import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.trixnet import create_model
from datasets import create_dataloaders
from utils import (
    get_config, setup_logger, TensorboardLogger, setup_seed,
    AverageMeter, accuracy, DeepSupervisionLoss, MetricTracker,
    save_checkpoint, load_checkpoint
)

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger, tb_logger):
    """Huấn luyện 1 Epoch"""
    model.train()
    
    losses = AverageMeter('Loss', ':.4f')
    accs = AverageMeter('Acc', ':.2f')
    
    # Thanh tiến trình
    pbar = tqdm(loader, desc=f"Train Epoch {epoch}", leave=False)
    
    for i, batch in enumerate(pbar):
        # 1. Chuyển dữ liệu sang GPU
        target = batch['label'].to(device)
        
        # Batch input cho model
        inputs = {
            'frequency': batch['frequency'].to(device),
            'flow': batch['flow'].to(device),
            'parts': {
                'eyes': batch['parts']['eyes'].to(device),
                'mouth': batch['parts']['mouth'].to(device)
            }
        }
        
        # 2. Forward Pass (Deep Supervision)
        # Model trả về: logits cuối, logits phụ, và features
        final_logits, aux_logits, _ = model(inputs)
        
        # 3. Tính Loss tổng hợp
        loss, loss_dict = criterion(final_logits, aux_logits, target)
        
        # 4. Backward & Optimize
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping (Tránh bùng nổ gradient)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 5. Đo đạc & Log
        acc = accuracy(final_logits, target)[0]
        losses.update(loss.item(), target.size(0))
        accs.update(acc.item(), target.size(0))
        
        # Cập nhật thanh tiến trình
        pbar.set_postfix({'loss': f"{losses.avg:.4f}", 'acc': f"{accs.avg:.2f}%"})
        
        # Log vào TensorBoard mỗi 10 bước
        step = epoch * len(loader) + i
        if i % 10 == 0:
            tb_logger.log_scalar('Train/Total_Loss', loss.item(), step)
            tb_logger.log_scalar('Train/Accuracy', acc.item(), step)
            # Log chi tiết loss từng nhánh
            for k, v in loss_dict.items():
                tb_logger.log_scalar(f'Train/Loss_{k}', v, step)

    logger.info(f"Epoch [{epoch}] Train Result: Loss {losses.avg:.4f} | Acc {accs.avg:.2f}%")
    return losses.avg, accs.avg

def validate(model, loader, criterion, device, epoch, logger, tb_logger, prefix="Val"):
    """Đánh giá model trên tập Val hoặc Test"""
    model.eval()
    
    losses = AverageMeter('Loss', ':.4f')
    metrics = MetricTracker() # Đo F1, AUC, Confusion Matrix
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{prefix} Epoch {epoch}", leave=False):
            target = batch['label'].to(device)
            
            inputs = {
                'frequency': batch['frequency'].to(device),
                'flow': batch['flow'].to(device),
                'parts': {
                    'eyes': batch['parts']['eyes'].to(device),
                    'mouth': batch['parts']['mouth'].to(device)
                }
            }
            
            # Forward
            final_logits, aux_logits, _ = model(inputs)
            
            # Loss
            loss, _ = criterion(final_logits, aux_logits, target)
            
            # Update metrics
            losses.update(loss.item(), target.size(0))
            metrics.update(final_logits, target)
            
    # Tính toán kết quả cuối cùng
    results = metrics.compute()
    results['loss'] = losses.avg
    
    # Log ra màn hình
    logger.info(f"Epoch [{epoch}] {prefix} Result:")
    logger.info(f"  >> Loss: {results['loss']:.4f}")
    logger.info(f"  >> Acc : {results['accuracy']*100:.2f}%")
    logger.info(f"  >> AUC : {results['auc']:.4f}")
    logger.info(f"  >> F1  : {results['f1']:.4f}")
    logger.info(f"  >> CM  : {results['confusion_matrix']}")
    
    # Log vào TensorBoard
    if tb_logger:
        tb_logger.log_metrics(results, epoch, prefix=prefix)
        
    return results

def main():
    parser = argparse.ArgumentParser(description="TriXNet Training")
    parser.add_argument('--config', default='configs/default.yaml', help='path to config file')
    # Cho phép override config nhanh từ dòng lệnh
    parser.add_argument('--batch_size', type=int, default=None) 
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    
    # 1. Load Configuration
    cfg = get_config(args.config)
    
    # Override từ command line nếu có
    if args.batch_size: cfg['dataset']['batch_size'] = args.batch_size
    if args.epochs: cfg['training']['epochs'] = args.epochs
    
    # Setup môi trường
    save_dir = os.path.join(cfg['training']['save_dir'], cfg['model']['name'])
    log_dir = os.path.join(cfg['training']['log_dir'], cfg['model']['name'])
    os.makedirs(save_dir, exist_ok=True)
    
    logger = setup_logger(log_dir)
    tb_logger = TensorboardLogger(log_dir)
    setup_seed(cfg['training']['seed'])
    
    logger.info(f" Starting training TriXNet with config: {args.config}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and cfg['training']['device'] == 'cuda' else 'cpu')
    logger.info(f"Device: {device}")
    
    # 2. DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(cfg)
    logger.info(f"Data loaded: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}")
    
    # 3. Model
    model = create_model(cfg).to(device)
    logger.info("Model created successfully.")
    
    # 4. Loss & Optimizer
    # Deep Supervision Loss: Trọng số lấy từ config trixnet.yaml
    weights = cfg['model'].get('branch_weights', {'frs': 1.0, 'dof': 1.0, 'lpc': 1.0})
    criterion = DeepSupervisionLoss(weights=weights, device=device).to(device)
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg['training']['optimizer']['lr'],
        weight_decay=cfg['training']['optimizer']['weight_decay']
    )
    
    # Scheduler: Giảm LR theo Cosine
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=cfg['training']['epochs'],
        eta_min=cfg['training']['scheduler']['min_lr']
    )
    
    # 5. Training Loop
    best_acc = 0.0
    start_epoch = 1
    
    # Resume nếu có checkpoint (Optional - logic load_checkpoint ở utils)
    # start_epoch, best_acc = load_checkpoint(model, optimizer, 'checkpoints/last.pth')
    
    try:
        for epoch in range(start_epoch, cfg['training']['epochs'] + 1):
            
            # --- TRAIN ---
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, logger, tb_logger
            )
            
            # --- VALIDATE ---
            val_results = validate(
                model, val_loader, criterion, device, epoch, logger, tb_logger
            )
            
            # Step Scheduler
            scheduler.step()
            
            # --- SAVE CHECKPOINT ---
            is_best = val_results['accuracy'] > best_acc
            if is_best:
                best_acc = val_results['accuracy']
            
            # Lưu state dict
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
                'config': cfg
            }
            
            save_checkpoint(state, is_best, save_dir, filename='last.pth')
            logger.info(f"Checkpoint saved. Best Acc: {best_acc*100:.2f}%")
            
            print("-" * 60)
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving emergency checkpoint")
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc
        }, False, save_dir, filename='interrupted.pth')
        
    logger.info("Training completed")
    tb_logger.close()

if __name__ == "__main__":
    main()