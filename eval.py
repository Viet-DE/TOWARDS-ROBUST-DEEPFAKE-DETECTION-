"""
TriXNet Evaluation Script
Đánh giá model trên tập Test với các metrics: Accuracy, AUC, EER, F1, Confusion Matrix
"""
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Import project modules
from models.trixnet import create_model
from datasets import create_dataloaders
from utils import get_config, setup_seed, accuracy

def compute_eer(labels, scores):
    """Tính Equal Error Rate (EER) - Chỉ số quan trọng cho Deepfake"""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh, fpr, tpr

def plot_confusion_matrix(cm, classes, save_path):
    """Vẽ và lưu Confusion Matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion Matrix saved to {save_path}")

def evaluate(model, loader, device):
    model.eval()
    
    all_targets = []
    all_probs = []   # Xác suất lớp Fake
    all_preds = []   # Nhãn dự đoán (0 hoặc 1)
    
    print("running evaluation...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            # Move to device
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
            # Lưu ý: evaluation chỉ quan tâm final_logits
            logits, _, _ = model(inputs)
            
            # Tính xác suất (Softmax)
            probs = torch.softmax(logits, dim=1)[:, 1] # Lấy cột Fake
            preds = torch.argmax(logits, dim=1)
            
            # Store
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    return np.array(all_targets), np.array(all_probs), np.array(all_preds)

def main():
    parser = argparse.ArgumentParser(description="TriXNet Evaluation")
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model_best.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_dir', type=str, default='results', help='Where to save plots')
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.save_dir, exist_ok=True)
    cfg = get_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}")
    
    # Load Data (Chỉ lấy Test Loader)
    _, _, test_loader = create_dataloaders(cfg)
    if test_loader is None:
        print("Error: Test split not found in config/dataset. Check your yaml.")
        return

    # Load Model
    print("Loading model...")
    model = create_model(cfg).to(device)
    
    
    
    
    # Load Checkpoint
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False) # Thêm tham số weights_only=False để cho phép load các biến số Numpy (bản torch mới nhất 2.4+ chỉ cho phép weights_only)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')} (Acc: {checkpoint.get('best_acc', 0):.2%})")
    else:
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    # Run Evaluation
    targets, probs, preds = evaluate(model, test_loader, device)
    
    # --- Metrics Calculation ---
    
    # 1. Classification Report
    print("\n" + "="*30)
    print("CLASSIFICATION REPORT")
    print("="*30)
    print(classification_report(targets, preds, target_names=['Real', 'Fake'], digits=4))
    
    # 2. AUC & ROC
    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)
    print(f"AUC Score: {roc_auc:.4f}")
    
    # 3. EER (Equal Error Rate)
    eer, thresh, _, _ = compute_eer(targets, probs)
    print(f"EER Score: {eer:.4f} (at threshold {thresh:.4f})")
    
    # 4. Confusion Matrix
    cm = confusion_matrix(targets, preds)
    plot_path = os.path.join(args.save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, ['Real', 'Fake'], plot_path)
    
    # Vẽ biểu đồ ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.save_dir, 'roc_curve.png'))
    print(f"ROC Curve saved to {os.path.join(args.save_dir, 'roc_curve.png')}")

if __name__ == "__main__":
    main()