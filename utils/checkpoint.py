import torch
import os
import shutil

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    """
    Lưu checkpoint.
    state: Dict chứa model_state_dict, optimizer, epoch...
    is_best: Nếu True, copy thêm 1 bản model_best.pth
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth')
        shutil.copyfile(filepath, best_path)
        print(f"Saved new best model to {best_path}")

def load_checkpoint(model, optimizer, path, device='cuda'):
    """Tải checkpoint để resume training"""
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0, 0.0 # start_epoch, best_acc
        
    print(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_acc = checkpoint.get('best_acc', 0.0)
    
    return start_epoch, best_acc