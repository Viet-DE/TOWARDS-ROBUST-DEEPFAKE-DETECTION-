import torch
import numpy as np
import cv2

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Đưa tensor ảnh về lại dạng numpy [0-255] để hiển thị"""
    tensor = tensor.clone().detach().cpu()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    img = tensor.numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def save_batch_grid(batch_images, save_path, nrow=4):
    """Lưu 1 batch ảnh thành 1 lưới ảnh (Grid)"""
    from torchvision.utils import save_image
    save_image(batch_images, save_path, nrow=nrow, normalize=True)