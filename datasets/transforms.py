import torch
import torchvision.transforms as T
import numpy as np

class TriXNetTransforms:
    """Transforms for TriXNet"""
    
    def __init__(self, is_train: bool = True, image_size: int = 224):
        self.is_train = is_train
        
        # Chỉ áp dụng Augmentation về màu sắc, KHÔNG dùng biến đổi hình học (Flip/Rotate)
        # để tránh làm lệch pha giữa RGB và các feature maps (Flow/Freq) đã tính sẵn.
        if is_train:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, img):
        return self.transform(img)