import torch
import numpy as np
import random
import os

def setup_seed(seed=42):
    """
    Fix random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Đảm bảo thuật toán CUDNN chạy determinictic (hơi chậm hơn nhưng kết quả giống hệt nhau)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")