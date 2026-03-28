import os
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def setup_logger(log_dir):
    """Thiết lập logging ra console và file"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = datetime.now().strftime('train_%Y%m%d_%H%M.log')
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

class TensorboardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        
    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        
    def log_metrics(self, metrics, step, prefix="Val"):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f"{prefix}/{k}", v, step)
                
    def close(self):
        self.writer.close()