import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

class MetricTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.preds = []
        self.targets = []
        self.probs = [] # Xác suất lớp Fake (class 1)
        
    def update(self, logits, target):
        """
        logits: Output raw của model [Batch, 2]
        target: Label [Batch]
        """
        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=1)[:, 1] # Lấy xác suất lớp 1 (Fake)
            predictions = torch.argmax(logits, dim=1)
            
            self.preds.extend(predictions.cpu().numpy())
            self.targets.extend(target.cpu().numpy())
            self.probs.extend(probabilities.cpu().numpy())
            
    def compute(self):
        """Trả về dictionary các metrics"""
        try:
            auc = roc_auc_score(self.targets, self.probs)
        except:
            auc = 0.5 # Trường hợp chỉ có 1 class trong batch
            
        f1 = f1_score(self.targets, self.preds, average='binary')
        acc = np.mean(np.array(self.preds) == np.array(self.targets))
        
        # Confusion Matrix
        cm = confusion_matrix(self.targets, self.preds)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0,0,0,0)
        
        return {
            'accuracy': acc,
            'auc': auc,
            'f1': f1,
            'confusion_matrix': {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}
        }