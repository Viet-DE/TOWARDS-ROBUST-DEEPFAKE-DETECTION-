import torch
import torch.nn as nn

class DeepSupervisionLoss(nn.Module):
    """
    Weighted Loss for Multi-stream Network with Class Balancing
    """
    def __init__(self, weights={'frs': 1.0, 'dof': 1.0, 'lpc': 1.0}, device='cuda'):
        super().__init__()
        self.weights = weights
        
        # TÍNH TOÁN TRỌNG SỐ CHO LỚP REAL
        # Vì data Real ít hơn Fake gấp 4-5 lần, ta tăng phạt khi đoán sai Real lên 4-5 lần.
        # Class 0: Real, Class 1: Fake
        # weight = [Tiền phạt cho Real, Tiền phạt cho Fake]
        class_weights = torch.tensor([4.0, 1.0]).to(device) 
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
    def forward(self, final_logits, aux_logits, target):
        # 1. Main Loss
        loss_final = self.criterion(final_logits, target)
        total_loss = loss_final
        
        # 2. Auxiliary Losses
        loss_details = {'final': loss_final.item()}
        
        for branch_name, branch_logit in aux_logits.items():
            if branch_name in self.weights:
                branch_loss = self.criterion(branch_logit, target)
                weight = self.weights[branch_name]
                
                total_loss += weight * branch_loss
                loss_details[branch_name] = branch_loss.item()
                
        return total_loss, loss_details