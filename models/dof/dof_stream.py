import torch
import torch.nn as nn
from .dof_backbone import DOFBackbone


class DOFStream(nn.Module):
    """Dense Optical Flow Stream"""
    
    def __init__(self,
                 backbone_name: str = 'efficientnet_b0',
                 pretrained: bool = True,
                 input_channels: int = 2,
                 feature_dim: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        
        self.backbone = DOFBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            input_channels=input_channels,
            feature_dim=feature_dim
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, flow_map):
        """
        Args:
            flow_map: [B, 2, H, W] optical flow
            
        Returns:
            features: [B, feature_dim]
        """
        # Extract features
        features = self.backbone(flow_map)
        features = self.dropout(features)
        
        return features