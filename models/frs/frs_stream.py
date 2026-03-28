
import torch
import torch.nn as nn
from .frs_backbone import FRSBackbone


class FRSStream(nn.Module):
    
    def __init__(self, 
                 backbone_name: str = 'efficientnet_b0',
                 pretrained: bool = True,
                 input_channels: int = 1,
                 feature_dim: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        
        self.backbone = FRSBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            input_channels=input_channels,
            feature_dim=feature_dim
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, frequency_map):
        """
        Args:
            frequency_map: [B, 1, H, W] frequency residual
            
        Returns:
            features: [B, feature_dim]
        """
        # Extract features
        features = self.backbone(frequency_map)
        features = self.dropout(features)
        
        return features