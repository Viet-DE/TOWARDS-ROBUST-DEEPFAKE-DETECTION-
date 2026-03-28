"""
LPC Stream - Complete local parts processing
"""
import torch
import torch.nn as nn
from .lpc_backbone import LPCBackbone


class LPCStream(nn.Module):
    """Local Part Consistency Stream"""
    
    def __init__(self,
                 backbone_name: str = 'efficientnet_b0',
                 pretrained: bool = True,
                 part_feature_dim: int = 256,
                 feature_dim: int = 512,
                 dropout: float = 0.3):
        super().__init__()
        
        self.part_feature_dim = part_feature_dim
        self.feature_dim = feature_dim
        
        # Siamese backbone (shared for eyes and mouth)
        self.backbone = LPCBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            part_feature_dim=part_feature_dim
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(part_feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, eyes, mouth):
        """
        Args:
            eyes: [B, 3, 64, 128] eyes region
            mouth: [B, 3, 64, 64] mouth region
            
        Returns:
            features: [B, feature_dim]
        """
        # Extract features using shared backbone
        eyes_features = self.backbone(eyes)      # [B, part_feature_dim]
        mouth_features = self.backbone(mouth)    # [B, part_feature_dim]
        
        # Concatenate
        combined = torch.cat([eyes_features, mouth_features], dim=1)  # [B, part_feature_dim * 2]
        
        # Fusion
        features = self.fusion(combined)  # [B, feature_dim]
        
        return features