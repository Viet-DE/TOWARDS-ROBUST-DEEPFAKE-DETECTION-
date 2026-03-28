"""
LPC Backbone - Siamese CNN for facial parts
"""
import torch
import torch.nn as nn
import torchvision.models as models


class LPCBackbone(nn.Module):
    def __init__(self,
                 backbone_name: str = 'efficientnet_b0',
                 pretrained: bool = True,
                 part_feature_dim: int = 256):
        super().__init__()
        self.backbone_name = backbone_name
        self.part_feature_dim = part_feature_dim
        # Shared backbone for both eyes and mouth
        if backbone_name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(pretrained=pretrained)
            # Get feature dimension
            backbone_feature_dim = backbone.classifier[1].in_features
            # Replace classifier with feature projection
            backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(backbone_feature_dim, part_feature_dim)
            )
            self.shared_backbone = backbone
        
        elif backbone_name == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            
            # Replace fc layer
            backbone_feature_dim = backbone.fc.in_features
            backbone.fc = nn.Linear(backbone_feature_dim, part_feature_dim)
            
            self.shared_backbone = backbone
        
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] facial part (eyes or mouth)
            
        Returns:
            features: [B, part_feature_dim]
        """
        return self.shared_backbone(x)