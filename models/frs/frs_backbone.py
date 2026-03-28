import torch
import torch.nn as nn
import torchvision.models as models


class FRSBackbone(nn.Module):
    def __init__(self, 
                 backbone_name: str = 'efficientnet_b0',
                 pretrained: bool = True,
                 input_channels: int = 1,
                 feature_dim: int = 512):
        super().__init__()
        self.backbone_name = backbone_name
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        # Load backbone
        if backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            # Modify first conv layer for single channel input
            if input_channels != 3:
                old_conv = self.backbone.features[0][0]
                self.backbone.features[0][0] = nn.Conv2d(
                    input_channels, 
                    old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False
                )
            
            backbone_feature_dim = self.backbone.classifier[1].in_features # Lay feature dimension tu backbone
            self.backbone.classifier = nn.Sequential(                      # Replace classifier with feature projection
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(backbone_feature_dim, feature_dim)
            )
        
        elif backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            
            # Modify first conv layer
            if input_channels != 3:
                self.backbone.conv1 = nn.Conv2d(
                    input_channels, 64,
                    kernel_size=7, stride=2, padding=3, bias=False
                )
            
            # Replace fc layer
            backbone_feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(backbone_feature_dim, feature_dim)
        
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] frequency residual map
            
        Returns:
            features: [B, feature_dim]
        """
        return self.backbone(x)