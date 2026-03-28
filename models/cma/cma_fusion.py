import torch
import torch.nn as nn
import math


class CMAFusion(nn.Module):
    def __init__(self,
                 feature_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 hidden_dim: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        # Positional encoding cho 3 modalities
        self.pos_encoding = nn.Parameter(torch.randn(1, 3, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(     # Transformer encoder
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    
    def forward(self, frs_features, dof_features, lpc_features):
        """
        Args:
            frs_features: [B, feature_dim] from FRS
            dof_features: [B, feature_dim] from DOF
            lpc_features: [B, feature_dim] from LPC
            
        Returns:
            fused_features: [B, feature_dim]
        """
        batch_size = frs_features.size(0)
        
        # Stack features into sequence [B, 3, feature_dim]
        # Order: FRS, DOF, LPC
        features = torch.stack([frs_features, dof_features, lpc_features], dim=1)
        
        # Add positional encoding
        features = features + self.pos_encoding
        
        # Apply transformer encoder (cross-attention between modalities)
        attended_features = self.transformer_encoder(features)  # [B, 3, feature_dim]
        
        # Flatten
        attended_features = attended_features.view(batch_size, -1)  # [B, feature_dim * 3]
        
        # Project to output dimension
        fused_features = self.output_proj(attended_features)  # [B, feature_dim]
        
        return fused_features