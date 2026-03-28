
import torch
import torch.nn as nn
from .frs.frs_stream import FRSStream
from .dof.dof_stream import DOFStream
from .lpc.lpc_stream import LPCStream
from .cma.cma_fusion import CMAFusion


class TriXNet(nn.Module):
    
    def __init__(self,
                 num_classes: int = 2,
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 dropout: float = 0.3,
                 backbone_name: str = 'efficientnet_b0',
                 pretrained: bool = True,
                 cma_num_heads: int = 8,
                 cma_num_layers: int = 2):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # --- Stream Branches ---
        # 1. Frequency Residual Stream
        self.frs = FRSStream(
            backbone_name=backbone_name,
            pretrained=pretrained,
            input_channels=1,
            feature_dim=feature_dim,
            dropout=dropout
        )
        
        # 2. Dense Optical Flow Stream
        self.dof = DOFStream(
            backbone_name=backbone_name,
            pretrained=pretrained,
            input_channels=2,
            feature_dim=feature_dim,
            dropout=dropout
        )
        
        # 3. Local Part Consistency Stream
        self.lpc = LPCStream(
            backbone_name=backbone_name,
            pretrained=pretrained,
            part_feature_dim=256,
            feature_dim=feature_dim,
            dropout=dropout
        )
        
        # --- Fusion Module ---
        self.cma = CMAFusion(
            feature_dim=feature_dim,
            num_heads=cma_num_heads,
            num_layers=cma_num_layers,
            hidden_dim=hidden_dim * 4,
            dropout=dropout
        )
        
        # --- Classifiers ---
        
        # 1. Main Classifier (Final Prediction)
        self.classifier = self._make_classifier(feature_dim, hidden_dim, num_classes, dropout)
        
        # 2. Auxiliary Classifiers (Deep Supervision)
        # Giúp từng nhánh tự học cách phân loại ngay cả khi chưa fusion
        self.aux_classifier_frs = self._make_classifier(feature_dim, hidden_dim // 2, num_classes, dropout)
        self.aux_classifier_dof = self._make_classifier(feature_dim, hidden_dim // 2, num_classes, dropout)
        self.aux_classifier_lpc = self._make_classifier(feature_dim, hidden_dim // 2, num_classes, dropout)
    
    def _make_classifier(self, input_dim, hidden_dim, num_classes, dropout):
        """Helper to create a standard classifier block"""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, batch):
        """
        Args:
            batch: dict containing 'frequency', 'flow', 'parts'
                
        Returns:
            final_logits: [B, num_classes] -> Dùng cho metrics chính
            aux_logits: dict of [B, num_classes] -> Dùng để tính Loss phụ
            branch_features: dict -> Dùng để visualize/debug
        """
        # 1. Feature Extraction (Independent Streams)
        frs_features = self.frs(batch['frequency'])           # [B, feature_dim]
        dof_features = self.dof(batch['flow'])                # [B, feature_dim]
        lpc_features = self.lpc(batch['parts']['eyes'], 
                               batch['parts']['mouth'])       # [B, feature_dim]
        
        # 2. Auxiliary Predictions (Deep Supervision)
        logits_frs = self.aux_classifier_frs(frs_features)
        logits_dof = self.aux_classifier_dof(dof_features)
        logits_lpc = self.aux_classifier_lpc(lpc_features)
        
        # 3. Fusion (Cross-Modality Attention)
        fused_features = self.cma(frs_features, dof_features, lpc_features)  # [B, feature_dim]
        
        # 4. Final Classification
        final_logits = self.classifier(fused_features)  # [B, num_classes]
        
        # Pack results
        aux_logits = {
            'frs': logits_frs,
            'dof': logits_dof,
            'lpc': logits_lpc
        }
        
        branch_features = {
            'frs': frs_features,
            'dof': dof_features,
            'lpc': lpc_features,
            'fused': fused_features
        }
        
        return final_logits, aux_logits, branch_features
    
    def predict(self, batch):
        """Prediction only uses the final fused logits"""
        logits, _, _ = self.forward(batch)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs


def create_model(config: dict) -> TriXNet:
    """Create TriXNet model from config"""
    model_config = config.get('model', {})
    
    # Đọc config hoặc dùng giá trị mặc định
    feature_dim = model_config.get('feature_dim', 512)
    dropout = model_config.get('dropout', 0.3)
    
    # Lấy thêm tham số CMA từ file config nếu có
    cma_config = model_config.get('cma', {})
    
    model = TriXNet(
        num_classes=model_config.get('num_classes', 2),
        feature_dim=feature_dim,
        hidden_dim=model_config.get('hidden_dim', 256),
        dropout=dropout,
        backbone_name='efficientnet_b0',
        pretrained=True,
        cma_num_heads=cma_config.get('num_heads', 8),
        cma_num_layers=cma_config.get('num_layers', 2)
    )
    
    return model