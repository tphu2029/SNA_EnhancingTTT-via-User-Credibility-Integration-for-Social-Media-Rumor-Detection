import math
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List

from src.models.user_encoder import UserCredibilityEncoder
from src.models.tree_transformer import TreeTransformer
from src.models.temporal_gru import TemporalGRU

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, depths: torch.Tensor) -> torch.Tensor:
        # depths: [batch_size, num_nodes] (integer depths)
        return self.pe[depths]

class UserCredibilityTemporalTreeTransformer(nn.Module):
    """
    Mô hình UC-TTT (User-Credibility-Aware Temporal Tree Transformer) kết hợp:
      1. Nội dung văn bản của tweet (Text projection / embedding)
      2. Cấu trúc cây và độ sâu lan truyền (Depth positional encoding + Tree Transformer)
      3. Thông tin thời gian (Equal-depth/time windows + Temporal GRU)
      4. Đặc trưng uy tín người dùng (User Credibility Encoder với Gated Fusion)
    """
    def __init__(
        self,
        text_input_dim: int = 128,          # Chiều vector text (hoặc GloVe/Sentence embedding)
        user_input_dim: int = 12,           # Chiều đặc trưng User Credibility
        d_model: int = 128,                 # Chiều không gian ẩn chung
        nhead: int = 4,                     # Số attention heads
        num_tree_layers: int = 2,           # Số lớp Tree Transformer
        num_time_windows: int = 4,          # Số lát cắt thời gian (Time windows)
        gru_hidden_dim: int = 128,          # Chiều ẩn Temporal GRU
        num_classes: int = 2,               # Rumour vs Non-rumour
        use_user_credibility: bool = True,  # Công tắc bật/tắt User Credibility (cho Baseline TTT gốc)
        fusion_type: str = "gated",         # 'gated', 'concat', 'additive'
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_user_credibility = use_user_credibility
        self.num_time_windows = num_time_windows
        self.d_model = d_model
        
        # 1. Text Projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 2. Positional / Depth Encoding
        self.depth_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=100)
        
        # 3. User Credibility Encoder
        if self.use_user_credibility:
            self.user_encoder = UserCredibilityEncoder(
                input_dim=user_input_dim,
                hidden_dim=64,
                output_dim=d_model,
                dropout=dropout,
                fusion_type=fusion_type,
            )
            
        # 4. Tree Transformer
        self.tree_transformer = TreeTransformer(
            num_layers=num_tree_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
        )
        
        # 5. Temporal GRU
        self.temporal_gru = TemporalGRU(
            input_dim=d_model,
            hidden_dim=gru_hidden_dim,
            num_layers=1,
            bidirectional=True,
            dropout=dropout,
        )
        
        # 6. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_dim + d_model, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        text_features: torch.Tensor,                # [B, N, text_input_dim]
        depths: torch.Tensor,                       # [B, N]
        user_features: Optional[torch.Tensor] = None, # [B, N, user_input_dim]
        time_window_assignments: Optional[torch.Tensor] = None, # [B, N] (0 to num_time_windows - 1)
        adj_mask: Optional[torch.Tensor] = None,    # [B, N, N]
        node_padding_mask: Optional[torch.Tensor] = None, # [B, N] (True là padding)
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = text_features.shape
        
        # 1. Chiếu text lên d_model
        h_text = self.text_proj(text_features)
        
        # 2. Cộng depth encoding
        h_depth = self.depth_enc(depths)
        h_node = h_text + h_depth
        
        # 3. Tích hợp User Credibility nếu được kích hoạt
        if self.use_user_credibility and user_features is not None:
            h_node = self.user_encoder(user_features, text_repr=h_node)
            
        # 4. Đi qua Tree Transformer
        # PyTorch MultiheadAttention yêu cầu key_padding_mask: [B, N]
        h_tree = self.tree_transformer(
            h_node,
            attn_mask=None,
            key_padding_mask=node_padding_mask
        )
        
        # Root node representation (vị trí index 0)
        root_repr = h_tree[:, 0, :]
        
        # 5. Phân bổ các node vào các time windows và Temporal GRU
        if time_window_assignments is None:
            # Mặc định chia đều theo index nếu không truyền time assignment
            time_window_assignments = (torch.arange(num_nodes, device=text_features.device) * self.num_time_windows // num_nodes).repeat(batch_size, 1)

        window_embeddings = []
        for w in range(self.num_time_windows):
            mask_w = (time_window_assignments == w).unsqueeze(-1).float()
            if node_padding_mask is not None:
                mask_w = mask_w * (~node_padding_mask).unsqueeze(-1).float()
            
            denom = mask_w.sum(dim=1).clamp(min=1e-6)
            pooled_w = (h_tree * mask_w).sum(dim=1) / denom
            window_embeddings.append(pooled_w)

        # [B, num_time_windows, d_model]
        seq_windows = torch.stack(window_embeddings, dim=1)
        
        # Temporal aggregation
        h_temporal = self.temporal_gru(seq_windows)
        
        # 6. Ghép root representation + temporal representation -> phân loại
        fused_graph = torch.cat([root_repr, h_temporal], dim=-1)
        logits = self.classifier(fused_graph)
        return logits
