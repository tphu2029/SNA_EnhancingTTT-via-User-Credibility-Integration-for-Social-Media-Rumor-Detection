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
        text_input_dim: int = 384,          # Chiều vector text đầu vào (MiniLM 384 hoặc BERT 768)
        user_input_dim: int = 12,           # Chiều đặc trưng User Credibility
        d_model: int = 768,                 # Chiều không gian ẩn (Wu et al. 2025: 768)
        nhead: int = 12,                    # Số attention heads (Wu et al. 2025: 12 heads)
        num_tree_layers: int = 6,           # Số lớp Tree Transformer (Wu et al. 2025: 6 layers)
        dim_feedforward: Optional[int] = None, # Chiều feedforward Transformer (mặc định 2048)
        num_time_windows: int = 8,          # Số lát cắt thời gian (Wu et al. 2025: 8 windows)
        gru_hidden_dim: int = 128,          # Chiều ẩn Temporal GRU (Wu et al. 2025: 128)
        fc_hidden_dim: int = 600,           # Chiều ẩn FC Classifier (Wu et al. 2025: 600)
        num_classes: int = 2,               # Rumour vs Non-rumour
        use_user_credibility: bool = True,  # Công tắc bật/tắt User Credibility (cho Baseline TTT gốc)
        fusion_type: str = "gated",         # 'gated', 'residual_gate', 'concat', 'additive'
        window_aggregation: str = "cumulative", # 'cumulative' (chuẩn bài báo Wu et al. 2025) hoặc 'disjoint'
        use_root_user_shortcut: bool = False,# Tắt root user shortcut để tránh overfitting out-of-domain
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_user_credibility = use_user_credibility
        self.use_root_user_shortcut = use_root_user_shortcut
        self.num_time_windows = num_time_windows
        self.d_model = d_model
        self.nhead = nhead
        self.window_aggregation = window_aggregation
        
        # 1. Text Projection (chiếu text đầu vào lên d_model)
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
                hidden_dim=128,
                output_dim=d_model,
                dropout=dropout,
                fusion_type=fusion_type,
            )
            
        # 4. Tree Transformer (6 layers, 12 heads, dim_feedforward=2048 theo Wu et al. 2025)
        ff_dim = dim_feedforward if dim_feedforward is not None else min(d_model * 4, 2048)
        self.tree_transformer = TreeTransformer(
            num_layers=num_tree_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
        )
        
        # 5. Temporal GRU (Bidirectional GRU)
        self.temporal_gru = TemporalGRU(
            input_dim=d_model,
            hidden_dim=gru_hidden_dim,
            num_layers=1,
            bidirectional=True,
            dropout=dropout,
        )
        
        # 6. Classifier Head (FC dimension 600 theo Wu et al. 2025)
        classifier_in_dim = gru_hidden_dim + d_model
        if self.use_user_credibility and self.use_root_user_shortcut:
            classifier_in_dim += d_model
            
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, fc_hidden_dim),
            nn.LayerNorm(fc_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward(
        self,
        text_features: torch.Tensor,                # [B, N, text_input_dim]
        depths: torch.Tensor,                       # [B, N]
        user_features: Optional[torch.Tensor] = None, # [B, N, user_input_dim]
        time_window_assignments: Optional[torch.Tensor] = None, # [B, N] (window index)
        adj_mask: Optional[torch.Tensor] = None,    # [B, N, N]
        node_padding_mask: Optional[torch.Tensor] = None, # [B, N] (True là padding)
        num_windows: Optional[torch.Tensor] = None, # [B] số cửa sổ thực tế của từng thread
    ) -> torch.Tensor:
        batch_size, num_nodes, _ = text_features.shape
        
        # 1. Chiếu text lên d_model
        h_text = self.text_proj(text_features)
        
        # 2. Cộng depth encoding
        h_depth = self.depth_enc(depths)
        h_node = h_text + h_depth
        
        # 3. Tích hợp User Credibility nếu được kích hoạt
        root_user_emb = None
        if self.use_user_credibility and user_features is not None:
            # Biểu diễn của root user (node index 0)
            u_all = self.user_encoder.encode_user(user_features)
            root_user_emb = u_all[:, 0, :]
            h_node = self.user_encoder(user_features, text_repr=h_node)
            
        # 4. Đi qua Tree Transformer (áp dụng Tree Adjacency Mask nếu có)
        tree_attn_mask = None
        kpm = node_padding_mask
        if adj_mask is not None:
            # Chuyển boolean adj_mask [B, N, N] thành additive float mask [B * nhead, N, N]
            # True -> 0.0 (được attend), False -> -inf (chặn attention)
            tree_attn_mask = torch.zeros(
                adj_mask.shape, dtype=h_node.dtype, device=h_node.device
            ).masked_fill(~adj_mask, float("-inf"))
            tree_attn_mask = tree_attn_mask.repeat_interleave(self.nhead, dim=0)

            # Đồng bộ key_padding_mask sang float để tránh deprecation warning của PyTorch
            if node_padding_mask is not None:
                kpm = torch.zeros_like(node_padding_mask, dtype=h_node.dtype).masked_fill(
                    node_padding_mask, float("-inf")
                )

        h_tree = self.tree_transformer(
            h_node,
            attn_mask=tree_attn_mask,
            key_padding_mask=kpm,
        )
        
        # Root node representation (vị trí index 0)
        root_repr = h_tree[:, 0, :]
        
        # 5. Phân bổ các node vào các time windows và Temporal GRU
        if time_window_assignments is None:
            # Mặc định chia đều theo index nếu không truyền time assignment
            num_w = self.num_time_windows
            time_window_assignments = (torch.arange(num_nodes, device=text_features.device) * num_w // num_nodes).repeat(batch_size, 1)
            total_windows = num_w
            if num_windows is None:
                num_windows = torch.full((batch_size,), num_w, device=text_features.device, dtype=torch.long)
        else:
            if num_windows is None:
                if node_padding_mask is not None:
                    valid_tw = time_window_assignments.masked_fill(node_padding_mask, 0)
                else:
                    valid_tw = time_window_assignments
                num_windows = valid_tw.max(dim=1).values + 1
            total_windows = int(time_window_assignments.max().item()) + 1
            total_windows = max(total_windows, 1)

        window_embeddings = []
        for w in range(total_windows):
            if self.window_aggregation == "cumulative":
                # Cumulative subtrees (Wu et al. 2025): Cửa sổ w chứa tất cả các node từ đầu đến w (<= w)
                mask_w = (time_window_assignments <= w).unsqueeze(-1).float()
            else:
                # Disjoint: Chỉ chứa các node thuộc riêng cửa sổ w (== w)
                mask_w = (time_window_assignments == w).unsqueeze(-1).float()

            if node_padding_mask is not None:
                mask_w = mask_w * (~node_padding_mask).unsqueeze(-1).float()
            
            denom = mask_w.sum(dim=1).clamp(min=1e-6)
            pooled_w = (h_tree * mask_w).sum(dim=1) / denom
            window_embeddings.append(pooled_w)

        # [B, total_windows, d_model]
        seq_windows = torch.stack(window_embeddings, dim=1)
        
        # Temporal aggregation
        h_temporal = self.temporal_gru(seq_windows, lengths=num_windows)
        
        # 6. Ghép root representation + temporal representation + root user shortcut -> phân loại
        features_to_cat = [root_repr, h_temporal]
        if self.use_user_credibility and self.use_root_user_shortcut and root_user_emb is not None:
            features_to_cat.append(root_user_emb)
            
        fused_graph = torch.cat(features_to_cat, dim=-1)
        logits = self.classifier(fused_graph)
        return logits
