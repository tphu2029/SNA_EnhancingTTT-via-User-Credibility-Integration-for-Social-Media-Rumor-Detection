import torch
import torch.nn as nn
from typing import Optional

class UserCredibilityEncoder(nn.Module):
    """
    Mã hóa vector đặc trưng uy tín người dùng (12 chiều) thành không gian biểu diễn ẩn.
    Hỗ trợ cơ chế Gated Fusion hoặc Residual Fusion với biểu diễn nội dung văn bản.
    """
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        output_dim: int = 128,
        dropout: float = 0.1,
        fusion_type: str = "gated",  # 'gated', 'concat', 'additive'
    ):
        super().__init__()
        self.fusion_type = fusion_type
        
        # User feature MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )
        
        # Gate mechanism if gated fusion is used
        if fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid()
            )
        elif fusion_type == "concat":
            self.proj = nn.Linear(output_dim * 2, output_dim)
            
    def forward(self, user_features: torch.Tensor, text_repr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        user_features: [batch_size, num_nodes, input_dim] hoặc [num_nodes, input_dim]
        text_repr: [batch_size, num_nodes, output_dim] (nếu cần fusion trực tiếp)
        """
        u_emb = self.mlp(user_features)
        
        if text_repr is None:
            return u_emb
            
        if self.fusion_type == "gated":
            # Concat text và user để tính trọng số cổng alpha
            combined = torch.cat([text_repr, u_emb], dim=-1)
            alpha = self.gate(combined)
            fused = alpha * text_repr + (1.0 - alpha) * u_emb
            return fused
            
        elif self.fusion_type == "concat":
            combined = torch.cat([text_repr, u_emb], dim=-1)
            return self.proj(combined)
            
        elif self.fusion_type == "additive":
            return text_repr + u_emb
            
        return u_emb
