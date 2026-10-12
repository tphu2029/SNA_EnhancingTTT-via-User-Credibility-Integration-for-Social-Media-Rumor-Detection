import torch
import torch.nn as nn
from typing import Optional

class UserCredibilityEncoder(nn.Module):
    """
    Mã hóa vector đặc trưng uy tín người dùng (12 chiều) thành không gian biểu diễn ẩn.
    Hỗ trợ chuẩn hóa đầu vào LayerNorm, Gated Fusion, Residual Gated Fusion, Concat Projection.
    """
    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        output_dim: int = 128,
        dropout: float = 0.1,
        fusion_type: str = "residual_gate",  # 'residual_gate', 'gated', 'concat', 'additive'
    ):
        super().__init__()
        self.fusion_type = fusion_type
        
        # 1. Chuẩn hóa đặc trưng đầu vào LayerNorm để cân bằng thang đo giữa log_followers và binary flags
        self.input_norm = nn.LayerNorm(input_dim)
        
        # 2. User feature MLP
        self.mlp = nn.Sequential(
            self.input_norm,
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )
        
        # 3. Gate / Projection mechanism
        if fusion_type in ("gated", "residual_gate"):
            self.gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid()
            )
        elif fusion_type == "concat":
            self.proj = nn.Linear(output_dim * 2, output_dim)
            
    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        """Trả về biểu diễn tiềm ẩn của user [B, N, output_dim]."""
        return self.mlp(user_features)
            
    def forward(self, user_features: torch.Tensor, text_repr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        user_features: [batch_size, num_nodes, input_dim] hoặc [num_nodes, input_dim]
        text_repr: [batch_size, num_nodes, output_dim] (nếu cần fusion trực tiếp)
        """
        u_emb = self.mlp(user_features)
        
        if text_repr is None:
            return u_emb
            
        if self.fusion_type == "residual_gate":
            # Residual Gated Fusion: Bảo toàn 100% text_repr gốc, cộng thêm điều biến từ User Credibility
            combined = torch.cat([text_repr, u_emb], dim=-1)
            alpha = self.gate(combined)
            fused = text_repr + alpha * u_emb
            return fused

        elif self.fusion_type == "gated":
            # Gated Attention truyền thống: cân bằng trọng số giữa text và user
            combined = torch.cat([text_repr, u_emb], dim=-1)
            alpha = self.gate(combined)
            fused = alpha * text_repr + (1.0 - alpha) * u_emb
            return fused
            
        elif self.fusion_type == "concat":
            combined = torch.cat([text_repr, u_emb], dim=-1)
            return text_repr + self.proj(combined)
            
        elif self.fusion_type in ("additive", "residual"):
            return text_repr + u_emb
            
        return u_emb
