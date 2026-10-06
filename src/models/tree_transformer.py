import torch
import torch.nn as nn
from typing import Optional

class TreeTransformerLayer(nn.Module):
    """
    Một lớp Transformer Encoder kết hợp mặt nạ cấu trúc cây (Tree Adjacency / Reachability Mask).
    """
    def __init__(self, d_model: int = 128, nhead: int = 4, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [batch_size, seq_len (num_nodes), d_model]
        attn_mask: [batch_size * nhead, seq_len, seq_len] hoặc [seq_len, seq_len] (0 cho kết nối hợp lệ, -inf cho không kết nối)
        key_padding_mask: [batch_size, seq_len] (True nếu là node padding)
        """
        # Multi-head attention với residual connection và LayerNorm
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = self.norm1(x + self.dropout1(attn_out))
        
        # Feed-forward network
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff_out))
        return x

class TreeTransformer(nn.Module):
    """
    Ngăn xếp nhiều tầng Tree Transformer Layers nhằm mô hình hóa sự tương tác và lan truyền thông tin trong cây hội thoại.
    """
    def __init__(self, num_layers: int = 2, d_model: int = 128, nhead: int = 4, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TreeTransformerLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.norm(x)
