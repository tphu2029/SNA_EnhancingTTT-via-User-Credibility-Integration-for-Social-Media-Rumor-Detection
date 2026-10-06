import torch
import torch.nn as nn
from typing import Optional

class TemporalGRU(nn.Module):
    """
    Temporal Recurrent Unit dùng để mô hình hóa sự thay đổi của cấu trúc lan truyền qua các khoảng thời gian (Equal-depth / Equal-duration Time Windows).
    """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, num_layers: int = 1, bidirectional: bool = True, dropout: float = 0.1):
        super().__init__()
        self.bidirectional = bidirectional
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(out_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, time_slice_embeddings: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        time_slice_embeddings: [batch_size, num_windows, input_dim]
        lengths: [batch_size] số lượng window thực tế của từng thread (nếu có padding)
        Trả về vector đặc trưng tổng hợp thời gian [batch_size, hidden_dim]
        """
        out, h_n = self.gru(time_slice_embeddings)
        
        # Nếu có lengths, lấy hidden state tại bước thời gian cuối cùng của từng batch item
        if lengths is not None:
            batch_size = time_slice_embeddings.size(0)
            last_indices = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, out.size(2))
            last_out = out.gather(1, last_indices).squeeze(1)
        else:
            # Lấy step cuối cùng
            last_out = out[:, -1, :]

        projected = self.layer_norm(self.fc(last_out))
        return projected
