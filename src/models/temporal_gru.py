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
        if lengths is not None:
            lengths_cpu = lengths.cpu().clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                time_slice_embeddings, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, h_n = self.gru(packed)
            if self.bidirectional:
                last_out = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                last_out = h_n[-1]
        else:
            out, h_n = self.gru(time_slice_embeddings)
            if self.bidirectional:
                last_out = torch.cat([h_n[-2], h_n[-1]], dim=-1)
            else:
                last_out = h_n[-1]

        projected = self.layer_norm(self.fc(last_out))
        return projected
