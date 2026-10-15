import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from src.models.ucttt_classifier import UserCredibilityTemporalTreeTransformer

def test_ttt_reimplemented_forward_backward():
    torch.manual_seed(42)
    B, N, D_text = 4, 10, 384
    
    # Mock data
    text = torch.randn(B, N, D_text)
    user = torch.randn(B, N, 12)
    depths = torch.randint(0, 5, (B, N))
    windows = torch.randint(0, 4, (B, N))
    num_windows = torch.tensor([3, 4, 2, 4])
    padding = torch.zeros(B, N, dtype=torch.bool)
    padding[0, 8:] = True
    padding[2, 6:] = True
    
    # Adjacency mask [B, N, N]
    adj_mask = torch.zeros(B, N, N, dtype=torch.bool)
    idx = torch.arange(N)
    adj_mask[:, idx, idx] = True  # self attend
    adj_mask[:, :, 0] = True      # all attend to root
    adj_mask[:, 0, :] = True      # root attends to all
    
    labels = torch.tensor([0, 1, 1, 0])
    
    # 1. Test TTT Baseline (use_user_credibility=False, cumulative windows, with adj_mask)
    model_ttt = UserCredibilityTemporalTreeTransformer(
        text_input_dim=D_text,
        d_model=64,
        nhead=4,
        num_tree_layers=2,
        num_time_windows=4,
        gru_hidden_dim=32,
        fc_hidden_dim=64,
        use_user_credibility=False,
        window_aggregation="cumulative",
    )
    
    logits_ttt = model_ttt(
        text_features=text,
        depths=depths,
        user_features=None,
        time_window_assignments=windows,
        adj_mask=adj_mask,
        node_padding_mask=padding,
        num_windows=num_windows,
    )
    assert logits_ttt.shape == (B, 2)
    assert not torch.isnan(logits_ttt).any()
    
    # Unweighted CrossEntropyLoss test
    loss_fn = nn.CrossEntropyLoss()
    loss_ttt = loss_fn(logits_ttt, labels)
    loss_ttt.backward()
    assert loss_ttt.item() > 0
    print("[OK] TTT Baseline forward + backward with adj_mask & cumulative subtrees passed.")
    
    # 2. Test UC-TTT Proposed (use_user_credibility=True, cumulative windows, with adj_mask)
    model_ucttt = UserCredibilityTemporalTreeTransformer(
        text_input_dim=D_text,
        user_input_dim=12,
        d_model=64,
        nhead=4,
        num_tree_layers=2,
        num_time_windows=4,
        gru_hidden_dim=32,
        fc_hidden_dim=64,
        use_user_credibility=True,
        fusion_type="gated",
        window_aggregation="cumulative",
    )
    
    logits_ucttt = model_ucttt(
        text_features=text,
        depths=depths,
        user_features=user,
        time_window_assignments=windows,
        adj_mask=adj_mask,
        node_padding_mask=padding,
        num_windows=num_windows,
    )
    assert logits_ucttt.shape == (B, 2)
    assert not torch.isnan(logits_ucttt).any()
    
    loss_ucttt = loss_fn(logits_ucttt, labels)
    loss_ucttt.backward()
    assert loss_ucttt.item() > 0
    print("[OK] UC-TTT Proposed forward + backward with adj_mask & cumulative subtrees passed.")

if __name__ == "__main__":
    test_ttt_reimplemented_forward_backward()
