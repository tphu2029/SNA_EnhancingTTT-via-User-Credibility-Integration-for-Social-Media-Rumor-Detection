import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
import torch

from src.data.dataset import PHEMETreeDataset, collate_tree_batch
from src.data.user_feature_builder import UserCredibilityExtractor
from src.models.ucttt_classifier import UserCredibilityTemporalTreeTransformer
from experiments.run_ablation import ABLATION_VARIANTS, normalize_variant_key, update_ablation_summary
from experiments.plot_ablation_results import plot_feature_contribution, plot_comprehensive_ablation


def test_feature_extractor_ablation_modes():
    extractor = UserCredibilityExtractor()
    feat = np.arange(12, dtype=np.float32)

    expected_dims = {
        "full": 12,
        "without_profile": 8,
        "without_counts": 7,
        "without_ratios": 9,
        "profile_only": 4,
        "counts_only": 5,
        "ratios_only": 3,
        "source_only": 12,
    }

    for mode, expected_dim in expected_dims.items():
        dim = extractor.get_feature_dim(mode)
        assert dim == expected_dim, f"Mode {mode} expected dim {expected_dim}, got {dim}"
        selected = extractor.select_feature_group(feat, mode=mode)
        assert len(selected) == expected_dim, f"Mode {mode} returned length {len(selected)}"


def test_dataset_source_only_ablation():
    threads = [{
        "thread_id": "test_1",
        "source_tweet": {"id": 1, "text": "hello", "created_at": "Wed Jan 07 11:00:00 +0000 2015", "credibility_features": [1.0] * 12},
        "reactions": [
            {"id": 2, "text": "reaction 1", "created_at": "Wed Jan 07 11:05:00 +0000 2015", "credibility_features": [2.0] * 12},
            {"id": 3, "text": "reaction 2", "created_at": "Wed Jan 07 11:10:00 +0000 2015", "credibility_features": [3.0] * 12},
        ],
        "edges": [[1, 2], [1, 3]],
        "label": 1,
        "event": "charliehebdo",
    }]
    emb = {
        "1": torch.randn(768),
        "2": torch.randn(768),
        "3": torch.randn(768),
    }

    # 1. Full
    ds_full = PHEMETreeDataset(threads, emb, feature_group="all")
    item_full = ds_full[0]
    assert item_full["node_users"].shape == (3, 12)
    assert torch.all(item_full["node_users"][1] == 2.0)

    # 2. Source only
    ds_source = PHEMETreeDataset(threads, emb, feature_group="source_only")
    item_source = ds_source[0]
    assert item_source["node_users"].shape == (3, 12)
    assert torch.all(item_source["node_users"][0] == 1.0)
    assert torch.all(item_source["node_users"][1:] == 0.0)

    # 3. Without profile (dim=8)
    ds_no_prof = PHEMETreeDataset(threads, emb, feature_group="without_profile")
    item_no_prof = ds_no_prof[0]
    assert item_no_prof["node_users"].shape == (3, 8)


def test_model_architectural_ablations():
    B, N = 2, 5
    text = torch.randn(B, N, 768)
    user = torch.randn(B, N, 12)
    depths = torch.zeros(B, N, dtype=torch.long)
    windows = torch.zeros(B, N, dtype=torch.long)
    pad = torch.zeros(B, N, dtype=torch.bool)
    adj = torch.eye(N, dtype=torch.bool).unsqueeze(0).repeat(B, 1, 1)

    # w/o Tree Transformer
    m_no_tree = UserCredibilityTemporalTreeTransformer(
        text_input_dim=768, user_input_dim=12, d_model=128, nhead=4,
        num_tree_layers=2, use_tree_transformer=False
    )
    out1 = m_no_tree(text, depths, user_features=user, time_window_assignments=windows, adj_mask=adj, node_padding_mask=pad)
    assert out1.shape == (B, 2)

    # w/o Temporal GRU
    m_no_gru = UserCredibilityTemporalTreeTransformer(
        text_input_dim=768, user_input_dim=12, d_model=128, nhead=4,
        num_tree_layers=2, use_temporal_gru=False
    )
    out2 = m_no_gru(text, depths, user_features=user, time_window_assignments=windows, adj_mask=adj, node_padding_mask=pad)
    assert out2.shape == (B, 2)


def test_ablation_summary_and_plotting():
    with tempfile.TemporaryDirectory() as tmpdir:
        summary_path = Path(tmpdir) / "ablation_summary.json"
        fig_feat = Path(tmpdir) / "ablation_feature_contribution.png"
        fig_comp = Path(tmpdir) / "ablation_full_summary.png"

        dummy_full = {
            "variant_key": "full",
            "display_name": "UC-TTT (Full)",
            "category": "baseline",
            "description": "Full model",
            "feature_group": "all",
            "user_input_dim": 12,
            "model_component": "full",
            "events_mode": "5",
            "num_folds": 5,
            "mean_loeo": {"accuracy": 0.7228, "accuracy_std": 0.0475, "macro_f1": 0.6793, "macro_f1_std": 0.0284},
            "pooled": {"accuracy": 0.7423, "macro_f1": 0.6979},
            "folds": [],
        }

        dummy_no_ratios = {
            "variant_key": "without_ratios",
            "display_name": "w/o Derived Ratios",
            "category": "feature",
            "description": "No ratios",
            "feature_group": "no_ratios",
            "user_input_dim": 9,
            "model_component": "full",
            "events_mode": "5",
            "num_folds": 5,
            "mean_loeo": {"accuracy": 0.7085, "accuracy_std": 0.0512, "macro_f1": 0.6610, "macro_f1_std": 0.0310},
            "pooled": {"accuracy": 0.7280, "macro_f1": 0.6780},
            "folds": [],
        }

        update_ablation_summary(summary_path, dummy_full)
        update_ablation_summary(summary_path, dummy_no_ratios)

        assert summary_path.exists()
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "full" in data["variants"]
        assert "without_ratios" in data["variants"]
        assert data["variants"]["without_ratios"]["delta_mean_loeo_macro_f1"] < 0

        # Test plot functions
        plot_feature_contribution(data, fig_feat)
        plot_comprehensive_ablation(data, fig_comp)

        assert fig_feat.exists()
        assert fig_comp.exists()
        assert fig_feat.stat().st_size > 1000
        assert fig_comp.stat().st_size > 1000
