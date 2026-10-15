import argparse
import json
import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import PHEMETreeDataset, collate_tree_batch
from src.data.user_feature_builder import UserCredibilityExtractor
from src.models.ucttt_classifier import UserCredibilityTemporalTreeTransformer
from src.training.engine import EarlyStopping, train_one_epoch, evaluate_epoch
from src.utils.metrics import compute_all_metrics, print_metrics_summary

PHEME_5_EVENTS = ["charliehebdo", "sydneysiege", "ferguson", "ottawashooting", "germanwings-crash"]
PHEME_9_EVENTS = [
    "charliehebdo", "sydneysiege", "ferguson", "ottawashooting", "germanwings-crash",
    "ebola-essien", "prince-toronto", "putinmissing", "gurlitt"
]

ABLATION_VARIANTS: Dict[str, Dict[str, Any]] = {
    # 1. Full UC-TTT (Chuẩn đối chiếu)
    "full": {
        "variant_key": "full",
        "display_name": "UC-TTT (Full)",
        "category": "baseline",
        "feature_group": "all",
        "model_component": "full",
        "description": "Đầy đủ 12 đặc trưng người dùng và toàn bộ kiến trúc UC-TTT",
    },
    # 2. Nhóm bóc tách đặc trưng người dùng (Feature-level Ablation)
    "without_profile": {
        "variant_key": "without_profile",
        "display_name": "w/o Profile Features",
        "category": "feature",
        "feature_group": "no_profile",
        "model_component": "full",
        "description": "Loại bỏ nhóm Profile & Verification (verified, avatar, bio, age; còn 8 đặc trưng)",
    },
    "without_counts": {
        "variant_key": "without_counts",
        "display_name": "w/o Raw Counts",
        "category": "feature",
        "feature_group": "no_counts",
        "model_component": "full",
        "description": "Loại bỏ nhóm Engagement Counts (followers, friends, statuses, favourites, listed; còn 7 đặc trưng)",
    },
    "without_ratios": {
        "variant_key": "without_ratios",
        "display_name": "w/o Derived Ratios",
        "category": "feature",
        "feature_group": "no_ratios",
        "model_component": "full",
        "description": "Loại bỏ nhóm Derived Ratios (reputation ratio, status rate, favourite rate; còn 9 đặc trưng)",
    },
    "profile_only": {
        "variant_key": "profile_only",
        "display_name": "Profile Only",
        "category": "feature",
        "feature_group": "profile_only",
        "model_component": "full",
        "description": "Chỉ sử dụng duy nhất nhóm Profile & Verification (4 đặc trưng)",
    },
    "counts_only": {
        "variant_key": "counts_only",
        "display_name": "Counts Only",
        "category": "feature",
        "feature_group": "counts_only",
        "model_component": "full",
        "description": "Chỉ sử dụng duy nhất nhóm Engagement Counts (5 đặc trưng)",
    },
    "ratios_only": {
        "variant_key": "ratios_only",
        "display_name": "Ratios Only",
        "category": "feature",
        "feature_group": "ratios_only",
        "model_component": "full",
        "description": "Chỉ sử dụng duy nhất nhóm Derived Ratios (3 đặc trưng)",
    },
    "source_only": {
        "variant_key": "source_only",
        "display_name": "Source User Only",
        "category": "feature",
        "feature_group": "source_only",
        "model_component": "full",
        "description": "Chỉ dùng thông tin user của tweet gốc (root node), đặt toàn bộ user phản hồi về 0",
    },
    # 3. Nhóm bóc tách thành phần kiến trúc (Model-level Ablation)
    "without_temporal_gru": {
        "variant_key": "without_temporal_gru",
        "display_name": "w/o Temporal GRU",
        "category": "model",
        "feature_group": "all",
        "model_component": "without_temporal_gru",
        "description": "Giữ Tree Transformer tĩnh, bỏ chiều động lực học thời gian (Static Tree)",
    },
    "without_tree_transformer": {
        "variant_key": "without_tree_transformer",
        "display_name": "w/o Tree Transformer",
        "category": "model",
        "feature_group": "all",
        "model_component": "without_tree_transformer",
        "description": "Chỉ dùng GRU chuỗi thời gian, bỏ cấu trúc cây lan truyền",
    },
}


def normalize_variant_key(key: str) -> str:
    k = key.lower().strip().replace("-", "_").replace(" ", "_")
    alias_map = {
        "all": "full",
        "uc_ttt": "full",
        "ucttt": "full",
        "baseline": "full",
        "no_profile": "without_profile",
        "w/o_profile": "without_profile",
        "no_counts": "without_counts",
        "w/o_counts": "without_counts",
        "no_ratios": "without_ratios",
        "w/o_ratios": "without_ratios",
        "source_user_only": "source_only",
        "root_only": "source_only",
        "no_temporal_gru": "without_temporal_gru",
        "w/o_temporal_gru": "without_temporal_gru",
        "static_tree": "without_temporal_gru",
        "no_tree_transformer": "without_tree_transformer",
        "w/o_tree_transformer": "without_tree_transformer",
        "pure_temporal": "without_tree_transformer",
    }
    return alias_map.get(k, k)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def split_stratified_loeo(threads, test_event, val_ratio=0.15, seed=42):
    test_threads = [th for th in threads if th["event"] == test_event]
    other_threads = [th for th in threads if th["event"] != test_event]

    rng = random.Random(seed)
    threads_c0 = [th for th in other_threads if th["label"] == 0]
    threads_c1 = [th for th in other_threads if th["label"] == 1]
    rng.shuffle(threads_c0)
    rng.shuffle(threads_c1)

    n_val_c0 = max(1, int(len(threads_c0) * val_ratio))
    n_val_c1 = max(1, int(len(threads_c1) * val_ratio))

    val_threads = threads_c0[:n_val_c0] + threads_c1[:n_val_c1]
    train_threads = threads_c0[n_val_c0:] + threads_c1[n_val_c1:]
    rng.shuffle(train_threads)
    rng.shuffle(val_threads)

    return train_threads, val_threads, test_threads


def run_single_variant_loeo(variant_cfg: Dict[str, Any], args, threads, embeddings, events_list: List[str]) -> Dict[str, Any]:
    variant_key = variant_cfg["variant_key"]
    display_name = variant_cfg["display_name"]
    feature_group = variant_cfg["feature_group"]
    model_component = variant_cfg["model_component"]

    device = torch.device(args.device)
    user_input_dim = UserCredibilityExtractor.get_feature_dim(feature_group)
    use_tree_transformer = (model_component != "without_tree_transformer")
    use_temporal_gru = (model_component != "without_temporal_gru")

    sample_emb = next(iter(embeddings.values()))
    text_dim = sample_emb.shape[0] if torch.is_tensor(sample_emb) else 768

    print("\n" + "=" * 80)
    print(f"🚀 BẮT ĐẦU ABLATION VARIANT: [{display_name.upper()}]")
    print(f"   Mô tả: {variant_cfg['description']}")
    print(f"   Cấu hình: feature_group='{feature_group}' (dim={user_input_dim}), "
          f"TreeTransformer={'ON' if use_tree_transformer else 'OFF'}, "
          f"TemporalGRU={'ON' if use_temporal_gru else 'OFF'}")
    print(f"   Fusion: {args.fusion_type.upper()} | Windows: {args.window_size} posts ({args.window_aggregation})")
    print("=" * 80)

    # Lựa chọn folds cần chạy
    if args.folds == "all":
        fold_indices = list(range(len(events_list)))
    else:
        fold_indices = [int(x.strip()) for x in args.folds.split(",") if x.strip()]

    fold_results = []
    all_test_preds = []
    all_test_labels = []
    all_test_probs = []

    checkpoint_dir = Path(args.output_dir) / "checkpoints" / variant_key
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_variant_time = time.time()

    for fold_idx in fold_indices:
        test_ev = events_list[fold_idx]
        print(f"\n>>> Fold {fold_idx + 1}/{len(events_list)} - Test Event: [{test_ev}]")

        train_th, val_th, test_th = split_stratified_loeo(
            threads, test_event=test_ev, val_ratio=args.val_ratio, seed=args.seed + fold_idx
        )

        train_ds = PHEMETreeDataset(
            train_th, embeddings, num_windows=args.num_time_windows,
            window_size=args.window_size, window_mode=args.window_mode,
            feature_group=feature_group
        )
        val_ds = PHEMETreeDataset(
            val_th, embeddings, num_windows=args.num_time_windows,
            window_size=args.window_size, window_mode=args.window_mode,
            feature_group=feature_group
        )
        test_ds = PHEMETreeDataset(
            test_th, embeddings, num_windows=args.num_time_windows,
            window_size=args.window_size, window_mode=args.window_mode,
            feature_group=feature_group
        )

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_tree_batch)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_tree_batch)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_tree_batch)

        fusion_key = "additive" if args.fusion_type == "residual" else args.fusion_type
        model = UserCredibilityTemporalTreeTransformer(
            text_input_dim=text_dim,
            user_input_dim=user_input_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_tree_layers=args.num_tree_layers,
            num_time_windows=args.num_time_windows,
            gru_hidden_dim=args.gru_hidden_dim,
            fc_hidden_dim=args.fc_hidden_dim,
            num_classes=2,
            use_user_credibility=True,
            fusion_type=fusion_key,
            window_aggregation=args.window_aggregation,
            use_root_user_shortcut=args.use_root_user_shortcut,
            use_tree_transformer=use_tree_transformer,
            use_temporal_gru=use_temporal_gru,
            dropout=args.dropout,
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        ckpt_path = checkpoint_dir / f"fold_{fold_idx}_{test_ev}.pt"
        early_stopping = EarlyStopping(
            patience=args.patience,
            mode="max",
            save_path=str(ckpt_path) if args.save_checkpoints else None,
        )

        for epoch in range(1, args.epochs + 1):
            train_loss, train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                scheduler=scheduler,
                clip_grad_norm=1.0,
            )
            val_loss, val_metrics, _, _, _ = evaluate_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )
            val_f1 = val_metrics["macro_f1"]
            improved = early_stopping(val_f1, model)

            if epoch % 5 == 0 or improved or epoch == args.epochs:
                mark = "*" if improved else " "
                print(f"    Epoch {epoch:02d}/{args.epochs:02d} {mark} | Train Loss: {train_loss:.4f}, F1: {train_metrics['macro_f1']:.4f} | Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}")

            if early_stopping.early_stop:
                print(f"    [Early Stopping] Dừng sớm tại Epoch {epoch}. Best Val F1: {early_stopping.best_score:.4f}")
                break

        # Tải lại trọng số tốt nhất cho Test
        early_stopping.load_best_weights(model)
        test_loss, test_metrics, preds, probs, labels = evaluate_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        test_acc = test_metrics["accuracy"]
        test_f1 = test_metrics["macro_f1"]
        print(f"    --> Kết quả [{test_ev}]: Acc = {test_acc * 100:.2f}% | Macro-F1 = {test_f1 * 100:.2f}%")

        fold_results.append({
            "fold_idx": fold_idx,
            "test_event": test_ev,
            "accuracy": float(test_acc),
            "macro_f1": float(test_f1),
            "rumour_precision": float(test_metrics.get("class_metrics", {}).get("Rumour", {}).get("precision", 0.0)),
            "rumour_recall": float(test_metrics.get("class_metrics", {}).get("Rumour", {}).get("recall", 0.0)),
            "rumour_f1": float(test_metrics.get("class_metrics", {}).get("Rumour", {}).get("f1", 0.0)),
            "non_rumour_f1": float(test_metrics.get("class_metrics", {}).get("Non-rumour", {}).get("f1", 0.0)),
        })

        all_test_preds.extend(preds)
        all_test_labels.extend(labels)
        all_test_probs.extend(probs)

    # Tính mean và pooled metrics
    accs = [f["accuracy"] for f in fold_results]
    f1s = [f["macro_f1"] for f in fold_results]
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    mean_f1 = float(np.mean(f1s))
    std_f1 = float(np.std(f1s))

    pooled_metrics = compute_all_metrics(all_test_labels, all_test_preds)
    elapsed = time.time() - start_variant_time

    print("\n" + "-" * 75)
    print(f" TỔNG KẾT VARIANT [{display_name}]:")
    print(f" Mean LOEO Accuracy: {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    print(f" Mean LOEO Macro-F1: {mean_f1 * 100:.2f}% ± {std_f1 * 100:.2f}%")
    print(f" Pooled Accuracy:    {pooled_metrics['accuracy'] * 100:.2f}%")
    print(f" Pooled Macro-F1:    {pooled_metrics['macro_f1'] * 100:.2f}%")
    print(f" Thời gian thực thi: {elapsed / 60:.2f} phút")
    print("-" * 75)

    variant_result = {
        "variant_key": variant_key,
        "display_name": display_name,
        "category": variant_cfg["category"],
        "description": variant_cfg["description"],
        "feature_group": feature_group,
        "user_input_dim": user_input_dim,
        "model_component": model_component,
        "events_mode": args.events_mode,
        "num_folds": len(fold_results),
        "mean_loeo": {
            "accuracy": mean_acc,
            "accuracy_std": std_acc,
            "macro_f1": mean_f1,
            "macro_f1_std": std_f1,
        },
        "pooled": {
            "accuracy": float(pooled_metrics["accuracy"]),
            "macro_f1": float(pooled_metrics["macro_f1"]),
            "macro_precision": float(pooled_metrics.get("macro_precision", 0.0)),
            "macro_recall": float(pooled_metrics.get("macro_recall", 0.0)),
            "class_metrics": pooled_metrics.get("class_metrics", {}),
        },
        "folds": fold_results,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Lưu kết quả riêng của variant
    var_file = Path(args.output_dir) / f"{variant_key}_results.json"
    with open(var_file, "w", encoding="utf-8") as f:
        json.dump(variant_result, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu kết quả chi tiết variant tại: {var_file}")

    return variant_result


def update_ablation_summary(summary_path: Path, new_result: Dict[str, Any], baseline_path: Optional[Path] = None):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "title": "Ablation Studies Summary (RQ2 & Hypothesis H2 - Social Media Rumor Detection)",
        "protocol": "Leave-One-Event-Out (LOEO) Cross-Validation",
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_full": None,
        "variants": {},
    }

    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and "variants" in loaded:
                    summary_data = loaded
        except Exception as e:
            print(f"Warning: Không thể đọc file summary cũ ({e}). Tạo file mới.")

    # Cập nhật kết quả variant vào summary_data
    v_key = new_result["variant_key"]
    summary_data["variants"][v_key] = new_result

    # Xác định baseline full
    if "full" in summary_data["variants"]:
        summary_data["baseline_full"] = summary_data["variants"]["full"]
    elif summary_data.get("baseline_full") is None:
        # Thử đọc từ kết quả ucttt_residual_gate_events5_results.json nếu có
        candidate_baseline = None
        candidates = [
            baseline_path,
            ROOT / "results/ucttt/ucttt_residual_gate_events5_results.json",
            ROOT / "results/ucttt_residual_gate_events5_results.json",
        ]
        for c in candidates:
            if c and c.exists():
                try:
                    with open(c, "r", encoding="utf-8") as bf:
                        candidate_baseline = json.load(bf)
                        break
                except Exception:
                    pass

        if candidate_baseline:
            m_loeo = candidate_baseline.get("mean_loeo", {})
            pooled = candidate_baseline.get("pooled", {})
            summary_data["baseline_full"] = {
                "variant_key": "full",
                "display_name": "UC-TTT (Full)",
                "category": "baseline",
                "mean_loeo": m_loeo,
                "pooled": pooled,
            }
            summary_data["variants"]["full"] = summary_data["baseline_full"]

    # Tính toán delta (so với baseline full)
    baseline = summary_data.get("baseline_full")
    if baseline:
        b_acc = baseline["mean_loeo"]["accuracy"]
        b_f1 = baseline["mean_loeo"]["macro_f1"]
        b_pooled_acc = baseline["pooled"]["accuracy"]
        b_pooled_f1 = baseline["pooled"]["macro_f1"]

        for k, v in summary_data["variants"].items():
            cur_acc = v["mean_loeo"]["accuracy"]
            cur_f1 = v["mean_loeo"]["macro_f1"]
            cur_pooled_acc = v["pooled"]["accuracy"]
            cur_pooled_f1 = v["pooled"]["macro_f1"]

            v["delta_mean_loeo_accuracy"] = round((cur_acc - b_acc) * 100, 2)
            v["delta_mean_loeo_macro_f1"] = round((cur_f1 - b_f1) * 100, 2)
            v["delta_pooled_accuracy"] = round((cur_pooled_acc - b_pooled_acc) * 100, 2)
            v["delta_pooled_macro_f1"] = round((cur_pooled_f1 - b_pooled_f1) * 100, 2)

    summary_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print_summary_table(summary_data)


def print_summary_table(summary_data: Dict[str, Any]):
    variants = summary_data.get("variants", {})
    if not variants:
        return

    print("\n" + "=" * 105)
    print(" BẢNG TỔNG HỢP KẾT QUẢ ABLATION STUDIES (BÓC TÁCH ĐẶC TRƯNG & THÀNH PHẦN KIẾN TRÚC)")
    print("=" * 105)
    print(f"{'Variant Name':<28} | {'Category':<8} | {'Mean Acc (%)':<16} | {'Mean F1 (%)':<16} | {'Δ Acc (%)':<10} | {'Δ F1 (%)':<10}")
    print("-" * 105)

    order = [
        "full",
        "without_profile",
        "without_counts",
        "without_ratios",
        "profile_only",
        "counts_only",
        "ratios_only",
        "source_only",
        "without_temporal_gru",
        "without_tree_transformer",
    ]

    for k in order:
        if k not in variants:
            continue
        v = variants[k]
        name = v.get("display_name", k)
        cat = v.get("category", "")
        acc_str = f"{v['mean_loeo']['accuracy'] * 100:.2f} ± {v['mean_loeo']['accuracy_std'] * 100:.2f}"
        f1_str = f"{v['mean_loeo']['macro_f1'] * 100:.2f} ± {v['mean_loeo']['macro_f1_std'] * 100:.2f}"
        d_acc = v.get("delta_mean_loeo_accuracy", 0.0)
        d_f1 = v.get("delta_mean_loeo_macro_f1", 0.0)

        d_acc_str = f"{d_acc:+.2f}%" if d_acc != 0 else "0.00% (Ref)"
        d_f1_str = f"{d_f1:+.2f}%" if d_f1 != 0 else "0.00% (Ref)"

        print(f"{name:<28} | {cat:<8} | {acc_str:<16} | {f1_str:<16} | {d_acc_str:<10} | {d_f1_str:<10}")

    print("=" * 105 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script điều phối Ablation Studies tự động cho UC-TTT (Feature & Model Architecture Ablations)"
    )
    parser.add_argument(
        "--feature_group",
        type=str,
        default="full",
        help="Nhóm đặc trưng người dùng: 'full', 'without_profile', 'without_counts', 'without_ratios', "
             "'profile_only', 'counts_only', 'ratios_only', 'source_only'",
    )
    parser.add_argument(
        "--model_component",
        type=str,
        default="full",
        help="Thành phần kiến trúc: 'full', 'without_temporal_gru', 'without_tree_transformer'",
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        default=False,
        help="Chạy tuần tự toàn bộ 10 cấu hình thực nghiệm bóc tách",
    )
    parser.add_argument(
        "--events_mode",
        type=str,
        choices=["5", "9"],
        default="5",
        help="'5' cho PHEME v5 (chuẩn bài báo Wu et al. 2025) hoặc '9' cho full PHEME",
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="data/processed/text_embeddings_bert.pt" if (ROOT / "data/processed/text_embeddings_bert.pt").exists() else "data/processed/text_embeddings.pt",
        help="Đường dẫn file text embeddings",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(ROOT / "data/processed/pheme_processed.pkl"),
        help="Đường dẫn file dataset PHEME (.pkl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "results/ablation"),
        help="Thư mục lưu kết quả thực nghiệm bóc tách",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default=str(ROOT / "results/ablation/ablation_summary.json"),
        help="Đường dẫn file tổng hợp kết quả ablation_summary.json",
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="residual_gate",
        choices=["residual_gate", "gated", "concat", "additive"],
        help="Cơ chế dung hợp đặc trưng uy tín người dùng",
    )
    parser.add_argument("--epochs", type=int, default=35, help="Số epochs tối đa")
    parser.add_argument("--patience", type=int, default=8, help="Patience cho Early Stopping")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=4.2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--nhead", type=int, default=12)
    parser.add_argument("--num_tree_layers", type=int, default=6)
    parser.add_argument("--num_time_windows", type=int, default=8)
    parser.add_argument("--window_mode", type=str, default="fixed_posts", choices=["fixed_posts", "fixed_windows"])
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--window_aggregation", type=str, default="cumulative", choices=["cumulative", "disjoint"])
    parser.add_argument("--gru_hidden_dim", type=int, default=128)
    parser.add_argument("--fc_hidden_dim", type=int, default=600)
    parser.add_argument("--use_root_user_shortcut", action="store_true", default=False)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=str, default="all", help="'all' hoặc chỉ số fold (ví dụ '0,1')")
    parser.add_argument("--save_checkpoints", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 80)
    print(" HỆ THỐNG THỰC NGHIỆM BÓC TÁCH (ABLATION STUDIES) - UC-TTT")
    print(f" Thời gian khởi chạy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Device: {args.device} | Events Mode: PHEME {args.events_mode} sự kiện")
    print(f" Dataset: {args.data_path}")
    print(f" Text Embeddings: {args.embeddings_path}")
    print("=" * 80)

    # 1. Đọc dữ liệu threads
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")

    with open(data_path, "rb") as f:
        threads_raw = pickle.load(f)

    events_list = PHEME_5_EVENTS if args.events_mode == "5" else PHEME_9_EVENTS
    threads = [th for th in threads_raw if th["event"] in events_list]
    print(f"Đã nạp {len(threads)} threads từ {len(events_list)} sự kiện: {', '.join(events_list)}")

    # 2. Đọc text embeddings
    emb_path = Path(args.embeddings_path)
    if not emb_path.is_absolute():
        emb_path = ROOT / emb_path
    if not emb_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file text embeddings: {emb_path}")

    print(f"Đang nạp text embeddings từ: {emb_path} ...")
    embeddings = torch.load(emb_path, map_location="cpu", weights_only=False)
    print(f"Đã nạp {len(embeddings)} text embeddings.")

    # 3. Xác định danh sách các variant cần chạy
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = Path(args.summary_path)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path

    if args.run_all:
        variants_to_run = list(ABLATION_VARIANTS.values())
        print(f"\n[CHẾ ĐỘ --run_all]: Sẽ chạy tuần tự {len(variants_to_run)} cấu hình thực nghiệm bóc tách:")
        for idx, v in enumerate(variants_to_run, 1):
            print(f"   {idx}. {v['display_name']} ({v['category']}) - {v['description']}")
    else:
        # Nhận diện cấu hình từ --feature_group hoặc --model_component
        norm_feat = normalize_variant_key(args.feature_group)
        norm_model = normalize_variant_key(args.model_component)

        # Ưu tiên nếu truyền model_component khác full
        if norm_model in ("without_temporal_gru", "without_tree_transformer"):
            target_key = norm_model
        elif norm_feat in ABLATION_VARIANTS:
            target_key = norm_feat
        else:
            target_key = "full"

        if target_key not in ABLATION_VARIANTS:
            raise ValueError(f"Không nhận diện được variant: feature_group={args.feature_group}, model_component={args.model_component}")

        variants_to_run = [ABLATION_VARIANTS[target_key]]

    # 4. Thực thi từng variant
    for variant_cfg in variants_to_run:
        result = run_single_variant_loeo(variant_cfg, args, threads, embeddings, events_list)
        update_ablation_summary(summary_path, result)

    print("\n🎉 HOÀN THÀNH TOÀN BỘ CHUỖI THỰC NGHIỆM BÓC TÁCH!")
    print(f"📂 File tổng hợp kết quả đã được cập nhật tại: {summary_path}")
    print("👉 Bạn có thể sinh biểu đồ phân tích bằng lệnh:")
    print(f"   python experiments/plot_ablation_results.py --input {summary_path}\n")


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
            sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except Exception:
            pass
    main()
