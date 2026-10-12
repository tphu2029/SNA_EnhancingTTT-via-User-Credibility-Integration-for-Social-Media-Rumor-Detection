import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.data.dataset import PHEMETreeDataset, collate_tree_batch
from src.models.ucttt_classifier import UserCredibilityTemporalTreeTransformer
from src.training.engine import EarlyStopping, train_one_epoch, evaluate_epoch
from src.utils.loeo_splitter import get_loeo_folds, split_dataset_by_loeo_fold
from src.utils.metrics import compute_all_metrics, print_metrics_summary


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_class_weights(labels: List[int]) -> torch.Tensor:
    total = len(labels)
    num_non_rumour = sum(1 for y in labels if y == 0)
    num_rumour = sum(1 for y in labels if y == 1)
    w0 = total / (2.0 * max(num_non_rumour, 1))
    w1 = total / (2.0 * max(num_rumour, 1))
    return torch.tensor([w0, w1], dtype=torch.float32)


PHEME_5_EVENTS = ["charliehebdo", "sydneysiege", "ferguson", "ottawashooting", "germanwings-crash"]


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện Mô hình Đề xuất: UC-TTT (User-Credibility Temporal Tree Transformer)"
    )
    parser.add_argument("--data_path", type=str, default=str(ROOT / "data/processed/pheme_processed.pkl"))
    parser.add_argument("--embeddings_path", type=str, default=str(ROOT / "data/processed/text_embeddings.pt"))
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "results/ucttt"))
    parser.add_argument(
        "--events_mode",
        type=str,
        choices=["5", "9"],
        default="5",
        help="'5' cho PHEME v5 (Wu et al. 2025 benchmark) hoặc '9' cho full PHEME",
    )
    parser.add_argument(
        "--fusion_type",
        type=str,
        choices=["gated", "concat", "additive", "residual"],
        default="gated",
        help="Chiến lược dung hợp đặc trưng uy tín: 'gated' (đề xuất chính), 'concat', 'additive'/'residual'",
    )
    parser.add_argument(
        "--val_strategy",
        type=str,
        choices=["stratified", "event"],
        default="stratified",
        help="'stratified' (15%% từ train events) hoặc 'event' (1 event riêng làm val)",
    )
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_tree_layers", type=int, default=2)
    parser.add_argument("--num_time_windows", type=int, default=4)
    parser.add_argument("--gru_hidden_dim", type=int, default=128)
    parser.add_argument("--user_input_dim", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", type=str, default="all", help="'all' hoặc chỉ số fold cụ thể (ví dụ '0,1')")
    parser.add_argument("--save_checkpoints", action="store_true", default=True)
    parser.add_argument(
        "--baseline_results_path",
        type=str,
        default=str(ROOT / "results/ttt_baseline_results.json"),
        help="Đường dẫn file kết quả Baseline 3 để so sánh đối chứng delta",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    # Chuẩn hóa tên fusion
    fusion_key = "additive" if args.fusion_type == "residual" else args.fusion_type

    print("=" * 80)
    print(" HUẤN LUYỆN MÔ HÌNH ĐỀ XUẤT: UC-TTT (User-Credibility Temporal Tree Transformer)")
    print("=" * 80)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.type == 'cuda' else 'CPU'})")
    print(f"Events Mode: {args.events_mode} sự kiện | Fusion Strategy: [{args.fusion_type.upper()}]")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr} | Patience: {args.patience}")
    print(f"Architecture: d_model={args.d_model}, nhead={args.nhead}, layers={args.num_tree_layers}, windows={args.num_time_windows}")
    print(f"use_user_credibility: True (12 User Credibility Features Enabled)")

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = str(ROOT / "results/checkpoints/ucttt")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1. Nạp dữ liệu
    print("\n[1/3] Đang nạp dữ liệu PHEME và Text Embeddings...")
    with open(args.data_path, "rb") as f:
        threads = pickle.load(f)
    embeddings = torch.load(args.embeddings_path, weights_only=False)

    if args.events_mode == "5":
        threads = [th for th in threads if th["event"] in PHEME_5_EVENTS]
        print(f"Lọc PHEME v5 (5 sự kiện đối chứng Wu et al. 2025): {len(threads):,} threads.")
    else:
        print(f"Sử dụng full PHEME (9 sự kiện): {len(threads):,} threads.")

    print(f"Loaded {len(embeddings):,} embeddings tensors.")

    # 2. Sinh folds LOEO
    events_in_data = sorted(list(set(th["event"] for th in threads)))
    if args.val_strategy == "event":
        folds = get_loeo_folds(events_in_data)
    else:
        folds = [
            {
                "fold_idx": i,
                "test_event": ev,
                "val_event": f"Stratified {int(args.val_ratio*100)}% train pool",
                "train_events": [e for e in events_in_data if e != ev],
            }
            for i, ev in enumerate(events_in_data)
        ]
    print(f"Đã tạo {len(folds)} LOEO folds trải trên {len(events_in_data)} sự kiện: {', '.join(events_in_data)}")

    selected_fold_indices = (
        list(range(len(folds)))
        if args.folds == "all"
        else [int(idx.strip()) for idx in args.folds.split(",") if idx.strip()]
    )

    # Nạp kết quả Baseline 3 để đối chứng nếu có
    baseline_data = None
    if os.path.exists(args.baseline_results_path):
        try:
            with open(args.baseline_results_path, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)
            print(f"Đã nạp kết quả đối chứng Baseline 3 (Pooled F1: {baseline_data['pooled']['macro_f1']*100:.2f}%)")
        except Exception as e:
            print(f"Lưu ý: Không thể đọc file baseline results: {e}")

    all_fold_results = []
    pooled_preds = []
    pooled_targets = []
    total_start_time = time.time()

    # 3. Chạy từng Fold
    for fold in folds:
        fold_idx = fold["fold_idx"]
        if fold_idx not in selected_fold_indices:
            continue

        test_ev = fold["test_event"]
        val_ev = fold["val_event"]
        train_evs = fold["train_events"]

        print("\n" + "-" * 70)
        print(f">>> FOLD {fold_idx + 1}/{len(folds)}: TEST EVENT = [{test_ev}] | VAL = [{val_ev}]")
        print(f"    Train events: {', '.join(train_evs)}")

        if args.val_strategy == "event":
            train_threads, val_threads, test_threads = split_dataset_by_loeo_fold(threads, fold)
        else:
            train_threads, val_threads, test_threads = split_stratified_loeo(
                threads, test_event=test_ev, val_ratio=args.val_ratio, seed=args.seed + fold_idx
            )

        print(f"    Threads count: Train={len(train_threads)}, Val={len(val_threads)}, Test={len(test_threads)}")

        train_ds = PHEMETreeDataset(train_threads, embeddings)
        val_ds = PHEMETreeDataset(val_threads, embeddings)
        test_ds = PHEMETreeDataset(test_threads, embeddings)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_tree_batch, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_tree_batch, num_workers=0
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_tree_batch, num_workers=0
        )

        train_labels = [th["label"] for th in train_threads]
        class_weights = calculate_class_weights(train_labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        sample_emb = next(iter(embeddings.values()))
        text_dim = sample_emb.shape[0] if torch.is_tensor(sample_emb) else 384

        model = UserCredibilityTemporalTreeTransformer(
            text_input_dim=text_dim,
            user_input_dim=args.user_input_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_tree_layers=args.num_tree_layers,
            num_time_windows=args.num_time_windows,
            gru_hidden_dim=args.gru_hidden_dim,
            num_classes=2,
            use_user_credibility=True,
            fusion_type=fusion_key,
            dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

        ckpt_path = os.path.join(checkpoint_dir, f"ucttt_{args.fusion_type}_mode{args.events_mode}_fold_{fold_idx}.pt")
        early_stopping = EarlyStopping(
            patience=args.patience,
            mode="max",
            save_path=ckpt_path if args.save_checkpoints else None,
        )

        fold_start_time = time.time()
        best_val_f1 = 0.0
        best_epoch = 0

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
            if improved:
                best_val_f1 = val_f1
                best_epoch = epoch

            if epoch % 5 == 0 or improved or epoch == args.epochs:
                mark = "*" if improved else " "
                print(
                    f"  Epoch {epoch:02d}/{args.epochs:02d} {mark} | "
                    f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['macro_f1']:.4f} | "
                    f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_f1:.4f}"
                )

            if early_stopping.early_stop:
                print(f"  [Early Stopping] Đạt ngưỡng kiên nhẫn tại epoch {epoch}. Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch}).")
                break

        # Tải lại weights tốt nhất cho Test
        early_stopping.load_best_weights(model)
        test_loss, test_metrics, test_preds, _, test_targets = evaluate_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        fold_time = time.time() - fold_start_time
        print(f"  >>> KẾT QUẢ FOLD {fold_idx + 1} (Test Event: {test_ev}, Time: {fold_time:.1f}s):")
        print_metrics_summary(test_metrics, prefix="      ")

        # So sánh fold này với baseline nếu có
        baseline_fold_f1 = None
        if baseline_data and "folds" in baseline_data:
            bf_match = next((b for b in baseline_data["folds"] if b.get("test_event") == test_ev), None)
            if bf_match:
                baseline_fold_f1 = bf_match.get("test_macro_f1")
                delta_f1 = test_metrics["macro_f1"] - baseline_fold_f1
                sign = "+" if delta_f1 >= 0 else ""
                print(f"      [Đối chứng Baseline 3] F1 gốc: {baseline_fold_f1*100:.2f}% -> UC-TTT: {test_metrics['macro_f1']*100:.2f}% ({sign}{delta_f1*100:.2f}%)")

        all_fold_results.append({
            "fold_idx": fold_idx,
            "test_event": test_ev,
            "val_event": val_ev,
            "best_epoch": best_epoch,
            "best_val_macro_f1": float(best_val_f1),
            "test_accuracy": float(test_metrics["accuracy"]),
            "test_macro_f1": float(test_metrics["macro_f1"]),
            "test_macro_precision": float(test_metrics["macro_precision"]),
            "test_macro_recall": float(test_metrics["macro_recall"]),
            "baseline_macro_f1": baseline_fold_f1,
            "delta_macro_f1": float(test_metrics["macro_f1"] - baseline_fold_f1) if baseline_fold_f1 is not None else None,
            "class_metrics": test_metrics["class_metrics"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "training_time_seconds": round(fold_time, 2),
        })

        pooled_preds.extend(test_preds)
        pooled_targets.extend(test_targets)

    # 4. Tổng kết đánh giá các Folds
    mean_acc = float(np.mean([r["test_accuracy"] for r in all_fold_results]))
    std_acc = float(np.std([r["test_accuracy"] for r in all_fold_results]))
    mean_f1 = float(np.mean([r["test_macro_f1"] for r in all_fold_results]))
    std_f1 = float(np.std([r["test_macro_f1"] for r in all_fold_results]))

    pooled_metrics = compute_all_metrics(np.array(pooled_targets), np.array(pooled_preds))

    total_time = time.time() - total_start_time

    # Tính Delta Pooled so với Baseline 3
    delta_pooled_acc = None
    delta_pooled_f1 = None
    if baseline_data and "pooled" in baseline_data:
        delta_pooled_acc = float(pooled_metrics["accuracy"] - baseline_data["pooled"]["accuracy"])
        delta_pooled_f1 = float(pooled_metrics["macro_f1"] - baseline_data["pooled"]["macro_f1"])

    summary = {
        "model_name": f"User-Credibility Temporal Tree Transformer (UC-TTT, Fusion: {args.fusion_type.upper()})",
        "use_user_credibility": True,
        "fusion_type": args.fusion_type,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": {
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_tree_layers": args.num_tree_layers,
            "num_time_windows": args.num_time_windows,
            "gru_hidden_dim": args.gru_hidden_dim,
            "user_input_dim": args.user_input_dim,
            "fusion_type": args.fusion_type,
            "dropout": args.dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed,
        },
        "mean_loeo": {
            "accuracy": mean_acc,
            "accuracy_std": std_acc,
            "macro_f1": mean_f1,
            "macro_f1_std": std_f1,
        },
        "pooled": {
            "accuracy": pooled_metrics["accuracy"],
            "macro_f1": pooled_metrics["macro_f1"],
            "macro_precision": pooled_metrics["macro_precision"],
            "macro_recall": pooled_metrics["macro_recall"],
            "delta_pooled_accuracy": delta_pooled_acc,
            "delta_pooled_macro_f1": delta_pooled_f1,
            "class_metrics": pooled_metrics["class_metrics"],
            "confusion_matrix": pooled_metrics["confusion_matrix"],
        },
        "folds": all_fold_results,
        "total_time_seconds": round(total_time, 2),
    }

    # Lưu kết quả
    fusion_res_file = os.path.join(args.output_dir, f"ucttt_{args.fusion_type}_events{args.events_mode}_results.json")
    with open(fusion_res_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Đồng thời lưu file theo mode tại results/
    root_events_file = str(ROOT / f"results/ucttt_{args.fusion_type}_events{args.events_mode}_results.json")
    with open(root_events_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Nếu là mô hình chính (gated, mode 5), đồng thời lưu file chuẩn ucttt_results.json
    if args.fusion_type == "gated" and args.events_mode == "5":
        root_results_file = str(ROOT / "results/ucttt_results.json")
        with open(root_results_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f" TỔNG KẾT BENCHMARK LOEO: UC-TTT (Fusion Strategy: {args.fusion_type.upper()})")
    print("=" * 80)
    print(f"Tổng thời gian chạy: {total_time:.1f}s ({total_time / 60:.2f} phút)")
    print(f"Mean LOEO Accuracy : {mean_acc * 100:.2f}% ± {std_acc * 100:.2f}%")
    print(f"Mean LOEO Macro-F1 : {mean_f1 * 100:.2f}% ± {std_f1 * 100:.2f}%")
    print(f"Pooled Accuracy    : {pooled_metrics['accuracy'] * 100:.2f}%", end="")
    if delta_pooled_acc is not None:
        sign = "+" if delta_pooled_acc >= 0 else ""
        print(f" ({sign}{delta_pooled_acc * 100:.2f}% so với Baseline 3)")
    else:
        print()
    print(f"Pooled Macro-F1    : {pooled_metrics['macro_f1'] * 100:.2f}%", end="")
    if delta_pooled_f1 is not None:
        sign = "+" if delta_pooled_f1 >= 0 else ""
        print(f" ({sign}{delta_pooled_f1 * 100:.2f}% so với Baseline 3)")
    else:
        print()

    print(f"\nChi tiết kết quả lưu tại: {fusion_res_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
