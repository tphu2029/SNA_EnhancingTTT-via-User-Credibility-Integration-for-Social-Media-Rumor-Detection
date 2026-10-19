import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def run_command(cmd, desc):
    print("\n" + "=" * 80)
    print(f" BẮT ĐẦU: {desc}")
    print(f" Lệnh: {' '.join(cmd)}")
    print("=" * 80 + "\n")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[ERROR] Lỗi khi chạy {desc}! Mã thoát: {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n[HOÀN THÀNH] {desc} trong {elapsed:.1f}s ({elapsed / 60:.2f} phút).")


def build_comparison_table():
    ttt_path = ROOT / "results/ttt_baseline_results.json"
    ucttt_path = ROOT / "results/ucttt/ucttt_gated_events5_results.json"

    if not ttt_path.exists():
        print(f"Không tìm thấy file kết quả TTT: {ttt_path}")
        return
    if not ucttt_path.exists():
        print(f"Không tìm thấy file kết quả UC-TTT: {ucttt_path}")
        return

    with open(ttt_path, "r", encoding="utf-8") as f:
        ttt_data = json.load(f)
    with open(ucttt_path, "r", encoding="utf-8") as f:
        ucttt_data = json.load(f)

    ttt_folds = {f["test_event"]: f for f in ttt_data.get("folds", [])}
    ucttt_folds = {f["test_event"]: f for f in ucttt_data.get("folds", [])}

    events = sorted(list(set(list(ttt_folds.keys()) + list(ucttt_folds.keys()))))

    lines = []
    lines.append("\n" + "=" * 90)
    lines.append(" BẢNG SO SÁNH HIỆU NĂNG: TTT BASELINE (WU ET AL. 2025) vs UC-TTT PROPOSED")
    lines.append(" Cấu hình đồng bộ: 6 layers, 12 heads, d_model=768, FC=600, K=8 windows, LR=4.2e-5")
    lines.append("=" * 90)
    lines.append(f"{'Event':<20} | {'TTT Acc':<9} {'TTT F1':<9} | {'UC-TTT Acc':<11} {'UC-TTT F1':<11} | {'Delta Acc':<10} {'Delta F1':<10}")
    lines.append("-" * 90)

    comparison_records = []
    for ev in events:
        t_fold = ttt_folds.get(ev, {})
        u_fold = ucttt_folds.get(ev, {})

        t_acc = t_fold.get("test_accuracy", 0.0) * 100
        t_f1 = t_fold.get("test_macro_f1", 0.0) * 100
        u_acc = u_fold.get("test_accuracy", 0.0) * 100
        u_f1 = u_fold.get("test_macro_f1", 0.0) * 100

        d_acc = u_acc - t_acc
        d_f1 = u_f1 - t_f1

        d_acc_str = f"{'+' if d_acc >= 0 else ''}{d_acc:.2f}%"
        d_f1_str = f"{'+' if d_f1 >= 0 else ''}{d_f1:.2f}%"

        lines.append(f"{ev:<20} | {t_acc:6.2f}%   {t_f1:6.2f}%   | {u_acc:8.2f}%   {u_f1:8.2f}%   | {d_acc_str:<10} {d_f1_str:<10}")

        comparison_records.append({
            "event": ev,
            "ttt_accuracy": t_acc / 100.0,
            "ttt_macro_f1": t_f1 / 100.0,
            "ucttt_accuracy": u_acc / 100.0,
            "ucttt_macro_f1": u_f1 / 100.0,
            "delta_accuracy": d_acc / 100.0,
            "delta_macro_f1": d_f1 / 100.0,
        })

    lines.append("-" * 90)

    # Pooled Metrics
    t_pool_acc = ttt_data.get("pooled", {}).get("accuracy", 0.0) * 100
    t_pool_f1 = ttt_data.get("pooled", {}).get("macro_f1", 0.0) * 100
    u_pool_acc = ucttt_data.get("pooled", {}).get("accuracy", 0.0) * 100
    u_pool_f1 = ucttt_data.get("pooled", {}).get("macro_f1", 0.0) * 100

    d_pool_acc = u_pool_acc - t_pool_acc
    d_pool_f1 = u_pool_f1 - t_pool_f1

    d_pool_acc_str = f"{'+' if d_pool_acc >= 0 else ''}{d_pool_acc:.2f}%"
    d_pool_f1_str = f"{'+' if d_pool_f1 >= 0 else ''}{d_pool_f1:.2f}%"

    lines.append(f"{'POOLED (Tổng hợp)':<20} | {t_pool_acc:6.2f}%   {t_pool_f1:6.2f}%   | {u_pool_acc:8.2f}%   {u_pool_f1:8.2f}%   | {d_pool_acc_str:<10} {d_pool_f1_str:<10}")

    # Mean LOEO Metrics
    t_mean_acc = ttt_data.get("mean_loeo", {}).get("accuracy", 0.0) * 100
    t_mean_f1 = ttt_data.get("mean_loeo", {}).get("macro_f1", 0.0) * 100
    u_mean_acc = ucttt_data.get("mean_loeo", {}).get("accuracy", 0.0) * 100
    u_mean_f1 = ucttt_data.get("mean_loeo", {}).get("macro_f1", 0.0) * 100

    d_mean_acc = u_mean_acc - t_mean_acc
    d_mean_f1 = u_mean_f1 - t_mean_f1

    d_mean_acc_str = f"{'+' if d_mean_acc >= 0 else ''}{d_mean_acc:.2f}%"
    d_mean_f1_str = f"{'+' if d_mean_f1 >= 0 else ''}{d_mean_f1:.2f}%"

    lines.append(f"{'MEAN LOEO':<20} | {t_mean_acc:6.2f}%   {t_mean_f1:6.2f}%   | {u_mean_acc:8.2f}%   {u_mean_f1:8.2f}%   | {d_mean_acc_str:<10} {d_mean_f1_str:<10}")
    lines.append("=" * 90)

    full_table_text = "\n".join(lines)
    print(full_table_text)

    # Lưu so sánh ra file JSON
    out_comparison = {
        "benchmark": "PHEME 5 Events LOEO (Wu et al. 2025 Synchronized)",
        "config": {
            "d_model": 768,
            "nhead": 12,
            "num_tree_layers": 6,
            "fc_hidden_dim": 600,
            "num_time_windows": 8,
            "learning_rate": 4.2e-5,
            "patience": 8,
            "epochs": 35,
        },
        "pooled": {
            "ttt_accuracy": t_pool_acc / 100.0,
            "ttt_macro_f1": t_pool_f1 / 100.0,
            "ucttt_accuracy": u_pool_acc / 100.0,
            "ucttt_macro_f1": u_pool_f1 / 100.0,
            "delta_accuracy": d_pool_acc / 100.0,
            "delta_macro_f1": d_pool_f1 / 100.0,
        },
        "mean_loeo": {
            "ttt_accuracy": t_mean_acc / 100.0,
            "ttt_macro_f1": t_mean_f1 / 100.0,
            "ucttt_accuracy": u_mean_acc / 100.0,
            "ucttt_macro_f1": u_mean_f1 / 100.0,
            "delta_accuracy": d_mean_acc / 100.0,
            "delta_macro_f1": d_mean_f1 / 100.0,
        },
        "per_event": comparison_records,
    }

    comp_file = ROOT / "results/paper_sync_comparison.json"
    with open(comp_file, "w", encoding="utf-8") as f:
        json.dump(out_comparison, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu bảng so sánh chi tiết tại: {comp_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chạy đối chứng chuẩn bài báo Wu et al. 2025: TTT Baseline vs UC-TTT")
    parser.add_argument("--embeddings_path", default="data/processed/text_embeddings_bert.pt" if (ROOT / "data/processed/text_embeddings_bert.pt").exists() else "data/processed/text_embeddings.pt",
                        help="Đường dẫn file text embeddings")
    parser.add_argument("--window_mode", default="fixed_posts", choices=["fixed_posts", "fixed_windows"],
                        help="Chế độ chia cửa sổ thời gian ('fixed_posts': k=8 posts theo Wu et al. 2025)")
    parser.add_argument("--window_size", type=int, default=8,
                        help="Số bài viết / cửa sổ (k=8)")
    parser.add_argument("--num_time_windows", type=int, default=8,
                        help="Số cửa sổ tối đa hoặc số lát cắt (K=8)")
    parser.add_argument("--fusion_type", default="residual_gate", choices=["residual_gate", "gated", "cross_attention", "concatenation"],
                        help="Loại fusion trong UC-TTT")
    parser.add_argument("--epochs", type=int, default=35, help="Số epochs")
    parser.add_argument("--patience", type=int, default=8, help="Patience early stopping")
    parser.add_argument("--lr", type=float, default=4.2e-5, help="Learning rate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    py_exec = sys.executable

    # 1. Chạy Lần 1: TTT Baseline gốc (Wu et al. 2025)
    cmd_ttt = [
        py_exec,
        str(ROOT / "experiments/train_ttt_baseline.py"),
        "--embeddings_path", args.embeddings_path,
        "--events_mode", "5",
        "--window_mode", args.window_mode,
        "--window_size", str(args.window_size),
        "--d_model", "768",
        "--nhead", "12",
        "--num_tree_layers", "6",
        "--num_time_windows", str(args.num_time_windows),
        "--fc_hidden_dim", "600",
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--device", args.device,
    ]
    run_command(cmd_ttt, "LẦN 1: HUẤN LUYỆN TTT BASELINE (WU ET AL. 2025 - KHÔNG BẬT UC)")

    # 2. Chạy Lần 2: UC-TTT Proposed (bật User Credibility với Gated/Residual-Gate Fusion)
    cmd_ucttt = [
        py_exec,
        str(ROOT / "experiments/train_ucttt.py"),
        "--embeddings_path", args.embeddings_path,
        "--events_mode", "5",
        "--window_mode", args.window_mode,
        "--window_size", str(args.window_size),
        "--fusion_type", args.fusion_type,
        "--d_model", "768",
        "--nhead", "12",
        "--num_tree_layers", "6",
        "--num_time_windows", str(args.num_time_windows),
        "--fc_hidden_dim", "600",
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--device", args.device,
    ]
    run_command(cmd_ucttt, f"LẦN 2: HUẤN LUYỆN UC-TTT PROPOSED (BẬT USER CREDIBILITY - {args.fusion_type.upper()})")

    # 3. Tổng hợp và in bảng so sánh
    build_comparison_table()


if __name__ == "__main__":
    import torch
    main()

