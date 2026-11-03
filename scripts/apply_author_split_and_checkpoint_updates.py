# -*- coding: utf-8 -*-
"""
Script cap nhat ty le chia du lieu 7:3 (val_ratio=0.30) va chien luoc chon checkpoint
theo phan hoi truc tiep qua email cua tac gia Sirong Wu (Wu et al. PLOS ONE 2025).
"""

import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def update_train_ttt_baseline():
    file_path = "experiments/train_ttt_baseline.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update split_stratified_loeo default val_ratio
    content = content.replace(
        "def split_stratified_loeo(threads, test_event, val_ratio=0.15, seed=42):",
        "def split_stratified_loeo(threads, test_event, val_ratio=0.30, seed=42):"
    )

    # 2. Update parser arguments
    old_args = (
        '    parser.add_argument("--val_strategy", type=str, choices=["stratified", "event"], default="stratified", help="\'stratified\' (15% từ train events) hoặc \'event\' (1 event riêng làm val)")\n'
        '    parser.add_argument("--val_ratio", type=float, default=0.15)'
    )
    new_args = (
        '    parser.add_argument("--val_strategy", type=str, choices=["stratified", "event"], default="stratified", help="\'stratified\' (30% từ train events, chuẩn 7:3 Wu et al. 2025) hoặc \'event\' (1 event riêng làm val)")\n'
        '    parser.add_argument(\n'
        '        "--val_ratio",\n'
        '        type=float,\n'
        '        default=0.30,\n'
        '        help="Tỷ lệ validation split từ 4 sự kiện còn lại (mặc định 0.30 tức 7:3 train:val theo chuẩn Wu et al. 2025)",\n'
        '    )\n'
        '    parser.add_argument(\n'
        '        "--checkpoint_criterion",\n'
        '        type=str,\n'
        '        choices=["loss_stabilize", "min_loss", "max_f1"],\n'
        '        default="loss_stabilize",\n'
        '        help="Tiêu chí chọn checkpoint theo phản hồi tác giả Wu et al. 2025: \'loss_stabilize\' (vừa bắt đầu ổn định), \'min_loss\', hoặc \'max_f1\'",\n'
        '    )\n'
        '    parser.add_argument(\n'
        '        "--stabilize_threshold",\n'
        '        type=float,\n'
        '        default=0.015,\n'
        '        help="Ngưỡng độ biến thiên tương đối của validation loss để phát hiện điểm bắt đầu ổn định (mặc định 0.015 tức 1.5%)",\n'
        '    )'
    )
    content = content.replace(old_args, new_args)

    # 3. Update EarlyStopping instantiation
    old_es = (
        '        ckpt_path = os.path.join(checkpoint_dir, f"ttt_baseline_fold_{fold_idx}.pt")\n'
        '        early_stopping = EarlyStopping(\n'
        '            patience=args.patience,\n'
        '            mode="max",\n'
        '            save_path=ckpt_path if args.save_checkpoints else None,\n'
        '        )'
    )
    new_es = (
        '        ckpt_path = os.path.join(checkpoint_dir, f"ttt_baseline_fold_{fold_idx}.pt")\n'
        '        early_stopping = EarlyStopping(\n'
        '            patience=args.patience,\n'
        '            mode=args.checkpoint_criterion,\n'
        '            stabilize_threshold=args.stabilize_threshold,\n'
        '            save_path=ckpt_path if args.save_checkpoints else None,\n'
        '        )'
    )
    content = content.replace(old_es, new_es)

    # 4. Update training epoch checkpoint check
    old_epoch_check = (
        '            val_f1 = val_metrics["macro_f1"]\n'
        '            improved = early_stopping(val_f1, model)\n'
        '            if improved:\n'
        '                best_val_f1 = val_f1\n'
        '                best_epoch = epoch'
    )
    new_epoch_check = (
        '            val_f1 = val_metrics["macro_f1"]\n'
        '            score = val_loss if args.checkpoint_criterion in ["loss_stabilize", "min_loss"] else val_f1\n'
        '            improved = early_stopping(score, model, epoch=epoch)\n'
        '            if improved:\n'
        '                best_epoch = early_stopping.best_epoch\n'
        '                best_val_f1 = val_f1'
    )
    content = content.replace(old_epoch_check, new_epoch_check)

    # 5. Update early stopping print & load
    old_es_print = (
        '            if early_stopping.early_stop:\n'
        '                print(f"  [Early Stopping] Đạt ngưỡng kiên nhẫn tại epoch {epoch}. Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch}).")\n'
        '                break\n\n'
        '        # Tải lại weights tốt nhất cho Test\n'
        '        early_stopping.load_best_weights(model)'
    )
    new_es_print = (
        '            if early_stopping.early_stop:\n'
        '                print(f"  [Early Stopping] Đạt ngưỡng kiên nhẫn tại epoch {epoch}. Selected Checkpoint tại Epoch {early_stopping.best_epoch} (tiêu chí \'{args.checkpoint_criterion}\').")\n'
        '                break\n\n'
        '        # Tải lại weights tốt nhất cho Test\n'
        '        early_stopping.load_best_weights(model)\n'
        '        print(f"  -> Đã nạp checkpoint được chọn tại Epoch {early_stopping.best_epoch} (tiêu chí \'{args.checkpoint_criterion}\') để đánh giá Test set.")'
    )
    content = content.replace(old_es_print, new_es_print)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Đã cập nhật thành công: {file_path}")


def update_run_ablation():
    file_path = "experiments/run_ablation.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update split_stratified_loeo default val_ratio
    content = content.replace(
        "def split_stratified_loeo(threads, test_event, val_ratio=0.15, seed=42):",
        "def split_stratified_loeo(threads, test_event, val_ratio=0.30, seed=42):"
    )

    # 2. Update parser argument
    old_parser = '    parser.add_argument("--val_ratio", type=float, default=0.15)'
    new_parser = (
        '    parser.add_argument(\n'
        '        "--val_ratio",\n'
        '        type=float,\n'
        '        default=0.30,\n'
        '        help="Tỷ lệ validation split từ 4 sự kiện còn lại (mặc định 0.30 tức 7:3 train:val theo chuẩn Wu et al. 2025)",\n'
        '    )\n'
        '    parser.add_argument(\n'
        '        "--checkpoint_criterion",\n'
        '        type=str,\n'
        '        choices=["loss_stabilize", "min_loss", "max_f1"],\n'
        '        default="loss_stabilize",\n'
        '        help="Tiêu chí chọn checkpoint theo phản hồi tác giả Wu et al. 2025: \'loss_stabilize\' (vừa bắt đầu ổn định), \'min_loss\', hoặc \'max_f1\'",\n'
        '    )\n'
        '    parser.add_argument(\n'
        '        "--stabilize_threshold",\n'
        '        type=float,\n'
        '        default=0.015,\n'
        '        help="Ngưỡng độ biến thiên tương đối của validation loss để phát hiện điểm bắt đầu ổn định (mặc định 0.015 tức 1.5%)",\n'
        '    )'
    )
    content = content.replace(old_parser, new_parser)

    # 3. Update EarlyStopping instantiation
    old_es = (
        '        ckpt_path = checkpoint_dir / f"fold_{fold_idx}_{test_ev}.pt"\n'
        '        early_stopping = EarlyStopping(\n'
        '            patience=args.patience,\n'
        '            mode="max",\n'
        '            save_path=str(ckpt_path) if args.save_checkpoints else None,\n'
        '        )'
    )
    new_es = (
        '        ckpt_path = checkpoint_dir / f"fold_{fold_idx}_{test_ev}.pt"\n'
        '        early_stopping = EarlyStopping(\n'
        '            patience=args.patience,\n'
        '            mode=args.checkpoint_criterion,\n'
        '            stabilize_threshold=args.stabilize_threshold,\n'
        '            save_path=str(ckpt_path) if args.save_checkpoints else None,\n'
        '        )'
    )
    content = content.replace(old_es, new_es)

    # 4. Update training loop
    old_epoch = (
        '            val_f1 = val_metrics["macro_f1"]\n'
        '            improved = early_stopping(val_f1, model)'
    )
    new_epoch = (
        '            val_f1 = val_metrics["macro_f1"]\n'
        '            score = val_loss if args.checkpoint_criterion in ["loss_stabilize", "min_loss"] else val_f1\n'
        '            improved = early_stopping(score, model, epoch=epoch)'
    )
    content = content.replace(old_epoch, new_epoch)

    # 5. Update early stopping message
    old_stop = (
        '            if early_stopping.early_stop:\n'
        '                print(f"    [Early Stopping] Dừng sớm tại Epoch {epoch}. Best Val F1: {early_stopping.best_score:.4f}")\n'
        '                break\n\n'
        '        # Tải lại trọng số tốt nhất cho Test\n'
        '        early_stopping.load_best_weights(model)'
    )
    new_stop = (
        '            if early_stopping.early_stop:\n'
        '                print(f"    [Early Stopping] Dừng sớm tại Epoch {epoch}. Selected Checkpoint tại Epoch {early_stopping.best_epoch} (tiêu chí \'{args.checkpoint_criterion}\').")\n'
        '                break\n\n'
        '        # Tải lại trọng số tốt nhất cho Test\n'
        '        early_stopping.load_best_weights(model)\n'
        '        print(f"    -> Đã nạp checkpoint được chọn tại Epoch {early_stopping.best_epoch} (tiêu chí \'{args.checkpoint_criterion}\') để đánh giá Test set.")'
    )
    content = content.replace(old_stop, new_stop)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Đã cập nhật thành công: {file_path}")


def update_run_paper_sync():
    file_path = "experiments/run_paper_sync_benchmarks.py"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update parser arguments
    old_parser = '    parser.add_argument("--epochs", type=int, default=35, help="Số epochs")'
    new_parser = (
        '    parser.add_argument("--val_ratio", type=float, default=0.30, help="Tỷ lệ validation split từ 4 sự kiện còn lại (mặc định 0.30 tức 7:3)")\n'
        '    parser.add_argument("--checkpoint_criterion", default="loss_stabilize", choices=["loss_stabilize", "min_loss", "max_f1"], help="Tiêu chí chọn checkpoint")\n'
        '    parser.add_argument("--stabilize_threshold", type=float, default=0.015, help="Ngưỡng độ biến thiên tương đối của validation loss")\n'
        '    parser.add_argument("--epochs", type=int, default=35, help="Số epochs")'
    )
    content = content.replace(old_parser, new_parser)

    # 2. Update cmd_ttt
    old_cmd_ttt = (
        '        "--epochs", str(args.epochs),\n'
        '        "--patience", str(args.patience),\n'
        '        "--device", args.device,\n'
        '    ]'
    )
    new_cmd_ttt = (
        '        "--val_ratio", str(args.val_ratio),\n'
        '        "--checkpoint_criterion", args.checkpoint_criterion,\n'
        '        "--stabilize_threshold", str(args.stabilize_threshold),\n'
        '        "--epochs", str(args.epochs),\n'
        '        "--patience", str(args.patience),\n'
        '        "--device", args.device,\n'
        '    ]'
    )
    content = content.replace(old_cmd_ttt, new_cmd_ttt, 1)

    # 3. Update cmd_ucttt
    old_cmd_ucttt = (
        '        "--epochs", str(args.epochs),\n'
        '        "--patience", str(args.patience),\n'
        '        "--device", args.device,\n'
        '    ]'
    )
    new_cmd_ucttt = (
        '        "--val_ratio", str(args.val_ratio),\n'
        '        "--checkpoint_criterion", args.checkpoint_criterion,\n'
        '        "--stabilize_threshold", str(args.stabilize_threshold),\n'
        '        "--epochs", str(args.epochs),\n'
        '        "--patience", str(args.patience),\n'
        '        "--device", args.device,\n'
        '    ]'
    )
    # replace second occurrence
    content = content.replace(old_cmd_ucttt, new_cmd_ucttt)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Đã cập nhật thành công: {file_path}")


def update_colab_notebook():
    file_path = "notebooks/PHEME_UCTTT_Colab_Training.ipynb"
    if not os.path.exists(file_path):
        print(f"⚠️ Không tìm thấy: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            # Cell 15: run_paper_sync_benchmarks
            if "run_paper_sync_benchmarks.py" in src and "--val_ratio" not in src:
                src = src.replace(
                    "--fusion_type residual_gate \\",
                    "--fusion_type residual_gate \\\n    --val_ratio 0.30 \\\n    --checkpoint_criterion loss_stabilize \\"
                )
                cell["source"] = [line + "\n" for line in src.split("\n")[:-1]] + [src.split("\n")[-1]]

            # Cell 17: train_ttt_baseline
            if "train_ttt_baseline.py" in src and "--val_ratio" not in src:
                src = src.replace(
                    "--num_tree_layers 6 \\",
                    "--num_tree_layers 6 \\\n    --val_ratio 0.30 \\\n    --checkpoint_criterion loss_stabilize \\"
                )
                cell["source"] = [line + "\n" for line in src.split("\n")[:-1]] + [src.split("\n")[-1]]

            # Cell 18: train_ucttt
            if "train_ucttt.py" in src and "--val_ratio" not in src:
                src = src.replace(
                    "--num_tree_layers 6 \\",
                    "--num_tree_layers 6 \\\n    --val_ratio 0.30 \\\n    --checkpoint_criterion loss_stabilize \\"
                )
                cell["source"] = [line + "\n" for line in src.split("\n")[:-1]] + [src.split("\n")[-1]]

            # Cell 20, 21: run_ablation
            if "run_ablation.py" in src and "--val_ratio" not in src:
                src = src.replace(
                    "--fusion_type residual_gate \\",
                    "--fusion_type residual_gate \\\n    --val_ratio 0.30 \\\n    --checkpoint_criterion loss_stabilize \\"
                )
                cell["source"] = [line + "\n" for line in src.split("\n")[:-1]] + [src.split("\n")[-1]]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✅ Đã cập nhật thành công: {file_path}")


if __name__ == "__main__":
    update_train_ttt_baseline()
    update_run_ablation()
    update_run_paper_sync()
    update_colab_notebook()
