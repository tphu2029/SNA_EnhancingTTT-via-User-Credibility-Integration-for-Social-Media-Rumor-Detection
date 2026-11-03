import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def load_ablation_data(input_path: Path) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file kết quả ablation: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def plot_feature_contribution(data: Dict[str, Any], output_path: Path):
    variants = data.get("variants", {})
    baseline = data.get("baseline_full") or variants.get("full")
    if not baseline:
        print("[!] Không tìm thấy cấu hình tham chiếu baseline_full trong dữ liệu.")
        return

    b_acc = baseline["mean_loeo"]["accuracy"] * 100
    b_f1 = baseline["mean_loeo"]["macro_f1"] * 100

    # Lọc các biến thể feature-level
    feature_order = [
        ("full", "UC-TTT (Full 12 features)"),
        ("without_profile", "w/o Profile Features"),
        ("without_counts", "w/o Raw Counts"),
        ("without_ratios", "w/o Derived Ratios"),
        ("source_only", "Source User Only (No Reactions)"),
        ("profile_only", "Profile Features Only"),
        ("counts_only", "Counts Features Only"),
        ("ratios_only", "Ratios Features Only"),
    ]

    names = []
    accs = []
    f1s = []
    delta_f1s = []
    delta_accs = []

    for key, display in feature_order:
        if key in variants:
            v = variants[key]
            names.append(display)
            acc = v["mean_loeo"]["accuracy"] * 100
            f1 = v["mean_loeo"]["macro_f1"] * 100
            accs.append(acc)
            f1s.append(f1)
            delta_f1s.append(v.get("delta_mean_loeo_macro_f1", f1 - b_f1))
            delta_accs.append(v.get("delta_mean_loeo_accuracy", acc - b_acc))

    if not names:
        print("[!] Không có biến thể feature ablation nào để vẽ.")
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.1, 1]})

    y_pos = np.arange(len(names))

    # --- PANEL 1: Giá trị Macro-F1 và Accuracy (%) ---
    bar_width = 0.38
    rects1 = ax1.barh(y_pos - bar_width / 2, f1s, bar_width, label="Macro-F1 (%)", color="#1f77b4", alpha=0.9, edgecolor="black")
    rects2 = ax1.barh(y_pos + bar_width / 2, accs, bar_width, label="Accuracy (%)", color="#2ca02c", alpha=0.85, edgecolor="black")

    ax1.axvline(b_f1, color="#d62728", linestyle="--", linewidth=1.5, label=f"Full Baseline F1 ({b_f1:.2f}%)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=11, fontweight="medium")
    ax1.invert_yaxis()  # Đưa full lên trên cùng
    ax1.set_xlabel("Hiệu năng (%)", fontsize=12, fontweight="bold")
    ax1.set_title("A. Hiệu năng Tuyệt đối theo Nhóm Đặc trưng Người dùng", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlim(left=max(0, min(f1s + accs) - 5), right=max(f1s + accs) + 5)
    ax1.legend(loc="lower right", frameon=True, fontsize=10)

    # Đính kèm số lên thanh bar
    for rect in rects1:
        w = rect.get_width()
        ax1.annotate(f"{w:.2f}%", (w, rect.get_y() + rect.get_height() / 2),
                     xytext=(4, 0), textcoords="offset points", va="center", ha="left", fontsize=9, fontweight="bold", color="#1f77b4")
    for rect in rects2:
        w = rect.get_width()
        ax1.annotate(f"{w:.2f}%", (w, rect.get_y() + rect.get_height() / 2),
                     xytext=(4, 0), textcoords="offset points", va="center", ha="left", fontsize=9, color="#2ca02c")

    # --- PANEL 2: Độ sụt giảm Macro-F1 (ΔF1) khi khuyết thiếu nhóm tính năng ---
    colors = ["#2b5c8f" if d == 0 else ("#e05252" if d < 0 else "#27ae60") for d in delta_f1s]
    bars_delta = ax2.barh(y_pos, delta_f1s, height=0.55, color=colors, edgecolor="black", alpha=0.9)

    ax2.axvline(0, color="black", linestyle="-", linewidth=1.2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # Ẩn nhãn bên phải cho gọn
    ax2.invert_yaxis()
    ax2.set_xlabel("Độ thay đổi Δ Macro-F1 (% points)", fontsize=12, fontweight="bold")
    ax2.set_title("B. Tác động của việc Khuyết thiếu Đặc trưng (Δ Macro-F1)", fontsize=13, fontweight="bold", pad=12)

    # Đính kèm số delta
    for bar, d in zip(bars_delta, delta_f1s):
        w = bar.get_width()
        offset = 5 if w >= 0 else -5
        ha = "left" if w >= 0 else "right"
        text_str = "0.00% (Ref)" if d == 0 else f"{d:+.2f}%"
        ax2.annotate(text_str, (w, bar.get_y() + bar.get_height() / 2),
                     xytext=(offset, 0), textcoords="offset points", va="center", ha=ha, fontsize=10, fontweight="bold")

    min_d = min(delta_f1s) if delta_f1s else 0
    max_d = max(delta_f1s) if delta_f1s else 0
    ax2.set_xlim(min(min_d * 1.3, -1.0), max(max_d * 1.3, 1.0))

    fig.suptitle("Phân tích Bóc tách Đặc trưng Người dùng (User Credibility Feature-Level Ablation)",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã xuất biểu đồ Feature Contribution tại: {output_path}")


def plot_comprehensive_ablation(data: Dict[str, Any], output_path: Path):
    variants = data.get("variants", {})
    baseline = data.get("baseline_full") or variants.get("full")
    if not baseline:
        return

    b_acc = baseline["mean_loeo"]["accuracy"] * 100
    b_f1 = baseline["mean_loeo"]["macro_f1"] * 100

    all_order = [
        ("full", "UC-TTT (Full)", "Baseline"),
        ("without_profile", "w/o Profile", "Feature"),
        ("without_counts", "w/o Raw Counts", "Feature"),
        ("without_ratios", "w/o Derived Ratios", "Feature"),
        ("source_only", "Source User Only", "Feature"),
        ("profile_only", "Profile Only", "Feature"),
        ("counts_only", "Counts Only", "Feature"),
        ("ratios_only", "Ratios Only", "Feature"),
        ("without_temporal_gru", "w/o Temporal GRU (Static Tree)", "Model"),
        ("without_tree_transformer", "w/o Tree Transformer", "Model"),
    ]

    names = []
    cats = []
    delta_f1s = []
    delta_accs = []
    f1s = []

    for key, display, cat in all_order:
        if key in variants:
            v = variants[key]
            names.append(display)
            cats.append(cat)
            f1 = v["mean_loeo"]["macro_f1"] * 100
            acc = v["mean_loeo"]["accuracy"] * 100
            f1s.append(f1)
            delta_f1s.append(v.get("delta_mean_loeo_macro_f1", f1 - b_f1))
            delta_accs.append(v.get("delta_mean_loeo_accuracy", acc - b_acc))

    if not names:
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(13, 7))

    x = np.arange(len(names))
    width = 0.35

    color_f1 = ["#1f77b4" if c == "Baseline" else ("#e67e22" if c == "Feature" else "#8e44ad") for c in cats]
    bars_f1 = ax.bar(x - width / 2, delta_f1s, width, label="Δ Macro-F1 (% points)", color="#c0392b", alpha=0.85, edgecolor="black")
    bars_acc = ax.bar(x + width / 2, delta_accs, width, label="Δ Accuracy (% points)", color="#2980b9", alpha=0.85, edgecolor="black")

    ax.axhline(0, color="black", linestyle="-", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=11, fontweight="medium")
    ax.set_ylabel("Mức độ Thay đổi (% points)", fontsize=12, fontweight="bold")
    ax.set_title("Tổng hợp Tác động của Bóc tách Đặc trưng & Kiến trúc Mô hình (Full Ablation Summary)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower left", fontsize=11, frameon=True)

    # Đính nhãn số trên các cột
    for bar in bars_f1:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(f"{h:+.2f}%" if h != 0 else "0.00%",
                    (bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -3), textcoords="offset points",
                    ha="center", va=va, fontsize=9, fontweight="bold", color="#922b21")

    for bar in bars_acc:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(f"{h:+.2f}%" if h != 0 else "0.00%",
                    (bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -3), textcoords="offset points",
                    ha="center", va=va, fontsize=9, color="#1b4f72")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Đã xuất biểu đồ Comprehensive Summary tại: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sinh biểu đồ trực quan hóa kết quả Ablation Studies")
    parser.add_argument(
        "--input",
        type=str,
        default=str(ROOT / "results/ablation/ablation_summary.json"),
        help="Đường dẫn file ablation_summary.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "results/figures/ablation_feature_contribution.png"),
        help="Đường dẫn file ảnh biểu đồ xuất ra",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = ROOT / input_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    print(f"Đang nạp số liệu từ: {input_path} ...")
    data = load_ablation_data(input_path)

    # 1. Vẽ biểu đồ đóng góp của đặc trưng (Feature Contribution)
    plot_feature_contribution(data, output_path)

    # 2. Vẽ biểu đồ tổng hợp cả kiến trúc và đặc trưng
    summary_fig_path = output_path.parent / "ablation_full_summary.png"
    plot_comprehensive_ablation(data, summary_fig_path)


if __name__ == "__main__":
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
