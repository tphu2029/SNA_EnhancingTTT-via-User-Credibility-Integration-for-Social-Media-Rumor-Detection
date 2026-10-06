import os
import sys
import json
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.pheme_parser import load_cached_dataset
from src.baselines.tfidf_lr import TfidfLogisticRegressionBaseline
from src.baselines.propagation_rf import PropagationRandomForestBaseline

def print_result_table(res):
    print("=" * 70)
    print(f"MÔ HÌNH: {res['model_name']}")
    print(f"Pooled Accuracy: {res['overall_pooled_metrics']['accuracy']:.4f} | Pooled Macro-F1: {res['overall_pooled_metrics']['macro_f1']:.4f}")
    print(f"Mean LOEO Acc  : {res['mean_accuracy']:.4f} ± {res['std_accuracy']:.4f}")
    print(f"Mean LOEO F1   : {res['mean_macro_f1']:.4f} ± {res['std_macro_f1']:.4f}")
    print("-" * 70)
    print(f"{'Fold / Test Event':<25} {'Samples':<10} {'Accuracy':<12} {'Macro-F1':<12}")
    print("-" * 70)
    for f in res["fold_results"]:
        print(f"{f['test_event']:<25} {f['num_samples']:<10} {f['accuracy']:<12.4f} {f['macro_f1']:<12.4f}")
    print("=" * 70)

def main():
    data_path = r"e:\UIT\MXH\PJ\data\processed\pheme_processed.pkl"
    if not os.path.exists(data_path):
        print(f"Không tìm thấy file: {data_path}. Vui lòng chạy preprocess_pheme.py trước!")
        return

    print("Đang tải dữ liệu PHEME đã tiền xử lý...")
    dataset = load_cached_dataset(data_path)
    print(f"Đã tải {len(dataset)} threads thành công.\n")

    results = {}
    
    # Baseline 1a: TF-IDF + LR (Source only)
    print(">>> Đang chạy Baseline 1a: TF-IDF + Logistic Regression (Source Tweet Only)...")
    b1_source = TfidfLogisticRegressionBaseline(include_reactions=False)
    res_b1_source = b1_source.evaluate_loeo(dataset)
    print_result_table(res_b1_source)
    results["baseline1_source_lr"] = res_b1_source

    # Baseline 1b: TF-IDF + LR (Source + Reactions)
    print("\n>>> Đang chạy Baseline 1b: TF-IDF + Logistic Regression (Source + Reactions)...")
    b1_all = TfidfLogisticRegressionBaseline(include_reactions=True)
    res_b1_all = b1_all.evaluate_loeo(dataset)
    print_result_table(res_b1_all)
    results["baseline1_all_lr"] = res_b1_all

    # Baseline 2: Propagation Features + Random Forest
    print("\n>>> Đang chạy Baseline 2: Propagation Features + Random Forest...")
    b2 = PropagationRandomForestBaseline(n_estimators=200, max_depth=12)
    res_b2 = b2.evaluate_loeo(dataset)
    print_result_table(res_b2)
    results["baseline2_prop_rf"] = res_b2

    # Lưu kết quả JSON
    out_json = r"e:\UIT\MXH\PJ\results\baseline_results.json"
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu toàn bộ kết quả Baselines vào: {out_json}")

if __name__ == "__main__":
    main()
