import json
import pickle
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import PHEMETreeDataset, collate_tree_batch
from src.models.ucttt_classifier import UserCredibilityTemporalTreeTransformer
from src.training.engine import evaluate_epoch
from src.utils.metrics import compute_all_metrics

def main():
    print("=" * 80)
    print("KIỂM TRA TOÀN DIỆN REPOSITORY: DATASET, 5 SỰ KIỆN, MODEL CHECKPOINTS & KẾT QUẢ")
    print("=" * 80)

    # 1. Check data
    data_path = ROOT / "data/processed/pheme_processed.pkl"
    bert_path = ROOT / "data/processed/text_embeddings_bert.pt"
    
    with open(data_path, "rb") as f:
        threads = pickle.load(f)
    print(f"1. Tổng số threads trong pheme_processed.pkl: {len(threads):,}")

    embs = torch.load(bert_path, weights_only=False)
    print(f"2. Embeddings BERT: {len(embs):,} items, dimension: {next(iter(embs.values())).shape[0]}")

    pheme_5_events = ["charliehebdo", "ferguson", "germanwings-crash", "ottawashooting", "sydneysiege"]
    threads_5 = [th for th in threads if th["event"] in pheme_5_events]
    print(f"3. Dữ liệu chuẩn 5 sự kiện (PHEME v5): {len(threads_5):,} threads")

    print("\nChi tiết 5 sự kiện benchmark (LOEO Folds):")
    for idx, ev in enumerate(pheme_5_events):
        ev_threads = [th for th in threads_5 if th["event"] == ev]
        n_rumour = sum(1 for th in ev_threads if th["label"] == 1)
        n_non = len(ev_threads) - n_rumour
        print(f"  Fold {idx}: {ev:20s} | Tổng={len(ev_threads):4d} | Non-rumour={n_non:4d} | Rumour={n_rumour:4d} ({n_rumour/len(ev_threads)*100:5.1f}%)")

    # 4. Check Checkpoints
    print("\n4. Kiểm tra các checkpoints đã lưu trong results/checkpoints/:")
    for idx, ev in enumerate(pheme_5_events):
        ttt_ckpt = ROOT / f"results/checkpoints/ttt_baseline/ttt_baseline_fold_{idx}.pt"
        uc_ckpt = ROOT / f"results/checkpoints/ucttt/ucttt_residual_gate_mode5_fold_{idx}.pt"
        print(f"  Fold {idx} ({ev}):")
        print(f"    - TTT: {ttt_ckpt.exists()} ({ttt_ckpt.stat().st_size:,} bytes)" if ttt_ckpt.exists() else f"    - TTT: Không tìm thấy!")
        print(f"    - UC : {uc_ckpt.exists()} ({uc_ckpt.stat().st_size:,} bytes)" if uc_ckpt.exists() else f"    - UC : Không tìm thấy!")

    # 5. Check results files
    print("\n5. Kiểm tra tính đồng bộ các file kết quả JSON trong results/:")
    paper_sync_file = ROOT / "results/paper_sync_comparison.json"
    if paper_sync_file.exists():
        with open(paper_sync_file, "r", encoding="utf-8") as f:
            sync_data = json.load(f)
        print("  - File paper_sync_comparison.json:")
        print(f"    TTT Pooled Acc:   {sync_data['pooled']['ttt_accuracy']*100:.2f}% | Macro-F1: {sync_data['pooled']['ttt_macro_f1']*100:.2f}%")
        print(f"    UC  Pooled Acc:   {sync_data['pooled']['ucttt_accuracy']*100:.2f}% | Macro-F1: {sync_data['pooled']['ucttt_macro_f1']*100:.2f}%")
        print(f"    TTT Mean LOEO F1: {sync_data['mean_loeo']['ttt_macro_f1']*100:.2f}%")
        print(f"    UC  Mean LOEO F1: {sync_data['mean_loeo']['ucttt_macro_f1']*100:.2f}%")
    else:
        print("  - Không tìm thấy results/paper_sync_comparison.json!")

if __name__ == "__main__":
    main()
