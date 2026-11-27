#!/usr/bin/env python3
"""
Rerun ONLY Fold 2 (germanwings-crash) of TTT Baseline (5,447 threads)
using updated loss_stabilize (min_epochs=5, consecutive_stable=2) AND
also evaluating min_loss & max_val_f1 checkpoints from the same training trajectory,
then updating Fold 2 inside results/ttt_baseline_results.json cleanly.
"""

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from experiments.train_ttt_baseline import set_seed, split_stratified_loeo, PHEME_5_EVENTS
from src.data.dataset import PHEMETreeDataset, collate_tree_batch
from src.models.ucttt_classifier import UserCredibilityTemporalTreeTransformer
from src.training.engine import EarlyStopping, train_one_epoch, evaluate_epoch
from src.utils.metrics import compute_all_metrics


def main():
    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Rerunning Fold 2 (germanwings-crash) on 5,447 threads")

    with open(ROOT / "data/processed/pheme_processed.pkl", "rb") as f:
        threads = pickle.load(f)
    threads = [th for th in threads if th["event"] in PHEME_5_EVENTS and len(th.get("reactions", [])) >= 1]
    print(f"Filtered PHEME v5 threads: {len(threads):,}")

    embeddings = torch.load(ROOT / "data/processed/text_embeddings_bert.pt", weights_only=False)
    sample_emb = next(iter(embeddings.values()))
    text_dim = sample_emb.shape[0]

    fold_idx = 2
    test_ev = "germanwings-crash"
    train_threads, val_threads, test_threads = split_stratified_loeo(
        threads, test_event=test_ev, val_ratio=0.30, seed=seed + fold_idx
    )
    print(f"Fold 2 split: Train={len(train_threads)}, Val={len(val_threads)}, Test={len(test_threads)}")

    train_ds = PHEMETreeDataset(train_threads, embeddings, num_windows=8, window_size=8, window_mode="fixed_posts")
    val_ds = PHEMETreeDataset(val_threads, embeddings, num_windows=8, window_size=8, window_mode="fixed_posts")
    test_ds = PHEMETreeDataset(test_threads, embeddings, num_windows=8, window_size=8, window_mode="fixed_posts")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_tree_batch, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_tree_batch, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_tree_batch, num_workers=0)

    model = UserCredibilityTemporalTreeTransformer(
        text_input_dim=text_dim,
        d_model=768,
        nhead=12,
        num_tree_layers=6,
        num_time_windows=8,
        gru_hidden_dim=128,
        fc_hidden_dim=600,
        num_classes=2,
        use_user_credibility=False,
        window_aggregation="cumulative",
        dropout=0.2,
    ).to(device)

    epochs = 35
    optimizer = torch.optim.AdamW(model.parameters(), lr=4.2e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    ckpt_path = str(ROOT / "results/checkpoints/ttt_baseline/ttt_baseline_fold_2.pt")
    early_stopping = EarlyStopping(
        patience=8,
        mode="loss_stabilize",
        stabilize_threshold=0.015,
        min_epochs=5,
        consecutive_stable=2,
        save_path=ckpt_path,
    )

    best_val_f1 = -1.0
    max_f1_epoch = 0
    max_f1_state = None
    val_f1_at_stab = 0.0
    val_f1_at_min_loss = 0.0

    start = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_m = train_one_epoch(model, train_loader, optimizer, criterion, device, scheduler=scheduler, clip_grad_norm=1.0)
        val_loss, val_m, _, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        val_f1 = val_m["macro_f1"]

        prev_min_loss_ep = early_stopping.min_loss_epoch
        improved = early_stopping(val_loss, model, epoch=epoch)

        if early_stopping.min_loss_epoch == epoch and prev_min_loss_ep != epoch:
            val_f1_at_min_loss = val_f1
        if early_stopping.stabilized_epoch == epoch:
            val_f1_at_stab = val_f1
        if improved and not early_stopping.stabilized:
            val_f1_at_stab = val_f1

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            max_f1_epoch = epoch
            max_f1_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        mark = "*" if improved else " "
        print(
            f"  Epoch {epoch:02d}/{epochs:02d} {mark} | "
            f"TrLoss={tr_loss:.4f} F1={tr_m['macro_f1']:.4f} | "
            f"ValLoss={val_loss:.4f} Acc={val_m['accuracy']:.4f} F1={val_f1:.4f}",
            flush=True
        )

        # Keep going until BOTH stabilized patience is done AND min_loss has stopped improving for >=4 epochs
        if early_stopping.early_stop and (epoch - early_stopping.min_loss_epoch >= 4):
            print(f"  [Early Stopping] Stop at epoch {epoch}. loss_stabilize_v2={early_stopping.best_epoch}, min_loss={early_stopping.min_loss_epoch}, max_val_f1={max_f1_epoch}")
            break

    elapsed = time.time() - start

    def eval_state(name, state, ep, vf1):
        model.load_state_dict(state)
        _, tm, _, _, _ = evaluate_epoch(model, test_loader, criterion, device)
        rm = tm["class_metrics"]["Rumour"]
        print(f"  [{name}] Epoch {ep:02d} | Test Acc={tm['accuracy']*100:.2f}% | Macro-F1={tm['macro_f1']*100:.2f}% | Rumour P={rm['precision']*100:.2f}% R={rm['recall']*100:.2f}% F1={rm['f1']*100:.2f}% | CM={tm['confusion_matrix']}")
        return {"name": name, "epoch": ep, "val_f1": vf1, "metrics": tm}

    print("\n=== FOLD 2 (GERMANWINGS CRASH) CHECKPOINT COMPARISON ===")
    res_stab = eval_state("loss_stabilize_v2 (min_ep=5, consec=2)", early_stopping.best_model_state, early_stopping.best_epoch, val_f1_at_stab)
    res_minl = eval_state("min_loss", early_stopping.min_loss_state, early_stopping.min_loss_epoch, val_f1_at_min_loss)
    res_maxf = eval_state("max_val_f1", max_f1_state, max_f1_epoch, best_val_f1)

    # Save comparison report JSON
    comp = {
        "test_event": test_ev,
        "elapsed_seconds": round(elapsed, 2),
        "candidates": [
            {"strategy": c["name"], "best_epoch": c["epoch"], "val_macro_f1": c["val_f1"],
             "test_accuracy": c["metrics"]["accuracy"], "test_macro_f1": c["metrics"]["macro_f1"],
             "rumour_recall": c["metrics"]["class_metrics"]["Rumour"]["recall"],
             "rumour_f1": c["metrics"]["class_metrics"]["Rumour"]["f1"],
             "confusion_matrix": c["metrics"]["confusion_matrix"]}
            for c in [res_stab, res_minl, res_maxf]
        ]
    }
    with open(ROOT / "results/ttt_baseline/fold2_checkpoint_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comp, f, indent=2, ensure_ascii=False)

    # Update Fold 2 inside results/ttt_baseline_results.json using loss_stabilize_v2 (and save min_loss copy too)
    chosen = res_stab
    # Save best_model_state (loss_stabilize_v2) as ttt_baseline_fold_2.pt
    torch.save(early_stopping.best_model_state, ckpt_path)

    for res_path in [ROOT / "results/ttt_baseline_results.json", ROOT / "results/ttt_baseline/ttt_baseline_results.json"]:
        with open(res_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for i, fld in enumerate(data["folds"]):
            if fld["test_event"] == test_ev:
                tm = chosen["metrics"]
                data["folds"][i] = {
                    "fold_idx": 2,
                    "test_event": test_ev,
                    "val_event": "Stratified 30% train pool",
                    "best_epoch": chosen["epoch"],
                    "best_val_macro_f1": float(chosen["val_f1"]),
                    "test_accuracy": float(tm["accuracy"]),
                    "test_macro_f1": float(tm["macro_f1"]),
                    "test_macro_precision": float(tm["macro_precision"]),
                    "test_macro_recall": float(tm["macro_recall"]),
                    "class_metrics": tm["class_metrics"],
                    "confusion_matrix": tm["confusion_matrix"],
                    "training_time_seconds": round(elapsed, 2),
                    "checkpoint_note": f"loss_stabilize_v2 (min_epochs=5, consecutive_stable=2)",
                }
        accs = [x["test_accuracy"] for x in data["folds"]]
        f1s = [x["test_macro_f1"] for x in data["folds"]]
        data["mean_loeo"] = {
            "accuracy": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "macro_f1": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
        }
        # Recompute exact pooled confusion matrix by summing 5 fold confusion matrices!
        cm_sum = np.zeros((2, 2), dtype=int)
        for x in data["folds"]:
            cm_sum += np.array(x["confusion_matrix"], dtype=int)
        tn, fp, fn, tp = cm_sum[0, 0], cm_sum[0, 1], cm_sum[1, 0], cm_sum[1, 1]
        y_true = np.array([0]*tn + [0]*fp + [1]*fn + [1]*tp)
        y_pred = np.array([0]*tn + [1]*fp + [0]*fn + [1]*tp)
        pm = compute_all_metrics(y_true, y_pred)
        data["pooled"] = {
            "accuracy": pm["accuracy"],
            "macro_f1": pm["macro_f1"],
            "macro_precision": pm["macro_precision"],
            "macro_recall": pm["macro_recall"],
            "class_metrics": pm["class_metrics"],
            "confusion_matrix": pm["confusion_matrix"],
        }
        data["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nUpdated results/ttt_baseline_results.json! New Mean LOEO F1={data['mean_loeo']['macro_f1']*100:.2f}%, Pooled F1={data['pooled']['macro_f1']*100:.2f}%")


if __name__ == "__main__":
    main()
