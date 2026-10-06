import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def compute_all_metrics(y_true, y_pred, target_names=("Non-rumour", "Rumour")):
    """
    Computes standard evaluation metrics for rumour detection.
    Target convention: 0 = Non-rumour, 1 = Rumour
    """
    acc = accuracy_score(y_true, y_pred)
    macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Class-specific scores
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "class_metrics": {
            target_names[i]: {
                "precision": float(prec_per_class[i]),
                "recall": float(rec_per_class[i]),
                "f1": float(f1_per_class[i]),
            }
            for i in range(len(target_names))
        },
        "confusion_matrix": cm,
    }

def print_metrics_summary(metrics, prefix=""):
    print(f"{prefix}Accuracy: {metrics['accuracy']:.4f}")
    print(f"{prefix}Macro-F1: {metrics['macro_f1']:.4f} (P: {metrics['macro_precision']:.4f}, R: {metrics['macro_recall']:.4f})")
    for cls_name, vals in metrics["class_metrics"].items():
        print(f"{prefix}  - {cls_name}: F1={vals['f1']:.4f}, P={vals['precision']:.4f}, R={vals['recall']:.4f}")
