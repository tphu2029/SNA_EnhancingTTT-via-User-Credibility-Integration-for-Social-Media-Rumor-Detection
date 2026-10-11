"""
Module tính toán các chỉ số đánh giá mô hình trong quá trình huấn luyện và kiểm thử.
Kế thừa và chuẩn hóa từ src.utils.metrics.
"""
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.utils.metrics import compute_all_metrics, print_metrics_summary


def calculate_metrics_from_tensors(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    target_names: Tuple[str, str] = ("Non-rumour", "Rumour")
) -> Dict[str, Any]:
    """
    Tính metrics trực tiếp từ PyTorch Tensors.
    """
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy()
    return compute_all_metrics(y_true, y_pred, target_names=target_names)


__all__ = [
    "compute_all_metrics",
    "print_metrics_summary",
    "calculate_metrics_from_tensors"
]
