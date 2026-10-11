import os
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import compute_all_metrics


class EarlyStopping:
    """
    Early Stopping theo dõi metric (mặc định: Validation Macro-F1).
    Nếu sau `patience` epochs không có cải thiện vượt quá `min_delta`, dừng huấn luyện.
    """
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
        save_path: Optional[str] = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.save_path = save_path
        self.counter = 0
        self.best_score = -float("inf") if mode == "max" else float("inf")
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, current_score: float, model: nn.Module) -> bool:
        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if self.save_path:
                os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                torch.save(self.best_model_state, self.save_path)
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def load_best_weights(self, model: nn.Module):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        elif self.save_path and os.path.exists(self.save_path):
            state = torch.load(self.save_path, weights_only=True)
            model.load_state_dict(state)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[Any] = None,
    clip_grad_norm: float = 1.0,
) -> Tuple[float, Dict[str, Any]]:
    """
    Huấn luyện 1 epoch.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in dataloader:
        text_features = batch["text_features"].to(device)
        depths = batch["depths"].to(device)
        node_padding_mask = batch["node_padding_mask"].to(device)
        time_windows = batch["time_window_assignments"].to(device)
        user_features = batch["user_features"].to(device) if "user_features" in batch else None
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        
        logits = model(
            text_features=text_features,
            depths=depths,
            user_features=user_features,
            time_window_assignments=time_windows,
            node_padding_mask=node_padding_mask,
        )

        loss = criterion(logits, labels)
        loss.backward()

        if clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)

        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(labels.detach().cpu().numpy())

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(len(all_targets), 1)
    metrics = compute_all_metrics(all_targets, all_preds)
    metrics["loss"] = avg_loss
    return avg_loss, metrics


def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """
    Đánh giá mô hình trên tập validation hoặc test.
    Trả về: (avg_loss, metrics, all_preds, all_probs, all_targets)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            text_features = batch["text_features"].to(device)
            depths = batch["depths"].to(device)
            node_padding_mask = batch["node_padding_mask"].to(device)
            time_windows = batch["time_window_assignments"].to(device)
            user_features = batch["user_features"].to(device) if "user_features" in batch else None
            labels = batch["labels"].to(device)

            logits = model(
                text_features=text_features,
                depths=depths,
                user_features=user_features,
                time_window_assignments=time_windows,
                node_padding_mask=node_padding_mask,
            )

            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(len(all_targets), 1)
    metrics = compute_all_metrics(all_targets, all_preds)
    metrics["loss"] = avg_loss

    return avg_loss, metrics, np.array(all_preds), np.array(all_probs), np.array(all_targets)
