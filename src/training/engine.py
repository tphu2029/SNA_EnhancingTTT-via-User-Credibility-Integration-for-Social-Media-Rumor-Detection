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
    Early Stopping & Checkpoint Selection.
    Hỗ trợ các chiến lược:
      - 'loss_stabilize' (Chiến lược tác giả Wu et al. 2025 - PLOS ONE):
        Chọn checkpoint tại thời điểm Validation Loss vừa bắt đầu ổn định (stabilize),
        thay vì chọn epoch có loss tối thiểu tuyệt đối (tránh overfit vào tập validation).
      - 'min' / 'min_loss': Chọn checkpoint có Validation Loss thấp nhất.
      - 'max' / 'max_f1': Chọn checkpoint có metric (ví dụ Macro-F1) cao nhất.
    """
    def __init__(
        self,
        patience: int = 8,
        min_delta: float = 1e-4,
        mode: str = "loss_stabilize",
        stabilize_threshold: float = 0.015,
        min_epochs: int = 3,
        save_path: Optional[str] = None,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.stabilize_threshold = stabilize_threshold
        self.min_epochs = min_epochs
        self.save_path = save_path

        self.counter = 0
        self.best_score = float("inf") if mode in ["min", "min_loss", "loss_stabilize"] else -float("inf")
        self.best_epoch = 0
        self.early_stop = False
        self.best_model_state = None

        # Lịch sử theo dõi cho loss_stabilize
        self.history: List[float] = []
        self.stabilized = False
        self.stabilized_epoch: Optional[int] = None
        self.stabilized_loss: Optional[float] = None
        self.min_loss = float("inf")
        self.min_loss_epoch = 0
        self.min_loss_state = None

    def __call__(self, current_score: float, model: nn.Module, epoch: Optional[int] = None) -> bool:
        ep = epoch if epoch is not None else (len(self.history) + 1)
        self.history.append(current_score)

        if self.mode == "loss_stabilize":
            val_loss = current_score
            improved = False

            # Cập nhật điểm min loss để dự phòng
            if val_loss < (self.min_loss - self.min_delta):
                self.min_loss = val_loss
                self.min_loss_epoch = ep
                self.min_loss_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Kiểm tra điều kiện bắt đầu ổn định theo Wu et al. 2025:
            # 1. Đã qua giai đoạn khởi tạo ban đầu (min_epochs)
            # 2. Chưa từng đánh dấu stabilize trước đó (chọn thời điểm 'just started to stabilize')
            if ep >= self.min_epochs and not self.stabilized and len(self.history) >= 2:
                prev_loss = self.history[-2]
                rel_decrease = (prev_loss - val_loss) / (abs(prev_loss) + 1e-8)

                # Điểm ổn định: độ giảm loss chậm lại rõ rệt (dưới ngưỡng stabilize_threshold, ví dụ < 1.5%)
                # và val_loss không bị bùng nổ phân kỳ so với mức thấp nhất đạt được (trong phạm vi 5%)
                is_stabilizing = (rel_decrease < self.stabilize_threshold) and (val_loss <= self.min_loss * 1.05)

                if is_stabilizing:
                    self.stabilized = True
                    self.stabilized_epoch = ep
                    self.stabilized_loss = val_loss
                    self.best_epoch = ep
                    self.best_score = val_loss
                    self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    improved = True

                    if self.save_path:
                        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                        torch.save(self.best_model_state, self.save_path)

                    print(
                        f"\n  [*] [Wu et al. 2025 Checkpoint Selection] Validation loss bat dau on dinh tai Epoch {ep} "
                        f"(Loss: {val_loss:.4f}, Rel Change: {rel_decrease * 100:.2f}% < {self.stabilize_threshold * 100:.1f}%). "
                        f"Da khoa checkpoint nay theo khuyen nghi tac gia!"
                    )

            # Quản lý Early Stopping
            if self.stabilized:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                if val_loss < (self.best_score - self.min_delta):
                    self.best_score = val_loss
                    self.best_epoch = ep
                    self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    self.counter = 0
                    improved = True
                    if self.save_path:
                        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                        torch.save(self.best_model_state, self.save_path)
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True

            return improved

        elif self.mode in ["min", "min_loss"]:
            improved = current_score < (self.best_score - self.min_delta)
            if improved:
                self.best_score = current_score
                self.best_epoch = ep
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

        else: # 'max' or 'max_f1'
            improved = current_score > (self.best_score + self.min_delta)
            if improved:
                self.best_score = current_score
                self.best_epoch = ep
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
        target_state = self.best_model_state
        if target_state is None and self.min_loss_state is not None:
            target_state = self.min_loss_state

        if target_state is not None:
            model.load_state_dict(target_state)
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
        num_windows = batch["num_windows"].to(device) if "num_windows" in batch else None
        user_features = batch["user_features"].to(device) if "user_features" in batch else None
        adj_mask = batch["adj_mask"].to(device) if "adj_mask" in batch else None
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        
        logits = model(
            text_features=text_features,
            depths=depths,
            user_features=user_features,
            time_window_assignments=time_windows,
            adj_mask=adj_mask,
            node_padding_mask=node_padding_mask,
            num_windows=num_windows,
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
            num_windows = batch["num_windows"].to(device) if "num_windows" in batch else None
            user_features = batch["user_features"].to(device) if "user_features" in batch else None
            adj_mask = batch["adj_mask"].to(device) if "adj_mask" in batch else None
            labels = batch["labels"].to(device)

            logits = model(
                text_features=text_features,
                depths=depths,
                user_features=user_features,
                time_window_assignments=time_windows,
                adj_mask=adj_mask,
                node_padding_mask=node_padding_mask,
                num_windows=num_windows,
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
