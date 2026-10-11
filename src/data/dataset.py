"""PHEMETreeDataset va collate_tree_batch (Issue #2)."""
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import Dataset

NUM_WINDOWS = 4
MAX_DEPTH = 99                      # SinusoidalPositionalEncoding(max_len=100)
_TIME_FMT = "%a %b %d %H:%M:%S %z %Y"


def _parse_time(s):
    try:
        return datetime.strptime(s, _TIME_FMT).timestamp()
    except (TypeError, ValueError):
        return 0.0


def build_tree(nodes, edges):
    """Tra ve (parent[N], depth[N], reach[N,N]).
    reach[i, j] = True neu j la to tien cua i hoac j == i.
    Node 0 la tweet goc. Node mo coi duoc gan vao goc."""
    n = len(nodes)
    id2idx = {str(x["id"]): i for i, x in enumerate(nodes)}
    parent = np.full(n, -1, dtype=np.int64)
    for p, c in edges:
        pi, ci = id2idx.get(str(p)), id2idx.get(str(c))
        if pi is None or ci is None or ci == 0 or pi == ci:
            continue
        parent[ci] = pi
    for i in range(1, n):
        if parent[i] == -1:
            parent[i] = 0

    depth = np.zeros(n, dtype=np.int64)
    reach = np.eye(n, dtype=bool)
    for i in range(1, n):
        cur, d, steps = i, 0, 0
        while cur != 0 and steps < n:          # steps: chong vong lap
            cur = parent[cur]
            reach[i, cur] = True
            d += 1
            steps += 1
        depth[i] = d
    return parent, depth, reach


def equal_depth_windows(times, k=NUM_WINDOWS):
    """Chia node thanh k cua so co so node xap xi bang nhau, theo thu tu thoi gian."""
    n = len(times)
    order = np.argsort(times, kind="stable")
    win = np.zeros(n, dtype=np.int64)
    win[order] = np.minimum(np.arange(n) * k // n, k - 1)
    return win


class PHEMETreeDataset(Dataset):
    def __init__(self, threads_data, text_embeddings, user_feature_extractor=None,
                 num_windows=NUM_WINDOWS):
        self.threads = threads_data
        self.emb = text_embeddings
        self.uext = user_feature_extractor
        self.k = num_windows

    def __len__(self):
        return len(self.threads)

    def _user_feat(self, node):
        # pkl da luu san 12 dac trung trong node["credibility_features"]
        if "credibility_features" in node:
            return np.asarray(node["credibility_features"], dtype=np.float32)
        if self.uext is not None:
            return np.asarray(self.uext.extract_from_tweet(node), dtype=np.float32)
        raise KeyError("Node khong co credibility_features va khong co extractor")

    def __getitem__(self, idx):
        th = self.threads[idx]
        nodes = [th["source_tweet"]] + list(th["reactions"])
        n = len(nodes)

        text = torch.stack([self.emb[str(x["id"])] for x in nodes]).float()      # [N, D]
        user = torch.from_numpy(np.stack([self._user_feat(x) for x in nodes]))   # [N, 12]

        parent, depth, reach = build_tree(nodes, th["edges"])
        times = np.array([_parse_time(x.get("created_at")) for x in nodes])
        win = equal_depth_windows(times, self.k)

        related = reach | reach.T                              # to tien <-> con chau
        same_win = win[:, None] == win[None, :]
        adj = related | same_win                               # True = duoc attend

        edges = [(int(parent[i]), i) for i in range(1, n)]
        return {
            "node_texts_emb": text,
            "node_users": user.float(),
            "tree_structure": {"edges": edges, "depth": torch.from_numpy(depth)},
            "timestamps": torch.from_numpy(times),
            "time_windows": torch.from_numpy(win),
            "adj_mask": torch.from_numpy(adj),
            "label": int(th["label"]),
            "event": th["event"],
            "thread_id": th["thread_id"],
            "num_nodes": n,
        }


def collate_tree_batch(batch):
    B = len(batch)
    max_n = max(b["num_nodes"] for b in batch)
    d_text = batch[0]["node_texts_emb"].shape[1]
    d_user = batch[0]["node_users"].shape[1]

    text = torch.zeros(B, max_n, d_text)
    user = torch.zeros(B, max_n, d_user)
    depths = torch.zeros(B, max_n, dtype=torch.long)
    windows = torch.zeros(B, max_n, dtype=torch.long)
    padding = torch.ones(B, max_n, dtype=torch.bool)           # True = node padding
    adj = torch.zeros(B, max_n, max_n, dtype=torch.bool)       # True = duoc attend

    for b, item in enumerate(batch):
        n = item["num_nodes"]
        text[b, :n] = item["node_texts_emb"]
        user[b, :n] = item["node_users"]
        depths[b, :n] = item["tree_structure"]["depth"].clamp(max=MAX_DEPTH)
        windows[b, :n] = item["time_windows"]
        padding[b, :n] = False
        adj[b, :n, :n] = item["adj_mask"]

    idx = torch.arange(max_n)
    adj[:, idx, idx] = True        # node padding tu attend chinh no, tranh softmax ra NaN

    return {
        "text_features": text,                       # [B, N, D_text]
        "user_features": user,                       # [B, N, 12]
        "depths": depths,                            # [B, N]
        "time_window_assignments": windows,          # [B, N] gia tri 0..K-1
        "adj_mask": adj,                             # [B, N, N] bool
        "attention_mask": adj,                       # ten theo issue (cung tensor)
        "node_padding_mask": padding,                # [B, N] True = padding
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "num_nodes": torch.tensor([b["num_nodes"] for b in batch]),
        "events": [b["event"] for b in batch],
        "thread_ids": [b["thread_id"] for b in batch],
    }


def to_additive_mask(adj_mask, nhead):
    """bool [B,N,N] (True = duoc attend) -> float [B*nhead,N,N] (0 / -inf)
    dung cho nn.MultiheadAttention (xem tree_transformer.py)."""
    m = torch.zeros(adj_mask.shape, dtype=torch.float32).masked_fill(~adj_mask, float("-inf"))
    return m.repeat_interleave(nhead, dim=0)