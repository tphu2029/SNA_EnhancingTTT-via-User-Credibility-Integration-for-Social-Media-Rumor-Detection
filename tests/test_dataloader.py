import pickle, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import PHEMETreeDataset, collate_tree_batch

torch.manual_seed(42)
np.random.seed(42)


def fake_node(i, t):
    return {"id": i, "text": "", "created_at": t,
            "credibility_features": np.zeros(12, np.float32)}


def test_toy_tree():
    """Cay: a -> b -> d, a -> c. Moi node nam trong 1 cua so rieng (N=4, K=4)."""
    a, b, c, d = "a", "b", "c", "d"
    th = {
        "thread_id": "toy", "event": "toy", "label": 1,
        "source_tweet": fake_node(a, "Mon Jan 05 10:00:00 +0000 2015"),
        "reactions": [fake_node(b, "Mon Jan 05 10:01:00 +0000 2015"),
                      fake_node(c, "Mon Jan 05 10:02:00 +0000 2015"),
                      fake_node(d, "Mon Jan 05 10:03:00 +0000 2015")],
        "edges": [(a, b), (a, c), (b, d)],
    }
    emb = {k: torch.zeros(384) for k in "abcd"}
    item = PHEMETreeDataset([th], emb)[0]

    assert item["tree_structure"]["depth"].tolist() == [0, 1, 1, 2]
    assert item["time_windows"].tolist() == [0, 1, 2, 3]
    adj = item["adj_mask"]
    assert adj[3, 1] and adj[1, 3], "d va b phai thay nhau"
    assert adj[3, 0] and adj[0, 3], "d va goc phai thay nhau"
    assert not adj[2, 3] and not adj[3, 2], "c va d khong lien quan"
    assert not adj[1, 2], "b va c khong lien quan"
    print("[OK] toy tree")


def test_real_batch():
    threads = pickle.load(open(ROOT / "data/processed/pheme_processed.pkl", "rb"))
    emb = torch.load(ROOT / "data/processed/text_embeddings.pt", weights_only=False)
    ds = PHEMETreeDataset(threads, emb)
    g = torch.Generator().manual_seed(42)
    dl = DataLoader(ds, batch_size=16, shuffle=True, generator=g,
                    collate_fn=collate_tree_batch, num_workers=0)
    it = iter(dl)
    batch = next(it)

    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"{k:26s} {tuple(v.shape)} {v.dtype}")

    B, N = batch["text_features"].shape[:2]
    pad = batch["node_padding_mask"]
    adj = batch["adj_mask"]
    assert batch["user_features"].shape == (B, N, 12)
    assert adj.shape == (B, N, N) and adj.dtype == torch.bool
    assert batch["depths"].shape == (B, N) and batch["depths"].max() <= 99
    tw = batch["time_window_assignments"]
    assert tw.shape == (B, N) and tw.min() >= 0 and tw.max() <= 3
    assert batch["labels"].shape == (B,) and set(batch["labels"].tolist()) <= {0, 1}
    assert not torch.isnan(batch["text_features"]).any()
    assert not torch.isnan(batch["user_features"]).any()
    assert (~pad).sum(1).tolist() == batch["num_nodes"].tolist()
    assert adj.any(-1).all(), "co hang mask rong -> NaN"
    for b in range(B):
        n = int(batch["num_nodes"][b])
        assert adj[b, :n, 0].all(), "moi node phai thay duoc tweet goc"
        assert not adj[b, :n, n:].any(), "node that khong duoc attend padding"
    print(f"[OK] batch shape: B={B}, max_nodes={N}")

    times = []
    for _ in range(10):
        t0 = time.time()
        next(it)
        times.append(time.time() - t0)
    mean = sum(times) / len(times)
    print(f"Thoi gian trung binh / batch: {mean:.3f}s (muc tieu < 0.1s)")
    if mean >= 0.1:
        print("[WARN] cham hon 0.1s")

    if torch.cuda.is_available():
        moved = {k: v.cuda() for k, v in batch.items() if torch.is_tensor(v)}
        print("CUDA OK, mem MB:", torch.cuda.memory_allocated() / 1e6)


if __name__ == "__main__":
    test_toy_tree()
    test_real_batch()
    print("ALL PASSED")