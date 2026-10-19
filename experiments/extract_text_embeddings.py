"""Trich xuat offline text embeddings cho PHEME (Issue #1)."""
import argparse, pickle, re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

URL_RE = re.compile(r"https?://\S+|www\.\S+")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text):
    text = URL_RE.sub("", text or "")
    text = CTRL_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def thread_nodes(th):
    return [th["source_tweet"]] + list(th.get("reactions", []))


def collect_texts(threads):
    """Tra ve {tweet_id(str): text}, tu khu trung lap theo id."""
    id2text = {}
    for th in threads:
        for n in thread_nodes(th):
            tid = str(n["id"])
            if tid not in id2text:
                id2text[tid] = clean_text(n["text"])
    return id2text


@torch.no_grad()
def encode(texts, model_name, batch_size, device, pooling="max"):
    if model_name.startswith("sentence-transformers/"):
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model_name, device=device)
        emb = st.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_tensor=True)
        return emb.cpu().float()

    from transformers import AutoTokenizer, AutoModel
    print(f"Loading tokenizer & model: {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()

    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding ({model_name}, pool={pooling})"):
        b = tok(texts[i:i + batch_size], padding=True, truncation=True,
                max_length=128, return_tensors="pt").to(device)
        h = mdl(**b).last_hidden_state  # [B, L, D]
        m = b["attention_mask"].unsqueeze(-1).float()  # [B, L, 1]

        if pooling == "max":
            # Max-pooling theo cong thuc (1) trong bai bao Wu et al. (PLOS ONE 2025)
            h_masked = h.masked_fill(m == 0, -1e9)
            pooled = h_masked.max(dim=1).values
        elif pooling == "cls":
            pooled = h[:, 0, :]
        else:
            # Mean-pooling
            denom = m.sum(1).clamp(min=1e-9)
            pooled = (h * m).sum(1) / denom

        out.append(pooled.cpu().float())
    return torch.cat(out)


def sanity_check(emb_dict, threads):
    expected = {str(n["id"]) for th in threads for n in thread_nodes(th)}
    assert len(emb_dict) == len(expected), f"{len(emb_dict)} != {len(expected)}"
    assert set(emb_dict) == expected, "Tap tweet_id khong khop"
    dims = {tuple(v.shape) for v in emb_dict.values()}
    assert len(dims) == 1, f"Dim khong dong nhat: {dims}"
    assert all(torch.isfinite(v).all() for v in emb_dict.values()), "Co NaN/Inf"
    print(f"[OK] {len(emb_dict)} keys, dim = {dims.pop()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="bert-base-uncased",
                    help="Model HuggingFace: 'bert-base-uncased' (chuan bai bao Wu et al. 2025) hoac 'all-MiniLM-L6-v2'")
    ap.add_argument("--batch_size", type=int, default=128,
                    help="Batch size (128 hoac 64 cho GPU 4GB VRAM)")
    ap.add_argument("--pooling", default="max", choices=["max", "mean", "cls"],
                    help="Pooling: 'max' (theo Wu et al. 2025), 'mean', 'cls'")
    ap.add_argument("--input", default="data/processed/pheme_processed.pkl")
    ap.add_argument("--output", default="data/processed/text_embeddings_bert.pt")
    ap.add_argument("--limit", type=int, default=0,
                    help="chi encode N tweet dau de test nhanh (0 = tat ca)")
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42); torch.cuda.manual_seed_all(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    if device == "cuda":
        print("GPU Name:", torch.cuda.get_device_name(0))

    with open(args.input, "rb") as f:
        threads = pickle.load(f)

    id2text = collect_texts(threads)
    ids = list(id2text.keys())
    print(f"{len(threads)} threads, {len(ids)} tweet duy nhat")
    if args.limit:
        ids = ids[:args.limit]
        print(f"[TEST MODE] chi encode {len(ids)} tweet")

    emb = encode([id2text[i] for i in ids], args.model_name, args.batch_size, device, pooling=args.pooling)
    emb_dict = {tid: emb[k].clone() for k, tid in enumerate(ids)}

    if not args.limit:
        sanity_check(emb_dict, threads)
    else:
        print("shape mau:", next(iter(emb_dict.values())).shape)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out = args.output if not args.limit else args.output.replace(".pt", "_test.pt")
    torch.save(emb_dict, out)
    print("Saved:", out)


if __name__ == "__main__":
    main()
