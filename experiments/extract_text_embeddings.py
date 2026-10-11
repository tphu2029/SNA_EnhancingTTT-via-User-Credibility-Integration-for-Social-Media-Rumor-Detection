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
def encode(texts, model_name, batch_size, device):
    if model_name == "microsoft/deberta-v3-small":
        # fallback: mean pooling, output 768 chieu (khac 384 cua MiniLM!)
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name).to(device).eval()
        out = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            b = tok(texts[i:i + batch_size], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(device)
            h = mdl(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1)).cpu())
        return torch.cat(out)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model_name, device=device)
    emb = st.encode(texts, batch_size=batch_size, show_progress_bar=True,
                    convert_to_tensor=True)
    return emb.cpu().float()


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
    ap.add_argument("--model_name", default="all-MiniLM-L6-v2")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--input", default="data/processed/pheme_processed.pkl")
    ap.add_argument("--output", default="data/processed/text_embeddings.pt")
    ap.add_argument("--limit", type=int, default=0,
                    help="chi encode N tweet dau de test nhanh (0 = tat ca)")
    args = ap.parse_args()

    torch.manual_seed(42); np.random.seed(42); torch.cuda.manual_seed_all(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    with open(args.input, "rb") as f:
        threads = pickle.load(f)

    id2text = collect_texts(threads)
    ids = list(id2text.keys())
    print(f"{len(threads)} threads, {len(ids)} tweet duy nhat")
    if args.limit:
        ids = ids[:args.limit]
        print(f"[TEST MODE] chi encode {len(ids)} tweet")

    emb = encode([id2text[i] for i in ids], args.model_name, args.batch_size, device)
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
