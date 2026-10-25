"""
build_embeddings.py
-------------------
Generate sentence embeddings for reviews and build a FAISS index.

Usage
-----
  python build_embeddings.py
  python build_embeddings.py --use-sample
  python build_embeddings.py --force-rebuild
  python build_embeddings.py --model-name paraphrase-MiniLM-L3-v2
"""


from __future__ import annotations
import sys as _sys_utf8
if hasattr(_sys_utf8.stdout, 'reconfigure'): _sys_utf8.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch  # Initialize PyTorch before pandas on Windows
import pandas as pd


# ---------------------------------------------------------------------------
# Optional imports with friendly error messages
# ---------------------------------------------------------------------------

def _require(pkg: str, install: str) -> None:
    try:
        __import__(pkg)
    except ImportError:
        print(f"\n[ERROR] Package '{pkg}' is required.\n  Install: pip install {install}\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect(df: pd.DataFrame, candidates: list) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _index_is_valid(index_path: Path, meta_path: Path, n_expected: int) -> bool:
    """Return True if a valid FAISS index exists for the given review count."""
    if not index_path.exists() or not meta_path.exists():
        return False
    try:
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        return meta.get("n_vectors", 0) == n_expected
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Embedding & indexing
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    """Return (N, D) float32 numpy array of embeddings."""
    from sentence_transformers import SentenceTransformer
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    print(f"[embeddings] Loading model '{model_name}' …")
    model = SentenceTransformer(model_name)
    print(f"[embeddings] Encoding {len(texts):,} texts in batches of {batch_size} …")

    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    if has_tqdm:
        iterator = tqdm(range(n_batches), desc="Encoding", unit="batch")
    else:
        iterator = range(n_batches)

    for i in iterator:
        batch = texts[i * batch_size : (i + 1) * batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embs.astype(np.float32))

        if not has_tqdm:
            pct = min((i + 1) / n_batches * 100, 100)
            print(f"  Batch {i+1}/{n_batches} ({pct:.0f}%)", end="\r")

    if not has_tqdm:
        print()

    return np.vstack(all_embeddings)


def build_faiss_index(embeddings: np.ndarray):
    """Build a flat L2 FAISS index; upgrade to IVF for large datasets."""
    _require("faiss", "faiss-cpu")
    import faiss  # type: ignore

    d = embeddings.shape[1]
    n = embeddings.shape[0]

    if n < 10_000:
        print(f"[embeddings] Building FlatIP index (d={d}, n={n}) …")
        index = faiss.IndexFlatIP(d)
        # Normalise for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
    else:
        nlist = min(int(np.sqrt(n)), 256)
        print(f"[embeddings] Building IVF index (nlist={nlist}, d={d}, n={n}) …")
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
        faiss.normalize_L2(embeddings)
        index.train(embeddings)
        index.add(embeddings)

    return index


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sentence-embedding FAISS index for ReviewGraph.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--processed-path",
        default=str(Path(__file__).parent / ".." / "data" / "processed" / "reviews_processed.csv"),
        help="Path to processed reviews CSV.",
    )
    parser.add_argument(
        "--models-path",
        default=str(Path(__file__).parent / ".." / "models"),
        help="Directory to save FAISS index and embedding cache.",
    )
    parser.add_argument(
        "--model-name",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding (default: 64).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild index even if a valid one already exists.",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Load ../data/sample/sample_reviews.csv instead.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # --- Check dependencies ---
    _require("sentence_transformers", "sentence-transformers")
    _require("faiss", "faiss-cpu")
    import faiss  # type: ignore

    # --- Resolve paths ---
    if args.use_sample:
        proc_path = Path(__file__).parent / ".." / "data" / "sample" / "sample_reviews.csv"
    else:
        proc_path = Path(args.processed_path)

    if not proc_path.exists():
        print(
            f"\n[ERROR] Processed CSV not found: {proc_path.resolve()}\n"
            "  → Run preprocess_data.py first.\n"
        )
        sys.exit(1)

    models_dir = Path(args.models_path)
    models_dir.mkdir(parents=True, exist_ok=True)

    index_path = models_dir / "review_embeddings.faiss"
    cache_path = models_dir / "review_embeddings_cache.pkl"
    meta_path = models_dir / "review_embeddings_meta.json"
    id_map_path = models_dir / "review_id_map.pkl"

    # --- Load reviews ---
    print(f"[embeddings] Loading reviews: {proc_path.resolve()}")
    df = pd.read_csv(proc_path, low_memory=False)
    print(f"[embeddings] Loaded {len(df):,} reviews")

    text_col = _detect(df, ["review_text_clean", "review_text", "text", "reviewtext", "review", "body", "content"])
    if text_col is None:
        print("[ERROR] No text column found in processed CSV.")
        sys.exit(1)

    id_col = _detect(df, ["review_id", "reviewid", "id"])
    if id_col is None:
        print("[embeddings] WARNING: No review_id column; using integer index as ID.")
        df["review_id"] = [f"REV{i:07d}" for i in range(len(df))]
        id_col = "review_id"

    texts = df[text_col].fillna("").tolist()
    review_ids = df[id_col].tolist()
    n = len(texts)

    # --- Check existing index ---
    if not args.force_rebuild and _index_is_valid(index_path, meta_path, n):
        print(
            f"[embeddings] Valid index already exists for {n:,} reviews.\n"
            f"  Path: {index_path.resolve()}\n"
            "  Use --force-rebuild to regenerate."
        )
        return

    # --- Generate embeddings ---
    embeddings = embed_texts(texts, args.model_name, args.batch_size)
    print(f"[embeddings] Embeddings shape: {embeddings.shape}")

    # --- Build FAISS index ---
    index = build_faiss_index(embeddings)
    print(f"[embeddings] Index total vectors: {index.ntotal:,}")

    # --- Save ---
    faiss.write_index(index, str(index_path))
    print(f"[embeddings] FAISS index saved → {index_path.resolve()}")

    # Embedding cache (numpy array + review IDs)
    cache = {"embeddings": embeddings, "review_ids": review_ids}
    with open(cache_path, "wb") as fh:
        pickle.dump(cache, fh)
    print(f"[embeddings] Embedding cache saved → {cache_path.resolve()}")

    # ID map: integer index → review_id
    id_map = {i: rid for i, rid in enumerate(review_ids)}
    with open(id_map_path, "wb") as fh:
        pickle.dump(id_map, fh)
    print(f"[embeddings] ID map saved → {id_map_path.resolve()}")

    elapsed = time.time() - t0
    index_size_mb = index_path.stat().st_size / 1_048_576

    meta = {
        "model_name": args.model_name,
        "n_vectors": n,
        "embedding_dim": int(embeddings.shape[1]),
        "index_type": "FlatIP" if n < 10_000 else "IVFFlat",
        "index_size_mb": round(index_size_mb, 3),
        "elapsed_seconds": round(elapsed, 2),
        "text_column": text_col,
        "id_column": id_col,
    }
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[embeddings] Metadata saved → {meta_path.resolve()}")

    print("\n" + "=" * 60)
    print("EMBEDDING BUILD COMPLETE")
    print("=" * 60)
    print(f"  Model           : {args.model_name}")
    print(f"  Total embedded  : {n:,}")
    print(f"  Embedding dim   : {embeddings.shape[1]}")
    print(f"  Index size      : {index_size_mb:.2f} MB")
    print(f"  Time taken      : {elapsed:.1f}s")
    print(f"  Output dir      : {models_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
