"""
preprocess_data.py
------------------
Preprocessing pipeline for ReviewGraph.

Usage
-----
  python preprocess_data.py
  python preprocess_data.py --use-sample
  python preprocess_data.py --dataset-path ../data/raw/fake_reviews.csv
  python preprocess_data.py --similarity-threshold 0.8
"""


from __future__ import annotations
import sys as _sys_utf8
if hasattr(_sys_utf8.stdout, 'reconfigure'): _sys_utf8.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def _detect(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_CONTRACTION_MAP = {
    "can't": "cannot", "won't": "will not", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
    "'m": " am",
}

_CONTRACTION_RE = re.compile(
    "(" + "|".join(re.escape(k) for k in _CONTRACTION_MAP) + ")",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = _CONTRACTION_RE.sub(lambda m: _CONTRACTION_MAP[m.group(0).lower()], text)
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_reviewer_stats(df: pd.DataFrame, reviewer_col: str) -> pd.DataFrame:
    stats = (
        df.groupby(reviewer_col)
        .agg(
            review_count=(reviewer_col, "count"),
        )
        .reset_index()
    )
    stats.rename(columns={reviewer_col: "reviewer_id"}, inplace=True)

    # avg rating per reviewer
    rating_col = _detect(df, ["rating", "stars", "score", "overall"])
    if rating_col:
        avg_rating = (
            df.groupby(reviewer_col)[rating_col]
            .mean()
            .reset_index()
            .rename(columns={reviewer_col: "reviewer_id", rating_col: "avg_rating"})
        )
        stats = stats.merge(avg_rating, on="reviewer_id", how="left")

    # fake review ratio if label available
    label_col = _detect(df, ["label", "fake", "is_fake", "class", "deceptive"])
    if label_col:
        df_tmp = df.copy()
        df_tmp["_is_fake"] = df_tmp[label_col].apply(
            lambda x: 1 if str(x).strip().upper() in {"1", "TRUE", "FAKE", "CG", "YES", "DECEPTIVE"} else 0
        )
        fake_rate = (
            df_tmp.groupby(reviewer_col)["_is_fake"]
            .mean()
            .reset_index()
            .rename(columns={reviewer_col: "reviewer_id", "_is_fake": "fake_ratio"})
        )
        stats = stats.merge(fake_rate, on="reviewer_id", how="left")

    return stats


def compute_product_stats(df: pd.DataFrame, product_col: str) -> pd.DataFrame:
    stats = (
        df.groupby(product_col)
        .agg(review_count=(product_col, "count"))
        .reset_index()
    )
    stats.rename(columns={product_col: "product_id"}, inplace=True)

    rating_col = _detect(df, ["rating", "stars", "score", "overall"])
    if rating_col:
        avg_rating = (
            df.groupby(product_col)[rating_col]
            .mean()
            .reset_index()
            .rename(columns={product_col: "product_id", rating_col: "avg_rating"})
        )
        stats = stats.merge(avg_rating, on="product_id", how="left")

    label_col = _detect(df, ["label", "fake", "is_fake", "class", "deceptive"])
    if label_col:
        df_tmp = df.copy()
        df_tmp["_is_fake"] = df_tmp[label_col].apply(
            lambda x: 1 if str(x).strip().upper() in {"1", "TRUE", "FAKE", "CG", "YES", "DECEPTIVE"} else 0
        )
        fake_rate = (
            df_tmp.groupby(product_col)["_is_fake"]
            .mean()
            .reset_index()
            .rename(columns={product_col: "product_id", "_is_fake": "fake_ratio"})
        )
        stats = stats.merge(fake_rate, on="product_id", how="left")

    return stats


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def compute_similarity_pairs(
    texts: pd.Series,
    review_ids: pd.Series,
    threshold: float = 0.75,
    max_reviews: int = 5000,
) -> pd.DataFrame:
    print(f"[preprocess] Computing TF-IDF cosine similarity (threshold={threshold}) …")

    n = min(len(texts), max_reviews)
    texts_subset = texts.iloc[:n].fillna("").tolist()
    ids_subset = review_ids.iloc[:n].tolist()

    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(texts_subset)
    print(f"[preprocess]   TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Process in chunks to avoid OOM
    chunk_size = 500
    pairs: List[Dict[str, Any]] = []

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = tfidf_matrix[start:end]
        sim_block = cosine_similarity(chunk, tfidf_matrix)  # (chunk_size, n)

        for i, row in enumerate(sim_block):
            global_i = start + i
            for j in range(global_i + 1, n):
                score = float(row[j])
                if score >= threshold:
                    pairs.append(
                        {
                            "review_id_1": ids_subset[global_i],
                            "review_id_2": ids_subset[j],
                            "similarity_score": round(score, 6),
                        }
                    )

        progress = min(end, n)
        print(
            f"[preprocess]   Processed {progress}/{n} reviews, "
            f"pairs so far: {len(pairs):,}",
            end="\r",
        )

    print()  # newline after \r
    print(f"[preprocess] Found {len(pairs):,} similar pairs above threshold {threshold}")
    return pd.DataFrame(pairs) if pairs else pd.DataFrame(columns=["review_id_1", "review_id_2", "similarity_score"])


# ---------------------------------------------------------------------------
# Main preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing transformations and return augmented dataframe."""
    text_col = _detect(df, ["review_text", "text", "reviewtext", "review", "body", "content"])
    reviewer_col = _detect(df, ["reviewer_id", "reviewerid", "reviewer", "user_id", "userid", "user"])
    product_col = _detect(df, ["product_id", "productid", "asin", "product", "item_id", "item"])
    rating_col = _detect(df, ["rating", "stars", "score", "overall"])
    label_col = _detect(df, ["label", "fake", "is_fake", "class", "deceptive"])
    time_col = _detect(df, ["timestamp", "date", "review_date", "reviewtime", "time"])
    id_col = _detect(df, ["review_id", "reviewid", "id"])

    out = df.copy()

    # --- Ensure review_id ---
    if id_col and id_col != "review_id":
        out.rename(columns={id_col: "review_id"}, inplace=True)
    elif id_col is None:
        out.insert(0, "review_id", [f"REV{i:07d}" for i in range(len(out))])

    # --- Normalise reviewer / product ---
    if reviewer_col and reviewer_col != "reviewer_id":
        out.rename(columns={reviewer_col: "reviewer_id"}, inplace=True)
        reviewer_col = "reviewer_id"
    if product_col and product_col != "product_id":
        out.rename(columns={product_col: "product_id"}, inplace=True)
        product_col = "product_id"

    # --- Clean text ---
    if text_col:
        print(f"[preprocess] Cleaning text column '{text_col}' …")
        out["review_text_clean"] = out[text_col].apply(clean_text)
        out["word_count"] = out["review_text_clean"].str.split().str.len()
        out["char_count"] = out["review_text_clean"].str.len()
        if text_col != "review_text":
            out["review_text"] = out[text_col]
    else:
        print("[preprocess] WARNING: no text column detected; skipping text cleaning.")

    # --- Rating ---
    if rating_col:
        out["rating"] = pd.to_numeric(out[rating_col], errors="coerce")
        if rating_col != "rating":
            out.drop(columns=[rating_col], inplace=True, errors="ignore")

    # --- Label normalisation ---
    if label_col:
        def norm_label(x: Any) -> int:
            s = str(x).strip().upper()
            return 1 if s in {"1", "TRUE", "FAKE", "CG", "YES", "DECEPTIVE"} else 0

        out["is_fake"] = out[label_col].apply(norm_label)
        if label_col not in ("is_fake", "label"):
            out["label_raw"] = out[label_col]
    else:
        print("[preprocess] No label column found; 'is_fake' column will not be created.")

    # --- Timestamp ---
    if time_col:
        try:
            out["timestamp"] = pd.to_datetime(out[time_col], errors="coerce")
        except Exception:
            pass

    # --- Reviewer-level features ---
    if "reviewer_id" in out.columns:
        rc = out["reviewer_id"].value_counts()
        out["reviewer_review_count"] = out["reviewer_id"].map(rc).fillna(0).astype(int)

    # --- Model score placeholder (filled by classifier later) ---
    if "model_score" not in out.columns:
        out["model_score"] = np.nan

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ReviewGraph preprocessing pipeline.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        default=str(
            Path(__file__).parent / ".." / "data" / "external"
            / "amazon_reviews_electronics.csv"
        ),
        help="Path to raw CSV/JSON dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / ".." / "data" / "processed"),
        help="Directory to save processed artefacts.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.75,
        help="Cosine-similarity threshold for similarity pairs (default: 0.75).",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Load ../data/sample/sample_reviews.csv instead of raw dataset.",
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Skip the (slow) TF-IDF similarity step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # --- Resolve input path ---
    if args.use_sample:
        dataset_path = Path(__file__).parent / ".." / "data" / "sample" / "sample_reviews.csv"
    else:
        dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        print(
            f"\n[ERROR] Input file not found: {dataset_path.resolve()}\n"
            "  → Run inspect_dataset.py --sample first to generate a sample, then\n"
            "    run this script with --use-sample, OR provide the raw dataset.\n"
        )
        sys.exit(1)

    print(f"[preprocess] Loading: {dataset_path.resolve()}")
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        df_raw = pd.read_csv(dataset_path, low_memory=False)
    elif suffix in {".json", ".jsonl"}:
        try:
            df_raw = pd.read_json(dataset_path)
        except ValueError:
            df_raw = pd.read_json(dataset_path, lines=True)
    else:
        print(f"[ERROR] Unsupported extension '{suffix}'.")
        sys.exit(1)

    print(f"[preprocess] Raw shape: {df_raw.shape}")

    # --- Preprocess ---
    print("[preprocess] Running preprocessing …")
    df = preprocess(df_raw)
    print(f"[preprocess] Processed shape: {df.shape}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Save processed CSV ---
    proc_path = output_dir / "reviews_processed.csv"
    df.to_csv(proc_path, index=False)
    print(f"[preprocess] Saved processed reviews → {proc_path.resolve()}")

    # --- Reviewer stats ---
    reviewer_col = "reviewer_id" if "reviewer_id" in df.columns else None
    if reviewer_col:
        rstats = compute_reviewer_stats(df_raw, _detect(df_raw, ["reviewer_id", "reviewerid", "reviewer", "user_id", "userid", "user"]) or reviewer_col)
        rstats_path = output_dir / "reviewer_stats.csv"
        rstats.to_csv(rstats_path, index=False)
        print(f"[preprocess] Saved reviewer stats → {rstats_path.resolve()} ({len(rstats):,} reviewers)")

    # --- Product stats ---
    product_col_raw = _detect(df_raw, ["product_id", "productid", "asin", "product", "item_id", "item"])
    if product_col_raw:
        pstats = compute_product_stats(df_raw, product_col_raw)
        pstats_path = output_dir / "product_stats.csv"
        pstats.to_csv(pstats_path, index=False)
        print(f"[preprocess] Saved product stats → {pstats_path.resolve()} ({len(pstats):,} products)")

    # --- Similarity pairs ---
    if not args.skip_similarity:
        text_col_proc = "review_text_clean" if "review_text_clean" in df.columns else (
            _detect(df, ["review_text", "text", "reviewtext", "review", "body", "content"])
        )
        if text_col_proc and "review_id" in df.columns:
            sim_df = compute_similarity_pairs(
                df[text_col_proc],
                df["review_id"],
                threshold=args.similarity_threshold,
            )
            sim_path = output_dir / "similarity_pairs.csv"
            sim_df.to_csv(sim_path, index=False)
            print(f"[preprocess] Saved similarity pairs → {sim_path.resolve()} ({len(sim_df):,} pairs)")
        else:
            print("[preprocess] Skipping similarity: no text or review_id column found.")
    else:
        print("[preprocess] Similarity computation skipped (--skip-similarity).")

    # --- Report ---
    elapsed = time.time() - t0
    report = {
        "input_file": str(dataset_path.resolve()),
        "raw_shape": {"rows": int(df_raw.shape[0]), "columns": int(df_raw.shape[1])},
        "processed_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "output_dir": str(output_dir.resolve()),
        "elapsed_seconds": round(elapsed, 2),
        "columns_added": [c for c in df.columns if c not in df_raw.columns],
        "similarity_threshold": args.similarity_threshold,
        "has_labels": "is_fake" in df.columns,
    }
    report_path = output_dir / "preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"[preprocess] Saved report → {report_path.resolve()}")

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"  Input rows         : {report['raw_shape']['rows']:,}")
    print(f"  Output rows        : {report['processed_shape']['rows']:,}")
    print(f"  Columns added      : {report['columns_added']}")
    print(f"  Labels available   : {report['has_labels']}")
    print(f"  Elapsed            : {elapsed:.1f}s")
    print(f"  Outputs in         : {output_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
