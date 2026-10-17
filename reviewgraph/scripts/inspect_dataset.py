"""
inspect_dataset.py
------------------
Standalone script to inspect the raw ReviewGraph dataset and produce a
comprehensive JSON report.  Optionally generate a 100-row synthetic sample.

Usage
-----
  python inspect_dataset.py
  python inspect_dataset.py --dataset-path ../data/raw/fake_reviews.csv
  python inspect_dataset.py --sample
"""


from __future__ import annotations
import sys as _sys_utf8
if hasattr(_sys_utf8.stdout, 'reconfigure'): _sys_utf8.stdout.reconfigure(encoding='utf-8')

import argparse
import json
import os
import sys
import random
import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_stats(series: pd.Series) -> Dict[str, Any]:
    """Return basic numeric stats as plain-Python types."""
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "median": float(series.median()),
    }


def _value_counts_top(series: pd.Series, n: int = 10) -> Dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().head(n).items()}


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first column name (case-insensitive) that matches any candidate."""
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(df: pd.DataFrame, path: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {}

    # --- Basic shape ---
    report["source_file"] = str(path)
    report["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    report["columns"] = list(df.columns)
    report["dtypes"] = {c: str(df[c].dtype) for c in df.columns}
    report["memory_usage_mb"] = round(
        df.memory_usage(deep=True).sum() / 1_048_576, 4
    )

    # --- Sample values (first 3 rows per column) ---
    report["sample_values"] = {
        c: [str(v) for v in df[c].head(3).tolist()] for c in df.columns
    }

    # --- Missing values ---
    missing = df.isnull().sum()
    report["missing_values"] = {
        c: {
            "count": int(missing[c]),
            "pct": round(float(missing[c]) / max(len(df), 1) * 100, 2),
        }
        for c in df.columns
    }

    # --- Duplicates ---
    report["duplicate_rows"] = int(df.duplicated().sum())

    # --- Identify key columns ---
    label_col = _detect_column(df, ["label", "fake", "is_fake", "class", "category", "deceptive"])
    rating_col = _detect_column(df, ["rating", "stars", "score", "overall"])
    text_col = _detect_column(df, ["review_text", "text", "reviewtext", "review", "body", "content"])
    reviewer_col = _detect_column(df, ["reviewer_id", "reviewerid", "reviewer", "user_id", "userid", "user"])
    product_col = _detect_column(df, ["product_id", "productid", "asin", "product", "item_id", "item"])
    time_col = _detect_column(df, ["timestamp", "date", "review_date", "reviewtime", "time"])

    report["detected_columns"] = {
        "label": label_col,
        "rating": rating_col,
        "text": text_col,
        "reviewer": reviewer_col,
        "product": product_col,
        "timestamp": time_col,
    }

    # --- Label analysis ---
    if label_col:
        lv = df[label_col]
        report["label_analysis"] = {
            "column": label_col,
            "unique_values": [str(v) for v in lv.unique().tolist()],
            "distribution": _value_counts_top(lv, 20),
        }
    else:
        report["label_analysis"] = {"column": None, "note": "No label column detected"}

    # --- Rating distribution ---
    if rating_col:
        rv = pd.to_numeric(df[rating_col], errors="coerce").dropna()
        report["rating_distribution"] = {
            "column": rating_col,
            "unique_values": sorted([float(x) for x in rv.unique().tolist()]),
            "distribution": {str(k): int(v) for k, v in rv.value_counts().sort_index().items()},
            **_safe_stats(rv),
        }
    else:
        report["rating_distribution"] = {"column": None, "note": "No rating column detected"}

    # --- Text length stats ---
    if text_col:
        lengths = df[text_col].astype(str).str.split().str.len()
        char_lengths = df[text_col].astype(str).str.len()
        report["text_length_stats"] = {
            "column": text_col,
            "word_count": _safe_stats(lengths),
            "char_count": _safe_stats(char_lengths),
        }
    else:
        report["text_length_stats"] = {"column": None, "note": "No text column detected"}

    # --- Reviewer stats ---
    if reviewer_col:
        rcounts = df[reviewer_col].value_counts()
        report["reviewer_stats"] = {
            "column": reviewer_col,
            "unique_reviewers": int(rcounts.shape[0]),
            "avg_reviews_per_reviewer": round(float(rcounts.mean()), 4),
            "max_reviews_per_reviewer": int(rcounts.max()),
            "top_reviewers": {str(k): int(v) for k, v in rcounts.head(5).items()},
        }
    else:
        report["reviewer_stats"] = {"column": None, "note": "No reviewer column detected"}

    # --- Product stats ---
    if product_col:
        pcounts = df[product_col].value_counts()
        report["product_stats"] = {
            "column": product_col,
            "unique_products": int(pcounts.shape[0]),
            "avg_reviews_per_product": round(float(pcounts.mean()), 4),
            "max_reviews_per_product": int(pcounts.max()),
            "top_products": {str(k): int(v) for k, v in pcounts.head(5).items()},
        }
    else:
        report["product_stats"] = {"column": None, "note": "No product column detected"}

    # --- Timestamp range ---
    if time_col:
        try:
            ts = pd.to_datetime(df[time_col], errors="coerce").dropna()
            report["timestamp_range"] = {
                "column": time_col,
                "min": str(ts.min()),
                "max": str(ts.max()),
                "span_days": int((ts.max() - ts.min()).days),
            }
        except Exception as exc:
            report["timestamp_range"] = {"column": time_col, "error": str(exc)}
    else:
        report["timestamp_range"] = {"column": None, "note": "No timestamp column detected"}

    return report


# ---------------------------------------------------------------------------
# Column mapping guidance
# ---------------------------------------------------------------------------

GUIDANCE = """
========================================================================
COLUMN MAPPING GUIDANCE
========================================================================
ReviewGraph expects the following fields from your dataset.
Map your actual columns accordingly in reviewgraph/config/settings.py
or via environment variables:

  label        -> is_fake (bool/int) or textual label (OR/CG etc.)
  rating       -> rating (float 1-5)
  review_text  -> review_text (str)
  reviewer_id  -> reviewer_id (str)
  product_id   -> product_id (str / ASIN)
  timestamp    -> timestamp (datetime parseable str)

Columns detected automatically are listed in 'detected_columns' of the
report JSON.  If detection missed a column, rename it or update the
_detect_column() candidate lists in this script.
========================================================================
"""


# ---------------------------------------------------------------------------
# Synthetic sample generator
# ---------------------------------------------------------------------------

def _rand_str(prefix: str, n: int) -> str:
    return prefix + "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


FAKE_WORDS = [
    "great", "product", "love", "it", "quality", "terrible", "waste", "money",
    "arrived", "broken", "excellent", "recommend", "return", "shipping", "fast",
    "slow", "cheap", "durable", "nice", "okay", "worst", "best", "ever",
    "bought", "again", "price", "value", "worth", "impressive", "disappointing",
]


def _random_review() -> str:
    length = random.randint(5, 40)
    return " ".join(random.choices(FAKE_WORDS, k=length))


def generate_sample(n: int = 100) -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)

    base_date = datetime(2022, 1, 1)
    rows = []
    for i in range(n):
        reviewer_id = _rand_str("R", 10)
        product_id = _rand_str("B", 10)
        rating = float(random.choice([1, 2, 3, 4, 5]))
        label = random.choice(["CG", "OR"])  # CG=Computer-Generated, OR=Original
        text = _random_review()
        ts = base_date + timedelta(days=random.randint(0, 730))
        rows.append(
            {
                "review_id": f"REV{i:05d}",
                "reviewer_id": reviewer_id,
                "product_id": product_id,
                "rating": rating,
                "review_text": text,
                "label": label,
                "timestamp": ts.strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a ReviewGraph raw dataset and produce a JSON report.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset-path",
        default=str(Path(__file__).parent / ".." / "data" / "raw" / "fake_reviews.csv"),
        help="Path to raw CSV or JSON dataset file.\n"
             "Default: ../data/raw/fake_reviews.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / ".." / "data" / "processed"),
        help="Directory to save dataset_report.json.\nDefault: ../data/processed",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Generate and save a 100-row synthetic sample CSV to\n"
             "../data/sample/sample_reviews.csv",
    )
    parser.add_argument(
        "--sample-dir",
        default=str(Path(__file__).parent / ".." / "data" / "sample"),
        help="Directory to save the synthetic sample.\nDefault: ../data/sample",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------ sample
    if args.sample:
        sample_dir = Path(args.sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / "sample_reviews.csv"
        print(f"[inspect_dataset] Generating 100-row synthetic sample …")
        df_sample = generate_sample(100)
        df_sample.to_csv(sample_path, index=False)
        print(f"[inspect_dataset] Sample saved → {sample_path.resolve()}")
        print(df_sample.head(5).to_string(index=False))

    # ------------------------------------------------------------------ load
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(
            f"\n[ERROR] Dataset not found: {dataset_path.resolve()}\n"
            "  → Place your raw CSV/JSON file at the path above, OR\n"
            "  → Run with --sample to generate a synthetic file and\n"
            "    re-run without --sample pointing to that file.\n"
        )
        sys.exit(1)

    print(f"\n[inspect_dataset] Loading dataset from: {dataset_path.resolve()}")
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(dataset_path, low_memory=False)
    elif suffix in {".json", ".jsonl"}:
        try:
            df = pd.read_json(dataset_path)
        except ValueError:
            df = pd.read_json(dataset_path, lines=True)
    else:
        print(f"[ERROR] Unsupported file extension '{suffix}'. Use .csv or .json/.jsonl")
        sys.exit(1)

    print(f"[inspect_dataset] Loaded {len(df):,} rows × {len(df.columns)} columns")

    # ------------------------------------------------------------------ report
    print("[inspect_dataset] Building report …")
    report = build_report(df, str(dataset_path.resolve()))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dataset_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"[inspect_dataset] Report saved → {report_path.resolve()}")

    # ------------------------------------------------------------------ print summary
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"  Rows            : {report['shape']['rows']:,}")
    print(f"  Columns         : {report['shape']['columns']}")
    print(f"  Memory usage    : {report['memory_usage_mb']} MB")
    print(f"  Duplicate rows  : {report['duplicate_rows']:,}")
    print(f"  Column names    : {', '.join(report['columns'])}")

    det = report["detected_columns"]
    print("\n  Auto-detected column mapping:")
    for field, col in det.items():
        marker = "✓" if col else "✗"
        print(f"    {marker}  {field:<12} → {col or 'NOT FOUND'}")

    miss = report["missing_values"]
    any_missing = [c for c, v in miss.items() if v["count"] > 0]
    if any_missing:
        print(f"\n  Columns with missing values:")
        for c in any_missing:
            print(f"    {c}: {miss[c]['count']} ({miss[c]['pct']}%)")
    else:
        print("\n  No missing values found.")

    if report["label_analysis"]["column"]:
        la = report["label_analysis"]
        print(f"\n  Label column   : {la['column']}")
        print(f"  Label values   : {la['unique_values']}")
        print(f"  Distribution   : {la['distribution']}")

    if report["rating_distribution"]["column"]:
        rd = report["rating_distribution"]
        print(f"\n  Rating column  : {rd['column']}")
        print(f"  Distribution   : {rd['distribution']}")

    if report["text_length_stats"]["column"]:
        tl = report["text_length_stats"]
        wc = tl["word_count"]
        print(
            f"\n  Review text word count: "
            f"min={wc['min']:.0f}, max={wc['max']:.0f}, "
            f"mean={wc['mean']:.1f}, std={wc['std']:.1f}"
        )

    if report["reviewer_stats"]["column"]:
        rs = report["reviewer_stats"]
        print(f"\n  Unique reviewers         : {rs['unique_reviewers']:,}")
        print(f"  Avg reviews/reviewer     : {rs['avg_reviews_per_reviewer']}")

    if report["product_stats"]["column"]:
        ps = report["product_stats"]
        print(f"  Unique products          : {ps['unique_products']:,}")
        print(f"  Avg reviews/product      : {ps['avg_reviews_per_product']}")

    if report["timestamp_range"]["column"]:
        tr = report["timestamp_range"]
        print(f"\n  Timestamp range: {tr.get('min')} → {tr.get('max')} ({tr.get('span_days')} days)")

    print(GUIDANCE)


if __name__ == "__main__":
    main()
