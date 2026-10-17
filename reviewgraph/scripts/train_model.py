"""
train_model.py
--------------
Train a fake-review classifier on the preprocessed dataset.

Usage
-----
  python train_model.py
  python train_model.py --use-sample
  python train_model.py --model-type random_forest
  python train_model.py --model-type both
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
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect(df: pd.DataFrame, candidates: list) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def build_features(df: pd.DataFrame) -> Tuple[pd.Series, Optional[pd.Series]]:
    """Return (text_series, label_series).  label_series is None if unavailable."""
    text_col = _detect(df, ["review_text_clean", "review_text", "text", "reviewtext", "review", "body", "content"])
    label_col = _detect(df, ["is_fake", "label", "fake", "class", "category", "deceptive"])

    if text_col is None:
        print("[train] ERROR: no text column found in processed dataset.")
        sys.exit(1)

    X = df[text_col].fillna("")
    y = None

    if label_col:
        raw = df[label_col]
        # Normalise to int
        if raw.dtype == object or raw.dtype.name == "category":
            def _norm(v: Any) -> int:
                return 1 if str(v).strip().upper() in {"1", "TRUE", "FAKE", "CG", "YES", "DECEPTIVE"} else 0
            y = raw.apply(_norm)
        else:
            y = raw.fillna(0).astype(int)

    return X, y


def build_pipeline(model_type: str) -> Pipeline:
    vectorizer = TfidfVectorizer(
        max_features=20_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )

    if model_type == "logistic":
        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="saga",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([("tfidf", vectorizer), ("clf", clf)])


def evaluate_pipeline(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    model_name: str,
) -> Dict[str, Any]:
    y_pred = pipeline.predict(X_test)

    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = float(roc_auc_score(y_test, y_prob))
    else:
        roc_auc = None

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "classification_report": report_dict,
        "test_set_size": len(y_test),
        "positive_class_ratio": round(float(y_test.mean()), 4),
    }


def print_metrics_table(results: list) -> None:
    print("\n" + "=" * 72)
    print(f"{'MODEL':<22} {'ACC':>6} {'PREC':>6} {'REC':>6} {'F1':>6} {'AUC':>7}")
    print("-" * 72)
    for r in results:
        auc_str = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "  N/A "
        print(
            f"{r['model']:<22} "
            f"{r['accuracy']:>6.4f} "
            f"{r['precision']:>6.4f} "
            f"{r['recall']:>6.4f} "
            f"{r['f1_score']:>6.4f} "
            f"{auc_str:>7}"
        )
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train fake-review classifier for ReviewGraph.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--processed-path",
        default=str(Path(__file__).parent / ".." / "data" / "processed" / "reviews_processed.csv"),
        help="Path to processed CSV from preprocess_data.py.",
    )
    parser.add_argument(
        "--models-path",
        default=str(Path(__file__).parent / ".." / "models"),
        help="Directory to save trained models and evaluation results.",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic", "random_forest", "both"],
        default="both",
        help="Which classifier to train (default: both).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2).",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Load ../data/sample/sample_reviews.csv as the processed input.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Cross-validation folds (default: 5). Set 0 to skip CV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # --- Resolve paths ---
    if args.use_sample:
        proc_path = Path(__file__).parent / ".." / "data" / "sample" / "sample_reviews.csv"
    else:
        proc_path = Path(args.processed_path)

    if not proc_path.exists():
        print(
            f"\n[ERROR] Processed dataset not found: {proc_path.resolve()}\n"
            "  → Run preprocess_data.py first (optionally with --use-sample).\n"
        )
        sys.exit(1)

    models_dir = Path(args.models_path)
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    print(f"[train] Loading: {proc_path.resolve()}")
    df = pd.read_csv(proc_path, low_memory=False)
    print(f"[train] Shape: {df.shape}")

    # --- Check labels ---
    X, y = build_features(df)

    if y is None:
        print(
            "\n[ERROR] No label column found in the processed dataset.\n"
            "  The classifier requires labelled data (is_fake, label, fake, etc.).\n"
            "  Please ensure your raw dataset has a label column and re-run\n"
            "  preprocess_data.py, or use a sample that includes labels.\n"
            "  Exiting without training.\n"
        )
        sys.exit(1)

    print(f"[train] Samples: {len(X):,}  |  Positive (fake): {y.sum():,}  |  Negative: {(y==0).sum():,}")

    if y.nunique() < 2:
        print("[ERROR] Only one class present in labels. Cannot train a classifier.")
        sys.exit(1)

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )
    print(f"[train] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    types_to_train = (
        ["logistic", "random_forest"] if args.model_type == "both" else [args.model_type]
    )

    results = []
    best_f1 = -1.0
    best_pipeline: Optional[Pipeline] = None
    best_name = ""

    for mtype in types_to_train:
        print(f"\n[train] Training {mtype} …")
        pipeline = build_pipeline(mtype)

        # Optional cross-validation
        if args.cv_folds > 1:
            print(f"[train]   Cross-validating ({args.cv_folds} folds) …")
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42),
                scoring="f1",
                n_jobs=-1,
            )
            print(f"[train]   CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        pipeline.fit(X_train, y_train)
        res = evaluate_pipeline(pipeline, X_test, y_test, model_name=mtype)
        results.append(res)

        # Save individual model
        model_file = models_dir / f"review_classifier_{mtype}.pkl"
        with open(model_file, "wb") as fh:
            pickle.dump(pipeline, fh)
        print(f"[train]   Saved → {model_file.resolve()}")

        if res["f1_score"] > best_f1:
            best_f1 = res["f1_score"]
            best_pipeline = pipeline
            best_name = mtype

    # --- Save best model ---
    best_path = models_dir / "review_classifier.pkl"
    with open(best_path, "wb") as fh:
        pickle.dump(best_pipeline, fh)
    print(f"\n[train] Best model ({best_name}, F1={best_f1:.4f}) saved → {best_path.resolve()}")

    # --- Save evaluation results ---
    elapsed = time.time() - t0
    eval_output = {
        "training_file": str(proc_path.resolve()),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "best_model": best_name,
        "best_f1": round(best_f1, 4),
        "elapsed_seconds": round(elapsed, 2),
        "models": results,
    }
    eval_path = models_dir / "evaluation_results.json"
    # Merge with existing file if present (don't overwrite retrieval metrics)
    if eval_path.exists():
        try:
            with open(eval_path, "r", encoding="utf-8") as fh:
                existing = json.load(fh)
        except Exception:
            existing = {}
        existing.update(eval_output)
        eval_output = existing
    with open(eval_path, "w", encoding="utf-8") as fh:
        json.dump(eval_output, fh, indent=2)
    print(f"[train] Evaluation results saved → {eval_path.resolve()}")

    # --- Print metrics table ---
    print_metrics_table(results)
    print(f"\n[train] Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
