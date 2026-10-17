"""
dataset_service.py
==================
DatasetService for the ReviewGraph project.

Handles loading, preprocessing, and synthetic sample generation for
fake-review datasets. Designed to work with several common CSV schemas
(e.g. Amazon review dumps, Yelp challenge datasets, custom labelled sets).

Label convention (internal):
    0 = genuine / OR (Original Review)
    1 = suspicious / CG (Computer Generated / fake)
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-name aliases accepted at ingestion time
# ---------------------------------------------------------------------------
_COL_ALIASES: Dict[str, List[str]] = {
    "review_id":   ["review_id", "id", "reviewID"],
    "reviewer_id": ["reviewer_id", "reviewerID", "user_id", "userID"],
    "product_id":  ["product_id", "asin", "productID", "item_id", "itemID"],
    "review_text": ["review_text", "reviewContent", "text", "reviewText", "content"],
    "rating":      ["rating", "overall", "stars", "score"],
    "timestamp":   ["timestamp", "date", "unixReviewTime", "reviewTime", "time"],
    "label":       ["label", "fake", "class", "flagged", "fraud"],
}

# ---------------------------------------------------------------------------
# Label normalisation maps
# ---------------------------------------------------------------------------
_LABEL_MAP = {
    # string aliases
    "cg": 1, "fake": 1, "fraud": 1, "1": 1, "suspicious": 1, "deceptive": 1,
    "or": 0, "genuine": 0, "real": 0, "0": 0, "authentic": 0, "truthful": 0,
}


# ---------------------------------------------------------------------------
# Synthetic review templates
# ---------------------------------------------------------------------------
_GENUINE_TEMPLATES = [
    "I bought this {product} about {weeks} weeks ago and I have been using it every day since. "
    "The build quality is solid and it does exactly what it says on the tin. "
    "Delivery was fast and the packaging was secure. Would recommend to a friend.",

    "Decent {product} for the price. It is not the fanciest option on the market but it works "
    "reliably. I have had it for {weeks} weeks without any issues. "
    "Setup was straightforward and the instructions were clear.",

    "I was a bit hesitant at first because of the mixed reviews, but I am glad I went ahead. "
    "The {product} feels well made and performs as expected. "
    "Customer service responded quickly when I had a question about the warranty.",

    "This is my second time buying this {product}. The first one lasted {years} years before "
    "showing wear, which is pretty good value. The new version feels slightly improved. "
    "Happy with the purchase overall.",

    "Good value for money. The {product} arrived on time and was exactly as described. "
    "It is not perfect — the {flaw} could be better — but for the price I cannot complain. "
    "I would buy again.",

    "Bought this as a gift and the recipient loved it. The {product} looks great in person, "
    "better than the photos suggest. Packaging was neat and it arrived before the estimated date.",

    "Works as advertised. I did a lot of research before choosing this {product} and it has "
    "met my expectations. I have been using it for {weeks} weeks and no complaints so far.",
]

_SUSPICIOUS_TEMPLATES = [
    "AMAZING PRODUCT!!! Best {product} I have ever bought in my entire life!!! "
    "Everyone should buy this RIGHT NOW!! Five stars is not enough!!! "
    "Perfect perfect perfect perfect!!!",

    "I received this {product} for free in exchange for an honest review. "
    "It is absolutely fantastic and I would give it ten stars if I could. "
    "Buy buy buy you will not regret it!!!",

    "This {product} changed my life! I bought it yesterday and already I can see incredible "
    "results. Amazing quality, amazing price, amazing everything. HIGHLY RECOMMEND!!!",

    "Best product on the market period. Do not waste your time looking at other {product} options "
    "just get this one. Fast shipping. Great seller. Five stars.",

    "I am so happy with this {product}! It is the best thing I have ever purchased! "
    "My whole family loves it! Perfect in every way! No complaints at all whatsoever!!!",

    "WOW just WOW!! This {product} exceeded every single one of my expectations!! "
    "I have bought many similar products and none come close to this one!! "
    "ORDER NOW you will thank yourself later!!!",

    "Incredible product. Incredible seller. Incredible everything. "
    "This {product} is exactly what I needed and more. Will definitely be ordering again soon. "
    "Cannot recommend highly enough. A+++++",
]

_PRODUCTS = [
    "laptop stand", "wireless mouse", "USB hub", "mechanical keyboard",
    "webcam", "desk lamp", "monitor arm", "phone charger",
    "noise-cancelling headphones", "portable speaker",
]

_FLAWS = [
    "cable management", "manual", "colour options", "carrying case",
    "software", "warranty card", "packaging",
]


class DatasetService:
    """
    Provides dataset ingestion, preprocessing, and synthetic data utilities
    for the ReviewGraph fake-review detection pipeline.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Rename incoming columns to the canonical internal names."""
        rename_map: Dict[str, str] = {}
        available = {c.lower(): c for c in df.columns}

        for canonical, aliases in _COL_ALIASES.items():
            if canonical in df.columns:
                continue  # already present under the right name
            for alias in aliases:
                if alias.lower() in available and alias.lower() != canonical:
                    rename_map[available[alias.lower()]] = canonical
                    break

        if rename_map:
            logger.info("Renaming columns: %s", rename_map)
            df = df.rename(columns=rename_map)

        return df

    @staticmethod
    def _stable_id(row_series: pd.Series, prefix: str = "rev") -> str:
        """Create a deterministic hash-based ID from a row."""
        raw = "|".join(str(v) for v in row_series.values)
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        return f"{prefix}_{digest}"

    @staticmethod
    def _normalise_label(val) -> Optional[int]:
        """Map heterogeneous label values → 0 (genuine) / 1 (suspicious) / NaN."""
        if pd.isna(val):
            return np.nan
        key = str(val).strip().lower()
        return _LABEL_MAP.get(key, np.nan)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataset(self, path: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Load a CSV dataset from *path*.

        Returns
        -------
        df : pd.DataFrame | None
            The loaded dataframe, or None if loading failed.
        stats : dict
            Basic ingestion metadata (row count, detected columns, etc.)
        """
        stats: Dict = {
            "path": path,
            "success": False,
            "error": None,
            "rows_raw": 0,
            "columns_detected": [],
            "columns_missing": [],
        }

        if not os.path.exists(path):
            msg = f"Dataset file not found: {path}"
            logger.error(msg)
            stats["error"] = msg
            return None, stats

        try:
            df = pd.read_csv(path, low_memory=False)
            logger.info("Loaded %d rows × %d cols from '%s'", len(df), len(df.columns), path)
        except Exception as exc:
            msg = f"Failed to parse CSV: {exc}"
            logger.exception(msg)
            stats["error"] = msg
            return None, stats

        stats["rows_raw"] = len(df)
        df = self._rename_columns(df)

        detected = [c for c in _COL_ALIASES if c in df.columns]
        missing = [c for c in _COL_ALIASES if c not in df.columns]
        stats["columns_detected"] = detected
        stats["columns_missing"] = missing
        stats["success"] = True

        if missing:
            logger.warning("Missing expected columns: %s", missing)

        return df, stats

    # ------------------------------------------------------------------

    def preprocess(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean and enrich the raw dataframe.

        Returns
        -------
        processed_df : pd.DataFrame
        stats : dict
            Detailed preprocessing report.
        """
        stats: Dict = {}
        df = df.copy()

        # ---- 1. Duplicates ----------------------------------------------------
        n_before = len(df)
        df.drop_duplicates(inplace=True)
        stats["duplicates_removed"] = n_before - len(df)
        logger.info("Removed %d duplicate rows.", stats["duplicates_removed"])

        # ---- 2. ID normalisation ----------------------------------------------
        if "review_id" not in df.columns or df["review_id"].isna().any():
            text_cols = [c for c in ["review_text", "reviewer_id", "product_id"] if c in df.columns]
            if text_cols:
                df["review_id"] = df[text_cols].apply(
                    lambda row: self._stable_id(row, prefix="rev"), axis=1
                )
            else:
                df["review_id"] = [f"rev_{i:07d}" for i in range(len(df))]
            logger.info("Generated stable review_id values.")

        for id_col in ("reviewer_id", "product_id"):
            if id_col not in df.columns:
                df[id_col] = f"unknown_{id_col}"

        # ---- 3. Missing values -------------------------------------------------
        missing_counts: Dict[str, int] = {}
        for col in df.columns:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                missing_counts[col] = n_missing
                if col != "label":
                    # Fill non-label columns sensibly
                    if df[col].dtype == object:
                        df[col].fillna("", inplace=True)
                    else:
                        df[col].fillna(df[col].median(), inplace=True)
        stats["missing_values"] = missing_counts

        # ---- 4. Text features --------------------------------------------------
        if "review_text" in df.columns:
            df["review_length"] = df["review_text"].astype(str).str.len()
            df["word_count"] = df["review_text"].astype(str).str.split().str.len()
            df["char_count"] = df["review_length"]  # alias kept for clarity
            logger.info("Computed text length features.")
        else:
            df["review_length"] = 0
            df["word_count"] = 0
            df["char_count"] = 0

        # ---- 5. Reviewer behavioural stats ------------------------------------
        if "reviewer_id" in df.columns:
            reviewer_stats = (
                df.groupby("reviewer_id")
                .agg(
                    reviewer_review_count=("review_id", "count"),
                    reviewer_distinct_products=("product_id", pd.Series.nunique)
                    if "product_id" in df.columns
                    else ("review_id", "count"),
                )
                .reset_index()
            )
            df = df.merge(reviewer_stats, on="reviewer_id", how="left")

        # ---- 6. Product stats --------------------------------------------------
        if "product_id" in df.columns:
            product_stats_agg: Dict = {"product_review_count": ("review_id", "count")}
            if "rating" in df.columns:
                product_stats_agg["product_avg_rating"] = ("rating", "mean")
            product_stats = (
                df.groupby("product_id")
                .agg(**product_stats_agg)
                .reset_index()
            )
            df = df.merge(product_stats, on="product_id", how="left")

            if "rating" in df.columns and "product_avg_rating" in df.columns:
                df["rating_deviation"] = df["rating"] - df["product_avg_rating"]

        # ---- 7. Label normalisation --------------------------------------------
        if "label" in df.columns:
            df["label"] = df["label"].apply(self._normalise_label)
            label_counts = df["label"].value_counts(dropna=False).to_dict()
            stats["label_distribution"] = {str(k): int(v) for k, v in label_counts.items()}
            stats["labeled_rows"] = int(df["label"].notna().sum())
            stats["unlabeled_rows"] = int(df["label"].isna().sum())
        else:
            df["label"] = np.nan
            stats["label_distribution"] = {}
            stats["labeled_rows"] = 0
            stats["unlabeled_rows"] = len(df)

        stats["rows_processed"] = len(df)
        logger.info("Preprocessing complete. %d rows ready.", len(df))
        return df, stats

    # ------------------------------------------------------------------

    def get_stats(self, df: pd.DataFrame) -> Dict:
        """Return a comprehensive statistics dictionary for a (processed) dataframe."""
        stats: Dict = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
        }

        if "label" in df.columns:
            vc = df["label"].value_counts(dropna=False)
            stats["genuine_count"] = int(vc.get(0, 0))
            stats["suspicious_count"] = int(vc.get(1, 0))
            stats["unlabeled_count"] = int(df["label"].isna().sum())
            labeled = stats["genuine_count"] + stats["suspicious_count"]
            stats["labeled_rows"] = labeled
            stats["class_imbalance_ratio"] = (
                round(stats["genuine_count"] / stats["suspicious_count"], 3)
                if stats["suspicious_count"] > 0
                else None
            )
            stats["label_distribution"] = {
                "genuine": stats["genuine_count"],
                "suspicious": stats["suspicious_count"],
                "unlabeled": stats["unlabeled_count"],
            }
        else:
            stats["label_distribution"] = {}
            stats["labeled_rows"] = 0

        stats["missing_values"] = {col: int(df[col].isna().sum()) for col in df.columns}

        for col in ("review_length", "word_count", "rating"):
            if col in df.columns:
                stats[f"{col}_mean"] = round(float(df[col].mean()), 3)
                stats[f"{col}_std"] = round(float(df[col].std()), 3)
                stats[f"{col}_min"] = round(float(df[col].min()), 3)
                stats[f"{col}_max"] = round(float(df[col].max()), 3)

        if "reviewer_id" in df.columns:
            stats["unique_reviewers"] = int(df["reviewer_id"].nunique())
        if "product_id" in df.columns:
            stats["unique_products"] = int(df["product_id"].nunique())

        return stats

    # ------------------------------------------------------------------

    def load_or_create_sample(self, sample_path: str) -> pd.DataFrame:
        """
        Return a 100-row synthetic sample dataframe.

        If *sample_path* already exists the cached sample is loaded directly.
        Otherwise a fresh synthetic sample is generated and saved.

        All rows have ``is_synthetic=True`` so downstream code can filter them.
        """
        if os.path.exists(sample_path):
            logger.info("Loading existing synthetic sample from '%s'.", sample_path)
            try:
                df = pd.read_csv(sample_path)
                logger.info("Loaded %d synthetic rows.", len(df))
                return df
            except Exception as exc:
                logger.warning("Failed to load sample (%s); regenerating.", exc)

        logger.info("Generating 100-row synthetic sample …")
        rng = np.random.default_rng(seed=42)

        n_total = 100
        n_suspicious = 30
        n_genuine = 70

        rows = []
        for i in range(n_genuine):
            template = _GENUINE_TEMPLATES[i % len(_GENUINE_TEMPLATES)]
            product = _PRODUCTS[i % len(_PRODUCTS)]
            weeks = int(rng.integers(2, 20))
            years = int(rng.integers(1, 4))
            flaw = _FLAWS[i % len(_FLAWS)]
            text = template.format(product=product, weeks=weeks, years=years, flaw=flaw)

            rows.append({
                "review_id":   f"syn_genuine_{i:04d}",
                "reviewer_id": f"syn_user_{rng.integers(1, 60):04d}",
                "product_id":  f"syn_prod_{rng.integers(1, 20):04d}",
                "review_text": text,
                "rating":      int(rng.choice([3, 4, 4, 4, 5])),
                "timestamp":   int(rng.integers(1_300_000_000, 1_700_000_000)),
                "label":       0,
                "is_synthetic": True,
            })

        for i in range(n_suspicious):
            template = _SUSPICIOUS_TEMPLATES[i % len(_SUSPICIOUS_TEMPLATES)]
            product = _PRODUCTS[i % len(_PRODUCTS)]
            text = template.format(product=product)

            rows.append({
                "review_id":   f"syn_suspicious_{i:04d}",
                "reviewer_id": f"syn_user_{rng.integers(61, 80):04d}",
                "product_id":  f"syn_prod_{rng.integers(1, 20):04d}",
                "review_text": text,
                "rating":      5,   # suspicious reviews tend to be extreme
                "timestamp":   int(rng.integers(1_300_000_000, 1_700_000_000)),
                "label":       1,
                "is_synthetic": True,
            })

        df = pd.DataFrame(rows)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        Path(sample_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(sample_path, index=False)
        logger.info("Synthetic sample saved to '%s'.", sample_path)
        return df

    # ------------------------------------------------------------------

    def save_processed(self, df: pd.DataFrame, path: str) -> None:
        """Persist a processed dataframe to *path* (Parquet preferred, CSV fallback)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(path, index=False)
            logger.info("Saved processed data (%d rows) → '%s' (Parquet).", len(df), path)
        except Exception:
            csv_path = str(path).replace(".parquet", ".csv")
            df.to_csv(csv_path, index=False)
            logger.info("Saved processed data (%d rows) → '%s' (CSV fallback).", len(df), csv_path)

    def load_processed(self, path: str) -> Optional[pd.DataFrame]:
        """Load a previously saved processed dataframe from *path*."""
        if not os.path.exists(path):
            logger.warning("Processed file not found: '%s'.", path)
            return None
        try:
            if path.endswith(".parquet"):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, low_memory=False)
            logger.info("Loaded processed data (%d rows) from '%s'.", len(df), path)
            return df
        except Exception as exc:
            logger.exception("Failed to load processed data: %s", exc)
            return None
