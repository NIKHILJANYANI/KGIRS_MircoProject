"""
test_dataset_service.py
-----------------------
Tests for DatasetService: loading, preprocessing, label normalization.
All tests use synthetic fixture data — never real dataset metrics.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from app.services.dataset_service import DatasetService


@pytest.fixture
def svc() -> DatasetService:
    return DatasetService()


class TestLoadSample:
    def test_load_sample_creates_synthetic_df(self, svc, tmp_data_dir):
        """load_or_create_sample returns a DataFrame with required columns."""
        sample_path = str(tmp_data_dir / "sample_reviews.csv")
        df = svc.load_or_create_sample(sample_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        required_cols = {"review_id", "reviewer_id", "product_id", "review_text",
                         "rating", "label", "is_synthetic"}
        assert required_cols.issubset(set(df.columns))

    def test_sample_is_labeled_synthetic(self, svc, tmp_data_dir):
        """All rows in the synthetic sample have is_synthetic=True."""
        sample_path = str(tmp_data_dir / "sample_reviews.csv")
        df = svc.load_or_create_sample(sample_path)
        assert df["is_synthetic"].all(), "Synthetic sample must have is_synthetic=True for every row"

    def test_sample_has_both_labels(self, svc, tmp_data_dir):
        """Synthetic sample contains both genuine (0) and suspicious (1) labels."""
        sample_path = str(tmp_data_dir / "sample_reviews.csv")
        df = svc.load_or_create_sample(sample_path)
        labels = set(df["label"].dropna().unique())
        assert 0 in labels or 0.0 in labels, "Sample must have genuine (0) labels"
        assert 1 in labels or 1.0 in labels, "Sample must have suspicious (1) labels"


class TestPreprocessing:
    def test_preprocess_removes_duplicates(self, svc, sample_df):
        """Duplicate rows are removed during preprocessing."""
        df_with_dupes = pd.concat([sample_df, sample_df.iloc[:3]], ignore_index=True)
        processed, stats = svc.preprocess(df_with_dupes)
        assert len(processed) == len(sample_df), "All duplicates should be removed"
        assert stats["duplicates_removed"] >= 3

    def test_preprocess_handles_missing_text(self, svc, sample_df):
        """Rows with missing review_text are handled (flagged/dropped, not crashed)."""
        df_missing = sample_df.copy()
        df_missing.loc[0, "review_text"] = None
        # Should not raise
        processed, stats = svc.preprocess(df_missing)
        assert isinstance(processed, pd.DataFrame)

    def test_preprocess_normalizes_label_cg_or(self, svc):
        """CG -> 1 (suspicious), OR -> 0 (genuine)."""
        df = pd.DataFrame({
            "review_id": ["r1", "r2"],
            "reviewer_id": ["rev1", "rev2"],
            "product_id": ["p1", "p1"],
            "review_text": ["good product", "bad product"],
            "rating": [5.0, 1.0],
            "label": ["CG", "OR"],
            "is_synthetic": [True, True],
        })
        processed, _ = svc.preprocess(df)
        label_col = next(c for c in processed.columns if "label" in c.lower())
        labels = processed[label_col].tolist()
        assert 1 in labels or 1.0 in labels, "CG should map to 1"
        assert 0 in labels or 0.0 in labels, "OR should map to 0"

    def test_preprocess_normalizes_label_fake_genuine(self, svc):
        """'fake' -> 1, 'genuine' -> 0."""
        df = pd.DataFrame({
            "review_id": ["r1", "r2"],
            "reviewer_id": ["rev1", "rev2"],
            "product_id": ["p1", "p1"],
            "review_text": ["great", "terrible"],
            "rating": [5.0, 1.0],
            "label": ["fake", "genuine"],
            "is_synthetic": [True, True],
        })
        processed, _ = svc.preprocess(df)
        label_col = next(c for c in processed.columns if "label" in c.lower())
        vals = set(processed[label_col].dropna().tolist())
        assert 1 in vals or 1.0 in vals
        assert 0 in vals or 0.0 in vals

    def test_preprocess_does_not_fill_missing_labels(self, svc, sample_df):
        """NaN labels remain NaN — never silently filled with 0."""
        df_missing_labels = sample_df.copy()
        df_missing_labels["label"] = None  # Wipe all labels
        processed, stats = svc.preprocess(df_missing_labels)
        label_col = next((c for c in processed.columns if "label" in c.lower()), None)
        if label_col:
            assert processed[label_col].isna().all(), \
                "Labels that were NaN must remain NaN — never auto-filled"

    def test_preprocess_computes_review_length(self, svc, sample_df):
        """Preprocessing adds word_count and char_count columns."""
        processed, _ = svc.preprocess(sample_df)
        assert "word_count" in processed.columns or "review_length" in processed.columns
        # At least one length feature must be present
        length_cols = [c for c in processed.columns if "count" in c or "length" in c]
        assert len(length_cols) > 0, "No review length features computed"


class TestGetStats:
    def test_get_stats_returns_complete_dict(self, svc, sample_df):
        """get_stats returns a dict with all expected top-level keys."""
        processed, _ = svc.preprocess(sample_df)
        stats = svc.get_stats(processed)
        required_keys = {"total_rows", "label_distribution", "missing_values"}
        assert required_keys.issubset(set(stats.keys())), \
            f"Missing stats keys. Got: {set(stats.keys())}"

    def test_get_stats_counts_labels(self, svc, sample_df):
        """Label distribution in stats matches actual label counts."""
        processed, _ = svc.preprocess(sample_df)
        stats = svc.get_stats(processed)
        dist = stats.get("label_distribution", {})
        # Should report both classes
        assert len(dist) >= 2 or stats.get("labeled_rows", 0) > 0
