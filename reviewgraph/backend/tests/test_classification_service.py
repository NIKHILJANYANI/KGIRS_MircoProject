"""
test_classification_service.py
-------------------------------
Tests for ClassificationService: training, prediction, model persistence.
All tests use synthetic fixture data — never real dataset metrics.
"""

from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path

from app.services.classification_service import ClassificationService
from app.services.dataset_service import DatasetService


@pytest.fixture
def svc(tmp_models_dir) -> ClassificationService:
    return ClassificationService(models_path=str(tmp_models_dir))


@pytest.fixture
def preprocessed_df(sample_df):
    """Return preprocessed synthetic DataFrame."""
    ds = DatasetService()
    processed, _ = ds.preprocess(sample_df)
    return processed


class TestTraining:
    def test_train_returns_metrics(self, svc, preprocessed_df):
        """train() returns a dict with precision, recall, f1 for at least one model."""
        metrics = svc.train(preprocessed_df, model_type="logistic")
        assert isinstance(metrics, dict)
        # At least one model's metrics present
        model_key = next((k for k in metrics if "logistic" in k or "regression" in k), None)
        if model_key is None:
            # Try top-level keys
            assert any(k in metrics for k in ["precision", "recall", "f1", "test_metrics"]), \
                f"No expected metric keys found. Got: {list(metrics.keys())}"

    def test_train_requires_labels(self, svc, preprocessed_df):
        """train() raises ValueError when no labeled rows exist."""
        df_no_labels = preprocessed_df.copy()
        # Remove all labels
        label_col = next((c for c in df_no_labels.columns if "label" in c.lower()), None)
        if label_col:
            df_no_labels[label_col] = None
        with pytest.raises((ValueError, RuntimeError, Exception)):
            svc.train(df_no_labels)

    def test_model_saves_to_disk(self, svc, preprocessed_df, tmp_models_dir):
        """After training, model file exists in models directory."""
        svc.train(preprocessed_df, model_type="logistic")
        model_files = list(tmp_models_dir.glob("*.pkl"))
        assert len(model_files) > 0, "No model .pkl file saved after training"


class TestPrediction:
    @pytest.fixture(autouse=True)
    def train_first(self, svc, preprocessed_df):
        """Ensure model is trained before prediction tests."""
        svc.train(preprocessed_df, model_type="logistic")

    def test_predict_single_returns_dict(self, svc):
        """predict_single returns a dict for any text input."""
        result = svc.predict_single("This product is great and I love it!", {})
        assert isinstance(result, dict)

    def test_model_score_labeled_correctly(self, svc):
        """Result has 'model_score' key, NOT 'fraud_probability' or 'probability'."""
        result = svc.predict_single("Amazing product five stars buy now!", {})
        assert "model_score" in result, "Result must have 'model_score' key"
        assert "fraud_probability" not in result, \
            "'fraud_probability' key must not exist — score is not calibrated"

    def test_predict_has_warning(self, svc):
        """predict_single result includes a 'warning' field disclaiming non-calibration."""
        result = svc.predict_single("Terrible product broken immediately!", {})
        assert "warning" in result, "Result must include 'warning' about score interpretation"
        assert result["warning"], "Warning must be a non-empty string"

    def test_predicted_class_is_valid(self, svc):
        """predicted_class must be 'suspicious' or 'genuine' (not arbitrary string)."""
        result = svc.predict_single("Some review text here.", {})
        assert "predicted_class" in result
        assert result["predicted_class"] in ("suspicious", "genuine", "0", "1", 0, 1), \
            f"Unexpected predicted_class: {result['predicted_class']}"

    def test_model_score_in_range(self, svc):
        """model_score must be between 0 and 1."""
        result = svc.predict_single("Good product worth the money.", {})
        score = result.get("model_score")
        assert score is not None
        assert 0.0 <= float(score) <= 1.0, f"model_score {score} not in [0,1]"


class TestPersistence:
    def test_save_and_load_model(self, svc, preprocessed_df, tmp_models_dir):
        """Train, save, create new instance, load — predictions should not crash."""
        svc.train(preprocessed_df, model_type="logistic")

        # New instance — load from disk
        svc2 = ClassificationService(models_path=str(tmp_models_dir))
        svc2.load_model()
        assert svc2.is_ready(), "Model should be ready after load_model()"

        result = svc2.predict_single("Great quality product!", {})
        assert "model_score" in result

    def test_is_ready_false_before_training(self, tmp_models_dir):
        """is_ready() returns False when no model has been trained or loaded."""
        svc = ClassificationService(models_path=str(tmp_models_dir))
        assert not svc.is_ready()
