"""
test_explanation_service.py
----------------------------
Tests for ExplanationService: no fabricated evidence, required structure.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def mock_investigation_data():
    """Minimal investigation data matching what graph_service.investigate_product() returns."""
    return {
        "product_id": "P001",
        "title": "Test Product",
        "suspiciousness_score": 0.45,
        "suspiciousness_level": "medium",
        "reviews": [
            {"review_id": "R001", "review_text": "Great product!", "rating": 5.0,
             "model_score": 0.72, "predicted_class": "suspicious"},
            {"review_id": "R002", "review_text": "Decent quality overall.", "rating": 4.0,
             "model_score": 0.18, "predicted_class": "genuine"},
            {"review_id": "R003", "review_text": "Worst purchase ever avoid!", "rating": 1.0,
             "model_score": 0.85, "predicted_class": "suspicious"},
        ],
        "signals": [
            {"signal_id": "S001", "signal_type": "HIGH_MODEL_SCORE",
             "signal_value": 0.72, "description": "Review R001 has high model score",
             "computation_method": "logistic_regression"},
        ],
        "similar_reviews": [
            {"review_id_a": "R001", "review_id_b": "R004", "score": 0.89, "method": "cosine_tfidf"}
        ],
        "connected_products": [{"product_id": "P002", "title": "Related Product"}],
        "connected_reviewers": [{"reviewer_id": "REV001", "review_count": 5}],
        "graph_paths": [],
        "source_records": ["R001", "R002", "R003"],
    }


@pytest.fixture
def explanation_service(mock_graph_service):
    from app.services.explanation_service import ExplanationService
    return ExplanationService(graph_service=mock_graph_service)


class TestExplanationService:
    @pytest.mark.asyncio
    async def test_explanation_only_cites_existing_reviews(
        self, explanation_service, mock_investigation_data
    ):
        """Every review_id in source_records must be from the actual input data."""
        explanation = await explanation_service.generate_product_explanation(
            mock_investigation_data
        )
        source_records = explanation.get("source_records", [])
        actual_review_ids = {r["review_id"] for r in mock_investigation_data["reviews"]}

        for rid in source_records:
            assert rid in actual_review_ids, \
                f"Explanation cites review '{rid}' not in input data — potential fabrication!"

    @pytest.mark.asyncio
    async def test_explanation_includes_limitations(
        self, explanation_service, mock_investigation_data
    ):
        """Explanation must include a non-empty 'limitations' list."""
        explanation = await explanation_service.generate_product_explanation(
            mock_investigation_data
        )
        limitations = explanation.get("limitations", [])
        assert isinstance(limitations, list), "'limitations' must be a list"
        assert len(limitations) >= 1, "Explanation must include at least one limitation"

    @pytest.mark.asyncio
    async def test_explanation_distinguishes_model_predictions(
        self, explanation_service, mock_investigation_data
    ):
        """model_predictions must be a separate section from observed_facts."""
        explanation = await explanation_service.generate_product_explanation(
            mock_investigation_data
        )
        assert "model_predictions" in explanation, \
            "'model_predictions' section must exist"
        assert "observed_facts" in explanation, \
            "'observed_facts' section must exist"

    @pytest.mark.asyncio
    async def test_no_unsupported_claims_in_summary(
        self, explanation_service, mock_investigation_data
    ):
        """Explanation summary must not contain certainty claims."""
        explanation = await explanation_service.generate_product_explanation(
            mock_investigation_data
        )
        summary = (explanation.get("explanation_summary") or "").lower()
        forbidden = ["proven", "definitely", "certainly", "confirmed fraud", "is fake", "guaranteed"]
        for word in forbidden:
            assert word not in summary, \
                f"Explanation summary contains unsupported claim: '{word}'"

    @pytest.mark.asyncio
    async def test_explanation_has_required_keys(
        self, explanation_service, mock_investigation_data
    ):
        """Explanation output must have all required top-level keys."""
        explanation = await explanation_service.generate_product_explanation(
            mock_investigation_data
        )
        required = {
            "investigated_entity", "observed_facts", "computed_signals",
            "model_predictions", "source_records", "explanation_summary", "limitations"
        }
        missing = required - set(explanation.keys())
        assert not missing, f"Explanation missing required keys: {missing}"
