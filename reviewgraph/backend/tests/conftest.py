"""
conftest.py
-----------
Shared pytest fixtures for ReviewGraph backend tests.

All fixtures in this file use SYNTHETIC data explicitly labeled as such.
Synthetic fixtures are NEVER mixed with real dataset metrics.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


# ─────────────────────────────────────────────────────────────────────────── #
# Synthetic test dataset (20 rows)
# ─────────────────────────────────────────────────────────────────────────── #

GENUINE_TEXTS = [
    "I bought this product three weeks ago and I am very happy with the build quality. The materials feel premium.",
    "Decent product for the price. Shipping was a bit slow but the item arrived in good condition.",
    "Works as described. I have been using it daily for a month without any issues.",
    "The color is slightly different from the pictures online but overall I like it. Good purchase.",
    "Very easy to set up. The instructions were clear and it took less than ten minutes.",
    "Not the best product I have owned but it does the job. Would consider buying again.",
    "Great value for money. It feels sturdy and looks nice. Satisfied with my purchase.",
    "Bought this as a gift and the recipient loved it. Packaging was neat and professional.",
    "Performs exactly as advertised. No complaints after two months of regular use.",
    "Solid product. Customer service was helpful when I had a question about the warranty.",
    "Had this for six months now. Still working perfectly. Highly recommend for the price point.",
    "The design is sleek and modern. Fits well in my kitchen without taking too much counter space.",
    "Quality could be slightly better but for the price I cannot complain. Does what it says.",
    "Ordered two of these for the house. Both work great. Fast delivery too.",
]

SUSPICIOUS_TEXTS = [
    "BEST PRODUCT EVER! Five stars! Buy now! You will love it! Amazing!",
    "Best product ever! Amazing quality! Buy this now! You will not regret it!",
    "Absolutely terrible. Broke after one day. Worst purchase of my life. Avoid at all costs!",
    "Five stars. Best product. Buy this. Amazing. Love it. Recommend.",
    "Do not buy this! It is a scam! Broke immediately! Worst product ever!",
    "Best product ever! Five stars! Amazing quality! Buy now!",
]


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    Synthetic 20-row DataFrame for testing.
    14 genuine (label=0), 6 suspicious (label=1).
    All rows are synthetic test data — not from any real dataset.
    """
    rows = []

    # Genuine reviews
    reviewer_ids = [f"REV{i:03d}" for i in range(1, 15)]
    product_ids = [f"PROD{i:02d}" for i in range(1, 6)]

    for i, (text, rid) in enumerate(zip(GENUINE_TEXTS, reviewer_ids)):
        rows.append({
            "review_id": f"SYNTH_GENUINE_{i:03d}",
            "reviewer_id": rid,
            "product_id": product_ids[i % len(product_ids)],
            "review_text": text,
            "rating": float(np.random.choice([3, 4, 5])),
            "timestamp": f"2024-0{(i % 9) + 1}-15",
            "label": 0,
            "is_synthetic": True,
        })

    # Suspicious reviews
    sus_reviewers = [f"REV{i:03d}" for i in range(95, 101)]
    for i, (text, rid) in enumerate(zip(SUSPICIOUS_TEXTS, sus_reviewers)):
        rows.append({
            "review_id": f"SYNTH_SUS_{i:03d}",
            "reviewer_id": rid,
            "product_id": product_ids[i % len(product_ids)],
            "review_text": text,
            "rating": float(np.random.choice([1, 5])),
            "timestamp": f"2024-03-{(i + 1):02d}",
            "label": 1,
            "is_synthetic": True,
        })

    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def sample_reviews_list(sample_df) -> list:
    """List-of-dicts version of sample_df."""
    return sample_df.to_dict(orient="records")


@pytest.fixture
def tmp_models_dir(tmp_path: Path) -> Path:
    """Temporary directory for model artifacts during tests."""
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary directory for data files during tests."""
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def mock_neo4j_session() -> AsyncMock:
    """AsyncMock Neo4j session returning canned responses."""
    session = AsyncMock()
    result = AsyncMock()
    result.data.return_value = [
        {
            "p": {"product_id": "P001", "title": "Test Product", "category": "Electronics"},
            "review_count": 5,
            "avg_rating": 4.2,
        }
    ]
    session.run.return_value = result
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.fixture
def mock_graph_service() -> MagicMock:
    """Mock GraphService with common async methods."""
    svc = MagicMock()
    svc.is_connected = AsyncMock(return_value=True)
    svc.close = AsyncMock(return_value=None)
    svc.create_constraints = AsyncMock(return_value=None)
    svc.get_stats = AsyncMock(return_value={
        "total_products": 10,
        "total_reviews": 100,
        "total_reviewers": 30,
        "total_signals": 42,
    })
    svc.get_product = AsyncMock(side_effect=lambda pid: None if "NONEXISTENT" in str(pid) else {
        "product_id": pid, "title": "Test Product", "review_count": 5
    })
    svc.investigate_product = AsyncMock(return_value={
        "product_id": "P001",
        "suspiciousness_score": 0.35,
        "suspiciousness_level": "medium",
        "reviews": [],
        "signals": [],
        "similar_reviews": [],
        "connected_products": [],
        "connected_reviewers": [],
        "graph_paths": [],
        "source_records": [],
    })
    svc.get_product_graph = AsyncMock(return_value={"nodes": [], "edges": []})
    svc.search_reviews_graph = AsyncMock(return_value=[])
    svc.get_review = AsyncMock(return_value=None)
    svc.get_related_reviews = AsyncMock(return_value=[])
    return svc


@pytest.fixture
def mock_classification_service() -> MagicMock:
    svc = MagicMock()
    svc.is_ready.return_value = True
    svc.predict_single.return_value = {
        "predicted_class": "genuine",
        "model_score": 0.23,
        "model_version": "test-v1",
        "warning": "score is a model score, not a calibrated probability of fraud",
    }
    svc.get_metadata.return_value = {
        "model_version": "test-v1",
        "training_date": "2024-01-01",
        "model_type": "logistic_regression",
    }
    return svc


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    svc = MagicMock()
    svc.is_ready.return_value = True
    svc.search = AsyncMock(return_value=[])
    svc.get_stats.return_value = {"total_indexed": 100, "model_name": "all-MiniLM-L6-v2"}
    return svc
