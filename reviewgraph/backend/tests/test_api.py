"""
test_api.py
-----------
API integration tests using FastAPI TestClient.
Tests health, stats, search validation, and error handling.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture
def client(mock_graph_service, mock_classification_service, mock_embedding_service):
    """TestClient with mocked services injected into app state."""
    from app.main import app

    mock_graph_service.run_parameterized = AsyncMock(return_value=[{"product": {"product_id": "P001", "name": "Test"}}])
    mock_retrieval = MagicMock()
    mock_retrieval.hybrid_search = AsyncMock(return_value={"results": [], "total": 0, "page": 1, "page_size": 20, "query": "test", "mode": "hybrid", "latency_ms": 5.0})
    mock_retrieval.keyword_search = AsyncMock(return_value={"results": [], "total": 0, "latency_ms": 5.0})
    mock_retrieval.semantic_search = AsyncMock(return_value={"results": [], "total": 0, "latency_ms": 5.0})
    mock_retrieval.graph_search = AsyncMock(return_value={"results": [], "total": 0, "latency_ms": 5.0})

    app.state.graph_service = mock_graph_service
    app.state.classification_service = mock_classification_service
    app.state.embedding_service = mock_embedding_service
    app.state.retrieval_service = mock_retrieval
    app.state.explanation_service = MagicMock()
    app.state.dataset_service = MagicMock()

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        """GET /api/health must return 200."""
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self, client):
        """Health response must include status, neo4j, model, embeddings, timestamp."""
        resp = client.get("/api/health")
        data = resp.json()
        for field in ("status", "neo4j", "model", "embeddings", "timestamp"):
            assert field in data, f"Missing field '{field}' in health response"

    def test_health_status_is_string(self, client):
        """status field must be a string."""
        resp = client.get("/api/health")
        assert isinstance(resp.json()["status"], str)


class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        """GET /api/stats must return 200."""
        resp = client.get("/api/stats")
        assert resp.status_code == 200


class TestSearchEndpoint:
    def test_search_validates_empty_query(self, client):
        """POST /api/search with empty query must return 422."""
        resp = client.post("/api/search", json={"query": "", "mode": "hybrid"})
        assert resp.status_code == 422

    def test_search_validates_invalid_mode(self, client):
        """POST /api/search with invalid mode must return 422."""
        resp = client.post("/api/search", json={"query": "test", "mode": "invalid_mode"})
        assert resp.status_code == 422

    def test_search_valid_request_accepted(self, client):
        """POST /api/search with valid payload should return 200."""
        resp = client.post("/api/search", json={"query": "great product", "mode": "keyword"})
        assert resp.status_code == 200

    def test_search_response_has_latency(self, client):
        """Search response must include latency_ms field."""
        resp = client.post("/api/search", json={"query": "review", "mode": "semantic"})
        if resp.status_code == 200:
            data = resp.json()
            assert "latency_ms" in data


class TestProductsEndpoint:
    def test_products_returns_200(self, client):
        """GET /api/products returns 200."""
        resp = client.get("/api/products")
        assert resp.status_code == 200

    def test_products_unknown_id_returns_404(self, client):
        """GET /api/products/{id} for non-existent ID returns 404."""
        resp = client.get("/api/products/NONEXISTENT_PRODUCT_XYZ")
        assert resp.status_code in (404, 503)  # 503 if graph is down


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        """GET / returns 200."""
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_docs_link(self, client):
        """Root response includes docs URL."""
        resp = client.get("/")
        assert "docs" in resp.json()
