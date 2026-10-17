"""
Pydantic v2 schemas for search, stats, health, graph visualisation, and
evaluation API endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

class SearchRequest(BaseModel):
    """Request body for the unified search endpoint."""

    model_config = ConfigDict(from_attributes=True)

    query: str = Field(..., min_length=1, description="Free-text search query")
    mode: Literal["keyword", "semantic", "graph", "hybrid"] = Field(
        default="hybrid",
        description=(
            "Search strategy: "
            "'keyword' – BM25-style token matching; "
            "'semantic' – dense vector similarity via FAISS; "
            "'graph'   – knowledge-graph traversal; "
            "'hybrid'  – weighted combination of all three"
        ),
    )
    page: int = Field(default=1, ge=1, description="1-based page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Results per page")
    product_id: Optional[str] = Field(
        default=None,
        description="Restrict results to a specific product",
    )
    min_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum combined_score threshold (inclusive)",
    )


class SearchResult(BaseModel):
    """A single item returned in a search response."""

    model_config = ConfigDict(from_attributes=True)

    item_id: str = Field(..., description="Identifier of the matched item (review / product)")
    item_type: str = Field(..., description="Type of the matched item, e.g. 'review' or 'product'")
    title: Optional[str] = Field(default=None, description="Display title or short label")
    snippet: Optional[str] = Field(
        default=None,
        description="Highlighted excerpt or preview of the matched text",
    )
    keyword_score: Optional[float] = Field(
        default=None, ge=0.0, description="Score from keyword matching"
    )
    semantic_score: Optional[float] = Field(
        default=None, ge=0.0, description="Score from dense vector similarity"
    )
    graph_score: Optional[float] = Field(
        default=None, ge=0.0, description="Score from graph-traversal ranking"
    )
    combined_score: float = Field(
        ..., ge=0.0, description="Final weighted combined score used for ranking"
    )
    model_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Classifier confidence that the review is fake (CG)",
    )


class SearchResponse(BaseModel):
    """Paginated search result envelope."""

    model_config = ConfigDict(from_attributes=True)

    query: str
    mode: str
    total: int = Field(..., ge=0, description="Total number of matching items (before pagination)")
    page: int
    page_size: int
    results: List[SearchResult] = Field(default_factory=list)
    latency_ms: float = Field(..., description="Server-side processing time in milliseconds")


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

class StatsResponse(BaseModel):
    """Aggregate dataset and system statistics returned by the /stats endpoint."""

    model_config = ConfigDict(from_attributes=True)

    total_products: int = Field(default=0)
    total_reviews: int = Field(default=0)
    total_reviewers: int = Field(default=0)
    labeled_reviews: int = Field(
        default=0, description="Reviews that have a ground-truth dataset label"
    )
    suspicious_reviews: int = Field(
        default=0,
        description="Reviews predicted or labelled as fake / computer-generated",
    )
    genuine_reviews: int = Field(
        default=0,
        description="Reviews predicted or labelled as genuine / original",
    )
    avg_rating: Optional[float] = Field(
        default=None, description="Mean star rating across all reviews"
    )
    label_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of reviews per label key (e.g. {'CG': 1234, 'OR': 5678})",
    )
    model_status: str = Field(
        default="unknown",
        description="Classifier model status: 'loaded' | 'not_loaded' | 'error'",
    )
    graph_status: str = Field(
        default="unknown",
        description="Neo4j connection status: 'connected' | 'disconnected'",
    )
    embedding_status: str = Field(
        default="unknown",
        description="FAISS / embedding model status: 'loaded' | 'not_loaded' | 'error'",
    )
    dataset_status: str = Field(
        default="unknown",
        description="Dataset availability: 'loaded' | 'not_loaded' | 'partial'",
    )


# --------------------------------------------------------------------------- #
# Health check
# --------------------------------------------------------------------------- #

class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    model_config = ConfigDict(from_attributes=True)

    status: str = Field(..., description="Overall application health: 'ok' | 'degraded' | 'error'")
    neo4j: str = Field(..., description="Neo4j connectivity: 'connected' | 'disconnected'")
    model: str = Field(..., description="Classifier model: 'loaded' | 'not_loaded'")
    embeddings: str = Field(..., description="Embedding model / FAISS: 'loaded' | 'not_loaded'")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp of the health check")


# --------------------------------------------------------------------------- #
# Graph visualisation
# --------------------------------------------------------------------------- #

class GraphVisualizationResponse(BaseModel):
    """
    Lightweight graph structure for front-end visualisation libraries
    (e.g. Cytoscape.js, D3-force, React Flow).
    """

    model_config = ConfigDict(from_attributes=True)

    nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of node descriptors. Each dict should contain at minimum "
            "{'id': str, 'label': str, 'properties': dict}"
        ),
    )
    edges: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of edge descriptors. Each dict should contain at minimum "
            "{'id': str, 'source': str, 'target': str, 'type': str}"
        ),
    )
    node_count: int = Field(..., ge=0)
    edge_count: int = Field(..., ge=0)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

class EvaluationResponse(BaseModel):
    """
    Aggregated evaluation metrics returned by the /evaluate endpoint.

    Metric dictionaries are intentionally typed as ``Dict[str, Any]`` to
    accommodate varying metric sets across classification, retrieval,
    explanation quality, and latency dimensions.
    """

    model_config = ConfigDict(from_attributes=True)

    classification_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Classifier performance metrics, e.g. accuracy, precision, recall, "
            "F1-score, ROC-AUC, confusion matrix"
        ),
    )
    retrieval_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Information-retrieval metrics, e.g. Precision@k, Recall@k, "
            "NDCG@k, MRR, MAP"
        ),
    )
    explanation_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Explanation quality metrics, e.g. faithfulness, completeness, "
            "path coverage"
        ),
    )
    latency_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "End-to-end and per-component latency stats (ms), "
            "e.g. p50, p95, p99 for search and investigation endpoints"
        ),
    )
    generated_at: str = Field(
        ...,
        description="ISO-8601 UTC timestamp when the evaluation was run",
    )
