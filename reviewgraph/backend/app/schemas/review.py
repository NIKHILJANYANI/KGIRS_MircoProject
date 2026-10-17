"""
Pydantic v2 schemas for Review-related API responses.

Hierarchy
---------
ReviewBase          – minimal read-only view
ReviewCreate        – write payload
ReviewResponse      – standard list item
ReviewDetailResponse – full detail with signals, related reviews, graph paths

Supporting schemas
------------------
EvidenceSignalResponse
GraphEntity
GraphRelationship
GraphPath
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Graph primitives
# --------------------------------------------------------------------------- #

class GraphEntity(BaseModel):
    """A node in the knowledge graph."""

    model_config = ConfigDict(from_attributes=True)

    node_id: str = Field(..., description="Unique identifier of the graph node")
    label: str = Field(..., description="Neo4j node label, e.g. 'Review', 'Product'")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value properties attached to the node",
    )


class GraphRelationship(BaseModel):
    """An edge in the knowledge graph."""

    model_config = ConfigDict(from_attributes=True)

    rel_id: str = Field(..., description="Unique identifier of the relationship")
    type: str = Field(..., description="Relationship type, e.g. 'REVIEWS', 'SIMILAR_TO'")
    source_id: str = Field(..., description="node_id of the source node")
    target_id: str = Field(..., description="node_id of the target node")
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value properties on the relationship",
    )


class GraphPath(BaseModel):
    """A traversal path through the knowledge graph with a human-readable explanation."""

    model_config = ConfigDict(from_attributes=True)

    path_id: str = Field(..., description="Unique identifier for this path")
    entities: List[GraphEntity] = Field(
        default_factory=list,
        description="Ordered list of nodes in the path",
    )
    relationships: List[GraphRelationship] = Field(
        default_factory=list,
        description="Ordered list of edges in the path (length == len(entities) - 1)",
    )
    explanation: str = Field(
        default="",
        description="Natural-language explanation of why this path is relevant",
    )


# --------------------------------------------------------------------------- #
# Evidence signal
# --------------------------------------------------------------------------- #

class EvidenceSignalResponse(BaseModel):
    """A computed signal that contributes to the suspiciousness assessment."""

    model_config = ConfigDict(from_attributes=True)

    signal_id: str = Field(..., description="Unique identifier for the signal instance")
    signal_type: str = Field(
        ...,
        description="Category of the signal, e.g. 'RATING_BURST', 'SEMANTIC_SIMILARITY'",
    )
    signal_value: Any = Field(
        ...,
        description="Computed value – may be a scalar, list, or dict depending on signal_type",
    )
    description: str = Field(
        default="",
        description="Human-readable description of what this signal represents",
    )
    computation_method: str = Field(
        default="",
        description="Short description of the algorithm used to compute the signal",
    )


# --------------------------------------------------------------------------- #
# Review schemas
# --------------------------------------------------------------------------- #

class ReviewBase(BaseModel):
    """Minimal, read-safe view of a review – safe to embed in list responses."""

    model_config = ConfigDict(from_attributes=True)

    review_id: str
    review_text: Optional[str] = None
    rating: Optional[float] = None
    timestamp: Optional[str] = None
    dataset_label: Optional[str] = Field(
        default=None,
        description="Ground-truth label from the original dataset (CG / OR)",
    )
    predicted_class: Optional[str] = Field(
        default=None,
        description="Classifier prediction: 'CG' (computer-generated) or 'OR' (original)",
    )
    model_score: Optional[float] = Field(
        default=None,
        description="Raw probability / confidence score from the classifier",
    )
    product_id: Optional[str] = None
    reviewer_id: Optional[str] = None


class ReviewCreate(BaseModel):
    """Payload for creating or ingesting a new review."""

    model_config = ConfigDict(from_attributes=True)

    review_id: str
    review_text: Optional[str] = None
    rating: Optional[float] = None
    timestamp: Optional[str] = None
    dataset_label: Optional[str] = None
    product_id: Optional[str] = None
    reviewer_id: Optional[str] = None


class ReviewResponse(ReviewBase):
    """Standard review item returned in list/search responses."""

    # Inherits all ReviewBase fields; extend here if needed.
    pass


class ReviewDetailResponse(ReviewBase):
    """
    Full detail response for a single review, including evidence signals,
    semantically related reviews, and knowledge-graph paths.
    """

    signals: List[EvidenceSignalResponse] = Field(
        default_factory=list,
        description="Evidence signals associated with this review",
    )
    related_reviews: List[ReviewBase] = Field(
        default_factory=list,
        description="Semantically or graph-similar reviews",
    )
    graph_paths: List[GraphPath] = Field(
        default_factory=list,
        description="Knowledge-graph traversal paths explaining the review's context",
    )
