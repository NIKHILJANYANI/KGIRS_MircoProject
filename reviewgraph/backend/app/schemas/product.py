"""
Pydantic v2 schemas for Product-related API responses.

Hierarchy
---------
ProductBase                   – minimal product info
ProductResponse               – adds aggregate stats
ProductDetailResponse         – adds reviews & rating distribution
ProductInvestigationResponse  – full investigation result with suspiciousness data

Supporting schemas
------------------
ReviewerBase – minimal reviewer view embedded in investigation responses
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.review import (
    EvidenceSignalResponse,
    GraphPath,
    ReviewBase,
)


# --------------------------------------------------------------------------- #
# Reviewer (lightweight – used only as embedded reference)
# --------------------------------------------------------------------------- #

class ReviewerBase(BaseModel):
    """Minimal reviewer data surfaced in investigation responses."""

    model_config = ConfigDict(from_attributes=True)

    reviewer_id: str
    review_count: int = Field(
        default=0,
        description="Total number of reviews posted by this reviewer",
    )
    distinct_products_reviewed: int = Field(
        default=0,
        description="Number of distinct products this reviewer has reviewed",
    )


# --------------------------------------------------------------------------- #
# Product schemas
# --------------------------------------------------------------------------- #

class ProductBase(BaseModel):
    """Minimal, read-safe product view."""

    model_config = ConfigDict(from_attributes=True)

    product_id: str
    title: Optional[str] = None
    category: Optional[str] = None


class ProductResponse(ProductBase):
    """Product item returned in list / search responses – adds aggregate stats."""

    average_rating: Optional[float] = Field(
        default=None,
        description="Mean star rating across all reviews for this product",
    )
    review_count: int = Field(
        default=0,
        description="Total number of reviews linked to this product",
    )


class ProductDetailResponse(ProductResponse):
    """Full product detail with all linked reviews and rating breakdown."""

    reviews: List[ReviewBase] = Field(
        default_factory=list,
        description="All reviews linked to this product",
    )
    rating_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Mapping of star rating (as string key, e.g. '1'..'5') "
            "to the count of reviews at that rating"
        ),
    )


class ProductInvestigationResponse(ProductResponse):
    """
    Comprehensive investigation result for a product.

    Combines suspiciousness scoring, evidence signals, suspicious review
    listings, related graph entities, traversal paths, and a human-readable
    explanation together with acknowledged limitations of the analysis.
    """

    # ------------------------------------------------------------------ #
    # Suspiciousness verdict
    # ------------------------------------------------------------------ #
    suspiciousness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalised suspiciousness score in [0, 1]",
    )
    suspiciousness_level: str = Field(
        ...,
        description="Human-readable verdict: 'low' | 'medium' | 'high' | 'critical'",
    )

    # ------------------------------------------------------------------ #
    # Evidence
    # ------------------------------------------------------------------ #
    signals: List[EvidenceSignalResponse] = Field(
        default_factory=list,
        description="Evidence signals that contributed to the suspiciousness score",
    )
    suspicious_reviews: List[ProductResponse] = Field(
        default_factory=list,
        description=(
            "Reviews flagged as potentially fake / computer-generated for this product. "
            "Typed as ProductResponse to carry per-review aggregate fields if needed."
        ),
    )

    # ------------------------------------------------------------------ #
    # Graph context
    # ------------------------------------------------------------------ #
    connected_products: List[ProductBase] = Field(
        default_factory=list,
        description="Other products connected to this one via shared reviewers or signals",
    )
    connected_reviewers: List[ReviewerBase] = Field(
        default_factory=list,
        description="Reviewers connected to this product in the knowledge graph",
    )
    graph_paths: List[GraphPath] = Field(
        default_factory=list,
        description="Knowledge-graph traversal paths explaining the investigation",
    )

    # ------------------------------------------------------------------ #
    # Explanation
    # ------------------------------------------------------------------ #
    explanation_summary: str = Field(
        default="",
        description="Natural-language summary of the investigation findings",
    )
    limitations: List[str] = Field(
        default_factory=list,
        description=(
            "Known limitations or caveats that apply to this investigation result, "
            "e.g. 'Small review count reduces confidence'"
        ),
    )
