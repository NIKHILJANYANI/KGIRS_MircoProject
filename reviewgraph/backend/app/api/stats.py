"""
stats.py
--------
GET /api/stats — Aggregate statistics from all services.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["stats"])


@router.get("/stats", summary="Aggregate system statistics")
async def get_stats(request: Request) -> Dict[str, Any]:
    """
    Return aggregated statistics from the graph, classification model,
    and embedding index.

    All numeric values default to **0** when the backing service is
    unavailable so the endpoint never returns fabricated numbers.
    """
    # ------------------------------------------------------------------ #
    # Graph statistics
    # ------------------------------------------------------------------ #
    graph_stats: Dict[str, Any] = {}
    graph_status = "unavailable"
    graph_service = getattr(request.app.state, "graph_service", None)
    if graph_service is not None:
        try:
            graph_stats = await graph_service.get_stats()
            graph_status = "ok"
        except Exception as exc:
            logger.warning("Stats – graph_service.get_stats() failed: %s", exc)

    nodes: Dict[str, int] = graph_stats.get("nodes", {})
    relationships: Dict[str, Any] = graph_stats.get("relationships", {})

    # ------------------------------------------------------------------ #
    # Classification model metadata
    # ------------------------------------------------------------------ #
    model_meta: Dict[str, Any] = {}
    model_status = "unavailable"
    classification_service = getattr(request.app.state, "classification_service", None)
    if classification_service is not None:
        try:
            model_meta = classification_service.get_metadata()
            model_status = "ok" if model_meta else "not_initialized"
        except Exception as exc:
            logger.warning("Stats – classification_service.get_metadata() failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Embedding / FAISS statistics
    # ------------------------------------------------------------------ #
    embedding_stats: Dict[str, Any] = {}
    embedding_status = "unavailable"
    embedding_service = getattr(request.app.state, "embedding_service", None)
    if embedding_service is not None:
        try:
            embedding_stats = embedding_service.get_stats()
            embedding_status = "ok" if embedding_stats else "not_initialized"
        except Exception as exc:
            logger.warning("Stats – embedding_service.get_stats() failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Dataset & Review distributions
    # ------------------------------------------------------------------ #
    from app.core.config import get_settings
    import os
    settings = get_settings()

    total_products = nodes.get("Product", 0)
    total_reviews = nodes.get("Review", 0)
    total_reviewers = nodes.get("Reviewer", 0)
    labeled_reviews = 0
    suspicious_reviews = 0
    genuine_reviews = 0
    avg_rating = None
    label_distribution = {"suspicious": 0, "genuine": 0, "unlabeled": 0}
    weak_label_distribution: Dict[str, int] = {}
    model_score_distribution: Dict[str, int] = {}

    # Load actual dataset stats from processed CSV if available
    reviews_file = os.path.join(settings.DATA_PROCESSED_PATH, "reviews_processed.csv")

    if os.path.exists(reviews_file):
        try:
            import pandas as pd
            df = pd.read_csv(reviews_file)
            if total_reviews == 0:
                total_reviews = len(df)
            if total_products == 0 and "product_id" in df.columns:
                total_products = int(df["product_id"].nunique())
            if total_reviewers == 0 and "reviewer_id" in df.columns:
                total_reviewers = int(df["reviewer_id"].nunique())

            if "rating" in df.columns:
                avg_rating = round(float(df["rating"].mean()), 2)

            label_col = "label" if "label" in df.columns else ("is_fake" if "is_fake" in df.columns else None)
            has_weak_labels = "label_source" in df.columns
            if label_col and not has_weak_labels:
                labeled_df = df[df[label_col].notna()]
                labeled_reviews = len(labeled_df)
                suspicious_reviews = int(
                    (labeled_df[label_col].astype(str).str.lower().isin(["1", "1.0", "true", "cg", "fake"])).sum()
                )
                genuine_reviews = labeled_reviews - suspicious_reviews
                unlabeled = len(df) - labeled_reviews
                label_distribution = {
                    "suspicious": suspicious_reviews,
                    "genuine": genuine_reviews,
                    "unlabeled": unlabeled,
                }
            else:
                label_distribution = {
                    "suspicious": 0,
                    "genuine": 0,
                    "unlabeled": len(df),
                }
            if has_weak_labels and label_col:
                weak_counts = df[label_col].fillna("unlabeled").astype(str).str.lower().value_counts()
                weak_label_distribution = {
                    "suspicious": int(weak_counts.get("suspicious", 0)),
                    "genuine": int(weak_counts.get("genuine", 0)),
                    "unlabeled": int(weak_counts.get("unlabeled", 0)),
                }
            if "model_score" in df.columns:
                scores = df["model_score"].dropna()
                if not scores.empty:
                    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    labels = ["0.0 - 0.2", "0.2 - 0.4", "0.4 - 0.6", "0.6 - 0.8", "0.8 - 1.0"]
                    bucketed = pd.cut(scores.clip(0, 1), bins=bins, labels=labels, include_lowest=True)
                    model_score_distribution = {
                        label: int((bucketed == label).sum()) for label in labels
                    }
        except Exception as exc:
            logger.warning("Stats – failed to read dataset CSV: %s", exc)

    return {
        "total_products": total_products,
        "total_reviews": total_reviews,
        "total_reviewers": total_reviewers,
        "labeled_reviews": labeled_reviews,
        "suspicious_reviews": suspicious_reviews,
        "genuine_reviews": genuine_reviews,
        "avg_rating": avg_rating,
        "label_distribution": label_distribution,
        "weak_label_distribution": weak_label_distribution,
        "label_status": (
            "weak_labels_not_ground_truth"
            if weak_label_distribution
            else "ground_truth_unavailable"
        ),
        "model_score_distribution": model_score_distribution,
        "model_version": model_meta.get("model_version", "1.0-pipeline"),
        "dataset_size": total_reviews,
        "model_status": "loaded" if model_status == "ok" else model_status,
        "graph_status": "connected" if graph_status == "ok" else graph_status,
        "embedding_status": "loaded" if embedding_status == "ok" else embedding_status,
        "dataset_status": "loaded",
        "graph": {
            "status": graph_status,
            "node_counts": {
                "products": nodes.get("Product", 0),
                "reviews": nodes.get("Review", 0),
                "reviewers": nodes.get("Reviewer", 0),
                "evidence_signals": nodes.get("EvidenceSignal", 0),
                "total": sum(nodes.values()),
            },
            "relationship_counts": {
                "total": relationships.get("total", 0),
                "WROTE": relationships.get("WROTE", 0),
                "REVIEWS": relationships.get("REVIEWS", 0),
                "HAS_SIGNAL": relationships.get("HAS_SIGNAL", 0),
                "SIMILAR_TO": relationships.get("SIMILAR_TO", 0),
            },
        },
        "model": {
            "status": model_status,
            "metadata": model_meta,
        },
        "embeddings": {
            "status": embedding_status,
            "index_size": embedding_stats.get("index_size", 0),
            "dimension": embedding_stats.get("dimension", 0),
            "model_name": embedding_stats.get("model_name", ""),
        },
    }
