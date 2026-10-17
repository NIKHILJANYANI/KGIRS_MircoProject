"""
reviews.py
----------
Review-related API endpoints:
  GET /api/reviews/{review_id}         — review detail with model predictions
  GET /api/reviews/{review_id}/related — related reviews via SIMILAR_TO
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reviews"])


def _require_graph(request: Request):
    svc = getattr(request.app.state, "graph_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="Graph service is unavailable. Please check Neo4j connectivity.",
        )
    return svc


# --------------------------------------------------------------------------- #
# GET /api/reviews/{review_id}
# --------------------------------------------------------------------------- #

@router.get("/reviews/{review_id}", summary="Review detail with model predictions")
async def get_review(
    review_id: str,
    request: Request,
) -> Dict[str, Any]:
    """
    Return a single review with:
    * All stored graph properties (rating, text, date, etc.)
    * Pre-stored model score / predicted class
    * Evidence signals attached to the review
    * A fresh single-review prediction if the classifier is ready
    """
    graph_service = _require_graph(request)

    try:
        review = await graph_service.get_review(review_id)
    except Exception as exc:
        logger.error("get_review(%s) graph error: %s", review_id, exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Graph query failed.") from exc

    if review is None:
        raise HTTPException(status_code=404, detail=f"Review '{review_id}' not found.")

    # ------------------------------------------------------------------ #
    # Live model prediction (if classifier is ready)
    # ------------------------------------------------------------------ #
    live_prediction: Optional[Dict[str, Any]] = None
    classification_service = getattr(request.app.state, "classification_service", None)
    if classification_service is not None:
        try:
            if classification_service.is_ready():
                review_text = review.get("review_text") or review.get("text") or ""
                if review_text.strip():
                    beh_dict = {
                        "word_count": review.get("word_count", 0),
                        "char_count": review.get("char_count", 0),
                        "review_length": review.get("char_count", 0),
                    }
                    pred = classification_service.predict_single(review_text, behavioral_dict=beh_dict)
                    raw_cls = pred.get("predicted_class")
                    pred_label = "Suspicious" if raw_cls in (1, "1", "fake", "suspicious", "CG") else "Genuine"
                    pred["predicted_class"] = pred_label
                    pred["score"] = pred.get("model_score", 0.0)
                    live_prediction = pred

                    # Fill into review if missing
                    if review.get("model_score") is None:
                        review["model_score"] = pred.get("model_score")
                    if review.get("predicted_class") is None:
                        review["predicted_class"] = pred_label
        except Exception as exc:
            logger.warning(
                "get_review(%s) live prediction failed: %s", review_id, exc
            )

    return {
        "review": review,
        "live_prediction": live_prediction,
        "signals": review.get("signals", []),
    }


# --------------------------------------------------------------------------- #
# GET /api/reviews/{review_id}/related
# --------------------------------------------------------------------------- #

@router.get("/reviews/{review_id}/related", summary="Related reviews via SIMILAR_TO")
async def get_related_reviews(
    review_id: str,
    request: Request,
    limit: int = Query(10, ge=1, le=50, description="Max number of related reviews"),
) -> Dict[str, Any]:
    """
    Return reviews that share a SIMILAR_TO relationship with the given review,
    ordered by similarity score descending.
    """
    graph_service = _require_graph(request)

    # Verify review exists first
    try:
        review = await graph_service.get_review(review_id)
    except Exception as exc:
        logger.error(
            "get_related_reviews(%s) graph error: %s", review_id, exc, exc_info=True
        )
        raise HTTPException(status_code=503, detail="Graph query failed.") from exc

    if review is None:
        raise HTTPException(status_code=404, detail=f"Review '{review_id}' not found.")

    try:
        related: List[Dict[str, Any]] = await graph_service.get_related_reviews(
            review_id, limit=limit
        )
    except Exception as exc:
        logger.error(
            "get_related_reviews(%s) fetch error: %s", review_id, exc, exc_info=True
        )
        raise HTTPException(status_code=503, detail="Graph query failed.") from exc

    return {
        "review_id": review_id,
        "related": related,
        "count": len(related),
    }
