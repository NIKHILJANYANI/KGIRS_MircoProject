"""
search.py
---------
POST /api/search — Hybrid / keyword / semantic / graph search over reviews.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["search"])


# --------------------------------------------------------------------------- #
# Request / Response schemas (local – no external schema file required)
# --------------------------------------------------------------------------- #

class SearchMode(str, Enum):
    hybrid = "hybrid"
    keyword = "keyword"
    semantic = "semantic"
    graph = "graph"


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
    mode: SearchMode = Field(SearchMode.hybrid, description="Search mode")
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    product_id: Optional[str] = Field(None, description="Optional product filter")
    min_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum relevance score filter"
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty")
        return v.strip()


# --------------------------------------------------------------------------- #
# POST /api/search
# --------------------------------------------------------------------------- #

@router.post("/search", summary="Hybrid/keyword/semantic/graph search")
async def search(
    body: SearchRequest,
    request: Request,
) -> Dict[str, Any]:
    """
    Unified search endpoint supporting four modes:

    * **hybrid**   — combines keyword + semantic scores (default)
    * **keyword**  — CONTAINS-style graph text search
    * **semantic** — FAISS vector similarity search
    * **graph**    — graph-traversal-based retrieval

    Results below *min_score* are filtered out after retrieval.
    Latency (wall-clock ms) is always included in the response.
    """
    start_ms = time.monotonic() * 1000

    retrieval_service = getattr(request.app.state, "retrieval_service", None)
    graph_service = getattr(request.app.state, "graph_service", None)

    results: List[Dict[str, Any]] = []
    search_status = "ok"
    warning: Optional[str] = None

    try:
        if retrieval_service is not None:
            top_k = body.page * body.page_size
            if body.mode == SearchMode.hybrid:
                raw = await retrieval_service.hybrid_search(
                    query=body.query,
                    mode="hybrid",
                    limit=top_k,
                    page=1,
                    product_id=body.product_id,
                )
            elif body.mode == SearchMode.keyword:
                raw = await retrieval_service.keyword_search(
                    query=body.query,
                    limit=top_k,
                    page=1,
                )
            elif body.mode == SearchMode.semantic:
                raw = await retrieval_service.semantic_search(
                    query=body.query,
                    limit=top_k,
                )
            elif body.mode == SearchMode.graph:
                raw = await retrieval_service.graph_search(
                    query=body.query,
                    product_id=body.product_id,
                    limit=top_k,
                )
            else:
                raw = {"results": []}
            # hybrid_search returns a dict; others may too
            results = raw.get("results", raw) if isinstance(raw, dict) else raw
        elif graph_service is not None:
            # Fallback: plain keyword search via graph service
            warning = (
                "retrieval_service is unavailable; falling back to basic keyword search."
            )
            results_raw = await graph_service.search_reviews_graph(
                query=body.query,
                product_id=body.product_id,
                limit=body.page * body.page_size,
            )
            results = [
                {
                    "review_id": r.get("review_id", ""),
                    "text": r.get("text", ""),
                    "score": r.get("model_score") or 0.0,
                    "product_id": r.get("product_id", ""),
                    "product_name": r.get("product_name", ""),
                    "predicted_class": r.get("predicted_class", "unknown"),
                    "source": "graph_keyword",
                }
                for r in results_raw
            ]
        else:
            raise HTTPException(
                status_code=503,
                detail="No search service available. Please check service configuration.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("search error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=503, detail=f"Search failed: {exc}"
        ) from exc

    # ------------------------------------------------------------------ #
    # Apply min_score filter
    # ------------------------------------------------------------------ #
    if body.min_score is not None:
        results = [r for r in results if (r.get("score") or 0.0) >= body.min_score]

    # ------------------------------------------------------------------ #
    # Paginate and normalize items
    # ------------------------------------------------------------------ #
    total = len(results)
    start_idx = (body.page - 1) * body.page_size
    end_idx = start_idx + body.page_size
    raw_page = results[start_idx:end_idx]

    def _clamp_score(val: Optional[Any]) -> Optional[float]:
        if val is None:
            return None
        try:
            v = float(val)
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return round(v / (1.0 + v), 4)
            return round(v, 4)
        except (ValueError, TypeError):
            return 0.0

    page_results = []
    for r in raw_page:
        item_id = r.get("id") or r.get("item_id") or r.get("review_id") or r.get("product_id") or ""
        item_type = str(r.get("type") or r.get("item_type") or "review").lower()
        snippet = r.get("text") or r.get("snippet") or r.get("review_text") or ""
        title = r.get("title") or r.get("product_name") or (f"Review {item_id}" if "review" in item_type else f"Product {item_id}")
        raw_sc = r.get("score") if r.get("score") is not None else r.get("combined_score")
        sc = _clamp_score(raw_sc) or 0.0
        comb = _clamp_score(r.get("combined_score")) if r.get("combined_score") is not None else sc
        page_results.append({
            "id": item_id,
            "item_id": item_id,
            "review_id": r.get("review_id") or item_id,
            "product_id": r.get("product_id"),
            "reviewer_id": r.get("reviewer_id"),
            "type": item_type,
            "item_type": item_type,
            "title": title,
            "text": snippet,
            "snippet": snippet,
            "score": sc,
            "combined_score": comb,
            "keyword_score": _clamp_score(r.get("keyword_score")),
            "semantic_score": _clamp_score(r.get("semantic_score")),
            "graph_score": _clamp_score(r.get("graph_score")),
            "model_score": float(r["model_score"]) if r.get("model_score") is not None else None,
            "predicted_class": r.get("predicted_class"),
        })

    latency_ms = round(time.monotonic() * 1000 - start_ms, 2)

    response: Dict[str, Any] = {
        "query": body.query,
        "mode": body.mode.value,
        "results": page_results,
        "items": page_results,
        "total": total,
        "page": body.page,
        "page_size": body.page_size,
        "latency_ms": latency_ms,
        "status": search_status,
    }
    if warning:
        response["warning"] = warning
    if body.min_score is not None:
        response["min_score_applied"] = body.min_score

    return response
