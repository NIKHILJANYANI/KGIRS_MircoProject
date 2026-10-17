"""
products.py
-----------
Product-related API endpoints:
  GET /api/products                       — paginated search / list
  GET /api/products/{product_id}          — product detail
  GET /api/products/{product_id}/investigation — full investigation
  GET /api/products/{product_id}/graph    — graph visualisation data
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)

router = APIRouter(tags=["products"])


# --------------------------------------------------------------------------- #
# Helper – guard against unavailable graph service
# --------------------------------------------------------------------------- #

def _require_graph(request: Request):
    svc = getattr(request.app.state, "graph_service", None)
    if svc is None:
        raise HTTPException(
            status_code=503,
            detail="Graph service is unavailable. Please check Neo4j connectivity.",
        )
    return svc


# --------------------------------------------------------------------------- #
# GET /api/products
# --------------------------------------------------------------------------- #

@router.get("/products", summary="Search or list products")
async def list_products(
    request: Request,
    query: Optional[str] = Query(None, description="Full-text search term for product title"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
) -> Dict[str, Any]:
    """
    Return a paginated list of products.

    * If *query* is provided, products are filtered by title/name match.
    * Results are ordered by computed fake-review suspiciousness descending.
    """
    graph_service = _require_graph(request)

    skip = (page - 1) * page_size

    if query and query.strip():
        cypher = """
        MATCH (p:Product)
        WHERE toLower(p.name) CONTAINS toLower($query)
           OR toLower(coalesce(p.category,'')) CONTAINS toLower($query)
        OPTIONAL MATCH (r:Review)-[:REVIEWS]->(p)
        WITH p,
             count(r)                                                    AS review_count,
             avg(r.model_score)                                          AS avg_model_score,
             sum(CASE WHEN r.predicted_class = 'fake' THEN 1 ELSE 0 END) AS fake_count
        RETURN p {
            .*,
            computed_review_count:     review_count,
            computed_avg_model_score:  avg_model_score,
            computed_fake_count:       fake_count
        } AS product
        ORDER BY avg_model_score DESC
        SKIP $skip LIMIT $page_size
        """
        count_cypher = """
        MATCH (p:Product)
        WHERE toLower(p.name) CONTAINS toLower($query)
           OR toLower(coalesce(p.category,'')) CONTAINS toLower($query)
        RETURN count(p) AS total
        """
        params = {"query": query.strip(), "skip": skip, "page_size": page_size}
        count_params = {"query": query.strip()}
    else:
        cypher = """
        MATCH (p:Product)
        OPTIONAL MATCH (r:Review)-[:REVIEWS]->(p)
        WITH p,
             count(r)                                                    AS review_count,
             avg(r.model_score)                                          AS avg_model_score,
             sum(CASE WHEN r.predicted_class = 'fake' THEN 1 ELSE 0 END) AS fake_count
        RETURN p {
            .*,
            computed_review_count:    review_count,
            computed_avg_model_score: avg_model_score,
            computed_fake_count:      fake_count
        } AS product
        ORDER BY avg_model_score DESC
        SKIP $skip LIMIT $page_size
        """
        count_cypher = "MATCH (p:Product) RETURN count(p) AS total"
        params = {"skip": skip, "page_size": page_size}
        count_params = {}

    try:
        products: List[Dict] = []
        rows = await graph_service.run_parameterized(cypher, params)
        for row in rows:
            products.append(dict(row.get("product", row)))

        total_rows = await graph_service.run_parameterized(count_cypher, count_params)
        total = total_rows[0].get("total", 0) if total_rows else len(products)

        return {
            "items": products,
            "total": total,
            "page": page,
            "page_size": page_size,
            "pages": math.ceil(total / page_size) if total else 1,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("list_products error: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Graph query failed.") from exc


# --------------------------------------------------------------------------- #
# GET /api/products/{product_id}
# --------------------------------------------------------------------------- #

@router.get("/products/{product_id}", summary="Product detail")
async def get_product(
    product_id: str,
    request: Request,
) -> Dict[str, Any]:
    """Return detailed information for a single product including computed stats."""
    graph_service = _require_graph(request)

    try:
        product = await graph_service.get_product(product_id)
    except Exception as exc:
        logger.error("get_product(%s) error: %s", product_id, exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Graph query failed.") from exc

    if product is None:
        raise HTTPException(status_code=404, detail=f"Product '{product_id}' not found.")

    return {"product": product}


# --------------------------------------------------------------------------- #
# GET /api/products/{product_id}/investigation
# --------------------------------------------------------------------------- #

@router.get("/products/{product_id}/investigation", summary="Full product investigation")
async def investigate_product(
    product_id: str,
    request: Request,
) -> Dict[str, Any]:
    """
    Run a comprehensive fake-review investigation for a product.

    Combines graph traversal, suspiciousness scoring, reviewer overlap
    analysis, and an NL explanation (if explanation service is available).
    """
    graph_service = _require_graph(request)

    try:
        investigation = await graph_service.investigate_product(product_id)
    except Exception as exc:
        logger.error("investigate_product(%s) graph error: %s", product_id, exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Graph investigation failed.") from exc

    if "error" in investigation:
        raise HTTPException(status_code=404, detail=investigation["error"])

    # Optional: enrich with NL explanation
    explanation_service = getattr(request.app.state, "explanation_service", None)
    if explanation_service is not None:
        try:
            explanation = await explanation_service.generate_product_explanation(investigation)
            investigation["explanation"] = explanation
        except Exception as exc:
            logger.warning(
                "investigate_product(%s) explanation failed: %s", product_id, exc
            )
            investigation["explanation"] = None

    return investigation


# --------------------------------------------------------------------------- #
# GET /api/products/{product_id}/graph
# --------------------------------------------------------------------------- #

@router.get("/products/{product_id}/graph", summary="Graph visualisation data")
async def get_product_graph(
    product_id: str,
    request: Request,
    depth: int = Query(2, ge=1, le=4, description="Traversal depth from product node"),
    max_nodes: int = Query(100, ge=10, le=500, description="Maximum nodes to return"),
) -> Dict[str, Any]:
    """
    Return a Cytoscape.js-compatible graph payload for the product neighbourhood.

    The response contains ``nodes`` and ``edges`` arrays suitable for direct
    consumption by a front-end graph renderer.
    """
    graph_service = _require_graph(request)

    try:
        graph_data = await graph_service.get_product_graph(
            product_id, depth=depth, max_nodes=max_nodes
        )
    except Exception as exc:
        logger.error(
            "get_product_graph(%s) error: %s", product_id, exc, exc_info=True
        )
        raise HTTPException(status_code=503, detail="Graph query failed.") from exc

    if not graph_data.get("nodes") and not graph_data.get("edges"):
        # May mean product not found — attempt a direct lookup
        graph_service2 = _require_graph(request)
        exists = await graph_service2.get_product(product_id)
        if exists is None:
            raise HTTPException(
                status_code=404, detail=f"Product '{product_id}' not found."
            )

    return {
        "product_id": product_id,
        "depth": depth,
        "nodes": graph_data.get("nodes", []),
        "edges": graph_data.get("edges", []),
        "node_count": len(graph_data.get("nodes", [])),
        "edge_count": len(graph_data.get("edges", [])),
    }
