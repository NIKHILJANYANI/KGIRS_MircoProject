"""
health.py
---------
GET /api/health — Returns overall system health including Neo4j, model,
embedding, and dataset status.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, Request

from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


def _safe_bool(coro_result: Any) -> bool:
    """Coerce any truthy value to bool safely."""
    return bool(coro_result)


@router.get("/health", summary="System health check")
async def health_check(request: Request) -> Dict[str, Any]:
    """
    Return the health status of every sub-system.

    Status values per component
    ---------------------------
    * ``ok``             — service reachable and ready
    * ``unavailable``    — service unreachable / threw an exception
    * ``not_initialized``— service exists but has not been initialised yet

    Overall status
    --------------
    * ``ok``      — Neo4j is reachable
    * ``degraded``— Neo4j is unavailable (everything else continues)
    """
    settings = get_settings()

    # ------------------------------------------------------------------ #
    # Neo4j connectivity
    # ------------------------------------------------------------------ #
    neo4j_status: str
    neo4j_detail: str = ""
    graph_service = getattr(request.app.state, "graph_service", None)
    if graph_service is None:
        neo4j_status = "unavailable"
        neo4j_detail = "GraphService not initialised"
    else:
        try:
            connected = await graph_service.is_connected()
            neo4j_status = "ok" if connected else "unavailable"
            if not connected:
                neo4j_detail = "Connection check returned False"
        except Exception as exc:
            neo4j_status = "unavailable"
            neo4j_detail = str(exc)
            logger.warning("Health – Neo4j check failed: %s", exc)

    # ------------------------------------------------------------------ #
    # ML model status
    # ------------------------------------------------------------------ #
    model_status: str
    classification_service = getattr(request.app.state, "classification_service", None)
    if classification_service is None:
        model_status = "unavailable"
    else:
        try:
            ready = classification_service.is_ready()
            model_status = "ok" if ready else "not_initialized"
        except Exception as exc:
            model_status = "unavailable"
            logger.warning("Health – model check failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Embedding service status
    # ------------------------------------------------------------------ #
    embedding_status: str
    embedding_service = getattr(request.app.state, "embedding_service", None)
    if embedding_service is None:
        embedding_status = "unavailable"
    else:
        try:
            ready = embedding_service.is_ready()
            embedding_status = "ok" if ready else "not_initialized"
        except Exception as exc:
            embedding_status = "unavailable"
            logger.warning("Health – embedding check failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Dataset status – check whether processed file exists on disk
    # ------------------------------------------------------------------ #
    processed_path = settings.DATA_PROCESSED_PATH
    dataset_status: str
    try:
        # Accept any CSV / parquet / json file inside the processed dir
        if os.path.isdir(processed_path):
            files = [
                f for f in os.listdir(processed_path)
                if f.endswith((".csv", ".parquet", ".json", ".jsonl"))
            ]
            dataset_status = "ok" if files else "not_initialized"
        else:
            dataset_status = "not_initialized"
    except Exception as exc:
        dataset_status = "unavailable"
        logger.warning("Health – dataset check failed: %s", exc)

    # ------------------------------------------------------------------ #
    # Overall status
    # ------------------------------------------------------------------ #
    overall_status = "ok" if neo4j_status == "ok" else "degraded"

    # Flat fields for frontend StatusBadge components
    response: Dict[str, Any] = {
        "status": "healthy" if overall_status == "ok" else overall_status,
        "neo4j": neo4j_status,
        "model": model_status,
        "embeddings": embedding_status,
        "dataset": dataset_status,
        "services": {
            "neo4j": "connected" if neo4j_status == "ok" else "unavailable",
            "model": "loaded" if model_status == "ok" else "unavailable",
            "faiss": "loaded" if embedding_status == "ok" else "unavailable",
            "dataset": "loaded" if dataset_status == "ok" else "unavailable",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        # Nested detail for programmatic consumers
        "components": {
            "neo4j": {"status": neo4j_status, "detail": neo4j_detail or ""},
            "model": {"status": model_status},
            "embeddings": {"status": embedding_status},
            "dataset": {"status": dataset_status},
        },
    }

    return response

