"""
main.py
-------
ReviewGraph FastAPI application entry-point.

Responsibilities
----------------
* Create the FastAPI app with metadata, CORS, and exception handlers.
* Manage service lifetimes via an async lifespan context manager.
* Wire all routers under the /api prefix.
* Expose a root health-probe route at GET /.
"""

from __future__ import annotations

# Initialize the numerical libraries before other app modules.
import numpy as np
import faiss
import torch

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings

# ─────────────────────────────────────────────────────────────────────────── #
# Logging bootstrap (runs before app creation so lifespan messages are visible)
# ─────────────────────────────────────────────────────────────────────────── #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────── #
# Dependency-injection helpers
# (imported by routers that need individual services)
# ─────────────────────────────────────────────────────────────────────────── #

def get_graph_service(request: Request):
    """Return the GraphService singleton stored on app.state."""
    return getattr(request.app.state, "graph_service", None)


def get_embedding_service(request: Request):
    """Return the EmbeddingService singleton stored on app.state."""
    return getattr(request.app.state, "embedding_service", None)


def get_classification_service(request: Request):
    """Return the ClassificationService singleton stored on app.state."""
    return getattr(request.app.state, "classification_service", None)


def get_retrieval_service(request: Request):
    """Return the RetrievalService singleton stored on app.state."""
    return getattr(request.app.state, "retrieval_service", None)


def get_explanation_service(request: Request):
    """Return the ExplanationService singleton stored on app.state."""
    return getattr(request.app.state, "explanation_service", None)


# ─────────────────────────────────────────────────────────────────────────── #
# Lifespan
# ─────────────────────────────────────────────────────────────────────────── #

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Async context manager that manages service startup and shutdown."""
    settings = get_settings()

    # Adjust log level from settings
    logging.getLogger().setLevel(settings.LOG_LEVEL.upper())

    logger.info("=== ReviewGraph API starting up (env=%s) ===", settings.APP_ENV)

    # ------------------------------------------------------------------ #
    # GraphService
    # ------------------------------------------------------------------ #
    graph_service = getattr(app.state, "graph_service", None)
    if graph_service is None:
        try:
            from app.services.graph_service import GraphService  # type: ignore

            graph_service = GraphService(
                uri=settings.NEO4J_URI,
                user=settings.NEO4J_USER,
                password=settings.NEO4J_PASSWORD,
            )
            await graph_service.create_constraints()
            logger.info("GraphService initialised and constraints ensured.")
        except Exception as exc:
            logger.error(
                "GraphService startup failed (API will run in degraded mode): %s", exc
            )
    app.state.graph_service = graph_service

    # ------------------------------------------------------------------ #
    # EmbeddingService
    # ------------------------------------------------------------------ #
    embedding_service = getattr(app.state, "embedding_service", None)
    if embedding_service is None:
        try:
            from app.services.embedding_service import EmbeddingService  # type: ignore

            embedding_service = EmbeddingService(
                model_name=settings.EMBEDDING_MODEL,
                index_path=settings.FAISS_INDEX_PATH,
                cache_path=settings.EMBEDDINGS_CACHE_PATH,
            )
            if embedding_service.is_ready():
                logger.info("EmbeddingService: FAISS index loaded.")
            else:
                logger.warning(
                    "EmbeddingService: FAISS index not ready at %s — "
                    "semantic search will be unavailable until index is built.",
                    settings.FAISS_INDEX_PATH,
                )
        except Exception as exc:
            logger.error(
                "EmbeddingService startup failed (semantic search disabled): %s", exc
            )
    app.state.embedding_service = embedding_service

    # ------------------------------------------------------------------ #
    # ClassificationService
    # ------------------------------------------------------------------ #
    classification_service = getattr(app.state, "classification_service", None)
    if classification_service is None:
        try:
            from app.services.classification_service import ClassificationService  # type: ignore

            classification_service = ClassificationService(
                models_path=settings.MODELS_PATH,
            )
            if classification_service.is_ready():
                logger.info("ClassificationService: model loaded.")
            else:
                logger.warning(
                    "ClassificationService: model not loaded from %s — "
                    "live predictions will be unavailable.",
                    settings.MODELS_PATH,
                )
        except Exception as exc:
            logger.error(
                "ClassificationService startup failed (predictions disabled): %s", exc
            )
    app.state.classification_service = classification_service

    # ------------------------------------------------------------------ #
    # RetrievalService
    # ------------------------------------------------------------------ #
    retrieval_service = getattr(app.state, "retrieval_service", None)
    if retrieval_service is None:
        try:
            from app.services.retrieval_service import RetrievalService  # type: ignore

            fts_db_path = os.path.join(settings.DATA_PROCESSED_PATH, "fts.db")
            retrieval_service = RetrievalService(
                graph_service=graph_service,
                embedding_service=embedding_service,
                fts_db_path=fts_db_path,
            )
            logger.info("RetrievalService initialised.")
        except Exception as exc:
            logger.error(
                "RetrievalService startup failed (advanced search disabled): %s", exc
            )
    app.state.retrieval_service = retrieval_service

    # ------------------------------------------------------------------ #
    # ExplanationService
    # ------------------------------------------------------------------ #
    explanation_service = getattr(app.state, "explanation_service", None)
    if explanation_service is None:
        try:
            from app.services.explanation_service import ExplanationService  # type: ignore

            explanation_service = ExplanationService(graph_service=graph_service)
            logger.info("ExplanationService initialised.")
        except Exception as exc:
            logger.error(
                "ExplanationService startup failed (NL explanations disabled): %s", exc
            )
    app.state.explanation_service = explanation_service

    logger.info("=== ReviewGraph API ready ===")

    # ── yield control to the running app ───────────────────────────────── #
    yield

    # ── Shutdown ────────────────────────────────────────────────────────── #
    logger.info("=== ReviewGraph API shutting down ===")
    if graph_service is not None:
        try:
            await graph_service.close()
            logger.info("GraphService driver closed.")
        except Exception as exc:
            logger.error("GraphService close error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────── #
# Application factory
# ─────────────────────────────────────────────────────────────────────────── #

def create_app() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title="ReviewGraph API",
        description=(
            "Explainable Graph-Based Information Retrieval for Fake Review Detection. "
            "Combines Neo4j graph queries, ML classification, and semantic embeddings "
            "to detect and explain suspicious review patterns."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------ #
    # CORS
    # ------------------------------------------------------------------ #
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # Exception handlers
    # ------------------------------------------------------------------ #
    from fastapi.encoders import jsonable_encoder

    @application.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        req: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("Validation error on %s: %s", req.url.path, exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "detail": jsonable_encoder(exc.errors()),
                "message": "Request validation failed.",
            },
        )

    @application.exception_handler(Exception)
    async def generic_exception_handler(req: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled exception on %s: %s",
            req.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal server error occurred.",
                "message": "Please check the server logs for details.",
            },
        )

    # ------------------------------------------------------------------ #
    # Routers
    # ------------------------------------------------------------------ #
    from app.api.health import router as health_router
    from app.api.stats import router as stats_router
    from app.api.products import router as products_router
    from app.api.reviews import router as reviews_router
    from app.api.search import router as search_router
    from app.api.evaluation import router as evaluation_router

    application.include_router(health_router, prefix="/api")
    application.include_router(stats_router, prefix="/api")
    application.include_router(products_router, prefix="/api")
    application.include_router(reviews_router, prefix="/api")
    application.include_router(search_router, prefix="/api")
    application.include_router(evaluation_router, prefix="/api")

    # ------------------------------------------------------------------ #
    # Root route
    # ------------------------------------------------------------------ #

    @application.get("/", tags=["root"], summary="API root")
    async def root() -> dict:
        """Simple liveness probe — returns API name and docs URL."""
        return {
            "message": "ReviewGraph API",
            "docs": "/api/docs",
            "redoc": "/api/redoc",
            "health": "/api/health",
        }

    return application


# ─────────────────────────────────────────────────────────────────────────── #
# Module-level app instance (used by uvicorn)
# ─────────────────────────────────────────────────────────────────────────── #
app = create_app()
