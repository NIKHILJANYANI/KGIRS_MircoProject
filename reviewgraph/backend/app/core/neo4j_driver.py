"""
Neo4j database driver module for the ReviewGraph backend.

Provides:
- ``Neo4jDatabase`` – lifecycle-managed driver wrapper with context-manager support
- ``init_db()``     – creates schema constraints and indexes on first run
- ``get_driver()``  – FastAPI dependency that yields the driver instance
- ``get_neo4j_session()`` – FastAPI dependency that yields an open session

The module is deliberately fault-tolerant: if Neo4j is unreachable at startup
the application continues to run and endpoints that require the database return
HTTP 503 instead of crashing the whole process.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable, AuthError

from app.core.config import get_settings

logger = logging.getLogger("reviewgraph.neo4j")

# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #
db_instance: Optional["Neo4jDatabase"] = None


# --------------------------------------------------------------------------- #
# Driver wrapper
# --------------------------------------------------------------------------- #

class Neo4jDatabase:
    """
    Thin wrapper around the Neo4j async driver.

    Usage (manual)::

        db = Neo4jDatabase(uri, user, password)
        await db.connect()
        ...
        await db.close()

    Usage (context manager)::

        async with Neo4jDatabase(uri, user, password) as db:
            async with db.driver.session() as session:
                ...
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self.driver: Optional[AsyncDriver] = None
        self._connected: bool = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def connect(self) -> bool:
        """
        Attempt to create and verify the driver.

        Returns ``True`` on success, ``False`` if Neo4j is unavailable.
        The caller should check ``self._connected`` before using the driver.
        """
        try:
            self.driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=30,
            )
            await self.driver.verify_connectivity()
            self._connected = True
            logger.info("Connected to Neo4j", extra={"uri": self._uri})
            return True
        except ServiceUnavailable as exc:
            logger.warning(
                "Neo4j is unavailable – running in degraded mode",
                extra={"uri": self._uri, "error": str(exc)},
            )
            self.driver = None
            self._connected = False
            return False
        except AuthError as exc:
            logger.error(
                "Neo4j authentication failed – check NEO4J_USER / NEO4J_PASSWORD",
                extra={"uri": self._uri, "error": str(exc)},
            )
            self.driver = None
            self._connected = False
            return False
        except Exception as exc:  # pragma: no cover
            logger.error(
                "Unexpected error connecting to Neo4j",
                extra={"uri": self._uri, "error": str(exc)},
            )
            self.driver = None
            self._connected = False
            return False

    async def close(self) -> None:
        """Close the driver and release all connections."""
        if self.driver is not None:
            await self.driver.close()
            self.driver = None
            self._connected = False
            logger.info("Neo4j driver closed")

    async def verify_connectivity(self) -> bool:
        """Return ``True`` if the driver can reach Neo4j right now."""
        if self.driver is None:
            return False
        try:
            await self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected and self.driver is not None

    # ------------------------------------------------------------------ #
    # Context manager
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> "Neo4jDatabase":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        await self.close()

    # Sync context manager (for use in non-async code / testing)
    def __enter__(self) -> "Neo4jDatabase":  # type: ignore[misc]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


# --------------------------------------------------------------------------- #
# Schema / index initialisation
# --------------------------------------------------------------------------- #

async def init_db(db: Neo4jDatabase) -> None:
    """
    Create uniqueness constraints and indexes required by the application.

    Safe to call multiple times – ``IF NOT EXISTS`` guards prevent errors on
    re-runs.
    """
    if not db.is_connected or db.driver is None:
        logger.warning("Skipping schema init – Neo4j not connected")
        return

    constraints = [
        # Unique constraints
        (
            "constraint_product_id",
            "CREATE CONSTRAINT constraint_product_id IF NOT EXISTS "
            "FOR (p:Product) REQUIRE p.product_id IS UNIQUE",
        ),
        (
            "constraint_reviewer_id",
            "CREATE CONSTRAINT constraint_reviewer_id IF NOT EXISTS "
            "FOR (r:Reviewer) REQUIRE r.reviewer_id IS UNIQUE",
        ),
        (
            "constraint_review_id",
            "CREATE CONSTRAINT constraint_review_id IF NOT EXISTS "
            "FOR (rv:Review) REQUIRE rv.review_id IS UNIQUE",
        ),
        (
            "constraint_signal_id",
            "CREATE CONSTRAINT constraint_signal_id IF NOT EXISTS "
            "FOR (s:EvidenceSignal) REQUIRE s.signal_id IS UNIQUE",
        ),
    ]

    indexes = [
        (
            "index_review_predicted_class",
            "CREATE INDEX index_review_predicted_class IF NOT EXISTS "
            "FOR (rv:Review) ON (rv.predicted_class)",
        ),
        (
            "index_review_model_score",
            "CREATE INDEX index_review_model_score IF NOT EXISTS "
            "FOR (rv:Review) ON (rv.model_score)",
        ),
        (
            "index_product_title",
            "CREATE INDEX index_product_title IF NOT EXISTS "
            "FOR (p:Product) ON (p.title)",
        ),
    ]

    async with db.driver.session() as session:
        for name, cypher in constraints + indexes:
            try:
                await session.run(cypher)
                logger.debug("Schema statement executed", extra={"name": name})
            except Exception as exc:
                logger.warning(
                    "Schema statement failed (non-fatal)",
                    extra={"name": name, "error": str(exc)},
                )

    logger.info("Neo4j schema initialisation complete")


# --------------------------------------------------------------------------- #
# FastAPI dependencies
# --------------------------------------------------------------------------- #

async def get_driver() -> AsyncGenerator[Optional[AsyncDriver], None]:
    """
    FastAPI dependency – yields the raw ``AsyncDriver`` or ``None`` when
    Neo4j is unavailable.  Endpoints must handle the ``None`` case.
    """
    global db_instance
    if db_instance is not None and db_instance.driver is not None:
        yield db_instance.driver
    else:
        yield None


async def get_neo4j_session() -> AsyncGenerator[Optional[AsyncSession], None]:
    """
    FastAPI dependency – yields an open ``AsyncSession`` or ``None`` when Neo4j
    is unavailable.  Endpoints should return HTTP 503 when they receive ``None``.

    Example::

        @router.get("/")
        async def my_endpoint(session: Optional[AsyncSession] = Depends(get_neo4j_session)):
            if session is None:
                raise HTTPException(503, "Graph database unavailable")
            ...
    """
    global db_instance
    if db_instance is None or db_instance.driver is None:
        yield None
        return

    async with db_instance.driver.session() as session:
        yield session


# --------------------------------------------------------------------------- #
# Module initialisation helper (called from app lifespan)
# --------------------------------------------------------------------------- #

async def create_db_instance() -> Neo4jDatabase:
    """
    Instantiate, connect, and store the module-level ``db_instance``.
    Called once during application startup (lifespan context).
    """
    global db_instance
    settings = get_settings()
    db_instance = Neo4jDatabase(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
    )
    connected = await db_instance.connect()
    if connected:
        await init_db(db_instance)
    return db_instance


async def close_db_instance() -> None:
    """Close and clear the module-level ``db_instance``."""
    global db_instance
    if db_instance is not None:
        await db_instance.close()
        db_instance = None
