"""
Application configuration using Pydantic Settings.
Reads from environment variables and .env file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Neo4j
    # ------------------------------------------------------------------ #
    NEO4J_URI: str = Field(default="bolt://localhost:7687")
    NEO4J_USER: str = Field(default="neo4j")
    # Fallback default so the app can start without a .env file
    NEO4J_PASSWORD: str = Field(default="reviewgraph123")

    # ------------------------------------------------------------------ #
    # Application server
    # ------------------------------------------------------------------ #
    APP_ENV: str = Field(default="development")
    APP_PORT: int = Field(default=8000)
    APP_HOST: str = Field(default="0.0.0.0")
    LOG_LEVEL: str = Field(default="INFO")

    # ------------------------------------------------------------------ #
    # File-system paths
    # ------------------------------------------------------------------ #
    DATA_RAW_PATH: str = Field(default="../data/raw")
    DATA_PROCESSED_PATH: str = Field(default="../data/processed")
    DATA_SAMPLE_PATH: str = Field(default="../data/sample")
    MODELS_PATH: str = Field(default="../models")

    MODEL_FILENAME: str = Field(default="review_classifier.pkl")
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")
    FAISS_INDEX_PATH: str = Field(default="../models/faiss_index.bin")
    EMBEDDINGS_CACHE_PATH: str = Field(default="../models/embeddings_cache.npz")

    DATASET_PATH: str = Field(default="../data/raw/fake_reviews.csv")
    SAMPLE_DATASET_PATH: str = Field(default="../data/sample/sample_reviews.csv")

    # ------------------------------------------------------------------ #
    # Algorithm / retrieval parameters
    # ------------------------------------------------------------------ #
    SIMILARITY_THRESHOLD: float = Field(default=0.75)
    MAX_GRAPH_TRAVERSAL_DEPTH: int = Field(default=3)
    MAX_RESULTS_PER_QUERY: int = Field(default=50)
    MAX_GRAPH_NODES: int = Field(default=200)

    # ------------------------------------------------------------------ #
    # CORS
    # ------------------------------------------------------------------ #
    CORS_ORIGINS: Union[List[str], str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"]
    )

    @field_validator("CORS_ORIGINS", mode="after")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        """Support both JSON lists and comma-separated strings for CORS_ORIGINS."""
        if isinstance(v, str):
            v_stripped = v.strip()
            if v_stripped.startswith("[") and v_stripped.endswith("]"):
                import json
                try:
                    parsed = json.loads(v_stripped)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except Exception:
                    pass
            return [item.strip() for item in v.split(",") if item.strip()]
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return ["http://localhost:5173", "http://localhost:3000"]

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #
    @property
    def model_path(self) -> str:
        """Absolute-style convenience path to the classifier pickle."""
        import os
        return os.path.join(self.MODELS_PATH, self.MODEL_FILENAME)

    @property
    def is_development(self) -> bool:
        return self.APP_ENV.lower() == "development"

    @property
    def is_production(self) -> bool:
        return self.APP_ENV.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
