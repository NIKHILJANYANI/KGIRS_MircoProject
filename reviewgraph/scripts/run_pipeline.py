"""
Run the complete ReviewGraph data pipeline.

Usage from the project root:
    py -3 scripts/run_pipeline.py

The pipeline deliberately does not start Neo4j, the backend, or the frontend.
Neo4j must be running before the graph-import stage.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "external" / "amazon_reviews_electronics.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


class PipelineError(RuntimeError):
    """Raised when a required pipeline stage fails."""


def _command_text(command: Sequence[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def run_stage(name: str, command: Sequence[str]) -> None:
    """Run one required subprocess and stop on any non-zero result."""
    print(f"\n=== {name} ===")
    print(f"Command: {_command_text(command)}")
    started = time.perf_counter()
    completed = subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        check=False,
        env=os.environ.copy(),
    )
    duration = time.perf_counter() - started
    if completed.returncode != 0:
        raise PipelineError(
            f"Stage '{name}' failed with exit code {completed.returncode} "
            f"after {duration:.1f}s."
        )
    print(f"Result: SUCCESS ({duration:.1f}s)")


def validate_artifacts() -> None:
    """Verify the processed CSV and every embedding artifact share one ID order."""
    print("\n=== Validate processed data and embedding alignment ===")
    started = time.perf_counter()
    processed_path = PROCESSED_DIR / "reviews_processed.csv"
    cache_path = MODELS_DIR / "review_embeddings_cache.pkl"
    index_path = MODELS_DIR / "review_embeddings.faiss"
    metadata_path = MODELS_DIR / "review_embeddings_meta.json"
    id_map_path = MODELS_DIR / "review_id_map.pkl"
    fts_path = PROCESSED_DIR / "fts.db"

    required = [
        processed_path,
        cache_path,
        index_path,
        metadata_path,
        id_map_path,
        fts_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise PipelineError(f"Validation failed; missing artifacts: {missing}")

    frame = pd.read_csv(processed_path, low_memory=False)
    if frame.empty or frame["review_id"].isna().any():
        raise PipelineError("Validation failed; processed review IDs are missing.")
    processed_ids = frame["review_id"].astype(str).tolist()
    if len(processed_ids) != len(set(processed_ids)):
        raise PipelineError("Validation failed; processed review IDs are not unique.")

    with sqlite3.connect(fts_path) as connection:
        fts_count = connection.execute(
            "SELECT count(*) FROM reviews_fts"
        ).fetchone()[0]
    if fts_count != len(processed_ids):
        raise PipelineError(
            f"Validation failed; FTS has {fts_count} rows for "
            f"{len(processed_ids)} processed reviews."
        )

    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    cache_ids = [str(value) for value in cache.get("review_ids", [])]
    if cache_ids != processed_ids:
        raise PipelineError(
            "Validation failed; embedding cache IDs do not match processed CSV order."
        )

    with id_map_path.open("rb") as handle:
        id_map = pickle.load(handle)
    ordered_map = [str(id_map[index]) for index in range(len(id_map))]
    if ordered_map != processed_ids:
        raise PipelineError(
            "Validation failed; review_id_map.pkl does not match processed CSV order."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("n_vectors", -1)) != len(processed_ids):
        raise PipelineError("Validation failed; embedding metadata vector count differs.")

    try:
        import faiss

        index = faiss.read_index(str(index_path))
    except Exception as exc:
        raise PipelineError(f"Validation failed; could not read FAISS index: {exc}") from exc
    if index.ntotal != len(processed_ids):
        raise PipelineError(
            f"Validation failed; FAISS has {index.ntotal} vectors for "
            f"{len(processed_ids)} processed reviews."
        )

    print(
        f"Result: SUCCESS ({time.perf_counter() - started:.1f}s) — "
        f"{len(processed_ids):,} reviews and aligned FAISS vectors"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ReviewGraph preprocessing, indexing, FTS, graph import, and validation."
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI passed to import_graph.py.",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username passed to import_graph.py.",
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.environ.get("NEO4J_PASSWORD", "reviewgraph123"),
        help="Neo4j password passed to import_graph.py.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not DATASET_PATH.exists():
        print(f"ERROR: active dataset not found: {DATASET_PATH}", file=sys.stderr)
        return 1

    scripts_dir = PROJECT_ROOT / "scripts"
    try:
        run_stage(
            "Preprocess active Amazon Electronics dataset",
            [
                sys.executable,
                str(scripts_dir / "preprocess_data.py"),
                "--dataset-path",
                str(DATASET_PATH),
                "--output-dir",
                str(PROCESSED_DIR),
            ],
        )
        run_stage(
            "Force-rebuild FAISS embeddings",
            [
                sys.executable,
                str(scripts_dir / "build_embeddings.py"),
                "--processed-path",
                str(PROCESSED_DIR / "reviews_processed.csv"),
                "--models-path",
                str(MODELS_DIR),
                "--force-rebuild",
            ],
        )
        run_stage(
            "Rebuild full-text search index",
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    f"sys.path.insert(0, {str(PROJECT_ROOT / 'backend')!r}); "
                    "from app.services.retrieval_service import RetrievalService; "
                    f"RetrievalService(None, None, {str(PROCESSED_DIR / 'fts.db')!r}); "
                    "print('FTS rebuilt from active processed CSV')"
                ),
            ],
        )
        run_stage(
            "Import processed data into Neo4j",
            [
                sys.executable,
                str(scripts_dir / "import_graph.py"),
                "--processed-path",
                str(PROCESSED_DIR),
                "--neo4j-uri",
                args.neo4j_uri,
                "--neo4j-user",
                args.neo4j_user,
                "--neo4j-password",
                args.neo4j_password,
            ],
        )
        validate_artifacts()
    except PipelineError as exc:
        print(f"\nPIPELINE FAILED: {exc}", file=sys.stderr)
        return 1

    print("\nPIPELINE COMPLETE: all required stages succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
