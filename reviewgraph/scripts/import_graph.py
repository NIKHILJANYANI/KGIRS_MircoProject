"""
import_graph.py
--------------
Import the processed ReviewGraph data into Neo4j.

Usage
-----
  python import_graph.py
  python import_graph.py --use-sample
  python import_graph.py --neo4j-uri bolt://localhost:7687 \\
                         --neo4j-user neo4j --neo4j-password secret
"""


from __future__ import annotations
import sys as _sys_utf8
if hasattr(_sys_utf8.stdout, 'reconfigure'): _sys_utf8.stdout.reconfigure(encoding='utf-8')

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _require_neo4j() -> None:
    try:
        import neo4j  # noqa: F401
    except ImportError:
        print("\n[ERROR] neo4j driver not installed.\n  Run: pip install neo4j\n")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _batched(iterable: List[Any], n: int) -> Iterable[List[Any]]:
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def _run_batch(tx, query: str, batch: List[Dict]) -> None:
    tx.run(query, batch=batch)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CONSTRAINTS = [
    "CREATE CONSTRAINT review_id_unique IF NOT EXISTS FOR (r:Review) REQUIRE r.review_id IS UNIQUE",
    "CREATE CONSTRAINT reviewer_id_unique IF NOT EXISTS FOR (rv:Reviewer) REQUIRE rv.reviewer_id IS UNIQUE",
    "CREATE CONSTRAINT product_id_unique IF NOT EXISTS FOR (p:Product) REQUIRE p.product_id IS UNIQUE",
    "CREATE CONSTRAINT evidence_id_unique IF NOT EXISTS FOR (e:Evidence) REQUIRE e.evidence_id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX review_model_score IF NOT EXISTS FOR (r:Review) ON (r.model_score)",
    "CREATE INDEX review_rating IF NOT EXISTS FOR (r:Review) ON (r.rating)",
    "CREATE INDEX review_is_fake IF NOT EXISTS FOR (r:Review) ON (r.is_fake)",
    "CREATE INDEX evidence_signal_type IF NOT EXISTS FOR (e:Evidence) ON (e.signal_type)",
]


def create_schema(session) -> None:
    print("[import_graph] Creating schema constraints and indexes …")
    for stmt in CONSTRAINTS:
        try:
            session.run(stmt)
        except Exception as exc:
            print(f"  WARNING: {exc}")
    for stmt in INDEXES:
        try:
            session.run(stmt)
        except Exception as exc:
            print(f"  WARNING: {exc}")
    print("[import_graph] Schema ready.")


# ---------------------------------------------------------------------------
# Import queries
# ---------------------------------------------------------------------------

MERGE_PRODUCT_Q = """
UNWIND $batch AS row
MERGE (p:Product {product_id: row.product_id})
SET p.review_count = row.review_count,
    p.avg_rating   = row.avg_rating,
    p.fake_ratio   = row.fake_ratio
"""

MERGE_REVIEWER_Q = """
UNWIND $batch AS row
MERGE (rv:Reviewer {reviewer_id: row.reviewer_id})
SET rv.review_count = row.review_count,
    rv.avg_rating   = row.avg_rating,
    rv.fake_ratio   = row.fake_ratio
"""

MERGE_REVIEW_Q = """
UNWIND $batch AS row
MERGE (r:Review {review_id: row.review_id})
SET r.rating       = row.rating,
    r.review_text  = row.review_text,
    r.word_count   = row.word_count,
    r.char_count   = row.char_count,
    r.is_fake      = row.is_fake,
    r.model_score  = row.model_score,
    r.timestamp    = row.timestamp
"""

WROTE_REL_Q = """
UNWIND $batch AS row
MATCH (rv:Reviewer {reviewer_id: row.reviewer_id})
MATCH (r:Review    {review_id:  row.review_id})
MERGE (rv)-[:WROTE]->(r)
"""

REVIEWS_REL_Q = """
UNWIND $batch AS row
MATCH (r:Review  {review_id:  row.review_id})
MATCH (p:Product {product_id: row.product_id})
MERGE (r)-[:REVIEWS]->(p)
"""

MERGE_EVIDENCE_Q = """
UNWIND $batch AS row
MERGE (e:Evidence {evidence_id: row.evidence_id})
SET e:EvidenceSignal,
    e.signal_id    = row.evidence_id,
    e.signal_type  = row.signal_type,
    e.signal_value = row.signal_value,
    e.description  = row.description
WITH e, row
MATCH (r:Review {review_id: row.review_id})
MERGE (r)-[:HAS_SIGNAL]->(e)
MERGE (r)-[:HAS_EVIDENCE]->(e)
"""

SIMILAR_TO_Q = """
UNWIND $batch AS row
MATCH (r1:Review {review_id: row.review_id_1})
MATCH (r2:Review {review_id: row.review_id_2})
MERGE (r1)-[s:SIMILAR_TO]-(r2)
SET s.similarity_score = row.similarity_score,
    s.score = row.similarity_score,
    s.similarity_type = 'tfidf_cosine'
"""


# ---------------------------------------------------------------------------
# Evidence signal detection
# ---------------------------------------------------------------------------

def build_evidence_rows(df: pd.DataFrame) -> List[Dict]:
    rows = []
    ev_id = 0

    model_score_col = _detect(df, ["model_score"])
    rating_col = _detect(df, ["rating"])
    word_count_col = _detect(df, ["word_count"])
    reviewer_count_col = _detect(df, ["reviewer_review_count"])
    review_id_col = _detect(df, ["review_id"])

    for _, row in df.iterrows():
        rid = str(row[review_id_col]) if review_id_col else f"REV{ev_id}"

        # HIGH_MODEL_SCORE
        if model_score_col and pd.notna(row.get(model_score_col)):
            score = float(row[model_score_col])
            if score > 0.7:
                rows.append({
                    "evidence_id": f"EV_{rid}_HMS",
                    "review_id": rid,
                    "signal_type": "HIGH_MODEL_SCORE",
                    "signal_value": round(score, 4),
                    "description": f"Model predicted fake probability {score:.2%}",
                })

        # EXTREME_RATING
        if rating_col and pd.notna(row.get(rating_col)):
            rating = float(row[rating_col])
            if rating in {1.0, 5.0}:
                rows.append({
                    "evidence_id": f"EV_{rid}_ER",
                    "review_id": rid,
                    "signal_type": "EXTREME_RATING",
                    "signal_value": rating,
                    "description": f"Extreme rating of {rating:.0f} stars",
                })

        # SHORT_REVIEW
        if word_count_col and pd.notna(row.get(word_count_col)):
            wc = int(row[word_count_col])
            if wc < 10:
                rows.append({
                    "evidence_id": f"EV_{rid}_SR",
                    "review_id": rid,
                    "signal_type": "SHORT_REVIEW",
                    "signal_value": wc,
                    "description": f"Review has only {wc} words",
                })

        # HIGH_REVIEWER_VOLUME
        if reviewer_count_col and pd.notna(row.get(reviewer_count_col)):
            rc = int(row[reviewer_count_col])
            if rc > 20:
                rows.append({
                    "evidence_id": f"EV_{rid}_HRV",
                    "review_id": rid,
                    "signal_type": "HIGH_REVIEWER_VOLUME",
                    "signal_value": rc,
                    "description": f"Reviewer has posted {rc} reviews",
                })

        ev_id += 1
    return rows


# ---------------------------------------------------------------------------
# Import functions
# ---------------------------------------------------------------------------

def import_products(session, pstats: pd.DataFrame, batch_size: int) -> int:
    rows = []
    for _, r in pstats.iterrows():
        rows.append({
            "product_id": str(r.get("product_id", "")),
            "review_count": int(r.get("review_count", 0)),
            "avg_rating": float(r["avg_rating"]) if "avg_rating" in r and pd.notna(r.get("avg_rating")) else None,
            "fake_ratio": float(r["fake_ratio"]) if "fake_ratio" in r and pd.notna(r.get("fake_ratio")) else None,
        })
    for batch in _batched(rows, batch_size):
        session.execute_write(_run_batch, MERGE_PRODUCT_Q, batch)
    return len(rows)


def import_reviewers(session, rstats: pd.DataFrame, batch_size: int) -> int:
    rows = []
    for _, r in rstats.iterrows():
        rows.append({
            "reviewer_id": str(r.get("reviewer_id", "")),
            "review_count": int(r.get("review_count", 0)),
            "avg_rating": float(r["avg_rating"]) if "avg_rating" in r and pd.notna(r.get("avg_rating")) else None,
            "fake_ratio": float(r["fake_ratio"]) if "fake_ratio" in r and pd.notna(r.get("fake_ratio")) else None,
        })
    for batch in _batched(rows, batch_size):
        session.execute_write(_run_batch, MERGE_REVIEWER_Q, batch)
    return len(rows)


def import_reviews(session, df: pd.DataFrame, batch_size: int) -> int:
    id_col = _detect(df, ["review_id", "reviewid", "id"])
    text_col = _detect(df, ["review_text_clean", "review_text", "text", "reviewtext", "review"])
    rating_col = _detect(df, ["rating"])
    wc_col = _detect(df, ["word_count"])
    cc_col = _detect(df, ["char_count"])
    fake_col = _detect(df, ["is_fake"])
    score_col = _detect(df, ["model_score"])
    ts_col = _detect(df, ["timestamp"])

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "review_id": str(r[id_col]) if id_col else f"REV{_}",
            "rating": float(r[rating_col]) if rating_col and pd.notna(r.get(rating_col)) else None,
            "review_text": str(r[text_col])[:2000] if text_col else "",
            "word_count": int(r[wc_col]) if wc_col and pd.notna(r.get(wc_col)) else None,
            "char_count": int(r[cc_col]) if cc_col and pd.notna(r.get(cc_col)) else None,
            "is_fake": int(r[fake_col]) if fake_col and pd.notna(r.get(fake_col)) else None,
            "model_score": float(r[score_col]) if score_col and pd.notna(r.get(score_col)) else None,
            "timestamp": str(r[ts_col]) if ts_col and pd.notna(r.get(ts_col)) else None,
        })

    for batch in _batched(rows, batch_size):
        session.execute_write(_run_batch, MERGE_REVIEW_Q, batch)
    return len(rows)


def import_wrote_rels(session, df: pd.DataFrame, batch_size: int) -> int:
    id_col = _detect(df, ["review_id", "reviewid", "id"])
    rev_col = _detect(df, ["reviewer_id"])
    if not id_col or not rev_col:
        print("[import_graph] SKIP WROTE rels: missing review_id or reviewer_id column.")
        return 0
    rows = [
        {"review_id": str(r[id_col]), "reviewer_id": str(r[rev_col])}
        for _, r in df.iterrows()
        if pd.notna(r.get(rev_col))
    ]
    for batch in _batched(rows, batch_size):
        session.execute_write(_run_batch, WROTE_REL_Q, batch)
    return len(rows)


def import_reviews_rels(session, df: pd.DataFrame, batch_size: int) -> int:
    id_col = _detect(df, ["review_id", "reviewid", "id"])
    prod_col = _detect(df, ["product_id"])
    if not id_col or not prod_col:
        print("[import_graph] SKIP REVIEWS rels: missing review_id or product_id column.")
        return 0
    rows = [
        {"review_id": str(r[id_col]), "product_id": str(r[prod_col])}
        for _, r in df.iterrows()
        if pd.notna(r.get(prod_col))
    ]
    for batch in _batched(rows, batch_size):
        session.execute_write(_run_batch, REVIEWS_REL_Q, batch)
    return len(rows)


def import_evidence(session, df: pd.DataFrame, batch_size: int) -> int:
    print("[import_graph] Building evidence signals …")
    ev_rows = build_evidence_rows(df)
    print(f"[import_graph]   {len(ev_rows):,} evidence signals generated")
    for batch in _batched(ev_rows, batch_size):
        session.execute_write(_run_batch, MERGE_EVIDENCE_Q, batch)
    return len(ev_rows)


def import_similarity(session, sim_df: pd.DataFrame, batch_size: int) -> int:
    rows = [
        {
            "review_id_1": str(r["review_id_1"]),
            "review_id_2": str(r["review_id_2"]),
            "similarity_score": float(r["similarity_score"]),
        }
        for _, r in sim_df.iterrows()
    ]
    for batch in _batched(rows, batch_size):
        session.execute_write(_run_batch, SIMILAR_TO_Q, batch)
    return len(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import ReviewGraph processed data into Neo4j.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--processed-path",
        default=str(Path(__file__).parent / ".." / "data" / "processed"),
        help="Directory containing processed CSVs.",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j URI (env: NEO4J_URI).",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username (env: NEO4J_USER).",
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.environ.get("NEO4J_PASSWORD", "password"),
        help="Neo4j password (env: NEO4J_PASSWORD).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Cypher UNWIND batch size (default: 500).",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Load from ../data/sample/ instead of processed/.",
    )
    parser.add_argument(
        "--skip-similarity",
        action="store_true",
        help="Skip SIMILAR_TO relationship import.",
    )
    return parser.parse_args()


def _load_csv(path: Path, label: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[import_graph] WARNING: {label} not found at {path.resolve()} — skipping.")
        return None
    df = pd.read_csv(path, low_memory=False)
    print(f"[import_graph] Loaded {label}: {len(df):,} rows")
    return df


def main() -> None:
    args = parse_args()
    t0 = time.time()

    _require_neo4j()
    from neo4j import GraphDatabase  # type: ignore

    # Load dotenv if available
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent / ".." / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            args.neo4j_uri = os.environ.get("NEO4J_URI", args.neo4j_uri)
            args.neo4j_user = os.environ.get("NEO4J_USER", args.neo4j_user)
            args.neo4j_password = os.environ.get("NEO4J_PASSWORD", args.neo4j_password)
            print(f"[import_graph] Loaded .env from {env_path.resolve()}")
    except ImportError:
        pass

    # --- Resolve paths ---
    if args.use_sample:
        data_dir = Path(__file__).parent / ".." / "data" / "sample"
    else:
        data_dir = Path(args.processed_path)

    proc_file = data_dir / "reviews_processed.csv"
    if not proc_file.exists() and not args.use_sample:
        proc_file = Path(args.processed_path) / "reviews_processed.csv"

    reviewer_file = Path(args.processed_path) / "reviewer_stats.csv"
    product_file = Path(args.processed_path) / "product_stats.csv"
    sim_file = Path(args.processed_path) / "similarity_pairs.csv"

    # --- Load data ---
    df_reviews = _load_csv(proc_file, "reviews_processed.csv")
    if df_reviews is None:
        print(
            f"[ERROR] Processed reviews not found at {proc_file.resolve()}\n"
            "  → Run preprocess_data.py first.\n"
        )
        sys.exit(1)

    df_reviewers = _load_csv(reviewer_file, "reviewer_stats.csv")
    df_products = _load_csv(product_file, "product_stats.csv")
    df_sim = None if args.skip_similarity else _load_csv(sim_file, "similarity_pairs.csv")

    # --- Auto-create reviewer/product stats if files missing ---
    if df_reviewers is None and "reviewer_id" in df_reviews.columns:
        print("[import_graph] Generating reviewer stats inline …")
        df_reviewers = (
            df_reviews.groupby("reviewer_id")
            .agg(review_count=("reviewer_id", "count"))
            .reset_index()
        )

    if df_products is None:
        prod_col = _detect(df_reviews, ["product_id"])
        if prod_col:
            print("[import_graph] Generating product stats inline …")
            df_products = (
                df_reviews.groupby(prod_col)
                .agg(review_count=(prod_col, "count"))
                .reset_index()
                .rename(columns={prod_col: "product_id"})
            )

    # --- Connect ---
    print(f"\n[import_graph] Connecting to Neo4j at {args.neo4j_uri} …")
    try:
        driver = GraphDatabase.driver(
            args.neo4j_uri,
            auth=(args.neo4j_user, args.neo4j_password),
        )
        driver.verify_connectivity()
        print("[import_graph] Connected successfully.")
    except Exception as exc:
        print(
            f"[ERROR] Cannot connect to Neo4j: {exc}\n"
            "  → Ensure Neo4j is running and credentials are correct.\n"
            "  → Set NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD environment variables.\n"
        )
        sys.exit(1)

    stats: Dict[str, int] = {}

    with driver.session() as session:
        create_schema(session)

        if df_products is not None:
            print(f"\n[import_graph] Importing {len(df_products):,} products …")
            stats["products"] = import_products(session, df_products, args.batch_size)
            print(f"  ✓ {stats['products']:,} products merged")

        if df_reviewers is not None:
            print(f"\n[import_graph] Importing {len(df_reviewers):,} reviewers …")
            stats["reviewers"] = import_reviewers(session, df_reviewers, args.batch_size)
            print(f"  ✓ {stats['reviewers']:,} reviewers merged")

        print(f"\n[import_graph] Importing {len(df_reviews):,} reviews …")
        stats["reviews"] = import_reviews(session, df_reviews, args.batch_size)
        print(f"  ✓ {stats['reviews']:,} reviews merged")

        print("\n[import_graph] Creating WROTE relationships …")
        stats["wrote_rels"] = import_wrote_rels(session, df_reviews, args.batch_size)
        print(f"  ✓ {stats['wrote_rels']:,} WROTE relationships")

        print("\n[import_graph] Creating REVIEWS relationships …")
        stats["reviews_rels"] = import_reviews_rels(session, df_reviews, args.batch_size)
        print(f"  ✓ {stats['reviews_rels']:,} REVIEWS relationships")

        print("\n[import_graph] Importing evidence signals …")
        stats["evidence"] = import_evidence(session, df_reviews, args.batch_size)
        print(f"  ✓ {stats['evidence']:,} evidence nodes")

        if df_sim is not None and len(df_sim) > 0:
            print(f"\n[import_graph] Importing {len(df_sim):,} SIMILAR_TO relationships …")
            stats["similar_to_rels"] = import_similarity(session, df_sim, args.batch_size)
            print(f"  ✓ {stats['similar_to_rels']:,} SIMILAR_TO relationships")

    driver.close()
    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("GRAPH IMPORT COMPLETE")
    print("=" * 60)
    for key, val in stats.items():
        print(f"  {key:<25} : {val:>10,}")
    print(f"  {'elapsed_seconds':<25} : {elapsed:>10.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
