"""
retrieval_service.py
--------------------
Hybrid search service for ReviewGraph: keyword (SQLite FTS5),
semantic (embedding), and graph-based search combined.
"""

import logging
import time
import math
import sqlite3
import asyncio
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class RetrievalService:
    """Combines keyword, semantic, and graph search into a unified hybrid retrieval pipeline."""

    def __init__(self, graph_service, embedding_service, fts_db_path: str):
        """Initialise the retrieval service."""
        self.graph_service = graph_service
        self.embedding_service = embedding_service
        self.fts_db_path = fts_db_path
        self._fts_ready = False
        self._reviews_cache: Dict[str, Dict] = {}
        logger.info("RetrievalService initialised. FTS db: %s", fts_db_path)
        self._ensure_fts_and_cache()

    def _ensure_fts_and_cache(self) -> None:
        """Load dataset into memory cache and ensure SQLite FTS5 table is populated."""
        if self._fts_ready and self._reviews_cache:
            return

        import os
        import pandas as pd

        candidate_paths = [
            os.path.join(os.path.dirname(self.fts_db_path), "reviews_processed.csv"),
            "data/processed/reviews_processed.csv",
            "../data/processed/reviews_processed.csv",
            "data/sample/sample_reviews.csv",
            "../data/sample/sample_reviews.csv",
        ]

        df: Optional[pd.DataFrame] = None
        for p in candidate_paths:
            if os.path.exists(p):
                try:
                    df = pd.read_csv(p)
                    break
                except Exception:
                    continue

        if df is None:
            return

        text_col = "review_text" if "review_text" in df.columns else ("text" if "text" in df.columns else "reviewContent")

        for _, r in df.iterrows():
            rid = str(r["review_id"])
            raw_text = str(r.get(text_col, ""))
            score_val = r.get("model_score")
            model_score = float(score_val) if pd.notna(score_val) else 0.0
            rating_val = r.get("rating")
            rating = float(rating_val) if pd.notna(rating_val) else 0.0

            self._reviews_cache[rid] = {
                "review_id": rid,
                "item_id": rid,
                "item_type": "Review",
                "product_id": str(r.get("product_id", "")),
                "reviewer_id": str(r.get("reviewer_id", "")),
                "title": f"Review {rid} for {r.get('product_id', '')}",
                "snippet": raw_text[:200] if len(raw_text) > 200 else raw_text,
                "text": raw_text,
                "rating": rating,
                "model_score": model_score,
                "predicted_class": "fake" if str(r.get("label", "")).strip() in ("1", "1.0", "True", "CG", "fake") else "genuine",
            }

        # Populate SQLite FTS table
        try:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(self.fts_db_path)), exist_ok=True)
            conn = sqlite3.connect(self.fts_db_path)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS reviews_fts
                USING fts5(
                    review_id UNINDEXED,
                    product_id UNINDEXED,
                    reviewer_id UNINDEXED,
                    text,
                    rating UNINDEXED,
                    date UNINDEXED,
                    model_score UNINDEXED,
                    predicted_class UNINDEXED,
                    tokenize='porter ascii'
                )
            """)
            count_row = conn.execute("SELECT count(*) FROM reviews_fts").fetchone()
            if not count_row or count_row[0] == 0:
                rows = [
                    (
                        item["review_id"],
                        item["product_id"],
                        item["reviewer_id"],
                        item["text"],
                        str(item["rating"]),
                        str(r.get("timestamp", "")),
                        str(item["model_score"]),
                        item["predicted_class"],
                    )
                    for item in self._reviews_cache.values()
                ]
                conn.executemany("INSERT INTO reviews_fts VALUES (?,?,?,?,?,?,?,?)", rows)
                conn.commit()
            conn.close()
            self._fts_ready = True
        except Exception as exc:
            logger.warning("FTS auto-initialisation failed: %s", exc)

    # ------------------------------------------------------------------
    # Public search methods
    # ------------------------------------------------------------------

    async def keyword_search(
        self,
        query: str,
        limit: int = 20,
        page: int = 1,
    ) -> Dict:
        """Search reviews using SQLite FTS5 with in-memory substring fallback."""
        t0 = time.perf_counter()
        limit = max(1, min(limit, 200))
        page = max(1, page)
        offset = (page - 1) * limit
        self._ensure_fts_and_cache()

        results: List[Dict] = []
        total = 0

        def _run_fts() -> Dict:
            conn = sqlite3.connect(self.fts_db_path)
            conn.row_factory = sqlite3.Row
            try:
                words = [w.strip().replace('"', '') for w in query.split() if len(w.strip()) > 1]
                if not words:
                    words = [query.strip().replace('"', '')]
                fts_query = " OR ".join(f'"{w}"' for w in words if w)
                if not fts_query:
                    fts_query = '""'

                count_row = conn.execute(
                    "SELECT count(*) AS total FROM reviews_fts WHERE reviews_fts MATCH ?",
                    (fts_query,),
                ).fetchone()
                tot = count_row["total"] if count_row else 0

                rows = conn.execute(
                    """
                    SELECT review_id, product_id, reviewer_id,
                           snippet(reviews_fts, 3, '<b>', '</b>', '…', 20) AS snippet,
                           bm25(reviews_fts)                                AS bm25_score,
                           model_score, predicted_class
                    FROM reviews_fts
                    WHERE reviews_fts MATCH ?
                    ORDER BY bm25(reviews_fts)
                    LIMIT ? OFFSET ?
                    """,
                    (fts_query, limit, offset),
                ).fetchall()

                res = []
                for row in rows:
                    raw_bm25 = abs(float(row["bm25_score"])) if row["bm25_score"] else 0.0
                    # Map unbounded FTS5 BM25 score monotonically to [0.0, 1.0]
                    score = round(raw_bm25 / (1.0 + raw_bm25), 4) if raw_bm25 > 0 else 0.0
                    rid = row["review_id"]
                    cached = self._reviews_cache.get(rid, {})
                    res.append(
                        {
                            "item_id": rid,
                            "review_id": rid,
                            "product_id": row["product_id"],
                            "reviewer_id": row["reviewer_id"],
                            "title": cached.get("title", f"Review {rid}"),
                            "snippet": row["snippet"] or cached.get("snippet", ""),
                            "keyword_score": score,
                            "score": score,
                            "model_score": float(row["model_score"] or 0.0),
                            "predicted_class": row["predicted_class"],
                            "item_type": "Review",
                        }
                    )
                return {"results": res, "total": tot}
            finally:
                conn.close()

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, _run_fts)
            results = data["results"]
            total = data["total"]
        except Exception:
            results = []
            total = 0

        # Fallback to in-memory substring match if FTS returns no matches
        if not results and self._reviews_cache:
            q_lower = query.lower()
            matched = []
            for item in self._reviews_cache.values():
                text_match = q_lower in item["text"].lower()
                pid_match = q_lower in item["product_id"].lower()
                if text_match or pid_match:
                    item_copy = dict(item)
                    item_copy["keyword_score"] = 1.0 if text_match else 0.5
                    item_copy["score"] = item_copy["keyword_score"]
                    matched.append(item_copy)
            total = len(matched)
            results = matched[offset: offset + limit]

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "results": results,
            "items": results,
            "total": total,
            "page": page,
            "limit": limit,
            "latency_ms": round(latency_ms, 2),
        }

    async def semantic_search(self, query: str, limit: int = 20) -> Dict:
        """Search using dense vector embeddings via embedding_service."""
        t0 = time.perf_counter()
        limit = max(1, min(limit, 200))
        self._ensure_fts_and_cache()
        results: List[Dict] = []

        try:
            if hasattr(self.embedding_service, "search"):
                # embedding_service.search is synchronous and takes k
                raw = self.embedding_service.search(query, k=limit)
                for item in raw:
                    rid = item.get("review_id", "")
                    cached = self._reviews_cache.get(rid, {})
                    score = float(item.get("score", 0.0))
                    results.append(
                        {
                            "item_id": rid,
                            "review_id": rid,
                            "product_id": cached.get("product_id", ""),
                            "reviewer_id": cached.get("reviewer_id", ""),
                            "title": cached.get("title", f"Review {rid}"),
                            "snippet": cached.get("snippet", ""),
                            "semantic_score": score,
                            "score": score,
                            "model_score": cached.get("model_score", 0.0),
                            "predicted_class": cached.get("predicted_class", "genuine"),
                            "item_type": "Review",
                        }
                    )
        except Exception as exc:
            logger.error("semantic_search error: %s", exc)

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "results": results,
            "items": results,
            "total": len(results),
            "latency_ms": round(latency_ms, 2),
        }

    async def graph_search(
        self,
        query: str,
        product_id: Optional[str] = None,
        limit: int = 20,
    ) -> Dict:
        """Search via graph traversal patterns in Neo4j."""
        t0 = time.perf_counter()
        limit = max(1, min(limit, 200))
        self._ensure_fts_and_cache()
        results: List[Dict] = []
        try:
            raw = await self.graph_service.search_reviews_graph(
                query, product_id=product_id, limit=limit
            )
            for item in raw:
                rid = item.get("review_id", "")
                cached = self._reviews_cache.get(rid, {})
                score = float(item.get("model_score") or 0.5)
                results.append(
                    {
                        "item_id": rid,
                        "review_id": rid,
                        "product_id": item.get("product_id") or cached.get("product_id", ""),
                        "reviewer_id": item.get("reviewer_id") or cached.get("reviewer_id", ""),
                        "title": cached.get("title", f"Review {rid}"),
                        "snippet": item.get("review_text") or cached.get("snippet", ""),
                        "graph_score": score,
                        "score": score,
                        "model_score": float(item.get("model_score") or cached.get("model_score", 0.0)),
                        "predicted_class": item.get("predicted_class") or cached.get("predicted_class", "genuine"),
                        "item_type": "Review",
                    }
                )
        except Exception as exc:
            logger.error("graph_search error: %s", exc)

        # Fallback to cache if graph returns no results
        if not results and self._reviews_cache:
            q_lower = query.lower()
            for item in list(self._reviews_cache.values())[:limit]:
                if q_lower in item["text"].lower():
                    results.append(dict(item, graph_score=0.5, score=0.5))

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "results": results,
            "items": results,
            "total": len(results),
            "latency_ms": round(latency_ms, 2),
        }

    async def hybrid_search(
        self,
        query: str,
        mode: str = "hybrid",
        limit: int = 20,
        page: int = 1,
        product_id: Optional[str] = None,
    ) -> Dict:
        """Run keyword, semantic, and/or graph search and fuse results.

        Weights for hybrid mode:
            keyword : 0.25
            semantic: 0.40
            graph   : 0.35

        Args:
            query: The search query.
            mode: One of 'keyword', 'semantic', 'graph', 'hybrid'.
            limit: Number of results to return.
            page: 1-based page (applied after fusion for keyword; others paginate on limit).
            product_id: Optional product scope.

        Returns:
            Unified SearchResponse dict.
        """
        t0 = time.perf_counter()
        mode = mode.lower()
        limit = max(1, min(limit, 200))
        page = max(1, page)

        WEIGHTS: Dict[str, float] = {
            "keyword": 0.25,
            "semantic": 0.40,
            "graph": 0.35,
        }

        # --- Collect results from each source ---
        keyword_results: List[Dict] = []
        semantic_results: List[Dict] = []
        graph_results: List[Dict] = []

        fetch_more = limit * 3  # fetch extra before dedup

        if mode in ("keyword", "hybrid"):
            kw = await self.keyword_search(query, limit=fetch_more, page=1)
            keyword_results = self._normalize_scores(kw["results"], "keyword_score")

        if mode in ("semantic", "hybrid"):
            sem = await self.semantic_search(query, limit=fetch_more)
            semantic_results = self._normalize_scores(sem["results"], "semantic_score")

        if mode in ("graph", "hybrid"):
            gr = await self.graph_search(query, product_id=product_id, limit=fetch_more)
            graph_results = self._normalize_scores(gr["results"], "graph_score")

        # --- Tag each result with source and weighted score ---
        def _tag(items: List[Dict], source: str, field: str, weight: float) -> List[Dict]:
            tagged = []
            for item in items:
                item = dict(item)
                norm = item.pop("_norm_score", 0.0)
                item["_sources"] = {source: norm * weight}
                item["_score"] = norm * weight
                tagged.append(item)
            return tagged

        all_results: List[Dict] = []
        if mode == "hybrid":
            all_results += _tag(keyword_results, "keyword", "keyword_score", WEIGHTS["keyword"])
            all_results += _tag(semantic_results, "semantic", "semantic_score", WEIGHTS["semantic"])
            all_results += _tag(graph_results, "graph", "graph_score", WEIGHTS["graph"])
        elif mode == "keyword":
            all_results += _tag(keyword_results, "keyword", "keyword_score", 1.0)
        elif mode == "semantic":
            all_results += _tag(semantic_results, "semantic", "semantic_score", 1.0)
        elif mode == "graph":
            all_results += _tag(graph_results, "graph", "graph_score", 1.0)

        # --- Deduplicate and rank ---
        fused = self._deduplicate_and_rank(all_results)

        # Pagination
        total = len(fused)
        offset = (page - 1) * limit
        page_results = fused[offset: offset + limit]

        # Clean internal fields for response
        for r in page_results:
            r.pop("_sources", None)
            r.pop("_score", None)

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "results": page_results,
            "total": total,
            "page": page,
            "limit": limit,
            "mode": mode,
            "query": query,
            "latency_ms": round(latency_ms, 2),
        }

    # ------------------------------------------------------------------
    # FTS initialisation
    # ------------------------------------------------------------------

    async def initialize_fts(self, reviews_df) -> None:
        """Populate the SQLite FTS5 index from a pandas DataFrame.

        Args:
            reviews_df: DataFrame with columns:
                review_id, product_id, reviewer_id, text,
                rating, date, model_score, predicted_class.
        """
        def _populate():
            conn = sqlite3.connect(self.fts_db_path)
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS reviews_fts
                    USING fts5(
                        review_id UNINDEXED,
                        product_id UNINDEXED,
                        reviewer_id UNINDEXED,
                        text,
                        rating UNINDEXED,
                        date UNINDEXED,
                        model_score UNINDEXED,
                        predicted_class UNINDEXED,
                        content='',
                        tokenize='porter ascii'
                    )
                """)
                # Clear existing data before re-inserting
                conn.execute("DELETE FROM reviews_fts")
                conn.commit()

                required_cols = [
                    "review_id", "product_id", "reviewer_id", "text",
                    "rating", "date", "model_score", "predicted_class",
                ]
                # Fill any missing columns with empty strings
                for col in required_cols:
                    if col not in reviews_df.columns:
                        reviews_df[col] = ""

                rows = reviews_df[required_cols].fillna("").values.tolist()
                conn.executemany(
                    """
                    INSERT INTO reviews_fts
                        (review_id, product_id, reviewer_id, text,
                         rating, date, model_score, predicted_class)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                conn.commit()
                logger.info("FTS index populated with %d rows.", len(rows))
            finally:
                conn.close()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _populate)
        self._fts_ready = True

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    async def get_stats(self) -> Dict:
        """Return service statistics.

        Returns:
            Dict with FTS row count, embedding and graph availability.
        """
        fts_count = 0

        def _count():
            try:
                conn = sqlite3.connect(self.fts_db_path)
                row = conn.execute(
                    "SELECT count(*) AS c FROM reviews_fts"
                ).fetchone()
                conn.close()
                return row[0] if row else 0
            except Exception:
                return 0

        loop = asyncio.get_event_loop()
        fts_count = await loop.run_in_executor(None, _count)

        graph_connected = False
        try:
            graph_connected = await self.graph_service.is_connected()
        except Exception:
            pass

        embedding_available = hasattr(self.embedding_service, "search")

        return {
            "fts_index_rows": fts_count,
            "fts_ready": self._fts_ready,
            "graph_connected": graph_connected,
            "embedding_available": embedding_available,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize_scores(self, results: List[Dict], score_field: str) -> List[Dict]:
        """Min-max normalise scores in the results list to [0, 1].

        Args:
            results: List of result dicts.
            score_field: Key of the score to normalise.

        Returns:
            Results with added '_norm_score' key.
        """
        if not results:
            return results
        scores = [float(r.get(score_field) or 0.0) for r in results]
        min_s = min(scores)
        max_s = max(scores)
        span = max_s - min_s if max_s != min_s else 1.0

        normalised = []
        for r, s in zip(results, scores):
            r = dict(r)
            r["_norm_score"] = (s - min_s) / span
            normalised.append(r)
        return normalised

    def _deduplicate_and_rank(self, all_results: List[Dict]) -> List[Dict]:
        """Merge duplicate results (by review_id) and sum their weighted scores.

        Args:
            all_results: Tagged result dicts from one or more sources.

        Returns:
            Deduplicated, ranked list sorted by descending combined score.
        """
        merged: Dict[str, Dict] = {}

        for item in all_results:
            rid = item.get("review_id") or item.get("id") or str(id(item))
            if rid not in merged:
                merged[rid] = dict(item)
                merged[rid]["_score"] = item.get("_score", 0.0)
                merged[rid]["_sources"] = dict(item.get("_sources", {}))
            else:
                # Accumulate score from additional sources
                merged[rid]["_score"] += item.get("_score", 0.0)
                merged[rid]["_sources"].update(item.get("_sources", {}))
                # Prefer the richer record (more non-empty fields)
                for k, v in item.items():
                    if k not in merged[rid] or (v and not merged[rid].get(k)):
                        merged[rid][k] = v

        ranked = sorted(merged.values(), key=lambda x: x.get("_score", 0.0), reverse=True)
        return ranked
