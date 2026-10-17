"""
graph_service.py
----------------
Async Neo4j graph service for ReviewGraph.
Handles all Cypher queries, node/relationship management, and investigation logic.
"""

from neo4j import AsyncGraphDatabase, AsyncSession
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


class GraphService:
    """Service layer for all Neo4j operations in ReviewGraph."""

    def __init__(self, uri: str, user: str, password: str):
        """Initialize the async Neo4j driver.

        Args:
            uri: Bolt/Neo4j URI (e.g. bolt://localhost:7687)
            user: Neo4j username
            password: Neo4j password
        """
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        logger.info("GraphService initialized with URI: %s", uri)

    async def close(self):
        """Close the async Neo4j driver."""
        await self._driver.close()
        logger.info("GraphService driver closed.")

    # ------------------------------------------------------------------
    # Schema helpers
    # ------------------------------------------------------------------

    async def create_constraints(self):
        """Create uniqueness constraints and indexes for all primary node types."""
        if not await self.is_connected():
            logger.warning("Neo4j database is unreachable. Skipping schema constraint creation.")
            return

        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product) REQUIRE p.product_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Reviewer) REQUIRE r.reviewer_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (rv:Review) REQUIRE rv.review_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (es:EvidenceSignal) REQUIRE es.signal_id IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (rv:Review) ON (rv.product_id)",
            "CREATE INDEX IF NOT EXISTS FOR (rv:Review) ON (rv.reviewer_id)",
            "CREATE INDEX IF NOT EXISTS FOR (rv:Review) ON (rv.model_score)",
            "CREATE INDEX IF NOT EXISTS FOR (es:EvidenceSignal) ON (es.signal_type)",
        ]
        async with self._driver.session() as session:
            for stmt in constraints + indexes:
                try:
                    await session.run(stmt)
                    logger.debug("Schema statement applied: %s", stmt[:80])
                except Exception as exc:
                    logger.warning("Schema statement failed (may already exist): %s", exc)
        logger.info("Graph constraints and indexes ensured.")

    # ------------------------------------------------------------------
    # Import helpers
    # ------------------------------------------------------------------

    async def import_products(self, products: List[Dict]):
        """MERGE product nodes to avoid duplicates.

        Args:
            products: List of dicts with keys: product_id, name, category, etc.
        """
        query = """
        UNWIND $products AS p
        MERGE (prod:Product {product_id: p.product_id})
        SET prod.name        = coalesce(p.name, prod.name),
            prod.category    = coalesce(p.category, prod.category),
            prod.brand       = coalesce(p.brand, prod.brand),
            prod.price       = coalesce(p.price, prod.price),
            prod.avg_rating  = coalesce(p.avg_rating, prod.avg_rating),
            prod.review_count= coalesce(p.review_count, prod.review_count)
        RETURN count(prod) AS merged
        """
        async with self._driver.session() as session:
            result = await session.run(query, products=products)
            record = await result.single()
            merged = record["merged"] if record else 0
            logger.info("import_products: merged %d products", merged)

    async def import_reviewers(self, reviewers: List[Dict]):
        """MERGE reviewer nodes.

        Args:
            reviewers: List of dicts with keys: reviewer_id, name, location, etc.
        """
        query = """
        UNWIND $reviewers AS r
        MERGE (rev:Reviewer {reviewer_id: r.reviewer_id})
        SET rev.name         = coalesce(r.name, rev.name),
            rev.location     = coalesce(r.location, rev.location),
            rev.review_count = coalesce(r.review_count, rev.review_count),
            rev.avg_rating   = coalesce(r.avg_rating, rev.avg_rating)
        RETURN count(rev) AS merged
        """
        async with self._driver.session() as session:
            result = await session.run(query, reviewers=reviewers)
            record = await result.single()
            merged = record["merged"] if record else 0
            logger.info("import_reviewers: merged %d reviewers", merged)

    async def import_reviews(self, reviews: List[Dict]):
        """MERGE review nodes and create WROTE / REVIEWS relationships.

        Args:
            reviews: List of dicts with keys: review_id, product_id, reviewer_id,
                     rating, text, model_score, predicted_class, date, etc.
        """
        query = """
        UNWIND $reviews AS rv
        MERGE (r:Review {review_id: rv.review_id})
        SET r.product_id       = coalesce(rv.product_id, r.product_id),
            r.reviewer_id      = coalesce(rv.reviewer_id, r.reviewer_id),
            r.rating           = coalesce(rv.rating, r.rating),
            r.text             = coalesce(rv.text, r.text),
            r.model_score      = coalesce(rv.model_score, r.model_score),
            r.predicted_class  = coalesce(rv.predicted_class, r.predicted_class),
            r.date             = coalesce(rv.date, r.date),
            r.verified_purchase= coalesce(rv.verified_purchase, r.verified_purchase),
            r.helpful_votes    = coalesce(rv.helpful_votes, r.helpful_votes)
        WITH r, rv
        MATCH (prod:Product {product_id: rv.product_id})
        MERGE (r)-[:REVIEWS]->(prod)
        WITH r, rv
        MATCH (reviewer:Reviewer {reviewer_id: rv.reviewer_id})
        MERGE (reviewer)-[:WROTE]->(r)
        RETURN count(r) AS merged
        """
        async with self._driver.session() as session:
            result = await session.run(query, reviews=reviews)
            record = await result.single()
            merged = record["merged"] if record else 0
            logger.info("import_reviews: merged %d reviews", merged)

    async def import_evidence_signals(self, signals: List[Dict]):
        """MERGE EvidenceSignal nodes and attach them to reviews via HAS_SIGNAL.

        Args:
            signals: List of dicts with keys: signal_id, review_id, signal_type,
                     value, confidence, description.
        """
        query = """
        UNWIND $signals AS s
        MERGE (es:EvidenceSignal {signal_id: s.signal_id})
        SET es.review_id   = coalesce(s.review_id, es.review_id),
            es.signal_type = coalesce(s.signal_type, es.signal_type),
            es.value       = coalesce(s.value, es.value),
            es.confidence  = coalesce(s.confidence, es.confidence),
            es.description = coalesce(s.description, es.description)
        WITH es, s
        MATCH (r:Review {review_id: s.review_id})
        MERGE (r)-[:HAS_SIGNAL]->(es)
        RETURN count(es) AS merged
        """
        async with self._driver.session() as session:
            result = await session.run(query, signals=signals)
            record = await result.single()
            merged = record["merged"] if record else 0
            logger.info("import_evidence_signals: merged %d signals", merged)

    async def create_similarity_relationships(
        self,
        pairs: List[Tuple[str, str, float, str]],
        threshold: float = 0.7,
    ):
        """Create SIMILAR_TO relationships between review pairs above the threshold.

        Args:
            pairs: List of (review_id_a, review_id_b, similarity_score, similarity_type)
            threshold: Minimum score required to create the relationship (default 0.7)
        """
        filtered = [
            {"a": a, "b": b, "score": score, "sim_type": sim_type}
            for a, b, score, sim_type in pairs
            if score >= threshold
        ]
        if not filtered:
            logger.info("create_similarity_relationships: no pairs above threshold %.2f", threshold)
            return

        query = """
        UNWIND $pairs AS pair
        MATCH (ra:Review {review_id: pair.a})
        MATCH (rb:Review {review_id: pair.b})
        MERGE (ra)-[s:SIMILAR_TO {similarity_type: pair.sim_type}]-(rb)
        SET s.score      = pair.score,
            s.updated_at = datetime()
        RETURN count(s) AS created
        """
        async with self._driver.session() as session:
            result = await session.run(query, pairs=filtered)
            record = await result.single()
            created = record["created"] if record else 0
            logger.info(
                "create_similarity_relationships: created/merged %d SIMILAR_TO relationships",
                created,
            )

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    async def get_product(self, product_id: str) -> Optional[Dict]:
        """Return a product node with computed review stats.

        Args:
            product_id: The unique product identifier.

        Returns:
            Dict with product properties + computed stats, or None if not found.
        """
        query = """
        MATCH (p:Product {product_id: $product_id})
        OPTIONAL MATCH (r:Review)-[:REVIEWS]->(p)
        WITH p,
             count(r)                                  AS review_count,
             avg(r.model_score)                        AS avg_model_score,
             avg(r.rating)                             AS avg_rating,
             sum(CASE WHEN r.predicted_class = 'fake' THEN 1 ELSE 0 END) AS fake_count,
             sum(CASE WHEN r.predicted_class = 'genuine' THEN 1 ELSE 0 END) AS genuine_count
        RETURN p {
            .*,
            computed_review_count: review_count,
            computed_avg_model_score: avg_model_score,
            computed_avg_rating: avg_rating,
            computed_fake_count: fake_count,
            computed_genuine_count: genuine_count
        } AS product
        """
        async with self._driver.session() as session:
            result = await session.run(query, product_id=product_id)
            record = await result.single()
            if record is None:
                return None
            return dict(record["product"])

    async def get_product_graph(
        self, product_id: str, depth: int = 2, max_nodes: int = 100
    ) -> Dict:
        """Return a Cytoscape.js-compatible graph for a product neighbourhood.

        Args:
            product_id: Root product to start traversal from.
            depth: Max hops from the product node (default 2).
            max_nodes: Hard cap on returned nodes to prevent runaway queries.

        Returns:
            {'nodes': [...], 'edges': [...]} with typed node/edge objects.
        """
        # Clamp inputs to safe ranges
        depth = max(1, min(depth, 4))
        max_nodes = max(10, min(max_nodes, 500))

        query = f"""
        MATCH path = (p:Product {{product_id: $product_id}})-[*1..{depth}]-(n)
        WITH p, nodes(path) AS path_nodes, relationships(path) AS path_rels
        UNWIND path_nodes AS node
        WITH collect(DISTINCT node) AS all_nodes, collect(DISTINCT path_rels) AS all_rel_lists, p
        UNWIND all_nodes AS n
        WITH n, all_rel_lists, p
        LIMIT $max_nodes
        WITH collect(n) AS collected_nodes, all_rel_lists
        UNWIND all_rel_lists AS rel_list
        UNWIND rel_list AS rel
        WITH collected_nodes, collect(DISTINCT rel) AS collected_rels
        RETURN collected_nodes AS nodes, collected_rels AS rels
        """
        nodes_out: List[Dict] = []
        edges_out: List[Dict] = []

        seen_node_ids: set = set()
        seen_edge_ids: set = set()

        async with self._driver.session() as session:
            result = await session.run(
                query, product_id=product_id, max_nodes=max_nodes
            )
            record = await result.single()
            if record is None:
                # Return just the product node if nothing found
                product = await self.get_product(product_id)
                if product:
                    nodes_out.append(
                        {
                            "id": product_id,
                            "label": product.get("name", product_id),
                            "type": "Product",
                            "properties": product,
                        }
                    )
                return {"nodes": nodes_out, "edges": edges_out}

            raw_nodes = record["nodes"]
            raw_rels = record["rels"]

        # Build nodes
        for node in raw_nodes:
            node_id = None
            labels = list(node.labels) if hasattr(node, "labels") else []
            props = dict(node)

            # Determine canonical ID field
            for id_field in ("product_id", "review_id", "reviewer_id", "signal_id", "evidence_id"):
                if id_field in props:
                    node_id = props[id_field]
                    break
            if node_id is None:
                node_id = str(node.id)  # fallback to internal Neo4j ID

            if node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)

            node_type = labels[0] if labels else "Unknown"
            label = (
                props.get("name")
                or props.get("text", "")[:50]
                or props.get("signal_type")
                or node_id
            )
            nodes_out.append(
                {
                    "id": node_id,
                    "label": label,
                    "type": node_type,
                    "properties": props,
                }
            )

        # Build edges
        for rel in raw_rels:
            rel_id = str(rel.id)
            if rel_id in seen_edge_ids:
                continue
            seen_edge_ids.add(rel_id)

            src_props = dict(rel.start_node)
            tgt_props = dict(rel.end_node)

            def _extract_id(p: Dict) -> str:
                for f in ("product_id", "review_id", "reviewer_id", "signal_id", "evidence_id"):
                    if f in p:
                        return p[f]
                return str(rel.start_node.id if p is src_props else rel.end_node.id)

            source_id = _extract_id(src_props)
            target_id = _extract_id(tgt_props)

            edges_out.append(
                {
                    "id": rel_id,
                    "source": source_id,
                    "target": target_id,
                    "type": rel.type,
                    "properties": dict(rel),
                }
            )

        return {"nodes": nodes_out, "edges": edges_out}

    async def investigate_product(self, product_id: str) -> Dict:
        """Run a full investigation of a product's review graph.

        Computes:
        - All reviews and their signals
        - Similar review pairs (SIMILAR_TO)
        - Connected reviewers and their other reviewed products
        - A suspiciousness_score with breakdown

        Args:
            product_id: Target product identifier.

        Returns:
            Structured investigation dict.
        """
        product = await self.get_product(product_id)
        if product is None:
            return {"error": f"Product {product_id!r} not found"}

        # 1. Fetch all reviews for this product
        reviews_query = """
        MATCH (r:Review)-[:REVIEWS]->(p:Product {product_id: $product_id})
        OPTIONAL MATCH (reviewer:Reviewer)-[:WROTE]->(r)
        RETURN r {
            .*,
            reviewer_name: reviewer.name,
            reviewer_id: reviewer.reviewer_id
        } AS review
        ORDER BY r.model_score DESC
        LIMIT 500
        """
        # 2. Fetch all signals for those reviews
        signals_query = """
        MATCH (r:Review)-[:REVIEWS]->(p:Product {product_id: $product_id})
        MATCH (r)-[:HAS_SIGNAL]->(es:EvidenceSignal)
        RETURN es { .*, review_id: r.review_id } AS signal
        LIMIT 1000
        """
        # 3. Fetch similar review pairs
        similar_query = """
        MATCH (r:Review)-[:REVIEWS]->(p:Product {product_id: $product_id})
        MATCH (r)-[s:SIMILAR_TO]-(r2:Review)
        RETURN r.review_id AS review_a,
               r2.review_id AS review_b,
               s.score AS score,
               s.similarity_type AS sim_type
        LIMIT 200
        """
        # 4. Reviewer overlap – reviewers who reviewed OTHER products
        reviewer_overlap_query = """
        MATCH (reviewer:Reviewer)-[:WROTE]->(r:Review)-[:REVIEWS]->(p:Product {product_id: $product_id})
        MATCH (reviewer)-[:WROTE]->(r2:Review)-[:REVIEWS]->(p2:Product)
        WHERE p2.product_id <> $product_id
        RETURN DISTINCT
            reviewer.reviewer_id AS reviewer_id,
            reviewer.name AS reviewer_name,
            p2.product_id AS other_product_id,
            p2.name AS other_product_name
        LIMIT 200
        """

        async with self._driver.session() as session:
            reviews: List[Dict] = []
            res = await session.run(reviews_query, product_id=product_id)
            async for rec in res:
                reviews.append(dict(rec["review"]))

            signals: List[Dict] = []
            res = await session.run(signals_query, product_id=product_id)
            async for rec in res:
                signals.append(dict(rec["signal"]))

            similar_pairs: List[Dict] = []
            res = await session.run(similar_query, product_id=product_id)
            async for rec in res:
                similar_pairs.append(dict(rec))

            reviewer_overlaps: List[Dict] = []
            res = await session.run(reviewer_overlap_query, product_id=product_id)
            async for rec in res:
                reviewer_overlaps.append(dict(rec))

        # --- Compute suspiciousness score ---
        total_reviews = len(reviews)

        avg_model_score: float = 0.0
        if reviews:
            scores = [r.get("model_score") or 0.0 for r in reviews]
            avg_model_score = sum(scores) / len(scores)

        suspicious_signals = [
            s for s in signals if str(s.get("signal_type", "")).lower() in
            {"duplicate_text", "burst_reviews", "rating_outlier", "unverified_purchase",
             "short_text", "high_similarity", "reviewer_collusion"}
        ]
        suspicious_signal_ratio: float = (
            len(suspicious_signals) / len(signals) if signals else 0.0
        )

        unique_reviewers_on_product = {r.get("reviewer_id") for r in reviews if r.get("reviewer_id")}
        overlap_reviewers = {o["reviewer_id"] for o in reviewer_overlaps}
        reviewer_overlap_ratio: float = (
            len(overlap_reviewers) / len(unique_reviewers_on_product)
            if unique_reviewers_on_product else 0.0
        )

        text_similarity_ratio: float = (
            len(similar_pairs) / total_reviews if total_reviews else 0.0
        )
        # Clamp to [0, 1]
        text_similarity_ratio = min(text_similarity_ratio, 1.0)

        suspiciousness_score = (
            avg_model_score * 0.4
            + suspicious_signal_ratio * 0.3
            + reviewer_overlap_ratio * 0.2
            + text_similarity_ratio * 0.1
        )
        suspiciousness_score = round(min(max(suspiciousness_score, 0.0), 1.0), 4)

        if suspiciousness_score < 0.25:
            level = "low"
        elif suspiciousness_score < 0.5:
            level = "medium"
        elif suspiciousness_score < 0.75:
            level = "high"
        else:
            level = "critical"

        # --- Build readable graph paths ---
        graph_paths: List[str] = []
        for pair in similar_pairs[:20]:  # limit paths in response
            path = (
                f"Product [{product_id}] → review [{pair['review_a']}] "
                f"→ SIMILAR_TO → review [{pair['review_b']}] "
                f"→ REVIEWS → Product [{product_id}]"
            )
            graph_paths.append(path)

        for overlap in reviewer_overlaps[:10]:
            path = (
                f"Product [{product_id}] ← REVIEWS ← review "
                f"← WROTE ← Reviewer [{overlap['reviewer_id']}] "
                f"→ WROTE → review → REVIEWS → Product [{overlap['other_product_id']}]"
            )
            graph_paths.append(path)

        connected_products = list(
            {o["other_product_id"]: o for o in reviewer_overlaps}.values()
        )

        return {
            "product": product,
            "product_id": product_id,
            "total_reviews": total_reviews,
            "reviews": reviews,
            "signals": signals,
            "similar_pairs": similar_pairs,
            "reviewer_overlaps": reviewer_overlaps,
            "connected_products": connected_products,
            "suspiciousness": {
                "score": suspiciousness_score,
                "level": level,
                "breakdown": {
                    "avg_model_score": round(avg_model_score, 4),
                    "suspicious_signal_ratio": round(suspicious_signal_ratio, 4),
                    "reviewer_overlap_ratio": round(reviewer_overlap_ratio, 4),
                    "text_similarity_ratio": round(text_similarity_ratio, 4),
                },
            },
            "graph_paths": graph_paths,
        }

    async def search_reviews_graph(
        self,
        query: str,
        product_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Full-text style search over Review nodes in the graph.

        Args:
            query: Search term(s) to match against review text.
            product_id: Optional product filter.
            limit: Maximum number of results.

        Returns:
            List of matching review dicts with product info.
        """
        limit = max(1, min(limit, 200))
        words = [w.strip().lower() for w in query.split() if len(w.strip()) > 1]
        if not words:
            words = [query.strip().lower()]

        if product_id:
            cypher = """
            MATCH (r:Review)-[:REVIEWS]->(p:Product {product_id: $product_id})
            WHERE toLower(coalesce(r.review_text, '')) CONTAINS toLower($term)
               OR toLower(coalesce(p.product_id, '')) CONTAINS toLower($term)
               OR ANY(w IN $words WHERE toLower(coalesce(r.review_text, '')) CONTAINS w)
            OPTIONAL MATCH (reviewer:Reviewer)-[:WROTE]->(r)
            RETURN r {
                .*,
                product_name: p.name,
                product_id: p.product_id,
                reviewer_id: reviewer.reviewer_id,
                reviewer_name: reviewer.name
            } AS review
            ORDER BY r.model_score DESC
            LIMIT $limit
            """
            params = {"product_id": product_id, "term": query, "words": words, "limit": limit}
        else:
            cypher = """
            MATCH (r:Review)-[:REVIEWS]->(p:Product)
            WHERE toLower(coalesce(r.review_text, '')) CONTAINS toLower($term)
               OR toLower(coalesce(p.product_id, '')) CONTAINS toLower($term)
               OR ANY(w IN $words WHERE toLower(coalesce(r.review_text, '')) CONTAINS w)
            OPTIONAL MATCH (reviewer:Reviewer)-[:WROTE]->(r)
            RETURN r {
                .*,
                product_name: p.name,
                product_id: p.product_id,
                reviewer_id: reviewer.reviewer_id,
                reviewer_name: reviewer.name
            } AS review
            ORDER BY r.model_score DESC
            LIMIT $limit
            """
            params = {"term": query, "words": words, "limit": limit}

        results: List[Dict] = []
        async with self._driver.session() as session:
            res = await session.run(cypher, parameters=params)
            async for rec in res:
                results.append(dict(rec["review"]))
        return results

    async def get_review(self, review_id: str) -> Optional[Dict]:
        """Return a single review node with all connected information.

        Args:
            review_id: Unique review identifier.

        Returns:
            Dict with review properties, or None.
        """
        cypher = """
        MATCH (r:Review {review_id: $review_id})
        OPTIONAL MATCH (reviewer:Reviewer)-[:WROTE]->(r)
        OPTIONAL MATCH (r)-[:REVIEWS]->(p:Product)
        OPTIONAL MATCH (r)-[:HAS_SIGNAL]->(es:EvidenceSignal)
        WITH r, reviewer, p, collect(es {.*}) AS signals
        RETURN r {
            .*,
            reviewer_id: reviewer.reviewer_id,
            reviewer_name: reviewer.name,
            product_id: p.product_id,
            product_name: p.name,
            signals: signals
        } AS review
        """
        async with self._driver.session() as session:
            result = await session.run(cypher, review_id=review_id)
            record = await result.single()
            if record is None:
                return None
            return dict(record["review"])

    async def get_related_reviews(
        self, review_id: str, limit: int = 10
    ) -> List[Dict]:
        """Return reviews related via SIMILAR_TO relationships.

        Args:
            review_id: Source review identifier.
            limit: Max number of related reviews.

        Returns:
            List of related review dicts with similarity score.
        """
        limit = max(1, min(limit, 100))
        cypher = """
        MATCH (r:Review {review_id: $review_id})-[s:SIMILAR_TO]-(r2:Review)
        OPTIONAL MATCH (r2)-[:REVIEWS]->(p:Product)
        RETURN r2 {
            .*,
            product_id: p.product_id,
            product_name: p.name,
            similarity_score: s.score,
            similarity_type: s.similarity_type
        } AS related
        ORDER BY s.score DESC
        LIMIT $limit
        """
        results: List[Dict] = []
        async with self._driver.session() as session:
            res = await session.run(cypher, review_id=review_id, limit=limit)
            async for rec in res:
                results.append(dict(rec["related"]))
        return results

    async def get_stats(self) -> Dict:
        """Return counts of all node and relationship types.

        Returns:
            Dict with node and relationship counts.
        """
        cypher = """
        CALL apoc.meta.stats()
        YIELD labels, relTypesCount
        RETURN labels, relTypesCount
        """
        # Fallback if APOC is not available
        fallback_cypher = """
        RETURN
            [(p:Product) | p] AS products,
            [(r:Review) | r] AS reviews,
            [(rv:Reviewer) | rv] AS reviewers,
            [(es:EvidenceSignal) | es] AS signals
        """
        counts_cypher = """
        OPTIONAL MATCH (p:Product) WITH count(DISTINCT p) AS products
        OPTIONAL MATCH (r:Review) WITH products, count(DISTINCT r) AS reviews
        OPTIONAL MATCH (rv:Reviewer) WITH products, reviews, count(DISTINCT rv) AS reviewers
        OPTIONAL MATCH (es) WHERE es:EvidenceSignal OR es:Evidence WITH products, reviews, reviewers, count(DISTINCT es) AS signals
        OPTIONAL MATCH ()-[rel]->() WITH products, reviews, reviewers, signals, count(DISTINCT rel) AS total_rels
        OPTIONAL MATCH ()-[rel_w:WROTE]->() WITH products, reviews, reviewers, signals, total_rels, count(DISTINCT rel_w) AS wrote_rels
        OPTIONAL MATCH ()-[rel_rev:REVIEWS]->() WITH products, reviews, reviewers, signals, total_rels, wrote_rels, count(DISTINCT rel_rev) AS reviews_rels
        OPTIONAL MATCH ()-[rel_s]->() WHERE type(rel_s) IN ['HAS_SIGNAL', 'HAS_EVIDENCE'] WITH products, reviews, reviewers, signals, total_rels, wrote_rels, reviews_rels, count(DISTINCT rel_s) AS signal_rels
        OPTIONAL MATCH ()-[rel_sim:SIMILAR_TO]-() WITH products, reviews, reviewers, signals, total_rels, wrote_rels, reviews_rels, signal_rels, count(DISTINCT rel_sim)/2 AS similar_rels
        RETURN products, reviews, reviewers, signals,
               total_rels, wrote_rels, reviews_rels, signal_rels, similar_rels
        """
        async with self._driver.session() as session:
            try:
                res = await session.run(counts_cypher)
                record = await res.single()
                if record:
                    return {
                        "nodes": {
                            "Product": record["products"],
                            "Review": record["reviews"],
                            "Reviewer": record["reviewers"],
                            "EvidenceSignal": record["signals"],
                        },
                        "relationships": {
                            "total": record["total_rels"],
                            "WROTE": record["wrote_rels"],
                            "REVIEWS": record["reviews_rels"],
                            "HAS_SIGNAL": record["signal_rels"],
                            "SIMILAR_TO": record["similar_rels"],
                        },
                    }
            except Exception as exc:
                logger.warning("get_stats query failed: %s", exc)
        return {"nodes": {}, "relationships": {}}

    async def is_connected(self) -> bool:
        """Verify the Neo4j driver can reach the database.

        Returns:
            True if a simple query succeeds, False otherwise.
        """
        try:
            async with self._driver.session() as session:
                result = await session.run("RETURN 1 AS ok")
                record = await result.single()
                return record is not None and record["ok"] == 1
        except Exception as exc:
            logger.error("is_connected check failed: %s", exc)
            return False

    async def run_parameterized(self, query: str, params: Dict) -> List[Dict]:
        """Execute an arbitrary parameterized Cypher query.

        Args:
            query: Cypher query string with $param placeholders.
            params: Parameter dict matching placeholders in the query.

        Returns:
            List of result record dicts.
        """
        results: List[Dict] = []
        async with self._driver.session() as session:
            res = await session.run(query, **params)
            async for rec in res:
                results.append(dict(rec))
        return results
