"""
explanation_service.py
-----------------------
Template-driven, non-hallucinating explanation service for ReviewGraph.
All explanations reference only entities present in the supplied investigation data.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Standard limitations that accompany every explanation
_STANDARD_LIMITATIONS = [
    "Similar text does not prove coordinated fraud.",
    "Model score is not a calibrated probability; treat it as a risk indicator only.",
    "Reviewer connections may be coincidental and do not imply collusion.",
    "Dataset labels may not reflect verified fraud and could contain annotation errors.",
    "Graph paths show structural patterns, not causal relationships.",
    "Absence of suspicious signals does not guarantee review authenticity.",
]


class ExplanationService:
    """Generates structured, evidence-grounded explanations for review investigations."""

    def __init__(self, graph_service):
        """Initialise with a GraphService instance.

        Args:
            graph_service: An initialised GraphService for optional supplemental queries.
        """
        self.graph_service = graph_service
        logger.info("ExplanationService initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_product_explanation(self, investigation_data: Dict) -> Dict:
        """Build a structured, verifiable explanation from raw investigation data.

        Args:
            investigation_data: Output of GraphService.investigate_product().

        Returns:
            Explanation dict with observed_facts, computed_signals, model_predictions,
            graph_paths, source_records, explanation_summary, limitations, and
            alternative_interpretations.
        """
        product = investigation_data.get("product") or {}
        product_id = investigation_data.get("product_id") or product.get("product_id", "UNKNOWN")
        reviews: List[Dict] = investigation_data.get("reviews", [])
        signals: List[Dict] = investigation_data.get("signals", [])
        similar_pairs: List[Dict] = investigation_data.get("similar_pairs", [])
        reviewer_overlaps: List[Dict] = investigation_data.get("reviewer_overlaps", [])
        connected_products: List[Dict] = investigation_data.get("connected_products", [])
        suspiciousness: Dict = investigation_data.get("suspiciousness", {})
        raw_paths: List[str] = investigation_data.get("graph_paths", [])

        # --- Investigated entity ---
        investigated_entity = {
            "type": "Product",
            "id": product_id,
            "name": product.get("name", ""),
            "category": product.get("category", ""),
            "total_reviews": len(reviews),
        }

        # --- Observed facts (from actual DB records) ---
        observed_facts = self._build_observed_facts(
            product, reviews, signals, similar_pairs, reviewer_overlaps
        )

        # --- Computed signals (from EvidenceSignal nodes) ---
        computed_signals = self._build_computed_signals(signals)

        # --- Model predictions ---
        model_predictions = self._build_model_predictions(reviews)

        # --- Graph paths ---
        graph_paths = self._build_graph_path_explanations(
            raw_paths, reviews, similar_pairs, reviewer_overlaps
        )

        # --- Source records ---
        source_records = list({r["review_id"] for r in reviews if r.get("review_id")})
        source_record_details = [
            {"review_id": r["review_id"], "source": "Review node"}
            for r in reviews if r.get("review_id")
        ]
        source_record_details.extend(
            {
                "review_id": p.get("review_a"),
                "related_review_id": p.get("review_b"),
                "relationship": "SIMILAR_TO",
                "score": p.get("score"),
            }
            for p in similar_pairs
            if p.get("review_a") and p.get("review_b")
        )

        # --- Reasoning chain ---
        reasoning_chain = self._build_reasoning_chain(
            product_id, reviews, similar_pairs, connected_products
        )

        # --- Summary ---
        explanation_summary = self._build_summary(
            product_id, product, suspiciousness, reviews, signals,
            similar_pairs, reviewer_overlaps
        )

        # --- Limitations ---
        limitations = self._generate_limitations(investigation_data)

        # --- Alternative interpretations ---
        alternative_interpretations = self._build_alternative_interpretations(
            suspiciousness, reviews, signals
        )

        explanation = {
            "investigated_entity": investigated_entity,
            "observed_facts": observed_facts,
            "computed_signals": computed_signals,
            "model_predictions": model_predictions,
            "reasoning_chain": reasoning_chain,
            "graph_paths": graph_paths,
            "source_records": source_records,
            "source_record_details": source_record_details,
            "explanation_summary": explanation_summary,
            "suspiciousness": suspiciousness,
            "limitations": limitations,
            "alternative_interpretations": alternative_interpretations,
        }

        validity = self._compute_explanation_validity(explanation)
        explanation["validity"] = validity

        return explanation

    async def generate_review_explanation(self, review_data: Dict) -> Dict:
        """Build an explanation for a single review node.

        Args:
            review_data: Output of GraphService.get_review() with optional
                         signals and related reviews included.

        Returns:
            Structured explanation dict for the review.
        """
        review_id = review_data.get("review_id", "UNKNOWN")
        product_id = review_data.get("product_id", "UNKNOWN")
        signals: List[Dict] = review_data.get("signals", [])

        model_score = review_data.get("model_score")
        predicted_class = review_data.get("predicted_class", "unknown")

        observed_facts: List[str] = []
        observed_facts.append(
            f"Review [{review_id}] was written for product [{product_id}]."
        )
        if review_data.get("reviewer_id"):
            observed_facts.append(
                f"Reviewer ID: {review_data['reviewer_id']}"
                + (f" ({review_data['reviewer_name']})" if review_data.get("reviewer_name") else "")
                + "."
            )
        if review_data.get("rating") is not None:
            observed_facts.append(f"Star rating: {review_data['rating']} out of 5.")
        if review_data.get("verified_purchase") is not None:
            vp = review_data["verified_purchase"]
            observed_facts.append(
                "Marked as a verified purchase." if vp else "NOT marked as a verified purchase."
            )
        if review_data.get("helpful_votes") is not None:
            observed_facts.append(f"Received {review_data['helpful_votes']} helpful votes.")
        if review_data.get("date"):
            observed_facts.append(f"Posted on: {review_data['date']}.")
        if review_data.get("text"):
            word_count = len(str(review_data["text"]).split())
            observed_facts.append(f"Review text contains approximately {word_count} words.")

        computed_signals = self._build_computed_signals(signals)

        model_predictions: List[Dict] = []
        if model_score is not None:
            model_predictions.append(
                {
                    "review_id": review_id,
                    "model_score": round(float(model_score), 4),
                    "predicted_class": predicted_class,
                    "warning": (
                        "Model score is a classifier output, not a calibrated probability."
                    ),
                }
            )

        # Summary
        if model_score is not None:
            score_pct = round(float(model_score) * 100, 1)
            summary = (
                f"Review [{review_id}] for product [{product_id}] received a model "
                f"suspiciousness score of {score_pct}% (predicted class: {predicted_class}). "
            )
        else:
            summary = f"Review [{review_id}] for product [{product_id}] has no model score recorded. "

        if signals:
            signal_types = list({s.get("signal_type", "unknown") for s in signals})
            summary += f"{len(signals)} evidence signal(s) detected: {', '.join(signal_types)}. "
        else:
            summary += "No evidence signals detected for this review. "

        limitations = self._generate_limitations({"reviews": [review_data], "signals": signals})

        return {
            "investigated_entity": {
                "type": "Review",
                "id": review_id,
                "product_id": product_id,
                "reviewer_id": review_data.get("reviewer_id"),
            },
            "observed_facts": observed_facts,
            "computed_signals": computed_signals,
            "model_predictions": model_predictions,
            "graph_paths": [],
            "source_records": [review_id],
            "explanation_summary": summary.strip(),
            "limitations": limitations,
            "alternative_interpretations": self._build_alternative_interpretations(
                {}, [review_data], signals
            ),
            "validity": self._compute_explanation_validity(
                {
                    "source_records": [review_id],
                    "graph_paths": [],
                    "investigated_entity": {"id": review_id},
                }
            ),
        }

    # ------------------------------------------------------------------
    # Private construction helpers
    # ------------------------------------------------------------------

    def _build_observed_facts(
        self,
        product: Dict,
        reviews: List[Dict],
        signals: List[Dict],
        similar_pairs: List[Dict],
        reviewer_overlaps: List[Dict],
    ) -> List[str]:
        """Build plain-English statements from actual database records only."""
        facts: List[str] = []
        pid = product.get("product_id", "")

        if product.get("name"):
            facts.append(f"Product name recorded in database: \"{product['name']}\".")
        if product.get("category"):
            facts.append(f"Product category: {product['category']}.")

        total = len(reviews)
        facts.append(f"Total reviews retrieved from graph for product [{pid}]: {total}.")

        if reviews:
            with_score = [r for r in reviews if r.get("model_score") is not None]
            if with_score:
                avg_s = sum(float(r["model_score"]) for r in with_score) / len(with_score)
                facts.append(
                    f"{len(with_score)} of {total} reviews have model scores; "
                    f"average score: {avg_s:.3f}."
                )
            fake_count = sum(
                1 for r in reviews if str(r.get("predicted_class", "")).lower() == "fake"
            )
            genuine_count = sum(
                1 for r in reviews if str(r.get("predicted_class", "")).lower() == "genuine"
            )
            verified = sum(1 for r in reviews if r.get("verified_purchase"))
            facts.append(f"{verified} of {total} reviews are marked as verified purchases.")

        if signals:
            signal_types = {}
            for s in signals:
                st = s.get("signal_type", "unknown")
                signal_types[st] = signal_types.get(st, 0) + 1
            type_summary = "; ".join(f"{t}: {c}" for t, c in signal_types.items())
            facts.append(
                f"{len(signals)} EvidenceSignal node(s) found in graph: {type_summary}."
            )
        else:
            facts.append("No EvidenceSignal nodes linked to this product's reviews.")

        if similar_pairs:
            facts.append(
                f"{len(similar_pairs)} SIMILAR_TO relationship(s) found among reviews of this product."
            )
        else:
            facts.append("No SIMILAR_TO relationships found among reviews of this product.")

        if reviewer_overlaps:
            unique_reviewers = len({o["reviewer_id"] for o in reviewer_overlaps})
            unique_other_products = len({o["other_product_id"] for o in reviewer_overlaps})
            facts.append(
                f"{unique_reviewers} reviewer(s) also reviewed {unique_other_products} "
                f"other product(s) in the dataset."
            )
        else:
            facts.append("No reviewer overlap with other products detected.")

        return facts

    def _build_computed_signals(self, signals: List[Dict]) -> List[Dict]:
        """Format EvidenceSignal nodes as structured signal entries."""
        result = []
        for s in signals:
            entry = {
                "signal_id": s.get("signal_id"),
                "review_id": s.get("review_id"),
                "signal_type": s.get("signal_type", "unknown"),
                "value": s.get("value"),
                "confidence": s.get("confidence"),
                "description": s.get("description", ""),
                "source": "EvidenceSignal node in Neo4j graph",
            }
            result.append(entry)
        return result

    def _build_model_predictions(self, reviews: List[Dict]) -> List[Dict]:
        """Extract model predictions with attached warnings."""
        predictions = []
        for r in reviews:
            if r.get("model_score") is not None or r.get("predicted_class") in {"fake", "genuine"}:
                predictions.append(
                    {
                        "review_id": r.get("review_id"),
                        "model_score": (
                            round(float(r["model_score"]), 4) if r.get("model_score") is not None else None
                        ),
                        "predicted_class": r.get("predicted_class"),
                        "warning": (
                            "Model score is a classifier output, not a calibrated probability. "
                            "It should not be interpreted as a definitive fraud determination."
                        ),
                    }
                )
        return predictions

    def _build_graph_path_explanations(
        self,
        raw_paths: List[str],
        reviews: List[Dict],
        similar_pairs: List[Dict],
        reviewer_overlaps: List[Dict],
    ) -> List[Dict]:
        """Convert raw path strings to structured, verifiable path objects.

        Only paths whose entity IDs appear in the supplied data are included.
        """
        known_review_ids = {r.get("review_id") for r in reviews if r.get("review_id")}
        known_reviewer_ids = {o.get("reviewer_id") for o in reviewer_overlaps if o.get("reviewer_id")}
        sim_review_ids = set()
        for pair in similar_pairs:
            sim_review_ids.add(pair.get("review_a"))
            sim_review_ids.add(pair.get("review_b"))

        path_objects = []
        for path_str in raw_paths:
            # Basic validity: path text should reference at least one known ID
            referenced = [rid for rid in (known_review_ids | known_reviewer_ids) if rid and rid in path_str]
            if not referenced:
                # Skip paths that don't reference any known entity
                continue
            path_objects.append(
                {
                    "path_text": path_str,
                    "referenced_entity_ids": referenced,
                    "verified": True,
                    "basis": "Derived from SIMILAR_TO or reviewer-overlap graph traversal.",
                }
            )
        return path_objects

    def _build_reasoning_chain(
        self,
        product_id: str,
        reviews: List[Dict],
        similar_pairs: List[Dict],
        connected_products: List[Dict],
    ) -> List[Dict]:
        """Build a step-by-step reasoning chain from verified entities only.

        IMPORTANT: Only entities present in the supplied arguments are referenced.
        No fabrication occurs.

        Args:
            product_id: Product under investigation.
            reviews: List of review dicts.
            similar_pairs: List of SIMILAR_TO pair dicts.
            connected_products: List of products connected via shared reviewers.

        Returns:
            List of reasoning step dicts.
        """
        chain: List[Dict] = []
        step_num = 1

        # Step 1: Review count
        chain.append(
            {
                "step": step_num,
                "path_text": f"Product [{product_id}] has {len(reviews)} review(s) in the graph.",
                "entities": [product_id],
                "confidence": "high",
                "basis": f"Direct MATCH count from Neo4j: {len(reviews)} Review nodes linked via :REVIEWS.",
            }
        )
        step_num += 1

        # Step 2: High-scoring reviews
        high_score_reviews = [
            r for r in reviews if (r.get("model_score") or 0.0) > 0.7
        ]
        if high_score_reviews:
            ids = [r["review_id"] for r in high_score_reviews[:5] if r.get("review_id")]
            chain.append(
                {
                    "step": step_num,
                    "path_text": (
                        f"{len(high_score_reviews)} review(s) have model_score > 0.7, "
                        f"indicating elevated risk. Sample IDs: {ids}."
                    ),
                    "entities": ids,
                    "confidence": "medium",
                    "basis": "model_score field on Review nodes; threshold is 0.7.",
                }
            )
            step_num += 1

        # Step 3: Similar pairs
        if similar_pairs:
            sample = similar_pairs[:3]
            chain.append(
                {
                    "step": step_num,
                    "path_text": (
                        str(len(similar_pairs)) + " SIMILAR_TO relationship(s) found. Sample: "
                        + ", ".join(
                            str(p.get("review_a", "?")) + " <-> " + str(p.get("review_b", "?"))
                            + " (score=" + str(p.get("score", "?")) + ")"
                            for p in sample
                        )
                    ),
                    "entities": list(
                        {p["review_a"] for p in similar_pairs if p.get("review_a")}
                        | {p["review_b"] for p in similar_pairs if p.get("review_b")}
                    )[:10],
                    "confidence": "medium",
                    "basis": "SIMILAR_TO edges in graph with score stored on relationship.",
                }
            )
            step_num += 1

        # Step 4: Connected products
        if connected_products:
            cp_ids = [cp.get("other_product_id") for cp in connected_products[:5] if cp.get("other_product_id")]
            chain.append(
                {
                    "step": step_num,
                    "path_text": (
                        f"Reviewers of product [{product_id}] also reviewed "
                        f"{len(connected_products)} other product(s): {cp_ids}."
                    ),
                    "entities": [product_id] + cp_ids,
                    "confidence": "low",
                    "basis": "Reviewer-overlap traversal: Reviewer-[:WROTE]->Review-[:REVIEWS]->OtherProduct.",
                }
            )
            step_num += 1

        return chain

    def _generate_limitations(self, investigation_data: Dict) -> List[str]:
        """Return standard + data-specific limitations.

        Args:
            investigation_data: Investigation dict; used to add contextual caveats.

        Returns:
            List of limitation strings.
        """
        limitations = list(_STANDARD_LIMITATIONS)

        reviews = investigation_data.get("reviews", [])
        coverage = investigation_data.get("coverage", {})
        if any(item.get("truncated") for item in coverage.values() if isinstance(item, dict)):
            limitations.append(
                "This investigation reached one or more query limits; reported evidence is a capped subset."
            )
        if reviews:
            without_score = [r for r in reviews if r.get("model_score") is None]
            if without_score:
                limitations.append(
                    f"{len(without_score)} of {len(reviews)} review(s) lack model scores "
                    "and are excluded from score-based analysis."
                )
            if not any(r.get("model_score") is not None or r.get("predicted_class") for r in reviews):
                limitations.append(
                    "No classifier predictions are available for these reviews; suspiciousness is based on observed and derived graph signals only."
                )
            if any(r.get("reviewer_id", "").startswith("sha256:") for r in reviews):
                limitations.append("Reviewer identifiers are anonymized hashes and cannot identify people.")

        signals = investigation_data.get("signals", [])
        if not signals:
            limitations.append(
                "No EvidenceSignal nodes were found; evidence-signal analysis is unavailable."
            )

        return limitations

    def _build_alternative_interpretations(
        self,
        suspiciousness: Dict,
        reviews: List[Dict],
        signals: List[Dict],
    ) -> List[str]:
        """Generate plausible innocent explanations for observed patterns."""
        interps: List[str] = []
        score = suspiciousness.get("score", 0.0) if suspiciousness else 0.0

        interps.append(
            "Text similarity between reviews may reflect shared product features "
            "or common usage contexts rather than coordinated fraud."
        )
        interps.append(
            "Reviewers who review many products may be active shoppers, not bad actors."
        )
        interps.append(
            "Burst review patterns can coincide with product launches or promotions."
        )

        high_rating_reviews = [r for r in reviews if (r.get("rating") or 0) >= 4]
        if len(high_rating_reviews) > len(reviews) * 0.7:
            interps.append(
                "The preponderance of high ratings may reflect genuine customer satisfaction "
                "rather than rating manipulation."
            )

        verified = [r for r in reviews if r.get("verified_purchase")]
        if verified and len(verified) > len(reviews) * 0.5:
            interps.append(
                f"{len(verified)} of {len(reviews)} reviews are verified purchases, "
                "suggesting real transactions."
            )

        return interps

    def _build_summary(
        self,
        product_id: str,
        product: Dict,
        suspiciousness: Dict,
        reviews: List[Dict],
        signals: List[Dict],
        similar_pairs: List[Dict],
        reviewer_overlaps: List[Dict],
    ) -> str:
        """Construct a template-driven summary paragraph.

        All values are sourced directly from supplied data. No LLM inference.
        """
        name = product.get("name") or product_id
        score = suspiciousness.get("score", 0.0)
        level = suspiciousness.get("level", "unknown")
        breakdown = suspiciousness.get("breakdown", {})
        avg_model_score = breakdown.get("avg_model_score")

        total_reviews = len(reviews)
        total_signals = len(signals)
        total_similar = len(similar_pairs)
        total_overlaps = len({o.get("reviewer_id") for o in reviewer_overlaps if o.get("reviewer_id")})

        summary = (
            f"Investigation of product [{product_id}] (\"{name}\") based on "
            f"{total_reviews} review(s) retrieved from the knowledge graph. "
            f"Overall suspiciousness score: {score:.3f} ({level.upper()}). "
        )
        model_component = (
            f"avg model score: {avg_model_score:.3f} (weight 0.40); "
            if avg_model_score is not None
            else "avg model score: unavailable; "
        )
        summary += (
            f"Score components — {model_component}"
            f"suspicious signal ratio: {breakdown.get('suspicious_signal_ratio', 0):.3f} (weight 0.30); "
            f"reviewer overlap ratio: {breakdown.get('reviewer_overlap_ratio', 0):.3f} (weight 0.20); "
            f"text similarity ratio: {breakdown.get('text_similarity_ratio', 0):.3f} (weight 0.10). "
        )
        summary += (
            f"{total_signals} EvidenceSignal node(s) found; "
            f"{total_similar} SIMILAR_TO relationship(s) detected; "
            f"{total_overlaps} overlapping reviewer(s) found across other products. "
        )
        summary += (
            "This summary is generated from database records only. "
            "Refer to the limitations section before drawing conclusions."
        )
        return summary

    # ------------------------------------------------------------------
    # Validity check
    # ------------------------------------------------------------------

    def _compute_explanation_validity(self, explanation: Dict) -> Dict:
        """Verify that all cited entities in the explanation exist in source_records or graph_paths.

        Args:
            explanation: The explanation dict as built by generate_product_explanation.

        Returns:
            {'valid': bool, 'issues': List[str], 'coverage': float}
        """
        issues: List[str] = []
        source_records: List[str] = explanation.get("source_records", [])
        graph_paths: List = explanation.get("graph_paths", [])
        entity_id: str = (explanation.get("investigated_entity") or {}).get("id", "")

        # Check that source_records list is non-empty when reviews exist
        if not source_records and not entity_id:
            issues.append("No source records and no investigated entity ID present.")

        # Check graph path entities reference real source records
        path_entities_covered = 0
        path_entities_total = 0
        for path in graph_paths:
            if isinstance(path, dict):
                entities = path.get("referenced_entity_ids", []) or path.get("entities", [])
                for eid in entities:
                    path_entities_total += 1
                    # Entity should appear in source_records OR be the investigated entity
                    if eid in source_records or eid == entity_id:
                        path_entities_covered += 1
                    else:
                        issues.append(
                            f"Graph path references entity [{eid}] not found in source_records."
                        )

        coverage = (
            path_entities_covered / path_entities_total
            if path_entities_total > 0
            else 1.0  # No paths = trivially valid
        )

        valid = len(issues) == 0
        return {
            "valid": valid,
            "issues": issues,
            "coverage": round(coverage, 4),
        }
