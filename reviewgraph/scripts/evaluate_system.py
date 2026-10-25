"""
evaluate_system.py
------------------
Comprehensive evaluation of ReviewGraph system:
  - Classification metrics (loaded from saved evaluation_results.json)
  - Retrieval metrics: Recall@5, Recall@10, MRR, nDCG@10 for keyword/semantic/graph/hybrid
  - Explainability metrics: edge existence, source coverage, unsupported claim detection
  - Response latency benchmarks

Usage:
    python scripts/evaluate_system.py --use-sample
    python scripts/evaluate_system.py --processed-path ../data/processed
"""


from __future__ import annotations
import sys as _sys_utf8
if hasattr(_sys_utf8.stdout, 'reconfigure'): _sys_utf8.stdout.reconfigure(encoding='utf-8')

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_system")

# ─────────────────────────────────────────────────────────────────────────── #
# Retrieval Metric Helpers
# ─────────────────────────────────────────────────────────────────────────── #

def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Recall@K: fraction of relevant items found in top-K retrieved."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """MRR component: 1/rank of first relevant item, 0 if none in list."""
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """DCG@K using binary relevance."""
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        rel = 1 if rid in relevant_ids else 0
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """nDCG@K: DCG@K / ideal DCG@K."""
    actual = dcg_at_k(retrieved_ids, relevant_ids, k)
    ideal_retrieved = list(relevant_ids)[:k]
    ideal = dcg_at_k(ideal_retrieved, relevant_ids, k)
    return actual / ideal if ideal > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────── #
# Benchmark Query Set
# ─────────────────────────────────────────────────────────────────────────── #

def build_benchmark_queries(df: pd.DataFrame) -> List[Dict]:
    """
    Build a small benchmark from the processed dataset.
    For each query we pick reviews that contain the query keywords,
    treating them as 'relevant' (silver standard — not verified ground truth).
    Label this clearly in outputs.
    """
    queries = []
    sample_terms = [
        "best product", "terrible quality", "love this", "waste of money",
        "highly recommend", "broken", "excellent", "disappointing",
        "great value", "never again"
    ]

    text_col = None
    for col in ["review_text", "reviewContent", "text"]:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        logger.warning("No review text column found — using empty benchmark.")
        return []

    id_col = None
    for col in ["review_id", "id"]:
        if col in df.columns:
            id_col = col
            break

    if id_col is None:
        logger.warning("No review ID column found — using empty benchmark.")
        return []

    df_text = df[[id_col, text_col]].dropna()
    df_text[text_col] = df_text[text_col].astype(str).str.lower()

    for term in sample_terms:
        mask = df_text[text_col].str.contains(term, na=False)
        relevant = df_text.loc[mask, id_col].tolist()
        if relevant:
            queries.append({
                "query": term,
                "relevant_ids": relevant[:20],  # cap at 20 to keep benchmark manageable
                "note": "Silver standard: reviews containing query term. Not manually verified."
            })

    logger.info("Built %d benchmark queries.", len(queries))
    return queries[:10]  # Use at most 10 queries


# ─────────────────────────────────────────────────────────────────────────── #
# API-based Retrieval Evaluation
# ─────────────────────────────────────────────────────────────────────────── #

async def evaluate_retrieval(
    queries: List[Dict],
    base_url: str,
    method: str,
    k_values: List[int] = (5, 10),
) -> Dict:
    """Run retrieval benchmark against live API."""
    recall_scores = {k: [] for k in k_values}
    mrr_scores = []
    ndcg10_scores = []
    latencies = []
    failed_requests = 0

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        for q in queries:
            payload = {
                "query": q["query"],
                "mode": method,
                "page": 1,
                "page_size": max(k_values),
            }
            t0 = time.perf_counter()
            try:
                resp = await client.post("/api/search", json=payload)
                latency_ms = (time.perf_counter() - t0) * 1000
                latencies.append(latency_ms)

                if resp.status_code != 200:
                    failed_requests += 1
                    logger.warning("Search returned %d for query '%s'", resp.status_code, q["query"])
                    continue

                data = resp.json()
                retrieved_ids = [r.get("item_id", "") for r in data.get("results", [])]
                relevant_ids = set(q["relevant_ids"])

                for k in k_values:
                    recall_scores[k].append(recall_at_k(retrieved_ids, relevant_ids, k))
                mrr_scores.append(reciprocal_rank(retrieved_ids, relevant_ids))
                ndcg10_scores.append(ndcg_at_k(retrieved_ids, relevant_ids, 10))

            except Exception as e:
                failed_requests += 1
                logger.error("Retrieval error for query '%s': %s", q["query"], e)

    return {
        "method": method,
        "recall_at_5": float(np.mean(recall_scores[5])) if recall_scores.get(5) else 0.0,
        "recall_at_10": float(np.mean(recall_scores[10])) if recall_scores.get(10) else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "ndcg_at_10": float(np.mean(ndcg10_scores)) if ndcg10_scores else 0.0,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "num_queries": len(mrr_scores),
        "failed_requests": failed_requests,
        "attempted_queries": len(queries),
        "note": "Silver-standard benchmark. Relevant items = reviews containing query term."
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Explainability Evaluation
# ─────────────────────────────────────────────────────────────────────────── #

async def evaluate_explainability(
    product_ids: List[str],
    base_url: str,
) -> Dict:
    """
    For each product, fetch investigation and verify:
    1. Every cited review_id exists in source_records
    2. Every graph_path has at least 2 verified entity references
    3. explanation_summary contains none of the forbidden claims
    4. limitations section is non-empty
    """
    forbidden_claims = ["proven", "definitely", "certainly", "confirmed fraud", "is fake"]
    results = []

    async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
        for pid in product_ids:
            try:
                resp = await client.get(f"/api/products/{pid}/investigation")
                if resp.status_code == 404:
                    logger.info("Product %s not found — skipping.", pid)
                    continue
                if resp.status_code != 200:
                    continue

                data = resp.json()
                explanation = data.get("explanation") or {}
                issues = []
                if not explanation:
                    issues.append("No structured explanation returned.")
                data_to_check = explanation or data

                # Check source records
                source_records = set(data_to_check.get("source_records", []))
                suspicious_reviews = [
                    r.get("review_id")
                    for r in data_to_check.get("suspicious_reviews", data_to_check.get("model_predictions", []))
                ]
                for rid in suspicious_reviews:
                    if rid and rid not in source_records:
                        issues.append(f"Review {rid} in suspicious_reviews but not in source_records")

                # Check graph paths have >= 2 entities
                for path in data_to_check.get("graph_paths", []):
                    entities = path.get("entities") or path.get("referenced_entity_ids", [])
                    if len(entities) < 2:
                        issues.append(f"Graph path has only {len(entities)} entity")

                # Check explanation summary for forbidden claims
                summary = (data_to_check.get("explanation_summary") or "").lower()
                for claim in forbidden_claims:
                    if claim in summary:
                        issues.append(f"Explanation summary contains unsupported claim: '{claim}'")

                # Check limitations non-empty
                limitations = data_to_check.get("limitations", [])
                if not limitations:
                    issues.append("No limitations listed in explanation")

                results.append({
                    "product_id": pid,
                    "valid": len(issues) == 0,
                    "issues": issues,
                    "source_records_count": len(source_records),
                    "graph_paths_count": len(data_to_check.get("graph_paths", [])),
                })
            except Exception as e:
                logger.error("Explainability check failed for %s: %s", pid, e)

    if not results:
        return {"status": "no_products_evaluated", "validity_rate": 0.0, "results": []}

    valid_count = sum(1 for r in results if r["valid"])
    return {
        "status": "evaluated",
        "validity_rate": valid_count / len(results),
        "valid_count": valid_count,
        "total_evaluated": len(results),
        "results": results,
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Latency Benchmarking
# ─────────────────────────────────────────────────────────────────────────── #

async def benchmark_latency(base_url: str, product_ids: List[str]) -> Dict:
    """Measure response times for key endpoints."""
    latencies: Dict[str, List[float]] = {
        "health": [], "stats": [], "products_list": [],
        "product_detail": [], "investigation": [],
    }

    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        for _ in range(5):
            t0 = time.perf_counter()
            await client.get("/api/health")
            latencies["health"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            await client.get("/api/stats")
            latencies["stats"].append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            await client.get("/api/products")
            latencies["products_list"].append((time.perf_counter() - t0) * 1000)

        if product_ids:
            pid = product_ids[0]
            for _ in range(3):
                t0 = time.perf_counter()
                await client.get(f"/api/products/{pid}")
                latencies["product_detail"].append((time.perf_counter() - t0) * 1000)

                t0 = time.perf_counter()
                await client.get(f"/api/products/{pid}/investigation")
                latencies["investigation"].append((time.perf_counter() - t0) * 1000)

    return {
        endpoint: {
            "avg_ms": float(np.mean(vals)),
            "min_ms": float(np.min(vals)),
            "max_ms": float(np.max(vals)),
            "p95_ms": float(np.percentile(vals, 95)) if len(vals) >= 2 else float(np.mean(vals)),
        }
        for endpoint, vals in latencies.items()
        if vals
    }


# ─────────────────────────────────────────────────────────────────────────── #
# Main Evaluation Runner
# ─────────────────────────────────────────────────────────────────────────── #

async def run_evaluation(args: argparse.Namespace) -> None:
    base_url = args.api_url
    processed_path = Path(args.processed_path)
    models_path = Path(args.models_path)

    results: Dict[str, Any] = {}

    # ── 1. Load existing classification metrics ──────────────────────────── #
    eval_file = models_path / "evaluation_results.json"
    if eval_file.exists():
        with open(eval_file) as f:
            existing = json.load(f)
        results["classification_metrics"] = existing.get("classification_metrics", {})
        logger.info("Loaded classification metrics from %s", eval_file)
    else:
        results["classification_metrics"] = {
            "status": "not_run",
            "message": "Run scripts/train_model.py first to generate classification metrics."
        }
        logger.warning("No classification metrics found. Run train_model.py first.")

    # ── 2. Load processed data for benchmark construction ─────────────────── #
    csv_path = processed_path / "reviews_processed.csv"
    if args.use_sample:
        csv_path = Path(args.sample_path)

    benchmark_queries: List[Dict] = []
    sample_product_ids: List[str] = []

    if csv_path.exists():
        logger.info("Loading dataset from %s for benchmark construction ...", csv_path)
        df = pd.read_csv(csv_path, nrows=5000)
        label_columns = [c for c in ("label", "is_fake", "fake", "fraud_label") if c in df.columns]
        valid_labels = any(
            df[c].notna().any() and df[c].astype(str).str.strip().ne("").any()
            for c in label_columns
        )
        if not valid_labels:
            results["classification_metrics"] = {
                "status": "not_evaluable",
                "message": "The active dataset has no valid ground-truth fraud labels.",
                "dataset": str(csv_path),
            }
        benchmark_queries = build_benchmark_queries(df)
        # Grab up to 5 product IDs for explainability tests
        pid_col = next((c for c in ["product_id", "asin", "productID"] if c in df.columns), None)
        if pid_col:
            sample_product_ids = df[pid_col].dropna().unique().tolist()[:5]
    else:
        logger.warning("Processed dataset not found at %s — benchmark will be empty.", csv_path)

    # ── 3. Check API availability ─────────────────────────────────────────── #
    api_available = False
    try:
        async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as client:
            resp = await client.get("/api/health")
            if resp.status_code == 200:
                api_available = True
                logger.info("✅ API at %s is reachable.", base_url)
    except Exception:
        logger.warning("⚠️  API at %s not reachable. Skipping live retrieval evaluation.", base_url)

    # ── 4. Retrieval Evaluation ───────────────────────────────────────────── #
    if api_available and benchmark_queries:
        logger.info("Running retrieval benchmarks (%d queries) ...", len(benchmark_queries))
        retrieval_results = {}
        for method in ["keyword", "semantic", "hybrid"]:
            logger.info("  Evaluating '%s' retrieval ...", method)
            retrieval_results[method] = await evaluate_retrieval(
                benchmark_queries, base_url, method
            )
        results["retrieval_metrics"] = retrieval_results
    elif not api_available:
        results["retrieval_metrics"] = {
            "status": "skipped",
            "message": f"API not reachable at {base_url}. Start the backend and re-run."
        }
    else:
        results["retrieval_metrics"] = {
            "status": "skipped",
            "message": "No benchmark queries could be built (dataset not found)."
        }

    # ── 5. Explainability Evaluation ─────────────────────────────────────── #
    if api_available and sample_product_ids:
        logger.info("Running explainability checks on %d products ...", len(sample_product_ids))
        results["explanation_metrics"] = await evaluate_explainability(
            sample_product_ids, base_url
        )
    elif not api_available:
        results["explanation_metrics"] = {"status": "skipped", "message": "API not reachable."}
    else:
        results["explanation_metrics"] = {
            "status": "skipped",
            "message": "No product IDs available for explainability testing."
        }

    # ── 6. Latency Benchmarking ───────────────────────────────────────────── #
    if api_available:
        logger.info("Running latency benchmarks ...")
        results["latency_metrics"] = await benchmark_latency(base_url, sample_product_ids)
    else:
        results["latency_metrics"] = {"status": "skipped", "message": "API not reachable."}

    # ── 7. Metadata ──────────────────────────────────────────────────────── #
    import datetime
    results["generated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    results["benchmark_note"] = (
        "Retrieval benchmark uses a silver-standard query set. "
        "Relevant items are reviews containing the query term — not manually verified. "
        "Do not treat these metrics as ground-truth IR performance."
    )

    # ── 8. Save results ───────────────────────────────────────────────────── #
    models_path.mkdir(parents=True, exist_ok=True)
    out_path = models_path / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("✅ Evaluation results saved to %s", out_path)

    # ── 9. Print summary table ────────────────────────────────────────────── #
    print("\n" + "=" * 70)
    print("  REVIEWGRAPH EVALUATION SUMMARY")
    print("=" * 70)

    # Classification
    cm = results.get("classification_metrics", {})
    if isinstance(cm, dict) and "logistic_regression" in cm:
        lr = cm["logistic_regression"]
        print(f"\n📊 CLASSIFICATION — Logistic Regression")
        print(f"   Precision : {lr.get('precision', 'N/A')}")
        print(f"   Recall    : {lr.get('recall', 'N/A')}")
        print(f"   F1-Score  : {lr.get('f1', 'N/A')}")
        print(f"   PR-AUC    : {lr.get('pr_auc', 'N/A')}")
    else:
        print(f"\n📊 CLASSIFICATION: {cm.get('status', 'unavailable')}")
        print(f"   {cm.get('message', '')}")

    # Retrieval
    print(f"\n🔍 RETRIEVAL METRICS")
    rm = results.get("retrieval_metrics", {})
    if isinstance(rm, dict) and "keyword" in rm:
        print(f"   {'Method':<12} {'Recall@5':>10} {'Recall@10':>11} {'MRR':>8} {'nDCG@10':>10} {'Latency ms':>12}")
        print(f"   {'-'*65}")
        for method in ["keyword", "semantic", "hybrid"]:
            m = rm.get(method, {})
            print(
                f"   {method:<12} "
                f"{m.get('recall_at_5', 0):.4f}     "
                f"{m.get('recall_at_10', 0):.4f}     "
                f"{m.get('mrr', 0):.4f}   "
                f"{m.get('ndcg_at_10', 0):.4f}     "
                f"{m.get('avg_latency_ms', 0):.1f}"
            )
    else:
        print(f"   Status: {rm.get('status', 'unavailable')}")
        print(f"   {rm.get('message', '')}")

    # Explainability
    em = results.get("explanation_metrics", {})
    print(f"\n🔬 EXPLAINABILITY METRICS")
    if isinstance(em, dict) and "validity_rate" in em:
        print(f"   Validity Rate     : {em.get('validity_rate', 0):.1%}")
        print(f"   Valid Explanations: {em.get('valid_count', 0)} / {em.get('total_evaluated', 0)}")
    else:
        print(f"   Status: {em.get('status', 'unavailable')}")

    # Latency
    lm = results.get("latency_metrics", {})
    print(f"\n⚡ LATENCY")
    if isinstance(lm, dict) and "health" in lm:
        for endpoint, stats in lm.items():
            print(f"   {endpoint:<20} avg={stats.get('avg_ms', 0):.1f}ms  p95={stats.get('p95_ms', 0):.1f}ms")
    else:
        print(f"   Status: {lm.get('status', 'unavailable')}")

    print("\n" + "=" * 70)
    print(f"Results written to: {out_path}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the ReviewGraph system: classification, retrieval, and explainability."
    )
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="Base URL of the running ReviewGraph API.")
    parser.add_argument("--processed-path", default="../data/processed",
                        help="Path to the processed dataset directory.")
    parser.add_argument("--sample-path", default="../data/sample/sample_reviews.csv",
                        help="Path to the sample dataset CSV.")
    parser.add_argument("--models-path", default="../models",
                        help="Path to the models directory.")
    parser.add_argument("--use-sample", action="store_true",
                        help="Use sample dataset for benchmark construction instead of full dataset.")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
