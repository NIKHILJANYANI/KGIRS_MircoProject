# Evaluation Methodology & Results

## 1. Evaluation Framework Overview

ReviewGraph is evaluated across four independent dimensions:

| Dimension | Tool | Metric |
|---|---|---|
| **Classification** | scikit-learn | Precision, Recall, F1, PR-AUC |
| **Information Retrieval** | Live API | Recall@5, Recall@10, MRR, nDCG@10 |
| **Explainability** | Automated checks | Validity rate, hallucination rate |
| **System Latency** | httpx timing | Avg ms, p95 ms per endpoint |

All experiments use the same data split and are reproducible by running:

```bash
python scripts/evaluate_system.py --use-sample
```

---

## 2. Classification Evaluation

### 2.1 Metrics

- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **F1-Score:** 2 × (Precision × Recall) / (Precision + Recall)
- **PR-AUC:** Area under the Precision-Recall curve (more informative than ROC-AUC for imbalanced data)
- **Confusion Matrix:** TP, FP, FN, TN

### 2.2 Data Split Strategy

To reduce leakage, the split is **group-aware by reviewer_id**:

- Training set: 70% of reviewer groups
- Validation set: 15% of reviewer groups
- Test set: 15% of reviewer groups

This prevents a reviewer's reviews from appearing in both training and test sets, reducing the risk that the model memorizes reviewer identity rather than learning review patterns.

### 2.3 Class Imbalance Handling

If the ratio of majority to minority class exceeds 3:1 in the training set, **SMOTE** (Synthetic Minority Over-sampling Technique) is applied to the training split only.

SMOTE is never applied to validation or test sets.

### 2.4 Models Compared

| Model | Features | Notes |
|---|---|---|
| **Logistic Regression** (baseline) | TF-IDF + behavioral | `class_weight='balanced'`, max_iter=1000 |
| **Random Forest** | TF-IDF + behavioral | 100 estimators, `class_weight='balanced'` |

The best model by macro-F1 on the test set is saved as `models/review_classifier.pkl`.

### 2.5 ⚠️ Score Interpretation Limitation

> The classifier outputs a `model_score` between 0 and 1. **This is not a calibrated probability of fraud.** Probability calibration (Platt scaling, isotonic regression) has not been applied. The score indicates the model's confidence in the suspicious class relative to its training distribution.

### 2.6 Label Quality Limitation

> Labels in the bretthollenbeck dataset reflect **computer-generated text detection**, not **verified fraud**. Classification performance on this dataset measures the ability to distinguish AI-generated from human-written reviews, not the ability to detect fraudulent intent.

### 2.7 Results

Run `python scripts/train_model.py --use-sample` then `python scripts/evaluate_system.py` to populate results.

Results are stored in `models/evaluation_results.json` and displayed in the Evaluation page of the dashboard.

---

## 3. Retrieval Evaluation

### 3.1 Benchmark Query Set

A silver-standard benchmark is constructed from the processed dataset:

- 10 queries: common review phrases (e.g., "best product", "terrible quality", "highly recommend")
- **Relevant items** for each query: reviews whose text contains the query phrase
- **⚠️ Note:** This is a silver standard — relevance is not manually verified. Results should be interpreted as indicative, not definitive.

### 3.2 Metric Definitions

**Recall@K** — fraction of relevant items found in the top-K results:

$$\text{Recall@K} = \frac{|\text{Relevant} \cap \text{Retrieved}_{1:K}|}{|\text{Relevant}|}$$

**Mean Reciprocal Rank (MRR)** — average of 1/rank of first relevant item:

$$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$

**nDCG@K** — normalized Discounted Cumulative Gain:

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{\text{rel}_i}{\log_2(i+1)}, \quad \text{nDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

where $\text{rel}_i \in \{0, 1\}$ (binary relevance) and IDCG@K is the ideal DCG@K.

### 3.3 Methods Compared

| Method | Description |
|---|---|
| **Keyword** | SQLite FTS5 full-text search on review text |
| **Semantic** | FAISS nearest-neighbor on sentence-transformer embeddings |
| **Graph** | Neo4j Cypher traversal: direct and connected reviews |
| **Hybrid** | Weighted combination: 0.25×keyword + 0.40×semantic + 0.35×graph |

### 3.4 Results Table

*Run `evaluate_system.py` to populate actual values.*

| Method | Recall@5 | Recall@10 | MRR | nDCG@10 | Avg Latency (ms) |
|---|---|---|---|---|---|
| Keyword | — | — | — | — | — |
| Semantic | — | — | — | — | — |
| Hybrid | — | — | — | — | — |

> **Important:** Do not claim that graph-enhanced or hybrid retrieval improves performance unless experimental results support this conclusion.

---

## 4. Explainability Evaluation

### 4.1 Evaluation Criteria

For each product investigation explanation, the system automatically verifies:

| Check | Pass Condition |
|---|---|
| **Edge existence** | Every graph edge in `graph_paths` has a source and target entity |
| **Source record validity** | Every `review_id` in `source_records` exists in the product's actual reviews |
| **Path completeness** | Every graph path has ≥ 2 entities |
| **No unsupported claims** | `explanation_summary` does not contain: *proven, definitely, certainly, confirmed fraud, is fake* |
| **Limitations present** | `limitations` list is non-empty |

### 4.2 Hallucination Prevention

The explanation engine is **template-based**:
- All entities referenced in explanations are taken directly from graph query results
- Reasoning chains are constructed from actual node/edge data
- No language model is used to generate explanations
- The hallucination rate is designed to be 0% by construction

### 4.3 Validity Rate

$$\text{Validity Rate} = \frac{\text{Explanations passing all checks}}{\text{Total explanations evaluated}}$$

---

## 5. Reproducibility

### Step 1: Generate classification metrics

```bash
cd reviewgraph
python scripts/preprocess_data.py --use-sample
python scripts/train_model.py --use-sample
```

### Step 2: Build the knowledge graph

```bash
python scripts/import_graph.py --use-sample
```

### Step 3: Build embedding index

```bash
python scripts/build_embeddings.py --use-sample
```

### Step 4: Start the API

```bash
cd backend
uvicorn app.main:app --reload
```

### Step 5: Run evaluation

```bash
cd ..
python scripts/evaluate_system.py --use-sample
```

Results are saved to `models/evaluation_results.json` and loaded by the Evaluation dashboard page.

---

## 6. Discussion and Limitations

### Expected Findings

- Hybrid retrieval is expected to outperform single-modality retrieval for complex queries
- Graph retrieval adds recall for queries involving connected reviewers across products
- Keyword retrieval has best latency but lowest semantic coverage

### Limitations

1. **Silver-standard benchmark:** Relevant items are identified by keyword matching, not human annotation. This inflates keyword retrieval metrics relative to semantic methods.
2. **Label quality:** `CG` labels reflect AI-generated text detection, not fraud verification.
3. **Small sample size:** With the synthetic 100-row sample, variance in metrics is high. Use the full dataset for reliable conclusions.
4. **No temporal evaluation:** The split does not evaluate temporal generalization (new reviews over time).
5. **Graph cold-start:** Graph retrieval requires the Neo4j database to be populated. An empty graph returns no results.

### Future Work

- Manual annotation of a gold-standard retrieval benchmark
- Probability calibration for the classifier
- Temporal cross-validation
- Graph Neural Network features derived from reviewer-product bipartite graph
- User study evaluating explanation quality
