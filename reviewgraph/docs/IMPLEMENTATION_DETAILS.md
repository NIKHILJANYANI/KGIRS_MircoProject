# ReviewGraph — Implementation Details

> **KGIRS Academic Project Documentation**  
> Explainable Graph-Based Information Retrieval for Fake Review Detection and E-commerce Trust Analysis

---

## 1. System Architecture

ReviewGraph is a full-stack application organized into six independent layers:

```
┌───────────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                          │
│   Overview │ Search │ Investigation │ Graph │ ReviewDetails │ Eval │
└─────────────────────────┬─────────────────────────────────────────┘
                           │ HTTP/JSON (REST)
┌─────────────────────────▼─────────────────────────────────────────┐
│                   FastAPI Backend (Python 3.11)                    │
│                                                                    │
│  /api/health  /api/stats  /api/products  /api/reviews             │
│  /api/search  /api/evaluation                                      │
│                                                                    │
│  ┌──────────────────┐  ┌─────────────────────────────────────────┐ │
│  │  DatasetService  │  │         ClassificationService           │ │
│  │  (pandas, I/O)   │  │  (TF-IDF + LogReg + RandomForest)      │ │
│  └──────────────────┘  └─────────────────────────────────────────┘ │
│  ┌──────────────────┐  ┌─────────────────────────────────────────┐ │
│  │  GraphService    │  │         EmbeddingService                │ │
│  │  (Neo4j Cypher)  │  │  (sentence-transformers + FAISS)        │ │
│  └──────────────────┘  └─────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │          RetrievalService (Hybrid Ranking)                    │ │
│  │    keyword(0.25) + semantic(0.40) + graph(0.35)              │ │
│  └──────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │       ExplanationService (Template-Based, No LLM)            │ │
│  └──────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
         │                    │                    │
    ┌────▼────┐          ┌────▼────┐         ┌────▼────┐
    │  Neo4j  │          │  FAISS  │         │ SQLite  │
    │  Graph  │          │  Index  │         │  FTS5   │
    └─────────┘          └─────────┘         └─────────┘
```

---

## 2. Knowledge Graph Schema

### Node Types

| Node | Key Property | Other Properties |
|---|---|---|
| **Product** | `product_id` | `title`, `category`, `average_rating`, `review_count` |
| **Reviewer** | `reviewer_id` | `review_count`, `distinct_products_reviewed` |
| **Review** | `review_id` | `review_text`, `rating`, `timestamp`, `dataset_label`, `predicted_class`, `model_score` |
| **EvidenceSignal** | `signal_id` | `signal_type`, `signal_value`, `description`, `computation_method` |

### Relationship Types

| Relationship | From | To | Properties |
|---|---|---|---|
| `WROTE` | Reviewer | Review | — |
| `REVIEWS` | Review | Product | — |
| `SIMILAR_TO` | Review | Review | `score`, `method` |
| `HAS_SIGNAL` | Review | EvidenceSignal | — |

### Evidence Signal Types

| Signal Type | Condition | Computation |
|---|---|---|
| `HIGH_MODEL_SCORE` | `model_score > 0.7` | Logistic Regression output |
| `EXTREME_RATING` | `rating in {1.0, 5.0}` | Direct from dataset |
| `SHORT_REVIEW` | `word_count < 10` | Text feature |
| `HIGH_REVIEWER_VOLUME` | `reviewer_review_count > 20` | Aggregated stat |
| `TEXT_SIMILARITY` | Cosine similarity > threshold | TF-IDF cosine |

---

## 3. Machine Learning Pipeline

### Feature Engineering

**Text features (TF-IDF):**
- TF-IDF with max 10,000 features, bigrams
- Review length (characters)
- Word count
- Average word length

**Behavioral features:**
- Reviewer total review count
- Reviewer distinct products reviewed
- Rating deviation from product average

**Sparse feature matrix:**  
TF-IDF sparse matrix + behavioral features concatenated with `scipy.sparse.hstack`.

### Training Protocol

1. Group-aware split by `reviewer_id` (70/15/15 train/val/test)
2. SMOTE applied to training split if class imbalance > 3:1
3. TF-IDF fitted on training text only (no leakage)
4. Logistic Regression (`class_weight='balanced'`, C=1.0, max_iter=1000)
5. Random Forest (`n_estimators=100`, `class_weight='balanced'`)
6. Best model by macro-F1 saved to `models/review_classifier.pkl`

### ⚠️ Score Disclaimer

The classifier outputs `model_score` (0–1). This is **not a calibrated probability of fraud**. It represents the model's confidence in the suspicious class relative to its training distribution.

---

## 4. Retrieval Engine

### A. Keyword Retrieval
- **Technology:** SQLite FTS5 (standard library)
- **Index:** Review text + product title
- **Scoring:** BM25-style rank from FTS5

### B. Semantic Retrieval
- **Model:** `all-MiniLM-L6-v2` (Sentence Transformers, ~90MB, free)
- **Index:** FAISS `IndexFlatIP` (inner product = cosine similarity after L2 normalization)
- **Cache:** Embeddings cached in `.npz`; index rebuilt only if `--force-rebuild` passed or cache missing

### C. Graph Retrieval
- **Database:** Neo4j Community Edition
- **Query pattern:** Product → Reviews → SIMILAR_TO → Reviews → Products → Reviewers
- **Limits:** Configurable traversal depth (default 3) and result limits (default 50)

### D. Hybrid Ranking

```
combined_score = 0.25 × keyword_score_normalized
              + 0.40 × semantic_score_normalized
              + 0.35 × graph_score_normalized
```

All scores normalized to [0,1] before combination using min-max normalization. Missing signals are treated as 0.

---

## 5. Explainability Engine

Every product investigation returns:

| Field | Source | Description |
|---|---|---|
| `investigated_entity` | Graph query | Product node data |
| `observed_facts` | Graph query | Retrieved review records |
| `computed_signals` | EvidenceSignal nodes | Computed signals with computation_method |
| `model_predictions` | ClassificationService | model_score + predicted_class + warning |
| `graph_paths` | Graph traversal | Actual Cypher traversal paths |
| `source_records` | Review IDs | IDs of every cited review |
| `explanation_summary` | Template engine | Non-hallucinated summary |
| `limitations` | Static + dynamic | Always shown |
| `alternative_interpretations` | Template engine | Alternative readings |

**Example reasoning path:**
```
Product [P001] → Review [R071] → SIMILAR_TO (score=0.89) → Review [R074] → Product [P003]
```

**What the system asserts:** Review R071 is textually similar to R074, which reviews a different product. Both are associated with reviewer REV026.  
**What the system does NOT assert:** This proves coordinated fraud.

---

## 6. API Design

All endpoints follow RESTful conventions. Defined in `backend/app/api/`.

| File | Endpoints |
|---|---|
| `health.py` | GET /api/health |
| `stats.py` | GET /api/stats |
| `products.py` | GET /api/products, GET /api/products/{id}, GET /api/products/{id}/investigation, GET /api/products/{id}/graph |
| `reviews.py` | GET /api/reviews/{id}, GET /api/reviews/{id}/related |
| `search.py` | POST /api/search |
| `evaluation.py` | GET /api/evaluation |

### Error Responses

| Situation | HTTP Status |
|---|---|
| Resource not found | 404 |
| Database unavailable | 503 |
| Input validation failed | 422 |
| Unhandled server error | 500 |

The API **never silently substitutes mock data** when a service is unavailable.

---

## 7. Frontend Dashboard (6 Pages)

| Page | Route | Key Features |
|---|---|---|
| **Overview** | `/` | Stats cards, pipeline diagram, service health, label distribution chart |
| **Search** | `/search` | 4-mode search (keyword/semantic/graph/hybrid), score bars, latency display |
| **Investigation** | `/investigation` | Suspiciousness gauge, evidence signals, graph reasoning paths, limitations |
| **Knowledge Graph** | `/graph` | Cytoscape.js interactive visualization, node inspector, legend |
| **Review Details** | `/reviews` | Full review, model score, signals, related reviews |
| **Evaluation** | `/evaluation` | Classification metrics, retrieval comparison table, explainability metrics |

### Key UI Components

| Component | Purpose |
|---|---|
| `SuspiciousnessGauge` | Animated SVG circular gauge (0→critical) |
| `EvidencePath` | Graph path as colored entity pills with arrows |
| `ReviewCard` | Review snippet with model score bar |
| `SignalCard` | Evidence signal with type-colored badge |
| `ScoreBar` | Animated progress bar with auto-color |
| `LoadingSkeleton` | Pulse-animated skeleton loaders |

---

## 8. Testing Strategy

Tests in `backend/tests/`:

| Test File | Covers |
|---|---|
| `test_dataset_service.py` | Loading, deduplication, label normalization, NaN handling |
| `test_classification_service.py` | Training, model_score key (not fraud_probability), warning, persistence |
| `test_api.py` | Health endpoint, search validation, 422/404 responses |
| `test_explanation_service.py` | Source record validity, limitations, forbidden certainty claims |

All tests use synthetic fixtures explicitly labeled as synthetic. No real dataset metrics are reported in test output.

---

## 9. Data Pipeline Scripts

| Script | Purpose |
|---|---|
| `inspect_dataset.py` | Print dataset report, detect columns, suggest mappings |
| `preprocess_data.py` | Clean, normalize, compute features, compute similarity pairs |
| `train_model.py` | Train and evaluate Logistic Regression + Random Forest |
| `build_embeddings.py` | Generate sentence embeddings, build FAISS index |
| `import_graph.py` | Import all entities and relationships into Neo4j |
| `evaluate_system.py` | Measure Recall@K, MRR, nDCG@10, explainability, latency |

---

## 10. Security Implementation

- All credentials in `.env` (never hardcoded)
- Parameterized Cypher queries (no string interpolation)
- Parameterized SQLite queries
- Pydantic input validation on all endpoints
- Configurable traversal depth and result limits
- CORS restricted to configured origins
- Sensitive fields not exposed beyond public dataset
