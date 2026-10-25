# ReviewGraph — Explainable E-commerce Trust Intelligence System

> **KGIRS Academic Project** | Knowledge Graphs and Information Retrieval Systems

ReviewGraph is an explainable investigation tool for e-commerce reviews. It helps users search for relevant reviews, inspect product and reviewer connections, examine observable evidence, and understand model signals without presenting similarity, heuristics, or predictions as proof of fraud.

The active application dataset is a 2,000-row Amazon Reviews 2023 Electronics sample. The original synthetic sample remains available for development and regression workflows, but it is not the active application dataset.

### Current data status

| Item | Current value |
|---|---:|
| Reviews | 2,000 |
| Products | 1,898 |
| Reviewers | 332 |
| Missing review text | 17 |
| Product titles | Unavailable in the review-only source |
| Verified fraud ground truth | None |
| Weak labels | 285 genuine, 38 suspicious, 1,677 unlabeled |
| FAISS vectors | 2,000 |
| Neo4j evidence nodes | 2,706 |

The weak labels are stored with `label_source` and `label_confidence` metadata. They are heuristic/manual-review placeholders, not human-verified ground truth, and must not be used as proof of fraud or validated classification metrics.

---

## 🏗 Architecture Overview

```
Dataset (CSV)
    │
    ▼
Preprocessing Pipeline
    │    ├── TF-IDF similarity pairs
    │    ├── Reviewer / Product stats
    │    └── Normalized labels
    ▼
┌─────────────────────────────────────────────────────────┐
│                     Backend (FastAPI)                    │
│                                                         │
│  ClassificationService  │  EmbeddingService (FAISS)     │
│  GraphService (Neo4j)   │  RetrievalService (Hybrid)    │
│  ExplanationService     │  DatasetService               │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Knowledge Graph (Neo4j)
    Product ─[:REVIEWS]─ Review ─[:WROTE]─ Reviewer
    Review ─[:SIMILAR_TO]─ Review
    Review ─[:HAS_SIGNAL]─ EvidenceSignal
    │
    ▼
Frontend Dashboard (React + Vite + Cytoscape.js)
    Overview │ Search │ Investigation │ Graph │ Review Details │ Evaluation
```

---

## 📋 Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11+ | Required |
| Node.js | 18+ | For frontend |
| Neo4j Aura | Managed cloud graph database |
| Git | Any | For cloning |

---

## 🚀 Quick Start (Local Development)

### Step 1 — Clone and Set Up Python Environment

```bash
git clone <your-repo-url>
cd reviewgraph

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r backend/requirements.txt
```

### Step 2 — Configure Environment

```bash
copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux
```

Edit `.env` and configure the Neo4j connection:
```
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password
```

For Neo4j Aura, use the URI and credentials supplied by Aura. Do not commit `.env` or expose credentials.

### Step 3 — Use the prepared active dataset

The active derived files are:

```text
data/external/amazon_reviews_electronics.csv
data/processed/reviews_processed.csv
data/processed/fts.db
models/review_embeddings.faiss
models/review_embeddings.pkl
models/review_embeddings_metadata.json
models/review_classifier.pkl
```

Dataset provenance and weak-label metadata are documented in [the dataset manifest](data/external/amazon_reviews_electronics_manifest.json).

Do not rebuild these artifacts unless you intentionally want to rerun the pipeline. To run the fail-fast orchestrator:

```powershell
Set-Location reviewgraph
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

### Step 4 — Verify Neo4j Aura access

In the Neo4j Aura console, copy the instance URI, username, and password into `.env`. The backend verifies connectivity during startup. No local Neo4j server is required.

### Step 5 — Train or inspect the classifier

```bash
python scripts/train_model.py --processed-path data/processed
```

The active model is [models/review_classifier.pkl](models/review_classifier.pkl). The service implementation is [classification_service.py](backend/app/services/classification_service.py). Model output is an exploratory signal, not a validated fraud probability.

To test one review directly:

```powershell
Set-Location reviewgraph\backend
..\.venv\Scripts\python.exe -c "from app.services.classification_service import ClassificationService; c=ClassificationService('../models'); print(c.is_ready()); print(c.predict_single('The battery drains too quickly and needs charging several times a day.'))"
```

### Step 6 — Build or verify the embedding index

```bash
python scripts/build_embeddings.py
```

This downloads `all-MiniLM-L6-v2` (~90MB) on first run and builds the FAISS index.

### Step 7 — Import Knowledge Graph

Ensure Neo4j is running, then:

```bash
python scripts/import_graph.py
```

### Step 8 — Start the Backend

```bash
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

API docs available at: http://localhost:8000/api/docs

### Step 9 — Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard available at: http://localhost:5173

---

## 📁 Project Structure

```
reviewgraph/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI application entry point
│   │   ├── api/                       # Route handlers
│   │   │   ├── health.py
│   │   │   ├── stats.py
│   │   │   ├── products.py
│   │   │   ├── reviews.py
│   │   │   ├── search.py
│   │   │   └── evaluation.py
│   │   ├── core/
│   │   │   ├── config.py              # Pydantic settings
│   │   │   ├── logging_config.py      # Structured logging
│   │   │   └── neo4j_driver.py        # Neo4j connection management
│   │   ├── schemas/                   # Pydantic request/response models
│   │   ├── services/
│   │   │   ├── dataset_service.py     # Data loading and preprocessing
│   │   │   ├── classification_service.py  # ML classifier
│   │   │   ├── embedding_service.py   # FAISS + sentence-transformers
│   │   │   ├── graph_service.py       # Neo4j operations
│   │   │   ├── retrieval_service.py   # Keyword/semantic/hybrid search
│   │   │   └── explanation_service.py # Evidence-based explanations
│   │   └── repositories/
│   ├── tests/                         # pytest test suite
│   ├── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/                     # 6 dashboard pages
│   │   │   ├── Overview.jsx
│   │   │   ├── Search.jsx
│   │   │   ├── Investigation.jsx
│   │   │   ├── GraphView.jsx
│   │   │   ├── ReviewDetails.jsx
│   │   │   └── Evaluation.jsx
│   │   ├── components/                # Reusable UI components
│   │   ├── services/api.js            # Axios API client
│   │   └── App.jsx
│   ├── package.json
├── data/
│   ├── raw/          # Download datasets here (not in git)
│   ├── processed/    # Generated by preprocess_data.py
│   └── sample/       # Synthetic sample CSV (in git)
├── models/           # Trained models and FAISS index (not in git)
├── scripts/          # Data pipeline scripts
├── docs/             # Architecture, dataset, evaluation docs
├── .env.example
└── .gitignore
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | System and service health check |
| GET | `/api/stats` | Dataset and graph statistics |
| GET | `/api/products` | Search/list products |
| GET | `/api/products/{id}` | Product details |
| GET | `/api/products/{id}/investigation` | Full suspiciousness investigation |
| GET | `/api/products/{id}/graph` | Graph visualization data |
| GET | `/api/reviews/{id}` | Review details with model score |
| GET | `/api/reviews/{id}/related` | Similar reviews via graph |
| POST | `/api/search` | Keyword/semantic/graph/hybrid search |
| GET | `/api/evaluation` | Stored evaluation results |

Full interactive docs: http://localhost:8000/api/docs

---

## 🔎 Using the application

The intended workflow is:

1. Search for a natural-language concern such as `battery drains too quickly`, `needs frequent charging`, or `wireless earbuds connection issues`.
2. Compare Keyword, Semantic, Graph, and Hybrid retrieval.
3. Open a review to inspect its text, metadata, evidence signals, and live classifier output when available.
4. Open a product investigation to inspect graph relationships, similar reviews, reviewer overlaps, and heuristic suspiciousness evidence.
5. Open the graph view with a current product ID such as `B0742FQ3Q3`.

`POST /api/search` runs the classifier live for returned review results when the model is loaded. Product results and reviews without usable text may not contain model fields. Missing model output remains unavailable; it is never converted into a genuine label.

---

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## 📊 Evaluation

```bash
# Ensure backend is running, then:
python scripts/evaluate_system.py
```

Results are saved to `models/evaluation_results.json` and shown in the Evaluation dashboard page. Retrieval metrics are meaningful only when relevance judgments are available. Classification quality is not validated on the active dataset because its labels are weak labels, not independently verified ground truth.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| Neo4j connection failure | Confirm the Neo4j Aura instance is running and `.env` contains the correct `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` |
| `No module named 'faiss'` | Install: `pip install faiss-cpu` |
| Embedding index not found | Run `python scripts/build_embeddings.py` from the project root |
| Model not found | Run `python scripts/train_model.py --processed-path data/processed` |
| Frontend blank page | Check browser console; ensure backend is running on port 8000 |
| CORS errors | Verify `CORS_ORIGINS` in `.env` includes your frontend URL |
| Neo4j auth failure | Check `NEO4J_PASSWORD` in `.env` matches Neo4j setup |

---

## ⚠️ Important Disclaimers

1. **Model scores are not calibrated probabilities of fraud.** The classifier outputs a raw model signal, not a verified probability.
2. **Weak labels are not ground truth.** The active labels include heuristic/manual-review metadata and an explicit `unlabeled` class.
3. **Graph connections** indicate potentially relevant patterns, not confirmed fraud.
4. **The active dataset has no independently verified fraud labels.** Do not report classification accuracy as validated performance.
5. **Synthetic sample data** is for development only and must not be used for published research results.
6. **This system is for academic research purposes only.** Do not use classifier outputs to make real-world fraud allegations.

---

## 📄 Documentation

- [System Architecture](docs/architecture.md)
- [Dataset Documentation](docs/dataset.md)
- [Evaluation Methodology](docs/evaluation.md)

---

## 📦 Technology Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI + Pydantic + Uvicorn |
| Knowledge Graph | Neo4j Aura |
| ML Classification | scikit-learn (Logistic Regression, Random Forest) |
| Semantic Retrieval | sentence-transformers + FAISS |
| Keyword Search | SQLite FTS5 |
| Frontend | React 18 + Vite + Tailwind CSS |
| Graph Visualization | Cytoscape.js |
| Charts | Recharts |
| Testing | pytest + pytest-asyncio |

All components are **free and open-source**. No paid APIs are required.
