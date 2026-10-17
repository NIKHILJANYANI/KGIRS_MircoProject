# ReviewGraph — Explainable E-commerce Trust Intelligence System

> **KGIRS Academic Project** | Knowledge Graphs and Information Retrieval Systems

ReviewGraph analyzes e-commerce product reviews to identify potentially suspicious activity, retrieve relevant evidence, discover relationships between reviewers and products, and explain why a product or review has been flagged — using a combination of machine learning, semantic retrieval, knowledge graphs, and explainable reasoning paths.

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
| Neo4j Community | 5.x | Graph database |
| Git | Any | For cloning |
| Docker + Docker Compose | Optional | For containerized setup |

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

Edit `.env` and set your Neo4j password:
```
NEO4J_PASSWORD=your_secure_password
```

### Step 3 — Start Neo4j

**Option A: Docker (recommended)**
```bash
docker run -d \
  --name reviewgraph_neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_secure_password \
  neo4j:5.20-community
```

**Option B: Neo4j Desktop**  
Download from https://neo4j.com/download/ and create a new database.

Wait ~30 seconds for Neo4j to start, then verify at http://localhost:7474.

### Step 4 — Prepare Data

**Using synthetic sample (no dataset download required):**
```bash
python scripts/preprocess_data.py --use-sample
```

**Using real dataset:**
1. Download from https://github.com/bretthollenbeck/fake-reviews-data
2. Place CSV in `data/raw/fake_reviews.csv`
3. Run:
```bash
python scripts/inspect_dataset.py --dataset-path data/raw/fake_reviews.csv
python scripts/preprocess_data.py --dataset-path data/raw/fake_reviews.csv
```

### Step 5 — Train the ML Model

```bash
python scripts/train_model.py --use-sample
# Or with real data:
python scripts/train_model.py --processed-path data/processed
```

Trained model is saved to `models/review_classifier.pkl`.

### Step 6 — Build Embedding Index

```bash
python scripts/build_embeddings.py --use-sample
```

This downloads `all-MiniLM-L6-v2` (~90MB) on first run and builds the FAISS index.

### Step 7 — Import Knowledge Graph

Ensure Neo4j is running, then:

```bash
python scripts/import_graph.py --use-sample \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-password your_secure_password
```

### Step 8 — Start the Backend

```bash
cd backend
uvicorn app.main:app --reload --port 8000
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

## 🐳 Docker Compose (Full Stack)

```bash
# Set your Neo4j password in .env first
docker-compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000/api/docs
- Neo4j Browser: http://localhost:7474

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
│   └── Dockerfile
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
│   └── Dockerfile
├── data/
│   ├── raw/          # Download datasets here (not in git)
│   ├── processed/    # Generated by preprocess_data.py
│   └── sample/       # Synthetic sample CSV (in git)
├── models/           # Trained models and FAISS index (not in git)
├── scripts/          # Data pipeline scripts
├── docs/             # Architecture, dataset, evaluation docs
├── docker-compose.yml
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

## 🧪 Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## 📊 Evaluation

```bash
# Ensure backend is running, then:
python scripts/evaluate_system.py --use-sample
```

Results are saved to `models/evaluation_results.json` and shown in the Evaluation dashboard page.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `Connection refused` to Neo4j | Ensure Neo4j is running: `docker ps` or check Neo4j Desktop |
| `No module named 'faiss'` | Install: `pip install faiss-cpu` |
| Embedding index not found | Run `python scripts/build_embeddings.py --use-sample` |
| Model not found | Run `python scripts/train_model.py --use-sample` |
| Frontend blank page | Check browser console; ensure backend is running on port 8000 |
| CORS errors | Verify `CORS_ORIGINS` in `.env` includes your frontend URL |
| Neo4j auth failure | Check `NEO4J_PASSWORD` in `.env` matches Neo4j setup |

---

## ⚠️ Important Disclaimers

1. **Model scores are not calibrated probabilities of fraud.** The classifier outputs a `model_score` indicating suspiciousness, not a verified probability.
2. **Dataset labels** (`CG`/`OR`) indicate computer-generated text detection, not verified fraudulent intent.
3. **Graph connections** indicate potentially relevant patterns, not confirmed fraud.
4. **Synthetic sample data** is for development only and must not be used for published research results.
5. **This system is for academic research purposes only.** Do not use classifier outputs to make real-world fraud allegations.

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
| Knowledge Graph | Neo4j 5.x Community Edition |
| ML Classification | scikit-learn (Logistic Regression, Random Forest) |
| Semantic Retrieval | sentence-transformers + FAISS |
| Keyword Search | SQLite FTS5 |
| Frontend | React 18 + Vite + Tailwind CSS |
| Graph Visualization | Cytoscape.js |
| Charts | Recharts |
| Testing | pytest + pytest-asyncio |
| Container | Docker + Docker Compose |

All components are **free and open-source**. No paid APIs are required.
