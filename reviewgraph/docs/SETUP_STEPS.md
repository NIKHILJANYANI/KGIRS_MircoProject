# ReviewGraph — Setup & Credentials Guide

> Complete step-by-step instructions for getting ReviewGraph running locally.  
> Follow these steps in order.

---

## Prerequisites Checklist

Before starting, install the following:

- [ ] **Python 3.11+** — https://www.python.org/downloads/
- [ ] **Node.js 18+** — https://nodejs.org/
- [ ] **Git** — https://git-scm.com/
- [ ] **Neo4j Community Edition 5.x** — (choose one of the options below)
- [ ] **Docker Desktop** *(optional, for Docker setup)* — https://www.docker.com/products/docker-desktop/

---

## Step 1 — Open the Project Folder

All commands below must be run from the `reviewgraph/` directory:

```powershell
cd "C:\Users\NIKHIL\Documents\Sem-07-Exps\KGIRS\MicroProject\reviewgraph"
```

---

## Step 2 — Create Python Virtual Environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
```

> ⚠️ **Note:** Installing `sentence-transformers` and `faiss-cpu` may take 3–5 minutes.  
> If `faiss-cpu` install fails, try: `pip install faiss-cpu --no-build-isolation`

---

## Step 3 — Configure Credentials (`.env` file)

### 3a. Create the `.env` file

```powershell
copy .env.example .env
```

### 3b. Edit `.env` — Required Settings

Open `.env` in any text editor and fill in:

```env
# ── REQUIRED ─────────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=YOUR_PASSWORD_HERE     ← Set this to your Neo4j password

# ── Optional: change if using different ports ─────────────
APP_PORT=8000
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

> 🔐 **Security:** Never commit `.env` to Git. It is already in `.gitignore`.

---

## Step 4 — Install and Start Neo4j

### Option A: Docker (Recommended — No Desktop App Needed)

```powershell
docker run -d `
  --name reviewgraph_neo4j `
  -p 7474:7474 -p 7687:7687 `
  -e NEO4J_AUTH=neo4j/YOUR_PASSWORD_HERE `
  neo4j:5.20-community
```

Replace `YOUR_PASSWORD_HERE` with the same password you set in `.env`.

Verify Neo4j is running:
- Open browser: http://localhost:7474
- Login: Username = `neo4j`, Password = your password

### Option B: Neo4j Desktop

1. Download Neo4j Desktop: https://neo4j.com/download/
2. Install and open it
3. Create a new **Local DBMS**
4. Set a password — use the same password you put in `.env`
5. Click **Start** to start the database
6. Verify at: http://localhost:7474

### Option C: Neo4j AuraDB Free (Cloud)

1. Create free account: https://neo4j.com/cloud/platform/aura-graph-database/
2. Create a free AuraDB instance
3. Copy the connection details into `.env`:
   ```env
   NEO4J_URI=neo4j+s://xxxxxxxx.databases.neo4j.io
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_aura_password
   ```

> ⏳ Wait ~30 seconds after starting Neo4j before proceeding.

---

## Step 5 — Download the Dataset (Optional — Sample Included)

A **synthetic sample** of 100 reviews is already provided at `data/sample/sample_reviews.csv`.  
You can skip to Step 6 and use `--use-sample` flags for all scripts.

**To use the real dataset:**

1. Visit: https://github.com/bretthollenbeck/fake-reviews-data
2. Download the CSV file
3. Place it at: `data\raw\fake_reviews.csv`
4. Run the inspection script:
   ```powershell
   python scripts\inspect_dataset.py --dataset-path data\raw\fake_reviews.csv
   ```
5. Read the output carefully — verify the column names and label values match expectations

---

## Step 6 — Preprocess the Dataset

**Using synthetic sample (recommended for first run):**
```powershell
python scripts\preprocess_data.py --use-sample
```

**Using real dataset:**
```powershell
python scripts\preprocess_data.py --dataset-path data\raw\fake_reviews.csv
```

Output files created in `data\processed\`:
- `reviews_processed.csv`
- `reviewer_stats.csv`
- `product_stats.csv`
- `similarity_pairs.csv`

---

## Step 7 — Train the ML Model

```powershell
python scripts\train_model.py --use-sample
```

This trains both Logistic Regression and Random Forest, evaluates them, and saves the best model to `models\review_classifier.pkl`.

Expected output:
```
Training Logistic Regression ...
Training Random Forest ...
Best model: logistic_regression  (F1=0.xx)
Model saved to models\review_classifier.pkl
```

If you see `No labeled rows found`, the label column was not detected. Run `inspect_dataset.py` to diagnose.

---

## Step 8 — Build the Embedding Index

```powershell
python scripts\build_embeddings.py --use-sample
```

**First run only:** Downloads `all-MiniLM-L6-v2` (~90MB from Hugging Face). Requires internet.

Subsequent runs reuse the cached model. Use `--force-rebuild` to regenerate.

Output files:
- `models\faiss_index.bin`
- `models\embeddings_cache.npz`

---

## Step 9 — Import the Knowledge Graph

Ensure Neo4j is running (Step 4), then:

```powershell
python scripts\import_graph.py `
  --use-sample `
  --neo4j-uri bolt://localhost:7687 `
  --neo4j-user neo4j `
  --neo4j-password YOUR_PASSWORD_HERE
```

Expected output:
```
Creating schema constraints ...
Importing 5 products ...
Importing 30 reviewers ...
Importing 100 reviews ...
Creating WROTE relationships ...
Creating REVIEWS relationships ...
Importing evidence signals ...
Creating SIMILAR_TO relationships ...
Import complete: 5 products, 30 reviewers, 100 reviews, XX signals, XX similarity edges
```

Verify in Neo4j Browser (http://localhost:7474):
```cypher
MATCH (n) RETURN labels(n)[0] AS type, count(n) AS count
```

---

## Step 10 — Start the Backend

Open a **new terminal** (keep Neo4j running):

```powershell
cd "C:\Users\NIKHIL\Documents\Sem-07-Exps\KGIRS\MicroProject\reviewgraph"
.venv\Scripts\activate
cd backend
uvicorn app.main:app --reload --port 8000
```

Verify backend is running:
- http://localhost:8000 → API info
- http://localhost:8000/api/docs → Interactive API documentation
- http://localhost:8000/api/health → Health check

You should see: `{"status":"ok","neo4j":"ok",...}`

---

## Step 11 — Start the Frontend

Open a **new terminal**:

```powershell
cd "C:\Users\NIKHIL\Documents\Sem-07-Exps\KGIRS\MicroProject\reviewgraph\frontend"
npm install
npm run dev
```

Open: **http://localhost:5173**

---

## Step 12 — Run Evaluation (Optional)

With backend running:

```powershell
cd "C:\Users\NIKHIL\Documents\Sem-07-Exps\KGIRS\MicroProject\reviewgraph"
.venv\Scripts\activate
python scripts\evaluate_system.py --use-sample
```

Results saved to `models\evaluation_results.json` and shown in the Evaluation page.

---

## Alternative: Docker Compose (All-in-One)

If you have Docker Desktop installed:

```powershell
# 1. Edit .env to set NEO4J_PASSWORD
# 2. Build and start everything:
docker-compose up --build
```

Services:
- Frontend: http://localhost:5173
- Backend: http://localhost:8000/api/docs
- Neo4j: http://localhost:7474

---

## Running Tests

```powershell
cd "C:\Users\NIKHIL\Documents\Sem-07-Exps\KGIRS\MicroProject\reviewgraph\backend"
.venv\Scripts\activate   # if not already activated from parent
pytest tests\ -v
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ServiceUnavailable: Failed to establish connection to Neo4j` | Start Neo4j (Step 4). Check password in `.env` |
| `FAISS index not found` | Run Step 8: `build_embeddings.py` |
| `Model not ready` | Run Step 7: `train_model.py` |
| `No module named 'sentence_transformers'` | Activate venv: `.venv\Scripts\activate` then `pip install -r backend\requirements.txt` |
| Frontend shows `Network Error` | Backend not running. Start uvicorn (Step 10) |
| `CORS error` in browser | Add frontend URL to `CORS_ORIGINS` in `.env` |
| `npm install` fails | Update Node.js to v18+. Delete `node_modules\` and retry |
| Graph is empty in dashboard | Run Step 9: `import_graph.py` |
| `tqdm` not found | `pip install tqdm` |

---

## What NOT to Do

- ❌ Do **not** commit `.env` to Git
- ❌ Do **not** place real dataset files in `data/raw/` if they are in `.gitignore`
- ❌ Do **not** use synthetic sample results for academic reporting — they are for development only
- ❌ Do **not** interpret `model_score` as a "fraud probability" — it is not calibrated
