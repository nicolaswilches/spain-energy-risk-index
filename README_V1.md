# Spain Energy Grid Risk Index: MLOps Pipeline

This guide outlines a step-by-step, practical roadmap for building, tracking, and deploying a machine learning pipeline that forecasts the next-day systemic risk of Spain's electrical grid. Each step includes a clear goal, what to do, and what "done" looks like.

> **IE University — MLOps Final Project · Group 8**  
> Alberto Cabezudo · Madelyn Ehni · Gilles Hamers · Nicolás Higuera · Salah Mneimne

---

## Prerequisites

Before you begin, ensure you have the following tools installed and configured on your system.

- **Python 3.12:** The core programming language for the project.
  - **To Install:** Download from [python.org](https://www.python.org/downloads/).
  - **To Verify:** Run `python --version` — must show `3.12.x`.

- **Git:** The version control system used to manage the project's source code.
  - **To Install:** Download from [git-scm.com](https://git-scm.com/downloads).
  - **To Verify:** Run `git --version`.

- **Docker Desktop:** Used to containerize the model serving application.
  - **To Install:** Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop).
  - **To Verify:** Run `docker --version`. Make sure Docker Desktop is **running** before building images.

- **(Recommended) VS Code:** A modern code editor with good Python support.
  - **To Install:** Download from [code.visualstudio.com](https://code.visualstudio.com/) with the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python).

---

## 0. Environment & Packaging

**Goal:** Deterministic local runs and easy hand-off across the team.

**Do:**
- Clone the repository.
- Create a single virtual environment at the repo root.
- Use `requirements.txt` with pinned versions to manage dependencies.
- Set `PYTHONPATH=src` so the `grid_risk` package is importable.

**How to set up:**
```bash
# 1. Clone the repo
git clone https://github.com/nicolaswilches/spain-energy-risk-index.git
cd spain-energy-risk-index

# 2. Create virtual environment
python3.12 -m venv .venv

# 3. Activate
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set PYTHONPATH (required for all scripts)
export PYTHONPATH=src            # Mac/Linux
$env:PYTHONPATH="src"            # Windows PowerShell
```

**Deliverable:**
- `pip install -r requirements.txt` completes without errors.
- `python -c "import pandas, lightgbm, mlflow, fastapi; print('OK')"` prints `OK`.

---

## 1. Data Extraction

**Goal:** A single clean daily dataset merged from REE and Open-Meteo APIs, ready for feature engineering.

**Do:**
- Fetch daily data from the **REE public API** (demand, generation mix, spot price) and **Open-Meteo** weather archive.
- Resample hourly REE data to daily means.
- Merge all streams on date index, impute small gaps, add calendar and degree-day features.
- Save to a single parquet file.
```bash
python scripts/run_extraction.py

# Optional: custom date range
python scripts/run_extraction.py --start 2021-01-01 --end 2025-12-31
```

> ⏳ Takes 10–15 minutes due to REE API rate limiting (1 request/second). No API keys required.

**Deliverable:**
- `data/merged_daily_data.parquet` exists with daily rows from 2021 to present.
- Less than 2% missing values per column.

---

## 2. Feature Engineering & Risk Index

**Goal:** A PCA-derived Grid Risk Index (0–1) and a clean train/val/test feature matrix with no data leakage.

**Do:**
- Compute the three core risk factors from raw data:
  - `flexibility_share` = (combined_cycle + hydro) / total_generation
  - `demand_forecast_error` = actual_demand − forecast_demand
  - `net_load` = actual_demand − (wind + solar_pv)
- Fit PCA on **training data only** → extract PC1 → MinMaxScale to [0, 1].
- Assign 5-tier risk categories: `Low / Stable / Elevated / Severe / Extreme`.
- Add 1-day and 7-day lag features.
- Split chronologically: train (≤ 2023-12-31), val (2024-01-01 to 2024-06-30), test (2024-07-01 onward).
```bash
python scripts/run_features.py
```

**Deliverable:**
- `data/train.parquet`, `data/val.parquet`, `data/test.parquet` produced.
- `data/artifacts/risk_index_fit.joblib` saved (fitted PCA pipeline + thresholds).
- No NaN values in the feature matrix. No future data leaks into the training set.

---

## 3. Model Training & Experiment Tracking (MLflow)

**Goal:** A trained LightGBM quantile model that outperforms the persistence baseline, with all runs tracked in MLflow.

**Do:**
- Train three models in sequence: persistence baseline → Ridge baseline → LightGBM.
- Use Optuna to tune LightGBM hyperparameters (minimizes quantile loss on validation set).
- Evaluate on the held-out chronological test set using business metrics:
  - **RMSE Skill Score** — % improvement over persistence baseline.
  - **Cost Matrix Penalty** — FN = 10 pts, FP = 1 pt (missing an Extreme event costs 10×).
  - **F₃ Score (Extreme class)** — recall weighted 9× over precision.
- Log all parameters, metrics, and model artifacts to MLflow.
```bash
# Full training with Optuna tuning (recommended)
python scripts/train_model.py --n-trials 30

# Fast run, skip tuning
python scripts/train_model.py --skip-tuning
```

**View MLflow UI:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

**Deliverable:**
- MLflow UI shows multiple runs with clearly logged parameters and metrics.
- `data/artifacts/lgbm_model.joblib` saved.
- `data/artifacts/metrics.json` saved.
- `run_id.txt` written with the active MLflow run ID.
- LightGBM achieves a positive RMSE Skill Score and F₃ > 0 on the test set.

---

## 4. Model Serving (FastAPI + Docker)

**Goal:** A containerized API that accepts a target date and returns a live risk prediction.

**Do:**
- Copy trained artifacts into `models/` for Docker.
- Run `app.py` as a FastAPI service on port `9696`.
- The `/predict` endpoint fetches the last 10 days of live REE + weather data, builds features in real time, and returns a prediction.
- Containerize with the Dockerfile — model artifacts are baked in at build time.
```bash
# Copy artifacts
mkdir -p models
cp data/artifacts/lgbm_model.joblib models/
cp data/artifacts/risk_index_fit.joblib models/

# Build and run
docker build -t grid-risk-app .
docker run -d -p 9696:9696 --name grid-risk grid-risk-app

# Health check
curl http://localhost:9696/health

# Predict
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"target_date": "2026-03-15"}'
```

**Expected prediction response:**
```json
{
  "date": "2026-03-15",
  "risk_index": 0.6231,
  "risk_category": "Elevated",
  "model_version": "b3bac1df"
}
```

**Interactive API docs:** `http://localhost:9696/docs`

**Deliverable:**
- `docker run` starts the container without errors.
- `/health` returns `"status": "ok"` with `model_loaded: true`.
- `/predict` returns a valid `risk_index` between 0 and 1 and a valid category.
- `docker stop grid-risk && docker rm grid-risk` cleans up cleanly.

---

## 5. Web App (Frontend UI)

**Goal:** A browser-based interface that lets anyone query the API without using the terminal.

**Do:**
- Open `webapp/index.html` directly in a browser (no server required).
- Set the API URL to `http://localhost:9696` (local) or your Render URL (production).
- Select a target date and click **Predict Risk**.
```bash
# Mac
open webapp/index.html

# Windows
start webapp/index.html

# Linux
xdg-open webapp/index.html
```

> **Note:** CORS must be enabled in `app.py`. Ensure `CORSMiddleware` is added and the Docker image is rebuilt before opening the web app.

**Deliverable:**
- Web app opens in browser with no errors.
- Selecting a date and clicking Predict Risk returns a colour-coded risk score and category.
- Works against both the local Docker container and the Render deployment.

---

## 6. CI/CD (GitHub Actions + GHCR + Render)

**Goal:** Automated training, testing, and image-based deployment on every push to `main`.

**Do:**
- `train.yml`: Reusable workflow that trains the model on sample data and uploads artifacts.
- `ci-cd.yml`: Main orchestrator — calls training → lints with flake8 → builds Docker image with model baked in → runs integration tests against a live container → pushes to GitHub Container Registry.
- Deploy on Render from the GHCR image. Manual deploy to pull the latest image after each push.

**Pipeline flow:**
```
Push to main
    │
    ▼
[train.yml]   Train on sample data (python train.py --skip-tuning)
    │
    ▼
[ci-cd.yml]   Lint (flake8) → Build Docker image → Test container → Push to GHCR
                    ghcr.io/nicolaswilches/spain-energy-risk-index:latest
                    ghcr.io/nicolaswilches/spain-energy-risk-index:<commit-sha>
```

**Push to GitHub:**
```bash
git add .
git commit -m "your message"
git push origin main
```

**Deploy on Render (first time):**
1. Go to [render.com](https://render.com) → **New → Web Service**
2. Choose **Deploy an existing image**
3. Image URL: `ghcr.io/nicolaswilches/spain-energy-risk-index:latest`
4. Port: `9696` · Health check path: `/health`
5. Click **Deploy**

**Deliverable:**
- Green CI pipeline on every push to `main`.
- Docker image published to GHCR with `:latest` and commit SHA tags.
- Live endpoint on Render returns a valid response from `/health` and `/predict`.

---

## Tips & Best Practices

- **PYTHONPATH=src always:** The `grid_risk` package lives under `src/`. Every script requires `export PYTHONPATH=src` to be set. Add it to your `~/.zshrc` to avoid repeating it.
- **Never shuffle time series data:** The train/val/test split is strictly chronological. Shuffling would leak future data into training and invalidate all metrics.
- **Quantile RMSE is expected to be worse than baseline:** The model predicts the 90th percentile on purpose. Use the Cost Matrix Penalty and F₃ Score as the real evaluation criteria — not RMSE alone.
- **Model artifacts are baked into Docker:** The `Dockerfile` copies `models/` at build time. Always run the `cp` commands before `docker build` after retraining.
- **Config, not constants:** All API URLs, column names, date ranges, and thresholds live in `src/grid_risk/config.py`. Change settings there, not inside scripts.
- **Sample data for CI:** `train.py` (the CI script) uses `data/sample_*.parquet` so the pipeline runs in under 2 minutes on GitHub Actions without hitting external APIs.