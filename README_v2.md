# Spain Energy Grid Risk Index ⚡

A production-grade machine learning system that forecasts the **next-day systemic risk** of Spain's electrical grid. The model predicts a continuous Grid Risk Index (0 = stable, 1 = critical) using a quantile LightGBM regressor trained on REE demand, generation mix, weather, and spot price data — deployed as a containerized FastAPI service with full CI/CD automation.

> **IE University — MLOps Final Project · Group 8**  
> Alberto Cabezudo · Madelyn Ehni · Gilles Hamers · Nicolás Higuera · Salah Mneimne

---

## Table of Contents

- [Business Problem](#business-problem)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Running with Docker](#running-with-docker)
- [Web App](#web-app)
- [API Reference](#api-reference)
- [CI/CD Pipeline](#cicd-pipeline)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Running Tests](#running-tests)
- [Deployment on Render](#deployment-on-render)
- [Best Practices Checklist](#best-practices-checklist)
- [Troubleshooting](#troubleshooting)

---

## Business Problem

Red Eléctrica de España (REE), the national transmission system operator, currently lacks a unified, day-ahead measure of systemic grid tension. While static forecasts for demand and generation exist in isolation, they fail to capture the overall stress the grid will face the following day.

As renewable energy penetration increases, the grid becomes more volatile. This system provides an **early-warning mechanism** — a single Next-Day Grid Risk Index — enabling REE operators to:

- Plan flexible energy source dispatch proactively
- Avoid costly last-minute emergency interventions (e.g. emergency gas-plant activation)
- Reduce the risk of unplanned load shedding

The model uses **quantile regression (α = 0.90)**, making it deliberately pessimistic: it predicts the 90th percentile of risk rather than the mean. Missing a genuine high-risk event (false negative) costs 10× more than a false alarm (false positive), so recall for the Extreme class is the primary evaluation criterion.

---

## How It Works

The system follows a two-stage approach.

**Stage 1 — Grid Risk Index Construction**

Three core stress factors are computed from daily REE data:

- `flexibility_share` = (combined_cycle + hydro) / total_generation — proportion of dispatchable generation
- `demand_forecast_error` = actual_demand − forecast_demand — real-time scheduling deviation
- `net_load` = actual_demand − (wind + solar_pv) — structural pressure on flexible sources

These are standardized and compressed into a single **Grid Risk Index** via PCA (PC1), then scaled to [0, 1] using MinMaxScaler. The PCA is fitted on training data only to prevent data leakage.

**Stage 2 — Next-Day Forecasting**

A LightGBM quantile regressor predicts tomorrow's risk index using 15 features:

- REE forecast: `forecast_demand_mw`
- Weather: `temperature_2m_max`, `temperature_2m_min`, `wind_speed_10m_max`, `shortwave_radiation_sum`, `precipitation_sum`
- Derived weather: `hdd` (heating degree days), `cdd` (cooling degree days)
- Calendar: `day_of_week`, `month`, `is_weekend`, `is_holiday`
- Spot price: `spot_price_eur_mwh`
- Temporal lags: `risk_index_lag_1d`, `risk_index_lag_7d`

**Risk Categories (5-tier):**

- `Low` — ≤ p20 — Normal operations
- `Stable` — p20 to p40 — Slightly elevated, no action needed
- `Elevated` — p40 to p60 — Monitor closely
- `Severe` — p60 to p80 — Prepare flexible reserves
- `Extreme` — > p80 — High-priority intervention required

---

## Project Structure
```
spain-energy-risk-index/
│
├── src/grid_risk/              # Core Python package (pure logic, no file I/O)
│   ├── __init__.py
│   ├── api_client.py           # REE + Open-Meteo HTTP clients (retry, pagination)
│   ├── extractors.py           # Raw API JSON → clean DataFrames per endpoint
│   ├── cleaning.py             # Alignment, imputation, calendar features, validation
│   ├── pipeline.py             # Orchestrates extraction → cleaning → merge → save
│   ├── config.py               # API URLs, date ranges, column mappings, thresholds
│   ├── features.py             # PCA risk index, feature engineering, train/val/test split
│   └── model.py                # LightGBM training, baselines, evaluation metrics
│
├── scripts/
│   ├── run_extraction.py       # Phase 1 CLI: fetch data from REE + Open-Meteo APIs
│   ├── run_features.py         # Phase 2 CLI: build features + fit risk index
│   └── train_model.py          # Phase 3 CLI: train baselines + LightGBM + evaluate
│
├── .github/
│   └── workflows/
│       ├── ci-cd.yml           # Main CI/CD pipeline (train → lint → build → push)
│       └── train.yml           # Reusable training workflow (sample data, no tuning)
│
├── notebooks/
│   └── etl_modeling.ipynb      # Exploratory analysis and model prototyping
│
├── data/
│   ├── sample_train.parquet    # Sample training data (for CI)
│   ├── sample_val.parquet      # Sample validation data (for CI)
│   └── sample_test.parquet     # Sample test data (for CI)
│
├── webapp/
│   └── index.html              # Frontend web app (connects to FastAPI)
│
├── app.py                      # FastAPI prediction service (live REE + weather data)
├── train.py                    # CI training script (uses sample data, self-contained)
├── Dockerfile                  # Containerized service (model baked in at build time)
├── render.yaml                 # Render.com deployment manifest
├── test_api.py                 # Integration tests for the API
├── requirements.txt            # Pinned production dependencies
├── pyproject.toml              # Package metadata and optional dependencies
├── run_id.txt                  # MLflow run ID of the deployed model
├── LESSONS.md                  # Engineering lessons learned
└── .gitignore
```

---

## Prerequisites

Ensure the following tools are installed before starting.

**Python 3.12**
```bash
python --version   # must show 3.12.x
```
Download from [python.org](https://www.python.org/downloads/) if needed.

**Git**
```bash
git --version
```
Download from [git-scm.com](https://git-scm.com/downloads) if needed.

**Docker Desktop**
```bash
docker --version
```
Download from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop) if needed. Make sure Docker Desktop is **running** before building images.

**VS Code (recommended)**  
Download from [code.visualstudio.com](https://code.visualstudio.com/) with the Python extension.

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/nicolaswilches/spain-energy-risk-index.git
cd spain-energy-risk-index
```

### 2. Create and activate a virtual environment
```bash
# Create
python3.12 -m venv .venv

# Activate — Mac/Linux
source .venv/bin/activate

# Activate — Windows PowerShell
.venv\Scripts\activate
```

You should see `(.venv)` at the start of your terminal prompt.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

Verify the install:
```bash
python -c "import pandas, sklearn, lightgbm, mlflow, fastapi; print('OK')"
```

### 4. Set PYTHONPATH

The `grid_risk` package lives under `src/`. All scripts require this to be set:
```bash
# Mac/Linux
export PYTHONPATH=src

# Windows PowerShell
$env:PYTHONPATH="src"
```

> **Note:** You need to re-run this export each time you open a new terminal. To make it permanent, add it to your `~/.zshrc` or `~/.bashrc`.

---

## Running the Full Pipeline

Run these steps in order. Each step produces output files consumed by the next.

### Phase 1 — Data Extraction

Fetches daily data from the **REE public API** and **Open-Meteo** weather archive. No API keys or authentication required.
```bash
python scripts/run_extraction.py

# Optional: custom date range
python scripts/run_extraction.py --start 2021-01-01 --end 2025-12-31
```

⏳ Takes **10–15 minutes** due to REE API rate limiting (1 request/second).

**Output:** `data/merged_daily_data.parquet`

Data streams fetched:
- REE demand (hourly → daily mean): actual and forecasted demand in MW
- REE generation mix (daily): wind, solar PV, hydro, combined cycle, nuclear
- REE spot price (hourly → daily mean): EUR/MWh
- Open-Meteo weather archive (daily): temperature, wind speed, solar radiation, precipitation

---

### Phase 2 — Feature Engineering

Computes the three core risk factors, fits the PCA-based Grid Risk Index on training data only, assigns risk categories, and adds lag features.
```bash
python scripts/run_features.py
```

**Output:**
```
data/train.parquet                    # training set  (≤ 2023-12-31)
data/val.parquet                      # validation set (2024-01-01 to 2024-06-30)
data/test.parquet                     # test set       (2024-07-01 onward)
data/daily_features.parquet           # full feature matrix
data/artifacts/risk_index_fit.joblib  # fitted PCA pipeline + thresholds
```

The chronological split is strict — no shuffling — to prevent data leakage.

---

### Phase 3 — Model Training

Trains three models in sequence (persistence baseline → Ridge baseline → LightGBM with Optuna tuning) and evaluates on the held-out chronological test set.
```bash
# Full training with 30 Optuna hyperparameter trials (recommended)
python scripts/train_model.py --n-trials 30

# Fast run, skip Optuna tuning
python scripts/train_model.py --skip-tuning

# Custom quantile alpha (default 0.90)
python scripts/train_model.py --quantile-alpha 0.95
```

⏳ With `--n-trials 30`, takes **5–10 minutes** depending on your machine.

**Output:**
```
data/artifacts/lgbm_model.joblib           # trained LightGBM model
data/artifacts/ridge_model.joblib          # Ridge baseline
data/artifacts/feature_importance.csv      # feature importance scores
data/artifacts/metrics.json                # all evaluation metrics
data/artifacts/test_predictions.parquet    # test set predictions
run_id.txt                                 # MLflow run ID
mlflow.db                                  # MLflow SQLite tracking database
```

Evaluation metrics printed to terminal:
- **RMSE Skill Score** — % improvement over persistence baseline
- **Cost Matrix Penalty** — FN = 10 pts, FP = 1 pt asymmetric cost
- **F₃ Score (Extreme class)** — recall weighted 9× over precision
- **RMSE / MAE / R²** — standard regression metrics

---

### Copy artifacts for Docker

After training, copy the model artifacts into `models/` so Docker can bake them into the image:
```bash
mkdir -p models
cp data/artifacts/lgbm_model.joblib models/
cp data/artifacts/risk_index_fit.joblib models/
```

---

## Running with Docker

### Build the image
```bash
docker build -t grid-risk-app .
```

### Run the container
```bash
docker run -d -p 9696:9696 --name grid-risk grid-risk-app
```

### Verify it started correctly
```bash
curl http://localhost:9696/health
```

Expected response:
```json
{
  "status": "ok",
  "run_id": "b3bac1df...",
  "model_loaded": true,
  "risk_fit_loaded": true
}
```

### Make a prediction
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"target_date": "2026-03-15"}'
```

Expected response:
```json
{
  "date": "2026-03-15",
  "risk_index": 0.6231,
  "risk_category": "Elevated",
  "model_version": "b3bac1df"
}
```

### Stop and remove the container
```bash
docker stop grid-risk && docker rm grid-risk
```

---

## Web App

A lightweight frontend is included at `webapp/index.html`. It requires no server — open it directly in a browser while the Docker container is running.
```bash
# Mac
open webapp/index.html

# Windows
start webapp/index.html

# Linux
xdg-open webapp/index.html
```

The app connects to `http://localhost:9696` by default. Select a target date and click **Predict Risk** to get a live prediction with a visual risk gauge and colour-coded category badge.

To point the webapp at the production Render deployment, update the default value in `webapp/index.html`:
```html
<input type="text" id="apiUrl" value="https://spain-grid-risk.onrender.com" />
```

---

## API Reference

Interactive documentation is available at `http://localhost:9696/docs` (Swagger UI) when the server is running.

### `GET /`

Returns a welcome message and link to the docs.

### `GET /health`

Returns model load status. Used for liveness and readiness checks.

**Response:**
```json
{
  "status": "ok",
  "run_id": "b3bac1df...",
  "model_loaded": true,
  "risk_fit_loaded": true
}
```

### `POST /predict`

Fetches the last 10 days of live data from REE and Open-Meteo, computes features, and returns a risk prediction for the target date.

**Request body:**
```json
{
  "target_date": "2026-03-15"
}
```

**Response:**
```json
{
  "date": "2026-03-15",
  "risk_index": 0.6231,
  "risk_category": "Elevated",
  "model_version": "b3bac1df"
}
```

Response fields:
- `date` — the date predicted for (ISO 8601)
- `risk_index` — predicted quantile risk score, float between 0.0 and 1.0
- `risk_category` — one of: Low / Stable / Elevated / Severe / Extreme
- `model_version` — MLflow run ID of the model in production

Error responses:
- `422` — invalid date format
- `503` — model not loaded, check `/health`
- `502` — could not fetch data from REE API
- `500` — prediction failed, see server logs

> **Note:** Each `/predict` call fetches live data from external APIs. Expect a response time of 5–15 seconds.

---

## CI/CD Pipeline

The pipeline runs automatically on every push to `main` via GitHub Actions.
```
Push to main
    │
    ▼
[train.yml] Train model on sample data
            python train.py --skip-tuning
    │
    ▼
[ci-cd.yml] Download trained model artifact
    │
    ├── Lint with flake8
    │   (app.py, train.py, test_api.py, src/)
    │
    ├── Build Docker image
    │   (model artifacts baked in)
    │
    ├── Start container → curl /health → pytest test_api.py
    │
    └── Push to GitHub Container Registry (GHCR)
            ghcr.io/nicolaswilches/spain-energy-risk-index:latest
            ghcr.io/nicolaswilches/spain-energy-risk-index:<commit-sha>
```

To trigger the pipeline manually without a push:
1. Go to your GitHub repo → **Actions** tab
2. Select **CI/CD Pipeline**
3. Click **Run workflow**

---

## MLflow Experiment Tracking

All training runs are tracked locally with MLflow using a SQLite backend.

### View the MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000
```

What is tracked per run:
- **Parameters** — `quantile_alpha`, `n_optuna_trials`, `train_rows`, `val_rows`, `test_rows`, best Optuna hyperparams
- **Baselines** — `persistence_val_rmse`, `persistence_test_rmse`, `ridge_val_rmse`, `ridge_test_rmse`
- **Model metrics** — `test_rmse`, `test_mae`, `test_r2`, `test_cost_penalty`, `test_cost_reduction_pct`, `test_f3_high`, `test_rmse_skill_score`
- **Artifacts** — trained LightGBM model logged via `mlflow.sklearn.log_model`

---

## Running Tests

Integration tests require the API server to be running.

### Start the server
```bash
docker run -d -p 9696:9696 --name grid-risk grid-risk-app
```

### Run all tests
```bash
pytest test_api.py -v
```

What is tested:
- `test_root_endpoint` — GET `/` returns 200 with message containing "Spain"
- `test_health_endpoint` — GET `/health` returns `status: ok`, both model and risk_fit loaded
- `test_predict_endpoint` — POST `/predict` returns valid `risk_index` in [0, 1] and a valid category
- `test_predict_invalid_date` — POST `/predict` with a bad date returns 422

> `test_predict_endpoint` makes a live call to REE. It is excluded in CI with `pytest -k "not test_predict_endpoint"` to avoid external API dependency in automated runs.

---

## Deployment on Render

The `render.yaml` manifest configures deployment on [Render.com](https://render.com) from the GHCR image.

### First-time setup

1. Go to [render.com](https://render.com) and create an account
2. Click **New → Web Service**
3. Choose **Deploy an existing image**
4. Set image URL: `ghcr.io/nicolaswilches/spain-energy-risk-index:latest`
5. Set port: `9696`
6. Set health check path: `/health`
7. Click **Deploy**

### Subsequent deployments

After each push to `main`, CI/CD pushes a new `:latest` image to GHCR. To deploy the updated image on Render, click **Manual Deploy → Deploy latest commit** in the Render dashboard.

Environment variable to set in Render dashboard:
- `PORT` = `9696`

---

## Best Practices Checklist

- **Modularization** — `src/grid_risk/` split into 7 focused modules: `api_client`, `extractors`, `cleaning`, `pipeline`, `config`, `features`, `model`
- **Configuration management** — all constants, API URLs, column mappings, and thresholds centralized in `src/grid_risk/config.py`
- **Logging and monitoring** — `logging` throughout all modules and `app.py`; MLflow tracks all experiments, metrics, and model artifacts
- **Error handling** — `try/except` throughout `app.py`; `HTTPException` with meaningful status codes (422, 500, 502, 503); `ValidationReport` dataclass in `cleaning.py`
- **Dependency management** — all dependencies pinned with exact versions in `requirements.txt` for full reproducibility
- **Documentation** — docstrings on all public functions and classes; `LESSONS.md` for engineering patterns; this README
- **Code quality** — PEP 8 enforced via `flake8` (configured in `.flake8`); runs automatically in every CI build
- **Testing** — integration test suite in `test_api.py` covering all 4 endpoints; runs in CI against a live Docker container
- **Security** — `.env` excluded in `.gitignore`; no secrets or API keys in source code; REE API is fully public with no authentication

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'grid_risk'` — run `export PYTHONPATH=src` (Mac/Linux) or `$env:PYTHONPATH="src"` (Windows)
- Docker port already in use — run `docker stop grid-risk && docker rm grid-risk` then re-run
- CORS error in browser when using webapp — ensure `CORSMiddleware` is added in `app.py` and the Docker image is rebuilt
- GitHub push asks for password — use a Personal Access Token (PAT) with `repo` scope instead of your password
- Render shows unhealthy — confirm `curl http://localhost:9696/health` returns `"status": "ok"` locally first
- `models/` folder missing — run `mkdir -p models && cp data/artifacts/lgbm_model.joblib models/ && cp data/artifacts/risk_index_fit.joblib models/`
- REE API timeout during extraction — the script retries automatically up to 3 times with exponential backoff; re-run if it still fails
- Negative RMSE skill score — expected with quantile regression (α = 0.90), the model intentionally overshoots; use cost penalty and F₃ to evaluate instead