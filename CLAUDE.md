# CLAUDE.md — Spain Energy Grid Risk Index

## Overview

Hourly risk forecasting for the Spanish electrical grid (REE). A quantile LightGBM model predicts the 90th percentile of a PCA-derived risk index, prioritizing safety: missing a blackout (FN) costs 10× more than a false alarm (FP).

## Project Structure

```
spain-energy-risk-index/
├── src/grid_risk/            # Core library (pure logic, no I/O)
│   ├── api_client.py         # REE API client with retry/pagination
│   ├── extractors.py         # Data extraction for each REE endpoint
│   ├── cleaning.py           # Timestamp alignment, imputation, QA
│   ├── pipeline.py           # Orchestrates extraction → cleaning
│   ├── config.py             # API URLs, date ranges, column mappings
│   ├── features.py           # Phase 2: risk index, features, splits
│   └── model.py              # Phase 3: training, metrics, evaluation
├── scripts/
│   ├── run_extraction.py     # CLI: fetch raw data from REE APIs
│   ├── run_features.py       # CLI: build features + train/val/test split
│   └── train_model.py        # CLI: baselines → Optuna → LightGBM → evaluate
├── notebooks/
│   ├── S01_data_exploration.ipynb
│   ├── S02_feature_engineering.ipynb
│   └── S03_model_evaluation.ipynb
├── data/
│   ├── spain_grid_hourly.parquet   # Raw hourly data (~45k rows)
│   ├── spain_grid_features.parquet # Full feature matrix
│   ├── train.parquet               # 2021-01 to 2023-12
│   ├── val.parquet                 # 2024-01 to 2024-06
│   ├── test.parquet                # 2024-07 to present
│   └── artifacts/
│       ├── risk_index_fit.joblib   # Fitted PCA pipeline (scaler, PCA, minmax, thresholds)
│       ├── lgbm_model.joblib       # Trained quantile LightGBM
│       ├── ridge_model.joblib      # Ridge baseline
│       ├── metrics.json            # All evaluation metrics
│       └── feature_importance.csv
├── execution_plan.md
├── project_guidelines.pdf
├── project_proposal/
├── pyproject.toml
└── .gitignore
```

---

## New Task Workflow

### 1. Identify

- Simple tasks are short, one-step, non-ambiguous nor expensive requests.
- Complex tasks require more than 3 steps, architectural and logical changes.
- If task complexity is ambiguous, ask.

### 1. Plan

- For complex tasks: Enter plan mode.
- For simple tasks no plan is required.
- Ask follow-up questions if instructions are not rational, logical and clear.
- Ask yourself: "what's the most efficient and elegant way to solve this task?"
- For complex tasks: write detailed final execution plan upfront to avoid ambiguity.

### 2. Execute

- For simple tasks: execute.
- For complex tasks: consider subagents to keep main context window clean.
- For complex tasks: Offload research, exploration and parallel analysis to subagents.
- For complex tasks: Assign only one task to one subagent for focused execution.
- If you encounter an error look at `LESSONS.md`

### 4. Verify

- For complex tasks: Use plan mode for verification steps.
- For complex tasks: Run appropriate tests to demonstrate completeness and correctness of your work.
- Never mark a task as complete without proving completeness and correctness.
- Report tests have been ran successfully.

### 5. Notify and log

- Once verified, deliver a brief description of what you achieved. Simple and clear.
- Once verified, mark the task a complete in `TASKS.md`
- If current plan is complete, update `TASKS.md` and flag pending tasks `TASKS.md`

### 6. Learn

- If an error is recurring: flag it, identify the best solution, log the problem and the solution to `LESSONS.md`. Keep short and direct.
- After any correction from the user: update `LESSONS.md` with the pattern and the solution. Keep short and direct.

---

## Architecture

### Data Pipeline (Phase 1)

REE API → `api_client.py` (paginated, retry with tenacity) → `extractors.py` (per-endpoint) → `cleaning.py` (align, impute, QA) → `spain_grid_hourly.parquet`

- **Granularity:** Hourly (`time_trunc=hour`) — preserves acute stress spikes
- **Coverage:** ~5 years (2021–present), ~45k rows, <0.5% missing
- **Timezone:** UTC index, Europe/Madrid for calendar features

### Target & Features (Phase 2)

Three core risk factors (computed from actuals):

- `flexibility_share` = (combined_cycle + hydro) / total_generation
- `demand_forecast_error` = actual_demand − forecast_demand
- `net_load` = actual_demand − (wind + solar_pv)

**Risk Index:** StandardScaler → PCA(3) → PC1 → MinMaxScaler → [0, 1]

- Fitted on train only; thresholds at p33/p67 → Low/Medium/High categories
- Thresholds: Low ≤ 0.410, Medium ≤ 0.587, High > 0.587

**9 Features (X):**

1. `forecast_demand_mw` — REE day-ahead demand forecast
2. `forecast_wind_mw` — REE day-ahead wind forecast
3. `forecast_solar_mw` — REE day-ahead solar forecast
4. `hour_of_day`, `day_of_week`, `month`, `is_weekend` — calendar (Europe/Madrid)
5. `risk_index_lag_24h`, `risk_index_lag_168h` — temporal lags

**Chronological split:** train (→2023-12), val (2024-01→2024-06), test (2024-07→present)

### Model (Phase 3)

**Key decision: Quantile regression (α=0.90)** — the model predicts the 90th percentile of risk, making it systematically pessimistic. Standard RMSE is demoted; four business metrics drive evaluation.

**Training pipeline:**

1. Persistence baseline (24h lag)
2. Ridge baseline
3. Optuna hyperparameter search (30 trials, minimizes quantile loss on val)
4. Final LightGBM with best params + early stopping (patience=50)

**4 Business Metrics:**

- **A. Peak Window Accuracy (±1h):** % of days where predicted peak hour matches actual within ±1h
- **B. Cost Matrix Penalty:** FN(High)=10pts, FP(High)=1pt — total penalty
- **C. F₃ Score (High class):** recall weighted 9× over precision (β=3)
- **D. RMSE Skill Score:** % improvement over persistence baseline

**Latest results (test set):**

- Cost reduction: 41.6% vs persistence
- F₃ = 0.781 (strong High-class recall)
- Near-zero High→Low misclassification

## Key Conventions

- **Pure logic in `src/`** — no file I/O in module functions; scripts handle I/O
- **PYTHONPATH=src** — required when running scripts (`PYTHONPATH=src uv run python scripts/...`)
- **joblib for serialization** — RiskIndexFit dataclass requires `grid_risk` on the import path
- **Notebooks** numbered S01, S02, S03... — each maps to a project phase
- **uv** as package manager — `pyproject.toml` at project root

## Running

```bash
# Full training pipeline (30 Optuna trials)
PYTHONPATH=src uv run python scripts/train_model.py --n-trials 30

# Quick run without tuning
PYTHONPATH=src uv run python scripts/train_model.py --skip-tuning

# Custom quantile alpha
PYTHONPATH=src uv run python scripts/train_model.py --quantile-alpha 0.95
```
