# Spain Energy Grid Risk Index - Execution Plan

## 1. Project Overview & Architecture

This project aims to forecast a 24-hour profile of the "Grid Risk Index" for the Spanish electrical grid (REE) using LightGBM. The system will be deployed locally using FastAPI, Streamlit, and Docker.

### 1.1 Key Architectural Decisions

* **Granularity:** Hourly predictions (`time_trunc=hour` via REE API) instead of 5-minute or daily aggregations, preventing the dilution of acute stress spikes.
* **Output:** A 24-hour risk profile (predicting hours 00:00 through 23:00 for the next day) rather than a single daily maximum.
* **Forecasting Strategy:** We bypass recursive "error accumulation" by using **Exogenous Day-Ahead Forecasts**. The model maps REE's official day-ahead predictions (Demand, Wind, Solar) at hour $t$, combined with historical lagged risk and calendar features, directly to the Grid Risk Index at hour $t$.

---

## 2. Phase 1: Data Engineering & API Integration

**Objective:** Fetch, clean, and align data using the REE API (`https://www.ree.es/es/datos/apidatos`).

* **API Strategy:** Implement a paginated/chunked Python extraction script using the `time_trunc=hour` parameter to avoid server timeouts and minimize data payload.
* **Required Data Points (Historical Actuals):**
  * Actual Demand (MW)
  * Actual Generation by Technology (Wind, Solar, Hydro, Combined-Cycle, Nuclear)
* **Required Data Points (Exogenous Features):**
  * Day-Ahead Demand Forecast (MW)
  * Day-Ahead Wind Generation Forecast (MW)
  * Day-Ahead Solar Generation Forecast (MW)
* **Outputs:** A single, strictly chronological, hourly Pandas/Polars DataFrame. Missing values must be imputed (forward-fill for small gaps), and timestamps localized to CET/CEST.

---

## 3. Phase 2: Target Creation & Feature Engineering

**Objective:** Calculate the "Ground Truth" Risk Index ($Y$) and engineer the predictive features ($X$) without introducing data leakage.

### 3.1 Chronological Split (CRITICAL)

Before any scaling or PCA, split the dataset chronologically:

* **Train Set:** e.g., 2021-01-01 to 2023-12-31
* **Validation Set:** e.g., 2024-01-01 to 2024-06-30 (for hyperparameter tuning)
* **Test Set:** e.g., 2024-07-01 to Present (locked away for final evaluation)

### 3.2 Target Variable Creation ($Y$)

1. **Core Factors:** Calculate the 3 operational stress variables for all rows based on *Historical Actuals*:
    * `Flexibility_Share` = (Combined-Cycle + Hydro) / Total Generation
    * `Demand_Forecast_Error` = Actual Demand - Forecasted Demand
    * `Net_Load` = Actual Demand - (Actual Wind + Actual Solar)
2. **PCA Indexing:**
    * Fit a StandardScaler and Principal Component Analysis (PCA) **ONLY** on the Train Set's 3 core factors.
    * Extract the weights from PC1.
    * Transform the Train, Validation, and Test sets using the fitted PCA to create a continuous `Risk_Index` (scaled between 0 and 1).
    * *Categorical mapping:* Determine thresholds for *Low*, *Medium*, and *High* risk based on percentiles in the Train set.

### 3.3 Predictive Features ($X$)

Engineer the features the model will use to predict the target:

1. **Exogenous Features:** REE's Day-Ahead Forecasts for Demand, Wind, and Solar at hour $t$.
2. **Calendar Features:** `Hour_of_Day` (0-23), `Day_of_Week` (0-6), `Month` (1-12), `Is_Weekend` (0/1).
3. **Lagged Features:** `Risk_Index_Lag_24h` (the Risk Index exactly 24 hours prior) and `Risk_Index_Lag_168h` (1 week prior).

---

## 4. Phase 3: Model Training & Evaluation (Asymmetric Strategy)

**Objective:** Train LightGBM with **quantile regression (α=0.90)** — the model predicts the 90th percentile of risk, making it pessimistic because missing a blackout (FN) costs 10× more than a false alarm (FP). Standard RMSE is demoted; four custom business metrics drive evaluation.

* **Baselines:** Persistence baseline ($t_{+24} = t_{0}$) and Ridge Regression. Document RMSE/MAE for reference.
* **LightGBM Configuration:**
  * **Objective:** `quantile` with `alpha=0.90` (not standard `regression`)
  * 1000 estimators, lr=0.05, 31 leaves, early stopping (patience=50)
  * Optuna hyperparameter tuning (30 trials) minimizing **validation quantile loss** (not RMSE)
  * Search space: `reg_alpha`, `reg_lambda` (log 1e-3..10), `num_leaves` (15..63), `learning_rate` (0.01..0.1), `min_child_samples` (10..50)
* **4 Business Metrics (Primary):**
  * **A. Peak Window Accuracy (±1h):** % of days where predicted peak hour is within ±1h of actual
  * **B. Cost Matrix Penalty:** FN(High) = 10 pts, FP(High) = 1 pt — total penalty (must beat persistence)
  * **C. F₃ Score (High class):** F-beta with β=3, recall weighted 9× over precision (target: >0.70)
  * **D. RMSE Skill Score:** % improvement over persistence (expected negative for quantile model — by design)
* **Standard Metrics (Secondary):** RMSE, MAE, R² — tracked for reference but not optimized
* **Artifact Tracking:** `metrics.json` + joblib artifacts (MLflow dropped — lightweight file-based tracking suffices)
* **CLI:** `scripts/train_model.py --n-trials 30 --quantile-alpha 0.90 [--skip-tuning]`

---

## 5. Phase 4: MLOps Packaging & Deployment

**Objective:** Containerize the trained model into a local prediction system using FastAPI and Streamlit.

### 5.1 Artifact Export

Save the finalized LightGBM model, the StandardScaler, and the PCA object (e.g., using `joblib`).

### 5.2 Backend (FastAPI)

* Build `app.py` using FastAPI.
* Create a `/predict` endpoint.
* **Workflow:** Endpoint receives a JSON payload containing tomorrow's REE day-ahead forecasts (24 hours). The endpoint fetches today's actual Risk Index to use as the 24h lag, computes calendar features, passes the array to the LightGBM model, and returns a JSON response with 24 Hourly Risk Index predictions.

### 5.3 Frontend (Streamlit)

* Build `dashboard.py`.
* Connect it to the FastAPI `/predict` endpoint.
* **UI Elements:** Display a 24-hour line chart showing tomorrow's Risk Index profile. Highlight hours that cross the *High* risk threshold. Include a visual breakdown of LightGBM's feature importance (e.g., via SHAP values) to explain *why* the risk is high.

### 5.4 Dockerization

* Create a `Dockerfile` for the FastAPI backend.
* Create a `Dockerfile` for the Streamlit frontend.
* Create a `docker-compose.yml` at the project root to orchestrate both containers, ensuring they communicate over a shared local Docker network.
