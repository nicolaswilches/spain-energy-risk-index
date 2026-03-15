# Tasks — Spain Energy Grid Risk Index

## Done

- [x] **Phase 1 (v1): Data Engineering — Hourly** — REE API pipeline, ~45k hourly rows (2021–present), <0.5% missing
- [x] **Phase 2 (v1): Feature Engineering — Hourly** — PCA-based risk index [0,1], 9 features, hourly split
- [x] **Phase 3 (v1): Model Training — Hourly** — Quantile LightGBM (α=0.90), 41.6% cost reduction, F₃=0.781
- [x] **Phase 1 (v2): Data Engineering — Daily Pivot** — 4 data streams (REE + Open-Meteo + Calendar + Spot Price/HDD/CDD), ~1,898 daily rows, <0.1% missing
- [x] **Phase 2 (v2): Feature Engineering — Daily** — PCA risk index, 15 features (weather, calendar, spot price, lags), train=1,088/val=182/test=620, 5 risk categories (Low/Stable/Elevated/Severe/Extreme)
- [x] **Phase 3 (v2): Model Training — Daily with MLflow** — Quantile LightGBM, 56.2% cost reduction, F₃=0.831, MLflow experiment tracking
- [x] **Consolidated Notebook** — etl_modeling.ipynb covering all 3 phases with Plotly visualizations + SHAP
- [x] **Phase 4: FastAPI Backend** — `/predict` endpoint (date-based, fetches live REE + Open-Meteo data), `/health`, `/` — all 4 integration tests pass
- [x] **Phase 5: Dockerization** — python:3.12-slim + libgomp1, model baked in, port 9696 — container tested locally
- [x] **Phase 6: CI/CD** — GitHub Actions: train.yml (reusable) + ci-cd.yml (train -> lint -> Docker -> test -> GHCR push)
- [x] **Phase 7: Deployment** — render.yaml pulling from ghcr.io

## Pending

- [ ] **README.md** — Setup, workflow, and usage documentation
