# Tasks — Spain Energy Grid Risk Index

## Done

- [x] **Phase 1 (v1): Data Engineering — Hourly** — REE API pipeline, ~45k hourly rows (2021–present), <0.5% missing
- [x] **Phase 2 (v1): Feature Engineering — Hourly** — PCA-based risk index [0,1], 9 features, hourly split
- [x] **Phase 3 (v1): Model Training — Hourly** — Quantile LightGBM (α=0.90), 41.6% cost reduction, F₃=0.781
- [x] **Phase 1 (v2): Data Engineering — Daily Pivot** — 4 data streams (REE + Open-Meteo + Calendar + Spot Price/HDD/CDD), ~1,898 daily rows, <0.1% missing
- [x] **Phase 2 (v2): Feature Engineering — Daily** — PCA risk index, 15 features (weather, calendar, spot price, lags), train=1,088/val=182/test=620
- [x] **Phase 3 (v2): Model Training — Daily with MLflow** — Quantile LightGBM, 56.2% cost reduction, F₃=0.831, MLflow experiment tracking
- [x] **Consolidated Notebook** — S01_full_pipeline.ipynb covering all 3 phases with Plotly visualizations + SHAP

## Pending

- [ ] **Phase 4: FastAPI Backend** — `/predict` endpoint serving the trained model
- [ ] **Phase 5: Dockerization** — Dockerfile, containerized service
- [ ] **Phase 6: CI/CD** — GitHub Actions workflow (lint + test + build + push to GHCR)
- [ ] **Phase 7: Deployment** — render.yaml, working online endpoint on Render.com
- [ ] **Phase 8: Streamlit Dashboard** — Interactive UI for risk predictions
- [ ] **README.md** — Setup, workflow, and usage documentation
