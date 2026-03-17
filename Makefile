.PHONY: setup extract features train train-fast promote docker-build docker-run docker-stop test clean

## Environment
setup:
	python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

## Pipeline

# Phase 1: Fetch data from REE + Open-Meteo (takes 10–15 min due to rate limiting)
extract:
	PYTHONPATH=src python scripts/run_extraction.py

# Phase 2: Compute risk index and build feature matrix
features:
	PYTHONPATH=src python scripts/run_features.py

# Phase 3: Train LightGBM with Optuna tuning (recommended, ~5–10 min)
train:
	PYTHONPATH=src python train.py --n-trials 30

# Phase 3 (fast): Train without Optuna tuning (~30 sec)
train-fast:
	PYTHONPATH=src python train.py --skip-tuning

# Copy trained artifacts from data/artifacts/ into models/ for Docker baking
promote:
	@mkdir -p models
	cp data/artifacts/lgbm_model.joblib models/
	cp data/artifacts/risk_index_fit.joblib models/
	cp data/artifacts/metrics.json models/
	cp data/artifacts/feature_importance.csv models/
	@echo "Artifacts promoted to models/"

## Docker
docker-build:
	docker build -t grid-risk-app .

docker-run:
	docker run -d -p 9696:9696 --name grid-risk grid-risk-app

docker-stop:
	docker stop grid-risk && docker rm grid-risk

## Testing
test:
	pytest tests/ -v

## Cleanup
clean:
	docker stop grid-risk && docker rm grid-risk 2>/dev/null || true
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
