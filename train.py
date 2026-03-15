"""
Unified Training Pipeline for Spain Energy Grid Risk Index.

Usage:
    python train.py                 # Train with sample data (CI/CD)
    python train.py --full          # Train with full data (Local production)
    python train.py --skip-tuning   # Skip Optuna hyperparameter searching
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from grid_risk.features import (
    assign_risk_category,
    feature_names,
    RiskIndexFit
)

from grid_risk.model import (
    cost_matrix_penalty,
    create_optuna_objective,
    evaluate,
    get_feature_importance,
    persistence_baseline,
    predict,
    ridge_baseline,
    train_lightgbm,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACTS = DATA / "artifacts"
DEPLOYMENT_MODEL_DIR = ROOT / "models"

MLFLOW_EXPERIMENT = "spain-grid-risk"


def load_data(full: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train/val/test splits.
    """
    prefix = "" if full else "sample_"
    train = pd.read_parquet(DATA / f"{prefix}train.parquet")
    val = pd.read_parquet(DATA / f"{prefix}val.parquet")
    test = pd.read_parquet(DATA / f"{prefix}test.parquet")
    print(f"Data loaded: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def fit_risk_index_from_train(train: pd.DataFrame) -> RiskIndexFit:
    """
    Fit the PCA-based risk index pipeline from training data.
    """
    factor_cols = ["flexibility_share", "demand_forecast_error", "net_load"]
    factors = train[factor_cols].dropna()

    scaler = StandardScaler().fit(factors)
    scaled = scaler.transform(factors)

    pca = PCA(n_components=3).fit(scaled)
    pc1 = pca.transform(scaled)[:, 0].reshape(-1, 1)

    minmax = MinMaxScaler().fit(pc1)
    risk_train = minmax.transform(pc1).ravel()

    p20 = float(np.percentile(risk_train, 20))
    p40 = float(np.percentile(risk_train, 40))
    p60 = float(np.percentile(risk_train, 60))
    p80 = float(np.percentile(risk_train, 80))

    return RiskIndexFit(
        scaler=scaler,
        pca=pca,
        minmax=minmax,
        pc1_weights=pca.components_[0],
        thresholds=(p20, p40, p60, p80),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train grid risk model")
    parser.add_argument("--full", action="store_true", help="Use full dataset")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials")
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--quantile-alpha", type=float, default=0.90)
    args = parser.parse_args()

    train, val, test = load_data(full=args.full)

    # ------------------------------------------------------------------
    # Fit or load Risk Index
    # ------------------------------------------------------------------
    risk_fit_path = ARTIFACTS / "risk_index_fit.joblib"
    if args.full and risk_fit_path.exists():
        fit: RiskIndexFit = joblib.load(risk_fit_path)
        print("Loaded existing risk_index_fit from artifacts")
    else:
        fit = fit_risk_index_from_train(train)
        print("Fitted risk index from training data")

    thresholds = fit.thresholds
    X_train = train[feature_names]
    y_train = train["risk_index"].values
    X_val = val[feature_names]
    y_val = val["risk_index"].values
    X_test = test[feature_names]
    y_test = test["risk_index"].values

    # ------------------------------------------------------------------
    # MLflow setup (local file-based)
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow run ID: {run_id}")

        mlflow.log_param("quantile_alpha", args.quantile_alpha)
        mlflow.log_param("n_optuna_trials", args.n_trials if not args.skip_tuning else 0)
        mlflow.log_param("dataset", "full" if args.full else "sample")
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("val_rows", len(val))
        mlflow.log_param("test_rows", len(test))

        # --------------------------------------------------------------
        # 1. Baselines
        # --------------------------------------------------------------
        print("\n-- Baseline Comparisons --")
        persist_test = persistence_baseline(y_test, test["risk_index_lag_1d"].values)
        _, ridge_test = ridge_baseline(X_train.values, y_train, X_test.values, y_test)

        mlflow.log_metric("persistence_test_rmse", persist_test.rmse)
        mlflow.log_metric("ridge_test_rmse", ridge_test.rmse)

        print(f"  Persistence Test RMSE: {persist_test.rmse:.4f}")
        print(f"  Ridge Test RMSE:       {ridge_test.rmse:.4f}")

        # --------------------------------------------------------------
        # 2. Optuna tuning
        # --------------------------------------------------------------
        best_params: dict = {}
        if not args.skip_tuning:
            print(f"\n-- Optuna Tuning ({args.n_trials} trials) --")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize")
            objective = create_optuna_objective(
                X_train.values, y_train, X_val.values, y_val, quantile_alpha=args.quantile_alpha
            )
            study.optimize(objective, n_trials=args.n_trials)
            best_params = study.best_params
            mlflow.log_metric("best_quantile_loss", study.best_value)
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # --------------------------------------------------------------
        # 3. Train final LightGBM
        # --------------------------------------------------------------
        print("\n-- Training Final LightGBM --")
        lgbm_model = train_lightgbm(
            X_train.values, y_train, X_val.values, y_val,
            params=best_params, quantile_alpha=args.quantile_alpha
        )
        mlflow.log_metric("best_iteration", lgbm_model.best_iteration_)

        # --------------------------------------------------------------
        # 4. Evaluate on test set
        # --------------------------------------------------------------
        y_pred_test = predict(lgbm_model, X_test.values)
        y_baseline_test = test["risk_index_lag_1d"].values

        mask = ~np.isnan(y_baseline_test)
        metrics = evaluate(
            y_true=y_test[mask],
            y_pred=y_pred_test[mask],
            y_baseline=y_baseline_test[mask],
            thresholds=thresholds,
        )

        base_cat = assign_risk_category(y_baseline_test[mask], thresholds)
        true_cat = assign_risk_category(y_test[mask], thresholds)
        baseline_cost = cost_matrix_penalty(true_cat, base_cat)
        cost_reduction = (
            (1 - metrics.cost_penalty / baseline_cost) * 100
        ) if baseline_cost > 0 else 0.0

        mlflow.log_metric("test_rmse", metrics.rmse)
        mlflow.log_metric("test_mae", metrics.mae)
        mlflow.log_metric("test_cost_penalty", metrics.cost_penalty)
        mlflow.log_metric("test_cost_reduction_pct", cost_reduction)
        mlflow.log_metric("test_f3_high", metrics.f3_extreme)
        mlflow.log_metric("test_rmse_skill_score", metrics.rmse_skill_score)

        mlflow.sklearn.log_model(lgbm_model, "model")

        print("\n-- Business Metrics (Test Set) --")
        print(
            f"  Cost Matrix Penalty:  {metrics.cost_penalty} pts  "
            f"(reduction: {cost_reduction:.1f}%)"
        )
        print(f"  F3 Score (Extreme):   {metrics.f3_extreme:.3f}")
        print("\n-- Standard Metrics --")
        print(f"  RMSE: {metrics.rmse:.4f}  |  MAE: {metrics.mae:.4f}")

    # ------------------------------------------------------------------
    # 5. Save artifacts to models/ for CI deployment
    # ------------------------------------------------------------------
    if DEPLOYMENT_MODEL_DIR.exists():
        shutil.rmtree(DEPLOYMENT_MODEL_DIR)
    DEPLOYMENT_MODEL_DIR.mkdir(parents=True)

    joblib.dump(lgbm_model, DEPLOYMENT_MODEL_DIR / "lgbm_model.joblib")
    joblib.dump(fit, DEPLOYMENT_MODEL_DIR / "risk_index_fit.joblib")

    feat_imp = get_feature_importance(lgbm_model, feature_names)
    feat_imp.to_csv(DEPLOYMENT_MODEL_DIR / "feature_importance.csv", index=False)

    metrics_dict = {
        "run_id": run_id,
        "quantile_alpha": args.quantile_alpha,
        "dataset": "full" if args.full else "sample",
        "best_params": best_params,
        "test": {
            "rmse": metrics.rmse,
            "cost_penalty": metrics.cost_penalty,
            "cost_reduction_pct": cost_reduction,
            "f3_high": metrics.f3_extreme,
            "rmse_skill_score": metrics.rmse_skill_score,
        },
    }
    (DEPLOYMENT_MODEL_DIR / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))

    # Save the risk model mapping out data predictions locally (optional dev step)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    test_preds = pd.DataFrame({
        "y_true": y_test[mask],
        "y_pred": y_pred_test[mask],
        "y_baseline": y_baseline_test[mask]
    })
    test_preds.to_parquet(ARTIFACTS / "test_predictions.parquet")

    # App root identifier
    (ROOT / "run_id.txt").write_text(run_id)

    print(f"\nTraining complete. Artifacts saved to {DEPLOYMENT_MODEL_DIR}/")


if __name__ == "__main__":
    main()
