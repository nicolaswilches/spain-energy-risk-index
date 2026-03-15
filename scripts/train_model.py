"""Phase 3 — Train baselines + LightGBM quantile model, evaluate, save artifacts.

Tracks experiments with MLflow (local file-based backend).

Usage:
    PYTHONPATH=src uv run python scripts/train_model.py
    PYTHONPATH=src uv run python scripts/train_model.py --n-trials 30
    PYTHONPATH=src uv run python scripts/train_model.py --skip-tuning
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd

from grid_risk.features import (
    assign_risk_category,
    feature_names as FEATURE_NAMES,
    RiskIndexFit,
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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = DATA / "artifacts"

MLFLOW_EXPERIMENT = "spain-grid-risk"


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"{name}.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 3 models")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--skip-tuning", action="store_true")
    parser.add_argument("--quantile-alpha", type=float, default=0.90)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading data...")
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    fit: RiskIndexFit = joblib.load(ARTIFACTS / "risk_index_fit.joblib")
    thresholds = fit.thresholds

    X_train = train[FEATURE_NAMES]
    y_train = train["risk_index"].values
    X_val = val[FEATURE_NAMES]
    y_val = val["risk_index"].values
    X_test = test[FEATURE_NAMES]
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
        mlflow.log_param(
            "n_optuna_trials", args.n_trials if not args.skip_tuning else 0
        )
        mlflow.log_param("train_rows", len(train))
        mlflow.log_param("val_rows", len(val))
        mlflow.log_param("test_rows", len(test))
        mlflow.log_param("n_features", len(FEATURE_NAMES))

        # --------------------------------------------------------------
        # 1. Persistence baseline (val + test)
        # --------------------------------------------------------------
        print("\n-- Persistence Baseline --")
        persist_val = persistence_baseline(y_val, val["risk_index_lag_1d"].values)
        persist_test = persistence_baseline(y_test, test["risk_index_lag_1d"].values)
        print(f"  Val  RMSE: {persist_val.rmse:.4f}  MAE: {persist_val.mae:.4f}")
        print(f"  Test RMSE: {persist_test.rmse:.4f}  MAE: {persist_test.mae:.4f}")

        mlflow.log_metric("persistence_val_rmse", persist_val.rmse)
        mlflow.log_metric("persistence_test_rmse", persist_test.rmse)

        # --------------------------------------------------------------
        # 2. Ridge baseline (val + test)
        # --------------------------------------------------------------
        print("\n-- Ridge Baseline --")
        ridge_model, ridge_val = ridge_baseline(X_train, y_train, X_val, y_val)
        _, ridge_test = ridge_baseline(X_train, y_train, X_test, y_test)
        print(f"  Val  RMSE: {ridge_val.rmse:.4f}  MAE: {ridge_val.mae:.4f}")
        print(f"  Test RMSE: {ridge_test.rmse:.4f}  MAE: {ridge_test.mae:.4f}")

        mlflow.log_metric("ridge_val_rmse", ridge_val.rmse)
        mlflow.log_metric("ridge_test_rmse", ridge_test.rmse)

        # --------------------------------------------------------------
        # 3. Optuna tuning
        # --------------------------------------------------------------
        best_params: dict = {}
        if not args.skip_tuning:
            print(f"\n-- Optuna Tuning ({args.n_trials} trials) --")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize")
            objective = create_optuna_objective(
                X_train,
                y_train,
                X_val,
                y_val,
                quantile_alpha=args.quantile_alpha,
            )
            study.optimize(objective, n_trials=args.n_trials)
            best_params = study.best_params
            print(f"  Best quantile loss: {study.best_value:.6f}")
            print(f"  Best params: {best_params}")

            mlflow.log_metric("best_quantile_loss", study.best_value)
            mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        # --------------------------------------------------------------
        # 4. Train final LightGBM
        # --------------------------------------------------------------
        print("\n-- Training Final LightGBM --")
        lgbm_model = train_lightgbm(
            X_train,
            y_train,
            X_val,
            y_val,
            params=best_params,
            quantile_alpha=args.quantile_alpha,
        )
        print(f"  Best iteration: {lgbm_model.best_iteration_}")
        mlflow.log_metric("best_iteration", lgbm_model.best_iteration_)

        # --------------------------------------------------------------
        # 5. Evaluate on test set
        # --------------------------------------------------------------
        y_pred_test = predict(lgbm_model, X_test)
        y_baseline_test = test["risk_index_lag_1d"].values

        # Align for baseline comparison (drop NaN in lag)
        mask = ~np.isnan(y_baseline_test)
        metrics = evaluate(
            y_true=y_test[mask],
            y_pred=y_pred_test[mask],
            y_baseline=y_baseline_test[mask],
            thresholds=thresholds,
        )

        # Baseline cost for comparison
        base_cat = assign_risk_category(y_baseline_test[mask], thresholds)
        true_cat = assign_risk_category(y_test[mask], thresholds)
        baseline_cost = cost_matrix_penalty(true_cat, base_cat)

        cost_reduction = (
            (1 - metrics.cost_penalty / baseline_cost) * 100
            if baseline_cost > 0
            else 0.0
        )

        # Log to MLflow
        mlflow.log_metric("test_rmse", metrics.rmse)
        mlflow.log_metric("test_mae", metrics.mae)
        mlflow.log_metric("test_r2", metrics.r2)
        mlflow.log_metric("test_cost_penalty", metrics.cost_penalty)
        mlflow.log_metric("test_baseline_cost", baseline_cost)
        mlflow.log_metric("test_cost_reduction_pct", cost_reduction)
        mlflow.log_metric("test_f3_high", metrics.f3_extreme)
        mlflow.log_metric("test_rmse_skill_score", metrics.rmse_skill_score)

        # Log model
        mlflow.sklearn.log_model(lgbm_model, "model")

        # --------------------------------------------------------------
        # 6. Print results
        # --------------------------------------------------------------
        print("\n-- Business Metrics (Test Set) --")
        print(
            f"  A. RMSE Skill Score:     {metrics.rmse_skill_score:.1f}% "
            f"improvement over persistence"
        )
        print(
            f"  B. Cost Matrix Penalty:  {metrics.cost_penalty} pts  "
            f"(baseline: {baseline_cost} pts, reduction: {cost_reduction:.1f}%)"
        )
        print(f"  C. F3 Score (Extreme):      {metrics.f3_extreme:.3f}")

        print("\n-- Standard Metrics (Test Set) --")
        print(f"  RMSE: {metrics.rmse:.4f}")
        print(f"  MAE:  {metrics.mae:.4f}")
        print(f"  R2:   {metrics.r2:.4f}")

        print("\n-- Confusion Matrix (Low / Stable / Elevated / Severe / Extreme) --")
        labels = ["Low", "Stable", "Elevated", "Severe", "Extreme"]
        header = "            " + "  ".join(f"{lbl:>10s}" for lbl in labels)
        print(header)
        for i, label in enumerate(labels):
            row = "  ".join(f"{metrics.confusion[i, j]:>10d}" for j in range(5))
            print(f"  {label:>10s}  {row}")

        # Store test predictions for notebook analysis
        test_predictions = pd.DataFrame(
            {
                "date": test.index[mask],
                "y_true": y_test[mask],
                "y_pred": y_pred_test[mask],
                "y_baseline": y_baseline_test[mask],
            }
        )
        test_predictions.to_parquet(ARTIFACTS / "test_predictions.parquet")

    # ------------------------------------------------------------------
    # 7. Save artifacts
    # ------------------------------------------------------------------
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    joblib.dump(lgbm_model, ARTIFACTS / "lgbm_model.joblib")
    joblib.dump(ridge_model, ARTIFACTS / "ridge_model.joblib")

    feat_imp = get_feature_importance(lgbm_model, FEATURE_NAMES)
    feat_imp.to_csv(ARTIFACTS / "feature_importance.csv", index=False)

    metrics_dict = {
        "mlflow_run_id": run_id,
        "quantile_alpha": args.quantile_alpha,
        "n_optuna_trials": args.n_trials if not args.skip_tuning else 0,
        "best_params": best_params,
        "test": {
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "r2": metrics.r2,
            "cost_penalty": metrics.cost_penalty,
            "baseline_cost_penalty": baseline_cost,
            "cost_reduction_pct": cost_reduction,
            "f3_high": metrics.f3_extreme,
            "rmse_skill_score": metrics.rmse_skill_score,
        },
        "baselines": {
            "persistence_test_rmse": persist_test.rmse,
            "ridge_test_rmse": ridge_test.rmse,
        },
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))

    # Save run_id for FastAPI serving
    (ROOT / "run_id.txt").write_text(run_id)

    print(f"\n-- Artifacts saved to {ARTIFACTS} --")
    print("  lgbm_model.joblib, ridge_model.joblib, metrics.json,")
    print("  feature_importance.csv, test_predictions.parquet")
    print(f"  MLflow run ID: {run_id}")


if __name__ == "__main__":
    main()
