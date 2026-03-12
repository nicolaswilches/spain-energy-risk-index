"""Phase 3 — Model training & evaluation with asymmetric quantile strategy (daily)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BaselineResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    rmse: float
    mae: float


@dataclass
class EvalMetrics:
    rmse: float
    mae: float
    r2: float
    cost_penalty: int  # Metric B
    f3_high: float  # Metric C
    rmse_skill_score: float  # Metric A (% improvement)
    confusion: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# 1. Persistence baseline (1-day lag)
# ---------------------------------------------------------------------------


def persistence_baseline(
    y_true: np.ndarray,
    lag_1d: np.ndarray,
) -> BaselineResult:
    """Baseline: predict risk_index = value from 1 day ago."""
    mask = ~(np.isnan(y_true) | np.isnan(lag_1d))
    yt, yp = y_true[mask], lag_1d[mask]
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    return BaselineResult(y_true=yt, y_pred=yp, rmse=rmse, mae=mae)


# ---------------------------------------------------------------------------
# 2. Ridge baseline
# ---------------------------------------------------------------------------


def ridge_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> tuple[Ridge, BaselineResult]:
    """Fit Ridge regression and evaluate."""
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_eval), 0.0, 1.0)
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    mae = float(mean_absolute_error(y_eval, y_pred))
    result = BaselineResult(y_true=y_eval, y_pred=y_pred, rmse=rmse, mae=mae)
    return model, result


# ---------------------------------------------------------------------------
# 3. LightGBM quantile training
# ---------------------------------------------------------------------------


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any] | None = None,
    quantile_alpha: float = 0.90,
):
    """Train LightGBM with quantile regression (default alpha=0.90)."""
    from lightgbm import LGBMRegressor, early_stopping

    defaults: dict[str, Any] = {
        "objective": "quantile",
        "alpha": quantile_alpha,
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbosity": -1,
        "random_state": 42,
        "n_jobs": -1,
    }
    if params:
        defaults.update(params)

    model = LGBMRegressor(**defaults)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )
    return model


# ---------------------------------------------------------------------------
# 4. Optuna objective
# ---------------------------------------------------------------------------


def create_optuna_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    quantile_alpha: float = 0.90,
):
    """Return an Optuna objective closure that minimizes val quantile loss."""

    def objective(trial):  # type: ignore[no-untyped-def]
        params = {
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.1,
                log=True,
            ),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        }
        model = train_lightgbm(
            X_train,
            y_train,
            X_val,
            y_val,
            params=params,
            quantile_alpha=quantile_alpha,
        )
        y_pred = predict(model, X_val)
        return quantile_loss(y_val, y_pred, quantile_alpha)

    return objective


def quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
) -> float:
    """Pinball / quantile loss."""
    residual = y_true - y_pred
    return float(
        np.mean(np.where(residual >= 0, alpha * residual, (alpha - 1) * residual))
    )


# ---------------------------------------------------------------------------
# 5. Predict (clipped)
# ---------------------------------------------------------------------------


def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict and clip to [0, 1]."""
    return np.clip(model.predict(X), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 6. Metric B — Cost matrix penalty
# ---------------------------------------------------------------------------


def cost_matrix_penalty(
    y_true_cat: list[str],
    y_pred_cat: list[str],
    fn_cost: int = 10,
    fp_cost: int = 1,
) -> int:
    """Asymmetric cost: FN (missed High) costs 10x more than FP (false High)."""
    penalty = 0
    for actual, predicted in zip(y_true_cat, y_pred_cat):
        if actual == "High" and predicted != "High":
            penalty += fn_cost
        elif actual != "High" and predicted == "High":
            penalty += fp_cost
    return penalty


# ---------------------------------------------------------------------------
# 7. Metric C — F-beta for High class
# ---------------------------------------------------------------------------


def f_beta_high(
    y_true_cat: list[str],
    y_pred_cat: list[str],
    beta: float = 3.0,
) -> float:
    """F3 score for the High class (recall weighted 9x over precision)."""
    y_true_bin = [1 if c == "High" else 0 for c in y_true_cat]
    y_pred_bin = [1 if c == "High" else 0 for c in y_pred_cat]
    return float(fbeta_score(y_true_bin, y_pred_bin, beta=beta, zero_division="warn"))


# ---------------------------------------------------------------------------
# 8. Metric A — Baseline-relative RMSE
# ---------------------------------------------------------------------------


def baseline_relative_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
) -> float:
    """Skill score: % improvement over baseline RMSE."""
    rmse_model = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_base = np.sqrt(mean_squared_error(y_true, y_baseline))
    if rmse_base == 0:
        return 0.0
    return float((1.0 - rmse_model / rmse_base) * 100)


# ---------------------------------------------------------------------------
# 9. Category assignment
# ---------------------------------------------------------------------------


def assign_categories(
    values: np.ndarray,
    thresholds: tuple[float, float],
) -> list[str]:
    """Map continuous values -> Low / Medium / High."""
    p33, p67 = thresholds
    cats = []
    for v in values:
        if np.isnan(v):
            cats.append("Low")  # fallback
        elif v <= p33:
            cats.append("Low")
        elif v <= p67:
            cats.append("Medium")
        else:
            cats.append("High")
    return cats


# ---------------------------------------------------------------------------
# 10. Feature importance
# ---------------------------------------------------------------------------


def get_feature_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return sorted feature importance DataFrame."""
    imp = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 11. Full evaluation
# ---------------------------------------------------------------------------


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
    thresholds: tuple[float, float],
) -> EvalMetrics:
    """Compute all 3 business metrics + standard metrics + confusion matrix."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    y_true_cat = assign_categories(y_true, thresholds)
    y_pred_cat = assign_categories(y_pred, thresholds)

    # Metric A — RMSE skill score
    skill = baseline_relative_rmse(y_true, y_pred, y_baseline)

    # Metric B — cost penalty
    cost = cost_matrix_penalty(y_true_cat, y_pred_cat)

    # Metric C — F3
    f3 = f_beta_high(y_true_cat, y_pred_cat)

    # Confusion matrix (Low, Medium, High)
    labels = ["Low", "Medium", "High"]
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels)

    return EvalMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        cost_penalty=cost,
        f3_high=f3,
        rmse_skill_score=skill,
        confusion=cm,
    )
