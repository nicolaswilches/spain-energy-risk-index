"""
Module 3: Model(s) Training and Evaluation.
This module contains:
1. Persistence baseline (1-day lag)
2. Ridge baseline
3. LightGBM quantile training
"""

# Imports
# ---------------------------------------------------------------------------
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
from grid_risk.features import assign_risk_category


# Classes
# ---------------------------------------------------------------------------
@dataclass
class BaselineResult:
    """
    Class to store baseline metrics and results.
    Each instance of this class acts as a scorecard for a model.
    """

    y_true: np.ndarray
    y_pred: np.ndarray
    rmse: float
    mae: float


@dataclass
class EvalMetrics:
    """
    Class to store evaluation metrics.
    Each instance of this class acts as a business scorecard for a model.
    """

    rmse: float
    mae: float
    r2: float
    cost_penalty: int  # Metric B
    f3_extreme: float  # Metric C
    rmse_skill_score: float  # Metric A (% improvement)
    confusion: np.ndarray = field(repr=False)


# Models
# ---------------------------------------------------------------------------
def persistence_baseline(
    y_true: np.ndarray,
    lag_1d: np.ndarray,
) -> BaselineResult:
    """
    Baseline model: predict risk_index = value from 1 day ago.
    """
    mask = ~(np.isnan(y_true) | np.isnan(lag_1d))
    yt, yp = y_true[mask], lag_1d[mask]
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    return BaselineResult(y_true=yt, y_pred=yp, rmse=rmse, mae=mae)


# ---------------------------------------------------------------------------
def ridge_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
) -> tuple[Ridge, BaselineResult]:
    """
    Fits a Ridge model using historical Risk Indexes.

    Returns:
    - An instance of the sklearn.linear_model.Ridge class. (trained model)
    - An instance of the BaselineResult class. (evaluation metrics)
    """
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = np.clip(model.predict(X_eval), 0.0, 1.0)
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    mae = float(mean_absolute_error(y_eval, y_pred))
    result = BaselineResult(y_true=y_eval, y_pred=y_pred, rmse=rmse, mae=mae)
    return model, result


# ---------------------------------------------------------------------------
def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict[str, Any] | None = None,
    quantile_alpha: float = 0.90,
):
    """
    Trains a LightGBM with quantile regression.
    GBM = Gradient Boosting Machine.

    Returns a trained instance of the lightgbm.LGBMRegressor class.
    This instance is specialized in predicting a specific "ceiling" (quantile_alpha)
    """
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


# Hyper parameters tunning
# ---------------------------------------------------------------------------
def create_optuna_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    quantile_alpha: float = 0.90,
):
    """
    Return an Optuna objective closure that minimizes val quantile loss.
    """

    def objective(trial):  # type: ignore[no-untyped-def]
        """
        Searches for the best hyperparameters.
        For given trial:
        1. Trial 1: Optuna picks random numbers within set ranges.
        2. Trial 2 and beyond:
            It stops being random.
            It looks at the results of Trial 1 and thinks,
            "When I increased num_leaves, the error went down.
            Let me try an even higher number."

        3. For each trial we train a LightGBM model.
        4. For each model we calculate its quantile loss value.
            Why Quantile Loss?
            This is the most important part.
            You aren't just looking for high accuracy;
            you are looking for a model that correctly
            predicts the "90th percentile" (the high-risk ceiling).
            This loss value is the "Grade" the model gets for that trial.
        5. The quantile loss value is the 'Grade' for the model. Low is good.

        6. Returns the quantile loss value.
        """
        params = {
            "reg_alpha": trial.suggest_float(
                "reg_alpha", 1e-3, 10.0, log=True
            ),  # Notice you use log=True for small numbers like learning rate. This tells Optuna to spend more time exploring the difference between 0.01 and 0.02 than between 0.09 and 0.10.
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        }
        # For each trial we train a model.
        model = train_lightgbm(
            X_train,
            y_train,
            X_val,
            y_val,
            params=params,
            quantile_alpha=quantile_alpha,
        )
        y_pred = predict(model, X_val)
        return quantile_loss(y_val, y_pred, quantile_alpha)  # defined below

    return objective


def quantile_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float,
) -> float:
    """
    Calculates the average quantile loss for a LightGBM model.
    - If the result is Positive: The real risk was higher than predicted (Under-prediction).
    - If the result is Negative: The real risk was lower than predicted (Over-prediction).

    Returns a float value with the average quantile loss.
    """
    residual = y_true - y_pred
    return float(
        np.mean(np.where(residual >= 0, alpha * residual, (alpha - 1) * residual))
    )


# Predictions generation
# ---------------------------------------------------------------------------
def predict(model: Any, X: np.ndarray) -> np.ndarray:
    """
    Wraps the predict method to ensure values are between 0 and 1.

    Returns a numpy array of Risk Indexpredictions.
    """
    return np.clip(model.predict(X), 0.0, 1.0)


# Model Evaluation by bussiness metrics
# ---------------------------------------------------------------------------


# Metric 1: Penalty / Cost Matrix
def cost_matrix_penalty(
    y_true_cat: list[str],
    y_pred_cat: list[str],
    fn_cost: int = 10,  # False negatives are penalized x10.
    fp_cost: int = 1,  # False positives are penalized x1.
) -> int:
    """
    Calculates a 'cost' by comparing the Actual vs. Predicted risk category for each time period.
    Asymmetric cost: False Negatives (missed Extreme) costs 10x more than False Positives (false Extreme).

    Returns an integer value with the total cost of the model.
    """
    penalty = 0
    for actual, predicted in zip(y_true_cat, y_pred_cat):
        if actual == "Extreme" and predicted != "Extreme":
            penalty += fn_cost
        elif actual != "Extreme" and predicted == "Extreme":
            penalty += fp_cost
    return penalty


# Metric 2: F-beta for High class
def f_beta_high(
    y_true_cat: list[str],  # risk index - historical
    y_pred_cat: list[str],  # risk index - predictions
    beta: float = 3.0,  # beta parameter to weight higher Recall
) -> float:
    """
    F3 score for the Extreme class (recall weighted 9x over precision).
    F-Score with beta = 3 to give x3 higher importance to Recall than Precision.

    Returns a float value with the F3 score.
    """
    y_true_bin = [1 if c == "Extreme" else 0 for c in y_true_cat]
    y_pred_bin = [1 if c == "Extreme" else 0 for c in y_pred_cat]
    return float(fbeta_score(y_true_bin, y_pred_bin, beta=beta, zero_division="warn"))


# Metric 3: Baseline-relative RMSE
def baseline_relative_rmse(
    y_true: np.ndarray, y_pred: np.ndarray, y_baseline: np.ndarray
) -> float:
    """
    Calculates the RMSE for the baseline and LightGBM model.

    Returns the relative difference in RSMEs between models.
    """
    rmse_model = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_base = np.sqrt(mean_squared_error(y_true, y_baseline))
    if rmse_base == 0:
        return 0.0
    return float((1.0 - rmse_model / rmse_base) * 100)


# Feature importance
# ---------------------------------------------------------------------------
def get_feature_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Estimates the feature importance of a LightGBM model.

    Returns sorted feature importance DataFrame.
    """
    importance = model.feature_importances_
    df = pd.DataFrame({"feature": feature_names, "importance": importance})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# Full evaluation
# ---------------------------------------------------------------------------


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
    thresholds: tuple[float, float, float, float],
) -> EvalMetrics:
    """
    Computes all evaluation metrics for the LightGBM model.

    Returns an instance of the EvalMetrics class.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    y_true_cat = assign_risk_category(y_true, thresholds)
    y_pred_cat = assign_risk_category(y_pred, thresholds)

    # Metric A — RMSE skill score
    skill = baseline_relative_rmse(y_true, y_pred, y_baseline)

    # Metric B — cost penalty
    cost = cost_matrix_penalty(y_true_cat, y_pred_cat)

    # Metric C — F3 (Extreme class)
    f3 = f_beta_high(y_true_cat, y_pred_cat)

    # Confusion matrix (5 categories)
    labels = ["Low", "Stable", "Elevated", "Severe", "Extreme"]
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels)

    return EvalMetrics(
        rmse=rmse,
        mae=mae,
        r2=r2,
        cost_penalty=cost,
        f3_extreme=f3,
        rmse_skill_score=skill,
        confusion=cm,
    )
