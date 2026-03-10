"""Phase 2 — Target creation & feature engineering (pure logic, no I/O)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Dataclass for fitted PCA pipeline
# ---------------------------------------------------------------------------

@dataclass
class RiskIndexFit:
    scaler: StandardScaler
    pca: PCA
    minmax: MinMaxScaler
    pc1_weights: np.ndarray          # PCA component weights for PC1
    thresholds: tuple[float, float]  # (p33, p67) on train risk_index


# ---------------------------------------------------------------------------
# 1. Chronological split
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / val / test by fixed UTC boundaries."""
    train = df.loc[:"2023-12-31 23:00:00+00:00"].copy()
    val = df.loc["2024-01-01 00:00:00+00:00":"2024-06-30 23:00:00+00:00"].copy()
    test = df.loc["2024-07-01 00:00:00+00:00":].copy()
    return train, val, test


# ---------------------------------------------------------------------------
# 2. Core risk factors
# ---------------------------------------------------------------------------

FACTOR_COLS = ["flexibility_share", "demand_forecast_error", "net_load"]


def compute_core_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 3-column DataFrame with the risk factors."""
    factors = pd.DataFrame(index=df.index)
    factors["flexibility_share"] = (
        (df["gen_combined_cycle_mw"] + df["gen_hydro_mw"]) / df["gen_total_mw"]
    )
    factors["demand_forecast_error"] = (
        df["actual_demand_mw"] - df["forecast_demand_mw"]
    )
    factors["net_load"] = (
        df["actual_demand_mw"] - (df["gen_wind_mw"] + df["gen_solar_pv_mw"])
    )
    return factors


# ---------------------------------------------------------------------------
# 3. Fit risk index (train only)
# ---------------------------------------------------------------------------

def fit_risk_index(train_factors: pd.DataFrame) -> RiskIndexFit:
    """Fit StandardScaler → PCA → MinMaxScaler on train factors."""
    clean = train_factors.dropna()

    scaler = StandardScaler().fit(clean)
    scaled = scaler.transform(clean)

    pca = PCA(n_components=3).fit(scaled)
    pc1 = pca.transform(scaled)[:, 0].reshape(-1, 1)

    minmax = MinMaxScaler().fit(pc1)
    risk_train = minmax.transform(pc1).ravel()

    p33 = float(np.percentile(risk_train, 33))
    p67 = float(np.percentile(risk_train, 67))

    return RiskIndexFit(
        scaler=scaler,
        pca=pca,
        minmax=minmax,
        pc1_weights=pca.components_[0],
        thresholds=(p33, p67),
    )


# ---------------------------------------------------------------------------
# 4. Transform risk index (any split)
# ---------------------------------------------------------------------------

def transform_risk_index(
    factors: pd.DataFrame,
    fit: RiskIndexFit,
) -> pd.Series:
    """Apply fitted pipeline to produce risk_index ∈ [0, 1]. NaN propagated."""
    result = pd.Series(np.nan, index=factors.index, name="risk_index")
    mask = factors.notna().all(axis=1)
    if mask.sum() == 0:
        return result

    clean = factors.loc[mask]
    scaled = fit.scaler.transform(clean)
    pc1 = fit.pca.transform(scaled)[:, 0].reshape(-1, 1)
    ri = fit.minmax.transform(pc1).ravel()
    ri = np.clip(ri, 0.0, 1.0)

    result.loc[mask] = ri
    return result


# ---------------------------------------------------------------------------
# 5. Risk category
# ---------------------------------------------------------------------------

def assign_risk_category(
    risk_index: pd.Series,
    thresholds: tuple[float, float],
) -> pd.Series:
    """Map risk_index → Low / Medium / High using percentile thresholds."""
    p33, p67 = thresholds
    cats = pd.Series(pd.NA, index=risk_index.index, name="risk_category", dtype="string")
    cats.loc[risk_index <= p33] = "Low"
    cats.loc[(risk_index > p33) & (risk_index <= p67)] = "Medium"
    cats.loc[risk_index > p67] = "High"
    return cats


# ---------------------------------------------------------------------------
# 6. Calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour_of_day, day_of_week, month, is_weekend from Europe/Madrid tz."""
    madrid = df.index.tz_convert("Europe/Madrid")
    df = df.copy()
    df["hour_of_day"] = madrid.hour
    df["day_of_week"] = madrid.dayofweek
    df["month"] = madrid.month
    df["is_weekend"] = (madrid.dayofweek >= 5).astype(int)
    return df


# ---------------------------------------------------------------------------
# 7. Lagged features
# ---------------------------------------------------------------------------

def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 24h and 168h (1-week) lags of risk_index."""
    df = df.copy()
    df["risk_index_lag_24h"] = df["risk_index"].shift(24)
    df["risk_index_lag_168h"] = df["risk_index"].shift(168)
    return df


# ---------------------------------------------------------------------------
# 8. Feature matrix
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "forecast_demand_mw",
    "forecast_wind_mw",
    "forecast_solar_mw",
    "hour_of_day",
    "day_of_week",
    "month",
    "is_weekend",
    "risk_index_lag_24h",
    "risk_index_lag_168h",
]


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (X_df, feature_names) — 9 feature columns ready for modeling."""
    return df[FEATURE_NAMES].copy(), list(FEATURE_NAMES)
