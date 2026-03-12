"""Phase 2 — Target creation & feature engineering (daily pipeline, pure logic)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ---------------------------------------------------------------------------
# Dataclass for fitted PCA pipeline
# ---------------------------------------------------------------------------


@dataclass
class RiskIndexFit:
    scaler: StandardScaler
    pca: PCA
    minmax: MinMaxScaler
    pc1_weights: np.ndarray  # PCA component weights for PC1
    thresholds: tuple[float, float]  # (p33, p67) on train risk_index


# ---------------------------------------------------------------------------
# 1. Chronological split (daily)
# ---------------------------------------------------------------------------

TRAIN_END = date(2023, 12, 31)
VAL_END = date(2024, 6, 30)


def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train / val / test by date boundaries.

    Train: <= 2023-12-31
    Val:   2024-01-01 to 2024-06-30
    Test:  2024-07-01 onward
    """
    train = df.loc[df.index <= TRAIN_END].copy()
    val = df.loc[(df.index > TRAIN_END) & (df.index <= VAL_END)].copy()
    test = df.loc[df.index > VAL_END].copy()
    return train, val, test


# ---------------------------------------------------------------------------
# 2. Core risk factors
# ---------------------------------------------------------------------------

FACTOR_COLS = ["flexibility_share", "demand_forecast_error", "net_load"]


def compute_core_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Return a 3-column DataFrame with the risk factors."""
    factors = pd.DataFrame(index=df.index)
    factors["flexibility_share"] = (
        df["gen_combined_cycle_mw"] + df["gen_hydro_mw"]
    ) / df["gen_total_mw"]
    factors["demand_forecast_error"] = df["actual_demand_mw"] - df["forecast_demand_mw"]
    factors["net_load"] = df["actual_demand_mw"] - (
        df["gen_wind_mw"] + df["gen_solar_pv_mw"]
    )
    return factors


# ---------------------------------------------------------------------------
# 3. Fit risk index (train only)
# ---------------------------------------------------------------------------


def fit_risk_index(train_factors: pd.DataFrame) -> RiskIndexFit:
    """Fit StandardScaler -> PCA -> MinMaxScaler on train factors."""
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
    """Apply fitted pipeline to produce risk_index in [0, 1]. NaN propagated."""
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
    """Map risk_index -> Low / Medium / High using percentile thresholds."""
    p33, p67 = thresholds
    cats = pd.Series(
        pd.NA,
        index=risk_index.index,
        name="risk_category",
        dtype="string",
    )
    cats.loc[risk_index <= p33] = "Low"
    cats.loc[(risk_index > p33) & (risk_index <= p67)] = "Medium"
    cats.loc[risk_index > p67] = "High"
    return cats


# ---------------------------------------------------------------------------
# 6. Lagged features (daily)
# ---------------------------------------------------------------------------


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 1-day and 7-day lags of risk_index."""
    df = df.copy()
    df["risk_index_lag_1d"] = df["risk_index"].shift(1)
    df["risk_index_lag_7d"] = df["risk_index"].shift(7)
    return df


# ---------------------------------------------------------------------------
# 7. Feature matrix (daily)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # REE day-ahead forecast
    "forecast_demand_mw",
    # Weather
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
    "precipitation_sum",
    # Derived weather
    "hdd",
    "cdd",
    # Calendar
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    # Stream D: spot price
    "spot_price_eur_mwh",
    # Lags
    "risk_index_lag_1d",
    "risk_index_lag_7d",
]


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (X_df, feature_names) ready for modeling."""
    return df[FEATURE_NAMES].copy(), list(FEATURE_NAMES)
