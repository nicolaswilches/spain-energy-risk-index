"""
Module 2: Feature Engineering.
This module contains:
1. Definition of the RiskIndexFit class.
2. Dataset chronological splits.
3. Feature engineering functions.
"""

from __future__ import annotations

# Imports
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Classes
# ---------------------------------------------------------------------------
@dataclass
class RiskIndexFit:
    """Container for the calibrated PCA risk-index pipeline.

    There is only 1 instance of this class. It stores the fitted
    weights for the 3 core factors of the trained model.

    Calibration parameters:
    - scaler: Mean/std of 3 input factors for standardization.
    - pca: PCA object that projects grid factors into 1 dimension.
    - minmax: Min/max of PCA scores to squeeze risk into [0, 1].
    - pc1_weights: Loadings (importance) for each input factor.
    - thresholds: Percentile thresholds to classify a Risk Index.
    """

    scaler: StandardScaler
    pca: PCA
    minmax: MinMaxScaler
    pc1_weights: np.ndarray
    thresholds: tuple[float, float, float, float]


# Functions
# ---------------------------------------------------------------------------
train_end = date(2023, 12, 31)
val_end = date(2024, 6, 30)


def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits dataframe into:
    - Training set (<= 2023-12-31)
    - Validation set (2024-01-01 to 2024-06-30)
    - Testing set (2024-07-01 onward)

    Returns a tuple of 3 pandas dataframes for: train, val, test.
    """
    train = df.loc[df.index <= train_end].copy()
    val = df.loc[(df.index > train_end) & (df.index <= val_end)].copy()
    test = df.loc[df.index > val_end].copy()
    return train, val, test


# ---------------------------------------------------------------------------
def compute_core_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the 3 core factors:
    1. flexibility_share
    2. demand_forecast_error
    3. net_load

    Returns a dataframe with the 3 core factors.
    """
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
def fit_risk_index(train_factors: pd.DataFrame) -> RiskIndexFit:
    """
    Using the 3 core factors dataframe (using only training data):
    > Fits StandardScaler.
    > Then fits PCA.
    > Then fits MinMaxScaler.
    > Declares the risk labels thresholds (50p, 80p)

    Returns a RiskIndexFit class instance or 'Calibrator' already tuned.
    """
    clean = train_factors.dropna()

    scaler = StandardScaler().fit(clean)
    scaled = scaler.transform(clean)

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


# ---------------------------------------------------------------------------
def transform_risk_index(
    factors: pd.DataFrame,
    calibrator: RiskIndexFit,
) -> pd.Series:
    """
    Uses:
    - Calculated core factors dataframe.
    - Calibrated RiskIndexFit class object.

    Returns a pandas Series with the Grid Risk Indexes.
    """
    result = pd.Series(np.nan, index=factors.index, name="risk_index")
    mask = factors.notna().all(axis=1)  # Days where all factors have data
    if mask.sum() == 0:
        return result

    clean = factors.loc[mask]  # clean means without nulls
    scaled = calibrator.scaler.transform(clean)
    pc1 = calibrator.pca.transform(scaled)[:, 0].reshape(-1, 1)
    risk_index = calibrator.minmax.transform(pc1).ravel()
    risk_index = np.clip(risk_index, 0.0, 1.0)  # bound index between 0 and 1

    result.loc[mask] = risk_index  # assign risk index to the days with full data.
    return result


# ---------------------------------------------------------------------------
def assign_risk_category(
    risk_index: pd.Series | np.ndarray, thresholds: tuple[float, float, float, float]
) -> pd.Series | list[str]:
    """
    Maps risk_index into categories.
    Returns a pandas Series for each Risk Index.
    """
    (
        p20,
        p40,
        p60,
        p80,
    ) = thresholds  # from fit_risk_index()
    if isinstance(risk_index, pd.Series):
        categories = pd.Series(
            pd.NA,
            index=risk_index.index,
            name="risk_category",
            dtype="string",
        )
        categories.loc[risk_index <= p20] = "Low"
        categories.loc[(risk_index > p20) & (risk_index <= p40)] = "Stable"
        categories.loc[(risk_index > p40) & (risk_index <= p60)] = "Elevated"
        categories.loc[(risk_index > p60) & (risk_index <= p80)] = "Severe"
        categories.loc[risk_index > p80] = "Extreme"
        return categories
    else:
        categories = []
        for v in risk_index:
            if np.isnan(v) or v <= p20:
                categories.append("Low")
            elif v <= p40:
                categories.append("Stable")
            elif v <= p60:
                categories.append("Elevated")
            elif v <= p80:
                categories.append("Severe")
            else:
                categories.append("Extreme")
        return categories


# ---------------------------------------------------------------------------
def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 1-day and 7-day lags of risk_index.
    """
    df = df.copy()
    df["risk_index_lag_1d"] = df["risk_index"].shift(1)
    df["risk_index_lag_7d"] = df["risk_index"].shift(7)
    return df


# ---------------------------------------------------------------------------
feature_names = [
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
    """
    Returns a tuple with (X_df, feature_names) ready for modeling.
    """
    return df[feature_names].copy(), list(feature_names)
