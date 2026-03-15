"""
Unit Tests for grid_risk.features

These tests verify isolated data transformations and math logic
without relying on external APIs, running models, or HTTP servers.
"""

import numpy as np
import pandas as pd

from grid_risk.features import (
    assign_risk_category,
    compute_core_factors,
)


def test_compute_core_factors_math():
    """
    Test that flexibility_share, demand_forecast_error, and net_load
    are mathematically calculated correctly from raw data.
    """
    # Create fake dataframe with raw inputs
    df = pd.DataFrame({
        "actual_demand_mw": [30000.0],
        "forecast_demand_mw": [29000.0],
        "gen_wind_mw": [5000.0],
        "gen_solar_pv_mw": [3000.0],
        "gen_hydro_mw": [2000.0],
        "gen_combined_cycle_mw": [4000.0],
        "gen_total_mw": [20000.0],
    })

    factors = compute_core_factors(df)

    # 1. Flexibility Share: (Hydro + CC) / Total
    # (2000 + 4000) / 20000 = 6000 / 20000 = 0.3
    assert factors.loc[0, "flexibility_share"] == 0.3

    # 2. Forecast Error: Actual - Forecast
    # 30000 - 29000 = 1000
    assert factors.loc[0, "demand_forecast_error"] == 1000.0

    # 3. Net Load: Actual - (Wind + Solar)
    # 30000 - (5000 + 3000) = 22000
    assert factors.loc[0, "net_load"] == 22000.0


def test_assign_risk_category_logic():
    """
    Test that array of Risk Index scores correctly map to the categorical
    labels based on defined thresholds.
    """
    thresholds = (0.2, 0.4, 0.6, 0.8)

    # Fake risk scores across all boundaries
    scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9, np.nan])

    categories = assign_risk_category(scores, thresholds)

    # Validate correctly assigned categories
    assert categories[0] == "Low"  # 0.1 <= 0.2
    assert categories[1] == "Stable"  # 0.3 <= 0.4
    assert categories[2] == "Elevated"  # 0.5 <= 0.6
    assert categories[3] == "Severe"  # 0.7 <= 0.8
    assert categories[4] == "Extreme"  # 0.9 > 0.8
    assert categories[5] == "Low"  # NaN defaults to Low
