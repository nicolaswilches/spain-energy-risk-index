"""
Phase 2.
Feature Engineering Pipeline.
"""

from __future__ import annotations

# Imports
import sys
from pathlib import Path
import joblib
import pandas as pd

# Helps python locate modules of the grid_risk package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Imports from the features.py script
from grid_risk.features import (  # noqa: E402
    feature_names,
    add_lagged_features,
    assign_risk_category,
    build_feature_matrix,
    chronological_split,
    compute_core_factors,
    fit_risk_index,
    transform_risk_index,
)

DATA = ROOT / "data"
ARTIFACTS = DATA / "artifacts"


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data/merged_daily_data.parquet ...")
    df = pd.read_parquet(DATA / "merged_daily_data.parquet")
    print(f"  Rows: {len(df):,}")

    # Compute and concat 3 core factors 
    factors = compute_core_factors(df)
    df = pd.concat([df, factors], axis=1) #horizontally

    # Execute chronological split into training, validation and testing data sets.
    train, val, test = chronological_split(df)
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    # -- Fit risk index on train only -----------------------------------------
    train_factors = train[["flexibility_share", "demand_forecast_error", "net_load"]]
    fit = fit_risk_index(train_factors)

    ev = fit.pca.explained_variance_ratio_
    print(f"  PCA explained variance: {ev[0]:.1%} / {ev[1]:.1%} / {ev[2]:.1%}")
    print(f"  PC1 weights: {fit.pc1_weights}")
    print(f"  Thresholds (p33, p67): {fit.thresholds}")

    # -- Transform all splits -------------------------------------------------
    for label, split in [("train", train), ("val", val), ("test", test)]:
        split_factors = split[
            ["flexibility_share", "demand_forecast_error", "net_load"]
        ]
        split["risk_index"] = transform_risk_index(split_factors, fit).values
        split["risk_category"] = assign_risk_category(
            split["risk_index"],
            fit.thresholds,
        ).values

    # -- Rejoin ---------------------------------------------------------------
    df = pd.concat([train, val, test])
    df.sort_index(inplace=True)

    # -- Lag features ---------------------------------------------------------
    df = add_lagged_features(df)

    # -- Drop NaN from lags ---------------------------------------------------
    before = len(df)
    df.dropna(
        subset=["risk_index", "risk_index_lag_1d", "risk_index_lag_7d"],
        inplace=True,
    )
    print(f"  Dropped {before - len(df)} rows with NaN lags")

    # -- Re-split -------------------------------------------------------------
    train, val, test = chronological_split(df)

    # -- Verification ---------------------------------------------------------
    for label, split in [("train", train), ("val", val), ("test", test)]:
        ri = split["risk_index"]
        assert ri.notna().all(), f"{label}: NaN in risk_index"
        assert (ri >= 0).all() and (ri <= 1).all(), f"{label}: risk_index out of [0,1]"

        X, names = build_feature_matrix(split)
        assert X.notna().all().all(), f"{label}: NaN in features"
        assert len(names) == len(feature_names), (
            f"{label}: expected {len(feature_names)} features, got {len(names)}"
        )

        cats = split["risk_category"].value_counts()
        print(
            f"  {label:5s}  rows={len(split):>5,}  "
            f"Low={cats.get('Low', 0):>4,}  Med={cats.get('Medium', 0):>4,}  "
            f"High={cats.get('High', 0):>4,}"
        )

    # -- Save -----------------------------------------------------------------
    df.to_parquet(DATA / "daily_features.parquet")
    train.to_parquet(DATA / "train.parquet")
    val.to_parquet(DATA / "val.parquet")
    test.to_parquet(DATA / "test.parquet")
    joblib.dump(fit, ARTIFACTS / "risk_index_fit.joblib")

    print("\nSaved:")
    print(f"  data/daily_features.parquet  ({len(df):,} rows)")
    print(f"  data/train.parquet           ({len(train):,} rows)")
    print(f"  data/val.parquet             ({len(val):,} rows)")
    print(f"  data/test.parquet            ({len(test):,} rows)")
    print("  data/artifacts/risk_index_fit.joblib")
    print(f"\nFeatures ({len(feature_names)}): {feature_names}")
    print("Phase 2 complete.")


if __name__ == "__main__":
    main()
