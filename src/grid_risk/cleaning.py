"""Data cleaning, imputation, calendar features, and validation (daily pipeline)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import holidays
import numpy as np
import pandas as pd

from grid_risk.config import (
    DEMAND_MIN_MW,
    DEMAND_MAX_MW,
    GENERATION_MIN_MW,
    MAX_CONSECUTIVE_GAP_DAYS,
    MAX_MISSING_PCT,
    OUTPUT_COLUMNS,
    HDD_BASE_TEMP,
    CDD_BASE_TEMP,
)

logger = logging.getLogger(__name__)


# -- Index helpers ------------------------------------------------------------


def ensure_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has a sorted date index named 'date'."""
    if "date" in df.columns:
        df = df.set_index("date")
    # Convert datetime index to date if needed
    if hasattr(df.index, "date") and not isinstance(df.index[0], date):
        df.index = df.index.date  # type: ignore[assignment]
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    return df


def reindex_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a complete daily range, exposing gaps as NaN."""
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="D",
    )
    full_dates = full_range.date
    return df.reindex(full_dates)


# -- Imputation ---------------------------------------------------------------


def impute_gaps(
    df: pd.DataFrame,
    max_gap: int = MAX_CONSECUTIVE_GAP_DAYS,
) -> pd.DataFrame:
    """Forward-fill gaps up to max_gap consecutive days."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        mask = df[col].isna()
        if not mask.any():
            continue

        # Identify consecutive NaN groups
        groups = (~mask).cumsum()
        gap_sizes = mask.groupby(groups).transform("sum")

        # Forward-fill, then restore NaN for gaps exceeding threshold
        df[col] = df[col].ffill()
        too_long = mask & (gap_sizes > max_gap)
        df.loc[too_long, col] = np.nan

    return df


# -- Calendar features --------------------------------------------------------


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_of_week, month, is_weekend, is_holiday from date index."""
    df = df.copy()
    dates = pd.to_datetime(pd.Series(df.index))

    df["day_of_week"] = dates.dt.dayofweek.values
    df["month"] = dates.dt.month.values
    df["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int).values

    # Spanish holidays
    years = sorted(set(d.year for d in df.index))
    es_holidays = holidays.Spain(years=years)
    df["is_holiday"] = [1 if d in es_holidays else 0 for d in df.index]

    return df


# -- Derived weather features -------------------------------------------------


def add_degree_days(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heating Degree Days (HDD) and Cooling Degree Days (CDD)."""
    df = df.copy()
    if "temperature_2m_max" in df.columns and "temperature_2m_min" in df.columns:
        avg_temp = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2.0
        df["hdd"] = np.maximum(0, HDD_BASE_TEMP - avg_temp)
        df["cdd"] = np.maximum(0, avg_temp - CDD_BASE_TEMP)
    else:
        df["hdd"] = np.nan
        df["cdd"] = np.nan
    return df


# -- Validation ---------------------------------------------------------------


@dataclass
class ValidationReport:
    """Results of dataset validation checks."""

    total_rows: int
    expected_rows: int
    duplicate_count: int
    missing_pct: dict[str, float]
    range_violations: dict[str, int]
    passed: bool
    messages: list[str]


def validate_dataset(df: pd.DataFrame) -> ValidationReport:
    """Run validation checks and return a report."""
    messages: list[str] = []
    passed = True

    # Row count
    date_min = pd.Timestamp(df.index.min())
    date_max = pd.Timestamp(df.index.max())
    expected_rows = (date_max - date_min).days + 1
    total_rows = len(df)
    if total_rows != expected_rows:
        messages.append(
            f"Row count mismatch: got {total_rows}, expected {expected_rows}"
        )

    # Duplicates
    dup_count = pd.Index(df.index).duplicated().sum()
    if dup_count > 0:
        messages.append(f"Found {dup_count} duplicate dates")
        passed = False

    # Missing percentage per column
    missing_pct: dict[str, float] = {}
    for col in OUTPUT_COLUMNS:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            missing_pct[col] = round(pct, 2)
            if pct > MAX_MISSING_PCT:
                messages.append(
                    f"Column '{col}' has {pct:.2f}% missing "
                    f"(threshold: {MAX_MISSING_PCT}%)"
                )
                passed = False

    # Value range checks
    range_violations: dict[str, int] = {}

    if "actual_demand_mw" in df.columns:
        demand = df["actual_demand_mw"].dropna()
        violations = ((demand < DEMAND_MIN_MW) | (demand > DEMAND_MAX_MW)).sum()
        if violations > 0:
            range_violations["actual_demand_mw"] = int(violations)
            messages.append(
                f"actual_demand_mw: {violations} values outside "
                f"[{DEMAND_MIN_MW}, {DEMAND_MAX_MW}] MW"
            )

    gen_cols = [c for c in df.columns if c.startswith("gen_")]
    for col in gen_cols:
        vals = df[col].dropna()
        neg = (vals < GENERATION_MIN_MW).sum()
        if neg > 0:
            range_violations[col] = int(neg)
            messages.append(f"{col}: {neg} negative values")
            passed = False

    if not messages:
        messages.append("All validation checks passed")

    report = ValidationReport(
        total_rows=total_rows,
        expected_rows=expected_rows,
        duplicate_count=dup_count,
        missing_pct=missing_pct,
        range_violations=range_violations,
        passed=passed,
        messages=messages,
    )

    for msg in messages:
        log_fn = logger.info if passed else logger.warning
        log_fn("Validation: %s", msg)

    return report
