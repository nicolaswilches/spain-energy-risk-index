"""Timezone handling, imputation, and validation for the merged dataset."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from grid_risk.config import (
    DEMAND_MIN_MW,
    DEMAND_MAX_MW,
    GENERATION_MIN_MW,
    MAX_CONSECUTIVE_GAP_HOURS,
    MAX_MISSING_PCT,
    OUTPUT_COLUMNS,
)

logger = logging.getLogger(__name__)


def set_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure UTC DatetimeIndex, truncated to the hour."""
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    df.index = pd.to_datetime(df.index, utc=True).floor("h")
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    return df


def reindex_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a complete hourly range, exposing gaps as NaN."""
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h",
        tz="UTC",
    )
    return df.reindex(full_range)


def add_local_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Add local_hour column in CET/CEST (Europe/Madrid)."""
    local = df.index.tz_convert("Europe/Madrid")
    df["local_hour"] = local.hour
    return df


def impute_gaps(df: pd.DataFrame, max_gap: int = MAX_CONSECUTIVE_GAP_HOURS) -> pd.DataFrame:
    """Forward-fill gaps up to max_gap consecutive hours. Flag longer gaps."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        mask = df[col].isna()
        if not mask.any():
            continue

        # Identify consecutive NaN groups
        groups = (~mask).cumsum()
        gap_sizes = mask.groupby(groups).transform("sum")

        # Only fill gaps within threshold
        fillable = mask & (gap_sizes <= max_gap)
        df[col] = df[col].ffill()
        # Restore NaN for gaps exceeding threshold
        too_long = mask & (gap_sizes > max_gap)
        df.loc[too_long, col] = pd.NA

    return df


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
    hours_span = (df.index.max() - df.index.min()).total_seconds() / 3600 + 1
    expected_rows = int(hours_span)
    total_rows = len(df)
    if total_rows != expected_rows:
        messages.append(
            f"Row count mismatch: got {total_rows}, expected {expected_rows}"
        )

    # Duplicates
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        messages.append(f"Found {dup_count} duplicate timestamps")
        passed = False

    # Missing percentage per column
    missing_pct: dict[str, float] = {}
    for col in OUTPUT_COLUMNS:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            missing_pct[col] = round(pct, 2)
            if pct > MAX_MISSING_PCT:
                messages.append(f"Column '{col}' has {pct:.2f}% missing (threshold: {MAX_MISSING_PCT}%)")
                passed = False

    # Value range checks
    range_violations: dict[str, int] = {}

    if "actual_demand_mw" in df.columns:
        demand = df["actual_demand_mw"].dropna()
        violations = ((demand < DEMAND_MIN_MW) | (demand > DEMAND_MAX_MW)).sum()
        if violations > 0:
            range_violations["actual_demand_mw"] = int(violations)
            messages.append(
                f"actual_demand_mw: {violations} values outside [{DEMAND_MIN_MW}, {DEMAND_MAX_MW}] MW"
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
