"""JSON response → DataFrame extractors for each data category."""

from __future__ import annotations

import logging

import pandas as pd

from grid_risk.config import REE_DEMAND_TITLES, REE_GENERATION_TITLES

logger = logging.getLogger(__name__)


def _parse_ree_series(
    responses: list[dict],
    target_title: str,
    col_name: str,
) -> pd.DataFrame:
    """Extract a single named series from REE 'included' array across chunks."""
    records: list[dict] = []

    for resp in responses:
        for item in resp.get("included", []):
            attrs = item.get("attributes", {})
            title = attrs.get("title", "")
            if title.strip().lower() != target_title.strip().lower():
                continue
            for val in attrs.get("values", []):
                records.append({
                    "datetime": val["datetime"],
                    col_name: val["value"],
                })
            break  # only first matching series per chunk

    if not records:
        logger.warning("No data found for series '%s'", target_title)
        return pd.DataFrame(columns=["datetime", col_name])

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.groupby("datetime", as_index=False).first()
    return df


def extract_demand(responses: list[dict]) -> pd.DataFrame:
    """Parse REE demanda-tiempo-real → DataFrame with actual + forecast demand.

    The endpoint returns 5-min data with series: Real, Forecasted, Scheduled.
    We extract Real and Forecasted, then resample to hourly means.
    """
    dfs: list[pd.DataFrame] = []

    for col_name, title in REE_DEMAND_TITLES.items():
        df = _parse_ree_series(responses, target_title=title, col_name=col_name)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        logger.warning("No demand data extracted")
        return pd.DataFrame()

    # Merge all demand series on datetime
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="datetime", how="outer")

    # Resample 5-min data to hourly means
    merged = merged.set_index("datetime").sort_index()
    merged = merged.resample("h").mean()
    merged = merged.reset_index()

    logger.info("Demand extracted: %d hourly rows", len(merged))
    return merged


def extract_generation_mix(responses: list[dict]) -> pd.DataFrame:
    """Parse REE generation mix (daily) → DataFrame with one column per technology + total.

    The generation endpoint only supports daily granularity. Values represent
    daily averages in MW for each technology.
    """
    tech_map = REE_GENERATION_TITLES
    dfs: list[pd.DataFrame] = []

    for col_name, title in tech_map.items():
        df = _parse_ree_series(responses, target_title=title, col_name=col_name)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        logger.warning("No generation data extracted")
        return pd.DataFrame()

    # Merge all tech columns on datetime
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="datetime", how="outer")

    # Compute total generation from columns that were actually extracted
    gen_cols = [c for c in tech_map.keys() if c in merged.columns]
    merged["gen_total_mw"] = merged[gen_cols].sum(axis=1)

    logger.info("Generation extracted: %d daily rows", len(merged))
    return merged


def broadcast_daily_to_hourly(
    daily_df: pd.DataFrame,
    hourly_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Broadcast daily generation values to each hour of that day.

    Each hour gets the daily average MW value for its date.
    Matching uses CET dates (Europe/Madrid) since REE daily data represents
    calendar days in Spanish local time.
    """
    if daily_df.empty:
        return pd.DataFrame(index=hourly_index)

    daily_df = daily_df.copy()
    if "datetime" in daily_df.columns:
        daily_df = daily_df.set_index("datetime")

    # Convert daily timestamps to CET date for matching
    daily_df["_cet_date"] = daily_df.index.tz_convert("Europe/Madrid").date
    daily_df = daily_df.set_index("_cet_date")
    daily_df = daily_df[~daily_df.index.duplicated(keep="first")]

    # Create hourly frame and match by CET calendar date
    hourly = pd.DataFrame(index=hourly_index)
    hourly["_cet_date"] = hourly.index.tz_convert("Europe/Madrid").date

    for col in daily_df.columns:
        date_map = daily_df[col].to_dict()
        hourly[col] = hourly["_cet_date"].map(date_map)

    hourly = hourly.drop(columns=["_cet_date"])
    return hourly


def extract_esios_forecast(
    responses: list[dict],
    col_name: str,
) -> pd.DataFrame:
    """Parse ESIOS indicator responses → DataFrame with given column name."""
    records: list[dict] = []

    for resp in responses:
        indicator = resp.get("indicator", {})
        for val in indicator.get("values", []):
            records.append({
                "datetime": val["datetime"],
                col_name: val["value"],
            })

    if not records:
        logger.warning("No ESIOS data found for '%s'", col_name)
        return pd.DataFrame(columns=["datetime", col_name])

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.groupby("datetime", as_index=False).first()
    return df
