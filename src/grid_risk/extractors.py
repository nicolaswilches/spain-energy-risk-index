"""JSON response -> DataFrame extractors for each data stream (daily pipeline)."""

from __future__ import annotations

import logging

import pandas as pd

from grid_risk.config import (
    REE_DEMAND_TITLES,
    REE_GENERATION_TITLES,
    REE_SPOT_TITLE,
    WEATHER_DAILY_VARS,
)

logger = logging.getLogger(__name__)


# -- REE helpers --------------------------------------------------------------


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
                records.append(
                    {
                        "datetime": val["datetime"],
                        col_name: val["value"],
                    }
                )
            break  # only first matching series per chunk

    if not records:
        logger.warning("No data found for series '%s'", target_title)
        return pd.DataFrame(columns=["datetime", col_name])

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.groupby("datetime", as_index=False).first()
    return df


# -- Stream A: REE Demand (hourly -> daily) -----------------------------------


def extract_demand_daily(responses: list[dict]) -> pd.DataFrame:
    """Parse REE demand (hourly) -> resample to daily mean MW.

    Returns a DataFrame indexed by date with columns:
    actual_demand_mw, forecast_demand_mw.
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

    # Resample hourly -> daily mean
    merged = merged.set_index("datetime").sort_index()
    merged = merged.resample("D").mean()
    merged.index = merged.index.date  # type: ignore[assignment]
    merged.index.name = "date"

    logger.info("Demand extracted: %d daily rows", len(merged))
    return merged


# -- Stream A: REE Generation (daily native) ----------------------------------


def extract_generation_daily(responses: list[dict]) -> pd.DataFrame:
    """Parse REE generation mix (daily) -> DataFrame with one col per tech + total.

    The generation endpoint returns daily MWh values. We convert to daily mean
    MW by dividing by 24 so units are consistent with demand.
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

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="datetime", how="outer")

    merged = merged.set_index("datetime").sort_index()

    # Convert MWh -> daily mean MW (divide by 24)
    gen_cols = [c for c in tech_map if c in merged.columns]
    for col in gen_cols:
        merged[col] = merged[col] / 24.0

    # Total generation
    merged["gen_total_mw"] = merged[gen_cols].sum(axis=1)

    # Convert to date index (REE daily timestamps are CET midnight)
    merged.index = merged.index.tz_convert("Europe/Madrid").date  # type: ignore[assignment]
    merged.index.name = "date"

    # Deduplicate (DST transitions may create two entries per day)
    merged = merged[~merged.index.duplicated(keep="first")]

    logger.info("Generation extracted: %d daily rows", len(merged))
    return merged


# -- Stream D: REE Spot Price (hourly -> daily) -------------------------------


def extract_spot_price_daily(responses: list[dict]) -> pd.DataFrame:
    """Parse REE spot market price (hourly) -> resample to daily mean EUR/MWh."""
    df = _parse_ree_series(
        responses,
        target_title=REE_SPOT_TITLE,
        col_name="spot_price_eur_mwh",
    )

    if df.empty:
        logger.warning("No spot price data extracted")
        return pd.DataFrame()

    df = df.set_index("datetime").sort_index()
    df = df.resample("D").mean()
    df.index = df.index.date  # type: ignore[assignment]
    df.index.name = "date"

    logger.info("Spot price extracted: %d daily rows", len(df))
    return df


# -- Stream B: Open-Meteo Weather (daily native) -----------------------------


def extract_weather_daily(responses: list[dict]) -> pd.DataFrame:
    """Parse Open-Meteo archive responses -> DataFrame with weather columns."""
    all_records: list[dict] = []

    for resp in responses:
        daily = resp.get("daily", {})
        times = daily.get("time", [])
        if not times:
            continue
        for i, dt_str in enumerate(times):
            row: dict = {"date": dt_str}
            for var in WEATHER_DAILY_VARS:
                vals = daily.get(var, [])
                row[var] = vals[i] if i < len(vals) else None
            all_records.append(row)

    if not all_records:
        logger.warning("No weather data extracted")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.drop_duplicates(subset=["date"]).set_index("date").sort_index()

    logger.info("Weather extracted: %d daily rows", len(df))
    return df
