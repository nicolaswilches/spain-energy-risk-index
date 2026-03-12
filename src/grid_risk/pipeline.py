"""Orchestrator: extract -> clean -> merge -> validate -> save (daily pipeline)."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from grid_risk.api_client import REEClient, OpenMeteoClient
from grid_risk.config import (
    REE_ENDPOINTS,
    OUTPUT_COLUMNS,
    OUTPUT_FILENAME,
)
from grid_risk.extractors import (
    extract_demand_daily,
    extract_generation_daily,
    extract_spot_price_daily,
    extract_weather_daily,
)
from grid_risk.cleaning import (
    ensure_date_index,
    reindex_daily,
    impute_gaps,
    add_calendar_features,
    add_degree_days,
    validate_dataset,
)

logger = logging.getLogger(__name__)


def run_pipeline(
    start: date,
    end: date,
    output_dir: Path,
) -> Path:
    """Run the full daily extraction pipeline and save to parquet.

    Merges four data streams:
        A. REE demand (hourly -> daily mean) + generation (daily)
        B. Open-Meteo weather (daily)
        C. Calendar features (computed)
        D. REE spot price (hourly -> daily mean) + HDD/CDD (derived)
    """

    # -- 1. REE extractions ---------------------------------------------------
    logger.info("Starting REE extractions from %s to %s", start, end)

    with REEClient() as ree:
        # Demand: hourly data, resampled to daily mean in extractor
        demand_raw = ree.fetch_demand_hourly(
            REE_ENDPOINTS["demand"],
            start,
            end,
        )
        # Generation: daily native
        gen_raw = ree.fetch_generation_daily(
            REE_ENDPOINTS["generation_mix"],
            start,
            end,
        )
        # Spot price: hourly data, resampled to daily mean
        spot_raw = ree.fetch_spot_price_hourly(
            REE_ENDPOINTS["spot_price"],
            start,
            end,
        )

    demand_df = extract_demand_daily(demand_raw)
    gen_df = extract_generation_daily(gen_raw)
    spot_df = extract_spot_price_daily(spot_raw)

    logger.info(
        "REE extraction complete: demand=%d, generation=%d, spot=%d daily rows",
        len(demand_df),
        len(gen_df),
        len(spot_df),
    )

    # -- 2. Open-Meteo weather ------------------------------------------------
    logger.info("Fetching Open-Meteo weather data")

    with OpenMeteoClient() as meteo:
        weather_raw = meteo.fetch_daily_weather(start, end)

    weather_df = extract_weather_daily(weather_raw)
    logger.info("Weather extraction complete: %d daily rows", len(weather_df))

    # -- 3. Merge all streams on date index -----------------------------------
    # Start with demand as base
    merged = ensure_date_index(demand_df)
    merged = reindex_daily(merged)

    # Join generation
    if not gen_df.empty:
        gen_df = ensure_date_index(gen_df)
        merged = merged.join(gen_df, how="left")

    # Join spot price (Stream D)
    if not spot_df.empty:
        spot_df = ensure_date_index(spot_df)
        merged = merged.join(spot_df, how="left")

    # Join weather
    if not weather_df.empty:
        weather_df = ensure_date_index(weather_df)
        merged = merged.join(weather_df, how="left")

    # -- 4. Calendar features (Stream C) --------------------------------------
    merged = add_calendar_features(merged)

    # -- 5. Derived features (HDD / CDD) -------------------------------------
    merged = add_degree_days(merged)

    # -- 6. Impute small gaps -------------------------------------------------
    merged = impute_gaps(merged)

    # -- 7. Ensure all output columns exist -----------------------------------
    for col in OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA

    # -- 8. Validate ----------------------------------------------------------
    report = validate_dataset(merged)
    logger.info("Validation passed: %s", report.passed)
    for msg in report.messages:
        logger.info("  %s", msg)

    # -- 9. Save --------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME

    final_cols = [c for c in OUTPUT_COLUMNS if c in merged.columns]
    merged[final_cols].to_parquet(output_path, engine="pyarrow")

    logger.info("Saved %d rows to %s", len(merged), output_path)
    return output_path
