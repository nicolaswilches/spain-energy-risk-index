"""Orchestrator: extract → clean → merge → validate → save."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from grid_risk.api_client import REEClient, ESIOSClient
from grid_risk.config import (
    REE_ENDPOINTS,
    ESIOS_INDICATORS,
    OUTPUT_COLUMNS,
    OUTPUT_FILENAME,
)
from grid_risk.extractors import (
    extract_demand,
    extract_generation_mix,
    extract_esios_forecast,
    broadcast_daily_to_hourly,
)
from grid_risk.cleaning import (
    set_datetime_index,
    reindex_hourly,
    add_local_hour,
    impute_gaps,
    validate_dataset,
)

logger = logging.getLogger(__name__)


def _create_forecast_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Create synthetic forecasts using 24h-lagged actuals when ESIOS is unavailable."""
    logger.warning(
        "ESIOS token not available — using 24h-lagged actuals as synthetic forecasts"
    )

    if "gen_wind_mw" in df.columns and "forecast_wind_mw" not in df.columns:
        df["forecast_wind_mw"] = df["gen_wind_mw"].shift(24)

    if "gen_solar_pv_mw" in df.columns and "forecast_solar_mw" not in df.columns:
        df["forecast_solar_mw"] = df["gen_solar_pv_mw"].shift(24)

    return df


def run_pipeline(
    start: date,
    end: date,
    output_dir: Path,
    esios_token: str | None = None,
) -> Path:
    """Run the full extraction pipeline and save to parquet."""

    # ── 1. REE Extractions ──────────────────────────────────────────
    logger.info("Starting REE extractions from %s to %s", start, end)

    with REEClient() as ree:
        # Demand: 5-min data, resampled to hourly inside extractor
        demand_raw = ree.fetch_demand(
            REE_ENDPOINTS["demand"], start, end, label="demand"
        )
        # Generation: daily only (API limitation)
        gen_raw = ree.fetch_generation(
            REE_ENDPOINTS["generation_mix"], start, end, label="generation mix"
        )

    demand_df = extract_demand(demand_raw)
    gen_daily_df = extract_generation_mix(gen_raw)

    logger.info(
        "REE extraction complete: demand=%d hourly rows, generation=%d daily rows",
        len(demand_df), len(gen_daily_df),
    )

    # ── 2. Build hourly index from demand, broadcast generation ─────
    demand_df = set_datetime_index(demand_df)
    demand_df = reindex_hourly(demand_df)

    gen_hourly = broadcast_daily_to_hourly(gen_daily_df, demand_df.index)
    merged = demand_df.join(gen_hourly, how="left")

    # ── 3. ESIOS or fallback ────────────────────────────────────────
    if esios_token:
        logger.info("Fetching ESIOS forecasts with token")
        with ESIOSClient(esios_token) as esios:
            for col_name, indicator_id in ESIOS_INDICATORS.items():
                raw = esios.fetch_indicator(
                    indicator_id, start, end, label=col_name
                )
                esios_df = extract_esios_forecast(raw, col_name)
                if not esios_df.empty:
                    esios_df = set_datetime_index(esios_df)
                    merged = merged.join(esios_df, how="left")
    else:
        merged = _create_forecast_fallback(merged)

    # ── 4. Clean ────────────────────────────────────────────────────
    merged = impute_gaps(merged)
    merged = add_local_hour(merged)

    # Ensure all output columns exist
    for col in OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA

    # ── 5. Validate ─────────────────────────────────────────────────
    report = validate_dataset(merged)
    logger.info("Validation passed: %s", report.passed)
    for msg in report.messages:
        logger.info("  %s", msg)

    # ── 6. Save ─────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME

    # Select final columns + local_hour
    final_cols = OUTPUT_COLUMNS + ["local_hour"]
    final_cols = [c for c in final_cols if c in merged.columns]
    merged[final_cols].to_parquet(output_path, engine="pyarrow")

    logger.info("Saved %d rows to %s", len(merged), output_path)
    return output_path
