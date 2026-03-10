"""Constants, API URLs, indicator IDs, and column mappings."""

from datetime import date

# ── Date range ──────────────────────────────────────────────────────
DEFAULT_START = date(2021, 1, 1)
DEFAULT_END = date.today()

# ── REE Public API (no auth) ───────────────────────────────────────
REE_BASE_URL = "https://apidatos.ree.es/en/datos"
REE_GEO_LIMIT = "peninsular"

# demanda-tiempo-real returns 5-min data (Real + Forecasted + Scheduled)
# estructura-generacion only works with time_trunc=day (500s on hour)
REE_ENDPOINTS = {
    "demand": "demanda/demanda-tiempo-real",       # 5-min → resample hourly
    "generation_mix": "generacion/estructura-generacion",  # daily only
}

# Titles in English API responses (/en/ endpoint)
REE_DEMAND_TITLES = {
    "actual_demand_mw": "Real",
    "forecast_demand_mw": "Forecasted",
}

REE_GENERATION_TITLES = {
    "gen_wind_mw": "Wind",
    "gen_solar_pv_mw": "Solar photovoltaic",
    "gen_hydro_mw": "Hydro",
    "gen_combined_cycle_mw": "Combined cycle",
    "gen_nuclear_mw": "Nuclear",
}

# ── ESIOS API (token required) ─────────────────────────────────────
ESIOS_BASE_URL = "https://api.esios.ree.es"
ESIOS_GEO_ID = 8741  # Peninsular Spain

ESIOS_INDICATORS = {
    "forecast_wind_mw": 541,
    "forecast_solar_mw": 545,
}

# ── Output schema ──────────────────────────────────────────────────
OUTPUT_COLUMNS = [
    "actual_demand_mw",
    "forecast_demand_mw",
    "gen_wind_mw",
    "gen_solar_pv_mw",
    "gen_hydro_mw",
    "gen_combined_cycle_mw",
    "gen_nuclear_mw",
    "gen_total_mw",
    "forecast_wind_mw",
    "forecast_solar_mw",
]

# ── Validation thresholds ──────────────────────────────────────────
DEMAND_MIN_MW = 15_000
DEMAND_MAX_MW = 50_000
GENERATION_MIN_MW = 0
MAX_CONSECUTIVE_GAP_HOURS = 3
MAX_MISSING_PCT = 1.0  # percent

# ── File paths ─────────────────────────────────────────────────────
OUTPUT_FILENAME = "spain_grid_hourly.parquet"
