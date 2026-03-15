"""
Module 1: Constants, API URLs, indicator IDs, and column mappings for daily pipeline.
Now loads configuration from a centralized settings.json file.
"""

import json
from datetime import date
from pathlib import Path

# Load settings.json
ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT / "config" / "settings.json"

with open(CONFIG_PATH, "r") as f:
    _cfg = json.load(f)

# ---------------------------------------------------------------------
# 1. Date ranges: From 2021 to current date
DEFAULT_START = date(2021, 1, 1)
DEFAULT_END = date.today()

# ---------------------------------------------------------------------
# 2. BASE URL + GEO
REE_BASE_URL = _cfg["api"]["ree_base_url"]
REE_GEO_LIMIT = _cfg["api"]["ree_geo_limit"]

# ---------------------------------------------------------------------
# 3. API ENDPOINTS
REE_ENDPOINTS = _cfg["endpoints"]["ree"]

# ---------------------------------------------------------------------
# 4. API PARAMETERS: DEMAND
REE_DEMAND_TITLES = _cfg["api_parameters"]["demand_titles"]

# ---------------------------------------------------------------------
# 5. API PARAMETERS: GENERATION
REE_GENERATION_TITLES = _cfg["api_parameters"]["generation_titles"]

# ---------------------------------------------------------------------
# 6. API PARAMETERS: PRICE
REE_SPOT_TITLE = _cfg["api_parameters"]["spot_title"]

# ---------------------------------------------------------------------
# 7. BASE URLS FOR OPEN METEO
OPEN_METEO_ARCHIVE_URL = _cfg["api"]["open_meteo_archive_url"]
OPEN_METEO_FORECAST_URL = _cfg["api"]["open_meteo_forecast_url"]

# ---------------------------------------------------------------------
# 8. LOCATION PARAMETERS FOR SPAIN
WEATHER_LAT = _cfg["location"]["weather_lat"]
WEATHER_LON = _cfg["location"]["weather_lon"]
WEATHER_TIMEZONE = _cfg["location"]["weather_timezone"]

WEATHER_DAILY_VARS = _cfg["features"]["weather_daily_vars"]

# ---------------------------------------------------------------------
# 9. HEATING AND COOLING BASELINE TEMPERATURES
HDD_BASE_TEMP = _cfg["features"]["hdd_base_temp"]
CDD_BASE_TEMP = _cfg["features"]["cdd_base_temp"]

# ---------------------------------------------------------------------
# 10. OUTPUT SCHEMA
# Complete list of features used in the model
OUTPUT_COLUMNS = [
    # REE demand (daily mean MW)
    "actual_demand_mw",
    "forecast_demand_mw",
    # REE generation (daily MWh converted to daily mean MW)
    "gen_wind_mw",
    "gen_solar_pv_mw",
    "gen_hydro_mw",
    "gen_combined_cycle_mw",
    "gen_nuclear_mw",
    "gen_total_mw",
    # Stream D: spot price (daily mean EUR/MWh)
    "spot_price_eur_mwh",
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
]

# ---------------------------------------------------------------------
# 11. VALIDATION THRESHOLDS
DEMAND_MIN_MW = _cfg["validation"]["demand_min_mw"]
DEMAND_MAX_MW = _cfg["validation"]["demand_max_mw"]
GENERATION_MIN_MW = _cfg["validation"]["generation_min_mw"]
MAX_CONSECUTIVE_GAP_DAYS = _cfg["validation"]["max_consecutive_gap_days"]
MAX_MISSING_PCT = _cfg["validation"]["max_missing_pct"]

# ---------------------------------------------------------------------
# 12. OUTPUT DF PATH
OUTPUT_FILENAME = "merged_daily_data.parquet"
