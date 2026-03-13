"""
Module 1: Constants, API URLs, indicator IDs, and column mappings for daily pipeline.
"""

from datetime import date

# ---------------------------------------------------------------------
# 1. Date ranges: From 2021 to current date
DEFAULT_START = date(2021, 1, 1)
DEFAULT_END = date.today()

# ---------------------------------------------------------------------
# 2. BASE URL + GEO
# - Define base_url for REE API
# - Location parameter
REE_BASE_URL = "https://apidatos.ree.es/en/datos"
REE_GEO_LIMIT = "peninsular"


# ---------------------------------------------------------------------
# 3. API ENDPOINTS: DEMAND, GENERATION, PRICE
# Define as dict() the end-points used for demand, generation and prices
REE_ENDPOINTS = {
    "demand": "demanda/demanda-tiempo-real",  # hourly → resample daily
    "generation_mix": "generacion/estructura-generacion",  # daily native
    "spot_price": "mercados/precios-mercados-tiempo-real",  # hourly → daily
}

# ---------------------------------------------------------------------
# 4. API PARAMETERS: DEMAND
# Define dict() with:
# - keys contain the variable names used in this project for demand
# - values contain the titles used in the 'demand' end-point for the REE API
REE_DEMAND_TITLES = {
    "actual_demand_mw": "Real",
    "forecast_demand_mw": "Forecasted",
}

# ---------------------------------------------------------------------
# 5. API PARAMETERS: GENERATION
# Define dict() with:
# - keys contain the variable names used in this project for generation
# - values contain the titles used in the 'generation_mix' endpoint of the REE API
REE_GENERATION_TITLES = {
    "gen_wind_mw": "Wind",
    "gen_solar_pv_mw": "Solar photovoltaic",
    "gen_hydro_mw": "Hydro",
    "gen_combined_cycle_mw": "Combined cycle",
    "gen_nuclear_mw": "Nuclear",
}

# ---------------------------------------------------------------------
# 6. API PARAMETERS: PRICE
# Title used for spot prices of energy in the REE API
REE_SPOT_TITLE = "Spot market price"  # EUR/MWh


# ---------------------------------------------------------------------
# 7. BASE URLS FOR OPEN METEO
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# ---------------------------------------------------------------------
# 8. LOCATION PARAMETERS FOR SPAIN
WEATHER_LAT = 40.41
WEATHER_LON = -3.70
WEATHER_TIMEZONE = "Europe/Madrid"

WEATHER_DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
    "precipitation_sum",
]

# ---------------------------------------------------------------------
# 9. HEATING AND COOLING BASELINE TEMPERATURES
HDD_BASE_TEMP = 18.0
CDD_BASE_TEMP = 24.0

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
# 11. VALIDATION THRESHOLDDS
DEMAND_MIN_MW = 15_000  # daily mean demand floor
DEMAND_MAX_MW = 50_000  # daily mean demand ceiling
GENERATION_MIN_MW = 0
MAX_CONSECUTIVE_GAP_DAYS = 3
MAX_MISSING_PCT = 2.0  # percent — slightly more tolerant for daily

# ---------------------------------------------------------------------
# 12. OUTPUT DF PATH
OUTPUT_FILENAME = "merged_daily_data.parquet"
