"""FastAPI service for Spain Energy Grid Risk Index prediction.

Accepts a target date and fetches live data from REE + Open-Meteo APIs
to produce a next-day risk forecast using the trained LightGBM model.

Model and risk_index_fit artifacts are baked into the Docker image at
build time under /app/models/.
"""

import logging
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from grid_risk.api_client import REEClient, OpenMeteoClient
from grid_risk.cleaning import (
    add_calendar_features,
    add_degree_days,
    ensure_date_index,
    impute_gaps,
)
from grid_risk.config import (
    OPEN_METEO_FORECAST_URL,
    REE_ENDPOINTS,
    WEATHER_DAILY_VARS,
    WEATHER_LAT,
    WEATHER_LON,
    WEATHER_TIMEZONE,
)
from grid_risk.extractors import (
    extract_demand_daily,
    extract_generation_daily,
    extract_spot_price_daily,
    extract_weather_daily,
)
from grid_risk.features import (
    assign_risk_category,
    feature_names as FEATURE_NAMES,
    RiskIndexFit,
    compute_core_factors,
    transform_risk_index,
)
from grid_risk.model import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state (loaded at startup)
# ---------------------------------------------------------------------------

model: Optional[Any] = None
risk_fit: Optional[RiskIndexFit] = None
run_id: Optional[str] = None

MODEL_DIR = Path("models")
RISK_FIT_PATH = MODEL_DIR / "risk_index_fit.joblib"
LGBM_MODEL_PATH = MODEL_DIR / "lgbm_model.joblib"
RUN_ID_PATH = Path("run_id.txt")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PredictionRequest(BaseModel):
    target_date: date = Field(
        ...,
        description=(
            "The date for which to predict the risk index. "
            "The system fetches weather forecasts and recent actuals "
            "to produce the prediction."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {"target_date": "2026-03-12"},
        }
    }


class PredictionResponse(BaseModel):
    date: str
    risk_index: float = Field(
        ..., ge=0.0, le=1.0, description="Predicted risk index (0=safe, 1=critical)"
    )
    risk_category: str = Field(
        ..., description="Low / Stable / Elevated / Severe / Extreme"
    )
    model_version: str


# ---------------------------------------------------------------------------
# Data fetching helpers (stateless, fetch per-request)
# ---------------------------------------------------------------------------


def _fetch_recent_data(target_date: date) -> pd.DataFrame:
    """
    Fetch the last 10 days of actual data from REE + Open-Meteo.

    We need at least 7 days of history to compute risk_index_lag_7d,
    plus the target date itself (weather forecast).
    Returns a DataFrame indexed by date with all raw columns.
    """
    # Fetch 10 days back to ensure we have 7+ valid lag values
    start = target_date - timedelta(days=10)
    end = target_date

    # -- REE data --
    with REEClient(rate_limit_sec=0.5) as ree:
        demand_raw = ree.fetch_demand_hourly(REE_ENDPOINTS["demand"], start, end)
        gen_raw = ree.fetch_generation_daily(
            REE_ENDPOINTS["generation_mix"], start, end
        )
        spot_raw = ree.fetch_spot_price_hourly(REE_ENDPOINTS["spot_price"], start, end)

    demand_df = extract_demand_daily(demand_raw)
    gen_df = extract_generation_daily(gen_raw)
    spot_df = extract_spot_price_daily(spot_raw)

    # -- Weather data --
    # For historical days use archive API; for today/tomorrow use forecast API
    yesterday = target_date - timedelta(days=1)
    with OpenMeteoClient() as meteo:
        # Historical weather (archive)
        weather_raw = meteo.fetch_daily_weather(start, yesterday)

    weather_df = extract_weather_daily(weather_raw)

    # Fetch forecast weather for target_date from Open-Meteo forecast API
    forecast_weather = _fetch_forecast_weather(target_date)
    if forecast_weather is not None:
        weather_df = pd.concat([weather_df, forecast_weather])
        weather_df = weather_df[~weather_df.index.duplicated(keep="last")]

    # -- Merge --
    merged = ensure_date_index(demand_df) if not demand_df.empty else pd.DataFrame()

    if merged.empty:
        raise HTTPException(
            status_code=502,
            detail="Could not fetch demand data from REE API",
        )

    if not gen_df.empty:
        gen_df = ensure_date_index(gen_df)
        merged = merged.join(gen_df, how="left")
    if not spot_df.empty:
        spot_df = ensure_date_index(spot_df)
        merged = merged.join(spot_df, how="left")
    if not weather_df.empty:
        weather_df = ensure_date_index(weather_df)
        merged = merged.join(weather_df, how="left")

    # Calendar + degree days
    merged = add_calendar_features(merged)
    merged = add_degree_days(merged)
    merged = impute_gaps(merged, max_gap=3)

    return merged


def _fetch_forecast_weather(target_date: date) -> pd.DataFrame | None:
    """Fetch weather forecast for a specific date from Open-Meteo forecast API."""
    import httpx

    try:
        params = {
            "latitude": str(WEATHER_LAT),
            "longitude": str(WEATHER_LON),
            "start_date": target_date.isoformat(),
            "end_date": target_date.isoformat(),
            "daily": ",".join(WEATHER_DAILY_VARS),
            "timezone": WEATHER_TIMEZONE,
        }
        resp = httpx.get(OPEN_METEO_FORECAST_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        times = daily.get("time", [])
        if not times:
            return None

        row: dict[str, Any] = {"date": times[0]}
        for var in WEATHER_DAILY_VARS:
            vals = daily.get(var, [])
            row[var] = vals[0] if vals else None

        df = pd.DataFrame([row])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.set_index("date")
        return df

    except Exception as exc:
        logger.warning("Failed to fetch forecast weather: %s", exc)
        return None


def _build_features(
    recent_data: pd.DataFrame,
    target_date: date,
    fit: RiskIndexFit,
) -> pd.DataFrame:
    """Build the 15-feature vector for the target date.

    Computes risk_index for all recent days using the fitted PCA pipeline,
    then derives lag features for the target date.
    """
    # Compute core risk factors for all days in the window
    factors = compute_core_factors(recent_data)
    risk_index = transform_risk_index(factors, fit)
    recent_data["risk_index"] = risk_index

    # Compute lag features
    recent_data["risk_index_lag_1d"] = recent_data["risk_index"].shift(1)
    recent_data["risk_index_lag_7d"] = recent_data["risk_index"].shift(7)

    # Extract the target date row
    if target_date not in recent_data.index:
        # If target_date data isn't available yet, use the latest available
        # and forward-fill the features
        available_dates = sorted(recent_data.index)
        if not available_dates:
            raise HTTPException(
                status_code=502,
                detail="No data available for prediction",
            )
        target_date = available_dates[-1]

    row = recent_data.loc[[target_date]]

    # Check which features are available
    missing_features = [f for f in FEATURE_NAMES if f not in row.columns]

    if missing_features:
        logger.warning("Missing features (will use 0): %s", missing_features)
        for f in missing_features:
            row[f] = 0.0

    return row[FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Lifespan: load model at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, risk_fit, run_id

    # Load run ID
    if RUN_ID_PATH.exists():
        run_id = RUN_ID_PATH.read_text().strip()
        logger.info("[startup] Run ID: %s", run_id)
    else:
        logger.warning("[startup] run_id.txt not found")
        run_id = None

    # Load LightGBM model
    if LGBM_MODEL_PATH.exists():
        try:
            model = joblib.load(LGBM_MODEL_PATH)
            logger.info("[startup] Model loaded from %s", LGBM_MODEL_PATH)
        except Exception as e:
            logger.error("[startup] Failed to load model: %s", e)
            model = None
    else:
        logger.warning("[startup] Model not found at %s", LGBM_MODEL_PATH)
        model = None

    # Load risk index fit (PCA pipeline + thresholds)
    if RISK_FIT_PATH.exists():
        try:
            risk_fit = joblib.load(RISK_FIT_PATH)
            logger.info("[startup] Risk fit loaded from %s", RISK_FIT_PATH)
        except Exception as e:
            logger.error("[startup] Failed to load risk fit: %s", e)
            risk_fit = None
    else:
        logger.warning("[startup] Risk fit not found at %s", RISK_FIT_PATH)
        risk_fit = None

    yield


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Spain Energy Grid Risk Index",
    description=(
        "Predict the daily risk index for the Spanish electrical grid. "
        "Accepts a date, fetches live data from REE and Open-Meteo, "
        "and returns a quantile-based risk prediction (0-1)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api")
def root():
    return {
        "message": "Spain Energy Grid Risk Index API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if (model is not None and risk_fit is not None) else "degraded",
        "run_id": run_id or "unknown",
        "model_loaded": model is not None,
        "risk_fit_loaded": risk_fit is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(request: PredictionRequest):
    if model is None or risk_fit is None:
        raise HTTPException(
            status_code=503,
            detail="Model or risk fit not loaded. Check /health.",
        )

    target = request.target_date
    logger.info("Prediction requested for date: %s", target)

    try:
        # 1. Fetch recent data from APIs
        recent_data = _fetch_recent_data(target)

        # 2. Build feature vector
        features = _build_features(recent_data, target, risk_fit)

        # 3. Predict
        y_pred = predict(model, features.values)
        risk_value = float(np.clip(y_pred[0], 0.0, 1.0))

        # 4. Assign category
        category = assign_risk_category(
            np.array([risk_value]),
            risk_fit.thresholds,
        )[0]

        return PredictionResponse(
            date=target.isoformat(),
            risk_index=round(risk_value, 4),
            risk_category=category,
            model_version=run_id or "unknown",
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed for date %s", target)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


# Mount the webapp after ALL API routes
# This serves index.html at the root "/" automatically
app.mount("/", StaticFiles(directory="webapp", html=True), name="webapp")


# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=9696, reload=True)
