# Lessons — Spain Energy Grid Risk Index

## joblib + custom dataclass requires PYTHONPATH

**Pattern:** `joblib.load("risk_index_fit.joblib")` fails with `ModuleNotFoundError: No module named 'grid_risk'` when the package isn't on the import path.
**Solution:** Always use `PYTHONPATH=src` when running scripts or loading artifacts that contain custom dataclasses.

## LightGBM feature name warnings with numpy arrays

**Pattern:** `UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names` when passing `.values` (numpy) to a model fitted with DataFrames.
**Solution:** Pass DataFrames consistently to `fit()` and `predict()`. If Optuna objective must pass arrays, suppress with `-W ignore::UserWarning` or accept the harmless warning.

## Quantile model RMSE is expected to be worse than baseline

**Pattern:** Negative RMSE skill score when using quantile regression (alpha=0.90) — looks like a bug but is by design.
**Solution:** Don't optimize for RMSE with quantile models. The model intentionally overshoots to minimize costly FNs. Evaluate with business metrics (cost penalty, F3) instead.

## ESIOS token has restricted access

**Pattern:** ESIOS API token `a90d42...` returns 403 Forbidden on all indicators (541, 545, 600, 805, 1739, etc.).
**Solution:** Use REE public API instead. Demand forecast comes from `demanda-tiempo-real` (hourly, Forecasted series). Wind/solar forecasts are unavailable — use lagged actuals or omit. Spot price available from `mercados/precios-mercados-tiempo-real` (hourly).

## REE demand endpoint only supports hourly granularity

**Pattern:** `demanda/demanda-tiempo-real` with `time_trunc=day` returns HTTP 500.
**Solution:** Fetch with `time_trunc=hour` and resample to daily mean in the extractor. The `demanda/evolucion` endpoint works for daily totals but only returns a single "Demand" series (no forecast).

## Open-Meteo archive API parameter name change

**Pattern:** The parameter is `wind_speed_10m_max` (with underscore), not `windspeed_10m_max`.
**Solution:** Check the Open-Meteo docs for exact parameter names. The archive API returns data through yesterday; for today/tomorrow use the forecast endpoint.
