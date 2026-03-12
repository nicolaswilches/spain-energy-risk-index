"""HTTP clients for REE public API and Open-Meteo API (daily pipeline)."""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm

from grid_risk.config import (
    REE_BASE_URL,
    REE_GEO_LIMIT,
    OPEN_METEO_ARCHIVE_URL,
    WEATHER_LAT,
    WEATHER_LON,
    WEATHER_TIMEZONE,
    WEATHER_DAILY_VARS,
)

logger = logging.getLogger(__name__)

# REE demand/spot endpoints return hourly data; 14-day chunks keep sizes sane
_HOURLY_CHUNK_DAYS = 14
# Generation endpoint is daily; month chunks are fine
_DAILY_CHUNK_DAYS = 31
# Open-Meteo archive supports up to ~1 year per request
_WEATHER_CHUNK_DAYS = 365


# -- Chunking helpers ---------------------------------------------------------


def _fixed_day_chunks(
    start: date,
    end: date,
    days: int,
) -> list[tuple[date, date]]:
    """Split a date range into chunks of at most *days* days."""
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


# -- REE Client ---------------------------------------------------------------


class REEClient:
    """Client for the REE public API (apidatos.ree.es)."""

    def __init__(self, rate_limit_sec: float = 1.0, timeout: float = 120.0):
        self._rate_limit = rate_limit_sec
        self._last_request_time = 0.0
        self._client = httpx.Client(
            base_url=REE_BASE_URL,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> REEClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self._rate_limit:
            time.sleep(self._rate_limit - elapsed)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.TransportError),
        ),
        reraise=True,
    )
    def _get(self, endpoint: str, params: dict[str, str]) -> dict:
        self._throttle()
        resp = self._client.get(endpoint, params=params)
        self._last_request_time = time.monotonic()
        resp.raise_for_status()
        return resp.json()

    def _fetch_chunks(
        self,
        endpoint: str,
        start: date,
        end: date,
        chunk_days: int,
        time_trunc: str,
        label: str,
    ) -> list[dict]:
        chunks = _fixed_day_chunks(start, end, chunk_days)
        responses: list[dict] = []

        for chunk_start, chunk_end in tqdm(chunks, desc=f"REE {label}"):
            params = {
                "start_date": f"{chunk_start}T00:00",
                "end_date": f"{chunk_end}T23:59",
                "time_trunc": time_trunc,
                "geo_limit": REE_GEO_LIMIT,
            }
            try:
                data = self._get(endpoint, params)
                responses.append(data)
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                logger.warning(
                    "REE %s chunk %s->%s failed: %s",
                    endpoint,
                    chunk_start,
                    chunk_end,
                    exc,
                )
        return responses

    def fetch_demand_hourly(
        self,
        endpoint: str,
        start: date,
        end: date,
    ) -> list[dict]:
        """Fetch demand endpoint (hourly) in 14-day chunks."""
        return self._fetch_chunks(
            endpoint,
            start,
            end,
            _HOURLY_CHUNK_DAYS,
            "hour",
            "demand",
        )

    def fetch_generation_daily(
        self,
        endpoint: str,
        start: date,
        end: date,
    ) -> list[dict]:
        """Fetch generation endpoint (daily) in month-sized chunks."""
        return self._fetch_chunks(
            endpoint,
            start,
            end,
            _DAILY_CHUNK_DAYS,
            "day",
            "generation",
        )

    def fetch_spot_price_hourly(
        self,
        endpoint: str,
        start: date,
        end: date,
    ) -> list[dict]:
        """Fetch spot price endpoint (hourly) in 14-day chunks."""
        return self._fetch_chunks(
            endpoint,
            start,
            end,
            _HOURLY_CHUNK_DAYS,
            "hour",
            "spot price",
        )


# -- Open-Meteo Client -------------------------------------------------------


class OpenMeteoClient:
    """Client for the Open-Meteo Archive API (daily weather data)."""

    def __init__(self, timeout: float = 60.0):
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenMeteoClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, httpx.TransportError),
        ),
        reraise=True,
    )
    def _get(self, url: str, params: dict[str, str]) -> dict:
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def fetch_daily_weather(
        self,
        start: date,
        end: date,
    ) -> list[dict]:
        """Fetch daily weather data in yearly chunks from archive API."""
        chunks = _fixed_day_chunks(start, end, _WEATHER_CHUNK_DAYS)
        responses: list[dict] = []

        for chunk_start, chunk_end in tqdm(chunks, desc="Weather"):
            params = {
                "latitude": str(WEATHER_LAT),
                "longitude": str(WEATHER_LON),
                "start_date": chunk_start.isoformat(),
                "end_date": chunk_end.isoformat(),
                "daily": ",".join(WEATHER_DAILY_VARS),
                "timezone": WEATHER_TIMEZONE,
            }
            try:
                data = self._get(OPEN_METEO_ARCHIVE_URL, params)
                responses.append(data)
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                logger.warning(
                    "Open-Meteo chunk %s->%s failed: %s",
                    chunk_start,
                    chunk_end,
                    exc,
                )
        return responses
