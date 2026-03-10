"""HTTP clients for REE public API and ESIOS API."""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Any

import httpx
from dateutil.relativedelta import relativedelta
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
    ESIOS_BASE_URL,
    ESIOS_GEO_ID,
)

logger = logging.getLogger(__name__)

# REE demand endpoint returns 5-min data; 14-day chunks keep response sizes sane
_DEMAND_CHUNK_DAYS = 14
# Generation endpoint only works daily; month chunks are fine
_GENERATION_CHUNK_DAYS = 31


def _fixed_day_chunks(start: date, end: date, days: int) -> list[tuple[date, date]]:
    """Split a date range into chunks of at most `days` days."""
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _quarter_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split a date range into ~90-day chunks for ESIOS."""
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + relativedelta(months=3) - relativedelta(days=1), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + relativedelta(days=1)
    return chunks


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
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
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
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                logger.warning(
                    "REE %s chunk %s→%s failed: %s", endpoint, chunk_start, chunk_end, e
                )
        return responses

    def fetch_demand(self, endpoint: str, start: date, end: date, label: str = "demand") -> list[dict]:
        """Fetch demand endpoint (5-min data) in 14-day chunks."""
        return self._fetch_chunks(endpoint, start, end, _DEMAND_CHUNK_DAYS, "hour", label)

    def fetch_generation(self, endpoint: str, start: date, end: date, label: str = "generation") -> list[dict]:
        """Fetch generation endpoint (daily only) in month-sized chunks."""
        return self._fetch_chunks(endpoint, start, end, _GENERATION_CHUNK_DAYS, "day", label)


class ESIOSClient:
    """Client for the ESIOS API (api.esios.ree.es). Requires auth token."""

    def __init__(self, token: str, timeout: float = 30.0):
        self._client = httpx.Client(
            base_url=ESIOS_BASE_URL,
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Token token={token}",
            },
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ESIOSClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        reraise=True,
    )
    def _get(self, path: str, params: dict[str, str]) -> dict:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def fetch_indicator(
        self,
        indicator_id: int,
        start: date,
        end: date,
        label: str = "",
    ) -> list[dict]:
        """Fetch an ESIOS indicator in quarterly chunks."""
        chunks = _quarter_chunks(start, end)
        responses: list[dict] = []
        desc = label or f"indicator {indicator_id}"

        for chunk_start, chunk_end in tqdm(chunks, desc=f"ESIOS {desc}"):
            params = {
                "start_date": f"{chunk_start}T00:00:00Z",
                "end_date": f"{chunk_end}T23:59:59Z",
                "geo_ids[]": str(ESIOS_GEO_ID),
            }
            try:
                data = self._get(f"/indicators/{indicator_id}", params)
                responses.append(data)
            except httpx.HTTPStatusError as e:
                logger.warning(
                    "ESIOS indicator %d chunk %s→%s failed: %s",
                    indicator_id, chunk_start, chunk_end, e,
                )
        return responses
