"""Integration tests for the Spain Energy Grid Risk Index API.

Requires the server running (e.g., via Docker or `python app.py`).
Tests issue real HTTP requests against localhost:9696.

Usage:
    pytest test_api.py -v
"""

import requests

BASE_URL = "http://localhost:9696"


def test_root_endpoint():
    resp = requests.get(f"{BASE_URL}/")
    assert resp.status_code == 200, (
        f"Unexpected status: {resp.status_code} body={resp.text}"
    )
    data = resp.json()
    assert "message" in data
    assert "Spain" in data["message"]


def test_health_endpoint():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, (
        f"Unexpected status: {resp.status_code} body={resp.text}"
    )
    data = resp.json()
    assert data.get("status") == "ok", f"Health check not ok: {data}"
    assert data.get("model_loaded") is True
    assert data.get("risk_fit_loaded") is True
    assert isinstance(data.get("run_id"), str) and len(data["run_id"]) > 5


def test_predict_endpoint():
    payload = {"target_date": "2026-03-10"}
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    assert resp.status_code == 200, (
        f"Unexpected status: {resp.status_code} body={resp.text}"
    )
    data = resp.json()

    # Validate response structure
    assert "risk_index" in data
    assert "risk_category" in data
    assert "date" in data
    assert "model_version" in data

    # Validate types and ranges
    assert isinstance(data["risk_index"], float)
    assert 0.0 <= data["risk_index"] <= 1.0
    assert data["risk_category"] in ("Low", "Stable", "Elevated", "Severe", "Extreme")
    assert isinstance(data["model_version"], str) and len(data["model_version"]) > 5


def test_predict_invalid_date():
    payload = {"target_date": "not-a-date"}
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    assert resp.status_code == 422, (
        f"Expected 422 for invalid date, got {resp.status_code}"
    )
