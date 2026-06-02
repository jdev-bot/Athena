"""Integration tests for new API endpoints (forward run, signals, metrics)."""
import pytest
from fastapi.testclient import TestClient

from athena.services.api import app
from athena.services.models import init_db


@pytest.fixture
def client():
    init_db()
    # fresh DB — clear any residual rows from prior test modules
    from athena.services.models import get_session, Signal
    session = get_session()
    session.query(Signal).delete()
    session.commit()
    session.close()
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "athena" in body["service"]


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert b"athena" in response.content


class TestSignalsEndpoint:
    def test_signals_list_empty(self, client):
        response = client.get("/signals")
        assert response.status_code == 200
        assert response.json() == []

    def test_signals_filter_by_status(self, client):
        response = client.get("/signals?status=open")
        assert response.status_code == 200
        assert response.json() == []


class TestForwardRunEndpoint:
    @pytest.mark.timeout(10)
    def test_forward_run_missing_strategy(self, client):
        response = client.post("/forward/run", json={"strategy_id": "missing_id"})
        assert response.status_code in (500, 422, 404)
