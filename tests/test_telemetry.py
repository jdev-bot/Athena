"""Tests for telemetry module."""
import pytest
from fastapi.testclient import TestClient

from athena.services.telemetry import (
    TelemetryCollector,
    generate_metrics,
    metrics_content_type,
)


@pytest.fixture
def tele():
    return TelemetryCollector()


class TestTelemetryCollector:
    def test_record_evaluation(self, tele):
        tele.record_evaluation(5)
        metrics = generate_metrics()
        assert b'athena_evaluations_total' in metrics

    def test_record_signal(self, tele):
        tele.record_signal("long")
        tele.record_signal("short")
        metrics = generate_metrics()
        assert b'athena_signals_total{direction="long"}' in metrics
        assert b'athena_signals_total{direction="short"}' in metrics

    def test_kill_switch(self, tele):
        assert tele._kill is False
        tele.set_kill_switch(True)
        assert tele._kill is True
        assert b'athena_kill_switch_active 1' in generate_metrics()

    def test_pnl(self, tele):
        tele.set_pnl(0.234)
        assert b'athena_pnl 0.234' in generate_metrics()

    def test_scheduler_cycle_and_promotion(self, tele):
        tele.record_scheduler_cycle()
        tele.record_scheduler_promotion()
        metrics = generate_metrics()
        assert b'athena_scheduler_cycles_total 1' in metrics
        assert b'athena_scheduler_promotions_total 1' in metrics

    def test_generate_metrics_content_type(self):
        assert "text/plain" in metrics_content_type()


class TestTelemetryIntegration:
    @pytest.fixture
    def client(self):
        from athena.services.api import app
        return TestClient(app)

    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert b"athena" in response.content

    def test_metrics_has_counters(self, client):
        response = client.get("/metrics")
        body = response.content
        assert b"athena_strategies_total" in body
        assert b"athena_api_requests_total" in body
