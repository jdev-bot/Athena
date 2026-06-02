"""Tests for real market data downloader."""
import pytest
from pathlib import Path

from athena.market.downloader import MarketDataDownloader, DEFAULT_DATA_DIR


@pytest.fixture
def downloader(tmp_path):
    return MarketDataDownloader(exchange="binance", data_dir=tmp_path / "data")


class TestMarketDataDownloader:
    def test_init_creates_exchange_dir(self, downloader):
        assert downloader._exchange_dir.exists()

    def test_file_path(self, downloader):
        p = downloader._file_path("BTC/USDT", "1h")
        assert "BTC_USDT-1h.feather" in str(p)

    def test_ensure_data_missing(self, downloader, monkeypatch):
        # Mock the internal _download to avoid actual network call
        calls = []
        def fake_download(pairs, tfs, days=None, timerange=None):
            calls.append((pairs, tfs, days, timerange))
        monkeypatch.setattr(downloader, "_download", fake_download)
        path = downloader.ensure_data(pair="BTC/USDT", timeframe="1h", days=7)
        assert len(calls) == 1
        assert path.name == "BTC_USDT-1h.feather"

    def test_ensure_data_already_present(self, downloader, monkeypatch):
        # Pre-create file
        downloader._exchange_dir.mkdir(parents=True, exist_ok=True)
        dummy = downloader._exchange_dir / "BTC_USDT-1h.feather"
        dummy.write_text("not real data")
        calls = []
        def fake_download(*a, **k):
            calls.append((a, k))
        monkeypatch.setattr(downloader, "_download", fake_download)
        path = downloader.ensure_data(pair="BTC/USDT", timeframe="1h")
        assert len(calls) == 0
        assert path == dummy

    def test_purge(self, downloader):
        downloader._exchange_dir.mkdir(parents=True, exist_ok=True)
        (downloader._exchange_dir / "dummy.feather").touch()
        downloader.purge()
        assert not downloader._exchange_dir.exists()

