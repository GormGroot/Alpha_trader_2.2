"""
Tests for DataPipeline, MarketCalendar og IndicatorStore.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import DataPipeline, MarketCalendar, IndicatorStore


ET = ZoneInfo("US/Eastern")


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-02", periods=250, freq="B")
    close = 150 + np.cumsum(np.random.randn(250) * 2)
    return pd.DataFrame(
        {
            "Open": close + np.random.randn(250) * 0.5,
            "High": close + abs(np.random.randn(250)) * 1.5,
            "Low": close - abs(np.random.randn(250)) * 1.5,
            "Close": close,
            "Volume": np.random.randint(1_000_000, 10_000_000, size=250),
        },
        index=dates,
    )


@pytest.fixture
def pipeline(tmp_path):
    return DataPipeline(
        symbols=["AAPL", "MSFT"],
        interval="1d",
        check_interval=60,
        cache_dir=str(tmp_path / "cache"),
    )


@pytest.fixture
def store(tmp_path):
    return IndicatorStore(tmp_path / "indicators.db")


# ── MarketCalendar tests ────────────────────────────────────

class TestMarketCalendar:

    def test_weekday_during_hours_is_open(self):
        cal = MarketCalendar("US/Eastern")
        # Onsdag kl 12:00 ET
        dt = datetime(2024, 6, 5, 12, 0, tzinfo=ET)
        assert cal.is_market_open(dt) is True

    def test_weekday_before_open_is_closed(self):
        cal = MarketCalendar("US/Eastern")
        dt = datetime(2024, 6, 5, 8, 0, tzinfo=ET)
        assert cal.is_market_open(dt) is False

    def test_weekday_after_close_is_closed(self):
        cal = MarketCalendar("US/Eastern")
        dt = datetime(2024, 6, 5, 17, 0, tzinfo=ET)
        assert cal.is_market_open(dt) is False

    def test_saturday_is_closed(self):
        cal = MarketCalendar("US/Eastern")
        dt = datetime(2024, 6, 8, 12, 0, tzinfo=ET)  # lørdag
        assert cal.is_market_open(dt) is False

    def test_sunday_is_closed(self):
        cal = MarketCalendar("US/Eastern")
        dt = datetime(2024, 6, 9, 12, 0, tzinfo=ET)  # søndag
        assert cal.is_market_open(dt) is False

    def test_christmas_is_closed(self):
        cal = MarketCalendar("US/Eastern")
        # Juledag 2024 er en onsdag
        dt = datetime(2024, 12, 25, 12, 0, tzinfo=ET)
        assert cal.is_market_open(dt) is False

    def test_at_exact_open(self):
        cal = MarketCalendar("US/Eastern")
        dt = datetime(2024, 6, 5, 9, 30, tzinfo=ET)
        assert cal.is_market_open(dt) is True

    def test_at_exact_close(self):
        cal = MarketCalendar("US/Eastern")
        dt = datetime(2024, 6, 5, 16, 0, tzinfo=ET)
        assert cal.is_market_open(dt) is True

    def test_seconds_until_open_when_open(self):
        cal = MarketCalendar("US/Eastern")
        dt = datetime(2024, 6, 5, 12, 0, tzinfo=ET)
        assert cal.seconds_until_open(dt) == 0.0

    def test_seconds_until_open_friday_evening(self):
        cal = MarketCalendar("US/Eastern")
        # Fredag kl 18:00 → næste åbning er mandag 09:30
        dt = datetime(2024, 6, 7, 18, 0, tzinfo=ET)
        secs = cal.seconds_until_open(dt)
        # Mandag 09:30 - Fredag 18:00 = ~63.5 timer
        assert 63 * 3600 < secs < 64 * 3600

    def test_seconds_until_close_when_closed(self):
        cal = MarketCalendar("US/Eastern")
        dt = datetime(2024, 6, 8, 12, 0, tzinfo=ET)  # lørdag
        assert cal.seconds_until_close(dt) == 0.0

    def test_seconds_until_close_when_open(self):
        cal = MarketCalendar("US/Eastern")
        dt = datetime(2024, 6, 5, 15, 0, tzinfo=ET)
        secs = cal.seconds_until_close(dt)
        assert secs == 3600.0  # 1 time til 16:00


# ── IndicatorStore tests ─────────────────────────────────────

class TestIndicatorStore:

    def test_save_and_load(self, store, sample_ohlcv):
        store.save("AAPL", "1d", sample_ohlcv)
        loaded = store.load("AAPL", "1d")
        assert loaded is not None
        assert len(loaded) == len(sample_ohlcv)
        # Parquet round-trip may change freq attribute, so compare values only
        pd.testing.assert_frame_equal(loaded, sample_ohlcv, check_freq=False)

    def test_load_nonexistent_returns_none(self, store):
        assert store.load("NOPE", "1d") is None

    def test_save_overwrites(self, store, sample_ohlcv):
        store.save("AAPL", "1d", sample_ohlcv)
        smaller = sample_ohlcv.iloc[:10]
        store.save("AAPL", "1d", smaller)
        loaded = store.load("AAPL", "1d")
        assert len(loaded) == 10

    def test_load_all(self, store, sample_ohlcv):
        store.save("AAPL", "1d", sample_ohlcv)
        store.save("MSFT", "1d", sample_ohlcv)
        all_data = store.load_all("1d")
        assert "AAPL" in all_data
        assert "MSFT" in all_data
        assert len(all_data) == 2

    def test_load_all_filters_by_interval(self, store, sample_ohlcv):
        store.save("AAPL", "1d", sample_ohlcv)
        store.save("AAPL", "1h", sample_ohlcv)
        daily = store.load_all("1d")
        assert len(daily) == 1


# ── DataPipeline tests ──────────────────────────────────────

class TestDataPipeline:

    @patch("src.data.pipeline.MarketDataFetcher")
    def test_run_once_processes_all_symbols(self, mock_fetcher_cls, sample_ohlcv, tmp_path):
        mock_fetcher = MagicMock()
        mock_fetcher.get_historical.return_value = sample_ohlcv
        mock_fetcher_cls.return_value = mock_fetcher

        pipe = DataPipeline(
            symbols=["AAPL", "MSFT", "GOOGL"],
            cache_dir=str(tmp_path / "cache"),
        )
        pipe.fetcher = mock_fetcher

        results = pipe.run_once()

        assert len(results) == 3
        assert all(not df.empty for df in results.values())
        # Indikatorer bør være tilføjet
        assert "RSI" in results["AAPL"].columns
        assert "MACD" in results["AAPL"].columns
        assert "SMA_50" in results["MSFT"].columns

    @patch("src.data.pipeline.MarketDataFetcher")
    def test_run_once_handles_api_failure(self, mock_fetcher_cls, sample_ohlcv, tmp_path):
        mock_fetcher = MagicMock()
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("API timeout")
            return sample_ohlcv

        mock_fetcher.get_historical.side_effect = side_effect
        mock_fetcher_cls.return_value = mock_fetcher

        pipe = DataPipeline(
            symbols=["AAPL", "BAD"],
            cache_dir=str(tmp_path / "cache"),
        )
        pipe.fetcher = mock_fetcher

        results = pipe.run_once()

        assert not results["AAPL"].empty
        assert results["BAD"].empty

    @patch("src.data.pipeline.MarketDataFetcher")
    def test_run_once_uses_cached_on_failure(self, mock_fetcher_cls, sample_ohlcv, tmp_path):
        mock_fetcher = MagicMock()
        mock_fetcher_cls.return_value = mock_fetcher

        pipe = DataPipeline(
            symbols=["AAPL"],
            cache_dir=str(tmp_path / "cache"),
        )
        pipe.fetcher = mock_fetcher

        # Gem noget i store med SAMME interval som pipeline bruger
        pipe.store.save("AAPL", pipe.interval, sample_ohlcv)

        # Lad API fejle
        mock_fetcher.get_historical.side_effect = Exception("Down")
        results = pipe.run_once()

        assert not results["AAPL"].empty
        assert len(results["AAPL"]) == len(sample_ohlcv)

    @patch("src.data.pipeline.MarketDataFetcher")
    def test_run_once_normalizes_column_names(self, mock_fetcher_cls, tmp_path):
        """Cache returnerer lowercase – pipeline bør normalisere."""
        mock_fetcher = MagicMock()

        dates = pd.date_range("2024-01-02", periods=50, freq="B")
        df_lower = pd.DataFrame(
            {
                "open": np.random.randn(50) + 100,
                "high": np.random.randn(50) + 101,
                "low": np.random.randn(50) + 99,
                "close": np.random.randn(50) + 100,
                "volume": np.random.randint(1_000_000, 5_000_000, size=50),
            },
            index=dates,
        )
        mock_fetcher.get_historical.return_value = df_lower
        mock_fetcher_cls.return_value = mock_fetcher

        pipe = DataPipeline(
            symbols=["TEST"],
            cache_dir=str(tmp_path / "cache"),
        )
        pipe.fetcher = mock_fetcher

        results = pipe.run_once()
        assert "Close" in results["TEST"].columns
        assert "RSI" in results["TEST"].columns

    @patch("src.data.pipeline.MarketDataFetcher")
    def test_get_latest_returns_stored_data(self, mock_fetcher_cls, sample_ohlcv, tmp_path):
        mock_fetcher = MagicMock()
        mock_fetcher.get_historical.return_value = sample_ohlcv
        mock_fetcher_cls.return_value = mock_fetcher

        pipe = DataPipeline(
            symbols=["AAPL"],
            cache_dir=str(tmp_path / "cache"),
        )
        pipe.fetcher = mock_fetcher
        pipe.run_once()

        latest = pipe.get_latest("AAPL")
        assert latest is not None
        assert "RSI" in latest.columns

    @patch("src.data.pipeline.MarketDataFetcher")
    def test_indicators_are_computed(self, mock_fetcher_cls, sample_ohlcv, tmp_path):
        mock_fetcher = MagicMock()
        mock_fetcher.get_historical.return_value = sample_ohlcv
        mock_fetcher_cls.return_value = mock_fetcher

        pipe = DataPipeline(
            symbols=["AAPL"],
            cache_dir=str(tmp_path / "cache"),
        )
        pipe.fetcher = mock_fetcher
        results = pipe.run_once()

        df = results["AAPL"]
        expected_cols = [
            "SMA_20", "SMA_50", "SMA_200",
            "EMA_12", "EMA_26",
            "RSI",
            "MACD", "MACD_Signal", "MACD_Hist",
            "BB_Upper", "BB_Middle", "BB_Lower", "BB_Width",
            "Volume_SMA", "Volume_Ratio", "OBV",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Mangler {col}"

    def test_stop_event(self, pipeline):
        """stop() sætter event, så run-loop kan afsluttes."""
        assert not pipeline._stop_event.is_set()
        pipeline.stop()
        assert pipeline._stop_event.is_set()

    def test_pipeline_run_count(self, tmp_path, sample_ohlcv):
        with patch("src.data.pipeline.MarketDataFetcher") as mock_cls:
            mock_fetcher = MagicMock()
            mock_fetcher.get_historical.return_value = sample_ohlcv
            mock_cls.return_value = mock_fetcher

            pipe = DataPipeline(
                symbols=["AAPL"],
                cache_dir=str(tmp_path / "cache"),
            )
            pipe.fetcher = mock_fetcher

            pipe.run_once()
            pipe.run_once()
            pipe.run_once()
            assert pipe._run_count == 3
