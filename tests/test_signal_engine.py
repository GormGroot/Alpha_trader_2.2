"""
Tests for SignalEngine, SignalStore og EngineResult.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.indicators import add_all_indicators
from src.strategy.base_strategy import BaseStrategy, Signal, StrategyResult
from src.strategy.sma_crossover import SMACrossoverStrategy
from src.strategy.rsi_strategy import RSIStrategy
from src.strategy.signal_engine import (
    SignalEngine,
    SignalStore,
    SymbolSignal,
    EngineResult,
)


# ── Helpers ──────────────────────────────────────────────────

class StubStrategy(BaseStrategy):
    """Stub der returnerer et fast signal."""

    def __init__(self, sig: Signal, conf: float = 60, name_: str = "Stub"):
        self._sig = sig
        self._conf = conf
        self._name = name_

    @property
    def name(self) -> str:
        return self._name

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        return StrategyResult(self._sig, self._conf, f"{self._name} stub")


class FailingStrategy(BaseStrategy):
    """Strategi der altid fejler."""

    @property
    def name(self) -> str:
        return "Failing"

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        raise RuntimeError("Strategy crashed")


def _make_sample_data(symbols: list[str], n: int = 200) -> dict[str, pd.DataFrame]:
    """Generér syntetisk data for flere symboler."""
    np.random.seed(42)
    data = {}
    for sym in symbols:
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        close = 100 + np.cumsum(np.random.randn(n) * 2)
        df = pd.DataFrame(
            {
                "Open": close * 0.999,
                "High": close * 1.005,
                "Low": close * 0.995,
                "Close": close,
                "Volume": np.random.randint(1_000_000, 10_000_000, size=n),
            },
            index=dates,
        )
        add_all_indicators(df)
        data[sym] = df
    return data


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    return _make_sample_data(["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"])


@pytest.fixture
def store(tmp_path):
    return SignalStore(tmp_path / "signals.db")


@pytest.fixture
def engine_buy(tmp_path):
    """Engine hvor alle strategier siger BUY."""
    return SignalEngine(
        strategies=[
            (StubStrategy(Signal.BUY, 80, "S1"), 1.0),
            (StubStrategy(Signal.BUY, 60, "S2"), 1.0),
        ],
        min_agreement=2,
        portfolio_value=100_000,
        max_position_pct=0.05,
        cache_dir=str(tmp_path / "cache"),
    )


@pytest.fixture
def engine_mixed(tmp_path):
    """Engine med modstridende signaler."""
    return SignalEngine(
        strategies=[
            (StubStrategy(Signal.BUY, 80, "BuyStrat"), 1.0),
            (StubStrategy(Signal.SELL, 70, "SellStrat"), 1.0),
        ],
        min_agreement=2,
        portfolio_value=100_000,
        cache_dir=str(tmp_path / "cache"),
    )


@pytest.fixture
def engine_real(tmp_path):
    """Engine med ægte strategier."""
    return SignalEngine(
        strategies=[
            (SMACrossoverStrategy(short_window=20, long_window=50), 1.5),
            (RSIStrategy(period=14), 1.0),
        ],
        min_agreement=2,
        portfolio_value=100_000,
        cache_dir=str(tmp_path / "cache"),
    )


# ── SignalStore tests ────────────────────────────────────────

class TestSignalStore:

    def test_save_and_get_history(self, store):
        sig = SymbolSignal(
            symbol="AAPL", signal=Signal.BUY, confidence=75,
            position_size_usd=3750, reason="test", timestamp="2024-06-01T12:00:00",
            strategy_details=[{"strategy": "SMA", "signal": "BUY", "confidence": 75}],
        )
        store.save(sig)

        history = store.get_history("AAPL")
        assert len(history) == 1
        assert history.iloc[0]["symbol"] == "AAPL"
        assert history.iloc[0]["signal"] == "BUY"
        assert history.iloc[0]["confidence"] == 75

    def test_save_batch(self, store):
        sigs = [
            SymbolSignal(
                symbol=sym, signal=Signal.BUY, confidence=60,
                position_size_usd=3000, reason="batch test",
                timestamp="2024-06-01T12:00:00",
            )
            for sym in ["AAPL", "MSFT", "GOOGL"]
        ]
        store.save_batch(sigs)
        assert store.count() == 3

    def test_get_history_filters_by_symbol(self, store):
        for sym in ["AAPL", "MSFT", "AAPL"]:
            store.save(SymbolSignal(
                symbol=sym, signal=Signal.HOLD, confidence=0,
                position_size_usd=0, reason="test",
                timestamp="2024-06-01T12:00:00",
            ))

        aapl = store.get_history("AAPL")
        assert len(aapl) == 2

        msft = store.get_history("MSFT")
        assert len(msft) == 1

    def test_get_history_respects_limit(self, store):
        for i in range(20):
            store.save(SymbolSignal(
                symbol="AAPL", signal=Signal.HOLD, confidence=0,
                position_size_usd=0, reason=f"test {i}",
                timestamp=f"2024-06-{i+1:02d}T12:00:00",
            ))

        history = store.get_history("AAPL", limit=5)
        assert len(history) == 5

    def test_count(self, store):
        assert store.count() == 0
        store.save(SymbolSignal(
            symbol="AAPL", signal=Signal.BUY, confidence=50,
            position_size_usd=0, reason="", timestamp="now",
        ))
        assert store.count() == 1
        assert store.count("AAPL") == 1
        assert store.count("MSFT") == 0


# ── SymbolSignal tests ──────────────────────────────────────

class TestSymbolSignal:

    def test_is_actionable_buy(self):
        sig = SymbolSignal(
            symbol="X", signal=Signal.BUY, confidence=50,
            position_size_usd=1000, reason="", timestamp="",
        )
        assert sig.is_actionable is True

    def test_is_actionable_hold(self):
        sig = SymbolSignal(
            symbol="X", signal=Signal.HOLD, confidence=0,
            position_size_usd=0, reason="", timestamp="",
        )
        assert sig.is_actionable is False

    def test_is_actionable_zero_confidence(self):
        sig = SymbolSignal(
            symbol="X", signal=Signal.BUY, confidence=0,
            position_size_usd=0, reason="", timestamp="",
        )
        assert sig.is_actionable is False


# ── EngineResult tests ───────────────────────────────────────

class TestEngineResult:

    def _make_result(self, signals: list[SymbolSignal]) -> EngineResult:
        return EngineResult(timestamp="now", signals=signals, run_duration_ms=10)

    def test_actionable_sorted_by_confidence(self):
        signals = [
            SymbolSignal("A", Signal.BUY, 50, 100, "", ""),
            SymbolSignal("B", Signal.BUY, 90, 100, "", ""),
            SymbolSignal("C", Signal.HOLD, 0, 0, "", ""),
        ]
        result = self._make_result(signals)
        actionable = result.actionable
        assert len(actionable) == 2
        assert actionable[0].symbol == "B"  # højest confidence
        assert actionable[1].symbol == "A"

    def test_buys_and_sells(self):
        signals = [
            SymbolSignal("A", Signal.BUY, 80, 100, "", ""),
            SymbolSignal("B", Signal.SELL, 70, 100, "", ""),
            SymbolSignal("C", Signal.HOLD, 0, 0, "", ""),
        ]
        result = self._make_result(signals)
        assert len(result.buys) == 1
        assert len(result.sells) == 1
        assert result.buys[0].symbol == "A"
        assert result.sells[0].symbol == "B"


# ── SignalEngine tests ───────────────────────────────────────

class TestSignalEngine:

    def test_requires_at_least_one_strategy(self, tmp_path):
        with pytest.raises(ValueError, match="Mindst én"):
            SignalEngine(strategies=[], cache_dir=str(tmp_path / "c"))

    def test_process_returns_engine_result(self, engine_buy, sample_data):
        result = engine_buy.process(sample_data)
        assert isinstance(result, EngineResult)
        assert len(result.signals) == 5
        assert result.run_duration_ms >= 0

    def test_all_buy_consensus(self, engine_buy, sample_data):
        result = engine_buy.process(sample_data)
        for sig in result.signals:
            assert sig.signal == Signal.BUY
            assert sig.confidence > 0
            assert sig.position_size_usd > 0

    def test_mixed_gives_hold(self, engine_mixed, sample_data):
        result = engine_mixed.process(sample_data)
        for sig in result.signals:
            assert sig.signal == Signal.HOLD

    def test_position_sizing(self, engine_buy, sample_data):
        result = engine_buy.process(sample_data, portfolio_value=200_000)
        for sig in result.signals:
            # max_position_pct=0.05 → max $10,000
            assert sig.position_size_usd <= 10_000
            assert sig.position_size_usd > 0

    def test_empty_dataframe_gives_hold(self, engine_buy):
        data = {"EMPTY": pd.DataFrame()}
        result = engine_buy.process(data)
        assert result.signals[0].signal == Signal.HOLD
        assert "Ingen data" in result.signals[0].reason

    def test_signals_saved_to_store(self, engine_buy, sample_data):
        engine_buy.process(sample_data)
        assert engine_buy.store.count() == 5

        engine_buy.process(sample_data)
        assert engine_buy.store.count() == 10

    def test_strategy_details_populated(self, engine_buy, sample_data):
        result = engine_buy.process(sample_data)
        sig = result.signals[0]
        assert len(sig.strategy_details) == 2
        assert sig.strategy_details[0]["strategy"] == "S1"
        assert sig.strategy_details[1]["strategy"] == "S2"

    def test_failing_strategy_handled(self, tmp_path, sample_data):
        engine = SignalEngine(
            strategies=[
                (FailingStrategy(), 1.0),
                (StubStrategy(Signal.BUY, 70, "Good"), 1.0),
            ],
            min_agreement=1,
            cache_dir=str(tmp_path / "cache"),
        )
        result = engine.process(sample_data)
        # Bør ikke crashe – failing strategy giver HOLD
        assert len(result.signals) == 5

    def test_actionable_ranking(self, tmp_path):
        """Symboler med højere confidence bør rangeres først."""
        engine = SignalEngine(
            strategies=[
                (StubStrategy(Signal.BUY, 90, "High"), 1.0),
                (StubStrategy(Signal.BUY, 90, "High2"), 1.0),
            ],
            min_agreement=2,
            cache_dir=str(tmp_path / "cache"),
        )
        data = _make_sample_data(["A", "B", "C"])
        result = engine.process(data)

        # Alle bør have BUY med confidence=90
        assert len(result.actionable) == 3
        assert all(s.confidence == 90 for s in result.actionable)

    def test_update_portfolio_value(self, engine_buy, sample_data):
        engine_buy.update_portfolio_value(50_000)
        result = engine_buy.process(sample_data)
        for sig in result.signals:
            # 5% af 50k = 2500 max
            assert sig.position_size_usd <= 2500

    def test_with_real_strategies(self, engine_real, sample_data):
        """Integrationstest med SMA + RSI strategier."""
        result = engine_real.process(sample_data)
        assert len(result.signals) == 5

        for sig in result.signals:
            assert sig.symbol in sample_data
            assert isinstance(sig.signal, Signal)
            assert 0 <= sig.confidence <= 100
            # Strategies may return HOLD/0 confidence (no signal) which
            # is skipped by the engine — so strategy_details may be empty
            assert len(sig.strategy_details) <= 2

    def test_run_count_increments(self, engine_buy, sample_data):
        assert engine_buy._run_count == 0
        engine_buy.process(sample_data)
        assert engine_buy._run_count == 1
        engine_buy.process(sample_data)
        assert engine_buy._run_count == 2

    def test_parallel_processing_many_symbols(self, tmp_path):
        """Test at parallel processering virker med mange symboler."""
        engine = SignalEngine(
            strategies=[
                (StubStrategy(Signal.BUY, 70, "A"), 1.0),
                (StubStrategy(Signal.BUY, 80, "B"), 1.0),
            ],
            min_agreement=2,
            cache_dir=str(tmp_path / "cache"),
            max_workers=4,
        )
        data = _make_sample_data([f"SYM{i}" for i in range(20)], n=50)
        result = engine.process(data)

        assert len(result.signals) == 20
        assert all(s.signal == Signal.BUY for s in result.signals)
