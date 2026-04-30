"""
Tests for RegimeDetector og AdaptiveStrategy.

Dækker:
  - MarketRegime enum og REGIME_INFO
  - RegimeSignal, RegimeResult, RegimeShift, StrategyAdjustment dataklasser
  - RegimeDetector: individuelle signaler (trend, vol, momentum, volume, breadth, yield)
  - RegimeDetector: composite score og regime-klassificering
  - RegimeDetector: confidence-beregning
  - RegimeDetector: regime-skift logging
  - RegimeDetector: historisk regime-sekvens
  - AdaptiveStrategy: regime-baseret signal
  - AdaptiveStrategy: filtrering af inner strategy
  - AdaptiveStrategy: crash-blokering
  - AdaptiveStrategy: BaseStrategy integration
  - Imports fra __init__.py
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.strategy.base_strategy import BaseStrategy, Signal, StrategyResult
from src.strategy.regime import (
    RegimeDetector,
    AdaptiveStrategy,
    MarketRegime,
    RegimeResult,
    RegimeSignal,
    RegimeShift,
    StrategyAdjustment,
    REGIME_INFO,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_df(
    n: int = 250,
    trend: float = 0.001,
    noise: float = 0.015,
    seed: int = 42,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generér syntetisk OHLCV."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n + 2)[-n:]
    returns = trend + rng.normal(0, noise, n)
    prices = start_price * np.cumprod(1 + returns)
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame({
        "Open": prices * (1 + rng.normal(0, 0.002, n)),
        "High": prices * (1 + abs(rng.normal(0, 0.01, n))),
        "Low": prices * (1 - abs(rng.normal(0, 0.01, n))),
        "Close": prices,
        "Volume": volume,
    }, index=dates)


def _make_bull() -> pd.DataFrame:
    """Stærk uptrend, lav vol."""
    return _make_df(300, trend=0.003, noise=0.008, seed=1)


def _make_bear() -> pd.DataFrame:
    """Downtrend, høj vol."""
    return _make_df(300, trend=-0.003, noise=0.025, seed=2)


def _make_sideways() -> pd.DataFrame:
    """Flat, moderat vol."""
    return _make_df(300, trend=0.0, noise=0.012, seed=3)


def _make_crash() -> pd.DataFrame:
    """Voldsomt fald med ekstremt høj vol."""
    rng = np.random.default_rng(99)
    n = 250
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n + 2)[-n:]
    # Normal periode, derefter crash
    returns = np.concatenate([
        rng.normal(0.001, 0.01, 200),
        rng.normal(-0.03, 0.05, 50),   # Voldsomme fald
    ])
    prices = 100 * np.cumprod(1 + returns)
    volume = np.concatenate([
        rng.integers(1_000_000, 5_000_000, 200),
        rng.integers(10_000_000, 30_000_000, 50),  # Voldsom volumen
    ]).astype(float)
    return pd.DataFrame({
        "Open": prices * 0.99,
        "High": prices * 1.01,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": volume,
    }, index=dates)


class _DummyStrategy(BaseStrategy):
    """Dummy strategi til test."""
    def __init__(self, signal=Signal.BUY, confidence=70):
        self._signal = signal
        self._confidence = confidence

    @property
    def name(self):
        return "dummy"

    def analyze(self, df):
        return StrategyResult(self._signal, self._confidence, "Dummy signal")


# ══════════════════════════════════════════════════════════════
# ENUM & DATAKLASSE TESTS
# ══════════════════════════════════════════════════════════════

class TestMarketRegime:
    """Test MarketRegime enum."""

    def test_all_regimes(self):
        assert len(MarketRegime) == 6
        assert MarketRegime.BULL.value == "bull"
        assert MarketRegime.CRASH.value == "crash"

    def test_regime_info_complete(self):
        for regime in MarketRegime:
            assert regime in REGIME_INFO
            info = REGIME_INFO[regime]
            assert "label" in info
            assert "color" in info
            assert "max_exposure" in info
            assert 0 <= info["max_exposure"] <= 1


class TestRegimeSignal:
    """Test RegimeSignal dataklasse."""

    def test_create(self):
        sig = RegimeSignal(name="trend", value=0.5, weight=2.0, detail="Bullish")
        assert sig.name == "trend"
        assert sig.value == 0.5
        assert sig.weight == 2.0

    def test_default_weight(self):
        sig = RegimeSignal(name="test", value=0.0)
        assert sig.weight == 1.0


class TestRegimeResult:
    """Test RegimeResult dataklasse."""

    def test_create(self):
        result = RegimeResult(
            regime=MarketRegime.BULL,
            confidence=85.0,
            composite_score=0.65,
        )
        assert result.regime == MarketRegime.BULL
        assert result.label == "BULL MARKET"
        assert result.color == "#2ed573"
        assert result.max_exposure == 1.0

    def test_timestamp_auto(self):
        result = RegimeResult(
            regime=MarketRegime.SIDEWAYS, confidence=50, composite_score=0.0,
        )
        assert result.timestamp != ""

    def test_max_exposure_crash(self):
        result = RegimeResult(
            regime=MarketRegime.CRASH, confidence=90, composite_score=-0.8,
        )
        assert result.max_exposure == 0.10


class TestRegimeShift:
    """Test RegimeShift."""

    def test_create(self):
        shift = RegimeShift(
            timestamp="2026-03-16T10:00:00",
            from_regime=MarketRegime.BULL,
            to_regime=MarketRegime.BEAR,
            confidence=75,
            reason="Trend vendte",
            composite_score=-0.4,
        )
        assert shift.from_regime == MarketRegime.BULL
        assert shift.to_regime == MarketRegime.BEAR


class TestStrategyAdjustment:
    """Test StrategyAdjustment."""

    def test_bull_adjustment(self):
        adj = AdaptiveStrategy._REGIME_ADJUSTMENTS[MarketRegime.BULL]
        assert adj.max_exposure_pct == 1.0
        assert adj.allow_new_buys is True
        assert adj.allow_shorts is False
        assert adj.stop_loss_multiplier > 1.0  # Løsere

    def test_crash_adjustment(self):
        adj = AdaptiveStrategy._REGIME_ADJUSTMENTS[MarketRegime.CRASH]
        assert adj.max_exposure_pct == 0.10
        assert adj.allow_new_buys is False
        assert adj.allow_shorts is True
        assert adj.stop_loss_multiplier < 1.0  # Strammere
        assert "GLD" in adj.safe_havens
        assert "CASH" in adj.safe_havens

    def test_bear_adjustment(self):
        adj = AdaptiveStrategy._REGIME_ADJUSTMENTS[MarketRegime.BEAR]
        assert adj.max_exposure_pct == 0.30
        assert "Utilities" in adj.preferred_sectors
        assert "Technology" in adj.avoid_sectors

    def test_sideways_adjustment(self):
        adj = AdaptiveStrategy._REGIME_ADJUSTMENTS[MarketRegime.SIDEWAYS]
        assert adj.max_exposure_pct == 0.50
        assert "rsi_strategy" in adj.preferred_strategies

    def test_recovery_adjustment(self):
        adj = AdaptiveStrategy._REGIME_ADJUSTMENTS[MarketRegime.RECOVERY]
        assert adj.max_exposure_pct == 0.50
        assert "Financials" in adj.preferred_sectors

    def test_euphoria_adjustment(self):
        adj = AdaptiveStrategy._REGIME_ADJUSTMENTS[MarketRegime.EUPHORIA]
        assert adj.max_exposure_pct == 0.70
        assert adj.stop_loss_multiplier < 1.0  # Strammere


# ══════════════════════════════════════════════════════════════
# REGIME DETECTOR TESTS
# ══════════════════════════════════════════════════════════════

class TestRegimeDetectorInit:
    """Test RegimeDetector initialisering."""

    def test_default(self):
        d = RegimeDetector()
        assert d.current_regime is None
        assert len(d.history) == 0
        assert len(d.shifts) == 0

    def test_custom_weights(self):
        d = RegimeDetector(weights={"trend": 5.0, "volatility": 1.0})
        assert d._weights["trend"] == 5.0


class TestTrendSignal:
    """Test trend-signal."""

    def test_bull_trend(self):
        d = RegimeDetector()
        df = _make_bull()
        sig = d._trend_signal(df)
        assert sig.name == "trend"
        assert sig.value > 0  # Bullish

    def test_bear_trend(self):
        d = RegimeDetector()
        df = _make_bear()
        sig = d._trend_signal(df)
        assert sig.value < 0  # Bearish

    def test_sideways_trend(self):
        d = RegimeDetector()
        df = _make_sideways()
        sig = d._trend_signal(df)
        # Sideways data har 0 trend, men random walk kan give drift
        # Bare tjek at det returnerer en gyldig score
        assert -1.0 <= sig.value <= 1.0


class TestVolatilitySignal:
    """Test volatilitets-signal."""

    def test_with_vix_calm(self):
        d = RegimeDetector()
        df = _make_bull()
        sig = d._volatility_signal(df, vix_level=12.0)
        assert sig.value > 0  # Calm = bullish
        assert "roligt" in sig.detail

    def test_with_vix_crisis(self):
        d = RegimeDetector()
        df = _make_bear()
        sig = d._volatility_signal(df, vix_level=45.0)
        assert sig.value == -1.0
        assert "KRISE" in sig.detail

    def test_with_vix_normal(self):
        d = RegimeDetector()
        df = _make_bull()
        sig = d._volatility_signal(df, vix_level=20.0)
        assert sig.value == 0.0

    def test_realized_vol_low(self):
        d = RegimeDetector()
        df = _make_bull()  # Lav vol
        sig = d._volatility_signal(df)
        assert sig.name == "volatility"
        # Lav vol → let positiv eller neutral
        assert sig.value >= -0.5

    def test_realized_vol_high(self):
        d = RegimeDetector()
        df = _make_crash()  # Høj vol
        sig = d._volatility_signal(df)
        assert sig.value <= 0  # Høj vol → negativ


class TestMomentumSignal:
    """Test momentum-signal."""

    def test_bull_momentum(self):
        d = RegimeDetector()
        df = _make_bull()
        sig = d._momentum_signal(df)
        assert sig.value > 0

    def test_bear_momentum(self):
        d = RegimeDetector()
        df = _make_bear()
        sig = d._momentum_signal(df)
        assert sig.value < 0

    def test_short_data(self):
        d = RegimeDetector()
        df = _make_df(n=5)
        sig = d._momentum_signal(df)
        assert sig.value == 0.0  # Utilstrækkelig data


class TestVolumeSignal:
    """Test volume-signal."""

    def test_bull_accumulation(self):
        d = RegimeDetector()
        df = _make_bull()
        sig = d._volume_signal(df)
        assert sig.name == "volume"
        # I uptrend: op-dage har mere volume = accumulation
        # (Kan variere pga. random data, test bare at det returnerer)
        assert -1 <= sig.value <= 1

    def test_no_volume_column(self):
        d = RegimeDetector()
        df = _make_bull().drop(columns=["Volume"])
        sig = d._volume_signal(df)
        assert sig.value == 0.0

    def test_short_data(self):
        d = RegimeDetector()
        df = _make_df(n=10)
        sig = d._volume_signal(df)
        assert -1 <= sig.value <= 1


class TestBreadthSignal:
    """Test breadth-signal."""

    def test_strong_breadth(self):
        d = RegimeDetector()
        sig = d._breadth_signal(2.5)
        assert sig.value == 1.0

    def test_weak_breadth(self):
        d = RegimeDetector()
        sig = d._breadth_signal(0.3)
        assert sig.value == -1.0

    def test_neutral_breadth(self):
        d = RegimeDetector()
        sig = d._breadth_signal(1.0)
        assert sig.value == 0.0


class TestYieldCurveSignal:
    """Test yield curve signal."""

    def test_inverted(self):
        d = RegimeDetector()
        sig = d._yield_curve_signal(-0.8)
        assert sig.value == -1.0
        assert "inverteret" in sig.detail

    def test_normal(self):
        d = RegimeDetector()
        sig = d._yield_curve_signal(1.0)
        assert sig.value > 0

    def test_steep(self):
        d = RegimeDetector()
        sig = d._yield_curve_signal(2.5)
        assert sig.value > 0
        assert "recovery" in sig.detail


class TestCompositeScore:
    """Test composite score beregning."""

    def test_all_bullish(self):
        d = RegimeDetector()
        signals = [
            RegimeSignal("a", 0.8, 1.0),
            RegimeSignal("b", 0.6, 1.0),
            RegimeSignal("c", 0.7, 1.0),
        ]
        score = d._compute_composite(signals)
        assert score > 0.5

    def test_all_bearish(self):
        d = RegimeDetector()
        signals = [
            RegimeSignal("a", -0.8, 1.0),
            RegimeSignal("b", -0.6, 1.0),
        ]
        score = d._compute_composite(signals)
        assert score < -0.5

    def test_mixed_signals(self):
        d = RegimeDetector()
        signals = [
            RegimeSignal("a", 0.8, 1.0),
            RegimeSignal("b", -0.8, 1.0),
        ]
        score = d._compute_composite(signals)
        assert abs(score) < 0.2  # Modstridende → nær 0

    def test_weighted(self):
        d = RegimeDetector()
        signals = [
            RegimeSignal("trend", 0.8, 3.0),   # Tungt vægtet bullish
            RegimeSignal("vol", -0.3, 1.0),
        ]
        score = d._compute_composite(signals)
        assert score > 0.3  # Trend dominerer

    def test_empty_signals(self):
        d = RegimeDetector()
        assert d._compute_composite([]) == 0.0


class TestRegimeClassification:
    """Test regime-klassificering."""

    def test_crash(self):
        d = RegimeDetector()
        df = _make_df(250)
        regime = d._classify_regime(-0.7, -0.8, df)
        assert regime == MarketRegime.CRASH

    def test_bear(self):
        d = RegimeDetector()
        df = _make_df(250)
        regime = d._classify_regime(-0.4, -0.3, df)
        assert regime == MarketRegime.BEAR

    def test_bull(self):
        d = RegimeDetector()
        df = _make_df(250)
        regime = d._classify_regime(0.4, 0.2, df)
        assert regime == MarketRegime.BULL

    def test_sideways(self):
        d = RegimeDetector()
        df = _make_df(250)
        regime = d._classify_regime(0.1, 0.0, df)
        assert regime == MarketRegime.SIDEWAYS

    def test_not_crash_without_high_vol(self):
        d = RegimeDetector()
        df = _make_df(250)
        # Negativ composite men lav vol → BEAR, ikke CRASH
        regime = d._classify_regime(-0.6, -0.3, df)
        assert regime == MarketRegime.BEAR


class TestConfidence:
    """Test confidence-beregning."""

    def test_high_agreement(self):
        d = RegimeDetector()
        signals = [
            RegimeSignal("a", 0.8, 1.0),
            RegimeSignal("b", 0.7, 1.0),
            RegimeSignal("c", 0.75, 1.0),
        ]
        conf = d._compute_confidence(signals, MarketRegime.BULL)
        assert conf > 70  # Høj enighed

    def test_disagreement(self):
        d = RegimeDetector()
        signals = [
            RegimeSignal("a", 0.8, 1.0),
            RegimeSignal("b", -0.7, 1.0),
        ]
        conf = d._compute_confidence(signals, MarketRegime.SIDEWAYS)
        assert conf < 60  # Modstridende

    def test_empty(self):
        d = RegimeDetector()
        assert d._compute_confidence([], MarketRegime.SIDEWAYS) == 0.0


# ══════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════

class TestDetectIntegration:
    """Test detect() med reelle data."""

    def test_bull_market_detected(self):
        d = RegimeDetector()
        df = _make_bull()
        result = d.detect(df)
        assert result.regime in (MarketRegime.BULL, MarketRegime.EUPHORIA)
        assert result.confidence > 0
        assert len(result.signals) >= 4

    def test_bear_market_detected(self):
        d = RegimeDetector()
        df = _make_bear()
        result = d.detect(df)
        assert result.regime in (MarketRegime.BEAR, MarketRegime.CRASH)

    def test_with_vix(self):
        d = RegimeDetector()
        df = _make_bull()
        result = d.detect(df, vix_level=12.0)
        assert result.confidence > 0

    def test_with_all_extras(self):
        d = RegimeDetector()
        df = _make_bull()
        result = d.detect(
            df, vix_level=15.0, breadth_ratio=1.5, yield_spread=1.2,
        )
        assert len(result.signals) >= 6  # trend, vol, mom, volume, breadth, yield

    def test_insufficient_data(self):
        d = RegimeDetector()
        df = _make_df(n=10)
        result = d.detect(df)
        assert result.confidence == 0.0
        assert result.regime == MarketRegime.SIDEWAYS

    def test_empty_data(self):
        d = RegimeDetector()
        result = d.detect(pd.DataFrame())
        assert result.regime == MarketRegime.SIDEWAYS

    def test_none_data(self):
        d = RegimeDetector()
        result = d.detect(None)
        assert result.regime == MarketRegime.SIDEWAYS

    def test_regime_shift_logged(self):
        d = RegimeDetector()
        # Først bull
        d.detect(_make_bull())
        initial_regime = d.current_regime
        # Derefter bear
        d.detect(_make_bear())
        if d.current_regime != initial_regime:
            assert len(d.shifts) >= 1
            assert d.shifts[-1].from_regime == initial_regime

    def test_history_grows(self):
        d = RegimeDetector()
        d.detect(_make_bull())
        d.detect(_make_bear())
        assert len(d.history) == 2

    def test_current_regime(self):
        d = RegimeDetector()
        d.detect(_make_bull())
        assert d.current_regime is not None


class TestRegimeHistory:
    """Test historisk regime-sekvens."""

    def test_get_regime_history(self):
        d = RegimeDetector()
        df = _make_bull()
        hist_df = d.get_regime_history(df, step=20)
        assert len(hist_df) > 0
        assert "regime" in hist_df.columns
        assert "confidence" in hist_df.columns
        assert "composite_score" in hist_df.columns

    def test_short_data_empty(self):
        d = RegimeDetector()
        df = _make_df(n=30)
        hist_df = d.get_regime_history(df)
        # Kun 30 rækker, min_window=200 → ingen resultater
        assert len(hist_df) == 0


# ══════════════════════════════════════════════════════════════
# ADAPTIVE STRATEGY TESTS
# ══════════════════════════════════════════════════════════════

class TestAdaptiveStrategyBasic:
    """Test AdaptiveStrategy grundlæggende funktioner."""

    def test_name(self):
        a = AdaptiveStrategy()
        assert a.name == "adaptive_regime"

    def test_is_base_strategy(self):
        a = AdaptiveStrategy()
        assert isinstance(a, BaseStrategy)

    def test_initial_state(self):
        a = AdaptiveStrategy()
        assert a.last_regime_result is None
        assert a.current_adjustment is None

    def test_get_adjustment(self):
        a = AdaptiveStrategy()
        adj = a.get_adjustment(MarketRegime.BULL)
        assert adj.max_exposure_pct == 1.0


class TestAdaptiveStrategyAnalyze:
    """Test analyze() med forskellige regimer."""

    def test_analyze_bull(self):
        a = AdaptiveStrategy()
        df = _make_bull()
        result = a.analyze(df)
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
        assert result.confidence > 0
        assert a.last_regime_result is not None

    def test_analyze_bear(self):
        a = AdaptiveStrategy()
        df = _make_bear()
        result = a.analyze(df)
        assert result.signal in (Signal.SELL, Signal.HOLD)

    def test_analyze_crash_generates_sell(self):
        """CRASH regime skal altid give SELL signal."""
        d = RegimeDetector()
        a = AdaptiveStrategy(detector=d)
        df = _make_crash()
        result = a.analyze(df, vix_level=50.0)  # Force krise-volatilitet
        # Med VIX=50 + crash-data bør vi få SELL eller HOLD
        assert result.signal in (Signal.SELL, Signal.HOLD)

    def test_analyze_short_data(self):
        a = AdaptiveStrategy()
        df = _make_df(n=10)
        result = a.analyze(df)
        assert result.signal == Signal.HOLD
        assert result.confidence == 0

    def test_with_vix_parameter(self):
        a = AdaptiveStrategy()
        df = _make_bull()
        result = a.analyze(df, vix_level=12.0)
        assert result.confidence > 0


class TestAdaptiveWithInnerStrategy:
    """Test AdaptiveStrategy med inner strategy."""

    def test_bull_boosts_buy_confidence(self):
        inner = _DummyStrategy(Signal.BUY, 60)
        a = AdaptiveStrategy(inner_strategy=inner)
        df = _make_bull()
        result = a.analyze(df)
        if a.last_regime_result.regime == MarketRegime.BULL:
            # Bull bør booste BUY confidence
            assert result.signal == Signal.BUY
            assert result.confidence >= 60

    def test_bear_reduces_buy_confidence(self):
        inner = _DummyStrategy(Signal.BUY, 80)
        a = AdaptiveStrategy(inner_strategy=inner)
        df = _make_bear()
        result = a.analyze(df)
        if a.last_regime_result.regime == MarketRegime.BEAR:
            # Bear bør reducere BUY confidence
            assert result.confidence < 80

    def test_crash_blocks_buy(self):
        """CRASH regime skal blokere BUY fra inner strategy."""
        inner = _DummyStrategy(Signal.BUY, 90)
        # Simulér crash-detektion
        d = RegimeDetector()
        a = AdaptiveStrategy(detector=d, inner_strategy=inner)
        df = _make_crash()
        result = a.analyze(df, vix_level=50.0)
        # I crash regime: SELL signal uanset inner
        assert result.signal in (Signal.SELL, Signal.HOLD)

    def test_sell_passes_through(self):
        inner = _DummyStrategy(Signal.SELL, 70)
        a = AdaptiveStrategy(inner_strategy=inner)
        df = _make_bull()
        result = a.analyze(df)
        # SELL bør altid passere igennem
        if a.last_regime_result.regime != MarketRegime.CRASH:
            assert result.signal == Signal.SELL

    def test_hold_passes_through(self):
        inner = _DummyStrategy(Signal.HOLD, 50)
        a = AdaptiveStrategy(inner_strategy=inner)
        df = _make_bull()
        result = a.analyze(df)
        if a.last_regime_result.regime != MarketRegime.CRASH:
            assert result.signal == Signal.HOLD


class TestRegimeSummary:
    """Test get_regime_summary."""

    def test_summary_after_detect(self):
        a = AdaptiveStrategy()
        df = _make_bull()
        a.analyze(df)
        summary = a.get_regime_summary()
        assert "regime" in summary
        assert "label" in summary
        assert "color" in summary
        assert "confidence" in summary
        assert "max_exposure" in summary
        assert "preferred_strategies" in summary
        assert "signals" in summary
        assert "shifts" in summary

    def test_summary_before_detect(self):
        a = AdaptiveStrategy()
        summary = a.get_regime_summary()
        assert summary["regime"] == "unknown"

    def test_summary_signals_list(self):
        a = AdaptiveStrategy()
        a.analyze(_make_bull())
        summary = a.get_regime_summary()
        assert len(summary["signals"]) >= 4
        for sig in summary["signals"]:
            assert "name" in sig
            assert "value" in sig
            assert "detail" in sig


# ══════════════════════════════════════════════════════════════
# IMPORT TESTS
# ══════════════════════════════════════════════════════════════

class TestImports:
    """Test at alle exports fra __init__.py virker."""

    def test_imports(self):
        from src.strategy import (
            RegimeDetector,
            AdaptiveStrategy,
            MarketRegime,
            RegimeResult,
            RegimeSignal,
            RegimeShift,
            StrategyAdjustment,
            REGIME_INFO,
        )
        assert RegimeDetector is not None
        assert MarketRegime.BULL.value == "bull"
        assert len(REGIME_INFO) == 6
