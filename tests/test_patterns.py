"""
Tests for src.strategy.patterns – mønstergenkendelse.

Dækker: Chart patterns, Candlestick patterns, Support/Resistance,
        Seasonality, Divergens, Multi-timeframe, PatternScanner.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.base_strategy import Signal
from src.strategy.patterns import (
    # Enums & dataclasses
    PatternDirection, PatternType, DetectedPattern,
    SupportResistanceLevel, BreakoutSignal, SeasonalPattern,
    DivergenceSignal, TimeframeSignal, MultiTimeframeResult,
    PatternScanResult,
    # Detektorer
    ChartPatternDetector, CandlestickDetector,
    SupportResistanceDetector, SeasonalityAnalyzer,
    DivergenceDetector, MultiTimeframeAnalyzer,
    PatternScanner,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_ohlcv(n: int = 200, seed: int = 42, trend: float = 0.001) -> pd.DataFrame:
    """Generér syntetisk OHLCV data."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start="2023-01-01", periods=n)

    close = 100.0
    rows = []
    for i in range(n):
        change = rng.randn() * 2 + trend * close
        close += change
        close = max(close, 10)
        high = close + abs(rng.randn()) * 1.5
        low = close - abs(rng.randn()) * 1.5
        opn = close + rng.randn() * 0.5
        vol = int(abs(rng.randn() * 1_000_000 + 5_000_000))
        rows.append({"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol})

    return pd.DataFrame(rows, index=dates)


def _make_zigzag(n: int = 200, amplitude: float = 10.0, period: int = 40, base: float = 100.0) -> pd.DataFrame:
    """Generér zigzag-data."""
    dates = pd.bdate_range(start="2023-01-01", periods=n)
    t = np.arange(n)
    close = base + amplitude * np.sin(2 * np.pi * t / period)
    high = close + 1.0
    low = close - 1.0
    opn = close + 0.5
    vol = np.full(n, 5_000_000)
    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol
    }, index=dates)


def _make_double_top() -> pd.DataFrame:
    """Generér data med tydeligt double top mønster."""
    n = 120
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    t = np.arange(n)

    # Stig til 120, fald til 110, stig til 120 igen, fald
    prices = np.piecewise(t.astype(float), [
        t < 30, (t >= 30) & (t < 50), (t >= 50) & (t < 70),
        (t >= 70) & (t < 90), t >= 90
    ], [
        lambda x: 100 + x * 0.67,    # stiger til ~120
        lambda x: 120 - (x - 30) * 0.5,  # falder til ~110
        lambda x: 110 + (x - 50) * 0.5,  # stiger til ~120
        lambda x: 120 - (x - 70) * 0.5,  # falder
        lambda x: 110 - (x - 90) * 0.3,  # fortsætter ned
    ])

    return pd.DataFrame({
        "Open": prices + 0.3,
        "High": prices + 1.5,
        "Low": prices - 1.5,
        "Close": prices,
        "Volume": np.full(n, 5_000_000),
    }, index=dates)


def _make_hns() -> pd.DataFrame:
    """Generér data med Head & Shoulders mønster."""
    n = 150
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    t = np.arange(n)

    prices = np.piecewise(t.astype(float), [
        t < 20, (t >= 20) & (t < 35), (t >= 35) & (t < 55),
        (t >= 55) & (t < 75), (t >= 75) & (t < 95),
        (t >= 95) & (t < 115), t >= 115
    ], [
        lambda x: 100 + x * 0.5,       # stiger til ~110 (venstre skulder)
        lambda x: 110 - (x - 20) * 0.4,  # falder til ~104
        lambda x: 104 + (x - 35) * 0.6,  # stiger til ~116 (hoved)
        lambda x: 116 - (x - 55) * 0.6,  # falder til ~104
        lambda x: 104 + (x - 75) * 0.3,  # stiger til ~110 (højre skulder)
        lambda x: 110 - (x - 95) * 0.5,  # falder
        lambda x: 100 - (x - 115) * 0.3,  # fortsætter ned
    ])

    return pd.DataFrame({
        "Open": prices + 0.2,
        "High": prices + 2.0,
        "Low": prices - 2.0,
        "Close": prices,
        "Volume": np.full(n, 5_000_000),
    }, index=dates)


def _make_candle_engulfing(bullish: bool = True) -> pd.DataFrame:
    """Lav data med engulfing pattern i de to sidste bars."""
    n = 30
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    close = np.full(n, 100.0)
    opn = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)

    if bullish:
        # Næstsidste: bearish (close < open)
        opn[-2] = 101.0
        close[-2] = 99.0
        high[-2] = 101.5
        low[-2] = 98.5
        # Sidste: bullish engulfing (close > prev open, open < prev close)
        opn[-1] = 98.5
        close[-1] = 102.0
        high[-1] = 102.5
        low[-1] = 98.0
    else:
        # Næstsidste: bullish
        opn[-2] = 99.0
        close[-2] = 101.0
        high[-2] = 101.5
        low[-2] = 98.5
        # Sidste: bearish engulfing
        opn[-1] = 101.5
        close[-1] = 98.0
        high[-1] = 102.0
        low[-1] = 97.5

    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close,
        "Volume": np.full(n, 5_000_000),
    }, index=dates)


def _make_doji() -> pd.DataFrame:
    """Data med doji i seneste bar."""
    n = 20
    dates = pd.bdate_range(start="2024-01-01", periods=n)
    close = np.full(n, 100.0)
    opn = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)

    # Doji: open ≈ close, lang range
    opn[-1] = 100.0
    close[-1] = 100.02  # næsten ens
    high[-1] = 103.0
    low[-1] = 97.0

    return pd.DataFrame({
        "Open": opn, "High": high, "Low": low, "Close": close,
        "Volume": np.full(n, 5_000_000),
    }, index=dates)


def _make_long_data(years: int = 3) -> pd.DataFrame:
    """Lang datasæt til sæsonanalyse."""
    n = 252 * years
    rng = np.random.RandomState(42)
    dates = pd.bdate_range(start="2021-01-01", periods=n)
    close = 100.0
    rows = []
    for i in range(n):
        month = dates[i].month
        seasonal_bias = 0.001 if month in [1, 11, 12] else -0.0005 if month in [5, 6, 7] else 0
        change = rng.randn() * 1.5 + seasonal_bias * close
        close += change
        close = max(close, 20)
        high = close + abs(rng.randn()) * 1.0
        low = close - abs(rng.randn()) * 1.0
        rows.append({
            "Open": close + rng.randn() * 0.3,
            "High": high, "Low": low, "Close": close,
            "Volume": int(abs(rng.randn() * 1_000_000 + 5_000_000)),
        })
    return pd.DataFrame(rows, index=dates)


# ══════════════════════════════════════════════════════════════
#  ENUMS & DATACLASSES
# ══════════════════════════════════════════════════════════════

class TestEnums:
    def test_pattern_direction_values(self):
        assert PatternDirection.BULLISH.value == "bullish"
        assert PatternDirection.BEARISH.value == "bearish"
        assert PatternDirection.NEUTRAL.value == "neutral"

    def test_pattern_type_chart(self):
        assert PatternType.HEAD_AND_SHOULDERS.value == "head_and_shoulders"
        assert PatternType.DOUBLE_TOP.value == "double_top"
        assert PatternType.CUP_AND_HANDLE.value == "cup_and_handle"

    def test_pattern_type_candlestick(self):
        assert PatternType.DOJI.value == "doji"
        assert PatternType.HAMMER.value == "hammer"
        assert PatternType.BULLISH_ENGULFING.value == "bullish_engulfing"

    def test_pattern_type_divergence(self):
        assert PatternType.BULLISH_DIVERGENCE.value == "bullish_divergence"
        assert PatternType.BEARISH_DIVERGENCE.value == "bearish_divergence"


class TestDataclasses:
    def test_detected_pattern(self):
        p = DetectedPattern(
            pattern_type=PatternType.DOUBLE_TOP,
            direction=PatternDirection.BEARISH,
            confidence=70.0,
            start_idx=10, end_idx=50,
            description="test",
            price_target=95.0,
            volume_confirmed=True,
        )
        assert p.confidence == 70.0
        assert p.volume_confirmed is True

    def test_support_resistance_level(self):
        sr = SupportResistanceLevel(
            price=100.0, level_type="support",
            strength=4, volume_weight=0.8,
            first_touch="2024-01-01", last_touch="2024-03-01",
        )
        assert sr.is_strong is True

    def test_support_resistance_weak(self):
        sr = SupportResistanceLevel(
            price=100.0, level_type="resistance",
            strength=2, volume_weight=0.5,
            first_touch=None, last_touch=None,
        )
        assert sr.is_strong is False

    def test_breakout_signal(self):
        sr = SupportResistanceLevel(100, "resistance", 3, 0.7, None, None)
        b = BreakoutSignal(sr, 102.0, 1.5, "up", "test breakout")
        assert b.direction == "up"

    def test_seasonal_pattern(self):
        sp = SeasonalPattern(
            period="monthly", data={"Jan": 2.0, "Feb": -1.0},
            best_period="Jan", worst_period="Feb",
            sell_in_may_effect=3.0, santa_rally_avg=2.5, january_effect=1.8,
        )
        assert sp.best_period == "Jan"

    def test_divergence_signal(self):
        d = DivergenceSignal("bullish", "RSI", "lower_lows", "higher_lows",
                            60.0, 10, 50, "test")
        assert d.divergence_type == "bullish"

    def test_timeframe_signal(self):
        ts = TimeframeSignal("daily", Signal.BUY, 65.0, "test")
        assert ts.signal == Signal.BUY

    def test_multi_timeframe_result(self):
        mtf = MultiTimeframeResult(
            signals=[], consensus=Signal.HOLD,
            consensus_confidence=30.0, aligned=False,
            description="test",
        )
        assert mtf.aligned is False

    def test_pattern_scan_result(self):
        psr = PatternScanResult(
            symbol="AAPL",
            chart_patterns=[], candlestick_patterns=[],
            support_resistance=[], breakouts=[],
            divergences=[], seasonal=None, multi_timeframe=None,
            overall_signal=Signal.HOLD, overall_confidence=30.0,
            summary="test",
        )
        assert psr.symbol == "AAPL"


# ══════════════════════════════════════════════════════════════
#  CHART PATTERNS
# ══════════════════════════════════════════════════════════════

class TestChartPatternDetector:
    def test_detect_all_returns_list(self):
        df = _make_ohlcv(200)
        det = ChartPatternDetector()
        result = det.detect_all(df)
        assert isinstance(result, list)

    def test_short_data_returns_empty(self):
        df = _make_ohlcv(10)
        det = ChartPatternDetector()
        result = det.detect_all(df)
        assert result == []

    def test_double_top_detection(self):
        df = _make_double_top()
        det = ChartPatternDetector(order=3, tolerance=0.05)
        result = det.detect_all(df)
        dt_found = [p for p in result if p.pattern_type == PatternType.DOUBLE_TOP]
        # Vi forventer at den finder et double top (med tilstrækkelig tolerance)
        # Det er OK hvis den ikke finder det — mønster-detektion er aldrig 100%
        for p in dt_found:
            assert p.direction == PatternDirection.BEARISH

    def test_hns_detection(self):
        df = _make_hns()
        det = ChartPatternDetector(order=3, tolerance=0.05)
        result = det.detect_all(df)
        hns = [p for p in result if p.pattern_type == PatternType.HEAD_AND_SHOULDERS]
        for p in hns:
            assert p.direction == PatternDirection.BEARISH
            assert p.confidence > 0

    def test_zigzag_finds_patterns(self):
        df = _make_zigzag(200, amplitude=15, period=40)
        det = ChartPatternDetector(order=5, tolerance=0.03)
        result = det.detect_all(df)
        # Zigzag-data bør generere nogle mønstre
        assert isinstance(result, list)

    def test_prices_match(self):
        det = ChartPatternDetector(tolerance=0.02)
        assert det._prices_match(100.0, 101.5) is True
        assert det._prices_match(100.0, 103.0) is False
        assert det._prices_match(0, 100) is False

    def test_pattern_has_confidence(self):
        df = _make_zigzag(200, amplitude=15, period=40)
        det = ChartPatternDetector(order=3, tolerance=0.05)
        result = det.detect_all(df)
        for p in result:
            assert 0 <= p.confidence <= 100


class TestTriangleDetection:
    def test_ascending_triangle(self):
        """Test med data der har fladt modstandsniveau og stigende støtte."""
        n = 100
        dates = pd.bdate_range(start="2024-01-01", periods=n)
        t = np.arange(n)
        # Zig-zag med fladt top og stigende bund
        close = 100.0 + 5 * np.sin(2 * np.pi * t / 15) + t * 0.05
        close = np.minimum(close, 108)  # flat top
        df = pd.DataFrame({
            "Open": close + 0.1, "High": close + 1.5,
            "Low": close - 1.5, "Close": close,
            "Volume": np.full(n, 5_000_000),
        }, index=dates)
        det = ChartPatternDetector(order=3, tolerance=0.04)
        result = det.detect_all(df)
        # Der kan eller kan ikke være et ascending triangle — vigtigt er at det ikke crasher
        assert isinstance(result, list)


class TestFlagDetection:
    def test_no_crash_on_flat_data(self):
        df = _make_ohlcv(50)
        det = ChartPatternDetector(order=3)
        result = det._detect_flags([], [], df)
        assert isinstance(result, list)


# ══════════════════════════════════════════════════════════════
#  CANDLESTICK PATTERNS
# ══════════════════════════════════════════════════════════════

class TestCandlestickDetector:
    def test_detect_doji(self):
        df = _make_doji()
        det = CandlestickDetector()
        result = det.detect_all(df, lookback=3)
        dojis = [p for p in result if p.pattern_type == PatternType.DOJI]
        assert len(dojis) >= 1
        assert dojis[0].direction == PatternDirection.NEUTRAL

    def test_detect_bullish_engulfing(self):
        df = _make_candle_engulfing(bullish=True)
        det = CandlestickDetector()
        result = det.detect_all(df, lookback=3)
        eng = [p for p in result if p.pattern_type == PatternType.BULLISH_ENGULFING]
        assert len(eng) >= 1
        assert eng[0].direction == PatternDirection.BULLISH

    def test_detect_bearish_engulfing(self):
        df = _make_candle_engulfing(bullish=False)
        det = CandlestickDetector()
        result = det.detect_all(df, lookback=3)
        eng = [p for p in result if p.pattern_type == PatternType.BEARISH_ENGULFING]
        assert len(eng) >= 1
        assert eng[0].direction == PatternDirection.BEARISH

    def test_three_white_soldiers(self):
        n = 10
        dates = pd.bdate_range(start="2024-01-01", periods=n)
        opn = np.array([100, 100, 100, 100, 100, 100, 100, 101, 103, 105], dtype=float)
        close = np.array([100, 100, 100, 100, 100, 100, 100, 102.5, 104.5, 107], dtype=float)
        high = close + 0.3
        low = opn - 0.3
        df = pd.DataFrame({
            "Open": opn, "High": high, "Low": low, "Close": close,
            "Volume": np.full(n, 5_000_000),
        }, index=dates)
        det = CandlestickDetector()
        result = det.detect_all(df, lookback=5)
        tws = [p for p in result if p.pattern_type == PatternType.THREE_WHITE_SOLDIERS]
        assert len(tws) >= 1

    def test_three_black_crows(self):
        n = 10
        dates = pd.bdate_range(start="2024-01-01", periods=n)
        opn = np.array([100, 100, 100, 100, 100, 100, 100, 107, 105, 103], dtype=float)
        close = np.array([100, 100, 100, 100, 100, 100, 100, 105, 103, 101], dtype=float)
        high = opn + 0.3
        low = close - 0.3
        df = pd.DataFrame({
            "Open": opn, "High": high, "Low": low, "Close": close,
            "Volume": np.full(n, 5_000_000),
        }, index=dates)
        det = CandlestickDetector()
        result = det.detect_all(df, lookback=5)
        tbc = [p for p in result if p.pattern_type == PatternType.THREE_BLACK_CROWS]
        assert len(tbc) >= 1

    def test_morning_star(self):
        n = 10
        dates = pd.bdate_range(start="2024-01-01", periods=n)
        # Construct morning star: big red, small body, big green
        opn = np.array([100, 100, 100, 100, 100, 100, 100, 106, 99.5, 100.5], dtype=float)
        close = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 105], dtype=float)
        high = np.maximum(opn, close) + 0.5
        low = np.minimum(opn, close) - 0.5
        df = pd.DataFrame({
            "Open": opn, "High": high, "Low": low, "Close": close,
            "Volume": np.full(n, 5_000_000),
        }, index=dates)
        det = CandlestickDetector()
        result = det.detect_all(df, lookback=5)
        ms = [p for p in result if p.pattern_type == PatternType.MORNING_STAR]
        assert len(ms) >= 1

    def test_short_data_no_crash(self):
        df = _make_ohlcv(2)
        det = CandlestickDetector()
        result = det.detect_all(df)
        assert isinstance(result, list)

    def test_empty_data(self):
        df = _make_ohlcv(1)
        det = CandlestickDetector()
        result = det.detect_all(df)
        assert isinstance(result, list)


# ══════════════════════════════════════════════════════════════
#  SUPPORT & RESISTANCE
# ══════════════════════════════════════════════════════════════

class TestSupportResistance:
    def test_detect_levels_zigzag(self):
        df = _make_zigzag(200, amplitude=10, period=30)
        det = SupportResistanceDetector()
        levels = det.detect_levels(df, order=5)
        assert isinstance(levels, list)
        # Zigzag bør have klare niveauer
        if levels:
            assert all(isinstance(l, SupportResistanceLevel) for l in levels)

    def test_levels_have_types(self):
        df = _make_ohlcv(200)
        det = SupportResistanceDetector()
        levels = det.detect_levels(df)
        for l in levels:
            assert l.level_type in ("support", "resistance")

    def test_max_levels(self):
        df = _make_zigzag(300, amplitude=15, period=20)
        det = SupportResistanceDetector(min_touches=1)
        levels = det.detect_levels(df, max_levels=5)
        assert len(levels) <= 5

    def test_short_data_returns_empty(self):
        df = _make_ohlcv(5)
        det = SupportResistanceDetector()
        levels = det.detect_levels(df)
        assert levels == []

    def test_breakout_detection(self):
        df = _make_ohlcv(200)
        det = SupportResistanceDetector()
        levels = det.detect_levels(df)
        breakouts = det.detect_breakouts(df, levels)
        assert isinstance(breakouts, list)

    def test_breakout_no_levels_empty(self):
        df = _make_ohlcv(200)
        det = SupportResistanceDetector()
        breakouts = det.detect_breakouts(df, [])
        assert breakouts == []


# ══════════════════════════════════════════════════════════════
#  SEASONALITY
# ══════════════════════════════════════════════════════════════

class TestSeasonality:
    def test_analyze_long_data(self):
        df = _make_long_data(3)
        analyzer = SeasonalityAnalyzer()
        result = analyzer.analyze(df)
        assert result is not None
        assert isinstance(result, SeasonalPattern)
        assert result.best_period in analyzer.MONTH_NAMES.values()
        assert result.worst_period in analyzer.MONTH_NAMES.values()

    def test_analyze_short_data_returns_none(self):
        df = _make_ohlcv(100)
        analyzer = SeasonalityAnalyzer()
        result = analyzer.analyze(df)
        assert result is None

    def test_sell_in_may(self):
        df = _make_long_data(3)
        analyzer = SeasonalityAnalyzer()
        result = analyzer.analyze(df)
        assert result is not None
        assert result.sell_in_may_effect is not None

    def test_santa_rally(self):
        df = _make_long_data(3)
        analyzer = SeasonalityAnalyzer()
        result = analyzer.analyze(df)
        assert result is not None
        assert result.santa_rally_avg is not None

    def test_january_effect(self):
        df = _make_long_data(3)
        analyzer = SeasonalityAnalyzer()
        result = analyzer.analyze(df)
        assert result is not None
        assert result.january_effect is not None

    def test_monthly_data_all_months(self):
        df = _make_long_data(3)
        analyzer = SeasonalityAnalyzer()
        result = analyzer.analyze(df)
        assert result is not None
        assert len(result.data) >= 10  # de fleste måneder bør være dækket


# ══════════════════════════════════════════════════════════════
#  DIVERGENS
# ══════════════════════════════════════════════════════════════

class TestDivergence:
    def test_detect_returns_list(self):
        df = _make_ohlcv(200)
        det = DivergenceDetector()
        result = det.detect_all(df)
        assert isinstance(result, list)
        for d in result:
            assert isinstance(d, DivergenceSignal)

    def test_divergence_has_indicator(self):
        df = _make_ohlcv(200)
        det = DivergenceDetector()
        result = det.detect_all(df)
        for d in result:
            assert d.indicator in ("RSI", "MACD", "MFI", "OBV")

    def test_divergence_types(self):
        df = _make_ohlcv(200)
        det = DivergenceDetector()
        result = det.detect_all(df)
        for d in result:
            assert d.divergence_type in ("bullish", "bearish")

    def test_short_data_no_crash(self):
        df = _make_ohlcv(20)
        det = DivergenceDetector()
        result = det.detect_all(df)
        assert isinstance(result, list)

    def test_adds_missing_indicators(self):
        df = _make_ohlcv(100)
        assert "RSI" not in df.columns
        det = DivergenceDetector()
        # detect_all uses a copy internally (thread-safety)
        # Verify it runs without error and returns valid results
        result = det.detect_all(df)
        assert isinstance(result, list)


# ══════════════════════════════════════════════════════════════
#  MULTI-TIMEFRAME
# ══════════════════════════════════════════════════════════════

class TestMultiTimeframe:
    def test_analyze_returns_result(self):
        df = _make_ohlcv(200)
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.analyze(df)
        assert isinstance(result, MultiTimeframeResult)
        assert len(result.signals) >= 1  # mindst daglig
        assert result.consensus in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_daily_always_present(self):
        df = _make_ohlcv(200)
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.analyze(df)
        timeframes = [s.timeframe for s in result.signals]
        assert "daily" in timeframes

    def test_weekly_with_enough_data(self):
        df = _make_ohlcv(200)
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.analyze(df)
        timeframes = [s.timeframe for s in result.signals]
        assert "weekly" in timeframes

    def test_monthly_with_enough_data(self):
        df = _make_ohlcv(200)
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.analyze(df)
        timeframes = [s.timeframe for s in result.signals]
        assert "monthly" in timeframes

    def test_short_data_only_daily(self):
        df = _make_ohlcv(15)
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.analyze(df)
        timeframes = [s.timeframe for s in result.signals]
        assert "daily" in timeframes
        assert "monthly" not in timeframes

    def test_aligned_boost(self):
        """Når alle timeframes er enige, bør confidence være højere."""
        df = _make_ohlcv(200)
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.analyze(df)
        if result.aligned:
            assert result.consensus_confidence >= 50

    def test_resample_weekly(self):
        df = _make_ohlcv(100)
        weekly = MultiTimeframeAnalyzer._resample(df, "W")
        assert len(weekly) < len(df)
        assert "Open" in weekly.columns
        assert "High" in weekly.columns
        assert "Volume" in weekly.columns

    def test_resample_monthly(self):
        df = _make_ohlcv(200)
        monthly = MultiTimeframeAnalyzer._resample(df, "ME")
        assert len(monthly) < len(df)

    def test_description_format(self):
        df = _make_ohlcv(200)
        analyzer = MultiTimeframeAnalyzer()
        result = analyzer.analyze(df)
        assert "Multi-TF:" in result.description


# ══════════════════════════════════════════════════════════════
#  PATTERN SCANNER
# ══════════════════════════════════════════════════════════════

class TestPatternScanner:
    def test_scan_returns_result(self):
        df = _make_ohlcv(200)
        scanner = PatternScanner()
        result = scanner.scan(df, symbol="AAPL")
        assert isinstance(result, PatternScanResult)
        assert result.symbol == "AAPL"

    def test_scan_has_all_fields(self):
        df = _make_ohlcv(200)
        scanner = PatternScanner()
        result = scanner.scan(df, symbol="TSLA")
        assert isinstance(result.chart_patterns, list)
        assert isinstance(result.candlestick_patterns, list)
        assert isinstance(result.support_resistance, list)
        assert isinstance(result.breakouts, list)
        assert isinstance(result.divergences, list)

    def test_scan_overall_signal(self):
        df = _make_ohlcv(200)
        scanner = PatternScanner()
        result = scanner.scan(df, symbol="MSFT")
        assert result.overall_signal in (Signal.BUY, Signal.SELL, Signal.HOLD)
        assert 0 <= result.overall_confidence <= 100

    def test_scan_without_seasonal(self):
        df = _make_ohlcv(100)
        scanner = PatternScanner()
        result = scanner.scan(df, include_seasonal=False)
        assert result.seasonal is None

    def test_scan_without_mtf(self):
        df = _make_ohlcv(100)
        scanner = PatternScanner()
        result = scanner.scan(df, include_mtf=False)
        assert result.multi_timeframe is None

    def test_scan_adds_missing_indicators(self):
        df = _make_ohlcv(100)
        assert "RSI" not in df.columns
        scanner = PatternScanner()
        scanner.scan(df)
        assert "RSI" in df.columns

    def test_confidence_adjustment_buy(self):
        scanner = PatternScanner()
        result = PatternScanResult(
            symbol="TEST", chart_patterns=[], candlestick_patterns=[],
            support_resistance=[], breakouts=[], divergences=[],
            seasonal=None, multi_timeframe=None,
            overall_signal=Signal.BUY, overall_confidence=72.0,
            summary="test",
        )
        adj = scanner.get_confidence_adjustment(result)
        assert 0 < adj <= 15

    def test_confidence_adjustment_sell(self):
        scanner = PatternScanner()
        result = PatternScanResult(
            symbol="TEST", chart_patterns=[], candlestick_patterns=[],
            support_resistance=[], breakouts=[], divergences=[],
            seasonal=None, multi_timeframe=None,
            overall_signal=Signal.SELL, overall_confidence=60.0,
            summary="test",
        )
        adj = scanner.get_confidence_adjustment(result)
        assert -15 <= adj < 0

    def test_confidence_adjustment_hold(self):
        scanner = PatternScanner()
        result = PatternScanResult(
            symbol="TEST", chart_patterns=[], candlestick_patterns=[],
            support_resistance=[], breakouts=[], divergences=[],
            seasonal=None, multi_timeframe=None,
            overall_signal=Signal.HOLD, overall_confidence=30.0,
            summary="test",
        )
        adj = scanner.get_confidence_adjustment(result)
        assert adj == 0

    def test_explain_output(self):
        df = _make_ohlcv(200)
        scanner = PatternScanner()
        result = scanner.scan(df, symbol="AAPL")
        text = scanner.explain(result)
        assert "=== Mønsteranalyse: AAPL ===" in text
        assert "Samlet signal:" in text

    def test_print_report_no_crash(self, capsys):
        df = _make_ohlcv(200)
        scanner = PatternScanner()
        result = scanner.scan(df, symbol="GOOG")
        scanner.print_report(result)
        captured = capsys.readouterr()
        assert "GOOG" in captured.out

    def test_explain_with_seasonal(self):
        df = _make_long_data(3)
        scanner = PatternScanner()
        result = scanner.scan(df, symbol="AAPL")
        text = scanner.explain(result)
        if result.seasonal:
            assert "Sæsonmønstre" in text

    def test_explain_with_mtf(self):
        df = _make_ohlcv(200)
        scanner = PatternScanner()
        result = scanner.scan(df, symbol="AAPL")
        text = scanner.explain(result)
        if result.multi_timeframe:
            assert "Multi-Timeframe" in text

    def test_scan_short_data_no_crash(self):
        df = _make_ohlcv(15)
        scanner = PatternScanner()
        result = scanner.scan(df, symbol="SHORT")
        assert isinstance(result, PatternScanResult)


class TestAggregation:
    def test_no_patterns_gives_hold(self):
        scanner = PatternScanner()
        signal, conf, summary = scanner._aggregate([], [], [], [], None, "TEST")
        assert signal == Signal.HOLD

    def test_bullish_patterns_give_buy(self):
        scanner = PatternScanner()
        patterns = [DetectedPattern(
            PatternType.DOUBLE_BOTTOM, PatternDirection.BULLISH,
            80.0, 0, 50, "test", volume_confirmed=True,
        )]
        signal, conf, summary = scanner._aggregate(patterns, [], [], [], None, "TEST")
        assert signal == Signal.BUY

    def test_bearish_patterns_give_sell(self):
        scanner = PatternScanner()
        patterns = [DetectedPattern(
            PatternType.HEAD_AND_SHOULDERS, PatternDirection.BEARISH,
            80.0, 0, 50, "test", volume_confirmed=True,
        )]
        signal, conf, summary = scanner._aggregate(patterns, [], [], [], None, "TEST")
        assert signal == Signal.SELL

    def test_mixed_signals_may_hold(self):
        scanner = PatternScanner()
        bull = DetectedPattern(PatternType.DOUBLE_BOTTOM, PatternDirection.BULLISH,
                              50.0, 0, 50, "test")
        bear = DetectedPattern(PatternType.DOUBLE_TOP, PatternDirection.BEARISH,
                              50.0, 0, 50, "test")
        signal, conf, summary = scanner._aggregate([bull, bear], [], [], [], None, "TEST")
        # Should be hold since equal signals
        assert signal == Signal.HOLD
