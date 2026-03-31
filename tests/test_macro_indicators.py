"""
Tests for MacroIndicatorTracker – FRED API, recession-sandsynlighed, surprise index.

Alle API-kald mockes – ingen netværksforbindelse kræves.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.macro_indicators import (
    MacroIndicatorTracker,
    MacroIndicator,
    CategorySummary,
    RecessionProbability,
    EconomicSurpriseIndex,
    MacroReport,
    IndicatorTrend,
    EconomicSignal,
    FRED_SERIES,
    CATEGORIES,
)


# ── Helpers ──────────────────────────────────────────────────

def _tmp_cache_dir() -> str:
    return tempfile.mkdtemp()


def _make_series(
    n: int = 30,
    start_val: float = 100.0,
    trend: float = 0.01,
    noise: float = 2.0,
    seed: int = 42,
) -> pd.Series:
    """Opret syntetisk tidsserie."""
    rng = np.random.RandomState(seed)
    # Brug fast slutdato for at undgå off-by-one med ME freq
    end_date = pd.Timestamp("2026-02-28")
    dates = pd.date_range(end=end_date, periods=n, freq="ME")
    values = start_val + np.arange(n) * trend * start_val + rng.randn(n) * noise
    return pd.Series(values, index=dates, name="test")


def _make_indicator(
    key: str = "housing_starts",
    current: float = 1500.0,
    previous: float = 1450.0,
    trend: IndicatorTrend = IndicatorTrend.IMPROVING,
    higher_is: str = "bullish",
    category: str = "housing",
) -> MacroIndicator:
    return MacroIndicator(
        key=key,
        name=FRED_SERIES.get(key, {}).get("name", key),
        category=category,
        current_value=current,
        previous_value=previous,
        change_pct=((current - previous) / abs(previous) * 100) if previous else 0,
        trend=trend,
        higher_is=higher_is,
    )


def _make_tracker() -> MacroIndicatorTracker:
    return MacroIndicatorTracker(cache_dir=_tmp_cache_dir())


# ── Test IndicatorTrend ──────────────────────────────────────

class TestIndicatorTrend:
    def test_all_values(self):
        assert len(IndicatorTrend) == 3

    def test_values(self):
        assert IndicatorTrend.IMPROVING.value == "improving"
        assert IndicatorTrend.DETERIORATING.value == "deteriorating"
        assert IndicatorTrend.STABLE.value == "stable"


# ── Test EconomicSignal ──────────────────────────────────────

class TestEconomicSignal:
    def test_all_values(self):
        assert len(EconomicSignal) == 4

    def test_values(self):
        assert EconomicSignal.EXPANSION.value == "expansion"
        assert EconomicSignal.RECESSION_WARNING.value == "recession_warning"


# ── Test MacroIndicator ──────────────────────────────────────

class TestMacroIndicator:
    def test_bullish_signal(self):
        ind = _make_indicator(higher_is="bullish", trend=IndicatorTrend.IMPROVING)
        assert ind.signal == "bullish"

    def test_bearish_signal(self):
        ind = _make_indicator(higher_is="bullish", trend=IndicatorTrend.DETERIORATING)
        assert ind.signal == "bearish"

    def test_inverted_bearish(self):
        """For bearish indicators: improving (stigning) = bearish."""
        ind = _make_indicator(
            key="initial_claims", higher_is="bearish",
            trend=IndicatorTrend.IMPROVING, category="labor",
        )
        assert ind.signal == "bearish"

    def test_inverted_bullish(self):
        """For bearish indicators: deteriorating (fald) = bullish."""
        ind = _make_indicator(
            key="initial_claims", higher_is="bearish",
            trend=IndicatorTrend.DETERIORATING, category="labor",
        )
        assert ind.signal == "bullish"

    def test_neutral_signal(self):
        ind = _make_indicator(trend=IndicatorTrend.STABLE)
        assert ind.signal == "neutral"

    def test_trend_arrow_bullish_improving(self):
        ind = _make_indicator(higher_is="bullish", trend=IndicatorTrend.IMPROVING)
        assert ind.trend_arrow == "↑"

    def test_trend_arrow_bearish_improving(self):
        ind = _make_indicator(higher_is="bearish", trend=IndicatorTrend.IMPROVING)
        assert ind.trend_arrow == "↓"

    def test_trend_arrow_stable(self):
        ind = _make_indicator(trend=IndicatorTrend.STABLE)
        assert ind.trend_arrow == "→"

    def test_color_green(self):
        ind = _make_indicator(higher_is="bullish", trend=IndicatorTrend.IMPROVING)
        assert ind.color == "green"

    def test_color_red(self):
        ind = _make_indicator(higher_is="bullish", trend=IndicatorTrend.DETERIORATING)
        assert ind.color == "red"

    def test_color_gray(self):
        ind = _make_indicator(trend=IndicatorTrend.STABLE)
        assert ind.color == "gray"


# ── Test CategorySummary ─────────────────────────────────────

class TestCategorySummary:
    def test_bullish_count(self):
        inds = [
            _make_indicator(key="a", trend=IndicatorTrend.IMPROVING),
            _make_indicator(key="b", trend=IndicatorTrend.IMPROVING),
            _make_indicator(key="c", trend=IndicatorTrend.DETERIORATING),
        ]
        cat = CategorySummary(
            category="test", name="Test", indicators=inds,
        )
        assert cat.bullish_count == 2
        assert cat.bearish_count == 1

    def test_empty(self):
        cat = CategorySummary(category="test", name="Test", indicators=[])
        assert cat.bullish_count == 0
        assert cat.bearish_count == 0


# ── Test RecessionProbability ────────────────────────────────

class TestRecessionProbability:
    def test_high_probability(self):
        rp = RecessionProbability(
            probability=70, level="high",
            key_warnings=["Yield curve inverteret"],
            key_positives=[],
            contributing_factors={"yield_curve": 30},
        )
        assert rp.color == "red"

    def test_moderate_probability(self):
        rp = RecessionProbability(
            probability=40, level="elevated",
            key_warnings=[], key_positives=[],
            contributing_factors={},
        )
        assert rp.color == "orange"

    def test_low_probability(self):
        rp = RecessionProbability(
            probability=15, level="low",
            key_warnings=[], key_positives=["Alt er godt"],
            contributing_factors={},
        )
        assert rp.color == "green"


# ── Test EconomicSurpriseIndex ───────────────────────────────

class TestEconomicSurpriseIndex:
    def test_positive_surprise(self):
        esi = EconomicSurpriseIndex(
            value=40, interpretation="Data overgår",
            beats=8, misses=2, total=10,
        )
        assert esi.value > 0
        assert esi.beats > esi.misses

    def test_negative_surprise(self):
        esi = EconomicSurpriseIndex(
            value=-30, interpretation="Data skuffer",
            beats=2, misses=8, total=10,
        )
        assert esi.value < 0


# ── Test MacroIndicatorTracker Init ──────────────────────────

class TestMacroTrackerInit:
    def test_creates_db(self):
        cache_dir = _tmp_cache_dir()
        tracker = MacroIndicatorTracker(cache_dir=cache_dir)
        db_path = Path(cache_dir) / "macro_indicators.db"
        assert db_path.exists()

    def test_db_tables(self):
        cache_dir = _tmp_cache_dir()
        tracker = MacroIndicatorTracker(cache_dir=cache_dir)
        db_path = Path(cache_dir) / "macro_indicators.db"
        conn = sqlite3.connect(db_path)
        tables = {t[0] for t in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "macro_data" in tables
        assert "macro_reports" in tables
        conn.close()

    def test_no_api_key(self):
        tracker = MacroIndicatorTracker(cache_dir=_tmp_cache_dir())
        assert tracker._fred is None

    def test_with_api_key(self):
        with patch("src.data.macro_indicators._HAS_FRED", True), \
             patch("src.data.macro_indicators.Fred", create=True) as mock_fred:
            tracker = MacroIndicatorTracker(
                fred_api_key="test_key_123",
                cache_dir=_tmp_cache_dir(),
            )
        mock_fred.assert_called_once_with(api_key="test_key_123")


# ── Test Determine Trend ─────────────────────────────────────

class TestDetermineTrend:
    def test_improving_bullish(self):
        series = pd.Series([100, 101, 102, 105, 108, 112])
        trend = MacroIndicatorTracker._determine_trend(series, "bullish")
        assert trend == IndicatorTrend.IMPROVING

    def test_deteriorating_bullish(self):
        series = pd.Series([112, 108, 105, 102, 101, 100])
        trend = MacroIndicatorTracker._determine_trend(series, "bullish")
        assert trend == IndicatorTrend.DETERIORATING

    def test_stable(self):
        series = pd.Series([100, 100.5, 99.5, 100.2, 99.8, 100.1])
        trend = MacroIndicatorTracker._determine_trend(series, "bullish")
        assert trend == IndicatorTrend.STABLE

    def test_bearish_indicator_rising(self):
        """For bearish indicators, rising = deteriorating."""
        series = pd.Series([100, 105, 110, 115, 120, 125])
        trend = MacroIndicatorTracker._determine_trend(series, "bearish")
        assert trend == IndicatorTrend.DETERIORATING

    def test_bearish_indicator_falling(self):
        """For bearish indicators, falling = improving."""
        series = pd.Series([125, 120, 115, 110, 105, 100])
        trend = MacroIndicatorTracker._determine_trend(series, "bearish")
        assert trend == IndicatorTrend.IMPROVING

    def test_too_short_series(self):
        series = pd.Series([100, 101])
        trend = MacroIndicatorTracker._determine_trend(series, "bullish")
        assert trend == IndicatorTrend.STABLE


# ── Test Build Indicator ─────────────────────────────────────

class TestBuildIndicator:
    def test_build_from_series(self):
        tracker = _make_tracker()
        series = _make_series(n=30, start_val=1500, trend=0.01)
        meta = FRED_SERIES["housing_starts"]
        ind = tracker._build_indicator("housing_starts", series, meta)

        assert ind.key == "housing_starts"
        assert ind.name == "Housing Starts"
        assert ind.category == "housing"
        assert ind.current_value > 0
        assert ind.previous_value > 0

    def test_single_value_series(self):
        tracker = _make_tracker()
        series = pd.Series([100.0], index=[pd.Timestamp("2026-01-01")])
        meta = FRED_SERIES["housing_starts"]
        ind = tracker._build_indicator("housing_starts", series, meta)
        assert ind.current_value == 100.0
        assert ind.previous_value == 100.0


# ── Test Cache ───────────────────────────────────────────────

class TestCache:
    def test_write_and_read(self):
        tracker = _make_tracker()
        series = _make_series(n=10)
        tracker._write_indicator_cache("housing_starts", "HOUST", series)
        cached = tracker._read_indicator_cache("housing_starts")
        assert cached is not None
        assert len(cached) == 10

    def test_cache_expired(self):
        tracker = _make_tracker()
        # Insert med gammel timestamp (monthly → 72h TTL)
        old_time = (datetime.now() - timedelta(hours=73)).isoformat()
        with tracker._get_conn() as conn:
            conn.execute(
                """INSERT INTO macro_data
                   (series_key, series_id, value, date, fetched_at)
                   VALUES (?, ?, ?, ?, ?)""",
                ("housing_starts", "HOUST", 1500.0,
                 datetime.now().strftime("%Y-%m-%d"), old_time),
            )
        cached = tracker._read_indicator_cache("housing_starts")
        assert cached is None

    def test_daily_cache_ttl(self):
        """Daily indikatorer har 6 timers TTL."""
        tracker = _make_tracker()
        # Recent cache (5 timer)
        recent_time = (datetime.now() - timedelta(hours=5)).isoformat()
        with tracker._get_conn() as conn:
            conn.execute(
                """INSERT INTO macro_data
                   (series_key, series_id, value, date, fetched_at)
                   VALUES (?, ?, ?, ?, ?)""",
                ("yield_spread_10y2y", "T10Y2Y", 1.5,
                 datetime.now().strftime("%Y-%m-%d"), recent_time),
            )
        cached = tracker._read_indicator_cache("yield_spread_10y2y")
        assert cached is not None


# ── Test Get Indicator ───────────────────────────────────────

class TestGetIndicator:
    def test_unknown_key(self):
        tracker = _make_tracker()
        result = tracker.get_indicator("nonexistent_key")
        assert result is None

    def test_with_cached_data(self):
        tracker = _make_tracker()
        series = _make_series(n=15, start_val=1500)
        tracker._write_indicator_cache("housing_starts", "HOUST", series)
        result = tracker.get_indicator("housing_starts")
        assert result is not None
        assert result.key == "housing_starts"
        assert result.current_value > 0

    def test_with_fred_mock(self):
        tracker = _make_tracker()
        mock_series = _make_series(n=20, start_val=1500)

        with patch.object(tracker, "_fetch_fred_series", return_value=mock_series):
            result = tracker.get_indicator("housing_starts", use_cache=False)

        assert result is not None
        assert result.name == "Housing Starts"

    def test_no_data_returns_none(self):
        tracker = _make_tracker()
        with patch.object(tracker, "_fetch_fred_series", return_value=None), \
             patch.object(tracker, "_fetch_yfinance_series", return_value=None):
            result = tracker.get_indicator("housing_starts", use_cache=False)
        assert result is None


# ── Test Get Category ────────────────────────────────────────

class TestGetCategory:
    def test_housing_category(self):
        tracker = _make_tracker()

        def mock_indicator(key, use_cache=True):
            if key in ("housing_starts", "building_permits"):
                return _make_indicator(
                    key=key, trend=IndicatorTrend.IMPROVING,
                    category="housing",
                )
            elif key == "mortgage_rate_30y":
                return _make_indicator(
                    key=key, higher_is="bearish",
                    trend=IndicatorTrend.DETERIORATING,
                    category="housing",
                )
            elif key == "case_shiller":
                return _make_indicator(
                    key=key, higher_is="neutral",
                    trend=IndicatorTrend.STABLE,
                    category="housing",
                )
            return None

        with patch.object(tracker, "get_indicator", side_effect=mock_indicator):
            cat = tracker.get_category("housing")

        assert cat.category == "housing"
        assert len(cat.indicators) > 0
        assert cat.name == "Ejendomsmarked"

    def test_all_categories_exist(self):
        for cat in CATEGORIES:
            assert cat in ["shipping", "housing", "energy", "consumer", "labor", "recession"]


# ── Test Recession Probability ───────────────────────────────

class TestRecessionCalc:
    def test_low_probability(self):
        tracker = _make_tracker()

        def mock_indicator(key, use_cache=True):
            if key == "yield_spread_10y2y":
                return _make_indicator(key=key, current=1.5, trend=IndicatorTrend.STABLE, category="recession")
            if key == "sahm_rule":
                return _make_indicator(key=key, current=0.1, higher_is="bearish", trend=IndicatorTrend.STABLE, category="recession")
            if key == "initial_claims":
                return _make_indicator(key=key, higher_is="bearish", trend=IndicatorTrend.IMPROVING, category="labor")
            if key == "housing_starts":
                return _make_indicator(key=key, trend=IndicatorTrend.IMPROVING, category="housing")
            if key == "michigan_sentiment":
                return _make_indicator(key=key, trend=IndicatorTrend.IMPROVING, category="consumer")
            if key == "leading_index":
                return _make_indicator(key=key, trend=IndicatorTrend.IMPROVING, category="recession")
            return None

        with patch.object(tracker, "get_indicator", side_effect=mock_indicator):
            result = tracker.calculate_recession_probability()

        assert result.probability < 20
        assert result.level in ("low", "moderate")
        assert len(result.key_positives) > 0

    def test_high_probability(self):
        tracker = _make_tracker()

        def mock_indicator(key, use_cache=True):
            if key == "yield_spread_10y2y":
                return _make_indicator(key=key, current=-0.5, trend=IndicatorTrend.DETERIORATING, category="recession")
            if key == "sahm_rule":
                return _make_indicator(key=key, current=0.8, higher_is="bearish", trend=IndicatorTrend.DETERIORATING, category="recession")
            if key == "initial_claims":
                return _make_indicator(key=key, higher_is="bearish", trend=IndicatorTrend.DETERIORATING, category="labor")
            if key == "housing_starts":
                return _make_indicator(key=key, trend=IndicatorTrend.DETERIORATING, category="housing")
            if key == "michigan_sentiment":
                return _make_indicator(key=key, trend=IndicatorTrend.DETERIORATING, category="consumer")
            if key == "leading_index":
                return _make_indicator(key=key, trend=IndicatorTrend.DETERIORATING, category="recession")
            return None

        with patch.object(tracker, "get_indicator", side_effect=mock_indicator):
            result = tracker.calculate_recession_probability()

        assert result.probability >= 50
        assert result.level in ("elevated", "high")
        assert len(result.key_warnings) > 0

    def test_no_data_low_probability(self):
        tracker = _make_tracker()
        with patch.object(tracker, "get_indicator", return_value=None):
            result = tracker.calculate_recession_probability()
        assert result.probability == 10  # Base only


# ── Test Surprise Index ──────────────────────────────────────

class TestSurpriseIndex:
    def test_positive_surprise(self):
        tracker = _make_tracker()
        count = 0

        def mock_indicator(key, use_cache=True):
            nonlocal count
            count += 1
            # Mest bullish
            return _make_indicator(
                key=key, trend=IndicatorTrend.IMPROVING,
                category=FRED_SERIES.get(key, {}).get("category", "test"),
                higher_is=FRED_SERIES.get(key, {}).get("higher_is", "bullish"),
            )

        with patch.object(tracker, "get_indicator", side_effect=mock_indicator):
            result = tracker.calculate_surprise_index()

        assert result.beats > result.misses
        assert result.value > 0

    def test_no_data(self):
        tracker = _make_tracker()
        with patch.object(tracker, "get_indicator", return_value=None):
            result = tracker.calculate_surprise_index()
        assert result.total == 0
        assert result.value == 0


# ── Test Full Report ─────────────────────────────────────────

class TestMacroReport:
    def test_report_structure(self):
        tracker = _make_tracker()

        def mock_indicator(key, use_cache=True):
            meta = FRED_SERIES.get(key, {})
            return _make_indicator(
                key=key,
                category=meta.get("category", "test"),
                higher_is=meta.get("higher_is", "bullish"),
                trend=IndicatorTrend.STABLE,
            )

        with patch.object(tracker, "get_indicator", side_effect=mock_indicator):
            report = tracker.get_macro_report()

        assert isinstance(report, MacroReport)
        assert len(report.indicators) > 0
        assert len(report.categories) > 0
        assert isinstance(report.recession_probability, RecessionProbability)
        assert isinstance(report.surprise_index, EconomicSurpriseIndex)
        assert report.overall_signal in EconomicSignal
        assert -15 <= report.confidence_adjustment <= 10

    def test_report_handles_errors(self):
        tracker = _make_tracker()
        with patch.object(tracker, "get_indicator", side_effect=Exception("fail")):
            report = tracker.get_macro_report()
        assert isinstance(report, MacroReport)
        assert len(report.indicators) == 0

    def test_expansion_signal(self):
        tracker = _make_tracker()

        def mock_indicator(key, use_cache=True):
            meta = FRED_SERIES.get(key, {})
            return _make_indicator(
                key=key,
                category=meta.get("category", "test"),
                higher_is=meta.get("higher_is", "bullish"),
                trend=IndicatorTrend.IMPROVING,
            )

        with patch.object(tracker, "get_indicator", side_effect=mock_indicator):
            report = tracker.get_macro_report()

        assert report.overall_signal in (EconomicSignal.EXPANSION, EconomicSignal.STABLE)
        assert report.confidence_adjustment >= 0


# ── Test Strategy Integration ────────────────────────────────

class TestStrategyIntegration:
    def test_confidence_adjustment(self):
        tracker = _make_tracker()
        mock_report = MacroReport(
            indicators={},
            categories={},
            recession_probability=RecessionProbability(
                probability=10, level="low",
                key_warnings=[], key_positives=[],
                contributing_factors={},
            ),
            surprise_index=EconomicSurpriseIndex(
                value=0, interpretation="Neutral",
                beats=0, misses=0, total=0,
            ),
            overall_signal=EconomicSignal.EXPANSION,
            confidence_adjustment=10,
        )
        with patch.object(tracker, "get_macro_report", return_value=mock_report):
            adj = tracker.get_confidence_adjustment()
        assert adj == 10


# ── Test Explain ─────────────────────────────────────────────

class TestExplain:
    def test_explain_contains_sections(self):
        tracker = _make_tracker()

        def mock_indicator(key, use_cache=True):
            meta = FRED_SERIES.get(key, {})
            return _make_indicator(
                key=key,
                category=meta.get("category", "test"),
                higher_is=meta.get("higher_is", "bullish"),
                trend=IndicatorTrend.STABLE,
            )

        with patch.object(tracker, "get_indicator", side_effect=mock_indicator):
            text = tracker.explain()

        assert "MAKROØKONOMISK RAPPORT" in text
        assert "RECESSION-SANDSYNLIGHED" in text
        assert "ECONOMIC SURPRISE INDEX" in text
        assert "SAMLET VURDERING" in text

    def test_print_report(self, capsys):
        tracker = _make_tracker()

        with patch.object(tracker, "get_indicator", return_value=None):
            tracker.print_report()

        captured = capsys.readouterr()
        assert "MAKROØKONOMISK RAPPORT" in captured.out


# ── Test FRED Series ─────────────────────────────────────────

class TestFREDSeries:
    def test_all_series_have_required_fields(self):
        for key, meta in FRED_SERIES.items():
            assert "id" in meta, f"{key} mangler 'id'"
            assert "name" in meta, f"{key} mangler 'name'"
            assert "category" in meta, f"{key} mangler 'category'"
            assert "higher_is" in meta, f"{key} mangler 'higher_is'"

    def test_all_categories_covered(self):
        found_cats = set(m["category"] for m in FRED_SERIES.values())
        for cat in CATEGORIES:
            assert cat in found_cats, f"Kategori '{cat}' har ingen indikatorer"

    def test_series_count(self):
        assert len(FRED_SERIES) >= 15

    def test_key_indicators_present(self):
        assert "housing_starts" in FRED_SERIES
        assert "initial_claims" in FRED_SERIES
        assert "yield_spread_10y2y" in FRED_SERIES
        assert "michigan_sentiment" in FRED_SERIES
        assert "sahm_rule" in FRED_SERIES
        assert "mortgage_rate_30y" in FRED_SERIES
