"""
Tests for EarningsTracker og MacroCalendar.

Dækker:
  - SurpriseType klassificering
  - EarningsAnalysis dataklasse
  - EarningsCalendar opbygning
  - PositionAdjustment beregning
  - Historisk earnings-reaktionsanalyse
  - Earnings surprise-analyse
  - MacroEventType klassificering
  - MacroAnalysis dataklasse
  - MacroCalendarView
  - ExposureAdjustment beregning
  - Historisk makro-reaktionsanalyse
  - FOMC-specifik logik
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.sentiment.earnings_tracker import (
    EarningsTracker,
    EarningsAnalysis,
    EarningsCalendar,
    PositionAdjustment,
    SurpriseType,
    classify_surprise,
)
from src.sentiment.macro_calendar import (
    MacroCalendar,
    MacroAnalysis,
    MacroCalendarView,
    MacroEventType,
    MacroImpact,
    ExposureAdjustment,
    classify_macro_event,
    get_event_impact,
)
from src.sentiment.news_fetcher import EarningsEvent, EconomicEvent


# ── Helpers ──────────────────────────────────────────────────

def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _days_from_now(n: int) -> str:
    return (datetime.now() + timedelta(days=n)).strftime("%Y-%m-%d")


def _make_earnings_event(
    symbol: str = "AAPL",
    date: str | None = None,
    hour: str = "amc",
    eps_estimate: float | None = 1.50,
    eps_actual: float | None = None,
    revenue_estimate: float | None = None,
    revenue_actual: float | None = None,
) -> EarningsEvent:
    return EarningsEvent(
        symbol=symbol,
        date=date or _days_from_now(5),
        hour=hour,
        eps_estimate=eps_estimate,
        eps_actual=eps_actual,
        revenue_estimate=revenue_estimate,
        revenue_actual=revenue_actual,
    )


def _make_economic_event(
    name: str = "FOMC Rate Decision",
    date: str | None = None,
    country: str = "US",
    estimate: float | str | None = None,
    actual: float | str | None = None,
    previous: float | str | None = None,
) -> EconomicEvent:
    return EconomicEvent(
        name=name,
        date=date or _days_from_now(5),
        country=country,
        time="",
        impact="",
        estimate=str(estimate) if estimate is not None else "",
        actual=str(actual) if actual is not None else "",
        previous=str(previous) if previous is not None else "",
    )


def _make_price_df(n: int = 100, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Generér syntetisk prisdata."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n + 2)[-n:]
    returns = rng.normal(0.001, 0.02, n)
    prices = start_price * np.cumprod(1 + returns)
    return pd.DataFrame({
        "Open": prices * 0.99,
        "High": prices * 1.01,
        "Low": prices * 0.98,
        "Close": prices,
        "Volume": rng.integers(1_000_000, 10_000_000, n),
    }, index=dates)


# ══════════════════════════════════════════════════════════════
# EARNINGS TRACKER TESTS
# ══════════════════════════════════════════════════════════════

class TestClassifySurprise:
    """Test classify_surprise funktion."""

    def test_big_beat(self):
        assert classify_surprise(0.15) == SurpriseType.BIG_BEAT

    def test_beat(self):
        assert classify_surprise(0.07) == SurpriseType.BEAT

    def test_small_beat(self):
        assert classify_surprise(0.03) == SurpriseType.SMALL_BEAT

    def test_inline(self):
        assert classify_surprise(0.0) == SurpriseType.INLINE
        assert classify_surprise(0.01) == SurpriseType.INLINE
        assert classify_surprise(-0.01) == SurpriseType.INLINE

    def test_small_miss(self):
        assert classify_surprise(-0.03) == SurpriseType.SMALL_MISS

    def test_miss(self):
        assert classify_surprise(-0.07) == SurpriseType.MISS

    def test_big_miss(self):
        assert classify_surprise(-0.15) == SurpriseType.BIG_MISS

    def test_boundary_values(self):
        # Exact boundaries: all use strict > comparison
        assert classify_surprise(0.10) == SurpriseType.BEAT       # NOT > 0.10, so BEAT
        assert classify_surprise(0.05) == SurpriseType.SMALL_BEAT # NOT > 0.05, so SMALL_BEAT
        assert classify_surprise(0.02) == SurpriseType.INLINE     # NOT > 0.02, so INLINE
        assert classify_surprise(-0.02) == SurpriseType.SMALL_MISS  # NOT > -0.02, so SMALL_MISS
        assert classify_surprise(-0.05) == SurpriseType.MISS      # NOT > -0.05, so MISS
        assert classify_surprise(-0.10) == SurpriseType.BIG_MISS  # NOT > -0.10, so BIG_MISS


class TestEarningsAnalysis:
    """Test EarningsAnalysis dataklasse."""

    def test_has_reported_true(self):
        ea = EarningsAnalysis(symbol="AAPL", date="2026-01-15", hour="amc", eps_actual=1.60)
        assert ea.has_reported is True

    def test_has_reported_false(self):
        ea = EarningsAnalysis(symbol="AAPL", date="2026-01-15", hour="amc")
        assert ea.has_reported is False

    def test_yoy_growth(self):
        ea = EarningsAnalysis(
            symbol="AAPL", date="2026-01-15", hour="amc",
            eps_actual=1.60, eps_previous=1.40,
        )
        expected = (1.60 - 1.40) / abs(1.40)
        assert ea.yoy_growth == pytest.approx(expected)

    def test_yoy_growth_none_when_no_actual(self):
        ea = EarningsAnalysis(symbol="AAPL", date="2026-01-15", hour="amc", eps_previous=1.40)
        assert ea.yoy_growth is None

    def test_yoy_growth_none_when_previous_zero(self):
        ea = EarningsAnalysis(
            symbol="AAPL", date="2026-01-15", hour="amc",
            eps_actual=1.60, eps_previous=0.0,
        )
        assert ea.yoy_growth is None

    def test_default_values(self):
        ea = EarningsAnalysis(symbol="TSLA", date="2026-03-01", hour="bmo")
        assert ea.is_upcoming is True
        assert ea.days_until == 0
        assert ea.avg_move_on_beat == 0.0
        assert ea.historical_beat_rate == 0.0


class TestEarningsCalendar:
    """Test EarningsCalendar dataklasse."""

    def test_has_earnings_today_true(self):
        cal = EarningsCalendar(
            today=[EarningsAnalysis(symbol="AAPL", date=_today(), hour="amc")],
        )
        assert cal.has_earnings_today is True

    def test_has_earnings_today_false(self):
        cal = EarningsCalendar()
        assert cal.has_earnings_today is False

    def test_symbols_reporting_soon(self):
        cal = EarningsCalendar(upcoming=[
            EarningsAnalysis(symbol="AAPL", date=_days_from_now(1), hour="amc", days_until=1),
            EarningsAnalysis(symbol="MSFT", date=_days_from_now(2), hour="bmo", days_until=2),
            EarningsAnalysis(symbol="GOOGL", date=_days_from_now(5), hour="amc", days_until=5),
        ])
        soon = cal.symbols_reporting_soon
        assert "AAPL" in soon
        assert "MSFT" in soon
        assert "GOOGL" not in soon

    def test_default_days_to_next(self):
        cal = EarningsCalendar()
        assert cal.days_to_next == 999


class TestPositionAdjustment:
    """Test PositionAdjustment dataklasse."""

    def test_repr(self):
        adj = PositionAdjustment(
            symbol="AAPL",
            reason="Earnings nærmer sig",
            reduction_pct=0.50,
            days_until_earnings=1,
            earnings_date="2026-03-17",
            earnings_hour="amc",
        )
        r = repr(adj)
        assert "AAPL" in r
        assert "50%" in r
        assert "1d" in r


class TestEarningsTracker:
    """Test EarningsTracker klasse."""

    def _make_tracker(self, events: list[EarningsEvent] | None = None):
        fetcher = MagicMock()
        fetcher.fetch_earnings_calendar.return_value = events or []
        return EarningsTracker(fetcher=fetcher), fetcher

    def test_get_calendar_empty(self):
        tracker, _ = self._make_tracker([])
        cal = tracker.get_calendar(["AAPL"])
        assert len(cal.upcoming) == 0
        assert len(cal.today) == 0
        assert len(cal.recent) == 0

    def test_get_calendar_upcoming(self):
        event = _make_earnings_event("AAPL", date=_days_from_now(3))
        tracker, _ = self._make_tracker([event])
        cal = tracker.get_calendar(["AAPL"])
        assert len(cal.upcoming) == 1
        assert cal.upcoming[0].symbol == "AAPL"
        assert cal.upcoming[0].days_until == 3

    def test_get_calendar_recent(self):
        event = _make_earnings_event(
            "AAPL",
            date=_days_from_now(-2),
            eps_actual=1.65,
            eps_estimate=1.50,
        )
        tracker, _ = self._make_tracker([event])
        cal = tracker.get_calendar(["AAPL"])
        assert len(cal.recent) == 1
        assert cal.recent[0].eps_actual == 1.65

    def test_get_calendar_next_event(self):
        events = [
            _make_earnings_event("AAPL", date=_days_from_now(5)),
            _make_earnings_event("MSFT", date=_days_from_now(2)),
        ]
        fetcher = MagicMock()
        fetcher.fetch_earnings_calendar.side_effect = [events[:1], events[1:]]
        tracker = EarningsTracker(fetcher=fetcher)
        cal = tracker.get_calendar(["AAPL", "MSFT"])
        assert cal.next_event is not None
        assert cal.next_event.symbol == "MSFT"
        assert cal.days_to_next == 2

    def test_position_adjustment_near_earnings(self):
        event = _make_earnings_event("AAPL", date=_days_from_now(1))
        tracker, _ = self._make_tracker([event])
        adjs = tracker.get_position_adjustments(["AAPL"])
        assert len(adjs) == 1
        assert adjs[0].symbol == "AAPL"
        assert adjs[0].reduction_pct == 0.50

    def test_no_adjustment_far_earnings(self):
        event = _make_earnings_event("AAPL", date=_days_from_now(10))
        tracker, _ = self._make_tracker([event])
        adjs = tracker.get_position_adjustments(["AAPL"])
        assert len(adjs) == 0

    def test_should_reduce_position_true(self):
        event = _make_earnings_event("AAPL", date=_days_from_now(1))
        tracker, _ = self._make_tracker([event])
        should, pct, reason = tracker.should_reduce_position("AAPL")
        assert should is True
        assert pct == 0.50
        assert reason != ""

    def test_should_reduce_position_false(self):
        tracker, _ = self._make_tracker([])
        should, pct, reason = tracker.should_reduce_position("AAPL")
        assert should is False
        assert pct == 0.0
        assert reason == ""

    def test_custom_reduction(self):
        event = _make_earnings_event("AAPL", date=_days_from_now(0))
        fetcher = MagicMock()
        fetcher.fetch_earnings_calendar.return_value = [event]
        tracker = EarningsTracker(
            fetcher=fetcher,
            pre_earnings_reduction=0.75,
            pre_earnings_days=2,
        )
        adjs = tracker.get_position_adjustments(["AAPL"])
        assert len(adjs) == 1
        assert adjs[0].reduction_pct == 0.75


class TestEarningsHistoricalAnalysis:
    """Test historisk earnings-reaktionsanalyse."""

    def test_empty_data(self):
        tracker, _ = TestEarningsTracker._make_tracker(TestEarningsTracker(), [])
        result = tracker.analyze_historical_reaction("AAPL", pd.DataFrame(), [])
        assert result["earnings_count"] == 0
        assert result["avg_move_pct"] == 0.0

    def test_none_data(self):
        tracker, _ = TestEarningsTracker._make_tracker(TestEarningsTracker(), [])
        result = tracker.analyze_historical_reaction("AAPL", None, ["2026-01-15"])
        assert result["earnings_count"] == 0

    def test_valid_analysis(self):
        tracker, _ = TestEarningsTracker._make_tracker(TestEarningsTracker(), [])
        df = _make_price_df(200)
        dates = [df.index[50].strftime("%Y-%m-%d"), df.index[100].strftime("%Y-%m-%d")]
        result = tracker.analyze_historical_reaction("AAPL", df, dates, window_days=5)
        assert result["earnings_count"] == 2
        assert result["symbol"] == "AAPL"
        assert len(result["moves"]) == 2

    def test_invalid_dates_skipped(self):
        tracker, _ = TestEarningsTracker._make_tracker(TestEarningsTracker(), [])
        df = _make_price_df(50)
        result = tracker.analyze_historical_reaction("AAPL", df, ["invalid-date"])
        assert result["earnings_count"] == 0


class TestEarningsSurpriseAnalysis:
    """Test analyze_surprises."""

    def test_empty_events(self):
        tracker, _ = TestEarningsTracker._make_tracker(TestEarningsTracker(), [])
        result = tracker.analyze_surprises([])
        assert result["total_reported"] == 0

    def test_mixed_surprises(self):
        tracker, _ = TestEarningsTracker._make_tracker(TestEarningsTracker(), [])
        events = [
            _make_earnings_event("AAPL", eps_estimate=1.50, eps_actual=1.80),  # beat
            _make_earnings_event("MSFT", eps_estimate=2.00, eps_actual=1.70),  # miss
            _make_earnings_event("GOOGL", eps_estimate=3.00, eps_actual=3.01),  # inline
        ]
        result = tracker.analyze_surprises(events)
        assert result["total_reported"] == 3
        assert result["beat_rate"] > 0
        assert "distribution" in result

    def test_only_upcoming_events(self):
        tracker, _ = TestEarningsTracker._make_tracker(TestEarningsTracker(), [])
        events = [_make_earnings_event("AAPL")]  # Ingen actual
        result = tracker.analyze_surprises(events)
        assert result["total_reported"] == 0


class TestPrintCalendar:
    """Test print_calendar output."""

    def test_print_calendar_no_crash(self, capsys):
        tracker, _ = TestEarningsTracker._make_tracker(TestEarningsTracker(), [])
        cal = EarningsCalendar(
            today=[EarningsAnalysis(symbol="AAPL", date=_today(), hour="amc", eps_estimate=1.50)],
            upcoming=[EarningsAnalysis(
                symbol="MSFT", date=_days_from_now(3), hour="bmo",
                eps_estimate=2.00, days_until=3,
            )],
            recent=[EarningsAnalysis(
                symbol="GOOGL", date=_days_from_now(-1), hour="amc",
                eps_estimate=3.00, eps_actual=3.30,
                surprise_pct=0.10, surprise_type=SurpriseType.BEAT,
            )],
        )
        tracker.print_calendar(cal)
        output = capsys.readouterr().out
        assert "EARNINGS KALENDER" in output
        assert "AAPL" in output
        assert "MSFT" in output
        assert "GOOGL" in output


# ══════════════════════════════════════════════════════════════
# MACRO CALENDAR TESTS
# ══════════════════════════════════════════════════════════════

class TestClassifyMacroEvent:
    """Test classify_macro_event funktion."""

    def test_fomc(self):
        assert classify_macro_event("FOMC Rate Decision") == MacroEventType.FOMC
        assert classify_macro_event("Federal Reserve Meeting") == MacroEventType.FOMC

    def test_nfp(self):
        assert classify_macro_event("Non-Farm Payrolls") == MacroEventType.NFP
        assert classify_macro_event("NFP Report") == MacroEventType.NFP

    def test_cpi(self):
        assert classify_macro_event("Consumer Price Index (YoY)") == MacroEventType.CPI
        assert classify_macro_event("Core CPI") == MacroEventType.CPI

    def test_ppi(self):
        assert classify_macro_event("Producer Price Index") == MacroEventType.PPI

    def test_gdp(self):
        assert classify_macro_event("GDP Growth Rate") == MacroEventType.GDP
        assert classify_macro_event("Gross Domestic Product") == MacroEventType.GDP

    def test_pmi(self):
        assert classify_macro_event("ISM Manufacturing PMI") == MacroEventType.PMI
        assert classify_macro_event("Purchasing Managers Index") == MacroEventType.PMI

    def test_retail_sales(self):
        assert classify_macro_event("Retail Sales (MoM)") == MacroEventType.RETAIL_SALES

    def test_unemployment(self):
        assert classify_macro_event("Unemployment Rate") == MacroEventType.UNEMPLOYMENT
        assert classify_macro_event("Initial Jobless Claims") == MacroEventType.UNEMPLOYMENT

    def test_other(self):
        assert classify_macro_event("Some Random Event") == MacroEventType.OTHER


class TestGetEventImpact:
    """Test get_event_impact funktion."""

    def test_critical_events(self):
        assert get_event_impact(MacroEventType.FOMC) == MacroImpact.CRITICAL
        assert get_event_impact(MacroEventType.NFP) == MacroImpact.CRITICAL

    def test_high_events(self):
        assert get_event_impact(MacroEventType.CPI) == MacroImpact.HIGH
        assert get_event_impact(MacroEventType.GDP) == MacroImpact.HIGH

    def test_medium_events(self):
        assert get_event_impact(MacroEventType.PPI) == MacroImpact.MEDIUM
        assert get_event_impact(MacroEventType.PMI) == MacroImpact.MEDIUM

    def test_low_events(self):
        assert get_event_impact(MacroEventType.OTHER) == MacroImpact.LOW


class TestMacroAnalysis:
    """Test MacroAnalysis dataklasse."""

    def test_has_reported(self):
        ma = MacroAnalysis(
            event_name="CPI", event_type=MacroEventType.CPI,
            impact=MacroImpact.HIGH, date="2026-03-01", actual=3.2,
        )
        assert ma.has_reported is True

    def test_not_reported(self):
        ma = MacroAnalysis(
            event_name="CPI", event_type=MacroEventType.CPI,
            impact=MacroImpact.HIGH, date="2026-03-01",
        )
        assert ma.has_reported is False

    def test_is_hot_cpi(self):
        ma = MacroAnalysis(
            event_name="CPI", event_type=MacroEventType.CPI,
            impact=MacroImpact.HIGH, date="2026-03-01",
            surprise=0.3,
        )
        assert ma.is_hot is True

    def test_is_hot_nfp(self):
        ma = MacroAnalysis(
            event_name="NFP", event_type=MacroEventType.NFP,
            impact=MacroImpact.CRITICAL, date="2026-03-01",
            surprise=50000,
        )
        assert ma.is_hot is True

    def test_is_cold_gdp(self):
        ma = MacroAnalysis(
            event_name="GDP", event_type=MacroEventType.GDP,
            impact=MacroImpact.HIGH, date="2026-03-01",
            surprise=-0.5,
        )
        assert ma.is_cold is True

    def test_not_hot_not_cold_when_no_surprise(self):
        ma = MacroAnalysis(
            event_name="CPI", event_type=MacroEventType.CPI,
            impact=MacroImpact.HIGH, date="2026-03-01",
        )
        assert ma.is_hot is False
        assert ma.is_cold is False


class TestMacroCalendarView:
    """Test MacroCalendarView dataklasse."""

    def test_has_critical_today_true(self):
        view = MacroCalendarView(today=[
            MacroAnalysis(
                event_name="FOMC", event_type=MacroEventType.FOMC,
                impact=MacroImpact.CRITICAL, date=_today(),
            ),
        ])
        assert view.has_critical_today is True

    def test_has_critical_today_false(self):
        view = MacroCalendarView(today=[
            MacroAnalysis(
                event_name="PMI", event_type=MacroEventType.PMI,
                impact=MacroImpact.MEDIUM, date=_today(),
            ),
        ])
        assert view.has_critical_today is False

    def test_has_critical_today_empty(self):
        view = MacroCalendarView()
        assert view.has_critical_today is False

    def test_critical_events_this_week(self):
        view = MacroCalendarView(upcoming=[
            MacroAnalysis(
                event_name="FOMC", event_type=MacroEventType.FOMC,
                impact=MacroImpact.CRITICAL, date=_days_from_now(3), days_until=3,
            ),
            MacroAnalysis(
                event_name="PMI", event_type=MacroEventType.PMI,
                impact=MacroImpact.MEDIUM, date=_days_from_now(2), days_until=2,
            ),
            MacroAnalysis(
                event_name="GDP", event_type=MacroEventType.GDP,
                impact=MacroImpact.HIGH, date=_days_from_now(10), days_until=10,
            ),
        ])
        critical = view.critical_events_this_week
        assert len(critical) == 1
        assert critical[0].event_type == MacroEventType.FOMC


class TestExposureAdjustment:
    """Test ExposureAdjustment dataklasse."""

    def test_repr(self):
        adj = ExposureAdjustment(
            reason="FOMC i morgen",
            reduction_pct=0.25,
            event_type=MacroEventType.FOMC,
            event_date="2026-03-17",
            days_until=1,
            impact=MacroImpact.CRITICAL,
        )
        r = repr(adj)
        assert "fomc" in r
        assert "25%" in r


class TestMacroCalendar:
    """Test MacroCalendar klasse."""

    def _make_calendar(self, events: list[EconomicEvent] | None = None):
        fetcher = MagicMock()
        fetcher.fetch_economic_calendar.return_value = events or []
        return MacroCalendar(fetcher=fetcher), fetcher

    def test_get_calendar_empty(self):
        cal, _ = self._make_calendar([])
        view = cal.get_calendar()
        assert len(view.upcoming) == 0
        assert len(view.today) == 0
        assert len(view.recent) == 0

    def test_get_calendar_upcoming_fomc(self):
        event = _make_economic_event("FOMC Rate Decision", date=_days_from_now(5))
        cal, _ = self._make_calendar([event])
        view = cal.get_calendar()
        assert len(view.upcoming) == 1
        assert view.upcoming[0].event_type == MacroEventType.FOMC
        assert view.upcoming[0].impact == MacroImpact.CRITICAL

    def test_get_calendar_recent_with_actual(self):
        event = _make_economic_event(
            "CPI (YoY)", date=_days_from_now(-2),
            estimate=3.0, actual=3.2, previous=2.9,
        )
        cal, _ = self._make_calendar([event])
        view = cal.get_calendar()
        assert len(view.recent) == 1
        assert view.recent[0].actual == 3.2
        assert view.recent[0].surprise == pytest.approx(0.2)

    def test_get_calendar_filters_country(self):
        events = [
            _make_economic_event("CPI (YoY)", country="US"),
            _make_economic_event("UK CPI", country="UK"),
        ]
        cal, _ = self._make_calendar(events)
        view = cal.get_calendar(country="US")
        # UK event should be filtered out
        all_events = view.upcoming + view.today + view.recent
        for e in all_events:
            assert e.country == "US"

    def test_next_critical_event(self):
        events = [
            _make_economic_event("PMI", date=_days_from_now(2)),
            _make_economic_event("FOMC Rate Decision", date=_days_from_now(7)),
        ]
        cal, _ = self._make_calendar(events)
        view = cal.get_calendar()
        assert view.next_critical is not None
        assert view.next_critical.event_type == MacroEventType.FOMC

    def test_exposure_adjustment_near_event(self):
        event = _make_economic_event("FOMC Rate Decision", date=_days_from_now(1))
        cal, _ = self._make_calendar([event])
        adjs = cal.get_exposure_adjustments()
        assert len(adjs) == 1
        assert adjs[0].reduction_pct == 0.25
        assert adjs[0].event_type == MacroEventType.FOMC

    def test_no_adjustment_far_event(self):
        event = _make_economic_event("FOMC Rate Decision", date=_days_from_now(15))
        cal, _ = self._make_calendar([event])
        adjs = cal.get_exposure_adjustments()
        assert len(adjs) == 0

    def test_exposure_adjustment_cpi(self):
        event = _make_economic_event("CPI (YoY)", date=_days_from_now(1))
        cal, _ = self._make_calendar([event])
        adjs = cal.get_exposure_adjustments()
        assert len(adjs) == 1
        assert adjs[0].reduction_pct == 0.15  # HIGH = 15%

    def test_no_adjustment_low_impact(self):
        event = _make_economic_event("Some Random Event", date=_days_from_now(0))
        cal, _ = self._make_calendar([event])
        adjs = cal.get_exposure_adjustments()
        assert len(adjs) == 0  # LOW impact = 0% reduction

    def test_total_reduction_takes_max(self):
        events = [
            _make_economic_event("FOMC Rate Decision", date=_days_from_now(1)),
            _make_economic_event("CPI (YoY)", date=_days_from_now(1)),
        ]
        cal, _ = self._make_calendar(events)
        reduction, reasons = cal.get_total_reduction()
        assert reduction == 0.25  # Max of 25% (FOMC) and 15% (CPI)
        assert len(reasons) == 2

    def test_total_reduction_no_events(self):
        cal, _ = self._make_calendar([])
        reduction, reasons = cal.get_total_reduction()
        assert reduction == 0.0
        assert len(reasons) == 0

    def test_should_reduce_exposure_true(self):
        event = _make_economic_event("FOMC Rate Decision", date=_days_from_now(0))
        cal, _ = self._make_calendar([event])
        should, pct, reason = cal.should_reduce_exposure()
        assert should is True
        assert pct == 0.25
        assert reason != ""

    def test_should_reduce_exposure_false(self):
        cal, _ = self._make_calendar([])
        should, pct, reason = cal.should_reduce_exposure()
        assert should is False
        assert pct == 0.0

    def test_custom_reduction_map(self):
        event = _make_economic_event("FOMC Rate Decision", date=_days_from_now(0))
        fetcher = MagicMock()
        fetcher.fetch_economic_calendar.return_value = [event]
        cal = MacroCalendar(
            fetcher=fetcher,
            reduction_map={
                MacroImpact.CRITICAL: 0.50,
                MacroImpact.HIGH: 0.30,
                MacroImpact.MEDIUM: 0.15,
                MacroImpact.LOW: 0.00,
            },
        )
        adjs = cal.get_exposure_adjustments()
        assert adjs[0].reduction_pct == 0.50

    def test_custom_pre_event_days(self):
        event = _make_economic_event("FOMC Rate Decision", date=_days_from_now(2))
        fetcher = MagicMock()
        fetcher.fetch_economic_calendar.return_value = [event]
        cal = MacroCalendar(fetcher=fetcher, pre_event_days=3)
        adjs = cal.get_exposure_adjustments()
        assert len(adjs) == 1  # 2 days <= 3 pre_event_days


class TestMacroFOMCSpecific:
    """Test FOMC-specifik logik."""

    def test_is_fomc_day_true(self):
        event = _make_economic_event("FOMC Rate Decision", date=_today())
        fetcher = MagicMock()
        fetcher.fetch_economic_calendar.return_value = [event]
        cal = MacroCalendar(fetcher=fetcher)
        assert cal.is_fomc_day(_today()) is True

    def test_is_fomc_day_false(self):
        event = _make_economic_event("CPI (YoY)", date=_today())
        fetcher = MagicMock()
        fetcher.fetch_economic_calendar.return_value = [event]
        cal = MacroCalendar(fetcher=fetcher)
        assert cal.is_fomc_day(_today()) is False

    def test_next_fomc_date(self):
        events = [
            _make_economic_event("CPI (YoY)", date=_days_from_now(5)),
            _make_economic_event("FOMC Rate Decision", date=_days_from_now(10)),
        ]
        fetcher = MagicMock()
        fetcher.fetch_economic_calendar.return_value = events
        cal = MacroCalendar(fetcher=fetcher)
        fomc_date = cal.next_fomc_date()
        assert fomc_date == _days_from_now(10)

    def test_next_fomc_date_none(self):
        fetcher = MagicMock()
        fetcher.fetch_economic_calendar.return_value = []
        cal = MacroCalendar(fetcher=fetcher)
        assert cal.next_fomc_date() is None


class TestMacroHistoricalAnalysis:
    """Test historisk makro-reaktionsanalyse."""

    def test_empty_data(self):
        cal, _ = TestMacroCalendar._make_calendar(TestMacroCalendar(), [])
        result = cal.analyze_historical_reaction(
            MacroEventType.FOMC, pd.DataFrame(), [],
        )
        assert result["event_count"] == 0
        assert result["avg_move_pct"] == 0.0

    def test_none_data(self):
        cal, _ = TestMacroCalendar._make_calendar(TestMacroCalendar(), [])
        result = cal.analyze_historical_reaction(
            MacroEventType.FOMC, None, ["2026-01-15"],
        )
        assert result["event_count"] == 0

    def test_valid_analysis(self):
        cal, _ = TestMacroCalendar._make_calendar(TestMacroCalendar(), [])
        df = _make_price_df(200)
        dates = [df.index[50].strftime("%Y-%m-%d"), df.index[100].strftime("%Y-%m-%d")]
        result = cal.analyze_historical_reaction(MacroEventType.FOMC, df, dates)
        assert result["event_count"] == 2
        assert result["event_type"] == "fomc"
        assert len(result["moves"]) == 2
        assert "avg_volatility" in result

    def test_invalid_dates_skipped(self):
        cal, _ = TestMacroCalendar._make_calendar(TestMacroCalendar(), [])
        df = _make_price_df(50)
        result = cal.analyze_historical_reaction(
            MacroEventType.CPI, df, ["not-a-date"],
        )
        assert result["event_count"] == 0


class TestMacroPrintCalendar:
    """Test print_calendar output."""

    def test_print_no_crash(self, capsys):
        cal, _ = TestMacroCalendar._make_calendar(TestMacroCalendar(), [])
        view = MacroCalendarView(
            today=[MacroAnalysis(
                event_name="FOMC Rate Decision",
                event_type=MacroEventType.FOMC,
                impact=MacroImpact.CRITICAL,
                date=_today(),
                estimate=5.25,
            )],
            upcoming=[MacroAnalysis(
                event_name="NFP",
                event_type=MacroEventType.NFP,
                impact=MacroImpact.CRITICAL,
                date=_days_from_now(3),
                days_until=3,
            )],
            recent=[MacroAnalysis(
                event_name="CPI (YoY)",
                event_type=MacroEventType.CPI,
                impact=MacroImpact.HIGH,
                date=_days_from_now(-1),
                estimate=3.0,
                actual=3.2,
                previous=2.9,
            )],
            next_critical=MacroAnalysis(
                event_name="NFP",
                event_type=MacroEventType.NFP,
                impact=MacroImpact.CRITICAL,
                date=_days_from_now(3),
                days_until=3,
            ),
            days_to_next_critical=3,
        )
        cal.print_calendar(view)
        output = capsys.readouterr().out
        assert "MAKROØKONOMISK KALENDER" in output
        assert "FOMC" in output
        assert "NFP" in output
        assert "CPI" in output


# ══════════════════════════════════════════════════════════════
# IMPORT TESTS
# ══════════════════════════════════════════════════════════════

class TestImports:
    """Test at alle exports fra __init__.py virker."""

    def test_earnings_tracker_imports(self):
        from src.sentiment import (
            EarningsTracker,
            EarningsAnalysis,
            EarningsCalendar,
            PositionAdjustment,
            SurpriseType,
            classify_surprise,
        )
        assert EarningsTracker is not None
        assert SurpriseType.BIG_BEAT.value == "big_beat"

    def test_macro_calendar_imports(self):
        from src.sentiment import (
            MacroCalendar,
            MacroAnalysis,
            MacroCalendarView,
            MacroEventType,
            MacroImpact,
            ExposureAdjustment,
            classify_macro_event,
            get_event_impact,
        )
        assert MacroCalendar is not None
        assert MacroEventType.FOMC.value == "fomc"
        assert MacroImpact.CRITICAL.value == "critical"
