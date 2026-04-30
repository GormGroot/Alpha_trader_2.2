"""
Tests for TaxAdvisor og Notifier:
  - QuarterlyEstimate (projektion, progressionsgrænse)
  - Tax-loss harvesting
  - Wash sale detektion
  - YearEndReport
  - Alerts og notifikationer
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.risk.portfolio_tracker import Position
from src.tax.tax_advisor import (
    TaxAdvisor,
    QuarterlyEstimate,
    TaxLossCandidate,
    WashSaleWarning,
    YearEndReport,
    TaxAlert,
)
from src.tax.tax_calculator import DanishTaxCalculator
from src.notifications.notifier import (
    Notifier,
    LogChannel,
    CallbackChannel,
)


# ══════════════════════════════════════════════════════════════
#  Hjælper-fixtures
# ══════════════════════════════════════════════════════════════


def _make_transactions(gains: list[float], losses: list[float] | None = None):
    """Opret test-transaktioner med givne gevinster/tab."""
    txs = []
    for i, g in enumerate(gains):
        txs.append({
            "symbol": f"SYM{i}",
            "qty": 10,
            "entry_value_dkk": 10_000,
            "exit_value_dkk": 10_000 + g,
            "entry_date": "2026-01-15",
            "trade_date": "2026-06-15",
            "realized_pnl_dkk": g,
        })
    for i, loss in enumerate(losses or []):
        txs.append({
            "symbol": f"LOSS{i}",
            "qty": 10,
            "entry_value_dkk": 10_000,
            "exit_value_dkk": 10_000 + loss,  # loss er negativt
            "entry_date": "2026-02-01",
            "trade_date": "2026-07-01",
            "realized_pnl_dkk": loss,
        })
    return txs


def _make_advisor(**kwargs) -> TaxAdvisor:
    return TaxAdvisor(
        progression_limit=kwargs.get("progression_limit", 61_000),
        carried_losses=kwargs.get("carried_losses", 0.0),
        fx_rate=kwargs.get("fx_rate", 6.90),
    )


def _make_position(symbol="AAPL", qty=10, entry_price=150.0,
                   current_price=140.0, entry_time=None) -> Position:
    """Opret en Position med urealiseret tab."""
    return Position(
        symbol=symbol,
        side="long",
        qty=qty,
        entry_price=entry_price,
        entry_time=entry_time or "2026-01-15T10:00:00",
        current_price=current_price,
    )


# ══════════════════════════════════════════════════════════════
#  QuarterlyEstimate
# ══════════════════════════════════════════════════════════════


class TestQuarterlyEstimate:
    def test_empty_transactions(self):
        advisor = _make_advisor()
        est = advisor.quarterly_estimate([], year=2026)
        assert isinstance(est, QuarterlyEstimate)
        assert est.net_ytd_dkk == 0.0
        assert est.num_trades_ytd == 0
        assert est.tax_ytd_dkk == 0.0

    def test_ytd_gains(self):
        advisor = _make_advisor()
        txs = _make_transactions(gains=[10_000, 5_000])
        est = advisor.quarterly_estimate(txs, year=2026)
        assert est.realized_gain_ytd_dkk == pytest.approx(15_000)
        assert est.net_ytd_dkk == pytest.approx(15_000)
        assert est.num_trades_ytd == 2

    def test_ytd_mixed(self):
        advisor = _make_advisor()
        txs = _make_transactions(gains=[20_000], losses=[-5_000])
        est = advisor.quarterly_estimate(txs, year=2026)
        assert est.realized_gain_ytd_dkk == pytest.approx(20_000)
        assert est.realized_loss_ytd_dkk == pytest.approx(-5_000)
        assert est.net_ytd_dkk == pytest.approx(15_000)

    def test_projected_annual_gain_positive(self):
        advisor = _make_advisor()
        txs = _make_transactions(gains=[30_000])
        est = advisor.quarterly_estimate(txs, year=2026)
        # Projektion skal være >= YTD (lineær fremskrivning)
        assert est.projected_annual_gain_dkk >= 30_000

    def test_progression_limit_tracking(self):
        advisor = _make_advisor(progression_limit=61_000)
        txs = _make_transactions(gains=[50_000])
        est = advisor.quarterly_estimate(txs, year=2026)
        assert est.pct_of_limit_used > 0
        assert est.remaining_before_42pct <= 61_000

    def test_projected_hits_limit(self):
        advisor = _make_advisor(progression_limit=20_000)
        txs = _make_transactions(gains=[15_000])
        est = advisor.quarterly_estimate(txs, year=2026)
        # Med 15k allerede og lineær fremskrivning, bør den ramme 20k grænsen
        assert est.projected_hits_limit is True

    def test_audit_notes_present(self):
        advisor = _make_advisor()
        txs = _make_transactions(gains=[10_000])
        est = advisor.quarterly_estimate(txs, year=2026)
        assert len(est.audit_notes) >= 2
        assert any("YTD" in note for note in est.audit_notes)

    def test_quarter_calculation(self):
        advisor = _make_advisor()
        est = advisor.quarterly_estimate([], year=2026)
        assert 1 <= est.quarter <= 4

    def test_carried_losses_affect_projection(self):
        advisor = _make_advisor(carried_losses=20_000)
        txs = _make_transactions(gains=[30_000])
        est = advisor.quarterly_estimate(txs, year=2026)
        # Taxable = 30_000 - 20_000 = 10_000
        assert est.pct_of_limit_used < 50  # Under 50% med carried losses


# ══════════════════════════════════════════════════════════════
#  Progression Warning
# ══════════════════════════════════════════════════════════════


class TestProgressionWarning:
    def test_no_warning_far_from_limit(self):
        advisor = _make_advisor(progression_limit=61_000)
        alert = advisor.check_progression_warning(10_000)
        assert alert is None

    def test_warning_near_limit(self):
        advisor = _make_advisor(progression_limit=61_000)
        alert = advisor.check_progression_warning(55_000)
        assert alert is not None
        assert alert.severity == "WARNING"
        assert alert.category == "progression"

    def test_warning_over_limit(self):
        advisor = _make_advisor(progression_limit=61_000)
        alert = advisor.check_progression_warning(80_000)
        assert alert is not None
        assert "overskredet" in alert.title.lower()

    def test_planned_sale_exceeds(self):
        advisor = _make_advisor(progression_limit=61_000)
        alert = advisor.check_progression_warning(55_000, planned_sale_gain_dkk=10_000)
        assert alert is not None
        assert alert.data.get("will_exceed_with_sale") is True

    def test_carried_losses_reduce_warning(self):
        advisor = _make_advisor(progression_limit=61_000, carried_losses=20_000)
        # 55_000 - 20_000 = 35_000 taxable → langt fra grænsen
        alert = advisor.check_progression_warning(55_000)
        assert alert is None


# ══════════════════════════════════════════════════════════════
#  Tax-Loss Harvesting
# ══════════════════════════════════════════════════════════════


class TestTaxLossHarvesting:
    def test_no_candidates_all_profit(self):
        advisor = _make_advisor()
        positions = [
            _make_position(current_price=160.0),  # Gevinst
        ]
        candidates = advisor.find_tax_loss_candidates(positions)
        assert len(candidates) == 0

    def test_candidates_with_loss(self):
        advisor = _make_advisor()
        positions = [
            _make_position(symbol="AAPL", current_price=130.0),  # Tab
            _make_position(symbol="MSFT", current_price=160.0),  # Gevinst
        ]
        candidates = advisor.find_tax_loss_candidates(
            positions, current_gain_dkk=20_000,
        )
        assert len(candidates) == 1
        assert candidates[0].symbol == "AAPL"
        assert candidates[0].unrealized_pnl_usd < 0

    def test_potential_saving_calculated(self):
        advisor = _make_advisor()
        positions = [
            _make_position(symbol="AAPL", qty=100, entry_price=150.0, current_price=140.0),
        ]
        candidates = advisor.find_tax_loss_candidates(
            positions, current_gain_dkk=30_000,
        )
        assert len(candidates) == 1
        assert candidates[0].potential_tax_saving_dkk > 0

    def test_sorted_by_saving(self):
        advisor = _make_advisor()
        positions = [
            _make_position(symbol="SMALL", qty=10, entry_price=100, current_price=95),
            _make_position(symbol="BIG", qty=100, entry_price=100, current_price=80),
        ]
        candidates = advisor.find_tax_loss_candidates(
            positions, current_gain_dkk=50_000,
        )
        assert len(candidates) == 2
        assert candidates[0].symbol == "BIG"  # Størst besparelse først

    def test_recommendation_text(self):
        advisor = _make_advisor()
        positions = [
            _make_position(symbol="AAPL", qty=100, entry_price=150, current_price=130),
        ]
        candidates = advisor.find_tax_loss_candidates(
            positions, current_gain_dkk=30_000,
        )
        assert candidates[0].recommendation
        assert "vejledende" in candidates[0].recommendation.lower()


# ══════════════════════════════════════════════════════════════
#  Wash Sale
# ══════════════════════════════════════════════════════════════


class TestWashSale:
    def test_no_warning_without_recent_sell(self):
        advisor = _make_advisor()
        warning = advisor.check_wash_sale("AAPL", "2026-06-15")
        assert warning is None

    def test_warning_within_30_days(self):
        advisor = _make_advisor()
        sells = [{
            "symbol": "AAPL",
            "date": "2026-06-01",
            "price": 140.0,
            "pnl_dkk": -5_000,  # Tab
        }]
        warning = advisor.check_wash_sale(
            "AAPL", "2026-06-20", recent_sells=sells,
        )
        assert warning is not None
        assert isinstance(warning, WashSaleWarning)
        assert warning.days_since_sell == 19
        assert "wash sale" in warning.warning.lower()

    def test_no_warning_after_30_days(self):
        advisor = _make_advisor()
        sells = [{
            "symbol": "AAPL",
            "date": "2026-05-01",
            "price": 140.0,
            "pnl_dkk": -5_000,
        }]
        warning = advisor.check_wash_sale(
            "AAPL", "2026-06-15", recent_sells=sells,
        )
        assert warning is None  # 45 dage > 30

    def test_no_warning_for_profit_sale(self):
        advisor = _make_advisor()
        sells = [{
            "symbol": "AAPL",
            "date": "2026-06-01",
            "price": 160.0,
            "pnl_dkk": 5_000,  # Gevinst, ikke tab
        }]
        warning = advisor.check_wash_sale(
            "AAPL", "2026-06-20", recent_sells=sells,
        )
        assert warning is None

    def test_no_warning_different_symbol(self):
        advisor = _make_advisor()
        sells = [{
            "symbol": "MSFT",
            "date": "2026-06-01",
            "price": 140.0,
            "pnl_dkk": -5_000,
        }]
        warning = advisor.check_wash_sale(
            "AAPL", "2026-06-10", recent_sells=sells,
        )
        assert warning is None

    def test_register_sell_and_detect(self):
        advisor = _make_advisor()
        advisor.register_sell("AAPL", "2026-06-01", 140.0, -5_000)
        warning = advisor.check_wash_sale("AAPL", "2026-06-20")
        assert warning is not None

    def test_old_sells_cleaned_up(self):
        advisor = _make_advisor()
        old_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        advisor.register_sell("AAPL", old_date, 140.0, -5_000)
        # Clean-up sker i register_sell
        advisor.register_sell("MSFT", datetime.now().strftime("%Y-%m-%d"), 300.0, 1_000)
        assert len(advisor._recent_sells) == 1  # Kun MSFT


# ══════════════════════════════════════════════════════════════
#  YearEndReport
# ══════════════════════════════════════════════════════════════


class TestYearEndReport:
    def test_empty_year(self):
        advisor = _make_advisor()
        report = advisor.year_end_report([], year=2026)
        assert isinstance(report, YearEndReport)
        assert report.year == 2026
        assert report.tax_result is not None

    def test_with_transactions(self):
        advisor = _make_advisor()
        txs = _make_transactions(gains=[20_000, 10_000], losses=[-5_000])
        report = advisor.year_end_report(txs, year=2026)
        assert report.tax_result.net_gain_dkk == pytest.approx(25_000)
        assert report.tax_result.total_tax_dkk > 0

    def test_actions_generated(self):
        advisor = _make_advisor()
        txs = _make_transactions(gains=[20_000], losses=[-5_000])
        report = advisor.year_end_report(txs, year=2026)
        assert len(report.actions) >= 1
        # Skal altid afslutte med disclaimer
        assert any("vejledende" in a.lower() for a in report.actions)

    def test_deadlines_present(self):
        advisor = _make_advisor()
        report = advisor.year_end_report([], year=2026)
        assert len(report.deadlines) >= 3
        assert any("december" in dl.lower() for dl in report.deadlines)
        assert any("marts" in dl.lower() or "march" in dl.lower()
                    for dl in report.deadlines)

    def test_harvest_candidates_included(self):
        advisor = _make_advisor()
        txs = _make_transactions(gains=[30_000])
        positions = [
            _make_position(symbol="LOSS1", qty=50, entry_price=100, current_price=80),
        ]
        report = advisor.year_end_report(txs, positions=positions, year=2026)
        assert len(report.harvest_candidates) >= 1
        assert report.potential_saving_dkk > 0

    def test_audit_trail(self):
        advisor = _make_advisor()
        txs = _make_transactions(gains=[10_000])
        report = advisor.year_end_report(txs, year=2026)
        assert len(report.audit_notes) >= 1

    def test_summary_lines_format(self):
        advisor = _make_advisor()
        txs = _make_transactions(gains=[20_000])
        report = advisor.year_end_report(txs, year=2026)
        lines = report.summary_lines
        assert any("SKATTEFORBEREDELSE" in l for l in lines)
        assert any("DEADLINE" in l.upper() for l in lines)
        assert any("vejledende" in l.lower() for l in lines)

    def test_near_progression_limit_action(self):
        advisor = _make_advisor(progression_limit=25_000)
        txs = _make_transactions(gains=[20_000])
        report = advisor.year_end_report(txs, year=2026)
        assert any("progressionsgrænse" in a.lower() for a in report.actions)


# ══════════════════════════════════════════════════════════════
#  TaxAlert
# ══════════════════════════════════════════════════════════════


class TestTaxAlert:
    def test_alert_defaults(self):
        alert = TaxAlert(
            severity="INFO",
            title="Test",
            message="Besked",
            category="test",
        )
        assert alert.timestamp != ""
        assert alert.severity == "INFO"

    def test_monthly_status(self):
        advisor = _make_advisor()
        txs = _make_transactions(gains=[15_000])
        alert = advisor.generate_monthly_status(txs, year=2026)
        assert alert.severity == "INFO"
        assert alert.category == "monthly"
        assert "YTD" in alert.message or "status" in alert.title.lower()

    def test_march_reminder(self):
        advisor = _make_advisor()
        alert = advisor.generate_march_reminder(2027)
        assert alert.severity == "CRITICAL"
        assert alert.category == "deadline"
        assert "Rubrik 66" in alert.message

    def test_collect_alerts_empty(self):
        advisor = _make_advisor()
        alerts = advisor.collect_pending_alerts([], year=2026)
        # Kan have december/marts alerts alt efter måned
        assert isinstance(alerts, list)

    def test_collect_alerts_sorted_by_severity(self):
        advisor = _make_advisor(progression_limit=10_000)
        txs = _make_transactions(gains=[15_000])
        alerts = advisor.collect_pending_alerts(txs, year=2026)
        if len(alerts) >= 2:
            severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
            for i in range(len(alerts) - 1):
                assert severity_order.get(alerts[i].severity, 3) <= \
                       severity_order.get(alerts[i + 1].severity, 3)


# ══════════════════════════════════════════════════════════════
#  Notifier
# ══════════════════════════════════════════════════════════════


class TestNotifier:
    @pytest.fixture(autouse=True)
    def _no_telegram(self, monkeypatch):
        """
        Isolér Notifier-tests fra Telegram-env-vars.

        Notifier auto-tilføjer en TelegramChannel hvis TELEGRAM_BOT_TOKEN
        og TELEGRAM_CHAT_ID er sat i miljøet. Det forstyrrer count-tests
        der antager kun LogChannel som baseline.
        """
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        yield

    def _make(self) -> Notifier:
        tmpdir = tempfile.mkdtemp()
        return Notifier(cache_dir=tmpdir)

    def test_send_returns_count(self):
        n = self._make()
        sent = n.send("INFO", "Test", "Besked", "test")
        assert sent >= 1  # LogChannel er altid aktiv

    def test_send_tax_alert(self):
        n = self._make()
        alert = TaxAlert(
            severity="WARNING",
            title="Test Alert",
            message="Testbesked",
            category="progression",
        )
        sent = n.send_tax_alert(alert)
        assert sent >= 1

    def test_callback_channel(self):
        n = self._make()
        received = []
        n.add_channel(CallbackChannel(
            lambda s, t, m, c: received.append((s, t, m, c))
        ))
        n.send("INFO", "Callback Test", "Hej", "test")
        assert len(received) == 1
        assert received[0][1] == "Callback Test"

    def test_history_saved(self):
        n = self._make()
        n.send("WARNING", "Historik Test", "Besked", "test")
        history = n.get_history()
        assert len(history) >= 1
        assert history[0]["title"] == "Historik Test"

    def test_history_filter_category(self):
        n = self._make()
        n.send("INFO", "A", "msg", "skat")
        n.send("INFO", "B", "msg", "risiko")
        history = n.get_history(category="skat")
        assert all(h["category"] == "skat" for h in history)

    def test_unread_count(self):
        n = self._make()
        n.send("INFO", "Test1", "msg", "test")
        n.send("INFO", "Test2", "msg", "test")
        count = n.get_unread_count()
        assert count >= 2

    def test_multiple_channels(self):
        n = self._make()
        counter = {"count": 0}

        def cb(s, t, m, c):
            counter["count"] += 1

        n.add_channel(CallbackChannel(cb))
        n.add_channel(CallbackChannel(cb))
        sent = n.send("INFO", "Multi", "msg", "test")
        # LogChannel + 2 callbacks = 3
        assert sent == 3
        assert counter["count"] == 2

    def test_failed_channel_doesnt_break(self):
        n = self._make()
        n.add_channel(CallbackChannel(
            lambda s, t, m, c: (_ for _ in ()).throw(RuntimeError("Boom"))
        ))
        # Skal ikke raise
        sent = n.send("INFO", "Error Test", "msg", "test")
        assert sent >= 1  # LogChannel succeeded
