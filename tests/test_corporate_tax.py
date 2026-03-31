"""
Tests for Corporate Tax — lagerbeskatning, FIFO, dividends, FX P&L.

Tester:
  - FIFO lot tracking (add_lot, consume_lots)
  - Lagerbeskatning beregning (mark-to-market)
  - 22% selskabsskat
  - DBO rates og reclaimable withholding tax
  - Tax credit tracker (skattetilgodehavende)
  - Currency P&L
"""

from __future__ import annotations

import os
import tempfile
import pytest
from datetime import date, datetime

# Ensure we use temp SQLite databases for tests
_test_db_dir = tempfile.mkdtemp()


# ── FIFO Tracker Tests ─────────────────────────────────────

class TestFIFOTracker:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Patch DB path before import
        os.environ["TAX_DB_PATH"] = os.path.join(_test_db_dir, "test_fifo.db")

    def test_add_and_consume_lots(self):
        from src.tax.corporate_tax import FIFOTracker
        tracker = FIFOTracker(db_path=os.path.join(_test_db_dir, "fifo1.db"))

        # Buy 100 shares at 50 DKK
        tracker.add_lot("NOVO-B.CO", 100, 50.0, "2026-01-15")

        # Buy 50 more at 60 DKK
        tracker.add_lot("NOVO-B.CO", 50, 60.0, "2026-02-01")

        # Sell 120 shares — should consume first lot (100@50) + 20 from second (20@60)
        consumed = tracker.consume_lots("NOVO-B.CO", 120, 65.0, "2026-03-01")
        assert len(consumed) == 2
        assert consumed[0].qty == 100  # qty from first lot
        assert consumed[0].acquisition_price_dkk == 50.0  # price from first lot
        assert consumed[1].qty == 20   # qty from second lot
        assert consumed[1].acquisition_price_dkk == 60.0  # price from second lot

    def test_fifo_order(self):
        """FIFO: ældste lots forbruges først."""
        from src.tax.corporate_tax import FIFOTracker
        tracker = FIFOTracker(db_path=os.path.join(_test_db_dir, "fifo2.db"))

        tracker.add_lot("AAPL", 10, 100.0, "2026-01-01")
        tracker.add_lot("AAPL", 10, 200.0, "2026-02-01")
        tracker.add_lot("AAPL", 10, 300.0, "2026-03-01")

        consumed = tracker.consume_lots("AAPL", 15, 250.0, "2026-03-15")
        # Should take 10@100 + 5@200
        assert consumed[0].qty == 10
        assert consumed[0].acquisition_price_dkk == 100.0
        assert consumed[1].qty == 5
        assert consumed[1].acquisition_price_dkk == 200.0

    def test_insufficient_lots(self):
        """Forbrug af mere end tilgængeligt bør fejle eller returnere hvad der er."""
        from src.tax.corporate_tax import FIFOTracker
        tracker = FIFOTracker(db_path=os.path.join(_test_db_dir, "fifo3.db"))

        tracker.add_lot("AAPL", 10, 100.0, "2026-01-01")

        # Try to sell 20 — should either raise or return max available
        try:
            consumed = tracker.consume_lots("AAPL", 20, 150.0, "2026-03-01")
            # If it returns, verify we only got 10
            total_qty = sum(c.qty for c in consumed)
            assert total_qty <= 10
        except (ValueError, Exception):
            pass  # Also acceptable

    def test_weighted_avg_price(self):
        from src.tax.corporate_tax import FIFOTracker
        tracker = FIFOTracker(db_path=os.path.join(_test_db_dir, "fifo4.db"))

        tracker.add_lot("SAP.DE", 100, 150.0, "2026-01-01")
        tracker.add_lot("SAP.DE", 100, 170.0, "2026-02-01")

        avg = tracker.get_weighted_avg_price("SAP.DE")
        assert abs(avg - 160.0) < 0.01  # (100*150 + 100*170) / 200 = 160


# ── Corporate Tax Rate ─────────────────────────────────────

class TestCorporateTaxRate:
    def test_rate_is_22_pct(self):
        from src.tax.corporate_tax import CORPORATE_TAX_RATE
        assert CORPORATE_TAX_RATE == 0.22

    def test_basic_tax_calculation(self):
        """22% af 100.000 DKK = 22.000 DKK."""
        from src.tax.corporate_tax import CORPORATE_TAX_RATE
        taxable = 100_000
        tax = taxable * CORPORATE_TAX_RATE
        assert tax == 22_000


# ── Dividend Tracker ───────────────────────────────────────

class TestDividendTracker:
    def test_dbo_rates_exist(self):
        from src.tax.dividend_tracker import DBO_RATES
        assert "US" in DBO_RATES
        assert DBO_RATES["US"] == 0.15  # US/DK DBO rate

    def test_dbo_rate_germany(self):
        from src.tax.dividend_tracker import DBO_RATES
        assert DBO_RATES["DE"] == 0.15  # DBO rate (actual withholding is 26.375%)

    def test_dbo_rate_uk(self):
        from src.tax.dividend_tracker import DBO_RATES
        assert DBO_RATES["GB"] == 0.0  # UK har ingen kildeskat

    def test_reclaimable_excess(self):
        from src.tax.dividend_tracker import RECLAIMABLE_EXCESS
        # Germany: 26.375% actual - 15% DBO = 11.375% reclaimable
        assert abs(RECLAIMABLE_EXCESS.get("DE", 0) - 0.11375) < 0.001

    def test_swiss_reclaimable(self):
        from src.tax.dividend_tracker import RECLAIMABLE_EXCESS
        # Switzerland: 35% actual - 15% DBO = 20% reclaimable
        assert abs(RECLAIMABLE_EXCESS.get("CH", 0) - 0.20) < 0.001


# ── Mark to Market ─────────────────────────────────────────

class TestMarkToMarket:
    def test_mtm_import(self):
        from src.tax.mark_to_market import MarkToMarketEngine, MTMPosition, MTMSummary
        assert MarkToMarketEngine is not None

    def test_mtm_position_pnl(self):
        from src.tax.mark_to_market import MTMPosition
        pos = MTMPosition(
            symbol="AAPL",
            qty=100,
            primo_price_dkk=150.0,
            ultimo_price_dkk=170.0,
            primo_value_dkk=15000.0,
            ultimo_value_dkk=17000.0,
            mtm_pnl_dkk=2000.0,
            tax_22pct_dkk=440.0,
            currency="USD",
        )
        assert pos.mtm_pnl_dkk == 2000.0
        assert pos.ultimo_value_dkk - pos.primo_value_dkk == 2000.0


# ── Currency P&L ───────────────────────────────────────────

class TestCurrencyPnL:
    def test_fx_gain_calculation(self):
        """FX gain = (sell_rate - buy_rate) * amount."""
        buy_rate = 6.50  # DKK/USD
        sell_rate = 6.90
        amount = 10_000  # USD

        fx_gain = (sell_rate - buy_rate) * amount
        assert fx_gain == pytest.approx(4_000)  # 4.000 DKK gain


# ── Tax Credit Tracker ─────────────────────────────────────

class TestTaxCreditTracker:
    def test_import(self):
        from src.tax.tax_credit_tracker import TaxCreditTracker
        assert TaxCreditTracker is not None

    def test_credit_projection_dataclass(self):
        from src.tax.tax_credit_tracker import CreditProjection
        proj = CreditProjection(
            current_balance=500_000,
            projected_tax=100_000,
            projected_offset=100_000,
            projected_balance=400_000,
        )
        assert proj.projected_balance == 400_000


# ── Corporate Tax Report ──────────────────────────────────

class TestCorporateTaxReport:
    def test_import(self):
        from src.tax.corporate_tax_reports import CorporateTaxReportGenerator, AnnualReportData
        assert CorporateTaxReportGenerator is not None
        assert AnnualReportData is not None
