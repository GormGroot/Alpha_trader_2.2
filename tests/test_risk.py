"""
Tests for RiskManager og PortfolioTracker.

Risikostyring er det vigtigste modul – disse tests er grundige og
dækker edge-cases for at sikre at kapital beskyttes korrekt.
"""

from __future__ import annotations

import pytest

from src.risk.portfolio_tracker import PortfolioTracker, Position, ClosedTrade
from src.risk.risk_manager import (
    RiskManager,
    RiskDecision,
    RejectionReason,
    ExitSignal,
)


# ════════════════════════════════════════════════════════════
# Position tests
# ════════════════════════════════════════════════════════════

class TestPosition:

    def test_basic_long_pnl(self):
        pos = Position("AAPL", "long", 10, 150.0, "2024-01-01")
        pos.update_price(160.0)
        assert pos.unrealized_pnl == 100.0  # (160-150) * 10
        assert pos.unrealized_pnl_pct == pytest.approx(100 / 1500)

    def test_basic_short_pnl(self):
        pos = Position("AAPL", "short", 10, 150.0, "2024-01-01")
        pos.update_price(140.0)
        assert pos.unrealized_pnl == 100.0  # (150-140) * 10

    def test_long_loss(self):
        pos = Position("AAPL", "long", 10, 150.0, "2024-01-01")
        pos.update_price(145.0)
        assert pos.unrealized_pnl == -50.0

    def test_peak_tracking(self):
        pos = Position("AAPL", "long", 10, 100.0, "2024-01-01")
        pos.update_price(110.0)
        pos.update_price(105.0)
        assert pos.peak_price == 110.0
        assert pos.pct_from_peak == pytest.approx(5 / 110)

    def test_market_value(self):
        pos = Position("AAPL", "long", 10, 100.0, "2024-01-01")
        pos.update_price(120.0)
        assert pos.market_value == 1200.0

    def test_cost_basis(self):
        pos = Position("AAPL", "long", 10, 100.0, "2024-01-01")
        assert pos.cost_basis == 1000.0

    def test_defaults(self):
        pos = Position("AAPL", "long", 5, 200.0, "now")
        assert pos.current_price == 200.0
        assert pos.peak_price == 200.0


# ════════════════════════════════════════════════════════════
# ClosedTrade tests
# ════════════════════════════════════════════════════════════

class TestClosedTrade:

    def test_profitable_long(self):
        trade = ClosedTrade(
            "AAPL", "long", 10, 100.0, 110.0, "t0", "t1", "take_profit",
        )
        assert trade.realized_pnl == 100.0
        assert trade.realized_pnl_pct == pytest.approx(0.1)

    def test_losing_long(self):
        trade = ClosedTrade(
            "AAPL", "long", 10, 100.0, 95.0, "t0", "t1", "stop_loss",
        )
        assert trade.realized_pnl == -50.0

    def test_profitable_short(self):
        trade = ClosedTrade(
            "AAPL", "short", 10, 100.0, 90.0, "t0", "t1", "take_profit",
        )
        assert trade.realized_pnl == 100.0


# ════════════════════════════════════════════════════════════
# PortfolioTracker tests
# ════════════════════════════════════════════════════════════

class TestPortfolioTracker:

    def test_initial_state(self):
        pt = PortfolioTracker(100_000)
        assert pt.total_equity == 100_000
        assert pt.cash == 100_000
        assert pt.open_position_count == 0
        assert pt.total_return_pct == 0.0

    def test_open_position_reduces_cash(self):
        pt = PortfolioTracker(100_000)
        pt.open_position("AAPL", "long", 10, 150.0)
        assert pt.cash == 98_500
        assert pt.open_position_count == 1
        assert pt.total_equity == 100_000  # cash + market value

    def test_close_position_returns_cash(self):
        pt = PortfolioTracker(100_000)
        pt.open_position("AAPL", "long", 10, 150.0)
        trade = pt.close_position("AAPL", 160.0, "take_profit")

        assert pt.cash == 100_100  # 98500 + 10*160
        assert pt.open_position_count == 0
        assert trade.realized_pnl == 100.0
        assert len(pt.closed_trades) == 1

    def test_duplicate_position_raises(self):
        pt = PortfolioTracker(100_000)
        pt.open_position("AAPL", "long", 10, 150.0)
        with pytest.raises(ValueError, match="eksisterer allerede"):
            pt.open_position("AAPL", "long", 5, 155.0)

    def test_close_nonexistent_raises(self):
        pt = PortfolioTracker(100_000)
        with pytest.raises(ValueError, match="Ingen position"):
            pt.close_position("AAPL", 150.0)

    def test_insufficient_cash_raises(self):
        pt = PortfolioTracker(1_000)
        with pytest.raises(ValueError, match="Ikke nok kontanter"):
            pt.open_position("AAPL", "long", 100, 150.0)

    def test_update_prices(self):
        pt = PortfolioTracker(100_000)
        pt.open_position("AAPL", "long", 10, 150.0)
        pt.update_prices({"AAPL": 160.0})

        assert pt.positions["AAPL"].current_price == 160.0
        assert pt.total_equity == 98_500 + 1600  # cash + 10*160

    def test_unrealized_pnl(self):
        pt = PortfolioTracker(100_000)
        pt.open_position("AAPL", "long", 10, 100.0)
        pt.update_prices({"AAPL": 105.0})
        assert pt.total_unrealized_pnl == 50.0

    def test_daily_pnl(self):
        pt = PortfolioTracker(100_000)
        pt.start_new_day()
        pt.open_position("AAPL", "long", 10, 100.0)
        pt.update_prices({"AAPL": 110.0})

        # Equity = 99000 + 1100 = 100100
        assert pt.daily_pnl == 100.0
        assert pt.daily_pnl_pct == pytest.approx(100 / 100_000)

    def test_win_rate(self):
        pt = PortfolioTracker(100_000)

        # 2 winners, 1 loser
        pt.open_position("A", "long", 10, 100.0)
        pt.close_position("A", 110.0, "take_profit")

        pt.open_position("B", "long", 10, 100.0)
        pt.close_position("B", 90.0, "stop_loss")

        pt.open_position("C", "long", 10, 100.0)
        pt.close_position("C", 105.0, "take_profit")

        assert pt.win_rate == pytest.approx(2 / 3)

    def test_profit_factor(self):
        pt = PortfolioTracker(100_000)

        pt.open_position("A", "long", 10, 100.0)
        pt.close_position("A", 120.0)  # +200

        pt.open_position("B", "long", 10, 100.0)
        pt.close_position("B", 90.0)   # -100

        assert pt.profit_factor == pytest.approx(2.0)

    def test_profit_factor_no_losses(self):
        pt = PortfolioTracker(100_000)
        pt.open_position("A", "long", 10, 100.0)
        pt.close_position("A", 110.0)
        assert pt.profit_factor == float("inf")

    def test_max_drawdown(self):
        pt = PortfolioTracker(100_000)
        # Simulér: equity stiger til 110k, falder til 99k
        pt._equity_history = [100_000, 105_000, 110_000, 102_000, 99_000, 103_000]
        assert pt.max_drawdown_pct == pytest.approx(11_000 / 110_000)

    def test_drawdown_from_initial(self):
        pt = PortfolioTracker(100_000)
        pt.open_position("AAPL", "long", 1000, 100.0)
        # cost = 100k, cash = 0. Equity = 1000 * 90 = 90k
        pt.update_prices({"AAPL": 90.0})
        assert pt.current_drawdown_pct == pytest.approx(0.10)

    def test_sharpe_ratio_flat(self):
        pt = PortfolioTracker(100_000)
        # Ingen ændringer → sharpe = 0
        assert pt.sharpe_ratio == 0.0

    def test_sharpe_ratio_positive(self):
        pt = PortfolioTracker(100_000)
        # Simulér stigende equity
        pt._equity_history = [100_000, 100_100, 100_300, 100_500, 100_800, 101_000]
        assert pt.sharpe_ratio > 0

    def test_position_weight(self):
        pt = PortfolioTracker(100_000)
        pt.open_position("AAPL", "long", 10, 100.0)
        # Market value = 1000, equity = 100000
        assert pt.get_position_weight("AAPL") == pytest.approx(0.01)

    def test_summary(self):
        pt = PortfolioTracker(100_000)
        s = pt.summary()
        assert "total_equity" in s
        assert "sharpe_ratio" in s
        assert s["total_equity"] == 100_000

    def test_total_return(self):
        pt = PortfolioTracker(100_000)
        pt.open_position("AAPL", "long", 10, 100.0)
        pt.close_position("AAPL", 110.0)
        # Profit = 100, return = 0.1%
        assert pt.total_return_pct == pytest.approx(100 / 100_000)

    def test_multiple_positions(self):
        pt = PortfolioTracker(100_000)
        pt.open_position("AAPL", "long", 10, 150.0)
        pt.open_position("MSFT", "long", 20, 300.0)
        assert pt.open_position_count == 2
        assert pt.cash == 100_000 - 1500 - 6000

    def test_start_new_day_resets_daily(self):
        pt = PortfolioTracker(100_000)
        pt.open_position("AAPL", "long", 10, 100.0)
        pt.update_prices({"AAPL": 110.0})
        pt.start_new_day()
        # Ny dag: equity = 99000 + 1100 = 100100
        assert pt.daily_pnl == 0.0


# ════════════════════════════════════════════════════════════
# RiskManager – Pre-trade checks
# ════════════════════════════════════════════════════════════

class TestRiskManagerPreTrade:

    def _make_rm(
        self,
        capital: float = 100_000,
        max_pos_pct: float = 0.02,
        max_daily: float = 0.05,
        max_positions: int = 10,
        max_dd: float = 0.10,
    ) -> tuple[RiskManager, PortfolioTracker]:
        pt = PortfolioTracker(capital)
        pt.start_new_day()
        rm = RiskManager(
            pt,
            max_position_pct=max_pos_pct,
            max_daily_loss_pct=max_daily,
            max_open_positions=max_positions,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            trailing_stop_pct=0.03,
            max_drawdown_pct=max_dd,
        )
        return rm, pt

    def test_approve_normal_order(self):
        rm, pt = self._make_rm()
        decision = rm.check_order("AAPL", "long", 1000, 150.0)
        assert decision.approved
        assert decision.reason == RejectionReason.APPROVED
        assert decision.adjusted_qty == 6  # int(1000/150)
        assert decision.adjusted_usd == 900.0

    def test_reject_when_halted(self):
        rm, pt = self._make_rm()
        rm._trading_halted = True
        rm._halt_reason = "test halt"
        decision = rm.check_order("AAPL", "long", 1000, 150.0)
        assert not decision.approved
        assert decision.reason == RejectionReason.TRADING_HALTED

    def test_max_position_size_caps_order(self):
        rm, pt = self._make_rm(max_pos_pct=0.02)
        # 2% af 100k = 2000. Request 5000 → capped til 2000
        decision = rm.check_order("AAPL", "long", 5000, 100.0)
        assert decision.approved
        assert decision.adjusted_usd <= 2000
        assert decision.adjusted_qty == 20  # int(2000/100)

    def test_max_open_positions_rejects(self):
        rm, pt = self._make_rm(max_positions=2)
        pt.open_position("A", "long", 1, 10.0)
        pt.open_position("B", "long", 1, 10.0)

        decision = rm.check_order("C", "long", 100, 10.0)
        assert not decision.approved
        assert decision.reason == RejectionReason.MAX_OPEN_POSITIONS

    def test_duplicate_position_rejects(self):
        rm, pt = self._make_rm()
        pt.open_position("AAPL", "long", 1, 100.0)

        decision = rm.check_order("AAPL", "long", 500, 105.0)
        assert not decision.approved
        assert decision.reason == RejectionReason.DUPLICATE_POSITION

    def test_daily_loss_limit_halts(self):
        rm, pt = self._make_rm(capital=100_000, max_daily=0.05)
        # Simulér 5% tab: åbn stor position, pris falder
        pt.open_position("AAPL", "long", 500, 100.0)
        pt.update_prices({"AAPL": 90.0})
        # Equity = 50000 + 45000 = 95000 → daily pnl = -5000 = -5%

        decision = rm.check_order("MSFT", "long", 1000, 50.0)
        assert not decision.approved
        assert decision.reason == RejectionReason.DAILY_LOSS_LIMIT
        assert rm.is_trading_halted

    def test_max_drawdown_halts(self):
        rm, pt = self._make_rm(capital=100_000, max_dd=0.10)
        # Simulér 10% drawdown
        pt.open_position("AAPL", "long", 1000, 100.0)
        # cash = 0, positions = 1000 * price
        pt.update_prices({"AAPL": 90.0})
        # Equity = 0 + 90000 = 90000, peak = 100000 → dd = 10%

        decision = rm.check_order("MSFT", "long", 1000, 50.0)
        assert not decision.approved
        assert decision.reason == RejectionReason.MAX_DRAWDOWN
        assert rm.is_trading_halted

    def test_insufficient_cash_adjusts(self):
        rm, pt = self._make_rm(capital=1000, max_pos_pct=1.0)
        # Request mere end tilgængeligt
        pt.open_position("A", "long", 5, 100.0)  # cash = 500

        decision = rm.check_order("B", "long", 1000, 100.0)
        assert decision.approved
        assert decision.adjusted_usd <= 500
        assert decision.adjusted_qty == 5

    def test_insufficient_cash_rejects_when_zero(self):
        rm, pt = self._make_rm(capital=1000, max_pos_pct=1.0)
        pt.open_position("A", "long", 10, 100.0)  # cash = 0

        decision = rm.check_order("B", "long", 100, 50.0)
        assert not decision.approved
        assert decision.reason == RejectionReason.INSUFFICIENT_CASH

    def test_price_too_high_for_budget(self):
        rm, pt = self._make_rm(capital=100_000, max_pos_pct=0.001)
        # 0.1% af 100k = $100. Price = $500 → qty=0
        decision = rm.check_order("BRK.A", "long", 100, 500.0)
        assert not decision.approved
        assert decision.reason == RejectionReason.MAX_POSITION_SIZE

    def test_resume_trading(self):
        rm, pt = self._make_rm()
        rm._trading_halted = True
        rm._halt_reason = "test"
        rm.resume_trading()
        assert not rm.is_trading_halted

        decision = rm.check_order("AAPL", "long", 1000, 100.0)
        assert decision.approved

    def test_summary(self):
        rm, pt = self._make_rm()
        s = rm.summary()
        assert "trading_halted" in s
        assert "drawdown_pct" in s
        assert s["max_positions"] == 10


# ════════════════════════════════════════════════════════════
# RiskManager – Post-trade monitoring
# ════════════════════════════════════════════════════════════

class TestRiskManagerPostTrade:

    def _make_rm(self) -> tuple[RiskManager, PortfolioTracker]:
        pt = PortfolioTracker(100_000)
        rm = RiskManager(
            pt,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            trailing_stop_pct=0.03,
            max_position_pct=0.10,
            max_daily_loss_pct=0.05,
            max_open_positions=10,
            max_drawdown_pct=0.10,
        )
        return rm, pt

    def test_stop_loss_triggered(self):
        rm, pt = self._make_rm()
        pt.open_position("AAPL", "long", 10, 100.0)
        # Pris falder 2%
        pt.update_prices({"AAPL": 97.0})

        exits = rm.check_positions({"AAPL": 97.0})
        assert len(exits) == 1
        assert exits[0].symbol == "AAPL"
        assert exits[0].reason == "stop_loss"

    def test_stop_loss_not_triggered(self):
        rm, pt = self._make_rm()
        pt.open_position("AAPL", "long", 10, 100.0)
        pt.update_prices({"AAPL": 99.0})  # kun 1% fald

        exits = rm.check_positions({"AAPL": 99.0})
        assert len(exits) == 0

    def test_take_profit_triggered(self):
        rm, pt = self._make_rm()
        pt.open_position("AAPL", "long", 10, 100.0)
        pt.update_prices({"AAPL": 106.0})  # +6%

        exits = rm.check_positions({"AAPL": 106.0})
        assert len(exits) == 1
        assert exits[0].reason == "take_profit"

    def test_trailing_stop_triggered(self):
        rm, pt = self._make_rm()
        pt.open_position("AAPL", "long", 10, 100.0)

        # Stiger til 104 (under 5% take-profit), derefter falder
        pt.update_prices({"AAPL": 104.0})
        exits = rm.check_positions({"AAPL": 104.0})
        assert len(exits) == 0  # stadig oppe

        # Falder 3%+ fra peak (104 → 100.8 = 3.08% fald fra peak)
        pt.update_prices({"AAPL": 100.8})
        exits = rm.check_positions({"AAPL": 100.8})
        assert len(exits) == 1
        assert exits[0].reason == "trailing_stop"

    def test_trailing_stop_not_triggered_small_dip(self):
        rm, pt = self._make_rm()
        pt.open_position("AAPL", "long", 10, 100.0)
        pt.update_prices({"AAPL": 104.0})
        pt.update_prices({"AAPL": 102.0})  # 1.9% fra peak

        exits = rm.check_positions({"AAPL": 102.0})
        assert len(exits) == 0

    def test_multiple_exits(self):
        rm, pt = self._make_rm()
        pt.open_position("AAPL", "long", 10, 100.0)
        pt.open_position("MSFT", "long", 10, 200.0)

        # AAPL falder 3% (stop-loss), MSFT stiger 6% (take-profit)
        pt.update_prices({"AAPL": 97.0, "MSFT": 212.0})
        exits = rm.check_positions({"AAPL": 97.0, "MSFT": 212.0})

        symbols = {e.symbol for e in exits}
        reasons = {e.reason for e in exits}
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "stop_loss" in reasons
        assert "take_profit" in reasons

    def test_no_exits_when_no_positions(self):
        rm, pt = self._make_rm()
        exits = rm.check_positions({})
        assert exits == []


# ════════════════════════════════════════════════════════════
# Integration tests
# ════════════════════════════════════════════════════════════

class TestRiskIntegration:

    def test_full_trade_lifecycle(self):
        """Åbn → overvåg → stop-loss → luk."""
        pt = PortfolioTracker(100_000)
        pt.start_new_day()
        rm = RiskManager(
            pt,
            max_position_pct=0.05,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
            trailing_stop_pct=0.03,
            max_daily_loss_pct=0.05,
            max_open_positions=10,
            max_drawdown_pct=0.10,
        )

        # 1. Check og åbn position
        decision = rm.check_order("AAPL", "long", 5000, 150.0)
        assert decision.approved

        pt.open_position("AAPL", "long", decision.adjusted_qty, 150.0)

        # 2. Pris stiger
        pt.update_prices({"AAPL": 155.0})
        exits = rm.check_positions({"AAPL": 155.0})
        assert len(exits) == 0

        # 3. Pris falder under stop-loss
        pt.update_prices({"AAPL": 146.0})
        exits = rm.check_positions({"AAPL": 146.0})
        assert len(exits) == 1
        assert exits[0].reason == "stop_loss"

        # 4. Luk position
        trade = pt.close_position("AAPL", 146.0, "stop_loss")
        assert trade.realized_pnl < 0

    def test_daily_limit_prevents_new_trades(self):
        """Når dagligt tab overskrides, kan ingen nye trades åbnes."""
        pt = PortfolioTracker(100_000)
        pt.start_new_day()
        rm = RiskManager(
            pt,
            max_position_pct=1.0,
            max_daily_loss_pct=0.03,
            max_open_positions=10,
            max_drawdown_pct=0.10,
        )

        # Tab 3%+
        pt.open_position("AAPL", "long", 500, 100.0)
        pt.update_prices({"AAPL": 94.0})
        # equity = 50000 + 47000 = 97000, daily_pnl = -3000 = -3%

        decision = rm.check_order("MSFT", "long", 1000, 50.0)
        assert not decision.approved
        assert rm.is_trading_halted

        # Ny dag → resume
        pt.start_new_day()
        rm.resume_trading()
        decision = rm.check_order("MSFT", "long", 1000, 50.0)
        assert decision.approved

    def test_drawdown_circuit_breaker(self):
        """Max drawdown stopper ALT handel."""
        pt = PortfolioTracker(100_000)
        pt.start_new_day()
        rm = RiskManager(
            pt,
            max_position_pct=1.0,
            max_daily_loss_pct=0.50,
            max_open_positions=10,
            max_drawdown_pct=0.10,
        )

        pt.open_position("AAPL", "long", 1000, 100.0)
        pt.update_prices({"AAPL": 90.0})
        # Equity = 0 + 90000, peak = 100000, dd = 10%

        decision = rm.check_order("MSFT", "long", 100, 50.0)
        assert not decision.approved
        assert decision.reason == RejectionReason.MAX_DRAWDOWN

    def test_position_size_scales_with_equity(self):
        """Max position pct beregnes fra aktuel equity, ikke initial capital."""
        pt = PortfolioTracker(100_000)
        pt.start_new_day()
        rm = RiskManager(pt, max_position_pct=0.02, max_drawdown_pct=0.50,
                         max_daily_loss_pct=0.50, max_open_positions=20)

        # Tjek: 2% af 100k = $2000
        d1 = rm.check_order("A", "long", 5000, 100.0)
        assert d1.adjusted_usd <= 2000

        # Tilføj profit
        pt.open_position("B", "long", 10, 100.0)
        pt.update_prices({"B": 1000.0})
        # Equity = 99000 + 10000 = 109000, 2% = $2180

        d2 = rm.check_order("C", "long", 5000, 100.0)
        assert d2.adjusted_usd <= 109_000 * 0.02 + 1

    def test_conservative_defaults_from_config(self):
        """Standardværdier bør være konservative."""
        pt = PortfolioTracker(100_000)
        rm = RiskManager(pt)

        # Disse bør matche config defaults (risk_sizing.json: 0.10)
        assert rm.max_position_pct <= 0.15   # maks 15%
        assert rm.max_daily_loss_pct <= 0.10  # maks 10%
        assert rm.stop_loss_pct <= 0.05       # maks 5%
        assert rm.max_drawdown_pct <= 0.15    # maks 15%
