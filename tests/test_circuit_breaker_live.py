"""
Phase A3: Stresstest af circuit breakers.

Bekræfter at:
  - Drawdown ≥ 10% udløser MAX_DRAWDOWN halt
  - Daily loss ≥ 5% udløser DAILY_LOSS_LIMIT halt
  - Halt blokerer NYE ordrer men tillader EXIT
  - resume_trading() kan genåbne handel
  - Circuit-breaker-reset task fra daily_scheduler kalder resume_trading()

Audit-fund 10. maj 2026: PWA viste drawdown 30% uden halt. Disse tests
verificerer at logikken VIRKER på korrekte data — selvom fejlen var at
data ikke nåede risk_manager (separat fix i A1 reconciliation).
"""
from __future__ import annotations

import os

import pytest


# Sæt env vars FØR import af moduler der læser dem
os.environ.setdefault("APP_USERNAME", "testuser")
os.environ.setdefault("APP_PASSWORD", "testpass123")
os.environ.setdefault("APP_SECRET_KEY", "x" * 64)


@pytest.fixture
def portfolio_at_drawdown():
    """PortfolioTracker hvor current_drawdown_pct = 11% (over 10% limit)."""
    from src.risk.portfolio_tracker import PortfolioTracker
    pt = PortfolioTracker(initial_capital=100_000)
    pt._peak_equity = 100_000
    # Set cash så total_equity = 89_000 → drawdown 11%
    pt.cash = 89_000
    return pt


@pytest.fixture
def portfolio_at_daily_loss():
    """PortfolioTracker hvor daily_pnl_pct = -6% (over 5% limit)."""
    from src.risk.portfolio_tracker import PortfolioTracker
    pt = PortfolioTracker(initial_capital=100_000)
    pt._peak_equity = 100_000
    pt._daily_start_equity = 100_000
    # Set cash så daily_pnl = -6_000 → -6%
    pt.cash = 94_000
    return pt


@pytest.fixture
def portfolio_healthy():
    """PortfolioTracker uden problemer."""
    from src.risk.portfolio_tracker import PortfolioTracker
    pt = PortfolioTracker(initial_capital=100_000)
    pt._peak_equity = 100_000
    pt._daily_start_equity = 100_000
    pt.cash = 100_000
    return pt


# ── Drawdown circuit breaker ───────────────────────────────────


class TestDrawdownCircuitBreaker:
    def test_drawdown_above_10pct_blocks_new_orders(self, portfolio_at_drawdown):
        from src.risk.risk_manager import RiskManager, RejectionReason
        rm = RiskManager(
            portfolio=portfolio_at_drawdown,
            max_drawdown_pct=0.10,
        )
        decision = rm.check_order(
            symbol="AAPL", side="long", requested_usd=1000.0, price=200.0,
        )
        assert decision.approved is False
        assert decision.reason == RejectionReason.MAX_DRAWDOWN
        assert "drawdown" in decision.message.lower()

    def test_drawdown_halt_persists_after_first_block(self, portfolio_at_drawdown):
        """Efter første drawdown-block skal trading_halted=True."""
        from src.risk.risk_manager import RiskManager
        rm = RiskManager(portfolio=portfolio_at_drawdown, max_drawdown_pct=0.10)
        rm.check_order(symbol="AAPL", side="long", requested_usd=1000.0, price=200.0)
        assert rm.is_trading_halted is True
        assert "drawdown" in rm._halt_reason.lower()

    def test_drawdown_block_allows_exit(self, portfolio_at_drawdown):
        """Halt skal IKKE blokere lukning af eksisterende positioner."""
        from src.risk.risk_manager import RiskManager
        portfolio_at_drawdown.open_position(
            symbol="MSFT", side="long", qty=10, price=300.0,
        )
        rm = RiskManager(portfolio=portfolio_at_drawdown, max_drawdown_pct=0.10)
        # Trigger halt
        rm.check_order(symbol="AAPL", side="long", requested_usd=1000.0, price=200.0)
        assert rm.is_trading_halted is True
        # Forsøg at lukke MSFT (long → "short" = exit)
        decision = rm.check_order(
            symbol="MSFT", side="short", requested_usd=3000.0, price=300.0,
        )
        assert decision.approved is True, "Exit skulle godkendes selv under halt"


# ── Daily-loss circuit breaker ─────────────────────────────────


class TestDailyLossCircuitBreaker:
    def test_daily_loss_above_5pct_blocks_new_orders(self, portfolio_at_daily_loss):
        from src.risk.risk_manager import RiskManager, RejectionReason
        rm = RiskManager(
            portfolio=portfolio_at_daily_loss,
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.20,  # høj så det IKKE er drawdown der trigger
        )
        decision = rm.check_order(
            symbol="AAPL", side="long", requested_usd=1000.0, price=200.0,
        )
        assert decision.approved is False
        assert decision.reason == RejectionReason.DAILY_LOSS_LIMIT

    def test_daily_loss_halt_persists(self, portfolio_at_daily_loss):
        from src.risk.risk_manager import RiskManager
        rm = RiskManager(
            portfolio=portfolio_at_daily_loss,
            max_daily_loss_pct=0.05,
            max_drawdown_pct=0.20,
        )
        rm.check_order(symbol="AAPL", side="long", requested_usd=1000.0, price=200.0)
        assert rm.is_trading_halted is True


# ── Resume_trading ─────────────────────────────────────────────


class TestResumeTrading:
    def test_resume_clears_halt_state(self, portfolio_healthy):
        from src.risk.risk_manager import RiskManager
        rm = RiskManager(portfolio=portfolio_healthy)
        rm._halt_trading("test halt")
        assert rm.is_trading_halted is True
        rm.resume_trading()
        assert rm.is_trading_halted is False
        assert rm._halt_reason == ""

    def test_orders_pass_after_resume(self, portfolio_healthy):
        from src.risk.risk_manager import RiskManager
        rm = RiskManager(portfolio=portfolio_healthy)
        rm._halt_trading("manual halt")
        rm.resume_trading()
        decision = rm.check_order(
            symbol="AAPL", side="long", requested_usd=1000.0, price=200.0,
        )
        assert decision.approved is True


# ── Healthy portfolio passes ───────────────────────────────────


class TestHealthyPortfolio:
    def test_no_halt_when_drawdown_under_limit(self, portfolio_healthy):
        from src.risk.risk_manager import RiskManager
        rm = RiskManager(portfolio=portfolio_healthy, max_drawdown_pct=0.10)
        decision = rm.check_order(
            symbol="AAPL", side="long", requested_usd=1000.0, price=200.0,
        )
        assert decision.approved is True
        assert rm.is_trading_halted is False


# ── Daily scheduler reset ──────────────────────────────────────


class TestCircuitBreakerReset:
    def test_reset_function_exists_in_scheduler(self):
        """Verificer at _circuit_breaker_reset function eksisterer."""
        from src.ops.daily_scheduler import _circuit_breaker_reset
        assert callable(_circuit_breaker_reset)

    def test_reset_task_in_default_tasks(self):
        """Verificer circuit_breaker_reset er registreret i DEFAULT_TASKS."""
        from src.ops.daily_scheduler import DailyScheduler
        sched = DailyScheduler()
        names = [t.name for t in sched._tasks]
        assert "circuit_breaker_reset" in names, \
            f"circuit_breaker_reset ikke i {names}"

    def test_reset_calls_resume_trading(self, monkeypatch):
        """_circuit_breaker_reset skal kalde resume_trading() på AutoTrader's RiskManager."""
        from src.ops.daily_scheduler import _circuit_breaker_reset
        from src.broker import registry

        # Mock auto-trader med en haltet risk-manager
        class MockRM:
            def __init__(self):
                self._trading_halted = True
                self.resumed = False

            @property
            def is_trading_halted(self):
                return self._trading_halted

            def resume_trading(self):
                self._trading_halted = False
                self.resumed = True

        class MockTrader:
            _risk_manager = MockRM()

        trader = MockTrader()
        monkeypatch.setattr(registry, "_auto_trader", trader)

        result = _circuit_breaker_reset()
        assert result.get("reset") is True
        assert trader._risk_manager.resumed is True
        assert trader._risk_manager.is_trading_halted is False


# ── Integration: drawdown beregning på reel portfolio ──────────


class TestDrawdownCalculation:
    def test_drawdown_correctly_reflects_equity_drop(self):
        """Sanity: når equity falder fra peak, opdateres current_drawdown_pct."""
        from src.risk.portfolio_tracker import PortfolioTracker
        pt = PortfolioTracker(initial_capital=100_000)
        # Simulér peak ved $110k
        pt._peak_equity = 110_000
        pt.cash = 110_000
        assert pt.current_drawdown_pct == 0.0

        # Tab til $99k → drawdown = 10/110 = 9.1%
        pt.cash = 99_000
        assert abs(pt.current_drawdown_pct - 11_000 / 110_000) < 0.001

    def test_daily_pnl_correctly_reflects_intraday_drop(self):
        from src.risk.portfolio_tracker import PortfolioTracker
        pt = PortfolioTracker(initial_capital=100_000)
        pt._daily_start_equity = 100_000
        pt.cash = 100_000
        assert pt.daily_pnl_pct == 0.0

        # Tab på $4k → daily P&L = -4%
        pt.cash = 96_000
        assert abs(pt.daily_pnl_pct - (-0.04)) < 0.001
