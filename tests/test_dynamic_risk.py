"""
Tests for DynamicRiskManager, CorrelationMonitor og VolatilityScaler.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.risk.portfolio_tracker import PortfolioTracker, Position
from src.risk.dynamic_risk import (
    DynamicRiskManager,
    RiskProfile,
    RISK_PROFILES,
    CircuitBreakerLevel,
    CircuitBreakerState,
    CircuitBreakerConfig,
    RiskTransition,
)
from src.risk.correlation_monitor import (
    CorrelationMonitor,
    CorrelationReport,
    CorrelationWarning,
    ConcentrationWarning,
    DiversificationSuggestion,
)
from src.risk.volatility_scaling import (
    VolatilityScaler,
    PositionSize,
    RiskParityAllocation,
)
from src.strategy.regime import MarketRegime, RegimeResult


# ── Helpers ──────────────────────────────────────────────────

def _make_portfolio(capital: float = 100_000) -> PortfolioTracker:
    return PortfolioTracker(initial_capital=capital)


def _make_regime_result(regime: MarketRegime, confidence: float = 80, score: float = 0.5):
    return RegimeResult(regime=regime, confidence=confidence, composite_score=score)


def _make_df(n=100, trend=0.001, noise=0.015, seed=42, start=100.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n + 2)[-n:]
    returns = trend + rng.normal(0, noise, n)
    prices = start * np.cumprod(1 + returns)
    return pd.DataFrame({
        "Open": prices * 0.99,
        "High": prices * (1 + abs(rng.normal(0, 0.01, n))),
        "Low": prices * (1 - abs(rng.normal(0, 0.01, n))),
        "Close": prices,
        "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=dates)


def _make_correlated_data(n=100, seed=42):
    """Generér 3 hoejt korrelerede + 1 ukorreleret aktie."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n + 2)[-n:]
    common = rng.normal(0.001, 0.015, n)

    return pd.DataFrame({
        "AAPL": 100 * np.cumprod(1 + common + rng.normal(0, 0.003, n)),
        "MSFT": 200 * np.cumprod(1 + common + rng.normal(0, 0.003, n)),
        "GOOGL": 150 * np.cumprod(1 + common + rng.normal(0, 0.003, n)),
        "GLD": 50 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)),  # Ukorreleret
    }, index=dates)


# ══════════════════════════════════════════════════════════════
# RISK PROFILES
# ══════════════════════════════════════════════════════════════

class TestRiskProfiles:
    """Test at alle regimer har profiles."""

    def test_all_regimes_have_profiles(self):
        for regime in MarketRegime:
            assert regime in RISK_PROFILES

    def test_bull_is_most_permissive(self):
        bull = RISK_PROFILES[MarketRegime.BULL]
        crash = RISK_PROFILES[MarketRegime.CRASH]
        assert bull.max_position_pct > crash.max_position_pct
        assert bull.max_open_positions > crash.max_open_positions
        assert bull.max_exposure_pct > crash.max_exposure_pct
        assert bull.cash_minimum_pct < crash.cash_minimum_pct

    def test_crash_is_most_restrictive(self):
        crash = RISK_PROFILES[MarketRegime.CRASH]
        assert crash.max_position_pct == 0.01
        assert crash.max_open_positions == 2
        assert crash.max_exposure_pct == 0.10
        assert crash.cash_minimum_pct == 0.90

    def test_profiles_are_frozen(self):
        profile = RISK_PROFILES[MarketRegime.BULL]
        with pytest.raises(AttributeError):
            profile.max_position_pct = 0.99


# ══════════════════════════════════════════════════════════════
# CIRCUIT BREAKER STATE
# ══════════════════════════════════════════════════════════════

class TestCircuitBreakerState:

    def test_default_inactive(self):
        cb = CircuitBreakerState()
        assert not cb.is_active
        assert not cb.requires_manual_reset

    def test_active_daily(self):
        cb = CircuitBreakerState(
            level=CircuitBreakerLevel.DAILY,
            triggered_at=datetime.now().isoformat(),
            reason="Test",
        )
        assert cb.is_active

    def test_can_auto_resume_expired(self):
        past = (datetime.now() - timedelta(hours=1)).isoformat()
        cb = CircuitBreakerState(
            level=CircuitBreakerLevel.DAILY,
            triggered_at=datetime.now().isoformat(),
            resume_at=past,
        )
        assert cb.can_auto_resume

    def test_cannot_auto_resume_future(self):
        future = (datetime.now() + timedelta(hours=24)).isoformat()
        cb = CircuitBreakerState(
            level=CircuitBreakerLevel.WEEKLY,
            triggered_at=datetime.now().isoformat(),
            resume_at=future,
        )
        assert not cb.can_auto_resume

    def test_critical_requires_manual(self):
        cb = CircuitBreakerState(
            level=CircuitBreakerLevel.CRITICAL,
            requires_manual_reset=True,
        )
        assert not cb.can_auto_resume


# ══════════════════════════════════════════════════════════════
# DYNAMIC RISK MANAGER
# ══════════════════════════════════════════════════════════════

class TestDynamicRiskInit:

    def test_default_regime_sideways(self):
        p = _make_portfolio()
        drm = DynamicRiskManager(p)
        assert drm.current_regime == MarketRegime.SIDEWAYS
        assert drm.max_position_pct == 0.03
        assert drm.max_open_positions == 10

    def test_custom_transition_days(self):
        p = _make_portfolio()
        drm = DynamicRiskManager(p, transition_days=5)
        assert drm._transition_days == 5

    def test_not_transitioning_initially(self):
        p = _make_portfolio()
        drm = DynamicRiskManager(p)
        assert not drm.is_transitioning
        assert drm.transition_progress == 1.0


class TestRegimeUpdate:

    def test_update_starts_transition(self):
        drm = DynamicRiskManager(_make_portfolio())
        drm.update_regime(_make_regime_result(MarketRegime.BULL))
        assert drm.target_regime == MarketRegime.BULL
        assert drm.is_transitioning

    def test_crash_immediate(self):
        drm = DynamicRiskManager(_make_portfolio())
        drm.update_regime(_make_regime_result(MarketRegime.CRASH))
        assert drm.current_regime == MarketRegime.CRASH
        assert not drm.is_transitioning
        assert drm.max_position_pct == 0.01
        assert drm.max_open_positions == 2

    def test_same_regime_no_change(self):
        drm = DynamicRiskManager(_make_portfolio())
        drm.update_regime(_make_regime_result(MarketRegime.SIDEWAYS))
        assert not drm.is_transitioning

    def test_transition_logs(self):
        drm = DynamicRiskManager(_make_portfolio())
        drm.update_regime(_make_regime_result(MarketRegime.CRASH))
        assert len(drm.transitions_log) > 0


class TestGradualTransition:

    def test_3_day_transition(self):
        drm = DynamicRiskManager(_make_portfolio(), transition_days=3)
        drm.update_regime(_make_regime_result(MarketRegime.BULL))

        sideways = RISK_PROFILES[MarketRegime.SIDEWAYS]
        bull = RISK_PROFILES[MarketRegime.BULL]

        # Dag 1: 1/3 af vejen
        drm.advance_transition()
        pos_day1 = drm.max_position_pct
        expected_day1 = sideways.max_position_pct + (bull.max_position_pct - sideways.max_position_pct) / 3
        assert pos_day1 == pytest.approx(expected_day1, abs=0.001)

        # Dag 2: 2/3 af vejen
        drm.advance_transition()
        pos_day2 = drm.max_position_pct
        assert pos_day2 > pos_day1

        # Dag 3: komplet
        drm.advance_transition()
        assert drm.max_position_pct == pytest.approx(bull.max_position_pct, abs=0.001)
        assert not drm.is_transitioning
        assert drm.current_regime == MarketRegime.BULL

    def test_no_advance_when_not_transitioning(self):
        drm = DynamicRiskManager(_make_portfolio())
        old_params = drm.current_parameters.copy()
        drm.advance_transition()
        assert drm.current_parameters == old_params

    def test_transition_progress(self):
        drm = DynamicRiskManager(_make_portfolio(), transition_days=4)
        drm.update_regime(_make_regime_result(MarketRegime.BEAR))
        assert drm.transition_progress == 0.0
        drm.advance_transition()
        assert drm.transition_progress == pytest.approx(0.25)
        drm.advance_transition()
        assert drm.transition_progress == pytest.approx(0.50)


class TestCircuitBreakers:

    def test_no_breach(self):
        drm = DynamicRiskManager(_make_portfolio())
        cb = drm.check_circuit_breakers()
        assert cb.level == CircuitBreakerLevel.NONE
        assert drm.is_trading_allowed

    def test_daily_breach(self):
        p = _make_portfolio(100_000)
        p.start_new_day()
        # Simulér 4% dagligt tab
        p.open_position("AAPL", "long", 100, 100.0)
        p.update_prices({"AAPL": 60.0})  # -40% paa position = stort tab
        drm = DynamicRiskManager(p)
        cb = drm.check_circuit_breakers()
        assert cb.level == CircuitBreakerLevel.DAILY
        assert not drm.is_trading_allowed

    def test_critical_requires_manual_reset(self):
        p = _make_portfolio(100_000)
        # Simulér 20% drawdown over flere dage (saa daily-loss ikke trigger foerst)
        p.open_position("AAPL", "long", 500, 100.0)
        # Dag 1: fald
        p.update_prices({"AAPL": 90.0})
        p.start_new_day()
        # Dag 2: mere fald
        p.update_prices({"AAPL": 80.0})
        p.start_new_day()
        # Dag 3: endnu mere – nu er drawdown > 15%
        p.update_prices({"AAPL": 65.0})
        # Peak equity: 100k, nu: cash(50k) + 500*65(32.5k) = 82.5k → 17.5% drawdown
        drm = DynamicRiskManager(p)
        cb = drm.check_circuit_breakers()
        assert cb.level == CircuitBreakerLevel.CRITICAL
        assert cb.requires_manual_reset
        assert not drm.is_trading_allowed

        # Manuel reset
        drm.manual_reset()
        assert drm.is_trading_allowed

    def test_start_new_day_resets_daily(self):
        p = _make_portfolio(100_000)
        p.start_new_day()
        p.open_position("AAPL", "long", 100, 100.0)
        p.update_prices({"AAPL": 60.0})
        drm = DynamicRiskManager(p)
        drm.check_circuit_breakers()
        assert drm.circuit_breaker.level == CircuitBreakerLevel.DAILY

        drm.start_new_day()
        assert drm.circuit_breaker.level == CircuitBreakerLevel.NONE


class TestExposureCheck:

    def test_no_exposure(self):
        drm = DynamicRiskManager(_make_portfolio())
        result = drm.check_exposure()
        assert result["current_exposure"] == 0
        assert not result["overexposed"]

    def test_overexposed(self):
        p = _make_portfolio(100_000)
        p.open_position("AAPL", "long", 400, 175.0)  # 70k af 100k
        drm = DynamicRiskManager(p)
        # SIDEWAYS: max 60%
        result = drm.check_exposure()
        assert result["overexposed"] is True
        assert "Reducér" in result["action"]


class TestDynamicRiskSummary:

    def test_summary_structure(self):
        drm = DynamicRiskManager(_make_portfolio())
        s = drm.summary()
        assert "current_regime" in s
        assert "circuit_breaker" in s
        assert "exposure" in s
        assert "trading_allowed" in s
        assert "parameters" in s
        assert "recent_transitions" in s


# ══════════════════════════════════════════════════════════════
# CORRELATION MONITOR
# ══════════════════════════════════════════════════════════════

class TestCorrelationMonitor:

    def test_too_few_positions(self):
        mon = CorrelationMonitor()
        report = mon.analyze(pd.DataFrame(), {"AAPL": 0.5})
        assert report.avg_correlation == 0.0

    def test_correlated_pairs_detected(self):
        mon = CorrelationMonitor(correlation_threshold=0.70)
        data = _make_correlated_data(100)
        positions = {"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.3, "GLD": 0.1}
        report = mon.analyze(data, positions)

        # AAPL, MSFT, GOOGL should be highly correlated
        assert len(report.highly_correlated_pairs) > 0
        assert report.max_correlation > 0.7

    def test_concentration_warning(self):
        mon = CorrelationMonitor(concentration_threshold=0.40)
        data = _make_correlated_data(100)
        positions = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "GLD": 0.25}
        report = mon.analyze(data, positions)

        # 3 tech aktier = 75% i Technology
        assert len(report.concentration_warnings) > 0
        tech_warning = next(
            (w for w in report.concentration_warnings if w.sector == "Technology"),
            None,
        )
        assert tech_warning is not None
        assert tech_warning.total_weight_pct >= 0.75

    def test_healthy_portfolio(self):
        mon = CorrelationMonitor(
            correlation_threshold=0.99,
            concentration_threshold=0.99,
        )
        data = _make_correlated_data(100)
        positions = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "GLD": 0.25}
        report = mon.analyze(data, positions)
        assert report.is_healthy
        assert report.risk_level == "low"

    def test_diversification_suggestions(self):
        mon = CorrelationMonitor(
            correlation_threshold=0.70,
            concentration_threshold=0.40,
        )
        data = _make_correlated_data(100)
        positions = {"AAPL": 0.3, "MSFT": 0.3, "GOOGL": 0.3, "GLD": 0.1}
        report = mon.analyze(data, positions)
        assert len(report.diversification_suggestions) > 0

    def test_correlation_matrix_returned(self):
        mon = CorrelationMonitor()
        data = _make_correlated_data(100)
        positions = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.25, "GLD": 0.25}
        report = mon.analyze(data, positions)
        assert report.correlation_matrix is not None
        assert report.correlation_matrix.shape == (4, 4)

    def test_risk_level(self):
        report = CorrelationReport(
            portfolio_beta=1.2, avg_correlation=0.8, max_correlation=0.95,
            highly_correlated_pairs=[
                CorrelationWarning("A", "B", 0.9, ""),
                CorrelationWarning("A", "C", 0.85, ""),
                CorrelationWarning("B", "C", 0.88, ""),
            ],
        )
        assert report.risk_level == "high"


class TestPortfolioBeta:

    def test_beta_with_market(self):
        mon = CorrelationMonitor(min_data_points=10)
        rng = np.random.default_rng(42)
        n = 60
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n + 2)[-n:]
        market_ret = rng.normal(0.001, 0.01, n)
        market = pd.DataFrame(
            {"Close": 100 * np.cumprod(1 + market_ret)},
            index=dates,
        )
        # 2 stocks for analyze() at acceptere (kraever >= 2)
        stock_a_ret = 1.5 * market_ret + rng.normal(0, 0.002, n)
        stock_b_ret = 0.8 * market_ret + rng.normal(0, 0.003, n)
        data = pd.DataFrame({
            "AAPL": 150 * np.cumprod(1 + stock_a_ret),
            "GLD": 50 * np.cumprod(1 + stock_b_ret),
        }, index=dates)
        positions = {"AAPL": 0.7, "GLD": 0.3}
        report = mon.analyze(data, positions, market_data=market)
        # Weighted beta: 0.7*1.5 + 0.3*0.8 ≈ 1.29
        assert 0.3 < report.portfolio_beta < 3.0

    def test_beta_without_market(self):
        mon = CorrelationMonitor()
        report = mon.analyze(pd.DataFrame(), {"AAPL": 1.0}, market_data=None)
        assert report.portfolio_beta == 0.0


class TestSectorMapping:

    def test_known_sectors(self):
        mon = CorrelationMonitor()
        assert mon.get_sector("AAPL") == "Technology"
        assert mon.get_sector("JPM") == "Financials"
        assert mon.get_sector("GLD") == "Safe Haven"

    def test_unknown_sector(self):
        mon = CorrelationMonitor()
        assert mon.get_sector("XYZZY") == "Unknown"

    def test_custom_mapping(self):
        mon = CorrelationMonitor()
        mon.add_sector_mapping("NOVO", "Healthcare")
        assert mon.get_sector("NOVO") == "Healthcare"


# ══════════════════════════════════════════════════════════════
# VOLATILITY SCALER
# ══════════════════════════════════════════════════════════════

class TestATR:

    def test_atr_positive(self):
        scaler = VolatilityScaler()
        df = _make_df(100)
        atr = scaler.calculate_atr(df)
        assert atr > 0

    def test_atr_empty_data(self):
        scaler = VolatilityScaler()
        assert scaler.calculate_atr(pd.DataFrame()) == 0.0
        assert scaler.calculate_atr(None) == 0.0

    def test_atr_short_data(self):
        scaler = VolatilityScaler()
        df = _make_df(3)
        atr = scaler.calculate_atr(df)
        assert atr >= 0


class TestPositionSizing:

    def test_basic_sizing(self):
        scaler = VolatilityScaler(equity=100_000, risk_per_trade_pct=0.01)
        df = _make_df(100)
        price = float(df["Close"].iloc[-1])
        size = scaler.calculate_position_size(df, "AAPL", price)
        assert size.shares > 0
        assert size.dollar_amount > 0
        assert size.weight_pct > 0
        assert size.atr > 0
        assert size.method == "atr"

    def test_high_vol_smaller_position(self):
        scaler = VolatilityScaler(equity=100_000)
        low_vol = _make_df(100, noise=0.005, seed=1)
        high_vol = _make_df(100, noise=0.04, seed=2)

        price_low = float(low_vol["Close"].iloc[-1])
        price_high = float(high_vol["Close"].iloc[-1])

        size_low = scaler.calculate_position_size(low_vol, "AAPL", price_low)
        size_high = scaler.calculate_position_size(high_vol, "TSLA", price_high)

        # Hoejere vol bør give faerre shares (relativt)
        assert size_low.risk_per_share < size_high.risk_per_share

    def test_zero_price(self):
        scaler = VolatilityScaler()
        df = _make_df(100)
        size = scaler.calculate_position_size(df, "AAPL", 0.0)
        assert size.shares == 0

    def test_max_position_cap(self):
        scaler = VolatilityScaler(equity=100_000, max_position_pct=0.05)
        df = _make_df(100, noise=0.001)  # Meget lav vol -> ville give stor position
        price = float(df["Close"].iloc[-1])
        size = scaler.calculate_position_size(df, "AAPL", price)
        assert size.dollar_amount <= 100_000 * 0.10 + price  # Rimelig cap


class TestVolatilityAdjustedWeight:

    def test_low_vol_higher_weight(self):
        scaler = VolatilityScaler(target_volatility=0.15)
        low_vol = _make_df(200, noise=0.005)
        high_vol = _make_df(200, noise=0.03)
        w_low = scaler.volatility_adjusted_weight(low_vol)
        w_high = scaler.volatility_adjusted_weight(high_vol)
        assert w_low > w_high

    def test_empty_data(self):
        scaler = VolatilityScaler()
        assert scaler.volatility_adjusted_weight(pd.DataFrame()) == 0.0

    def test_short_data(self):
        scaler = VolatilityScaler()
        assert scaler.volatility_adjusted_weight(_make_df(5)) == 0.0


class TestRiskParity:

    def test_risk_parity_allocation(self):
        scaler = VolatilityScaler()
        data = {
            "AAPL": _make_df(200, noise=0.015, seed=1),
            "GLD": _make_df(200, noise=0.008, seed=2),
            "TSLA": _make_df(200, noise=0.035, seed=3),
        }
        alloc = scaler.risk_parity(data)
        assert len(alloc.allocations) == 3
        assert sum(alloc.allocations.values()) == pytest.approx(1.0, abs=0.01)

        # GLD (lavest vol) bør faa hoejest vaegt
        assert alloc.allocations["GLD"] > alloc.allocations["TSLA"]

    def test_risk_parity_empty(self):
        scaler = VolatilityScaler()
        alloc = scaler.risk_parity({})
        assert len(alloc.allocations) == 0

    def test_risk_contributions_present(self):
        scaler = VolatilityScaler()
        data = {
            "A": _make_df(200, noise=0.01, seed=1),
            "B": _make_df(200, noise=0.02, seed=2),
        }
        alloc = scaler.risk_parity(data)
        assert len(alloc.risk_contributions) == 2


class TestVolTargetLeverage:

    def test_low_vol_leverage_up(self):
        scaler = VolatilityScaler(target_volatility=0.15)
        df = _make_df(200, noise=0.005)  # ~8% annualiseret
        lev = scaler.vol_target_leverage(df)
        assert lev > 1.0  # Bør leverage op

    def test_high_vol_leverage_down(self):
        scaler = VolatilityScaler(target_volatility=0.15)
        df = _make_df(200, noise=0.04)  # ~64% annualiseret
        lev = scaler.vol_target_leverage(df)
        assert lev < 1.0  # Bør leverage ned

    def test_empty_data(self):
        scaler = VolatilityScaler()
        assert scaler.vol_target_leverage(pd.DataFrame()) == 1.0


class TestBatchSizing:

    def test_size_all(self):
        scaler = VolatilityScaler(equity=100_000)
        data = {
            "AAPL": _make_df(100, seed=1),
            "MSFT": _make_df(100, seed=2),
        }
        prices = {
            "AAPL": float(data["AAPL"]["Close"].iloc[-1]),
            "MSFT": float(data["MSFT"]["Close"].iloc[-1]),
        }
        sizes = scaler.size_all_positions(data, prices)
        assert "AAPL" in sizes
        assert "MSFT" in sizes
        assert sizes["AAPL"].shares > 0


class TestEquityProperty:

    def test_set_equity(self):
        scaler = VolatilityScaler(equity=50_000)
        assert scaler.equity == 50_000
        scaler.equity = 75_000
        assert scaler.equity == 75_000

    def test_negative_equity_capped(self):
        scaler = VolatilityScaler()
        scaler.equity = -1000
        assert scaler.equity == 0


# ══════════════════════════════════════════════════════════════
# IMPORT TESTS
# ══════════════════════════════════════════════════════════════

class TestImports:

    def test_dynamic_risk_imports(self):
        from src.risk import (
            DynamicRiskManager, RiskProfile, RISK_PROFILES,
            CircuitBreakerLevel, CircuitBreakerState, CircuitBreakerConfig,
        )
        assert DynamicRiskManager is not None

    def test_correlation_imports(self):
        from src.risk import (
            CorrelationMonitor, CorrelationReport,
            CorrelationWarning, ConcentrationWarning,
        )
        assert CorrelationMonitor is not None

    def test_volatility_imports(self):
        from src.risk import (
            VolatilityScaler, PositionSize, RiskParityAllocation,
        )
        assert VolatilityScaler is not None
