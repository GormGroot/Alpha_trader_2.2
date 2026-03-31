"""
DynamicRiskManager – regime-adaptiv risikostyring med circuit breakers.

Justerer automatisk alle risiko-parametre baseret på markedsregime:

    Parameter          | Bull  | Sideways | Bear  | Crash
    -------------------|-------|----------|-------|------
    Max position size  | 5%    | 3%       | 2%    | 1%
    Max dagligt tab    | 3%    | 2%       | 1%    | 0.5%
    Max aabne posit.   | 15    | 10       | 5     | 2
    Stop-loss default  | 8%    | 5%       | 3%    | 2%
    Max eksponering    | 95%   | 60%      | 30%   | 10%
    Cash minimum       | 5%    | 40%      | 70%   | 90%

Circuit breakers:
  - 3% tab paa 1 dag  -> stop nye handler resten af dagen
  - 7% tab paa 1 uge  -> stop alt i 48 timer + alert
  - 15% fra peak      -> stop ALT, kraev manuel genstart

Overgange sker gradvist over 3-5 dage for at undgaa whipsaw.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from loguru import logger

from src.risk.portfolio_tracker import PortfolioTracker
from src.strategy.regime import MarketRegime, RegimeResult


# ── Risk Profile per Regime ──────────────────────────────────

@dataclass(frozen=True)
class RiskProfile:
    """Risiko-parametre for ét regime."""
    regime: MarketRegime
    max_position_pct: float
    max_daily_loss_pct: float
    max_open_positions: int
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    max_exposure_pct: float
    cash_minimum_pct: float


# Standard risiko-profiler
RISK_PROFILES: dict[MarketRegime, RiskProfile] = {
    MarketRegime.BULL: RiskProfile(
        regime=MarketRegime.BULL,
        max_position_pct=0.05,
        max_daily_loss_pct=0.03,
        max_open_positions=15,
        stop_loss_pct=0.08,
        take_profit_pct=0.15,
        trailing_stop_pct=0.06,
        max_exposure_pct=0.95,
        cash_minimum_pct=0.05,
    ),
    MarketRegime.SIDEWAYS: RiskProfile(
        regime=MarketRegime.SIDEWAYS,
        max_position_pct=0.03,
        max_daily_loss_pct=0.02,
        max_open_positions=10,
        stop_loss_pct=0.05,
        take_profit_pct=0.08,
        trailing_stop_pct=0.04,
        max_exposure_pct=0.60,
        cash_minimum_pct=0.40,
    ),
    MarketRegime.BEAR: RiskProfile(
        regime=MarketRegime.BEAR,
        max_position_pct=0.02,
        max_daily_loss_pct=0.01,
        max_open_positions=5,
        stop_loss_pct=0.03,
        take_profit_pct=0.05,
        trailing_stop_pct=0.03,
        max_exposure_pct=0.30,
        cash_minimum_pct=0.70,
    ),
    MarketRegime.CRASH: RiskProfile(
        regime=MarketRegime.CRASH,
        max_position_pct=0.01,
        max_daily_loss_pct=0.005,
        max_open_positions=2,
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        trailing_stop_pct=0.02,
        max_exposure_pct=0.10,
        cash_minimum_pct=0.90,
    ),
    MarketRegime.RECOVERY: RiskProfile(
        regime=MarketRegime.RECOVERY,
        max_position_pct=0.03,
        max_daily_loss_pct=0.02,
        max_open_positions=8,
        stop_loss_pct=0.04,
        take_profit_pct=0.10,
        trailing_stop_pct=0.04,
        max_exposure_pct=0.50,
        cash_minimum_pct=0.50,
    ),
    MarketRegime.EUPHORIA: RiskProfile(
        regime=MarketRegime.EUPHORIA,
        max_position_pct=0.04,
        max_daily_loss_pct=0.02,
        max_open_positions=12,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        trailing_stop_pct=0.04,
        max_exposure_pct=0.70,
        cash_minimum_pct=0.30,
    ),
}


# ── Circuit Breaker ──────────────────────────────────────────

class CircuitBreakerLevel(Enum):
    NONE = "none"
    DAILY = "daily"        # 3% tab paa 1 dag
    WEEKLY = "weekly"      # 7% tab paa 1 uge
    CRITICAL = "critical"  # 15% fra peak – kraev manuel genstart


@dataclass
class CircuitBreakerState:
    """Status for circuit breaker."""
    level: CircuitBreakerLevel = CircuitBreakerLevel.NONE
    triggered_at: str = ""
    reason: str = ""
    resume_at: str = ""             # Hvornaar kan handel genoptages
    requires_manual_reset: bool = False

    @property
    def is_active(self) -> bool:
        return self.level != CircuitBreakerLevel.NONE

    @property
    def can_auto_resume(self) -> bool:
        """Kan handelen genoptages automatisk?"""
        if self.requires_manual_reset:
            return False
        if not self.resume_at:
            return False
        try:
            resume_time = datetime.fromisoformat(self.resume_at)
            return datetime.now() >= resume_time
        except (ValueError, TypeError):
            return False


@dataclass
class CircuitBreakerConfig:
    """Konfigurerbare circuit breaker-niveauer."""
    daily_loss_pct: float = 0.03      # 3% tab paa 1 dag
    weekly_loss_pct: float = 0.07     # 7% tab paa 1 uge
    peak_drawdown_pct: float = 0.15   # 15% fra peak
    daily_cooldown_hours: int = 0     # Stop resten af dagen (0 = til ny dag)
    weekly_cooldown_hours: int = 48   # 48 timer stop


@dataclass
class RiskTransition:
    """Log-entry for en risiko-parameter ændring."""
    timestamp: str
    parameter: str
    old_value: float
    new_value: float
    from_regime: str
    to_regime: str
    transition_day: int     # Dag i overgangsperioden (1-N)
    total_days: int         # Total antal overgangsdage


# ── DynamicRiskManager ───────────────────────────────────────

class DynamicRiskManager:
    """
    Regime-adaptiv risikostyring med graduelle overgange og circuit breakers.

    Alle risiko-parametre justeres automatisk baseret paa markedsregime.
    Overgange sker gradvist over transition_days for at undgaa whipsaw.

    Brug:
        drm = DynamicRiskManager(portfolio)
        drm.update_regime(regime_result)
        params = drm.current_parameters
        cb_status = drm.check_circuit_breakers()
    """

    def __init__(
        self,
        portfolio: PortfolioTracker,
        profiles: dict[MarketRegime, RiskProfile] | None = None,
        cb_config: CircuitBreakerConfig | None = None,
        transition_days: int = 3,
    ) -> None:
        """
        Args:
            portfolio: PortfolioTracker.
            profiles: Custom risiko-profiler per regime.
            cb_config: Circuit breaker konfiguration.
            transition_days: Antal dage for gradvis overgang (3-5).
        """
        self._portfolio = portfolio
        self._profiles = profiles or RISK_PROFILES.copy()
        self._cb_config = cb_config or CircuitBreakerConfig()
        self._transition_days = max(1, transition_days)

        # Nuvaerende tilstand
        self._current_regime = MarketRegime.SIDEWAYS
        self._target_regime = MarketRegime.SIDEWAYS
        self._transition_start: datetime | None = None
        self._transition_day = 0

        # Aktuelle parametre (starter med SIDEWAYS)
        default = self._profiles[MarketRegime.SIDEWAYS]
        self._current_params: dict[str, float] = {
            "max_position_pct": default.max_position_pct,
            "max_daily_loss_pct": default.max_daily_loss_pct,
            "max_open_positions": float(default.max_open_positions),
            "stop_loss_pct": default.stop_loss_pct,
            "take_profit_pct": default.take_profit_pct,
            "trailing_stop_pct": default.trailing_stop_pct,
            "max_exposure_pct": default.max_exposure_pct,
            "cash_minimum_pct": default.cash_minimum_pct,
        }

        # Circuit breaker
        self._cb_state = CircuitBreakerState()

        # Weekly tracking
        self._week_start_equity: float = portfolio.total_equity
        self._week_start_time: datetime = datetime.now()

        # Historik
        self._transitions: list[RiskTransition] = []
        self._max_transitions = 1000

    # ── Properties ───────────────────────────────────────────

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    @property
    def target_regime(self) -> MarketRegime:
        return self._target_regime

    @property
    def is_transitioning(self) -> bool:
        return self._current_regime != self._target_regime

    @property
    def transition_progress(self) -> float:
        """Fremskridt i overgang (0.0 til 1.0)."""
        if not self.is_transitioning:
            return 1.0
        return min(1.0, self._transition_day / self._transition_days)

    @property
    def current_parameters(self) -> dict[str, float]:
        return self._current_params.copy()

    @property
    def circuit_breaker(self) -> CircuitBreakerState:
        return self._cb_state

    @property
    def transitions_log(self) -> list[RiskTransition]:
        return self._transitions

    # ── Core: max_position_pct etc. direkte tilgaengelige ────

    @property
    def max_position_pct(self) -> float:
        return self._current_params["max_position_pct"]

    @property
    def max_daily_loss_pct(self) -> float:
        return self._current_params["max_daily_loss_pct"]

    @property
    def max_open_positions(self) -> int:
        return int(self._current_params["max_open_positions"])

    @property
    def stop_loss_pct(self) -> float:
        return self._current_params["stop_loss_pct"]

    @property
    def take_profit_pct(self) -> float:
        return self._current_params["take_profit_pct"]

    @property
    def trailing_stop_pct(self) -> float:
        return self._current_params["trailing_stop_pct"]

    @property
    def max_exposure_pct(self) -> float:
        return self._current_params["max_exposure_pct"]

    @property
    def cash_minimum_pct(self) -> float:
        return self._current_params["cash_minimum_pct"]

    # ── Regime Update ────────────────────────────────────────

    def update_regime(self, regime_result: RegimeResult) -> None:
        """
        Opdatér regime og start gradvis overgang.

        CRASH er en undtagelse: overgaar STRAKS (ingen gradvis).
        """
        new_regime = regime_result.regime

        if new_regime == self._current_regime and not self.is_transitioning:
            return  # Ingen ændring

        # CRASH: oejebliklig overgang!
        if new_regime == MarketRegime.CRASH:
            old = self._current_regime
            self._apply_immediate(new_regime)
            logger.critical(
                f"[risk] CRASH REGIME – oejebliklig overgang fra {old.value}! "
                f"Alle parametre strammet."
            )
            return

        # Sæt ny target og start overgang
        if new_regime != self._target_regime:
            self._target_regime = new_regime
            self._transition_start = datetime.now()
            self._transition_day = 0
            logger.info(
                f"[risk] Regime-overgang startet: {self._current_regime.value} "
                f"-> {new_regime.value} (over {self._transition_days} dage)"
            )

    def advance_transition(self) -> None:
        """
        Ryd 1 dag i overgangen. Kald dette dagligt.

        Interpolerer mellem nuvaerende og target profil.
        """
        if not self.is_transitioning:
            return

        self._transition_day += 1
        progress = min(1.0, self._transition_day / self._transition_days)

        source_profile = self._profiles[self._current_regime]
        target_profile = self._profiles[self._target_regime]

        # Interpolér alle parametre
        for param in self._current_params:
            old_val = getattr(source_profile, param)
            new_val = getattr(target_profile, param)
            interpolated = old_val + (new_val - old_val) * progress

            if self._current_params[param] != interpolated:
                self._transitions.append(RiskTransition(
                    timestamp=datetime.now().isoformat(),
                    parameter=param,
                    old_value=self._current_params[param],
                    new_value=interpolated,
                    from_regime=self._current_regime.value,
                    to_regime=self._target_regime.value,
                    transition_day=self._transition_day,
                    total_days=self._transition_days,
                ))
                self._current_params[param] = interpolated

        if len(self._transitions) > self._max_transitions:
            self._transitions = self._transitions[-self._max_transitions:]

        logger.info(
            f"[risk] Overgang dag {self._transition_day}/{self._transition_days}: "
            f"{self._current_regime.value} -> {self._target_regime.value} "
            f"({progress:.0%} komplet)"
        )

        # Faerdig?
        if progress >= 1.0:
            old = self._current_regime
            self._current_regime = self._target_regime
            self._transition_start = None
            logger.info(
                f"[risk] Overgang faerdig: {old.value} -> {self._current_regime.value}"
            )

    def _apply_immediate(self, regime: MarketRegime) -> None:
        """Anvend en profil oejeblikkeligt (bruges ved CRASH)."""
        profile = self._profiles[regime]
        for param in self._current_params:
            old_val = self._current_params[param]
            new_val = getattr(profile, param)
            if isinstance(new_val, int):
                new_val = float(new_val)
            if old_val != new_val:
                self._transitions.append(RiskTransition(
                    timestamp=datetime.now().isoformat(),
                    parameter=param,
                    old_value=old_val,
                    new_value=new_val,
                    from_regime=self._current_regime.value,
                    to_regime=regime.value,
                    transition_day=0,
                    total_days=0,
                ))
            self._current_params[param] = float(new_val)

        if len(self._transitions) > self._max_transitions:
            self._transitions = self._transitions[-self._max_transitions:]

        self._current_regime = regime
        self._target_regime = regime
        self._transition_start = None
        self._transition_day = 0

    # ── Circuit Breakers ─────────────────────────────────────

    def check_circuit_breakers(self) -> CircuitBreakerState:
        """
        Tjek alle circuit breaker-niveauer.

        Niveauer (hoejeste prioritet foerst):
          1. 15% fra peak -> CRITICAL (stop ALT, manuel genstart)
          2. 7% paa 1 uge -> WEEKLY (stop 48 timer)
          3. 3% paa 1 dag -> DAILY (stop resten af dagen)

        Returns:
            CircuitBreakerState med nuvaerende status.
        """
        # Allerede aktiv og kraever manuel reset?
        if self._cb_state.requires_manual_reset:
            return self._cb_state

        # Auto-resume check
        if self._cb_state.is_active and self._cb_state.can_auto_resume:
            logger.info(
                f"[circuit_breaker] Auto-resume: {self._cb_state.level.value} "
                f"udloebet, handler genoptages"
            )
            self._cb_state = CircuitBreakerState()

        portfolio = self._portfolio

        # Level 3: 15% fra peak
        dd = portfolio.current_drawdown_pct
        if dd >= self._cb_config.peak_drawdown_pct:
            if self._cb_state.level != CircuitBreakerLevel.CRITICAL:
                self._cb_state = CircuitBreakerState(
                    level=CircuitBreakerLevel.CRITICAL,
                    triggered_at=datetime.now().isoformat(),
                    reason=(
                        f"Drawdown {dd:.1%} >= {self._cb_config.peak_drawdown_pct:.0%} "
                        f"fra peak – STOP ALT!"
                    ),
                    requires_manual_reset=True,
                )
                logger.critical(f"[circuit_breaker] CRITICAL: {self._cb_state.reason}")
            return self._cb_state

        # Level 2: 7% paa 1 uge
        week_pnl_pct = self._get_weekly_pnl_pct()
        if week_pnl_pct <= -self._cb_config.weekly_loss_pct:
            if self._cb_state.level not in (CircuitBreakerLevel.WEEKLY, CircuitBreakerLevel.CRITICAL):
                resume_at = (
                    datetime.now() + timedelta(hours=self._cb_config.weekly_cooldown_hours)
                ).isoformat()
                self._cb_state = CircuitBreakerState(
                    level=CircuitBreakerLevel.WEEKLY,
                    triggered_at=datetime.now().isoformat(),
                    reason=(
                        f"Ugentligt tab {week_pnl_pct:.1%} <= "
                        f"-{self._cb_config.weekly_loss_pct:.0%} – "
                        f"stop i {self._cb_config.weekly_cooldown_hours}t"
                    ),
                    resume_at=resume_at,
                )
                logger.critical(f"[circuit_breaker] WEEKLY: {self._cb_state.reason}")
            return self._cb_state

        # Level 1: 3% paa 1 dag
        daily_pnl = portfolio.daily_pnl_pct
        if daily_pnl <= -self._cb_config.daily_loss_pct:
            if self._cb_state.level == CircuitBreakerLevel.NONE:
                # Stop resten af dagen (til ny dag starter)
                if self._cb_config.daily_cooldown_hours > 0:
                    resume_at = (
                        datetime.now() + timedelta(hours=self._cb_config.daily_cooldown_hours)
                    ).isoformat()
                else:
                    # Antag markedet lukker kl. 22:00 lokal tid
                    today = datetime.now().replace(hour=22, minute=0, second=0)
                    if datetime.now() >= today:
                        today += timedelta(days=1)
                    resume_at = today.isoformat()

                self._cb_state = CircuitBreakerState(
                    level=CircuitBreakerLevel.DAILY,
                    triggered_at=datetime.now().isoformat(),
                    reason=(
                        f"Dagligt tab {daily_pnl:.1%} <= "
                        f"-{self._cb_config.daily_loss_pct:.0%} – "
                        f"stop resten af dagen"
                    ),
                    resume_at=resume_at,
                )
                logger.warning(f"[circuit_breaker] DAILY: {self._cb_state.reason}")
            return self._cb_state

        return self._cb_state

    def manual_reset(self) -> None:
        """Manuel genstart af circuit breaker (kraeves ved CRITICAL)."""
        if self._cb_state.requires_manual_reset:
            logger.info(
                f"[circuit_breaker] Manuel reset: {self._cb_state.level.value} "
                f"ophævet af bruger"
            )
        self._cb_state = CircuitBreakerState()

    @property
    def is_trading_allowed(self) -> bool:
        """Returnér True hvis handel er tilladt (ingen aktive circuit breakers)."""
        if not self._cb_state.is_active:
            return True
        if self._cb_state.can_auto_resume:
            return True
        return False

    def reset_circuit_breaker(self) -> None:
        """Eksplicit nulstilling af circuit breaker — kald kun bevidst."""
        logger.info("[risk] Circuit breaker nulstillet manuelt")
        self._cb_state = CircuitBreakerState()

    def _get_weekly_pnl_pct(self) -> float:
        """Beregn P&L for den aktuelle uge."""
        if self._week_start_equity <= 0:
            return 0.0
        current = self._portfolio.total_equity
        return (current - self._week_start_equity) / self._week_start_equity

    def start_new_week(self) -> None:
        """Markér start af ny uge."""
        self._week_start_equity = self._portfolio.total_equity
        self._week_start_time = datetime.now()

    def start_new_day(self) -> None:
        """Markér start af ny dag. Auto-resume DAILY circuit breaker."""
        self._portfolio.start_new_day()
        if self._cb_state.level == CircuitBreakerLevel.DAILY:
            logger.info("[circuit_breaker] Ny dag – DAILY circuit breaker ophævet")
            self._cb_state = CircuitBreakerState()

    # ── Exposure Check ───────────────────────────────────────

    def check_exposure(self) -> dict:
        """
        Tjek nuvaerende eksponering mod regime-graenser.

        Returns:
            Dict med current_exposure, max_allowed, overexposed, action.
        """
        equity = self._portfolio.total_equity
        if equity <= 0:
            return {"current_exposure": 0, "max_allowed": 0, "overexposed": False, "action": ""}

        # Netto-eksponering: longs minus shorts (shorts reducerer eksponering)
        invested = sum(
            p.market_value if p.side == "long" else -p.market_value
            for p in self._portfolio.positions.values()
        )
        exposure_pct = invested / equity
        max_allowed = self.max_exposure_pct
        cash_pct = self._portfolio.cash / equity
        min_cash = self.cash_minimum_pct

        overexposed = exposure_pct > max_allowed
        cash_too_low = cash_pct < min_cash

        action = ""
        if overexposed:
            excess = exposure_pct - max_allowed
            action = f"Reducér eksponering med {excess:.1%} (nuv. {exposure_pct:.1%}, max {max_allowed:.1%})"
        elif cash_too_low:
            deficit = min_cash - cash_pct
            action = f"Oeg cash med {deficit:.1%} (nuv. {cash_pct:.1%}, min {min_cash:.1%})"

        return {
            "current_exposure": exposure_pct,
            "max_allowed": max_allowed,
            "cash_pct": cash_pct,
            "cash_minimum": min_cash,
            "overexposed": overexposed,
            "cash_too_low": cash_too_low,
            "action": action,
        }

    # ── Summary ──────────────────────────────────────────────

    def summary(self) -> dict:
        """Komplet status-oversigt."""
        exposure = self.check_exposure()
        return {
            "current_regime": self._current_regime.value,
            "target_regime": self._target_regime.value,
            "is_transitioning": self.is_transitioning,
            "transition_progress": self.transition_progress,
            "parameters": self.current_parameters,
            "circuit_breaker": {
                "level": self._cb_state.level.value,
                "is_active": self._cb_state.is_active,
                "reason": self._cb_state.reason,
                "requires_manual_reset": self._cb_state.requires_manual_reset,
                "resume_at": self._cb_state.resume_at,
            },
            "exposure": exposure,
            "trading_allowed": self.is_trading_allowed,
            "recent_transitions": [
                {
                    "timestamp": t.timestamp,
                    "parameter": t.parameter,
                    "old": t.old_value,
                    "new": t.new_value,
                    "day": f"{t.transition_day}/{t.total_days}",
                }
                for t in self._transitions[-10:]
            ],
        }
