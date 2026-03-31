"""
RiskManager – håndhæver risikoregler FØR og EFTER handler.

Pre-trade checks:
  - Max position størrelse (% af portefølje)
  - Max antal åbne positioner
  - Max dagligt tab
  - Max drawdown fra peak
  - Tilstrækkelig kontant-dækning

Post-trade checks (monitoring):
  - Stop-loss per position
  - Take-profit per position
  - Trailing stop per position

Alle grænser er konfigurerbare via config/default_config.yaml og .env.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from loguru import logger

from config.settings import settings
from src.risk.portfolio_tracker import PortfolioTracker, Position


class RejectionReason(Enum):
    APPROVED = "APPROVED"
    MAX_POSITION_SIZE = "MAX_POSITION_SIZE"
    MAX_OPEN_POSITIONS = "MAX_OPEN_POSITIONS"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"
    DUPLICATE_POSITION = "DUPLICATE_POSITION"
    TRADING_HALTED = "TRADING_HALTED"


@dataclass
class RiskDecision:
    """Resultat af en risk-check."""
    approved: bool
    reason: RejectionReason
    message: str
    # Justeret order-størrelse (kan være reduceret)
    adjusted_qty: float = 0.0
    adjusted_usd: float = 0.0

    def __repr__(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"
        return f"RiskDecision({status}: {self.message})"


class ExitSignal:
    """Signal om at lukke en position pga. risikoregler."""

    __slots__ = ("symbol", "reason", "message", "trigger_price")

    def __init__(self, symbol: str, reason: str, message: str, trigger_price: float) -> None:
        self.symbol = symbol
        self.reason = reason
        self.message = message
        self.trigger_price = trigger_price

    def __repr__(self) -> str:
        return f"ExitSignal({self.symbol}, {self.reason}, ${self.trigger_price:.2f})"


class RiskManager:
    """
    Central risikostyring. Alle handler går igennem check_order() først.

    Standardværdier (konservative):
      - max_position_pct: 2 % af portefølje per position
      - max_daily_loss_pct: 5 % dagligt tab
      - max_open_positions: 10
      - stop_loss_pct: 2 % per position
      - take_profit_pct: 5 % per position
      - trailing_stop_pct: 3 % fra peak
      - max_drawdown_pct: 10 % fra portefølje-peak
    """

    def __init__(
        self,
        portfolio: PortfolioTracker,
        max_position_pct: float | None = None,
        max_daily_loss_pct: float | None = None,
        max_open_positions: int | None = None,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        trailing_stop_pct: float | None = None,
        max_drawdown_pct: float | None = None,
    ) -> None:
        self.portfolio = portfolio

        # Prioritet: 1) eksplicit parameter, 2) JSON config-override, 3) YAML default_config
        self.max_position_pct = max_position_pct if max_position_pct is not None else settings.risk.max_position_pct
        # Load persisted position sizing override (kun hvis IKKE sat eksplicit)
        if max_position_pct is None:
            try:
                import json
                from pathlib import Path as _Path
                _rs = _Path(__file__).resolve().parent.parent.parent / "config" / "risk_sizing.json"
                if _rs.exists():
                    _rsd = json.loads(_rs.read_text())
                    if "max_position_pct" in _rsd:
                        _val = _rsd["max_position_pct"]
                        # Værdi i JSON er allerede decimal (0.15 = 15%)
                        if isinstance(_val, (int, float)) and 0.001 <= _val <= 0.25:
                            self.max_position_pct = _val
                        else:
                            logger.warning(f"[risk] Ugyldig max_position_pct i risk_sizing.json: {_val} — ignoreret (tilladt: 0.001-0.25)")
            except Exception as e:
                logger.warning(f"[risk] Kunne ikke læse risk_sizing.json: {e}")
        self.max_daily_loss_pct = max_daily_loss_pct if max_daily_loss_pct is not None else settings.risk.max_daily_loss_pct
        self.max_open_positions = max_open_positions if max_open_positions is not None else settings.risk.max_open_positions
        # Load persisted override if available (kun hvis IKKE sat eksplicit)
        if max_open_positions is None:
            try:
                import json
                from pathlib import Path as _Path
                _mp = _Path(__file__).resolve().parent.parent.parent / "config" / "max_positions.json"
                if _mp.exists():
                    _mp_val = json.loads(_mp.read_text()).get("max_open_positions", self.max_open_positions)
                    if isinstance(_mp_val, int) and 1 <= _mp_val <= 50:
                        self.max_open_positions = _mp_val
                    else:
                        logger.warning(f"[risk] Ugyldig max_open_positions: {_mp_val} — ignoreret (tilladt: 1-50)")
            except Exception as e:
                logger.warning(f"[risk] Kunne ikke læse max_positions.json: {e}")
        self.stop_loss_pct = stop_loss_pct if stop_loss_pct is not None else settings.risk.stop_loss_pct
        # Load persisted global stop-loss override (kun hvis IKKE sat eksplicit)
        if stop_loss_pct is None:
            try:
                import json
                from pathlib import Path as _Path
                _gsl = _Path(__file__).resolve().parent.parent.parent / "config" / "global_stop_loss.json"
                if _gsl.exists():
                    _gsl_raw = json.loads(_gsl.read_text()).get("stop_loss_pct", self.stop_loss_pct)
                    # Auto-detect format: >1 = procent (f.eks. 5 = 5%), <=1 = decimal (f.eks. 0.05)
                    _gsl_val = _gsl_raw / 100.0 if _gsl_raw > 1 else _gsl_raw
                    if isinstance(_gsl_val, (int, float)) and 0.005 <= _gsl_val <= 0.20:
                        self.stop_loss_pct = _gsl_val
                    else:
                        logger.warning(f"[risk] Ugyldig stop_loss_pct: {_gsl_val} — ignoreret (tilladt: 0.005-0.20)")
            except Exception as e:
                logger.warning(f"[risk] Kunne ikke læse global_stop_loss.json: {e}")
        self.take_profit_pct = take_profit_pct if take_profit_pct is not None else settings.risk.take_profit_pct
        self.trailing_stop_pct = trailing_stop_pct if trailing_stop_pct is not None else settings.risk.trailing_stop_pct
        self.max_drawdown_pct = max_drawdown_pct if max_drawdown_pct is not None else settings.risk.max_drawdown_pct

        self._trading_halted = False
        self._halt_reason = ""

    # ── Pre-trade check ──────────────────────────────────────

    def check_order(
        self,
        symbol: str,
        side: str,
        requested_usd: float,
        price: float,
        is_exit: bool = False,
    ) -> RiskDecision:
        """
        Tjek om en ordre overholder alle risikoregler.

        Args:
            symbol: Ticker.
            side: "long" eller "short".
            requested_usd: Ønsket handelsbeløb i USD.
            price: Aktuel pris per aktie.
            is_exit: True hvis dette er en lukning af eksisterende position.

        Returns:
            RiskDecision med approved/rejected + eventuel justeret størrelse.
        """
        # Auto-detect exit: kun hvis symbolet har en position OG side er modsat (sell af long, cover af short)
        _is_exit = is_exit
        if not _is_exit and symbol in self.portfolio.positions:
            existing_side = self.portfolio.positions[symbol].side
            _is_exit = (existing_side != side)  # long+short=exit, short+long=exit

        equity = self.portfolio.total_equity

        # 1. Trading halted?
        if self._trading_halted:
            # EXIT BYPASS: tillad lukning af eksisterende positioner selvom handel er stoppet
            if _is_exit and symbol in self.portfolio.positions:
                qty = self.portfolio.positions[symbol].qty
                return RiskDecision(approved=True, reason=RejectionReason.APPROVED, message=f"Exit godkendt (halted): sælg {qty} {symbol}", adjusted_qty=qty, adjusted_usd=qty * price)
            return self._reject(
                RejectionReason.TRADING_HALTED,
                f"Handel stoppet: {self._halt_reason}",
            )

        # 2. Max drawdown
        dd = self.portfolio.current_drawdown_pct
        if dd >= self.max_drawdown_pct:
            # EXIT BYPASS: tillad lukning selvom drawdown er nået
            if _is_exit and symbol in self.portfolio.positions:
                qty = self.portfolio.positions[symbol].qty
                return RiskDecision(approved=True, reason=RejectionReason.APPROVED, message=f"Exit godkendt (drawdown): sælg {qty} {symbol}", adjusted_qty=qty, adjusted_usd=qty * price)
            self._halt_trading(f"Max drawdown nået: {dd:.1%} >= {self.max_drawdown_pct:.1%}")
            return self._reject(
                RejectionReason.MAX_DRAWDOWN,
                f"Max drawdown overskredet: {dd:.1%} >= {self.max_drawdown_pct:.1%}",
            )

        # 3. Dagligt tab
        daily_loss = -self.portfolio.daily_pnl_pct
        if daily_loss >= self.max_daily_loss_pct:
            self._halt_trading(f"Dagligt tab: {daily_loss:.1%} >= {self.max_daily_loss_pct:.1%}")
            return self._reject(
                RejectionReason.DAILY_LOSS_LIMIT,
                f"Dagligt tab overskredet: {daily_loss:.1%} >= {self.max_daily_loss_pct:.1%}",
            )

        # 4. Duplicate position — blokerer kun for SAMME retning (ny long når long eksisterer)
        # Exits (sælg eksisterende) tillades altid
        if symbol in self.portfolio.positions:
            existing_side = self.portfolio.positions[symbol].side
            if existing_side == side:
                return self._reject(
                    RejectionReason.DUPLICATE_POSITION,
                    f"Position i {symbol} ({existing_side}) eksisterer allerede",
                )

        # 5. Max åbne positioner
        if self.portfolio.open_position_count >= self.max_open_positions:
            return self._reject(
                RejectionReason.MAX_OPEN_POSITIONS,
                f"Max positioner nået: {self.portfolio.open_position_count} >= {self.max_open_positions}",
            )

        # 6. Max position size – justér ned hvis nødvendigt
        max_usd = equity * self.max_position_pct
        adjusted_usd = min(requested_usd, max_usd)

        if adjusted_usd <= 0:
            return self._reject(
                RejectionReason.MAX_POSITION_SIZE,
                f"Position ville være $0 (equity=${equity:,.0f}, max={self.max_position_pct:.0%})",
            )

        # 7. Cash check
        if adjusted_usd > self.portfolio.cash:
            adjusted_usd = self.portfolio.cash
            if adjusted_usd <= 0:
                return self._reject(
                    RejectionReason.INSUFFICIENT_CASH,
                    f"Ingen kontanter tilgængelige (cash=${self.portfolio.cash:,.2f})",
                )

        # Beregn qty — fractional for commodities, forex og crypto
        is_fractional = any(symbol.endswith(s) for s in ("=F", "=X", "-USD"))
        if is_fractional:
            qty = round(adjusted_usd / price, 4) if price > 0 else 0
        else:
            qty = int(adjusted_usd / price) if price > 0 else 0
        if qty <= 0:
            return self._reject(
                RejectionReason.MAX_POSITION_SIZE,
                f"Kan ikke købe nogen aktier (pris=${price:.2f}, budget=${adjusted_usd:.2f})",
            )

        final_usd = qty * price

        was_reduced = final_usd < requested_usd
        msg = (
            f"Godkendt: {qty} {symbol} @ ${price:.2f} = ${final_usd:,.2f}"
            + (f" (reduceret fra ${requested_usd:,.2f})" if was_reduced else "")
        )

        logger.info(f"Risk check: {msg}")

        return RiskDecision(
            approved=True,
            reason=RejectionReason.APPROVED,
            message=msg,
            adjusted_qty=qty,
            adjusted_usd=final_usd,
        )

    # ── Post-trade monitoring ────────────────────────────────

    def check_positions(self, prices: dict[str, float]) -> list[ExitSignal]:
        """
        Tjek alle åbne positioner mod stop-loss, take-profit og trailing stop.

        Kald denne metode efter update_prices() i portfolio tracker.

        Returns:
            Liste af ExitSignals for positioner der bør lukkes.
        """
        exits: list[ExitSignal] = []

        for symbol, pos in self.portfolio.positions.items():
            price = prices.get(symbol, pos.current_price)
            pos.update_price(price)

            # Stop-loss
            if pos.unrealized_pnl_pct <= -self.stop_loss_pct:
                exits.append(ExitSignal(
                    symbol=symbol,
                    reason="stop_loss",
                    message=(
                        f"Stop-loss udløst: {pos.unrealized_pnl_pct:.2%} "
                        f"<= -{self.stop_loss_pct:.2%}"
                    ),
                    trigger_price=price,
                ))
                continue

            # Take-profit
            if pos.unrealized_pnl_pct >= self.take_profit_pct:
                exits.append(ExitSignal(
                    symbol=symbol,
                    reason="take_profit",
                    message=(
                        f"Take-profit udløst: {pos.unrealized_pnl_pct:.2%} "
                        f">= {self.take_profit_pct:.2%}"
                    ),
                    trigger_price=price,
                ))
                continue

            # Trailing stop
            if pos.pct_from_peak >= self.trailing_stop_pct:
                exits.append(ExitSignal(
                    symbol=symbol,
                    reason="trailing_stop",
                    message=(
                        f"Trailing stop udløst: {pos.pct_from_peak:.2%} fald "
                        f"fra peak ${pos.peak_price:.2f}"
                    ),
                    trigger_price=price,
                ))

        for exit_sig in exits:
            logger.warning(f"EXIT: {exit_sig.symbol} – {exit_sig.message}")

        return exits

    # ── Portfolio-level checks ───────────────────────────────

    def check_daily_limit(self) -> bool:
        """Returnér True hvis daglig tabsgrænse er overskredet."""
        daily_loss = -self.portfolio.daily_pnl_pct
        return daily_loss >= self.max_daily_loss_pct

    def check_drawdown_limit(self) -> bool:
        """Returnér True hvis max drawdown er overskredet."""
        return self.portfolio.current_drawdown_pct >= self.max_drawdown_pct

    # ── Halt / Resume ────────────────────────────────────────

    def _halt_trading(self, reason: str) -> None:
        if not self._trading_halted:
            self._trading_halted = True
            self._halt_reason = reason
            logger.critical(f"HANDEL STOPPET: {reason}")

    def resume_trading(self) -> None:
        """Genoptag handel (f.eks. ved start af ny dag)."""
        if self._trading_halted:
            logger.info(f"Handel genoptaget (var stoppet: {self._halt_reason})")
            self._trading_halted = False
            self._halt_reason = ""

    @property
    def is_trading_halted(self) -> bool:
        return self._trading_halted

    # ── Helpers ──────────────────────────────────────────────

    def _reject(self, reason: RejectionReason, message: str) -> RiskDecision:
        logger.warning(f"Risk AFVIST: {message}")
        return RiskDecision(
            approved=False,
            reason=reason,
            message=message,
        )

    def summary(self) -> dict:
        """Returnér risk-status som dict."""
        return {
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "daily_pnl_pct": self.portfolio.daily_pnl_pct,
            "daily_limit_pct": self.max_daily_loss_pct,
            "drawdown_pct": self.portfolio.current_drawdown_pct,
            "drawdown_limit_pct": self.max_drawdown_pct,
            "open_positions": self.portfolio.open_position_count,
            "max_positions": self.max_open_positions,
            "max_position_size_pct": self.max_position_pct,
        }
