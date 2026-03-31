"""
Market Handoff Module
=====================
Feeds the result of closed market sessions into the signal engine
for markets that open later in the day.

Session order (CET):
  22:00 - 03:00  New Zealand (NZX)
  01:00 - 07:00  Australia (ASX)
  01:00 - 07:30  Tokyo (TSE)
  02:00 - 08:00  Hong Kong (HKEX)
  04:45 - 11:15  Mumbai (NSE)
  09:00 - 17:30  EU + Nordic
  09:00 - 17:30  London (LSE)
  10:00 - 15:30  US Pre-market
  15:30 - 22:00  US Regular
  22:00 - 02:00  US Post-market
  24/7           Crypto
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from zoneinfo import ZoneInfo
from loguru import logger

TZ_CET = ZoneInfo("Europe/Copenhagen")


@dataclass
class SessionResult:
    """Result of a completed market session."""
    market:          str
    date:            date
    change_pct:      float        # Session return %
    volatility:      float        # Intraday volatility
    volume_ratio:    float        # Volume vs 20-day average
    breadth:         float        # % of stocks that rose (0-1)
    sentiment_score: float        # -1.0 (bearish) to +1.0 (bullish)
    regime:          str          # "bull" | "bear" | "sideways" | "crash"
    top_movers_up:   list[str] = field(default_factory=list)
    top_movers_down: list[str] = field(default_factory=list)


@dataclass
class HandoffSignal:
    """
    Adjustment signal passed from one session to the next.
    Used by AutoTrader to scale position sizes and signal thresholds.
    """
    source_market:               str
    target_market:               str
    position_size_multiplier:    float   # 0.5 = half, 1.5 = 50% larger
    signal_threshold_adjustment: float   # +0.1 = require stronger signal
    risk_off:                    bool    # True = reduce all exposure
    composite_score:             float = 0.0
    notes:                       str = ""


class MarketHandoffEngine:
    """
    Tracks completed sessions and generates adjustment signals
    for markets that open next.
    """

    # How much weight each prior session has on the next
    HANDOFF_WEIGHTS = {
        "new_zealand": {"australia": 0.4, "japan": 0.3, "hong_kong": 0.2},
        "australia":   {"japan": 0.5,     "hong_kong": 0.3, "india": 0.2},
        "japan":       {"hong_kong": 0.6, "india": 0.3, "eu": 0.2},
        "hong_kong":   {"india": 0.5,     "eu": 0.3, "us": 0.1},
        "india":       {"eu": 0.4,        "us": 0.2},
        "eu":          {"us": 0.5},  # NB: "eu" dækker EU+Nordic (exchange_limits bruger "eu_nordic")
        "us":          {"new_zealand": 0.7, "australia": 0.5, "japan": 0.4},
    }

    def __init__(self):
        self._sessions: dict[str, SessionResult] = {}

    def record_session(self, result: SessionResult) -> None:
        """Call this when a market session closes."""
        self._sessions[result.market] = result
        logger.info(
            f"[handoff] {result.market.upper()} session recorded: "
            f"{result.change_pct:+.2f}%, regime={result.regime}, "
            f"sentiment={result.sentiment_score:+.2f}"
        )

    def get_handoff_signal(self, target_market: str) -> HandoffSignal:
        """
        Generate an adjustment signal for target_market based on
        all sessions that have already closed today.
        """
        composite_score = 0.0
        total_weight    = 0.0
        notes           = []

        for source, weights in self.HANDOFF_WEIGHTS.items():
            if target_market not in weights:
                continue
            if source not in self._sessions:
                continue

            session = self._sessions[source]
            weight  = weights[target_market]
            composite_score += session.sentiment_score * weight
            total_weight    += weight
            notes.append(
                f"{source.upper()} {session.change_pct:+.1f}% "
                f"(regime: {session.regime})"
            )

        if total_weight > 0:
            composite_score /= total_weight

        # Translate composite score into trading adjustments
        if composite_score >= 0.3:
            multiplier    = 1.2
            threshold_adj = -0.05
            risk_off      = False
        elif composite_score >= 0.1:
            multiplier    = 1.0
            threshold_adj = 0.0
            risk_off      = False
        elif composite_score >= -0.2:
            multiplier    = 0.8
            threshold_adj = 0.05
            risk_off      = False
        elif composite_score >= -0.4:
            multiplier    = 0.5
            threshold_adj = 0.10
            risk_off      = False
        else:
            # Crash / very bearish
            multiplier    = 0.25
            threshold_adj = 0.20
            risk_off      = True

        signal = HandoffSignal(
            source_market=", ".join(self._sessions.keys()) or "none",
            target_market=target_market,
            position_size_multiplier=multiplier,
            signal_threshold_adjustment=threshold_adj,
            risk_off=risk_off,
            composite_score=composite_score,
            notes=" | ".join(notes) if notes else "No prior sessions today",
        )

        logger.info(
            f"[handoff] Signal for {target_market.upper()}: "
            f"size_mult={multiplier:.2f}, risk_off={risk_off}, "
            f"score={composite_score:+.2f}"
        )
        return signal

    def get_global_mood(self) -> str:
        """Summary of all sessions so far today."""
        if not self._sessions:
            return "No sessions completed yet today"
        order  = ["new_zealand", "australia", "japan", "hong_kong",
                   "india", "eu", "us"]
        parts  = []
        for market in order:
            if market in self._sessions:
                s = self._sessions[market]
                parts.append(f"{market.upper()}: {s.change_pct:+.1f}% ({s.regime})")
        return " → ".join(parts)

    def reset_daily(self) -> None:
        """Call at midnight CET to clear previous day's sessions."""
        self._sessions.clear()
        logger.info("[handoff] Daily session data reset")


# ── Singleton ──────────────────────────────────────────────
_engine: MarketHandoffEngine | None = None

def get_handoff_engine() -> MarketHandoffEngine:
    global _engine
    if _engine is None:
        _engine = MarketHandoffEngine()
    return _engine
