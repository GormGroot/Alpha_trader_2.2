"""
RegimeDetector & AdaptiveStrategy – markedsregime-detektion og tilpasning.

Regime-typer:
  - BULL: Opadgående trend, lav volatilitet
  - BEAR: Nedadgående trend, stigende volatilitet
  - SIDEWAYS: Ingen klar retning, range-bound
  - CRASH: Hurtigt fald, ekstremt høj volatilitet
  - RECOVERY: Bunden er fundet, tidlig opadgående trend
  - EUPHORIA: Meget stærk stigning, mulig boble

Detektionsmetoder:
  - Trend (SMA 50/200)
  - Volatilitet (VIX-niveau eller realiseret vol)
  - Breadth (advance/decline)
  - Momentum (ROC acceleration)
  - Volume (distribution vs. accumulation)
  - Yield curve (spread 10Y-2Y)
  - Hidden Markov Model (statistisk regime-skift)

AdaptiveStrategy tilpasser eksponering, stop-loss og strategi efter regimet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from collections import deque

import numpy as np
import pandas as pd
from loguru import logger

from src.strategy.base_strategy import BaseStrategy, Signal, StrategyResult


# ══════════════════════════════════════════════════════════════
#  Enums & Dataklasser
# ══════════════════════════════════════════════════════════════

class MarketRegime(Enum):
    """Markedets overordnede tilstand."""
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    CRASH = "crash"
    RECOVERY = "recovery"
    EUPHORIA = "euphoria"


# Regime-metadata: farve, beskrivelse, default eksponering
REGIME_INFO: dict[MarketRegime, dict[str, Any]] = {
    MarketRegime.BULL: {
        "label": "BULL MARKET",
        "color": "#2ed573",
        "icon": "bi-graph-up-arrow",
        "description": "Opadgående trend, sunde forhold",
        "max_exposure": 1.00,
    },
    MarketRegime.BEAR: {
        "label": "BEAR MARKET",
        "color": "#ff4757",
        "icon": "bi-graph-down-arrow",
        "description": "Nedadgående trend, stigende risiko",
        "max_exposure": 0.30,
    },
    MarketRegime.SIDEWAYS: {
        "label": "SIDEWAYS",
        "color": "#ffa502",
        "icon": "bi-arrows-expand",
        "description": "Ingen klar retning, range-bound",
        "max_exposure": 0.50,
    },
    MarketRegime.CRASH: {
        "label": "CRASH",
        "color": "#ff0000",
        "icon": "bi-exclamation-triangle-fill",
        "description": "Hurtigt fald, ekstremt volatilt!",
        "max_exposure": 0.10,
    },
    MarketRegime.RECOVERY: {
        "label": "RECOVERY",
        "color": "#3498db",
        "icon": "bi-arrow-up-circle",
        "description": "Bunden fundet, tidlig recovery",
        "max_exposure": 0.50,
    },
    MarketRegime.EUPHORIA: {
        "label": "EUPHORIA",
        "color": "#a855f7",
        "icon": "bi-lightning-fill",
        "description": "Ekstremt bullish, boble-risiko!",
        "max_exposure": 0.70,
    },
}


@dataclass
class RegimeSignal:
    """Signal fra én detektionsmetode."""
    name: str
    value: float           # Normaliseret -1 (meget bearish) til +1 (meget bullish)
    weight: float = 1.0    # Vægt i den samlede vurdering
    detail: str = ""       # Forklaring


@dataclass
class RegimeResult:
    """Samlet regime-vurdering."""
    regime: MarketRegime
    confidence: float              # 0-100%
    composite_score: float         # -1 til +1 (samlet score)
    signals: list[RegimeSignal] = field(default_factory=list)
    timestamp: str = ""
    reason: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @property
    def label(self) -> str:
        return REGIME_INFO[self.regime]["label"]

    @property
    def color(self) -> str:
        return REGIME_INFO[self.regime]["color"]

    @property
    def max_exposure(self) -> float:
        return REGIME_INFO[self.regime]["max_exposure"]


@dataclass
class RegimeShift:
    """Log-entry for et regime-skift."""
    timestamp: str
    from_regime: MarketRegime
    to_regime: MarketRegime
    confidence: float
    reason: str
    composite_score: float


@dataclass
class StrategyAdjustment:
    """Anbefalet strategi-tilpasning baseret på regime."""
    regime: MarketRegime
    max_exposure_pct: float
    stop_loss_multiplier: float     # 1.0 = standard, 0.5 = strammere, 1.5 = løsere
    preferred_strategies: list[str]
    preferred_sectors: list[str]
    avoid_sectors: list[str]
    safe_havens: list[str]
    allow_new_buys: bool
    allow_shorts: bool
    notes: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
#  RegimeDetector
# ══════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    Detektér markedets regime via multiple metoder.

    Kombinerer trend, volatilitet, momentum, breadth, volume og
    valgfrit en Hidden Markov Model for statistisk regime-skift.

    Brug:
        detector = RegimeDetector()
        result = detector.detect(market_df)
        result = detector.detect(market_df, vix_level=22.5)
    """

    # Vægte for de forskellige signaler
    DEFAULT_WEIGHTS: dict[str, float] = {
        "trend": 2.0,
        "volatility": 1.5,
        "momentum": 1.0,
        "volume": 0.8,
        "breadth": 1.0,
        "yield_curve": 0.7,
        "hmm": 1.5,
    }

    # VIX-niveauer
    VIX_CALM = 15.0
    VIX_NORMAL = 25.0
    VIX_NERVOUS = 35.0
    # Over 35 = krise

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        sma_short: int = 50,
        sma_long: int = 200,
        vol_window: int = 20,
        momentum_window: int = 20,
    ) -> None:
        self._weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._sma_short = sma_short
        self._sma_long = sma_long
        self._vol_window = vol_window
        self._momentum_window = momentum_window
        self._history: deque[RegimeResult] = deque(maxlen=500)
        self._shifts: deque[RegimeShift] = deque(maxlen=200)
        self._hmm_model = None
        self._hmm_last_trained: datetime | None = None
        self._hmm_retrain_days: int = 7

    @property
    def history(self) -> list[RegimeResult]:
        return list(self._history)

    @property
    def shifts(self) -> list[RegimeShift]:
        return list(self._shifts)

    @property
    def current_regime(self) -> MarketRegime | None:
        return self._history[-1].regime if self._history else None

    # ── Hovedmetode ───────────────────────────────────────────

    def detect(
        self,
        market_data: pd.DataFrame,
        vix_level: float | None = None,
        breadth_ratio: float | None = None,
        yield_spread: float | None = None,
        use_hmm: bool = False,
    ) -> RegimeResult:
        """
        Detektér markedets regime.

        Args:
            market_data: OHLCV DataFrame (mindst 200 rækker ideelt).
            vix_level: VIX-indeks (valgfrit, ellers beregnes realiseret vol).
            breadth_ratio: Advance/decline ratio (valgfrit).
            yield_spread: 10Y-2Y rentespread (valgfrit).
            use_hmm: Brug Hidden Markov Model (kræver hmmlearn).

        Returns:
            RegimeResult med regime, confidence og detaljer.
        """
        if market_data is None or market_data.empty or len(market_data) < 20:
            return RegimeResult(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.0,
                composite_score=0.0,
                reason="Utilstrækkelig data",
            )

        signals: list[RegimeSignal] = []

        # 1. Trend-signal
        signals.append(self._trend_signal(market_data))

        # 2. Volatilitets-signal
        signals.append(self._volatility_signal(market_data, vix_level))

        # 3. Momentum-signal
        signals.append(self._momentum_signal(market_data))

        # 4. Volume-signal
        signals.append(self._volume_signal(market_data))

        # 5. Breadth-signal (hvis tilgængeligt)
        if breadth_ratio is not None:
            signals.append(self._breadth_signal(breadth_ratio))

        # 6. Yield curve (hvis tilgængeligt)
        if yield_spread is not None:
            signals.append(self._yield_curve_signal(yield_spread))

        # 7. HMM (valgfrit)
        if use_hmm:
            hmm_sig = self._hmm_signal(market_data)
            if hmm_sig is not None:
                signals.append(hmm_sig)

        # Beregn composite score
        composite = self._compute_composite(signals)

        # Bestem regime fra composite + volatilitet
        vol_signal = next((s for s in signals if s.name == "volatility"), None)
        vol_value = vol_signal.value if vol_signal else 0.0

        regime = self._classify_regime(composite, vol_value, market_data)

        # Confidence: baseret på hvor enige signalerne er
        confidence = self._compute_confidence(signals, regime)

        # Byg reason-string
        top_signals = sorted(signals, key=lambda s: abs(s.value) * s.weight, reverse=True)
        reason_parts = [f"{s.name}: {s.detail}" for s in top_signals[:3] if s.detail]
        reason = " | ".join(reason_parts)

        result = RegimeResult(
            regime=regime,
            confidence=confidence,
            composite_score=composite,
            signals=signals,
            reason=reason,
        )

        # Log regime-skift
        if self._history:
            prev = self._history[-1]
            if prev.regime != regime:
                shift = RegimeShift(
                    timestamp=result.timestamp,
                    from_regime=prev.regime,
                    to_regime=regime,
                    confidence=confidence,
                    reason=reason,
                    composite_score=composite,
                )
                self._shifts.append(shift)
                logger.info(
                    f"[regime] SKIFT: {prev.regime.value} → {regime.value} "
                    f"(confidence={confidence:.0f}%, score={composite:+.2f}) – {reason}"
                )

        self._history.append(result)
        return result

    # ── Individuelle Signaler ────────────────────────────────

    def _trend_signal(self, df: pd.DataFrame) -> RegimeSignal:
        """
        Trend-signal baseret på SMA 50 og 200.

        - Pris > SMA200 > SMA50: Stærk bull (+1.0)
        - Pris > SMA200: Moderat bull (+0.5)
        - Pris < SMA200 < SMA50: Stærk bear (-1.0)
        - Pris < SMA200: Moderat bear (-0.5)
        """
        close = df["Close"]
        n = len(close)

        if n < self._sma_long:
            # Brug kortere perioder hvis data er begrænset
            sma_s = close.rolling(min(self._sma_short, n // 2)).mean()
            sma_l = close.rolling(min(self._sma_long, n)).mean()
        else:
            sma_s = close.rolling(self._sma_short).mean()
            sma_l = close.rolling(self._sma_long).mean()

        current_price = float(close.iloc[-1])
        sma_short_val = float(sma_s.iloc[-1]) if not pd.isna(sma_s.iloc[-1]) else current_price
        sma_long_val = float(sma_l.iloc[-1]) if not pd.isna(sma_l.iloc[-1]) else current_price

        # Pris relativt til SMA200
        pct_from_sma200 = (current_price - sma_long_val) / sma_long_val if sma_long_val > 0 else 0
        # SMA50 relativt til SMA200 (golden/death cross)
        sma_cross = (sma_short_val - sma_long_val) / sma_long_val if sma_long_val > 0 else 0

        # Kombiner til score
        score = np.clip(pct_from_sma200 * 5 + sma_cross * 10, -1, 1)

        if score > 0.5:
            detail = f"Pris {pct_from_sma200:+.1%} over SMA200, golden cross aktiv"
        elif score > 0:
            detail = f"Pris {pct_from_sma200:+.1%} over SMA200"
        elif score > -0.5:
            detail = f"Pris {pct_from_sma200:+.1%} under SMA200"
        else:
            detail = f"Pris {pct_from_sma200:+.1%} under SMA200, death cross aktiv"

        return RegimeSignal(
            name="trend",
            value=float(score),
            weight=self._weights.get("trend", 2.0),
            detail=detail,
        )

    def _volatility_signal(
        self, df: pd.DataFrame, vix_level: float | None = None,
    ) -> RegimeSignal:
        """
        Volatilitets-signal.

        Bruger VIX hvis tilgængeligt, ellers realiseret volatilitet.
        Høj volatilitet = bearish, lav = bullish.
        """
        if vix_level is not None:
            # VIX-baseret
            if vix_level < self.VIX_CALM:
                score = 0.5   # Roligt marked = let bullish
                detail = f"VIX={vix_level:.1f} (roligt)"
            elif vix_level < self.VIX_NORMAL:
                score = 0.0   # Normal
                detail = f"VIX={vix_level:.1f} (normalt)"
            elif vix_level < self.VIX_NERVOUS:
                score = -0.5  # Nervøst
                detail = f"VIX={vix_level:.1f} (nervøst)"
            else:
                score = -1.0  # Krise
                detail = f"VIX={vix_level:.1f} (KRISE!)"
        else:
            # Realiseret volatilitet
            returns = df["Close"].pct_change().dropna()
            if len(returns) < self._vol_window:
                return RegimeSignal(
                    name="volatility", value=0.0,
                    weight=self._weights.get("volatility", 1.5),
                    detail="Utilstrækkelig vol-data",
                )

            recent_vol = float(returns.iloc[-self._vol_window:].std()) * np.sqrt(252) * 100
            long_vol = float(returns.std()) * np.sqrt(252) * 100

            vol_ratio = recent_vol / long_vol if long_vol > 0 else 1.0

            if vol_ratio < 0.7:
                score = 0.5
                detail = f"Vol={recent_vol:.1f}% (lavt, {vol_ratio:.1f}x normalt)"
            elif vol_ratio < 1.3:
                score = 0.0
                detail = f"Vol={recent_vol:.1f}% (normalt)"
            elif vol_ratio < 2.0:
                score = -0.5
                detail = f"Vol={recent_vol:.1f}% (forhøjet, {vol_ratio:.1f}x normalt)"
            else:
                score = -1.0
                detail = f"Vol={recent_vol:.1f}% (EKSTREMT, {vol_ratio:.1f}x normalt!)"

        return RegimeSignal(
            name="volatility",
            value=float(score),
            weight=self._weights.get("volatility", 1.5),
            detail=detail,
        )

    def _momentum_signal(self, df: pd.DataFrame) -> RegimeSignal:
        """
        Momentum-signal: Rate of Change og acceleration.

        Positiv ROC + acceleration = stærk bull.
        Negativ ROC + deceleration = bear.
        """
        close = df["Close"]
        n = len(close)
        win = min(self._momentum_window, n - 1)

        if win < 5:
            return RegimeSignal(
                name="momentum", value=0.0,
                weight=self._weights.get("momentum", 1.0),
                detail="Utilstrækkelig data",
            )

        # Rate of Change (20-dages)
        roc = float((close.iloc[-1] / close.iloc[-win] - 1))

        # Acceleration: ROC nu vs. ROC for halv-perioden siden
        half = max(win // 2, 1)
        if n > win + half:
            roc_prev = float((close.iloc[-half] / close.iloc[-half - win] - 1))
            acceleration = roc - roc_prev
        else:
            acceleration = 0.0

        # Score
        score = np.clip(roc * 5 + acceleration * 10, -1, 1)

        if roc > 0.05 and acceleration > 0:
            detail = f"ROC={roc:+.1%}, accelererende"
        elif roc > 0:
            detail = f"ROC={roc:+.1%}, positivt momentum"
        elif roc > -0.05:
            detail = f"ROC={roc:+.1%}, svagt momentum"
        else:
            detail = f"ROC={roc:+.1%}, stærkt negativt momentum"

        return RegimeSignal(
            name="momentum",
            value=float(score),
            weight=self._weights.get("momentum", 1.0),
            detail=detail,
        )

    def _volume_signal(self, df: pd.DataFrame) -> RegimeSignal:
        """
        Volume-signal: Distribution vs. Accumulation.

        Stigende volumen på ned-dage = distribution (bearish).
        Stigende volumen på op-dage = accumulation (bullish).
        """
        if "Volume" not in df.columns or len(df) < 20:
            return RegimeSignal(
                name="volume", value=0.0,
                weight=self._weights.get("volume", 0.8),
                detail="Ingen volume-data",
            )

        close = df["Close"]
        volume = df["Volume"]
        returns = close.pct_change()

        win = min(20, len(df) - 1)
        recent = slice(-win, None)

        up_days = returns.iloc[recent] > 0
        down_days = returns.iloc[recent] < 0

        up_volume = float(volume.iloc[recent][up_days].mean()) if up_days.any() else 0
        down_volume = float(volume.iloc[recent][down_days].mean()) if down_days.any() else 0

        if up_volume + down_volume == 0:
            score = 0.0
            detail = "Ingen volume-aktivitet"
        else:
            # Accumulation/Distribution ratio
            ad_ratio = (up_volume - down_volume) / (up_volume + down_volume)
            score = np.clip(float(ad_ratio) * 2, -1, 1)

            if ad_ratio > 0.2:
                detail = f"Accumulation (op-vol={up_volume/1e6:.0f}M > ned-vol={down_volume/1e6:.0f}M)"
            elif ad_ratio < -0.2:
                detail = f"Distribution (ned-vol={down_volume/1e6:.0f}M > op-vol={up_volume/1e6:.0f}M)"
            else:
                detail = f"Neutral volume (AD={ad_ratio:+.2f})"

        return RegimeSignal(
            name="volume",
            value=float(score),
            weight=self._weights.get("volume", 0.8),
            detail=detail,
        )

    def _breadth_signal(self, breadth_ratio: float) -> RegimeSignal:
        """
        Breadth-signal: Advance/Decline ratio.

        > 2.0 = stærk markedsbredde (bullish)
        < 0.5 = dårlig markedsbredde (bearish)
        """
        # Normaliser: 1.0 = neutral, >2 = bullish, <0.5 = bearish
        if breadth_ratio > 2.0:
            score = 1.0
            detail = f"A/D={breadth_ratio:.1f} (stærk bredde)"
        elif breadth_ratio > 1.2:
            score = 0.5
            detail = f"A/D={breadth_ratio:.1f} (god bredde)"
        elif breadth_ratio > 0.8:
            score = 0.0
            detail = f"A/D={breadth_ratio:.1f} (neutral)"
        elif breadth_ratio > 0.5:
            score = -0.5
            detail = f"A/D={breadth_ratio:.1f} (dårlig bredde)"
        else:
            score = -1.0
            detail = f"A/D={breadth_ratio:.1f} (kapitulation)"

        return RegimeSignal(
            name="breadth",
            value=score,
            weight=self._weights.get("breadth", 1.0),
            detail=detail,
        )

    def _yield_curve_signal(self, yield_spread: float) -> RegimeSignal:
        """
        Yield curve signal: 10Y-2Y spread.

        Inverteret (< 0) = recession-risiko.
        Normalt (0.5-2.0) = sund økonomi.
        Stejl (> 2.0) = tidlig recovery.
        """
        if yield_spread < -0.5:
            score = -1.0
            detail = f"Spread={yield_spread:.2f}% (dybt inverteret, recession!)"
        elif yield_spread < 0:
            score = -0.5
            detail = f"Spread={yield_spread:.2f}% (inverteret, advarsel)"
        elif yield_spread < 0.5:
            score = 0.0
            detail = f"Spread={yield_spread:.2f}% (fladt)"
        elif yield_spread < 2.0:
            score = 0.3
            detail = f"Spread={yield_spread:.2f}% (normalt)"
        else:
            score = 0.5
            detail = f"Spread={yield_spread:.2f}% (stejlt, tidlig recovery)"

        return RegimeSignal(
            name="yield_curve",
            value=score,
            weight=self._weights.get("yield_curve", 0.7),
            detail=detail,
        )

    def _hmm_signal(self, df: pd.DataFrame) -> RegimeSignal | None:
        """
        Hidden Markov Model for statistisk regime-detektion.

        Bruger hmmlearn's GaussianHMM med 3 states:
        - State 0: Low vol (bull)
        - State 1: Medium vol (sideways)
        - State 2: High vol (bear/crash)
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.debug("[regime] hmmlearn ikke installeret, springer HMM over")
            return None

        returns = df["Close"].pct_change().dropna().values.reshape(-1, 1)
        if len(returns) < 50:
            return None

        try:
            # Retrain if model is None or older than _hmm_retrain_days
            needs_retrain = (
                self._hmm_model is None
                or self._hmm_last_trained is None
                or (datetime.now() - self._hmm_last_trained).days >= self._hmm_retrain_days
            )
            if needs_retrain:
                model = GaussianHMM(
                    n_components=3, covariance_type="full",
                    n_iter=100, random_state=42,
                )
                model.fit(returns)
                self._hmm_model = model
                self._hmm_last_trained = datetime.now()
                logger.debug("[regime] HMM model (re)trained")

            hidden_states = self._hmm_model.predict(returns)
            current_state = int(hidden_states[-1])

            # Sortér states efter gennemsnitlig return
            means = self._hmm_model.means_.flatten()
            state_order = np.argsort(means)  # Laveste til højeste

            # Map til score
            if current_state == state_order[2]:
                score = 0.8   # Highest mean = bullish
                detail = f"HMM: bull-state (mean={means[current_state]:.4f})"
            elif current_state == state_order[0]:
                score = -0.8  # Lowest mean = bearish
                detail = f"HMM: bear-state (mean={means[current_state]:.4f})"
            else:
                score = 0.0
                detail = f"HMM: neutral-state (mean={means[current_state]:.4f})"

            return RegimeSignal(
                name="hmm",
                value=float(score),
                weight=self._weights.get("hmm", 1.5),
                detail=detail,
            )

        except Exception as exc:
            logger.warning(f"[regime] HMM fejl: {exc}")
            return None

    # ── Composite & Classification ───────────────────────────

    def _compute_composite(self, signals: list[RegimeSignal]) -> float:
        """Beregn vægtet composite score fra alle signaler."""
        total_weight = sum(s.weight for s in signals)
        if total_weight == 0:
            return 0.0
        weighted = sum(s.value * s.weight for s in signals)
        return float(np.clip(weighted / total_weight, -1, 1))

    def _classify_regime(
        self,
        composite: float,
        vol_signal: float,
        df: pd.DataFrame,
    ) -> MarketRegime:
        """
        Klassificér regime baseret på composite score og volatilitet.

        Regime-regler:
          CRASH:    composite < -0.5 OG vol_signal < -0.7
          BEAR:     composite < -0.3
          EUPHORIA: composite > 0.7 OG pris > 20% over SMA200
          BULL:     composite > 0.2
          RECOVERY: composite > 0 OG seneste 20d return > 10% OG var nyligt bearish
          SIDEWAYS: alt andet
        """
        # Tjek for crash først (højeste prioritet)
        if composite < -0.5 and vol_signal <= -0.7:
            return MarketRegime.CRASH

        # Tjek for ekstremt bearish
        if composite < -0.3:
            return MarketRegime.BEAR

        # Tjek for euphoria
        if composite > 0.7:
            close = df["Close"]
            if len(close) >= 200:
                sma200 = float(close.rolling(200).mean().iloc[-1])
                if sma200 > 0 and float(close.iloc[-1]) > sma200 * 1.20:
                    return MarketRegime.EUPHORIA

        # Tjek for recovery (var nyligt bearish, nu opad)
        if 0 < composite < 0.5:
            close = df["Close"]
            recent_return = float(close.iloc[-1] / close.iloc[-min(20, len(close) - 1)] - 1)
            was_bearish = any(
                r.regime in (MarketRegime.BEAR, MarketRegime.CRASH)
                for r in list(self._history)[-5:]
            ) if self._history else False
            if recent_return > 0.10 and was_bearish:
                return MarketRegime.RECOVERY

        # Bull
        if composite > 0.2:
            return MarketRegime.BULL

        # Default: sideways
        return MarketRegime.SIDEWAYS

    def _compute_confidence(
        self, signals: list[RegimeSignal], regime: MarketRegime,
    ) -> float:
        """
        Beregn confidence baseret på enighed mellem signaler.

        Høj confidence = alle signaler peger samme vej.
        Lav confidence = modstridende signaler.
        """
        if not signals:
            return 0.0

        values = [s.value for s in signals]
        weights = [s.weight for s in signals]

        # Vægtet standardafvigelse (lav = enighed = høj confidence)
        weighted_mean = np.average(values, weights=weights)
        variance = np.average(
            [(v - weighted_mean) ** 2 for v in values],
            weights=weights,
        )
        std = float(np.sqrt(variance))

        # Konvertér: lav std → høj confidence
        # std=0 → 95%, std=0.5 → 70%, std=1.0 → 40%
        confidence = max(10, min(95, 95 - std * 55))

        # Boost confidence hvis composite er stærkt
        composite = abs(weighted_mean)
        if composite > 0.7:
            confidence = min(95, confidence + 10)

        return float(confidence)

    # ── Historisk Regime-sekvens ──────────────────────────────

    def get_regime_history(
        self, df: pd.DataFrame, step: int = 5,
    ) -> pd.DataFrame:
        """
        Beregn regime for hvert tidsskridt i en DataFrame.

        Args:
            df: OHLCV DataFrame.
            step: Beregn regime hvert N rækker.

        Returns:
            DataFrame med dato, regime, confidence, score.
        """
        results = []
        n = len(df)
        min_window = max(50, self._sma_long)

        for i in range(min_window, n, step):
            subset = df.iloc[:i + 1]
            result = self.detect(subset)
            results.append({
                "date": df.index[i] if hasattr(df.index, '__getitem__') else i,
                "regime": result.regime.value,
                "confidence": result.confidence,
                "composite_score": result.composite_score,
            })

        return pd.DataFrame(results) if results else pd.DataFrame(
            columns=["date", "regime", "confidence", "composite_score"]
        )


# ══════════════════════════════════════════════════════════════
#  AdaptiveStrategy
# ══════════════════════════════════════════════════════════════

class AdaptiveStrategy(BaseStrategy):
    """
    Strategi der tilpasser sig markedets regime.

    Bruger RegimeDetector til at bestemme markedstilstand og
    justerer eksponering, stop-loss og strategi-valg.

    BULL:     100% eksponering, momentum-strategier, growth
    BEAR:     30% eksponering, defensive, strammere stops, inverse ETF
    SIDEWAYS: 50% eksponering, mean-reversion, udbytte-aktier
    CRASH:    10% eksponering, STOP køb, safe havens
    RECOVERY: 50% → 75%, value plays, cykliske
    EUPHORIA: 70% eksponering, trailing stops, obs boble
    """

    # Regime → StrategyAdjustment
    _REGIME_ADJUSTMENTS: dict[MarketRegime, StrategyAdjustment] = {
        MarketRegime.BULL: StrategyAdjustment(
            regime=MarketRegime.BULL,
            max_exposure_pct=1.00,
            stop_loss_multiplier=1.5,     # Løsere stops
            preferred_strategies=["momentum", "sma_crossover", "ml_strategy"],
            preferred_sectors=["Technology", "Consumer Discretionary", "Communication Services"],
            avoid_sectors=[],
            safe_havens=[],
            allow_new_buys=True,
            allow_shorts=False,
            notes=["Fuld eksponering tilladt", "Favorisér growth og tech",
                    "Lad vindere løbe med løsere stop-loss"],
        ),
        MarketRegime.BEAR: StrategyAdjustment(
            regime=MarketRegime.BEAR,
            max_exposure_pct=0.30,
            stop_loss_multiplier=0.6,     # Strammere stops
            preferred_strategies=["rsi_strategy", "mean_reversion"],
            preferred_sectors=["Utilities", "Healthcare", "Consumer Staples"],
            avoid_sectors=["Technology", "Consumer Discretionary", "Financials"],
            safe_havens=["GLD", "TLT", "SHY", "BIL"],
            allow_new_buys=True,          # Selektivt
            allow_shorts=True,            # Via inverse ETF
            notes=["Max 30% eksponering", "Skift til defensive sektorer",
                    "Overvej inverse ETF (SH, PSQ)", "Stramme stop-losses",
                    "Øg allokering til guld (GLD) og obligationer (TLT)"],
        ),
        MarketRegime.SIDEWAYS: StrategyAdjustment(
            regime=MarketRegime.SIDEWAYS,
            max_exposure_pct=0.50,
            stop_loss_multiplier=1.0,     # Standard
            preferred_strategies=["rsi_strategy", "mean_reversion", "combined"],
            preferred_sectors=["Utilities", "Real Estate", "Consumer Staples"],
            avoid_sectors=[],
            safe_havens=["TLT"],
            allow_new_buys=True,
            allow_shorts=False,
            notes=["50% eksponering", "Mean-reversion: køb lav RSI, sælg høj RSI",
                    "Fokusér på aktier med høj udbytte (SCHD, VYM)",
                    "Smal handelsrange – brug Bollinger Bands"],
        ),
        MarketRegime.CRASH: StrategyAdjustment(
            regime=MarketRegime.CRASH,
            max_exposure_pct=0.10,
            stop_loss_multiplier=0.4,     # Meget stramme stops
            preferred_strategies=[],      # STOP alle strategier
            preferred_sectors=[],
            avoid_sectors=["ALL"],
            safe_havens=["GLD", "SHY", "BIL", "TLT", "CASH"],
            allow_new_buys=False,         # STOP alle nye køb!
            allow_shorts=True,
            notes=["⚠️ CRASH REGIME – STOP alle nye køb!",
                    "Reducér til max 10% eksponering",
                    "Flyt til safe havens: guld, korte obligationer, kontanter",
                    "Alle stop-losses stramt",
                    "Send URGENT alert til bruger!"],
        ),
        MarketRegime.RECOVERY: StrategyAdjustment(
            regime=MarketRegime.RECOVERY,
            max_exposure_pct=0.50,
            stop_loss_multiplier=0.8,
            preferred_strategies=["ml_strategy", "sma_crossover"],
            preferred_sectors=["Financials", "Industrials", "Materials", "Consumer Discretionary"],
            avoid_sectors=[],
            safe_havens=["GLD"],
            allow_new_buys=True,
            allow_shorts=False,
            notes=["Langsomt øg eksponering: 25% → 50% → 75%",
                    "Køb kvalitetsaktier der er faldet mest (value plays)",
                    "Favorisér cykliske aktier",
                    "Brug dollar-cost averaging"],
        ),
        MarketRegime.EUPHORIA: StrategyAdjustment(
            regime=MarketRegime.EUPHORIA,
            max_exposure_pct=0.70,
            stop_loss_multiplier=0.7,     # Strammere – beskyt gevinster
            preferred_strategies=["momentum", "sma_crossover"],
            preferred_sectors=["Technology"],
            avoid_sectors=["SPACs", "Meme Stocks"],
            safe_havens=["GLD", "TLT"],
            allow_new_buys=True,          # Men forsigtigere
            allow_shorts=False,
            notes=["⚠️ EUPHORIA – boble-risiko!",
                    "Max 70% eksponering, reducér gradvist",
                    "Stramme trailing stops for at beskytte gevinster",
                    "Undgå FOMO – undgå spekulative aktier",
                    "Overvej at tage profit på de bedste positioner"],
        ),
    }

    def __init__(
        self,
        detector: RegimeDetector | None = None,
        inner_strategy: BaseStrategy | None = None,
    ) -> None:
        """
        Args:
            detector: RegimeDetector (oprettes automatisk hvis None).
            inner_strategy: Underliggende strategi til signal-generering.
        """
        self._detector = detector or RegimeDetector()
        self._inner = inner_strategy
        self._last_result: RegimeResult | None = None

    @property
    def name(self) -> str:
        return "adaptive_regime"

    @property
    def last_regime_result(self) -> RegimeResult | None:
        return self._last_result

    @property
    def current_adjustment(self) -> StrategyAdjustment | None:
        if self._last_result is None:
            return None
        return self._REGIME_ADJUSTMENTS.get(self._last_result.regime)

    def get_adjustment(self, regime: MarketRegime) -> StrategyAdjustment:
        """Hent strategi-tilpasning for et specifikt regime."""
        return self._REGIME_ADJUSTMENTS[regime]

    def analyze(self, df: pd.DataFrame, **kwargs) -> StrategyResult:
        """
        Analysér markedet med regime-tilpasning.

        Args:
            df: OHLCV DataFrame.
            **kwargs: Ekstra parametre til RegimeDetector.detect().

        Returns:
            StrategyResult med regime-justeret signal og confidence.
        """
        if not self.validate_data(df, min_rows=20):
            return StrategyResult(
                signal=Signal.HOLD,
                confidence=0,
                reason="Utilstrækkelig data for regime-detektion",
            )

        # Detektér regime
        vix = kwargs.get("vix_level")
        breadth = kwargs.get("breadth_ratio")
        yield_spread = kwargs.get("yield_spread")
        use_hmm = kwargs.get("use_hmm", False)

        result = self._detector.detect(
            df, vix_level=vix, breadth_ratio=breadth,
            yield_spread=yield_spread, use_hmm=use_hmm,
        )
        self._last_result = result

        adjustment = self._REGIME_ADJUSTMENTS[result.regime]

        # CRASH: stop alle køb
        if result.regime == MarketRegime.CRASH:
            return StrategyResult(
                signal=Signal.SELL,
                confidence=int(result.confidence),
                reason=(
                    f"CRASH REGIME (score={result.composite_score:+.2f}, "
                    f"confidence={result.confidence:.0f}%) – "
                    f"REDUCÉR EKSPONERING TIL {adjustment.max_exposure_pct:.0%}! "
                    f"{result.reason}"
                ),
            )

        # Brug inner strategy hvis den findes
        if self._inner is not None:
            inner_result = self._inner.analyze(df)
            return self._apply_regime_filter(inner_result, result, adjustment)

        # Ellers: generer signal baseret på regime
        return self._regime_based_signal(result, adjustment, df)

    def _apply_regime_filter(
        self,
        inner: StrategyResult,
        regime: RegimeResult,
        adj: StrategyAdjustment,
    ) -> StrategyResult:
        """
        Filtrér inner strategy's signal med regime-regler.

        - Blokér BUY i CRASH
        - Reducér confidence i BEAR
        - Boost confidence i BULL
        """
        signal = inner.signal
        confidence = inner.confidence

        # Blokér køb i crash/bear (medmindre det er safe havens)
        if signal == Signal.BUY and not adj.allow_new_buys:
            return StrategyResult(
                signal=Signal.HOLD,
                confidence=0,
                reason=f"Køb blokeret af {regime.regime.value.upper()} regime – {inner.reason}",
            )

        # Justér confidence efter regime
        if regime.regime == MarketRegime.BULL:
            if signal == Signal.BUY:
                confidence = min(95, int(confidence * 1.2))
        elif regime.regime == MarketRegime.BEAR:
            if signal == Signal.BUY:
                confidence = max(10, int(confidence * 0.5))
            elif signal == Signal.SELL:
                confidence = min(95, int(confidence * 1.3))
        elif regime.regime == MarketRegime.EUPHORIA:
            if signal == Signal.BUY:
                confidence = max(10, int(confidence * 0.7))

        regime_note = (
            f"[{regime.regime.value.upper()} regime, "
            f"eksponering max {adj.max_exposure_pct:.0%}]"
        )

        return StrategyResult(
            signal=signal,
            confidence=confidence,
            reason=f"{regime_note} {inner.reason}",
        )

    def _regime_based_signal(
        self,
        regime: RegimeResult,
        adj: StrategyAdjustment,
        df: pd.DataFrame,
    ) -> StrategyResult:
        """Generer signal udelukkende baseret på regime."""
        score = regime.composite_score

        if score > 0.3:
            signal = Signal.BUY
        elif score < -0.3:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        # Blokér køb i crash
        if signal == Signal.BUY and not adj.allow_new_buys:
            signal = Signal.HOLD

        return StrategyResult(
            signal=signal,
            confidence=int(regime.confidence),
            reason=(
                f"{regime.label} regime (score={score:+.2f}, "
                f"confidence={regime.confidence:.0f}%) – "
                f"Max eksponering {adj.max_exposure_pct:.0%}. "
                f"{regime.reason}"
            ),
        )

    # ── Utility ──────────────────────────────────────────────

    def get_regime_summary(self) -> dict:
        """Hent opsummering af nuværende regime og tilpasninger."""
        if self._last_result is None:
            return {"regime": "unknown", "adjustments": None}

        result = self._last_result
        adj = self._REGIME_ADJUSTMENTS[result.regime]

        return {
            "regime": result.regime.value,
            "label": result.label,
            "color": result.color,
            "confidence": result.confidence,
            "composite_score": result.composite_score,
            "max_exposure": adj.max_exposure_pct,
            "stop_loss_multiplier": adj.stop_loss_multiplier,
            "preferred_strategies": adj.preferred_strategies,
            "preferred_sectors": adj.preferred_sectors,
            "avoid_sectors": adj.avoid_sectors,
            "safe_havens": adj.safe_havens,
            "allow_new_buys": adj.allow_new_buys,
            "notes": adj.notes,
            "signals": [
                {"name": s.name, "value": s.value, "detail": s.detail}
                for s in result.signals
            ],
            "shifts": [
                {
                    "timestamp": sh.timestamp,
                    "from": sh.from_regime.value,
                    "to": sh.to_regime.value,
                    "confidence": sh.confidence,
                    "reason": sh.reason,
                }
                for sh in self._detector.shifts
            ],
        }
