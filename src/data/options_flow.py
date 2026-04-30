"""
Options Flow Module – Unusual Options Activity, Put/Call Ratio, Max Pain, IV Analyse.

Funktionalitet:
  - Unusual Options Activity (UOA): detektér store/usædvanlige handler
  - Put/Call ratio: per aktie + samlet marked (contrarian indikator)
  - Max Pain: beregn options expiration price magnet
  - Implied Volatility: IV Rank, IV Percentile, IV vs HV
  - SQLite-cache for alle data

Datakilde: yfinance options chains (gratis, ingen API-nøgle).
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from config.settings import settings

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False
    logger.warning("yfinance ikke installeret – options data utilgængelig")


# ── Konstanter ───────────────────────────────────────────────

_MIN_REQUEST_GAP = 0.40          # Rate limiting (sekunder)
UOA_VOLUME_MULTIPLIER = 5.0     # Volume > 5x normal = usædvanlig
UOA_MIN_PREMIUM = 50_000        # Minimum premium i USD for UOA
BLOCK_TRADE_THRESHOLD = 1_000_000  # $1M+ = block trade
DEFAULT_IV_LOOKBACK = 252        # 1 år (handelsdage) for HV beregning
DEFAULT_PCR_LOOKBACK = 20        # 20 dage for glidende put/call ratio


# ── Dataclasses ──────────────────────────────────────────────

@dataclass
class UnusualOption:
    """Én usædvanlig options-handel."""
    symbol: str
    expiration: str              # "2026-04-17"
    strike: float
    option_type: str             # "call" eller "put"
    volume: int
    open_interest: int
    implied_volatility: float
    last_price: float
    bid: float
    ask: float
    premium_total: float         # volume * last_price * 100
    volume_oi_ratio: float       # volume / open_interest
    volume_vs_normal: float      # volume / avg volume
    is_block_trade: bool         # premium > $1M
    in_the_money: bool
    detected_at: datetime = field(default_factory=datetime.now)

    @property
    def signal(self) -> str:
        """Bullish/bearish baseret på option type og volume."""
        if self.option_type == "call" and self.volume_vs_normal > UOA_VOLUME_MULTIPLIER:
            return "BULLISH"
        elif self.option_type == "put" and self.volume_vs_normal > UOA_VOLUME_MULTIPLIER:
            return "BEARISH"
        return "NEUTRAL"

    @property
    def alert_text(self) -> str:
        """Menneskelæselig alert-tekst."""
        emoji = "🟢" if self.option_type == "call" else "🔴"
        itm = " (ITM)" if self.in_the_money else ""
        return (
            f"{emoji} {self.symbol}: Usædvanlig {self.option_type.upper()}-volume "
            f"${self.strike:.0f} exp {self.expiration}{itm} — "
            f"{self.volume_vs_normal:.1f}x normalt, premium ${self.premium_total:,.0f}"
        )


@dataclass
class PutCallRatio:
    """Put/Call ratio for et symbol eller samlet marked."""
    symbol: str                  # "AAPL" eller "MARKET"
    put_volume: int
    call_volume: int
    ratio: float                 # put_vol / call_vol
    put_oi: int                  # Open interest puts
    call_oi: int                 # Open interest calls
    oi_ratio: float              # put_oi / call_oi
    signal: str                  # "bullish", "bearish", "neutral"
    date: datetime = field(default_factory=datetime.now)

    @property
    def interpretation(self) -> str:
        """Contrarian tolkning af put/call ratio."""
        if self.ratio > 1.2:
            return "Høj P/C ratio – markedet er nervøst (contrarian BULLISH)"
        elif self.ratio > 0.9:
            return "Normal P/C ratio – neutral sentiment"
        elif self.ratio > 0.6:
            return "Lav P/C ratio – markedet er selvsikkert (contrarian BEARISH)"
        return "Meget lav P/C ratio – ekstremt bullish marked (contrarian WARNING)"


@dataclass
class MaxPainResult:
    """Max Pain analyse for et options expiration."""
    symbol: str
    expiration: str
    max_pain_price: float        # Pris med mindst smerte for options-sælgere
    current_price: float
    distance_pct: float          # % afstand fra current til max pain
    call_oi_by_strike: dict[float, int] = field(default_factory=dict)
    put_oi_by_strike: dict[float, int] = field(default_factory=dict)
    total_pain_by_strike: dict[float, float] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        """Retning aktien forventes at bevæge sig mod max pain."""
        if self.distance_pct > 2.0:
            return "NED mod max pain"
        elif self.distance_pct < -2.0:
            return "OP mod max pain"
        return "Tæt på max pain"


@dataclass
class IVAnalysis:
    """Implied Volatility analyse for et symbol."""
    symbol: str
    current_iv: float            # Nuværende gennemsnitlig IV
    historical_vol: float        # Historisk volatilitet (annualiseret)
    iv_rank: float               # 0–100: IV placering i forhold til seneste år
    iv_percentile: float         # 0–100: % af dage med lavere IV
    iv_hv_ratio: float           # IV / HV – over 1.0 = IV er "dyr"
    iv_high_52w: float           # Højeste IV seneste 52 uger
    iv_low_52w: float            # Laveste IV seneste 52 uger
    is_elevated: bool            # IV Rank > 50
    date: datetime = field(default_factory=datetime.now)

    @property
    def alert_text(self) -> str | None:
        """Alert-tekst hvis IV er usædvanligt høj."""
        if self.iv_rank >= 90:
            return (
                f"🔥 {self.symbol} IV Rank er {self.iv_rank:.0f}% — "
                f"markedet forventer stor bevægelse"
            )
        if self.iv_rank >= 70:
            return (
                f"⚡ {self.symbol} IV Rank er {self.iv_rank:.0f}% — "
                f"forhøjet forventning til bevægelse"
            )
        return None

    @property
    def interpretation(self) -> str:
        """Menneskelæselig tolkning."""
        if self.iv_rank >= 80:
            return "Meget høj IV – overvej at sælge premium (covered calls, iron condors)"
        elif self.iv_rank >= 50:
            return "Moderat IV – normal optionsmarked"
        elif self.iv_rank >= 20:
            return "Lav IV – overvej at købe premium (long calls/puts)"
        return "Meget lav IV – markedet forventer minimal bevægelse"


@dataclass
class OptionsFlowSummary:
    """Samlet options flow rapport for et symbol."""
    symbol: str
    unusual_activity: list[UnusualOption]
    put_call_ratio: PutCallRatio | None
    max_pain: MaxPainResult | None
    iv_analysis: IVAnalysis | None
    overall_signal: str = "neutral"   # "bullish", "bearish", "neutral"
    alerts: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def confidence_adjustment(self) -> int:
        """Confidence-justering for strategier (−10 til +10 points)."""
        adj = 0
        # Usædvanlig call-volume = bullish
        bullish_uoa = sum(1 for u in self.unusual_activity if u.signal == "BULLISH")
        bearish_uoa = sum(1 for u in self.unusual_activity if u.signal == "BEARISH")
        if bullish_uoa > bearish_uoa:
            adj += min(bullish_uoa * 3, 8)
        elif bearish_uoa > bullish_uoa:
            adj -= min(bearish_uoa * 3, 8)

        # Put/call ratio contrarian signal
        if self.put_call_ratio and self.put_call_ratio.ratio > 1.3:
            adj += 3   # Contrarian bullish
        elif self.put_call_ratio and self.put_call_ratio.ratio < 0.5:
            adj -= 3   # Contrarian bearish

        return max(-10, min(10, adj))


# ── Options Flow Tracker ─────────────────────────────────────

class OptionsFlowTracker:
    """
    Tracker for options flow data.

    Henter options chains via yfinance, beregner UOA, P/C ratio,
    max pain og IV analyse. Cacher alt i SQLite.
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        self._last_request_time: float = 0.0

        cache_path = Path(cache_dir or settings.market_data.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self._db_path = cache_path / "options_flow.db"
        self._init_db()

    def _throttle(self) -> None:
        """Rate limiting."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < _MIN_REQUEST_GAP:
            time.sleep(_MIN_REQUEST_GAP - elapsed)
        self._last_request_time = time.monotonic()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        """Opret cache-tabeller."""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS unusual_options (
                    symbol          TEXT NOT NULL,
                    expiration      TEXT NOT NULL,
                    strike          REAL NOT NULL,
                    option_type     TEXT NOT NULL,
                    volume          INTEGER,
                    open_interest   INTEGER,
                    implied_vol     REAL,
                    last_price      REAL,
                    premium_total   REAL,
                    volume_vs_normal REAL,
                    is_block_trade  INTEGER DEFAULT 0,
                    in_the_money    INTEGER DEFAULT 0,
                    detected_at     TEXT NOT NULL,
                    fetched_at      TEXT NOT NULL,
                    PRIMARY KEY (symbol, expiration, strike, option_type, detected_at)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS put_call_ratio (
                    symbol          TEXT NOT NULL,
                    put_volume      INTEGER,
                    call_volume     INTEGER,
                    ratio           REAL,
                    put_oi          INTEGER,
                    call_oi         INTEGER,
                    oi_ratio        REAL,
                    signal          TEXT,
                    date            TEXT NOT NULL,
                    fetched_at      TEXT NOT NULL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS iv_history (
                    symbol          TEXT NOT NULL,
                    current_iv      REAL,
                    historical_vol  REAL,
                    iv_rank         REAL,
                    iv_percentile   REAL,
                    iv_hv_ratio     REAL,
                    iv_high_52w     REAL,
                    iv_low_52w      REAL,
                    date            TEXT NOT NULL,
                    fetched_at      TEXT NOT NULL,
                    PRIMARY KEY (symbol, date)
                )
            """)

    # ── Options Chain Hentning ────────────────────────────────

    def _get_options_chain(
        self, symbol: str, expiration: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, float] | None:
        """
        Hent options chain for et symbol via yfinance.

        Returns:
            (calls_df, puts_df, current_price) eller None ved fejl.
        """
        if not _HAS_YF:
            logger.warning("yfinance ikke tilgængelig")
            return None

        try:
            self._throttle()
            ticker = yf.Ticker(symbol)

            # Hent nuværende pris
            info = ticker.fast_info
            current_price = info.get("lastPrice") or info.get("last_price", 0)
            if not current_price:
                hist = ticker.history(period="1d")
                current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0

            # Hent expirations
            expirations = ticker.options
            if not expirations:
                logger.warning(f"[options] Ingen options fundet for {symbol}")
                return None

            # Vælg expiration
            if expiration and expiration in expirations:
                exp = expiration
            else:
                # Brug nærmeste expiration (mindst 7 dage ud)
                now = datetime.now()
                exp = None
                for e in expirations:
                    exp_date = datetime.strptime(e, "%Y-%m-%d")
                    if exp_date - now >= timedelta(days=7):
                        exp = e
                        break
                if not exp:
                    exp = expirations[0]

            # Hent chain
            self._throttle()
            chain = ticker.option_chain(exp)
            return chain.calls, chain.puts, float(current_price)

        except Exception as exc:
            logger.error(f"[options] Fejl ved hentning for {symbol}: {exc}")
            return None

    def _get_all_chains(
        self, symbol: str,
    ) -> list[tuple[str, pd.DataFrame, pd.DataFrame]] | None:
        """Hent alle options chains for et symbol (alle expirations)."""
        if not _HAS_YF:
            return None

        try:
            self._throttle()
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                return None

            chains = []
            # Begræns til de næste 4 expirations for performance
            for exp in expirations[:4]:
                self._throttle()
                try:
                    chain = ticker.option_chain(exp)
                    chains.append((exp, chain.calls, chain.puts))
                except Exception:
                    continue

            return chains if chains else None

        except Exception as exc:
            logger.error(f"[options] Fejl ved all chains for {symbol}: {exc}")
            return None

    # ── Unusual Options Activity ──────────────────────────────

    def detect_unusual_activity(
        self,
        symbol: str,
        volume_multiplier: float = UOA_VOLUME_MULTIPLIER,
        min_premium: float = UOA_MIN_PREMIUM,
    ) -> list[UnusualOption]:
        """
        Detektér usædvanlige options-handler for et symbol.

        Kriterier:
          - Volume > volume_multiplier × open interest
          - Premium > min_premium USD
          - Block trades > $1M

        Args:
            symbol: Ticker symbol.
            volume_multiplier: Multiplier for at definere "usædvanlig".
            min_premium: Minimum samlet premium i USD.

        Returns:
            Sorteret liste af UnusualOption (største premium først).
        """
        symbol = symbol.upper()
        result = self._get_options_chain(symbol)
        if not result:
            return []

        calls, puts, current_price = result
        unusual: list[UnusualOption] = []

        for opt_type, df in [("call", calls), ("put", puts)]:
            if df.empty:
                continue

            for _, row in df.iterrows():
                volume = int(row.get("volume", 0) or 0)
                oi = int(row.get("openInterest", 0) or 0)
                last_price = float(row.get("lastPrice", 0) or 0)
                iv = float(row.get("impliedVolatility", 0) or 0)
                bid = float(row.get("bid", 0) or 0)
                ask = float(row.get("ask", 0) or 0)
                strike = float(row.get("strike", 0) or 0)
                itm = bool(row.get("inTheMoney", False))

                if volume == 0:
                    continue

                # Premium = volume × last_price × 100 (1 kontrakt = 100 aktier)
                premium = volume * last_price * 100

                # Volume vs normal (brug OI som proxy for normal)
                vol_vs_normal = volume / max(oi, 1)

                # Filtrer
                if premium < min_premium:
                    continue
                if vol_vs_normal < volume_multiplier and premium < BLOCK_TRADE_THRESHOLD:
                    continue

                is_block = premium >= BLOCK_TRADE_THRESHOLD

                unusual.append(UnusualOption(
                    symbol=symbol,
                    expiration=str(row.get("contractSymbol", ""))[:17] if "contractSymbol" in row else "",
                    strike=strike,
                    option_type=opt_type,
                    volume=volume,
                    open_interest=oi,
                    implied_volatility=iv,
                    last_price=last_price,
                    bid=bid,
                    ask=ask,
                    premium_total=premium,
                    volume_oi_ratio=vol_vs_normal,
                    volume_vs_normal=vol_vs_normal,
                    is_block_trade=is_block,
                    in_the_money=itm,
                ))

        # Cache
        if unusual:
            self._write_uoa_cache(unusual)

        return sorted(unusual, key=lambda u: u.premium_total, reverse=True)

    def _write_uoa_cache(self, options: list[UnusualOption]) -> None:
        """Gem UOA til cache."""
        now = datetime.now().isoformat()
        rows = [
            (
                o.symbol, o.expiration, o.strike, o.option_type,
                o.volume, o.open_interest, o.implied_volatility,
                o.last_price, o.premium_total, o.volume_vs_normal,
                int(o.is_block_trade), int(o.in_the_money),
                o.detected_at.isoformat(), now,
            )
            for o in options
        ]
        with self._get_conn() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO unusual_options
                   (symbol, expiration, strike, option_type,
                    volume, open_interest, implied_vol,
                    last_price, premium_total, volume_vs_normal,
                    is_block_trade, in_the_money, detected_at, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        logger.debug(f"[options] Cached {len(rows)} UOA entries")

    def get_recent_uoa(self, symbol: str, hours: int = 24) -> list[UnusualOption]:
        """Hent seneste UOA fra cache."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT symbol, expiration, strike, option_type,
                          volume, open_interest, implied_vol,
                          last_price, premium_total, volume_vs_normal,
                          is_block_trade, in_the_money, detected_at
                   FROM unusual_options
                   WHERE symbol = ? AND fetched_at >= ?
                   ORDER BY premium_total DESC""",
                (symbol.upper(), cutoff),
            ).fetchall()

        return [
            UnusualOption(
                symbol=r[0], expiration=r[1], strike=r[2], option_type=r[3],
                volume=r[4], open_interest=r[5], implied_volatility=r[6],
                last_price=r[7], bid=0, ask=0, premium_total=r[8],
                volume_oi_ratio=r[9], volume_vs_normal=r[9],
                is_block_trade=bool(r[10]), in_the_money=bool(r[11]),
                detected_at=datetime.fromisoformat(r[12]) if r[12] else datetime.now(),
            )
            for r in rows
        ]

    # ── Put/Call Ratio ────────────────────────────────────────

    def get_put_call_ratio(self, symbol: str) -> PutCallRatio | None:
        """
        Beregn put/call ratio for et symbol.

        Summerer volume og OI for alle expirations.
        Contrarian tolkning:
          - Ratio > 1.2 → markedet er nervøst → contrarian BULLISH
          - Ratio < 0.6 → markedet er selvsikkert → contrarian BEARISH

        Returns:
            PutCallRatio eller None.
        """
        symbol = symbol.upper()
        chains = self._get_all_chains(symbol)
        if not chains:
            return None

        total_call_vol = 0
        total_put_vol = 0
        total_call_oi = 0
        total_put_oi = 0

        for _exp, calls, puts in chains:
            if not calls.empty:
                total_call_vol += int(calls["volume"].fillna(0).sum())
                total_call_oi += int(calls["openInterest"].fillna(0).sum())
            if not puts.empty:
                total_put_vol += int(puts["volume"].fillna(0).sum())
                total_put_oi += int(puts["openInterest"].fillna(0).sum())

        if total_call_vol == 0:
            ratio = float("inf") if total_put_vol > 0 else 0.0
        else:
            ratio = total_put_vol / total_call_vol

        if total_call_oi == 0:
            oi_ratio = 0.0
        else:
            oi_ratio = total_put_oi / total_call_oi

        # Signal (contrarian)
        if ratio > 1.2:
            signal = "bullish"     # Contrarian: høj P/C → folk er bange → bund nær
        elif ratio < 0.6:
            signal = "bearish"     # Contrarian: lav P/C → for complacent → top nær
        else:
            signal = "neutral"

        pcr = PutCallRatio(
            symbol=symbol,
            put_volume=total_put_vol,
            call_volume=total_call_vol,
            ratio=round(ratio, 3) if ratio != float("inf") else 99.0,
            put_oi=total_put_oi,
            call_oi=total_call_oi,
            oi_ratio=round(oi_ratio, 3),
            signal=signal,
        )

        # Cache
        self._write_pcr_cache(pcr)
        return pcr

    def _write_pcr_cache(self, pcr: PutCallRatio) -> None:
        """Gem put/call ratio til cache."""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO put_call_ratio
                   (symbol, put_volume, call_volume, ratio,
                    put_oi, call_oi, oi_ratio, signal, date, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    pcr.symbol, pcr.put_volume, pcr.call_volume, pcr.ratio,
                    pcr.put_oi, pcr.call_oi, pcr.oi_ratio, pcr.signal,
                    pcr.date.strftime("%Y-%m-%d"), now,
                ),
            )

    def get_market_pcr(self, spy_symbol: str = "SPY") -> PutCallRatio | None:
        """Hent samlet marked put/call ratio via SPY options."""
        return self.get_put_call_ratio(spy_symbol)

    # ── Max Pain ──────────────────────────────────────────────

    def calculate_max_pain(
        self, symbol: str, expiration: str | None = None,
    ) -> MaxPainResult | None:
        """
        Beregn Max Pain pris for en given expiration.

        Max Pain er den strike-pris hvor den samlede værdi af
        udestående call og put options har mindst intrinsic value.
        Aktier trækkes ofte mod Max Pain nær expiration.

        Args:
            symbol: Ticker symbol.
            expiration: Options expiration dato. None = nærmeste.

        Returns:
            MaxPainResult eller None.
        """
        symbol = symbol.upper()
        result = self._get_options_chain(symbol, expiration)
        if not result:
            return None

        calls, puts, current_price = result

        if calls.empty or puts.empty:
            return None

        # Saml OI per strike
        call_oi: dict[float, int] = {}
        put_oi: dict[float, int] = {}

        for _, row in calls.iterrows():
            strike = float(row.get("strike", 0))
            oi = int(row.get("openInterest", 0) or 0)
            call_oi[strike] = call_oi.get(strike, 0) + oi

        for _, row in puts.iterrows():
            strike = float(row.get("strike", 0))
            oi = int(row.get("openInterest", 0) or 0)
            put_oi[strike] = put_oi.get(strike, 0) + oi

        # Alle unikke strikes
        all_strikes = sorted(set(list(call_oi.keys()) + list(put_oi.keys())))

        if not all_strikes:
            return None

        # Beregn total pain per strike
        # Pain = sum of (intrinsic value × OI) for alle options
        pain_by_strike: dict[float, float] = {}

        for test_price in all_strikes:
            total_pain = 0.0

            # Pain for call holders (de taber når prisen er under deres strike)
            for strike, oi in call_oi.items():
                if test_price > strike:
                    total_pain += (test_price - strike) * oi * 100
                # Calls under test_price udløber worthless → ingen yderligere pain

            # Pain for put holders (de taber når prisen er over deres strike)
            for strike, oi in put_oi.items():
                if test_price < strike:
                    total_pain += (strike - test_price) * oi * 100

            pain_by_strike[test_price] = total_pain

        # Max pain = strike med MINDST total pain
        max_pain_price = min(pain_by_strike, key=pain_by_strike.get)  # type: ignore
        distance_pct = ((current_price - max_pain_price) / max_pain_price * 100
                        if max_pain_price > 0 else 0.0)

        # Find expiration fra chain
        exp_str = expiration or "nearest"

        return MaxPainResult(
            symbol=symbol,
            expiration=exp_str,
            max_pain_price=max_pain_price,
            current_price=current_price,
            distance_pct=round(distance_pct, 2),
            call_oi_by_strike=call_oi,
            put_oi_by_strike=put_oi,
            total_pain_by_strike=pain_by_strike,
        )

    # ── Implied Volatility Analyse ────────────────────────────

    def analyze_iv(
        self,
        symbol: str,
        lookback_days: int = DEFAULT_IV_LOOKBACK,
    ) -> IVAnalysis | None:
        """
        Analysér Implied Volatility for et symbol.

        Beregner:
          - Nuværende IV (gennemsnit af ATM options)
          - Historisk volatilitet (annualiseret)
          - IV Rank: (current - 52w_low) / (52w_high - 52w_low) × 100
          - IV Percentile: % af dage seneste år med lavere IV

        Args:
            symbol: Ticker symbol.
            lookback_days: Antal dage for HV beregning.

        Returns:
            IVAnalysis eller None.
        """
        symbol = symbol.upper()

        if not _HAS_YF:
            return None

        try:
            # Hent options for nuværende IV
            result = self._get_options_chain(symbol)
            if not result:
                return None

            calls, puts, current_price = result

            # Beregn nuværende IV (gennemsnit af ATM options)
            current_iv = self._calc_current_iv(calls, puts, current_price)

            # Hent historisk data for HV
            self._throttle()
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=f"{lookback_days}d")

            if hist.empty or len(hist) < 30:
                return None

            # Historisk volatilitet (annualiseret)
            returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
            hv = float(returns.std() * np.sqrt(252))

            # IV historik fra cache for IV Rank beregning
            iv_history = self._get_iv_history(symbol)

            if iv_history and len(iv_history) >= 10:
                iv_values = [v for v in iv_history if v > 0]
                iv_high = max(iv_values) if iv_values else current_iv
                iv_low = min(iv_values) if iv_values else current_iv

                # IV Rank
                iv_range = iv_high - iv_low
                if iv_range > 0:
                    iv_rank = (current_iv - iv_low) / iv_range * 100
                else:
                    iv_rank = 50.0

                # IV Percentile
                below = sum(1 for v in iv_values if v < current_iv)
                iv_percentile = below / len(iv_values) * 100
            else:
                iv_high = current_iv * 1.5
                iv_low = current_iv * 0.5
                iv_rank = 50.0
                iv_percentile = 50.0

            # IV / HV ratio
            iv_hv_ratio = current_iv / hv if hv > 0 else 1.0

            analysis = IVAnalysis(
                symbol=symbol,
                current_iv=round(current_iv, 4),
                historical_vol=round(hv, 4),
                iv_rank=round(max(0, min(100, iv_rank)), 1),
                iv_percentile=round(max(0, min(100, iv_percentile)), 1),
                iv_hv_ratio=round(iv_hv_ratio, 2),
                iv_high_52w=round(iv_high, 4),
                iv_low_52w=round(iv_low, 4),
                is_elevated=iv_rank > 50,
            )

            # Cache
            self._write_iv_cache(analysis)
            return analysis

        except Exception as exc:
            logger.error(f"[options] IV analyse fejl for {symbol}: {exc}")
            return None

    @staticmethod
    def _calc_current_iv(
        calls: pd.DataFrame, puts: pd.DataFrame, current_price: float,
    ) -> float:
        """
        Beregn nuværende IV som gennemsnit af near-the-money options.

        Vælger options med strike ±10% af current_price.
        """
        ivs: list[float] = []

        low = current_price * 0.90
        high = current_price * 1.10

        for df in [calls, puts]:
            if df.empty:
                continue
            mask = (df["strike"] >= low) & (df["strike"] <= high)
            near_money = df[mask]
            if not near_money.empty:
                iv_vals = near_money["impliedVolatility"].dropna()
                ivs.extend(iv_vals.tolist())

        if not ivs:
            # Fallback: brug alle options
            for df in [calls, puts]:
                if not df.empty:
                    iv_vals = df["impliedVolatility"].dropna()
                    ivs.extend(iv_vals.tolist())

        return float(np.mean(ivs)) if ivs else 0.0

    def _get_iv_history(self, symbol: str) -> list[float]:
        """Hent IV historik fra cache."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT current_iv FROM iv_history
                   WHERE symbol = ? ORDER BY date DESC LIMIT 252""",
                (symbol,),
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    def _write_iv_cache(self, iv: IVAnalysis) -> None:
        """Gem IV analyse til cache."""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO iv_history
                   (symbol, current_iv, historical_vol, iv_rank,
                    iv_percentile, iv_hv_ratio, iv_high_52w, iv_low_52w,
                    date, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    iv.symbol, iv.current_iv, iv.historical_vol,
                    iv.iv_rank, iv.iv_percentile, iv.iv_hv_ratio,
                    iv.iv_high_52w, iv.iv_low_52w,
                    iv.date.strftime("%Y-%m-%d"), now,
                ),
            )

    # ── Full Report ───────────────────────────────────────────

    def get_options_flow_summary(self, symbol: str) -> OptionsFlowSummary:
        """
        Generér komplet options flow rapport for et symbol.

        Kombinerer UOA, P/C ratio, max pain og IV analyse.
        """
        alerts: list[str] = []

        # Early-exit: tjek om symbolet har options overhovedet (undgå 4x redundante API-kald)
        if _HAS_YF:
            try:
                self._throttle()
                ticker = yf.Ticker(symbol.upper())
                if not ticker.options:
                    logger.debug(f"[options] {symbol} har ingen options — springer over")
                    return OptionsFlowSummary(
                        symbol=symbol.upper(),
                        unusual_activity=[],
                        put_call_ratio=None,
                        max_pain=None,
                        iv_analysis=None,
                    )
            except Exception:
                return OptionsFlowSummary(
                    symbol=symbol.upper(),
                    unusual_activity=[],
                    put_call_ratio=None,
                    max_pain=None,
                    iv_analysis=None,
                )

        # 1. Unusual activity
        try:
            uoa = self.detect_unusual_activity(symbol)
            for u in uoa[:5]:  # Top 5 alerts
                alerts.append(u.alert_text)
        except Exception as exc:
            logger.error(f"[options] UOA fejl for {symbol}: {exc}")
            uoa = []

        # 2. Put/Call ratio
        try:
            pcr = self.get_put_call_ratio(symbol)
            if pcr:
                alerts.append(
                    f"📊 P/C Ratio: {pcr.ratio:.2f} – {pcr.interpretation}"
                )
        except Exception as exc:
            logger.error(f"[options] PCR fejl for {symbol}: {exc}")
            pcr = None

        # 3. Max Pain
        try:
            mp = self.calculate_max_pain(symbol)
            if mp:
                alerts.append(
                    f"🎯 Max Pain: ${mp.max_pain_price:.2f} "
                    f"({mp.distance_pct:+.1f}% fra nuværende ${mp.current_price:.2f}) — "
                    f"{mp.direction}"
                )
        except Exception as exc:
            logger.error(f"[options] Max Pain fejl for {symbol}: {exc}")
            mp = None

        # 4. IV analyse
        try:
            iv = self.analyze_iv(symbol)
            if iv and iv.alert_text:
                alerts.append(iv.alert_text)
        except Exception as exc:
            logger.error(f"[options] IV fejl for {symbol}: {exc}")
            iv = None

        # Samlet signal
        summary = OptionsFlowSummary(
            symbol=symbol.upper(),
            unusual_activity=uoa,
            put_call_ratio=pcr,
            max_pain=mp,
            iv_analysis=iv,
            alerts=alerts,
        )

        # Bestem overall signal
        adj = summary.confidence_adjustment
        if adj > 3:
            summary.overall_signal = "bullish"
        elif adj < -3:
            summary.overall_signal = "bearish"
        else:
            summary.overall_signal = "neutral"

        return summary

    # ── Batch Operations ──────────────────────────────────────

    def scan_symbols(
        self, symbols: list[str],
    ) -> dict[str, OptionsFlowSummary]:
        """Scan flere symboler for options flow."""
        results: dict[str, OptionsFlowSummary] = {}

        for symbol in symbols:
            try:
                results[symbol] = self.get_options_flow_summary(symbol)
                logger.info(f"[options] Scannet {symbol}: {results[symbol].overall_signal}")
            except Exception as exc:
                logger.error(f"[options] Scan fejl for {symbol}: {exc}")

        return results

    def get_iv_ranking(
        self, symbols: list[str],
    ) -> list[tuple[str, float]]:
        """
        Rangér symboler efter IV Rank (højeste først).

        Nyttigt til at finde aktier med forventet stor bevægelse.

        Returns:
            Sorteret liste af (symbol, iv_rank).
        """
        rankings: list[tuple[str, float]] = []

        for symbol in symbols:
            try:
                iv = self.analyze_iv(symbol)
                if iv:
                    rankings.append((symbol, iv.iv_rank))
            except Exception:
                continue

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    # ── Strategy Integration ──────────────────────────────────

    def get_confidence_adjustment(self, symbol: str) -> int:
        """
        Beregn confidence-justering baseret på options flow.

        Returns:
            -10 til +10 points.
        """
        summary = self.get_options_flow_summary(symbol)
        return summary.confidence_adjustment

    # ── Explain ───────────────────────────────────────────────

    def explain(self, symbol: str) -> str:
        """Forklar options flow i simple termer."""
        summary = self.get_options_flow_summary(symbol)
        lines = [
            f"═══ OPTIONS FLOW RAPPORT: {symbol.upper()} ═══",
            "",
        ]

        # UOA
        if summary.unusual_activity:
            lines.append(f"🔔 USÆDVANLIG OPTIONS AKTIVITET ({len(summary.unusual_activity)} fund)")
            for u in summary.unusual_activity[:5]:
                block = " [BLOCK TRADE]" if u.is_block_trade else ""
                lines.append(
                    f"   {u.option_type.upper()} ${u.strike:.0f} — "
                    f"vol {u.volume:,} ({u.volume_vs_normal:.1f}x), "
                    f"prem ${u.premium_total:,.0f}{block}"
                )
            lines.append("")

        # Put/Call Ratio
        if summary.put_call_ratio:
            pcr = summary.put_call_ratio
            lines.append("📊 PUT/CALL RATIO")
            lines.append(f"   Ratio: {pcr.ratio:.2f} (volume-baseret)")
            lines.append(f"   OI Ratio: {pcr.oi_ratio:.2f} (open interest)")
            lines.append(f"   Put vol: {pcr.put_volume:,} | Call vol: {pcr.call_volume:,}")
            lines.append(f"   {pcr.interpretation}")
            lines.append("")

        # Max Pain
        if summary.max_pain:
            mp = summary.max_pain
            lines.append("🎯 MAX PAIN")
            lines.append(f"   Max Pain pris: ${mp.max_pain_price:.2f}")
            lines.append(f"   Nuværende pris: ${mp.current_price:.2f}")
            lines.append(f"   Afstand: {mp.distance_pct:+.1f}% — {mp.direction}")
            lines.append("")

        # IV analyse
        if summary.iv_analysis:
            iv = summary.iv_analysis
            lines.append("📈 IMPLIED VOLATILITY")
            lines.append(f"   Nuværende IV: {iv.current_iv:.1%}")
            lines.append(f"   Historisk Vol: {iv.historical_vol:.1%}")
            lines.append(f"   IV Rank: {iv.iv_rank:.0f}% | IV Percentile: {iv.iv_percentile:.0f}%")
            lines.append(f"   IV/HV Ratio: {iv.iv_hv_ratio:.2f}")
            lines.append(f"   52-ugers range: {iv.iv_low_52w:.1%} – {iv.iv_high_52w:.1%}")
            lines.append(f"   {iv.interpretation}")
            lines.append("")

        # Samlet
        lines.append("📋 SAMLET VURDERING")
        lines.append(f"   Signal: {summary.overall_signal.upper()}")
        lines.append(f"   Confidence justering: {summary.confidence_adjustment:+d} points")

        return "\n".join(lines)

    def print_report(self, symbol: str) -> None:
        """Print options flow rapport til konsollen."""
        print(self.explain(symbol))
