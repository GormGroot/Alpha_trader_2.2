"""
Aggregated Portfolio — unified portfolio-view på tværs af alle brokers.

Features:
  - Saml positioner fra alle brokers i ét view
  - Merge duplicate symbols (samme aktie hos flere brokers)
  - Konvertér alt til DKK (eller anden base currency)
  - Breakdown: per broker, per asset type, per currency, per land
  - Combined trade history til skat og P&L
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from src.broker.models import AccountInfo
from src.risk.portfolio_tracker import Position


# ── FX Rates ────────────────────────────────────────────────

# Statiske fallback-kurser (bruges hvis yfinance fejler)
_FALLBACK_FX: dict[str, float] = {
    "USD/DKK": 6.85,
    "EUR/DKK": 7.46,
    "GBP/DKK": 8.65,
    "SEK/DKK": 0.65,
    "NOK/DKK": 0.65,
    "CHF/DKK": 7.70,
    "CAD/DKK": 5.05,
    "JPY/DKK": 0.046,
    "HKD/DKK": 0.88,
    "INR/DKK": 0.082,
    "AUD/DKK": 4.45,
    "NZD/DKK": 4.05,
    "SGD/DKK": 5.15,
    "KRW/DKK": 0.005,
    "CNY/DKK": 0.95,
    "BRL/DKK": 1.20,
    "DKK/DKK": 1.0,
}


def get_fx_rate(from_ccy: str, to_ccy: str = "DKK") -> float:
    """
    Hent FX-kurs. Prøver yfinance, falder tilbage til statiske kurser.

    Args:
        from_ccy: Kildecurrency (f.eks. "USD")
        to_ccy: Målcurrency (default "DKK")

    Returns:
        FX-kurs som float.
    """
    from_ccy = from_ccy.upper()
    to_ccy = to_ccy.upper()

    if from_ccy == to_ccy:
        return 1.0

    pair = f"{from_ccy}/{to_ccy}"
    reverse_pair = f"{to_ccy}/{from_ccy}"

    # Prøv yfinance
    try:
        import yfinance as yf
        yf_symbol = f"{from_ccy}{to_ccy}=X"
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period="1d")
        if len(hist) > 0:
            rate = float(hist["Close"].iloc[-1])
            if rate > 0:
                return rate
    except Exception as exc:
        logger.debug(f"[fx] yfinance fejl for {pair}: {exc}")

    # Fallback
    if pair in _FALLBACK_FX:
        return _FALLBACK_FX[pair]
    if reverse_pair in _FALLBACK_FX:
        return 1.0 / _FALLBACK_FX[reverse_pair]

    logger.warning(f"[fx] Ingen kurs for {pair}, bruger 1.0")
    return 1.0


# ── Dataklasser ─────────────────────────────────────────────

@dataclass
class AggregatedPosition:
    """Position med broker-source og DKK-konvertering."""
    symbol: str
    side: str
    qty: float
    entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    currency: str                    # Original currency
    market_value_dkk: float = 0.0   # Konverteret til DKK
    broker_source: str = ""          # Hvilken broker
    asset_type: str = ""             # "stock_us", "stock_nordic", "crypto", etc.
    exchange: str = ""               # Exchange code

    @classmethod
    def from_position(
        cls,
        pos: Position,
        broker_name: str,
        currency: str = "USD",
        fx_rate: float = 1.0,
    ) -> AggregatedPosition:
        """Konvertér en Position til AggregatedPosition."""
        return cls(
            symbol=pos.symbol,
            side=pos.side,
            qty=pos.qty,
            entry_price=pos.entry_price,
            current_price=pos.current_price,
            market_value=pos.market_value,
            cost_basis=pos.cost_basis,
            unrealized_pnl=pos.unrealized_pnl,
            unrealized_pnl_pct=pos.unrealized_pnl_pct,
            currency=currency,
            market_value_dkk=pos.market_value * fx_rate,
            broker_source=broker_name,
        )


@dataclass
class PortfolioSummary:
    """Samlet portfolio-oversigt."""
    total_value_dkk: float
    total_cash_dkk: float
    total_equity_dkk: float
    total_unrealized_pnl_dkk: float
    position_count: int

    # Breakdowns
    by_broker: dict[str, float] = field(default_factory=dict)
    by_asset_type: dict[str, float] = field(default_factory=dict)
    by_currency: dict[str, float] = field(default_factory=dict)
    by_sector: dict[str, float] = field(default_factory=dict)

    # Broker accounts
    broker_accounts: dict[str, dict] = field(default_factory=dict)

    timestamp: str = ""


@dataclass
class Trade:
    """En handel til combined trade history."""
    symbol: str
    side: str           # "buy" / "sell"
    qty: float
    price: float
    value: float        # qty * price
    value_dkk: float
    currency: str
    broker: str
    timestamp: str
    order_id: str = ""
    fees: float = 0.0


# ── Currency Guesser ────────────────────────────────────────

# Broker → default currency mapping
_BROKER_CURRENCIES: dict[str, str] = {
    "alpaca": "USD",
    "ibkr": "USD",     # Multi-currency, men default USD
    "saxo": "DKK",     # Saxo DK account
    "nordnet": "DKK",  # Nordnet DK account
}

# Exchange → currency
_EXCHANGE_CURRENCIES: dict[str, str] = {
    "CSE": "DKK",
    "SFB": "SEK",
    "OSE": "NOK",
    "HEX": "EUR",
    "XETRA": "EUR",
    "SBF": "EUR",
    "AEB": "EUR",
    "LSE": "GBP",
    "EBS": "CHF",
    "MIL": "EUR",
    "BME": "EUR",
    "NYSE": "USD",
    "NASDAQ": "USD",
    # ── Tilføjet: Asia-Pacific, Oceanien, Latinamerika ──
    "TSE": "JPY",       # Tokyo Stock Exchange
    "SEHK": "HKD",      # Hong Kong Stock Exchange
    "NSE": "INR",       # National Stock Exchange India
    "BSE": "INR",       # Bombay Stock Exchange
    "ASX": "AUD",       # Australian Securities Exchange
    "NZX": "NZD",       # New Zealand Exchange
    "SGX": "SGD",       # Singapore Exchange
    "KRX": "KRW",       # Korea Exchange
    "SSE": "CNY",       # Shanghai Stock Exchange
    "SZSE": "CNY",      # Shenzhen Stock Exchange
    "B3": "BRL",        # B3 São Paulo
    "TSX": "CAD",       # Toronto Stock Exchange
}


def guess_currency(symbol: str, broker_name: str = "") -> str:
    """Gæt currency baseret på symbol suffix, exchange og broker."""
    # 1. Prøv exchange-detection
    from src.broker.broker_router import detect_exchange

    exchange = detect_exchange(symbol)
    if exchange and exchange in _EXCHANGE_CURRENCIES:
        return _EXCHANGE_CURRENCIES[exchange]

    # 2. Suffix-baseret fallback (fanger hvad detect_exchange misser)
    #    detect_exchange returnerer None for .T, .HK, .NZ og US-aktier
    _SUFFIX_CURRENCIES = {
        ".T": "JPY",       # Tokyo
        ".HK": "HKD",      # Hong Kong
        ".NS": "INR",      # NSE India
        ".BO": "INR",      # BSE India
        ".AX": "AUD",      # Australia
        ".NZ": "NZD",      # New Zealand
        ".CO": "DKK",      # Copenhagen
        ".ST": "SEK",      # Stockholm
        ".OL": "NOK",      # Oslo
        ".HE": "EUR",      # Helsinki
        ".DE": "EUR",      # Frankfurt/Xetra
        ".PA": "EUR",      # Paris
        ".AS": "EUR",      # Amsterdam
        ".MI": "EUR",      # Milan
        ".MC": "EUR",      # Madrid
        ".L": "GBP",       # London
        ".SW": "CHF",      # Zürich
        ".SI": "SGD",      # Singapore
        ".KS": "KRW",      # Korea
        ".SS": "CNY",      # Shanghai
        ".SZ": "CNY",      # Shenzhen
        ".SA": "BRL",      # São Paulo
        ".TO": "CAD",      # Toronto
    }
    for suffix, ccy in _SUFFIX_CURRENCIES.items():
        if symbol.upper().endswith(suffix.upper()):
            return ccy

    # 3. Broker default
    if broker_name and broker_name in _BROKER_CURRENCIES:
        return _BROKER_CURRENCIES[broker_name]

    # 4. Intet suffix = US stock (AAPL, TSLA, SPY, BTC-USD, ES=F, etc.)
    return "USD"


# ── Aggregated Portfolio ────────────────────────────────────

class AggregatedPortfolio:
    """
    Unified portfolio-view på tværs af alle brokers.

    Brug:
        from src.broker.broker_router import BrokerRouter
        portfolio = AggregatedPortfolio(router)

        # Alle positioner
        positions = portfolio.get_all_positions()

        # Total value i DKK
        summary = portfolio.get_total_value("DKK")
        print(f"Total: {summary.total_value_dkk:,.0f} DKK")

        # Per broker
        for broker, value in summary.by_broker.items():
            print(f"  {broker}: {value:,.0f} DKK")
    """

    # Cache: positions refreshes every 30 seconds
    CACHE_TTL_SECONDS = 30

    def __init__(self, router: Any) -> None:
        """
        Args:
            router: BrokerRouter instance med registrerede brokers.
        """
        self._router = router
        self._cache: dict[str, Any] = {}
        self._cache_time: float = 0.0

    def _is_cache_valid(self) -> bool:
        return (time.time() - self._cache_time) < self.CACHE_TTL_SECONDS

    def invalidate_cache(self) -> None:
        """Force cache invalidation (f.eks. efter en trade)."""
        self._cache_time = 0.0
        self._cache.clear()

    # ── Positions ───────────────────────────────────────────

    def get_all_positions(
        self,
        base_currency: str = "DKK",
        force_refresh: bool = False,
    ) -> list[AggregatedPosition]:
        """
        Hent alle positioner fra alle brokers med DKK-konvertering.

        Args:
            base_currency: Target currency for value conversion.
            force_refresh: Bypass cache.

        Returns:
            List af AggregatedPosition sorteret efter market_value_dkk.
        """
        cache_key = f"positions_{base_currency}"
        if not force_refresh and self._is_cache_valid() and cache_key in self._cache:
            return self._cache[cache_key]

        positions: list[AggregatedPosition] = []
        broker_positions = self._router.get_positions_by_broker()

        for broker_name, broker_pos_list in broker_positions.items():
            for pos in broker_pos_list:
                currency = guess_currency(pos.symbol, broker_name)
                fx_rate = get_fx_rate(currency, base_currency)

                agg = AggregatedPosition.from_position(
                    pos, broker_name, currency, fx_rate
                )
                positions.append(agg)

        # Sortér efter markedsværdi (DKK), størst først
        positions.sort(key=lambda p: p.market_value_dkk, reverse=True)

        # Cache
        self._cache[cache_key] = positions
        self._cache_time = time.time()

        logger.info(
            f"[portfolio] {len(positions)} positioner fra "
            f"{len(broker_positions)} brokers"
        )
        return positions

    def get_merged_positions(
        self,
        base_currency: str = "DKK",
    ) -> list[AggregatedPosition]:
        """
        Merge positioner for samme symbol på tværs af brokers.

        Hvis du har AAPL hos både Alpaca og IBKR, merger den til én.
        """
        all_pos = self.get_all_positions(base_currency)
        merged: dict[str, AggregatedPosition] = {}

        for pos in all_pos:
            key = pos.symbol.upper()
            if key in merged:
                existing = merged[key]
                # Merge: summer qty og values
                total_qty = existing.qty + pos.qty
                total_cost = existing.cost_basis + pos.cost_basis
                total_value = existing.market_value + pos.market_value
                total_value_dkk = existing.market_value_dkk + pos.market_value_dkk
                total_pnl = existing.unrealized_pnl + pos.unrealized_pnl

                merged[key] = AggregatedPosition(
                    symbol=existing.symbol,
                    side=existing.side,
                    qty=total_qty,
                    entry_price=total_cost / total_qty if total_qty > 0 else 0,
                    current_price=pos.current_price,  # Brug seneste pris
                    market_value=total_value,
                    cost_basis=total_cost,
                    unrealized_pnl=total_pnl,
                    unrealized_pnl_pct=(
                        total_pnl / total_cost if abs(total_cost) > 0.01 else 0
                    ),
                    currency=existing.currency,
                    market_value_dkk=total_value_dkk,
                    broker_source=f"{existing.broker_source}+{pos.broker_source}",
                )
            else:
                merged[key] = pos

        result = list(merged.values())
        result.sort(key=lambda p: p.market_value_dkk, reverse=True)
        return result

    # ── Portfolio Summary ───────────────────────────────────

    def get_total_value(
        self,
        base_currency: str = "DKK",
    ) -> PortfolioSummary:
        """
        Samlet portfolio-værdi med breakdowns.

        Returns:
            PortfolioSummary med total value, breakdowns per broker/currency/type.
        """
        positions = self.get_all_positions(base_currency)
        accounts = self._router.get_accounts_by_broker()

        # Beregn totaler
        long_value_dkk = sum(p.market_value_dkk for p in positions if p.side == "long")
        short_value_dkk = sum(p.market_value_dkk for p in positions if p.side == "short")
        total_position_value = long_value_dkk - short_value_dkk
        total_pnl = 0.0
        total_cash_dkk = 0.0

        # Breakdowns
        by_broker: dict[str, float] = {}
        by_currency: dict[str, float] = {}
        by_asset_type: dict[str, float] = {}
        broker_account_info: dict[str, dict] = {}

        # Cash fra alle brokers
        for broker_name, account in accounts.items():
            ccy = account.currency or _BROKER_CURRENCIES.get(broker_name, "USD")
            fx = get_fx_rate(ccy, base_currency)
            cash_dkk = account.cash * fx
            total_cash_dkk += cash_dkk

            broker_account_info[broker_name] = {
                "cash": account.cash,
                "cash_dkk": round(cash_dkk, 2),
                "equity": account.equity,
                "equity_dkk": round(account.equity * fx, 2),
                "currency": ccy,
                "fx_rate": round(fx, 4),
            }

        # Positions breakdown
        for pos in positions:
            # By broker
            by_broker[pos.broker_source] = (
                by_broker.get(pos.broker_source, 0) + pos.market_value_dkk
            )
            # By currency
            by_currency[pos.currency] = (
                by_currency.get(pos.currency, 0) + pos.market_value_dkk
            )
            # By asset type
            atype = pos.asset_type or "unknown"
            by_asset_type[atype] = (
                by_asset_type.get(atype, 0) + pos.market_value_dkk
            )
            # P&L
            total_pnl += pos.unrealized_pnl * get_fx_rate(pos.currency, base_currency)

        # Add cash til by_broker
        for broker_name, acct_info in broker_account_info.items():
            by_broker[broker_name] = (
                by_broker.get(broker_name, 0) + acct_info["cash_dkk"]
            )

        total_value = total_position_value + total_cash_dkk

        return PortfolioSummary(
            total_value_dkk=round(total_value, 2),
            total_cash_dkk=round(total_cash_dkk, 2),
            total_equity_dkk=round(total_position_value, 2),
            total_unrealized_pnl_dkk=round(total_pnl, 2),
            position_count=len(positions),
            by_broker={k: round(v, 2) for k, v in by_broker.items()},
            by_asset_type={k: round(v, 2) for k, v in by_asset_type.items()},
            by_currency={k: round(v, 2) for k, v in by_currency.items()},
            broker_accounts=broker_account_info,
            timestamp=datetime.now().isoformat(),
        )

    # ── Combined Trades ─────────────────────────────────────

    def get_combined_trades(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Trade]:
        """
        Kombineret trade-historik fra alle brokers.

        TODO: Implementeres når individuelle broker-integrationer
        har get_trade_history() metoder (T1-T3).

        Returns:
            Sorteret liste af trades (nyeste først).
        """
        logger.info(
            "[portfolio] get_combined_trades() — venter på T1-T3 broker impls"
        )
        return []

    # ── Convenience ─────────────────────────────────────────

    def top_positions(self, n: int = 10, base_currency: str = "DKK") -> list[dict]:
        """Top N positioner by value."""
        positions = self.get_merged_positions(base_currency)[:n]
        return [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "value_dkk": round(p.market_value_dkk, 2),
                "pnl_pct": round(p.unrealized_pnl_pct * 100, 2),
                "broker": p.broker_source,
                "currency": p.currency,
            }
            for p in positions
        ]

    def allocation_breakdown(self, base_currency: str = "DKK") -> dict:
        """Asset allocation i procent."""
        summary = self.get_total_value(base_currency)
        total = summary.total_value_dkk

        if total <= 0:
            return {"cash_pct": 100.0, "positions": {}}

        return {
            "cash_pct": round(summary.total_cash_dkk / total * 100, 1),
            "equity_pct": round(summary.total_equity_dkk / total * 100, 1),
            "by_broker_pct": {
                k: round(v / total * 100, 1)
                for k, v in summary.by_broker.items()
            },
            "by_currency_pct": {
                k: round(v / total * 100, 1)
                for k, v in summary.by_currency.items()
            },
        }
