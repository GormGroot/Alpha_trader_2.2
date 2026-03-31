"""
BrokerRouter — unified interface til alle fire brokers.

Routing-logik:
  1. Exact symbol match (BTC-USD → alpaca)
  2. Exchange detection fra symbol suffix (.CO → nordnet, .DE → ibkr)
  3. Asset type routing (forex → ibkr, crypto → alpaca)
  4. Fallback chain: ibkr → saxo → alpaca

Alle BaseBroker-metoder delegerer til den broker der resolver for symbolet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from src.broker.base_broker import BaseBroker
from src.broker.models import (
    AccountInfo,
    BrokerError,
    Order,
    OrderType,
    OrderValidationError,
)
from src.risk.portfolio_tracker import Position


# ── Exceptions ──────────────────────────────────────────────

class RoutingError(BrokerError):
    """Ingen broker kan håndtere dette symbol/asset type."""


# ── Routing Configuration ───────────────────────────────────

@dataclass
class RoutingConfig:
    """Routing-konfiguration — kan loades fra YAML/config."""

    # Exact symbol → broker
    exact_matches: dict[str, str] = field(default_factory=lambda: {
        "BTC-USD": "alpaca",
        "ETH-USD": "alpaca",
        "BTC/USD": "alpaca",
        "ETH/USD": "alpaca",
        "BTCUSD": "alpaca",
        "ETHUSD": "alpaca",
        "DOGE-USD": "alpaca",
        "SOL-USD": "alpaca",
        "AVAX-USD": "alpaca",
        "LINK-USD": "alpaca",
    })

    # Symbol suffix → exchange code
    suffix_to_exchange: dict[str, str] = field(default_factory=lambda: {
        ".CO": "CSE",       # Copenhagen
        ".CPH": "CSE",      # Copenhagen alternativ
        ".ST": "SFB",       # Stockholm
        ".OL": "OSE",       # Oslo
        ".HE": "HEX",       # Helsinki
        ".DE": "XETRA",     # Tyskland / XETRA
        ".F": "XETRA",      # Frankfurt
        ".PA": "SBF",       # Euronext Paris
        ".AS": "AEB",       # Euronext Amsterdam
        ".BR": "AEB",       # Euronext Bruxelles
        ".L": "LSE",        # London
        ".SW": "EBS",       # SIX Swiss
        ".MI": "MIL",       # Milano
        ".MC": "BME",       # Madrid
        ".NS": "NSE",       # National Stock Exchange India
        ".TO": "TSX",       # Toronto
        ".AX": "ASX",       # Australia
        ".NZ": "NZX",       # New Zealand
    })

    # Exchange code → primary broker
    exchange_to_broker: dict[str, str] = field(default_factory=lambda: {
        # US exchanges → Alpaca
        "NYSE": "alpaca",
        "NASDAQ": "alpaca",
        "ARCA": "alpaca",
        "AMEX": "alpaca",
        # Nordic → Nordnet (med saxo/ibkr fallback)
        "CSE": "nordnet",
        "SFB": "nordnet",
        "OSE": "nordnet",
        "HEX": "nordnet",
        # Europa → IBKR
        "XETRA": "ibkr",
        "SBF": "ibkr",
        "AEB": "ibkr",
        "LSE": "ibkr",
        "EBS": "ibkr",
        "MIL": "ibkr",
        "BME": "ibkr",
        # Derivater → IBKR
        "CME": "ibkr",
        "CBOE": "ibkr",
        "EUREX": "ibkr",
        # Andre
        "NSE": "ibkr",
        "TSX": "ibkr",
        "ASX": "ibkr",
        "NZX": "ibkr",
    })

    # Exchange → fallback brokers (prøves i rækkefølge efter primary)
    exchange_fallbacks: dict[str, list[str]] = field(default_factory=lambda: {
        "CSE": ["saxo", "ibkr"],
        "SFB": ["saxo", "ibkr"],
        "OSE": ["saxo", "ibkr"],
        "HEX": ["saxo", "ibkr"],
    })

    # Asset type → broker
    asset_type_to_broker: dict[str, str] = field(default_factory=lambda: {
        "forex": "ibkr",
        "futures": "ibkr",
        "options": "ibkr",
        "commodity": "ibkr",
        "etf_eu": "saxo",
        "bond": "saxo",
        "fund_dk": "nordnet",
        "crypto": "alpaca",
        "stock_us": "alpaca",
        "stock_nordic": "nordnet",
        "stock_eu": "ibkr",
    })

    # Global fallback chain — paper er altid sidste udvej
    fallback_chain: list[str] = field(
        default_factory=lambda: ["ibkr", "saxo", "alpaca", "paper"]
    )


# ── Symbol Utilities ────────────────────────────────────────

# Regex til at fange suffix som .CO, .ST, .DE osv.
_SUFFIX_PATTERN = re.compile(r"(\.[A-Z]{1,3})$")

# US-aktier har typisk INGEN suffix og 1-5 bogstaver
_US_STOCK_PATTERN = re.compile(r"^[A-Z]{1,5}$")

# Crypto patterns
_CRYPTO_PATTERN = re.compile(
    r"^(BTC|ETH|SOL|DOGE|AVAX|ADA|LINK|DOT|UNI|MATIC|SHIB|XRP)"
    r"[/-]?(USD|USDT|EUR)?$",
    re.IGNORECASE,
)


def detect_exchange(symbol: str) -> str | None:
    """
    Detektér exchange fra symbol suffix.

    Eksempler:
        "NOVO-B.CO" → "CSE"
        "VOLVO-B.ST" → "SFB"
        "SAP.DE" → "XETRA"
        "AAPL" → None (antages US)
    """
    config = RoutingConfig()  # Default for suffix lookup

    match = _SUFFIX_PATTERN.search(symbol)
    if match:
        suffix = match.group(1).upper()
        exchange = config.suffix_to_exchange.get(suffix)
        if exchange:
            return exchange

    return None


def detect_asset_type(symbol: str) -> str | None:
    """
    Heuristisk asset type detection.

    Bruger symbol-patterns til at gætte asset type.
    """
    upper = symbol.upper()

    # Crypto
    if _CRYPTO_PATTERN.match(upper):
        return "crypto"

    # Forex pairs (EUR/USD, GBP/JPY, osv.)
    forex_currencies = {"EUR", "USD", "GBP", "JPY", "CHF", "AUD", "CAD",
                        "NZD", "SEK", "NOK", "DKK"}
    parts = re.split(r"[/]", upper)
    if len(parts) == 2 and parts[0] in forex_currencies and parts[1] in forex_currencies:
        return "forex"

    # Futures (ES=F, NQ=F, CL=F format)
    if upper.endswith("=F") or upper.startswith("/"):
        return "futures"

    # Nordic suffix
    exchange = detect_exchange(symbol)
    if exchange in ("CSE", "SFB", "OSE", "HEX"):
        return "stock_nordic"

    # EU suffix
    if exchange in ("XETRA", "SBF", "AEB", "LSE", "EBS", "MIL", "BME"):
        return "stock_eu"

    # US stock (no suffix, 1-5 chars)
    if _US_STOCK_PATTERN.match(upper):
        return "stock_us"

    return None


# ── BrokerRouter ────────────────────────────────────────────

class BrokerRouter(BaseBroker):
    """
    Unified broker interface der router til den rigtige broker.

    Brug:
        router = BrokerRouter()
        router.register("alpaca", alpaca_broker)
        router.register("ibkr", ibkr_broker)
        router.register("saxo", saxo_broker)
        router.register("nordnet", nordnet_broker)

        # Automatisk routing
        order = router.buy("AAPL", qty=10)         # → alpaca
        order = router.buy("NOVO-B.CO", qty=5)     # → nordnet
        order = router.buy("SAP.DE", qty=3)        # → ibkr
        order = router.buy("BTC-USD", qty=0.5)     # → alpaca

        # Manuel override
        order = router.buy("AAPL", qty=10, broker_override="ibkr")
    """

    def __init__(self, config: RoutingConfig | None = None) -> None:
        self._config = config or RoutingConfig()
        self._brokers: dict[str, BaseBroker] = {}

    @property
    def name(self) -> str:
        return "router"

    # ── Broker Registration ─────────────────────────────────

    def register(self, name: str, broker: BaseBroker) -> None:
        """Registrér en broker med et navn."""
        self._brokers[name.lower()] = broker
        logger.info(f"[router] Registreret broker: {name}")

    def unregister(self, name: str) -> None:
        """Fjern en broker."""
        self._brokers.pop(name.lower(), None)

    @property
    def available_brokers(self) -> list[str]:
        """Navne på alle registrerede brokers."""
        return list(self._brokers.keys())

    def get_broker(self, name: str) -> BaseBroker | None:
        """Hent en specifik broker by name."""
        return self._brokers.get(name.lower())

    # ── Routing Logic ───────────────────────────────────────

    def resolve_broker(
        self,
        symbol: str,
        asset_type: str | None = None,
        broker_override: str | None = None,
    ) -> tuple[str, BaseBroker]:
        """
        Resolve hvilken broker der skal håndtere et symbol.

        Resolution order:
          1. Explicit broker override
          2. Exact symbol match
          3. Exchange detection (symbol suffix → exchange → broker)
          4. Asset type routing
          5. Fallback chain

        Args:
            symbol: Trading-symbol (f.eks. "AAPL", "NOVO-B.CO", "BTC-USD").
            asset_type: Valgfri asset type override.
            broker_override: Tving en specifik broker.

        Returns:
            (broker_name, broker_instance)

        Raises:
            RoutingError: Hvis ingen broker kan håndtere symbolet.
        """
        # 1. Explicit override
        if broker_override:
            broker = self._brokers.get(broker_override.lower())
            if broker:
                return broker_override.lower(), broker
            raise RoutingError(
                f"Broker override '{broker_override}' ikke registreret. "
                f"Tilgængelige: {self.available_brokers}"
            )

        # 2. Exact match
        upper = symbol.upper()
        exact_broker = self._config.exact_matches.get(upper)
        if exact_broker and exact_broker in self._brokers:
            return exact_broker, self._brokers[exact_broker]

        # 3. Exchange detection
        exchange = detect_exchange(symbol)
        if exchange:
            primary = self._config.exchange_to_broker.get(exchange)
            if primary and primary in self._brokers:
                return primary, self._brokers[primary]

            # Exchange fallbacks
            fallbacks = self._config.exchange_fallbacks.get(exchange, [])
            for fb in fallbacks:
                if fb in self._brokers:
                    logger.debug(
                        f"[router] {symbol}: primary '{primary}' unavailable, "
                        f"fallback til '{fb}'"
                    )
                    return fb, self._brokers[fb]

        # 4. Asset type routing
        resolved_type = asset_type or detect_asset_type(symbol)
        if resolved_type:
            type_broker = self._config.asset_type_to_broker.get(resolved_type)
            if type_broker and type_broker in self._brokers:
                return type_broker, self._brokers[type_broker]

        # 5. Fallback chain
        for fb in self._config.fallback_chain:
            if fb in self._brokers:
                logger.debug(
                    f"[router] {symbol}: ingen specifik route, "
                    f"fallback til '{fb}'"
                )
                return fb, self._brokers[fb]

        # Ingen broker fundet
        raise RoutingError(
            f"Kan ikke route '{symbol}' (asset_type={resolved_type}). "
            f"Registrerede brokers: {self.available_brokers}"
        )

    def explain_routing(self, symbol: str, asset_type: str | None = None) -> dict:
        """
        Forklar routing-beslutningen for et symbol.

        Nyttigt til debugging og UI.
        """
        result: dict[str, Any] = {
            "symbol": symbol,
            "detected_exchange": detect_exchange(symbol),
            "detected_asset_type": detect_asset_type(symbol),
            "asset_type_override": asset_type,
        }

        try:
            broker_name, broker = self.resolve_broker(symbol, asset_type)
            result["resolved_broker"] = broker_name
            result["broker_name"] = broker.name
            result["success"] = True
        except RoutingError as exc:
            result["resolved_broker"] = None
            result["error"] = str(exc)
            result["success"] = False

        return result

    # ── BaseBroker Implementation ───────────────────────────

    def buy(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        broker_override: str | None = None,
    ) -> Order:
        """Køb via den broker der matcher symbolet."""
        self._validate_order(symbol, qty, order_type, limit_price)
        broker_name, broker = self.resolve_broker(
            symbol, broker_override=broker_override
        )
        logger.info(
            f"[router] BUY {qty} {symbol} via {broker_name} "
            f"({order_type.value}, limit={limit_price})"
        )
        return broker.buy(symbol, qty, order_type, limit_price)

    def sell(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        broker_override: str | None = None,
        short: bool = False,
    ) -> Order:
        """Sælg via den broker der matcher symbolet.

        For non-short sells, automatically routes to the broker that holds
        the position (instead of using generic routing which may pick a
        different broker).

        Args:
            short: Hvis True, åbn en short position (sælg uden at eje).
        """
        self._validate_order(symbol, qty, order_type, limit_price)

        # For closing positions (not short-opening), find the broker
        # that actually holds this position to avoid routing mismatches.
        if not short and not broker_override:
            upper = symbol.upper()
            for name, b in self._brokers.items():
                try:
                    for pos in b.get_positions():
                        if getattr(pos, "symbol", "").upper() == upper and getattr(pos, "qty", 0) > 0:
                            logger.info(
                                f"[router] SELL {qty} {symbol} via {name} "
                                f"(position found in {name}, {order_type.value}, limit={limit_price})"
                            )
                            return b.sell(symbol, qty, order_type, limit_price, short=False)
                except Exception:
                    continue

        broker_name, broker = self.resolve_broker(
            symbol, broker_override=broker_override
        )
        action = "SHORT" if short else "SELL"
        logger.info(
            f"[router] {action} {qty} {symbol} via {broker_name} "
            f"({order_type.value}, limit={limit_price})"
        )
        return broker.sell(symbol, qty, order_type, limit_price, short=short)

    def get_positions(self) -> list[Position]:
        """Hent positioner fra ALLE registrerede brokers."""
        all_positions: list[Position] = []
        for name, broker in self._brokers.items():
            try:
                positions = broker.get_positions()
                all_positions.extend(positions)
                logger.debug(f"[router] {name}: {len(positions)} positioner")
            except Exception as exc:
                logger.warning(f"[router] Fejl ved hentning fra {name}: {exc}")
        return all_positions

    def get_account(self) -> AccountInfo:
        """
        Hent aggregeret kontoinformation.

        Summerer cash, equity og buying power fra alle brokers.
        Returnerer i USD som base currency (konvertering sker i AggregatedPortfolio).

        WARNING: This is a raw sum without FX conversion. Brokers may report
        in different currencies (USD, DKK, EUR, etc.), so the totals are
        approximate when multiple currency accounts are active.
        """
        total_cash = 0.0
        total_equity = 0.0
        total_portfolio = 0.0
        total_buying_power = 0.0
        accounts_fetched = 0
        currencies_seen: set[str] = set()

        for name, broker in self._brokers.items():
            try:
                account = broker.get_account()
                # WARNING: raw sum without FX conversion — values may be in
                # different currencies (USD, DKK, EUR, etc.)
                total_cash += account.cash
                total_equity += account.equity
                total_portfolio += account.portfolio_value
                total_buying_power += account.buying_power
                accounts_fetched += 1
                if hasattr(account, "currency") and account.currency:
                    currencies_seen.add(account.currency)
            except Exception as exc:
                logger.warning(f"[router] Kontofejl fra {name}: {exc}")

        mixed = len(currencies_seen) > 1
        if mixed:
            logger.warning(
                f"[router] Aggregated account sums mix currencies: {currencies_seen}. "
                f"Values are NOT FX-converted — treat as approximate."
            )

        return AccountInfo(
            account_id=f"router_aggregated_{accounts_fetched}_brokers"
                       + ("_MIXED_CURRENCIES" if mixed else ""),
            cash=total_cash,
            portfolio_value=total_portfolio,
            buying_power=total_buying_power,
            equity=total_equity,
            currency="USD" if not mixed else f"MIXED({','.join(sorted(currencies_seen))})",
        )

    def get_order_status(self, order_id: str) -> Order:
        """
        Hent ordrestatus — prøver alle brokers da vi ikke ved hvilken.

        Med OrderManager bruges unified IDs, men dette er fallback.
        """
        for name, broker in self._brokers.items():
            try:
                return broker.get_order_status(order_id)
            except Exception:
                continue

        raise BrokerError(
            f"Ordre '{order_id}' ikke fundet hos nogen broker. "
            f"Prøvede: {self.available_brokers}"
        )

    def cancel_order(self, order_id: str) -> bool:
        """Annullér en ordre — prøver alle brokers."""
        for name, broker in self._brokers.items():
            try:
                result = broker.cancel_order(order_id)
                if result:
                    logger.info(f"[router] Ordre {order_id} annulleret via {name}")
                    return True
            except Exception:
                continue

        logger.warning(f"[router] Kunne ikke annullere ordre {order_id}")
        return False

    # ── Utility ─────────────────────────────────────────────

    def get_positions_by_broker(self) -> dict[str, list[Position]]:
        """Hent positioner grupperet per broker."""
        result = {}
        for name, broker in self._brokers.items():
            try:
                result[name] = broker.get_positions()
            except Exception as exc:
                logger.warning(f"[router] Fejl ved {name}: {exc}")
                result[name] = []
        return result

    def get_accounts_by_broker(self) -> dict[str, AccountInfo]:
        """Hent kontoinformation per broker."""
        result = {}
        for name, broker in self._brokers.items():
            try:
                result[name] = broker.get_account()
            except Exception as exc:
                logger.warning(f"[router] Kontofejl fra {name}: {exc}")
        return result

    def status(self) -> dict[str, Any]:
        """Samlet status for alle brokers."""
        return {
            "brokers_registered": self.available_brokers,
            "broker_count": len(self._brokers),
            "routing_config": {
                "exact_matches": len(self._config.exact_matches),
                "exchanges": len(self._config.exchange_to_broker),
                "asset_types": len(self._config.asset_type_to_broker),
                "fallback_chain": self._config.fallback_chain,
            },
        }
