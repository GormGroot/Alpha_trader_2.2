"""
NordnetBroker — Nordnet integration via uofficiel web-API.

Primært til: Danske aktier, svenske aktier, norske aktier, danske fonde.

VIGTIGT:
  - Uofficiel API — kan bryde ved Nordnet opdateringer
  - Saxo Bank er backup (BrokerRouter fallback)
  - Defensiv coding: Alle responses valideres
  - Max 1 req/sec (aggressiv for at undgå ban)
  - Alle data caches lokalt
"""

from __future__ import annotations

import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable

from loguru import logger

from src.broker.base_broker import BaseBroker
from src.broker.models import (
    AccountInfo,
    BrokerError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationError,
)
from src.broker.nordnet_auth import NordnetAuthError, NordnetConfig, NordnetSession
from src.risk.portfolio_tracker import Position


# ── Retry Decorator ─────────────────────────────────────────

def _retry_nordnet(
    max_retries: int = 2,
    base_delay: float = 2.0,
) -> Callable:
    """Retry med backoff — konservativt for uofficiel API."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except NordnetAuthError:
                    raise
                except BrokerError as exc:
                    if "400" in str(exc) or "422" in str(exc):
                        raise
                    last_exc = exc
                except Exception as exc:
                    last_exc = exc

                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"[nordnet] API-fejl (forsøg {attempt + 1}/"
                        f"{max_retries}): {last_exc} — retry om {delay:.1f}s"
                    )
                    time.sleep(delay)

            raise BrokerError(
                f"Nordnet API-fejl efter {max_retries} forsøg: {last_exc}"
            ) from last_exc

        return wrapper
    return decorator


# ── NordnetBroker ───────────────────────────────────────────

class NordnetBroker(BaseBroker):
    """
    Nordnet broker-integration via uofficiel web-API.

    Brug:
        config = NordnetConfig.from_env()
        session = NordnetSession(config)
        broker = NordnetBroker(session)

        broker.connect()
        positions = broker.get_positions()
        order = broker.buy("NOVO-B.CO", qty=5, order_type=OrderType.LIMIT, limit_price=850.0)
    """

    def __init__(
        self,
        session: NordnetSession | None = None,
        config: NordnetConfig | None = None,
    ) -> None:
        self._config = config or NordnetConfig.from_env()
        self._session = session or NordnetSession(self._config)
        self._connected = False
        self._account_id: str = ""

        # Instrument cache: symbol → instrument_id
        self._instrument_cache: dict[str, dict] = {}

        # Position cache (30 sec TTL)
        self._position_cache: list[Position] = []
        self._position_cache_time: float = 0.0
        self._POSITION_CACHE_TTL = 30

    @property
    def name(self) -> str:
        return "nordnet"

    # ── Connection ──────────────────────────────────────────

    def connect(self) -> dict:
        """Log ind og hent account info."""
        result = self._session.login()
        self._account_id = result.get("account_id", "")
        self._connected = True
        logger.info(f"[nordnet] Forbundet — account: {self._account_id}")
        return result

    def _ensure_connected(self) -> None:
        if not self._connected:
            self.connect()
        self._session.ensure_logged_in()

    # ── Instrument Lookup ───────────────────────────────────

    def _resolve_instrument(self, symbol: str) -> dict:
        """
        Resolve symbol til Nordnet instrument.

        Nordnet bruger instrument_id (numerisk) internt.
        """
        upper = symbol.upper()

        # Strip exchange suffixes for search
        clean = upper
        for suffix in (".CO", ".CPH", ".ST", ".OL", ".HE"):
            if clean.endswith(suffix):
                clean = clean[: -len(suffix)]
                break

        # Check cache
        if upper in self._instrument_cache:
            return self._instrument_cache[upper]

        # Søg
        try:
            results = self._session.get(
                f"/instruments",
                params={"query": clean, "limit": 10},
            )

            instruments = []
            if isinstance(results, list):
                instruments = results
            elif isinstance(results, dict):
                instruments = results.get("Data", results.get("instruments", []))

            if not instruments:
                raise OrderValidationError(
                    f"Instrument '{symbol}' ikke fundet hos Nordnet"
                )

            # Find best match
            best = instruments[0]
            for inst in instruments:
                inst_symbol = str(inst.get("symbol", inst.get("name", ""))).upper()
                if inst_symbol == clean or inst_symbol == upper:
                    best = inst
                    break

            self._instrument_cache[upper] = best
            logger.debug(
                f"[nordnet] Resolved '{symbol}' → "
                f"id={best.get('instrument_id', best.get('id', '?'))}"
            )
            return best

        except OrderValidationError:
            raise
        except Exception as exc:
            raise BrokerError(
                f"Nordnet instrument lookup fejl for '{symbol}': {exc}"
            ) from exc

    def _get_instrument_id(self, instrument: dict) -> str:
        """Extract instrument_id fra instrument dict (defensivt)."""
        for key in ("instrument_id", "id", "identifier", "Identifier"):
            if key in instrument:
                return str(instrument[key])
        raise BrokerError("Kan ikke finde instrument_id i Nordnet response")

    # ── BaseBroker Implementation ───────────────────────────

    @_retry_nordnet()
    def buy(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order:
        """Placér en købsordre via Nordnet."""
        self._validate_order(symbol, qty, order_type, limit_price)
        self._ensure_connected()

        instrument = self._resolve_instrument(symbol)
        instrument_id = self._get_instrument_id(instrument)

        order_data: dict[str, Any] = {
            "identifier": instrument_id,
            "side": "BUY",
            "volume": qty,
            "order_type": "LIMIT" if order_type == OrderType.LIMIT else "MARKET",
            "validity": {"type": "DAY"},
        }

        if order_type == OrderType.LIMIT and limit_price is not None:
            order_data["price"] = limit_price
        elif order_type == OrderType.MARKET:
            # Nordnet kræver ofte en pris selv for market orders
            # Sæt en høj limit som "market" simulering
            order_data["order_type"] = "LIMIT"
            # Hent current price og sæt +2%
            try:
                cur_price = float(instrument.get("last_price", 0))
                if cur_price > 0:
                    order_data["price"] = round(cur_price * 1.02, 2)
            except (ValueError, TypeError):
                if limit_price:
                    order_data["price"] = limit_price
                else:
                    raise OrderValidationError(
                        "Nordnet kræver limit-pris. "
                        "Angiv limit_price for denne ordre."
                    )

        data = self._session.post(
            f"/accounts/{self._account_id}/orders",
            data=order_data,
        )

        order_id = str(data.get("order_id", data.get("orderId", "unknown")))

        # Invalidér position cache
        self._position_cache_time = 0

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=order_type,
            qty=qty,
            status=OrderStatus.SUBMITTED,
            limit_price=limit_price or order_data.get("price"),
            submitted_at=datetime.now().isoformat(),
        )

        logger.info(
            f"[nordnet] KØB {qty} {symbol} ({order_type.value}) → {order_id}"
        )
        return order

    @_retry_nordnet()
    def sell(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        short: bool = False,
    ) -> Order:
        """Placér en salgsordre via Nordnet."""
        if short:
            raise NotImplementedError("Short-selling er ikke understøttet via Nordnet")
        self._validate_order(symbol, qty, order_type, limit_price)
        self._ensure_connected()

        instrument = self._resolve_instrument(symbol)
        instrument_id = self._get_instrument_id(instrument)

        order_data: dict[str, Any] = {
            "identifier": instrument_id,
            "side": "SELL",
            "volume": qty,
            "order_type": "LIMIT" if order_type == OrderType.LIMIT else "MARKET",
            "validity": {"type": "DAY"},
        }

        if order_type == OrderType.LIMIT and limit_price is not None:
            order_data["price"] = limit_price
        elif order_type == OrderType.MARKET:
            order_data["order_type"] = "LIMIT"
            try:
                cur_price = float(instrument.get("last_price", 0))
                if cur_price > 0:
                    order_data["price"] = round(cur_price * 0.98, 2)
            except (ValueError, TypeError):
                if limit_price:
                    order_data["price"] = limit_price
                else:
                    raise OrderValidationError(
                        "Nordnet kræver limit-pris for sell orders."
                    )

        data = self._session.post(
            f"/accounts/{self._account_id}/orders",
            data=order_data,
        )

        order_id = str(data.get("order_id", data.get("orderId", "unknown")))
        self._position_cache_time = 0

        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=order_type,
            qty=qty,
            status=OrderStatus.SUBMITTED,
            limit_price=limit_price or order_data.get("price"),
            submitted_at=datetime.now().isoformat(),
        )

        logger.info(
            f"[nordnet] SÆLG {qty} {symbol} ({order_type.value}) → {order_id}"
        )
        return order

    @_retry_nordnet()
    def get_positions(self) -> list[Position]:
        """Hent alle åbne positioner fra Nordnet."""
        self._ensure_connected()

        # Cache check
        if (time.time() - self._position_cache_time) < self._POSITION_CACHE_TTL:
            return self._position_cache

        data = self._session.get(
            f"/accounts/{self._account_id}/positions"
        )

        raw_positions = []
        if isinstance(data, list):
            raw_positions = data
        elif isinstance(data, dict):
            raw_positions = data.get("Data", data.get("positions", []))

        positions: list[Position] = []
        for p in raw_positions:
            # Defensiv extraction — Nordnet's schema kan ændre sig
            symbol = self._safe_get_str(p, ["instrument", "symbol", "name"])
            qty = self._safe_get_float(p, ["qty", "volume", "amount", "quantity"])
            acq_price = self._safe_get_float(
                p, ["acq_price", "acquisition_price", "avg_price", "acquiredPrice"]
            )
            market_value = self._safe_get_float(
                p, ["market_value", "marketValue", "value"]
            )

            if qty == 0:
                continue

            side = "long" if qty > 0 else "short"
            current_price = (
                market_value / abs(qty) if qty != 0 and market_value > 0
                else acq_price
            )

            positions.append(Position(
                symbol=symbol or "UNKNOWN",
                side=side,
                qty=abs(qty),
                entry_price=acq_price,
                entry_time="",
                current_price=current_price,
            ))

        # Cache
        self._position_cache = positions
        self._position_cache_time = time.time()

        return positions

    @_retry_nordnet()
    def get_account(self) -> AccountInfo:
        """Hent kontoinformation fra Nordnet."""
        self._ensure_connected()

        data = self._session.get(
            f"/accounts/{self._account_id}/info"
        )

        # Defensiv parsing
        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        account_value = self._safe_get_float(
            data, ["account_sum", "total_value", "accountSum", "totalValue"]
        )
        cash = self._safe_get_float(
            data, ["own_capital_morning", "cash_balance", "cashBalance", "cash"]
        )
        # Nordnet har ikke eksplicit buying_power — brug cash
        buying_power = cash

        return AccountInfo(
            account_id=self._account_id or "nordnet",
            cash=cash,
            portfolio_value=account_value,
            buying_power=buying_power,
            equity=account_value,
            currency="DKK",
        )

    @_retry_nordnet()
    def get_order_status(self, order_id: str) -> Order:
        """Hent status for en specifik ordre."""
        self._ensure_connected()

        data = self._session.get(
            f"/accounts/{self._account_id}/orders"
        )

        orders = []
        if isinstance(data, list):
            orders = data
        elif isinstance(data, dict):
            orders = data.get("Data", data.get("orders", []))

        for o in orders:
            oid = str(o.get("order_id", o.get("orderId", "")))
            if oid == str(order_id):
                return self._map_nordnet_order(o)

        raise BrokerError(f"Ordre {order_id} ikke fundet hos Nordnet")

    @_retry_nordnet()
    def cancel_order(self, order_id: str) -> bool:
        """Annullér en ordre."""
        self._ensure_connected()

        try:
            self._session.delete(
                f"/accounts/{self._account_id}/orders/{order_id}"
            )
            self._position_cache_time = 0
            logger.info(f"[nordnet] Ordre {order_id} annulleret")
            return True
        except Exception as exc:
            logger.warning(f"[nordnet] Cancel fejl: {exc}")
            return False

    # ── Mapping Helpers ─────────────────────────────────────

    def _map_nordnet_order(self, data: dict) -> Order:
        """Map Nordnet order dict til vores Order model."""
        # Status mapping
        status_raw = str(
            data.get("state", data.get("status", ""))
        ).lower()
        status_map = {
            "local": OrderStatus.PENDING,
            "on_market": OrderStatus.SUBMITTED,
            "filled": OrderStatus.FILLED,
            "cancelled": OrderStatus.CANCELLED,
            "deleted": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
        }
        status = status_map.get(status_raw, OrderStatus.PENDING)

        side_raw = str(data.get("side", data.get("order_side", "BUY"))).upper()
        side = OrderSide.BUY if side_raw == "BUY" else OrderSide.SELL

        return Order(
            order_id=str(data.get("order_id", data.get("orderId", ""))),
            symbol=self._safe_get_str(data, ["instrument", "symbol", "name"]),
            side=side,
            order_type=OrderType.LIMIT,  # Nordnet bruger primært limit
            qty=self._safe_get_float(data, ["volume", "qty", "amount"]),
            status=status,
            limit_price=self._safe_get_float(data, ["price", "limit_price"]) or None,
            filled_qty=self._safe_get_float(
                data, ["traded_volume", "filled_volume", "filledQty"]
            ),
            filled_avg_price=self._safe_get_float(
                data, ["traded_price", "avg_price", "filledPrice"]
            ),
        )

    # ── Defensive Helpers ───────────────────────────────────

    @staticmethod
    def _safe_get_str(data: dict, keys: list[str]) -> str:
        """Hent string fra dict med multiple possible keys."""
        for key in keys:
            if "." in key:
                parts = key.split(".")
                d = data
                for part in parts:
                    if isinstance(d, dict):
                        d = d.get(part, {})
                    else:
                        break
                if isinstance(d, str):
                    return d
            elif key in data:
                val = data[key]
                if isinstance(val, dict):
                    # Nested instrument object
                    for subkey in ("symbol", "name", "ticker"):
                        if subkey in val:
                            return str(val[subkey])
                return str(val)
        return ""

    @staticmethod
    def _safe_get_float(data: dict, keys: list[str]) -> float:
        """Hent float fra dict med multiple possible keys."""
        for key in keys:
            if key in data:
                try:
                    val = data[key]
                    if isinstance(val, dict):
                        val = val.get("value", val.get("amount", 0))
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return 0.0

    # ── Extra: Instrument Search ────────────────────────────

    def search_instruments(self, query: str) -> list[dict]:
        """Søg efter instrumenter hos Nordnet."""
        self._ensure_connected()
        try:
            data = self._session.get(
                "/instruments",
                params={"query": query, "limit": 20},
            )
            if isinstance(data, list):
                return data
            return data.get("Data", data.get("instruments", []))
        except Exception as exc:
            logger.warning(f"[nordnet] Instrument search fejl: {exc}")
            return []

    # ── Status ──────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        return {
            "broker": "nordnet",
            "connected": self._connected,
            "account_id": self._account_id,
            "session": self._session.status(),
            "instrument_cache_size": len(self._instrument_cache),
            "position_cache_age": (
                round(time.time() - self._position_cache_time, 1)
                if self._position_cache_time > 0 else None
            ),
        }
