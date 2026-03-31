"""
IBKRBroker — Interactive Brokers integration via ib_insync.

Primært til: EU aktier, UK aktier, råstoffer, forex, futures, options.
Forbindelse: TWS/IB Gateway på localhost (port 4001 live / 4002 paper).

Kræver:
  - pip install ib_insync
  - TWS eller IB Gateway kørende lokalt
"""

from __future__ import annotations

import os
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
from src.risk.portfolio_tracker import Position


# ── Retry Decorator ─────────────────────────────────────────

def _retry_ibkr(
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Callable:
    """Retry med exponential backoff for IBKR API-kald."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except BrokerError:
                    raise  # Validation/permanent fejl
                except Exception as exc:
                    last_exc = exc

                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"[ibkr] API-fejl (forsøg {attempt + 1}/{max_retries}): "
                        f"{last_exc} — retry om {delay:.1f}s"
                    )
                    time.sleep(delay)

            raise BrokerError(
                f"IBKR API-fejl efter {max_retries} forsøg: {last_exc}"
            ) from last_exc

        return wrapper
    return decorator


# ── Status Mapping ──────────────────────────────────────────

_IBKR_STATUS_MAP: dict[str, OrderStatus] = {
    "submitted": OrderStatus.SUBMITTED,
    "presubmitted": OrderStatus.SUBMITTED,
    "filled": OrderStatus.FILLED,
    "cancelled": OrderStatus.CANCELLED,
    "inactive": OrderStatus.REJECTED,
    "pendingsubmit": OrderStatus.PENDING,
    "pendingcancel": OrderStatus.PENDING,
    "apicancelled": OrderStatus.CANCELLED,
}


# ── Contract Builders ───────────────────────────────────────

def _build_stock_contract(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Any:
    """Byg en aktie-contract for IBKR."""
    try:
        from ib_insync import Stock
        return Stock(symbol, exchange, currency)
    except ImportError:
        raise BrokerError("ib_insync er ikke installeret — kør: pip install ib_insync")


def _build_forex_contract(pair: str) -> Any:
    """Byg en forex contract. F.eks. 'EUR/USD' → Forex('EUR', 'IDEALPRO', 'USD')."""
    try:
        from ib_insync import Forex
        parts = pair.replace("/", "").replace(".", "")
        if len(parts) == 6:
            base = parts[:3]
            quote = parts[3:]
            return Forex(base, "IDEALPRO", quote)
        raise OrderValidationError(f"Ugyldig forex pair: {pair}")
    except ImportError:
        raise BrokerError("ib_insync er ikke installeret")


def _build_future_contract(symbol: str) -> Any:
    """Byg en futures contract (continuous)."""
    try:
        from ib_insync import ContFuture
        # Strip =F suffix (Yahoo format)
        clean = symbol.replace("=F", "").replace("/", "")
        return ContFuture(clean, "SMART")
    except ImportError:
        raise BrokerError("ib_insync er ikke installeret")


def _build_commodity_contract(symbol: str) -> Any:
    """Byg commodity futures contract."""
    # Commodity symbol mapping
    commodity_map = {
        "GC": ("GC", "COMEX", "USD"),   # Gold
        "SI": ("SI", "COMEX", "USD"),   # Silver
        "CL": ("CL", "NYMEX", "USD"),  # Crude Oil
        "NG": ("NG", "NYMEX", "USD"),  # Natural Gas
        "HG": ("HG", "COMEX", "USD"),  # Copper
    }
    clean = symbol.replace("=F", "").upper()

    if clean in commodity_map:
        sym, exch, ccy = commodity_map[clean]
        try:
            from ib_insync import ContFuture
            return ContFuture(sym, exch, ccy)
        except ImportError:
            raise BrokerError("ib_insync er ikke installeret")

    return _build_future_contract(symbol)


# ── Exchange → Currency Mapping ─────────────────────────────

_EXCHANGE_CURRENCY: dict[str, tuple[str, str]] = {
    # suffix → (IBKR exchange, currency)
    ".DE": ("IBIS", "EUR"),      # XETRA
    ".F": ("FWB", "EUR"),        # Frankfurt
    ".PA": ("SBF", "EUR"),       # Paris
    ".AS": ("AEB", "EUR"),       # Amsterdam
    ".BR": ("BELFOX", "EUR"),    # Brussels
    ".MI": ("BVME", "EUR"),      # Milan
    ".MC": ("BM", "EUR"),        # Madrid
    ".L": ("LSE", "GBP"),       # London
    ".SW": ("EBS", "CHF"),      # Zürich
    ".CO": ("CSE", "DKK"),      # Copenhagen
    ".ST": ("SFB", "SEK"),      # Stockholm
    ".OL": ("OSE", "NOK"),      # Oslo
    ".HE": ("HEX", "EUR"),      # Helsinki
}


def _detect_ibkr_contract(symbol: str) -> Any:
    """
    Auto-detect contract type fra symbol.

    Understøtter:
        - EU aktier: "SAP.DE", "ASML.AS", "NOVO-B.CO"
        - UK aktier: "VOD.L", "HSBA.L"
        - US aktier: "AAPL", "MSFT" (fallback)
        - Forex: "EUR/USD", "GBP/DKK"
        - Futures: "ES=F", "NQ=F", "/CL"
        - Commodities: "GC", "CL", "SI"
    """
    upper = symbol.upper()

    # Forex
    if "/" in symbol and len(symbol.replace("/", "")) == 6:
        return _build_forex_contract(symbol)

    # Futures
    if upper.endswith("=F") or symbol.startswith("/"):
        return _build_future_contract(symbol)

    # Commodities
    if upper in ("GC", "SI", "CL", "NG", "HG"):
        return _build_commodity_contract(symbol)

    # European stocks (suffix-based)
    for suffix, (exchange, currency) in _EXCHANGE_CURRENCY.items():
        if upper.endswith(suffix.upper()):
            clean_symbol = upper[: -len(suffix)]
            # IBKR bruger ofte '-' som ' ' for B-aktier (NOVO-B → NOVO B)
            return _build_stock_contract(
                clean_symbol.replace("-", " "),
                exchange,
                currency,
            )

    # Default: US stock via SMART routing
    return _build_stock_contract(upper, "SMART", "USD")


# ── IBKRBroker ──────────────────────────────────────────────

class IBKRBroker(BaseBroker):
    """
    Interactive Brokers integration via ib_insync.

    Brug:
        broker = IBKRBroker()
        broker.connect()

        # Standard BaseBroker interface
        positions = broker.get_positions()
        order = broker.buy("SAP.DE", qty=10)
        order = broker.buy("EUR/USD", qty=10000)

    Kræver TWS eller IB Gateway kørende lokalt.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
    ) -> None:
        self._host = host or os.environ.get("IBKR_HOST", "127.0.0.1")
        self._port = port or int(os.environ.get("IBKR_PORT", "4002"))
        self._client_id = client_id or int(os.environ.get("IBKR_CLIENT_ID", "1"))
        self._ib: Any = None
        self._connected = False

    @property
    def name(self) -> str:
        return "ibkr"

    # ── Connection ──────────────────────────────────────────

    def connect(self) -> dict:
        """Forbind til TWS/IB Gateway."""
        try:
            from ib_insync import IB
        except ImportError:
            raise BrokerError(
                "ib_insync er ikke installeret. "
                "Kør: pip install ib_insync"
            )

        self._ib = IB()
        try:
            self._ib.connect(
                self._host,
                self._port,
                clientId=self._client_id,
                timeout=15,
            )
            self._connected = True

            accounts = self._ib.managedAccounts()
            logger.info(
                f"[ibkr] Forbundet til {self._host}:{self._port} — "
                f"accounts: {accounts}"
            )
            return {
                "host": self._host,
                "port": self._port,
                "accounts": accounts,
                "connected": True,
            }

        except Exception as exc:
            self._connected = False
            raise BrokerError(
                f"Kan ikke forbinde til IBKR på "
                f"{self._host}:{self._port}: {exc}"
            ) from exc

    def disconnect(self) -> None:
        """Afbryd forbindelse."""
        if self._ib and self._connected:
            self._ib.disconnect()
            self._connected = False
            logger.info("[ibkr] Afbrudt")

    def _ensure_connected(self) -> None:
        """Sikr at vi er forbundet."""
        if not self._ib or not self._connected:
            raise BrokerError(
                "Ikke forbundet til IBKR. Kald connect() først."
            )
        if not self._ib.isConnected():
            logger.warning("[ibkr] Forbindelse tabt — reconnecting...")
            self.connect()

    # ── BaseBroker Implementation ───────────────────────────

    @_retry_ibkr()
    def buy(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order:
        """Placér en købsordre via IBKR."""
        self._validate_order(symbol, qty, order_type, limit_price)
        self._ensure_connected()

        from ib_insync import LimitOrder, MarketOrder

        contract = _detect_ibkr_contract(symbol)
        self._ib.qualifyContracts(contract)

        if order_type == OrderType.LIMIT and limit_price is not None:
            ib_order = LimitOrder("BUY", qty, limit_price)
        else:
            ib_order = MarketOrder("BUY", qty)

        trade = self._ib.placeOrder(contract, ib_order)
        self._ib.sleep(0.5)  # Vent kort på confirmation

        order = self._map_trade(trade, symbol, OrderSide.BUY)
        logger.info(
            f"[ibkr] KØB {qty} {symbol} ({order_type.value}) → {order.order_id}"
        )
        return order

    @_retry_ibkr()
    def sell(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        short: bool = False,
    ) -> Order:
        """Placér en salgsordre via IBKR."""
        if short:
            raise NotImplementedError("Short-selling not yet implemented for this broker")
        self._validate_order(symbol, qty, order_type, limit_price)
        self._ensure_connected()

        from ib_insync import LimitOrder, MarketOrder

        contract = _detect_ibkr_contract(symbol)
        self._ib.qualifyContracts(contract)

        if order_type == OrderType.LIMIT and limit_price is not None:
            ib_order = LimitOrder("SELL", qty, limit_price)
        else:
            ib_order = MarketOrder("SELL", qty)

        trade = self._ib.placeOrder(contract, ib_order)
        self._ib.sleep(0.5)

        order = self._map_trade(trade, symbol, OrderSide.SELL)
        logger.info(
            f"[ibkr] SÆLG {qty} {symbol} ({order_type.value}) → {order.order_id}"
        )
        return order

    @_retry_ibkr()
    def get_positions(self) -> list[Position]:
        """Hent alle åbne positioner fra IBKR."""
        self._ensure_connected()

        ib_positions = self._ib.positions()
        positions: list[Position] = []

        for p in ib_positions:
            symbol = p.contract.localSymbol or p.contract.symbol
            qty = float(p.position)
            side = "long" if qty > 0 else "short"
            avg_cost = float(p.avgCost)

            # Hent current price via marketPrice (hvis tilgængelig)
            current_price = avg_cost  # Fallback
            try:
                ticker = self._ib.reqTickers(p.contract)
                if ticker and ticker[0].marketPrice():
                    current_price = float(ticker[0].marketPrice())
            except Exception:
                pass

            positions.append(Position(
                symbol=symbol,
                side=side,
                qty=abs(qty),
                entry_price=avg_cost,
                entry_time="",
                current_price=current_price,
            ))

        return positions

    @_retry_ibkr()
    def get_account(self) -> AccountInfo:
        """Hent kontoinformation fra IBKR."""
        self._ensure_connected()

        summary = self._ib.accountSummary()

        values: dict[str, float] = {}
        currency = "USD"
        account_id = ""

        for item in summary:
            if item.tag in (
                "TotalCashValue", "NetLiquidation",
                "BuyingPower", "GrossPositionValue",
            ):
                values[item.tag] = float(item.value)
            if item.tag == "NetLiquidation":
                currency = item.currency
                account_id = item.account

        return AccountInfo(
            account_id=account_id or "ibkr",
            cash=values.get("TotalCashValue", 0),
            portfolio_value=values.get("GrossPositionValue", 0),
            buying_power=values.get("BuyingPower", 0),
            equity=values.get("NetLiquidation", 0),
            currency=currency,
        )

    @_retry_ibkr()
    def get_order_status(self, order_id: str) -> Order:
        """Hent status for en specifik ordre."""
        self._ensure_connected()

        for trade in self._ib.trades():
            if str(trade.order.orderId) == str(order_id):
                return self._map_trade(trade)

        raise BrokerError(f"Ordre {order_id} ikke fundet hos IBKR")

    @_retry_ibkr()
    def cancel_order(self, order_id: str) -> bool:
        """Annullér en ordre."""
        self._ensure_connected()

        for trade in self._ib.openTrades():
            if str(trade.order.orderId) == str(order_id):
                self._ib.cancelOrder(trade.order)
                self._ib.sleep(0.5)
                logger.info(f"[ibkr] Ordre {order_id} annulleret")
                return True

        logger.warning(f"[ibkr] Ordre {order_id} ikke fundet")
        return False

    # ── Mapping ─────────────────────────────────────────────

    def _map_trade(
        self,
        trade: Any,
        symbol: str | None = None,
        side: OrderSide | None = None,
    ) -> Order:
        """Map en ib_insync Trade til vores Order model."""
        status_str = trade.orderStatus.status.lower() if trade.orderStatus else "pending"
        status = _IBKR_STATUS_MAP.get(status_str, OrderStatus.PENDING)

        if side is None:
            side = (
                OrderSide.BUY
                if trade.order.action.upper() == "BUY"
                else OrderSide.SELL
            )

        order_type = (
            OrderType.LIMIT
            if trade.order.orderType == "LMT"
            else OrderType.MARKET
        )

        filled_qty = float(trade.orderStatus.filled) if trade.orderStatus else 0
        avg_price = float(trade.orderStatus.avgFillPrice) if trade.orderStatus else 0

        return Order(
            order_id=str(trade.order.orderId),
            symbol=symbol or trade.contract.symbol,
            side=side,
            order_type=order_type,
            qty=float(trade.order.totalQuantity),
            status=status,
            limit_price=(
                float(trade.order.lmtPrice)
                if trade.order.orderType == "LMT"
                else None
            ),
            filled_qty=filled_qty,
            filled_avg_price=avg_price,
            submitted_at=(
                trade.log[-1].time.isoformat()
                if trade.log
                else datetime.now().isoformat()
            ),
        )

    # ── Extra: Instrument Search ────────────────────────────

    def search_instruments(self, query: str) -> list[dict]:
        """Søg efter instrumenter via IBKR contract search."""
        self._ensure_connected()

        try:
            results = self._ib.reqMatchingSymbols(query)
            return [
                {
                    "symbol": r.contract.symbol,
                    "secType": r.contract.secType,
                    "exchange": r.contract.primaryExchange,
                    "currency": r.contract.currency,
                    "description": str(getattr(r, "derivativeSecTypes", "")),
                }
                for r in (results or [])
            ]
        except Exception as exc:
            logger.warning(f"[ibkr] Instrument search fejl: {exc}")
            return []

    # ── Status ──────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        return {
            "broker": "ibkr",
            "connected": self._connected,
            "host": self._host,
            "port": self._port,
            "client_id": self._client_id,
            "is_connected": (
                self._ib.isConnected() if self._ib else False
            ),
        }
