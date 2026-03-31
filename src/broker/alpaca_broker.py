"""
AlpacaBroker – integration med Alpaca Paper/Live Trading API.

Bruger alpaca-trade-api SDK til at placere handler, hente positioner
og overvåge ordrestatus. Inkluderer retry-logik for transiente fejl.
"""

from __future__ import annotations

import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable

import alpaca_trade_api as tradeapi
from loguru import logger

from config.settings import settings
from src.broker.base_broker import BaseBroker
from src.broker.models import (
    AccountInfo,
    BrokerError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.risk.portfolio_tracker import Position


# ── Retry-decorator ───────────────────────────────────────────

def _retry_on_transient(
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> Callable:
    """Decorator: retry med eksponentiel backoff for transiente API-fejl."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except tradeapi.rest.APIError as exc:
                    # 4xx fejl er permanente – retry hjælper ikke
                    code = getattr(exc, "status_code", 500)
                    if 400 <= code < 500:
                        raise BrokerError(f"Alpaca API-fejl ({code}): {exc}") from exc
                    last_exc = exc
                except Exception as exc:
                    last_exc = exc

                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Alpaca API-fejl (forsøg {attempt + 1}/{max_retries}): "
                        f"{last_exc} – prøver igen om {delay:.1f}s"
                    )
                    time.sleep(delay)

            raise BrokerError(
                f"Alpaca API-fejl efter {max_retries} forsøg: {last_exc}"
            ) from last_exc

        return wrapper
    return decorator


# ── Status-mapping ────────────────────────────────────────────

_ALPACA_STATUS_MAP: dict[str, OrderStatus] = {
    "new": OrderStatus.SUBMITTED,
    "accepted": OrderStatus.SUBMITTED,
    "pending_new": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.SUBMITTED,
    "filled": OrderStatus.FILLED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "canceled": OrderStatus.CANCELLED,
    "expired": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "stopped": OrderStatus.CANCELLED,
    "suspended": OrderStatus.CANCELLED,
}


class AlpacaBroker(BaseBroker):
    """
    Broker-implementation for Alpaca Paper/Live Trading.

    Opretter forbindelse via alpaca-trade-api SDK.
    Alle API-kald har automatisk retry ved transiente fejl.
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api = tradeapi.REST(
            key_id=api_key or settings.broker.api_key,
            secret_key=secret_key or settings.broker.secret_key,
            base_url=base_url or settings.broker.base_url,
            api_version=settings.broker.api_version,
        )
        logger.info(
            f"AlpacaBroker forbundet til {base_url or settings.broker.base_url}"
        )

    @property
    def name(self) -> str:
        return "alpaca"

    # ── Ordrer ────────────────────────────────────────────────

    @_retry_on_transient()
    def buy(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order:
        """Placér en købsordre via Alpaca API."""
        self._validate_order(symbol, qty, order_type, limit_price)

        alpaca_order = self._api.submit_order(
            symbol=symbol.upper(),
            qty=qty,
            side="buy",
            type=order_type.value,
            time_in_force="day",
            limit_price=limit_price,
        )

        order = self._map_order(alpaca_order)
        logger.info(
            f"[alpaca] KØB {qty} {symbol} ({order_type.value}) "
            f"→ ordre {order.order_id}"
        )
        return order

    @_retry_on_transient()
    def sell(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        short: bool = False,
    ) -> Order:
        """Placér en salgsordre via Alpaca API."""
        if short:
            raise NotImplementedError("Short-selling not yet implemented for this broker")
        self._validate_order(symbol, qty, order_type, limit_price)

        alpaca_order = self._api.submit_order(
            symbol=symbol.upper(),
            qty=qty,
            side="sell",
            type=order_type.value,
            time_in_force="day",
            limit_price=limit_price,
        )

        order = self._map_order(alpaca_order)
        logger.info(
            f"[alpaca] SÆLG {qty} {symbol} ({order_type.value}) "
            f"→ ordre {order.order_id}"
        )
        return order

    @_retry_on_transient()
    def get_positions(self) -> list[Position]:
        """Hent alle åbne positioner fra Alpaca."""
        alpaca_positions = self._api.list_positions()
        positions = []
        for p in alpaca_positions:
            side = "long" if float(p.qty) > 0 else "short"
            positions.append(
                Position(
                    symbol=p.symbol,
                    side=side,
                    qty=abs(float(p.qty)),
                    entry_price=float(p.avg_entry_price),
                    entry_time="",
                    current_price=float(p.current_price),
                )
            )
        return positions

    @_retry_on_transient()
    def get_account(self) -> AccountInfo:
        """Hent kontoinformation fra Alpaca."""
        acct = self._api.get_account()
        return AccountInfo(
            account_id=acct.id,
            cash=float(acct.cash),
            portfolio_value=float(acct.portfolio_value),
            buying_power=float(acct.buying_power),
            equity=float(acct.equity),
            currency=acct.currency,
        )

    @_retry_on_transient()
    def get_order_status(self, order_id: str) -> Order:
        """Hent status for en specifik ordre fra Alpaca."""
        alpaca_order = self._api.get_order(order_id)
        return self._map_order(alpaca_order)

    def cancel_order(self, order_id: str) -> bool:
        """Annullér en ordre via Alpaca API."""
        try:
            self._api.cancel_order(order_id)
            logger.info(f"[alpaca] Ordre {order_id} annulleret")
            return True
        except tradeapi.rest.APIError as exc:
            # Ikke-transient fejl (f.eks. ordre allerede fyldt) — log og returner False
            logger.warning(f"[alpaca] Kunne ikke annullere {order_id}: {exc}")
            return False

    # ── Mapping ───────────────────────────────────────────────

    @staticmethod
    def _map_order(alpaca_order: Any) -> Order:
        """Konvertér Alpaca-ordre til vores Order-model."""
        status_str = str(alpaca_order.status).lower()
        status = _ALPACA_STATUS_MAP.get(status_str, OrderStatus.PENDING)

        side = OrderSide.BUY if alpaca_order.side == "buy" else OrderSide.SELL
        otype = (
            OrderType.LIMIT
            if alpaca_order.type == "limit"
            else OrderType.MARKET
        )

        filled_price = float(alpaca_order.filled_avg_price or 0)
        filled_qty = float(alpaca_order.filled_qty or 0)
        limit_price = (
            float(alpaca_order.limit_price) if alpaca_order.limit_price else None
        )

        return Order(
            order_id=str(alpaca_order.id),
            symbol=alpaca_order.symbol,
            side=side,
            order_type=otype,
            qty=float(alpaca_order.qty),
            status=status,
            limit_price=limit_price,
            filled_qty=filled_qty,
            filled_avg_price=filled_price,
            submitted_at=str(alpaca_order.submitted_at or ""),
            filled_at=str(alpaca_order.filled_at or ""),
        )
