"""
PaperBroker – lokal handelssimulator med rigtige markedspriser.

Bruger MarketDataFetcher til at hente reelle priser, men udfører
handler lokalt uden API-kald. Perfekt til test og udvikling.

Market-ordrer fyldes øjeblikkeligt til aktuel pris.
Limit-ordrer gemmes som PENDING og tjekkes via process_pending_orders().
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from loguru import logger

from src.broker.base_broker import BaseBroker
from src.broker.models import (
    AccountInfo,
    BrokerError,
    InsufficientFundsError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationError,
)
from src.data.market_data import MarketDataFetcher
from src.fees.fee_calculator import FeeCalculator
from src.risk.portfolio_tracker import PortfolioTracker, Position


class PaperBroker(BaseBroker):
    """
    Lokal papirhandels-simulator.

    Simulerer ordreudførelse med rigtige markedspriser fra yfinance,
    men uden at ramme nogen mægler-API.
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        market_data: MarketDataFetcher | None = None,
        portfolio: PortfolioTracker | None = None,
        broker_fees: str = "paper",
    ) -> None:
        self._initial_capital = initial_capital
        self._market_data = market_data or MarketDataFetcher()
        self._portfolio = portfolio or PortfolioTracker(initial_capital)
        self._orders: dict[str, Order] = {}
        self._pending_orders: list[str] = []
        self._fee_calc = FeeCalculator(broker=broker_fees)

    @property
    def name(self) -> str:
        return "paper"

    @property
    def portfolio(self) -> PortfolioTracker:
        """Adgang til den underliggende portfolio-tracker."""
        return self._portfolio

    # ── Ordrer ────────────────────────────────────────────────

    def buy(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
    ) -> Order:
        """Placér en købsordre. Hvis short position eksisterer → cover short."""
        self._validate_order(symbol, qty, order_type, limit_price)

        order = self._create_order(symbol, OrderSide.BUY, qty, order_type, limit_price)

        if order_type == OrderType.MARKET:
            price = self._get_price(symbol)
            # Tjek om vi har en short position → cover den
            sym = symbol.upper()
            if sym in self._portfolio.positions and self._portfolio.positions[sym].side == "short":
                self._fill_short_close(order, price)
            else:
                self._fill_buy(order, price)
        else:
            self._pending_orders.append(order.order_id)
            order.status = OrderStatus.SUBMITTED
            logger.info(
                f"[paper] Limit-køb {qty} {symbol} @ ${limit_price:.2f} – venter"
            )

        return order

    def sell(
        self,
        symbol: str,
        qty: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        short: bool = False,
    ) -> Order:
        """Placér en salgsordre.

        Args:
            short: Hvis True, åbn en short position (sælg uden at eje).
        """
        self._validate_order(symbol, qty, order_type, limit_price)
        if not short:
            self._validate_sell(symbol, qty)

        order = self._create_order(symbol, OrderSide.SELL, qty, order_type, limit_price)

        if order_type == OrderType.MARKET:
            price = self._get_price(symbol)
            if short:
                self._fill_short_open(order, price)
            else:
                self._fill_sell(order, price)
        else:
            self._pending_orders.append(order.order_id)
            order.status = OrderStatus.SUBMITTED
            logger.info(
                f"[paper] Limit-{'short' if short else 'salg'} {qty} {symbol} "
                f"@ ${limit_price:.2f} – venter"
            )

        return order

    def get_positions(self) -> list[Position]:
        """Hent alle åbne positioner."""
        return list(self._portfolio.positions.values())

    def get_account(self) -> AccountInfo:
        """Hent kontoinformation."""
        return AccountInfo(
            account_id="paper-account",
            cash=self._portfolio.cash,
            portfolio_value=self._portfolio.total_equity,
            buying_power=self._portfolio.cash,
            equity=self._portfolio.total_equity,
        )

    def get_order_status(self, order_id: str) -> Order:
        """Hent status for en specifik ordre."""
        if order_id not in self._orders:
            raise BrokerError(f"Ordre {order_id} findes ikke")
        return self._orders[order_id]

    def cancel_order(self, order_id: str) -> bool:
        """Annullér en ventende ordre."""
        if order_id not in self._orders:
            raise BrokerError(f"Ordre {order_id} findes ikke")

        order = self._orders[order_id]
        if order.status not in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
            logger.warning(
                f"[paper] Kan ikke annullere ordre {order_id} "
                f"med status {order.status.value}"
            )
            return False

        order.status = OrderStatus.CANCELLED
        if order_id in self._pending_orders:
            self._pending_orders.remove(order_id)

        logger.info(f"[paper] Ordre {order_id} annulleret")
        return True

    # ── Limit-ordre behandling ────────────────────────────────

    def process_pending_orders(self) -> list[Order]:
        """
        Tjek alle ventende limit-ordrer mod aktuelle priser.

        Kald denne metode periodisk i trading-loopet for at
        fylde limit-ordrer når prisen rammer limit.

        Returns:
            Liste af ordrer der blev fyldt i dette kald.
        """
        filled: list[Order] = []
        still_pending: list[str] = []

        for order_id in self._pending_orders:
            order = self._orders[order_id]
            try:
                price = self._get_price(order.symbol)
            except BrokerError:
                still_pending.append(order_id)
                continue

            should_fill = False
            if order.limit_price is None:
                # Market-ordre i pending-listen — fyld altid
                should_fill = True
            elif order.side == OrderSide.BUY and price <= order.limit_price:
                should_fill = True
            elif order.side == OrderSide.SELL and price >= order.limit_price:
                should_fill = True

            if should_fill:
                # Fyld til markedspris (ikke limit) — markedet kan have bevæget sig i vores favør
                if order.side == OrderSide.BUY:
                    self._fill_buy(order, price)
                else:
                    self._fill_sell(order, price)
                filled.append(order)
            else:
                still_pending.append(order_id)

        self._pending_orders = still_pending
        return filled

    # ── Interne hjælpefunktioner ──────────────────────────────

    def _create_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        order_type: OrderType,
        limit_price: float | None,
    ) -> Order:
        """Opret et Order-objekt og registrér det."""
        order = Order(
            order_id=f"paper-{uuid4().hex[:8]}",
            symbol=symbol.upper(),
            side=side,
            order_type=order_type,
            qty=qty,
            limit_price=limit_price,
            submitted_at=datetime.now().isoformat(),
        )
        self._orders[order.order_id] = order
        return order

    def _get_price(self, symbol: str) -> float:
        """Hent aktuel pris via MarketDataFetcher."""
        try:
            return self._market_data.get_latest_price(symbol)
        except Exception as exc:
            raise BrokerError(f"Kunne ikke hente pris for {symbol}: {exc}") from exc

    def _fill_buy(self, order: Order, price: float) -> None:
        """Fyld en købsordre med realistiske handelsomkostninger."""
        fee = self._fee_calc.calculate(order.symbol, "buy", order.qty, price)
        cost = order.qty * price + fee.total
        if cost > self._portfolio.cash:
            order.status = OrderStatus.REJECTED
            order.error_message = (
                f"Ikke nok kontanter: kræver ${cost:,.2f} "
                f"(inkl. ${fee.total:,.2f} gebyr), "
                f"har ${self._portfolio.cash:,.2f}"
            )
            raise InsufficientFundsError(order.error_message)

        # Deduct fees from cash
        self._portfolio.cash -= fee.total

        self._portfolio.open_position(
            symbol=order.symbol,
            side="long",
            qty=order.qty,
            price=price,
            timestamp=order.submitted_at,
        )

        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.filled_avg_price = price
        order.filled_at = datetime.now().isoformat()
        order.fees = fee.total

        logger.info(
            f"[paper] KØB {order.qty} {order.symbol} @ ${price:.2f} "
            f"(${cost:,.2f} inkl. gebyr ${fee.total:.2f})"
        )

    def _fill_sell(self, order: Order, price: float) -> None:
        """Fyld en salgsordre (supports partial sells) med realistiske omkostninger."""
        fee = self._fee_calc.calculate(order.symbol, "sell", order.qty, price)

        self._portfolio.close_position(
            symbol=order.symbol,
            price=price,
            reason="signal",
            timestamp=order.submitted_at,
            qty=order.qty,
        )

        # Deduct fees from cash
        self._portfolio.cash -= fee.total

        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.filled_avg_price = price
        order.filled_at = datetime.now().isoformat()
        order.fees = fee.total

        proceeds = order.qty * price
        logger.info(
            f"[paper] SÆLG {order.qty} {order.symbol} @ ${price:.2f} "
            f"(${proceeds:,.2f} minus gebyr ${fee.total:.2f})"
        )

    def _fill_short_open(self, order: Order, price: float) -> None:
        """Åbn en short position — sælg aktier vi ikke ejer."""
        fee = self._fee_calc.calculate(order.symbol, "sell", order.qty, price)

        try:
            self._portfolio.open_position(
                symbol=order.symbol,
                side="short",
                qty=order.qty,
                price=price,
                timestamp=order.submitted_at,
            )
        except ValueError as e:
            # Margin-check fejlede — refunder IKKE fee (den blev aldrig trukket)
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            raise InsufficientFundsError(str(e))

        # Fee trækkes først EFTER vellykket position-åbning
        self._portfolio.cash -= fee.total

        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.filled_avg_price = price
        order.filled_at = datetime.now().isoformat()
        order.fees = fee.total

        proceeds = order.qty * price
        logger.info(
            f"[paper] SHORT {order.qty} {order.symbol} @ ${price:.2f} "
            f"(proceeds=${proceeds:,.2f} minus gebyr ${fee.total:.2f})"
        )

    def _fill_short_close(self, order: Order, price: float) -> None:
        """Luk en short position — køb aktier tilbage."""
        fee = self._fee_calc.calculate(order.symbol, "buy", order.qty, price)
        self._portfolio.cash -= fee.total

        self._portfolio.close_position(
            symbol=order.symbol,
            price=price,
            reason="cover",
            timestamp=order.submitted_at,
        )

        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.filled_avg_price = price
        order.filled_at = datetime.now().isoformat()
        order.fees = fee.total

        cost = order.qty * price
        logger.info(
            f"[paper] COVER {order.qty} {order.symbol} @ ${price:.2f} "
            f"(cost=${cost:,.2f})"
        )

    def _validate_sell(self, symbol: str, qty: float) -> None:
        """Tjek at vi har en position at sælge."""
        symbol = symbol.upper()
        if symbol not in self._portfolio.positions:
            raise OrderValidationError(f"Ingen position i {symbol} at sælge")
        pos = self._portfolio.positions[symbol]
        if qty > pos.qty:
            raise OrderValidationError(
                f"Kan ikke sælge {qty} {symbol} – har kun {pos.qty}"
            )
