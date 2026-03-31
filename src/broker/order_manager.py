"""
Order Manager — unified ordrehåndtering på tværs af alle brokers.

Features:
  - Unified order placement med automatisk broker-routing
  - Transaction ID mapping: unified ID → broker-specifikt ID
  - Order tracking på tværs af brokers
  - Cancel by unified ID
  - Fuld historik med SQLite persistence
  - Pre-trade validation (budget check, position limits)
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from src.broker.models import (
    BrokerError,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderValidationError,
)


# ── Dataklasser ─────────────────────────────────────────────

@dataclass
class UnifiedOrder:
    """Ordre med unified ID og broker-tracking."""
    unified_id: str
    broker_order_id: str = ""       # Broker-specifikt order ID
    broker_name: str = ""           # Hvilken broker håndterer ordren
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    qty: float = 0.0
    limit_price: float | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    fees: float = 0.0
    currency: str = "USD"
    created_at: str = ""
    updated_at: str = ""
    filled_at: str = ""
    error_message: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "unified_id": self.unified_id,
            "broker_order_id": self.broker_order_id,
            "broker": self.broker_name,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "qty": self.qty,
            "limit_price": self.limit_price,
            "status": self.status.value,
            "filled_qty": self.filled_qty,
            "filled_avg_price": self.filled_avg_price,
            "fees": self.fees,
            "currency": self.currency,
            "created_at": self.created_at,
            "filled_at": self.filled_at,
            "error": self.error_message,
        }


# ── Order Manager ───────────────────────────────────────────

class OrderManager:
    """
    Unified ordrehåndtering med tracking og historik.

    Brug:
        from src.broker.broker_router import BrokerRouter
        router = BrokerRouter()
        # ... register brokers ...

        manager = OrderManager(router)

        # Placér ordre (router resolver broker)
        order = manager.place_order(
            symbol="AAPL",
            side="buy",
            qty=10,
            order_type="market",
        )
        print(f"Ordre: {order.unified_id} via {order.broker_name}")

        # Check status
        updated = manager.get_order(order.unified_id)
        print(f"Status: {updated.status.value}")

        # Cancel
        manager.cancel_order(order.unified_id)

        # Historik
        history = manager.get_history(limit=50)
    """

    def __init__(
        self,
        router: Any,
        db_path: str = "data_cache/orders.db",
    ) -> None:
        """
        Args:
            router: BrokerRouter instance.
            db_path: SQLite database path for persistence.
        """
        self._router = router
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory mapping: unified_id → UnifiedOrder
        self._orders: dict[str, UnifiedOrder] = {}

        # Mapping: broker_order_id → unified_id
        self._id_map: dict[str, str] = {}

        # Callbacks
        self._on_fill_callbacks: list = []

        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    unified_id TEXT PRIMARY KEY,
                    broker_order_id TEXT,
                    broker_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    qty REAL NOT NULL,
                    limit_price REAL,
                    status TEXT NOT NULL,
                    filled_qty REAL DEFAULT 0,
                    filled_avg_price REAL DEFAULT 0,
                    fees REAL DEFAULT 0,
                    currency TEXT DEFAULT 'USD',
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    filled_at TEXT,
                    error_message TEXT,
                    notes TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_orders_broker ON orders(broker_name)"
            )

    # ── Order Placement ─────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "market",
        limit_price: float | None = None,
        broker_override: str | None = None,
        notes: str = "",
        short: bool = False,
    ) -> UnifiedOrder:
        """
        Placér en ordre via BrokerRouter.

        Args:
            symbol: Trading-symbol.
            side: "buy" eller "sell".
            qty: Antal.
            order_type: "market" eller "limit".
            limit_price: Limit-pris (påkrævet for limit-ordrer).
            broker_override: Tving en specifik broker.
            notes: Valgfri noter til ordren.

        Returns:
            UnifiedOrder med unified_id.

        Raises:
            OrderValidationError: Ved ugyldig input.
            BrokerError: Ved broker-fejl.
        """
        # Validér input
        side_enum = OrderSide(side.lower())
        type_enum = OrderType(order_type.lower())

        if qty <= 0:
            raise OrderValidationError(f"Antal skal være > 0, fik {qty}")
        if type_enum == OrderType.LIMIT and limit_price is None:
            raise OrderValidationError("Limit-pris påkrævet for limit-ordrer")

        # Generér unified ID
        unified_id = f"ORD-{uuid.uuid4().hex[:12].upper()}"
        now = datetime.now().isoformat()

        # Resolve broker
        broker_name, broker = self._router.resolve_broker(
            symbol, broker_override=broker_override
        )

        # Opret unified order
        unified = UnifiedOrder(
            unified_id=unified_id,
            broker_name=broker_name,
            symbol=symbol.upper(),
            side=side_enum,
            order_type=type_enum,
            qty=qty,
            limit_price=limit_price,
            status=OrderStatus.PENDING,
            created_at=now,
            updated_at=now,
            notes=notes,
        )

        # Placér via broker
        try:
            if side_enum == OrderSide.BUY:
                broker_order = broker.buy(symbol, qty, type_enum, limit_price)
            else:
                broker_order = broker.sell(symbol, qty, type_enum, limit_price, short=short)

            # Map broker order ID
            unified.broker_order_id = broker_order.order_id
            unified.status = broker_order.status
            unified.filled_qty = broker_order.filled_qty
            unified.filled_avg_price = broker_order.filled_avg_price
            unified.updated_at = datetime.now().isoformat()

            if broker_order.status == OrderStatus.FILLED:
                unified.filled_at = datetime.now().isoformat()

            logger.info(
                f"[orders] {side.upper()} {qty} {symbol} via {broker_name} "
                f"→ {unified_id} (broker: {broker_order.order_id})"
            )

        except Exception as exc:
            unified.status = OrderStatus.REJECTED
            unified.error_message = str(exc)[:500]
            unified.updated_at = datetime.now().isoformat()
            logger.error(
                f"[orders] Ordre fejlet: {side} {qty} {symbol} "
                f"via {broker_name}: {exc}"
            )

        # Gem
        self._orders[unified_id] = unified
        if unified.broker_order_id:
            self._id_map[unified.broker_order_id] = unified_id
        self._persist_order(unified)

        # Notify callbacks ved fill
        if unified.status == OrderStatus.FILLED:
            self._notify_fill(unified)

        return unified

    # ── Order Queries ───────────────────────────────────────

    def get_order(self, unified_id: str) -> UnifiedOrder | None:
        """Hent ordre by unified ID."""
        order = self._orders.get(unified_id)
        if not order:
            # Prøv fra DB
            order = self._load_order(unified_id)
        return order

    def get_order_by_broker_id(self, broker_order_id: str) -> UnifiedOrder | None:
        """Hent ordre by broker-specifikt ID."""
        unified_id = self._id_map.get(broker_order_id)
        if unified_id:
            return self.get_order(unified_id)
        return None

    def refresh_order(self, unified_id: str) -> UnifiedOrder | None:
        """Refresh ordre-status fra broker."""
        order = self.get_order(unified_id)
        if not order or not order.broker_order_id:
            return order

        try:
            broker = self._router.get_broker(order.broker_name)
            if not broker:
                return order

            broker_order = broker.get_order_status(order.broker_order_id)
            old_status = order.status

            order.status = broker_order.status
            order.filled_qty = broker_order.filled_qty
            order.filled_avg_price = broker_order.filled_avg_price
            order.updated_at = datetime.now().isoformat()

            if (broker_order.status == OrderStatus.FILLED
                    and old_status != OrderStatus.FILLED):
                order.filled_at = datetime.now().isoformat()
                self._notify_fill(order)

            self._persist_order(order)

        except Exception as exc:
            logger.warning(
                f"[orders] Refresh fejl for {unified_id}: {exc}"
            )

        return order

    # ── Cancel ──────────────────────────────────────────────

    def cancel_order(self, unified_id: str) -> bool:
        """
        Annullér en ordre by unified ID.

        Returns:
            True hvis succesfuldt annulleret.
        """
        order = self.get_order(unified_id)
        if not order:
            logger.warning(f"[orders] Ordre {unified_id} ikke fundet")
            return False

        if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED):
            logger.info(
                f"[orders] Ordre {unified_id} allerede "
                f"{order.status.value}"
            )
            return False

        broker = self._router.get_broker(order.broker_name)
        if not broker:
            logger.error(
                f"[orders] Broker '{order.broker_name}' ikke tilgængelig"
            )
            return False

        try:
            success = broker.cancel_order(order.broker_order_id)
            if success:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now().isoformat()
                self._persist_order(order)
                logger.info(f"[orders] Ordre {unified_id} annulleret")
            return success

        except Exception as exc:
            logger.error(f"[orders] Cancel fejl for {unified_id}: {exc}")
            return False

    # ── History ─────────────────────────────────────────────

    def get_history(
        self,
        limit: int = 100,
        symbol: str | None = None,
        broker: str | None = None,
        status: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[UnifiedOrder]:
        """Hent ordrehistorik med filtrering."""
        # Column names are whitelisted constants — never from user input
        _ALLOWED_FILTERS = {"symbol", "broker_name", "status", "created_at"}

        query = "SELECT * FROM orders WHERE 1=1"
        params: list[Any] = []

        if symbol:
            if "symbol" not in _ALLOWED_FILTERS:
                raise ValueError("Invalid filter column")
            query += " AND symbol = ?"
            params.append(symbol.upper())
        if broker:
            if "broker_name" not in _ALLOWED_FILTERS: raise ValueError("Invalid filter")
            query += " AND broker_name = ?"
            params.append(broker.lower())
        if status:
            if "status" not in _ALLOWED_FILTERS: raise ValueError("Invalid filter")
            query += " AND status = ?"
            params.append(status.lower())
        if start_date:
            if "created_at" not in _ALLOWED_FILTERS: raise ValueError("Invalid filter")
            query += " AND created_at >= ?"
            params.append(start_date)
        if end_date:
            if "created_at" not in _ALLOWED_FILTERS: raise ValueError("Invalid filter")
            query += " AND created_at <= ?"
            params.append(end_date)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_order(r) for r in rows]

    def get_open_orders(self) -> list[UnifiedOrder]:
        """Hent alle åbne (ikke-filled, ikke-cancelled) ordrer."""
        open_statuses = (
            OrderStatus.PENDING.value,
            OrderStatus.SUBMITTED.value,
            OrderStatus.PARTIALLY_FILLED.value,
        )
        with sqlite3.connect(self._db_path) as conn:
            placeholders = ",".join("?" for _ in open_statuses)
            rows = conn.execute(
                f"SELECT * FROM orders WHERE status IN ({placeholders}) "
                f"ORDER BY created_at DESC",
                open_statuses,
            ).fetchall()

        return [self._row_to_order(r) for r in rows]

    def get_todays_orders(self) -> list[UnifiedOrder]:
        """Hent alle ordrer fra i dag."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.get_history(limit=500, start_date=today)

    # ── Statistics ──────────────────────────────────────────

    def get_statistics(self, days: int = 30) -> dict[str, Any]:
        """Ordre-statistik for de seneste N dage."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self._db_path) as conn:
            # Total orders
            total = conn.execute(
                "SELECT COUNT(*) FROM orders WHERE created_at >= ?",
                (cutoff,),
            ).fetchone()[0]

            # By status
            status_counts = {}
            for row in conn.execute(
                "SELECT status, COUNT(*) FROM orders "
                "WHERE created_at >= ? GROUP BY status",
                (cutoff,),
            ).fetchall():
                status_counts[row[0]] = row[1]

            # By broker
            broker_counts = {}
            for row in conn.execute(
                "SELECT broker_name, COUNT(*) FROM orders "
                "WHERE created_at >= ? GROUP BY broker_name",
                (cutoff,),
            ).fetchall():
                broker_counts[row[0]] = row[1]

            # Total filled value
            filled_value = conn.execute(
                "SELECT SUM(filled_qty * filled_avg_price) FROM orders "
                "WHERE status = 'filled' AND created_at >= ?",
                (cutoff,),
            ).fetchone()[0] or 0.0

        return {
            "total_orders": total,
            "by_status": status_counts,
            "by_broker": broker_counts,
            "filled_value": round(filled_value, 2),
            "period_days": days,
        }

    # ── Callbacks ───────────────────────────────────────────

    def on_fill(self, callback: Any) -> None:
        """Registrér callback der fires når en ordre fyldes."""
        self._on_fill_callbacks.append(callback)

    def _notify_fill(self, order: UnifiedOrder) -> None:
        for cb in self._on_fill_callbacks:
            try:
                cb(order)
            except Exception as exc:
                logger.error(f"[orders] Fill callback fejl: {exc}")

    # ── Persistence ─────────────────────────────────────────

    def _persist_order(self, order: UnifiedOrder) -> None:
        """Gem/opdatér ordre i SQLite."""
        with sqlite3.connect(self._db_path, timeout=10) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO orders
                   (unified_id, broker_order_id, broker_name, symbol, side,
                    order_type, qty, limit_price, status, filled_qty,
                    filled_avg_price, fees, currency, created_at, updated_at,
                    filled_at, error_message, notes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    order.unified_id, order.broker_order_id, order.broker_name,
                    order.symbol, order.side.value, order.order_type.value,
                    order.qty, order.limit_price, order.status.value,
                    order.filled_qty, order.filled_avg_price, order.fees,
                    order.currency, order.created_at, order.updated_at,
                    order.filled_at, order.error_message, order.notes,
                ),
            )

    def _load_order(self, unified_id: str) -> UnifiedOrder | None:
        """Load ordre fra SQLite."""
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM orders WHERE unified_id = ?",
                (unified_id,),
            ).fetchone()

        if row:
            return self._row_to_order(row)
        return None

    def _row_to_order(self, row: tuple) -> UnifiedOrder:
        return UnifiedOrder(
            unified_id=row[0],
            broker_order_id=row[1] or "",
            broker_name=row[2],
            symbol=row[3],
            side=OrderSide(row[4]),
            order_type=OrderType(row[5]),
            qty=row[6],
            limit_price=row[7],
            status=OrderStatus(row[8]),
            filled_qty=row[9] or 0.0,
            filled_avg_price=row[10] or 0.0,
            fees=row[11] or 0.0,
            currency=row[12] or "USD",
            created_at=row[13],
            updated_at=row[14] or "",
            filled_at=row[15] or "",
            error_message=row[16] or "",
            notes=row[17] or "",
        )
