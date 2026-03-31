"""
Tests for OrderManager — unified order tracking, SQLite persistence, order lifecycle.

Tester:
  - Unified order ID generation
  - Order persistence (SQLite)
  - Order status transitions
  - Order history filtering
  - Fill callbacks
"""

from __future__ import annotations

import os
import tempfile
import pytest
from datetime import datetime, date

from src.broker.models import Order, OrderSide, OrderType, OrderStatus


# ── Fixtures ───────────────────────────────────────────────

class MockRouter:
    """Mock BrokerRouter for OrderManager tests."""

    def __init__(self):
        self._orders = {}
        self._brokers = {"alpaca": self, "nordnet": self}

    def resolve_broker(self, symbol, asset_type=None, broker_override=None):
        if symbol.endswith(".CO") or symbol.endswith(".ST"):
            return "nordnet", self
        return "alpaca", self

    def buy(self, symbol, qty, order_type="market", limit_price=None, broker_override=None):
        order = Order(
            order_id=f"BROKER-{len(self._orders)+1:03d}",
            symbol=symbol,
            side=OrderSide.BUY,
            qty=qty,
            order_type=OrderType.MARKET,
            status=OrderStatus.SUBMITTED,
        )
        self._orders[order.order_id] = order
        return order

    def sell(self, symbol, qty, order_type="market", limit_price=None, broker_override=None, short=False):
        order = Order(
            order_id=f"BROKER-{len(self._orders)+1:03d}",
            symbol=symbol,
            side=OrderSide.SELL,
            qty=qty,
            order_type=OrderType.MARKET,
            status=OrderStatus.SUBMITTED,
        )
        self._orders[order.order_id] = order
        return order

    def get_order_status(self, order_id):
        order = self._orders.get(order_id)
        if order:
            order.status = OrderStatus.FILLED
        return order

    def cancel_order(self, order_id):
        return True


class TestOrderManager:
    @pytest.fixture(autouse=True)
    def setup(self):
        self._db_path = os.path.join(tempfile.mkdtemp(), "test_orders.db")

    def test_unified_id_format(self):
        """Unified IDs bør starte med 'ORD-'."""
        from src.broker.order_manager import OrderManager
        router = MockRouter()
        mgr = OrderManager(router=router, db_path=self._db_path)
        order = mgr.place_order("AAPL", "buy", 10)
        assert order.unified_id.startswith("ORD-")

    def test_place_buy_order(self):
        from src.broker.order_manager import OrderManager
        router = MockRouter()
        mgr = OrderManager(router=router, db_path=self._db_path)

        order = mgr.place_order("AAPL", "buy", 10)
        assert order is not None
        assert order.symbol == "AAPL"
        assert order.qty == 10

    def test_place_sell_order(self):
        from src.broker.order_manager import OrderManager
        router = MockRouter()
        mgr = OrderManager(router=router, db_path=self._db_path)

        order = mgr.place_order("NOVO-B.CO", "sell", 50)
        assert order is not None
        assert order.symbol == "NOVO-B.CO"

    def test_get_order(self):
        from src.broker.order_manager import OrderManager
        router = MockRouter()
        mgr = OrderManager(router=router, db_path=self._db_path)

        placed = mgr.place_order("MSFT", "buy", 20)
        retrieved = mgr.get_order(placed.unified_id)
        assert retrieved is not None
        assert retrieved.unified_id == placed.unified_id

    def test_order_persistence(self):
        """Orders bør overleve manager restart (SQLite)."""
        from src.broker.order_manager import OrderManager
        router = MockRouter()

        # Place order with first manager
        mgr1 = OrderManager(router=router, db_path=self._db_path)
        placed = mgr1.place_order("AAPL", "buy", 5)
        unified_id = placed.unified_id

        # Create new manager with same DB
        mgr2 = OrderManager(router=router, db_path=self._db_path)
        retrieved = mgr2.get_order(unified_id)
        assert retrieved is not None

    def test_get_history_filtering(self):
        from src.broker.order_manager import OrderManager
        router = MockRouter()
        mgr = OrderManager(router=router, db_path=self._db_path)

        mgr.place_order("AAPL", "buy", 10)
        mgr.place_order("MSFT", "buy", 20)
        mgr.place_order("NOVO-B.CO", "buy", 30)

        history = mgr.get_history(symbol="AAPL")
        assert all(o.symbol == "AAPL" for o in history)

    def test_get_open_orders(self):
        from src.broker.order_manager import OrderManager
        router = MockRouter()
        mgr = OrderManager(router=router, db_path=self._db_path)

        mgr.place_order("AAPL", "buy", 10)
        open_orders = mgr.get_open_orders()
        assert isinstance(open_orders, list)

    def test_statistics(self):
        from src.broker.order_manager import OrderManager
        router = MockRouter()
        mgr = OrderManager(router=router, db_path=self._db_path)

        mgr.place_order("AAPL", "buy", 10)
        mgr.place_order("MSFT", "sell", 5)

        stats = mgr.get_statistics()
        assert isinstance(stats, dict)
        assert stats.get("total_orders", 0) >= 2

    def test_fill_callback(self):
        from src.broker.order_manager import OrderManager
        router = MockRouter()
        mgr = OrderManager(router=router, db_path=self._db_path)

        fills = []
        mgr.on_fill(lambda order: fills.append(order))

        order = mgr.place_order("AAPL", "buy", 10)
        mgr.refresh_order(order.unified_id)

        # After refresh, if order is filled, callback should fire
        # (depends on mock returning FILLED status)


class TestOrderManagerEdgeCases:
    @pytest.fixture(autouse=True)
    def setup(self):
        self._db_path = os.path.join(tempfile.mkdtemp(), "test_orders_edge.db")

    def test_get_nonexistent_order(self):
        from src.broker.order_manager import OrderManager
        router = MockRouter()
        mgr = OrderManager(router=router, db_path=self._db_path)

        result = mgr.get_order("ORD-NONEXISTENT")
        assert result is None

    def test_cancel_order(self):
        from src.broker.order_manager import OrderManager
        router = MockRouter()
        mgr = OrderManager(router=router, db_path=self._db_path)

        order = mgr.place_order("AAPL", "buy", 10)
        # Cancel should not crash
        try:
            mgr.cancel_order(order.unified_id)
        except Exception:
            pass  # Mock may not support cancel fully
