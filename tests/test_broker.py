"""
Tests for broker-modulet: models, base_broker, paper_broker, alpaca_broker, factory.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock

import pytest

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
from src.broker.base_broker import BaseBroker
from src.broker.paper_broker import PaperBroker
from src.broker.alpaca_broker import AlpacaBroker, _ALPACA_STATUS_MAP
from src.broker import create_broker


# ══════════════════════════════════════════════════════════════
#  Test Models
# ══════════════════════════════════════════════════════════════


class TestModels:
    def test_order_defaults(self):
        o = Order(
            order_id="t1", symbol="AAPL", side=OrderSide.BUY,
            order_type=OrderType.MARKET, qty=10,
        )
        assert o.status == OrderStatus.PENDING
        assert o.filled_qty == 0.0
        assert o.filled_avg_price == 0.0

    def test_order_limit(self):
        o = Order(
            order_id="t2", symbol="MSFT", side=OrderSide.SELL,
            order_type=OrderType.LIMIT, qty=5, limit_price=300.0,
        )
        assert o.limit_price == 300.0
        assert o.order_type == OrderType.LIMIT

    def test_account_info(self):
        a = AccountInfo(
            account_id="acc1", cash=50_000, portfolio_value=100_000,
            buying_power=50_000, equity=100_000,
        )
        assert a.currency == "USD"
        assert a.cash == 50_000

    def test_broker_error_hierarchy(self):
        assert issubclass(OrderValidationError, BrokerError)
        assert issubclass(InsufficientFundsError, BrokerError)


# ══════════════════════════════════════════════════════════════
#  Test BaseBroker
# ══════════════════════════════════════════════════════════════


class TestBaseBroker:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseBroker()

    def test_validate_order_empty_symbol(self):
        broker = self._make_concrete()
        with pytest.raises(OrderValidationError, match="Symbol"):
            broker._validate_order("", 10, OrderType.MARKET, None)

    def test_validate_order_zero_qty(self):
        broker = self._make_concrete()
        with pytest.raises(OrderValidationError, match="Antal"):
            broker._validate_order("AAPL", 0, OrderType.MARKET, None)

    def test_validate_order_negative_qty(self):
        broker = self._make_concrete()
        with pytest.raises(OrderValidationError, match="Antal"):
            broker._validate_order("AAPL", -5, OrderType.MARKET, None)

    def test_validate_order_limit_without_price(self):
        broker = self._make_concrete()
        with pytest.raises(OrderValidationError, match="Limit-pris kræves"):
            broker._validate_order("AAPL", 10, OrderType.LIMIT, None)

    def test_validate_order_negative_limit_price(self):
        broker = self._make_concrete()
        with pytest.raises(OrderValidationError, match="Limit-pris"):
            broker._validate_order("AAPL", 10, OrderType.LIMIT, -50.0)

    def test_validate_order_ok(self):
        broker = self._make_concrete()
        broker._validate_order("AAPL", 10, OrderType.MARKET, None)
        broker._validate_order("AAPL", 10, OrderType.LIMIT, 150.0)

    @staticmethod
    def _make_concrete():
        """Opret en minimal konkret subklasse til test af validering."""
        class _ConcreteBroker(BaseBroker):
            @property
            def name(self): return "test"
            def buy(self, *a, **kw): pass
            def sell(self, *a, **kw): pass
            def get_positions(self): return []
            def get_account(self): return None
            def get_order_status(self, oid): return None
            def cancel_order(self, oid): return False
        return _ConcreteBroker()


# ══════════════════════════════════════════════════════════════
#  Test PaperBroker – Market Orders
# ══════════════════════════════════════════════════════════════


def _make_paper_broker(initial_capital: float = 100_000, price: float = 150.0):
    """Opret PaperBroker med mocket MarketDataFetcher."""
    mock_md = MagicMock()
    mock_md.get_latest_price.return_value = price
    return PaperBroker(initial_capital=initial_capital, market_data=mock_md)


class TestPaperBrokerMarketOrders:
    def test_buy_market_fills_immediately(self):
        pb = _make_paper_broker(price=150.0)
        order = pb.buy("AAPL", 10)

        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == 10
        assert order.filled_avg_price == 150.0
        assert order.side == OrderSide.BUY

    def test_buy_reduces_cash(self):
        pb = _make_paper_broker(price=100.0)
        pb.buy("AAPL", 10)
        acct = pb.get_account()
        # 100k - 10*100 - fee (0.15% of 1000 = 1.50) = 98998.50
        assert acct.cash == pytest.approx(98_998.5)

    def test_buy_creates_position(self):
        pb = _make_paper_broker(price=100.0)
        pb.buy("AAPL", 10)
        positions = pb.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].qty == 10

    def test_sell_market_fills_immediately(self):
        pb = _make_paper_broker(price=100.0)
        pb.buy("AAPL", 10)

        pb._market_data.get_latest_price.return_value = 110.0
        order = pb.sell("AAPL", 10)

        assert order.status == OrderStatus.FILLED
        assert order.filled_avg_price == 110.0

    def test_sell_removes_position(self):
        pb = _make_paper_broker(price=100.0)
        pb.buy("AAPL", 10)
        pb.sell("AAPL", 10)
        assert len(pb.get_positions()) == 0

    def test_sell_returns_cash(self):
        pb = _make_paper_broker(price=100.0)
        pb.buy("AAPL", 10)
        pb._market_data.get_latest_price.return_value = 120.0
        pb.sell("AAPL", 10)
        acct = pb.get_account()
        # Started 100k, bought 10@100=1000+fee1.50, sold 10@120=1200-fee1.80
        # cash = 100000 - 1001.50 + 1198.20 = 100196.70
        assert acct.cash == pytest.approx(100_196.7)

    def test_multiple_buys_different_symbols(self):
        pb = _make_paper_broker(price=100.0)
        pb.buy("AAPL", 10)
        pb._market_data.get_latest_price.return_value = 200.0
        pb.buy("MSFT", 5)
        assert len(pb.get_positions()) == 2

    def test_order_id_unique(self):
        pb = _make_paper_broker(price=100.0)
        o1 = pb.buy("AAPL", 10)
        pb._market_data.get_latest_price.return_value = 200.0
        pb.buy("MSFT", 5)  # nyt symbol
        pb.sell("AAPL", 10)
        o3 = pb.get_order_status(o1.order_id)
        assert o3.order_id == o1.order_id

    def test_symbol_uppercased(self):
        pb = _make_paper_broker(price=100.0)
        order = pb.buy("aapl", 10)
        assert order.symbol == "AAPL"


# ══════════════════════════════════════════════════════════════
#  Test PaperBroker – Limit Orders
# ══════════════════════════════════════════════════════════════


class TestPaperBrokerLimitOrders:
    def test_limit_buy_stays_pending(self):
        pb = _make_paper_broker(price=150.0)
        order = pb.buy("AAPL", 10, OrderType.LIMIT, limit_price=140.0)
        assert order.status == OrderStatus.SUBMITTED
        assert len(pb.get_positions()) == 0

    def test_limit_buy_fills_when_price_drops(self):
        pb = _make_paper_broker(price=150.0)
        order = pb.buy("AAPL", 10, OrderType.LIMIT, limit_price=140.0)

        # Pris falder til 140
        pb._market_data.get_latest_price.return_value = 140.0
        filled = pb.process_pending_orders()

        assert len(filled) == 1
        assert filled[0].order_id == order.order_id
        assert filled[0].status == OrderStatus.FILLED
        assert len(pb.get_positions()) == 1

    def test_limit_buy_not_filled_when_price_above(self):
        pb = _make_paper_broker(price=150.0)
        pb.buy("AAPL", 10, OrderType.LIMIT, limit_price=140.0)

        pb._market_data.get_latest_price.return_value = 145.0
        filled = pb.process_pending_orders()

        assert len(filled) == 0
        assert len(pb.get_positions()) == 0

    def test_limit_sell_fills_when_price_rises(self):
        pb = _make_paper_broker(price=100.0)
        pb.buy("AAPL", 10)  # Køb først

        pb._market_data.get_latest_price.return_value = 100.0
        order = pb.sell("AAPL", 10, OrderType.LIMIT, limit_price=110.0)
        assert order.status == OrderStatus.SUBMITTED

        # Pris stiger til 110
        pb._market_data.get_latest_price.return_value = 110.0
        filled = pb.process_pending_orders()

        assert len(filled) == 1
        assert filled[0].status == OrderStatus.FILLED

    def test_cancel_pending_limit(self):
        pb = _make_paper_broker(price=150.0)
        order = pb.buy("AAPL", 10, OrderType.LIMIT, limit_price=140.0)

        assert pb.cancel_order(order.order_id)
        assert pb.get_order_status(order.order_id).status == OrderStatus.CANCELLED

        # Skal ikke fyldes efter cancel
        pb._market_data.get_latest_price.return_value = 130.0
        filled = pb.process_pending_orders()
        assert len(filled) == 0


# ══════════════════════════════════════════════════════════════
#  Test PaperBroker – Account & Positions
# ══════════════════════════════════════════════════════════════


class TestPaperBrokerAccount:
    def test_initial_account(self):
        pb = _make_paper_broker()
        acct = pb.get_account()
        assert acct.account_id == "paper-account"
        assert acct.cash == 100_000
        assert acct.equity == 100_000

    def test_equity_includes_positions(self):
        pb = _make_paper_broker(price=100.0)
        pb.buy("AAPL", 100)
        acct = pb.get_account()
        # cash = 100k - 10000 - fee(15) = 89985, position = 100*100 = 10000
        # equity = 89985 + 10000 = 99985
        assert acct.equity == pytest.approx(99_985)

    def test_portfolio_property(self):
        pb = _make_paper_broker()
        assert pb.portfolio is not None
        assert pb.portfolio.initial_capital == 100_000

    def test_name(self):
        pb = _make_paper_broker()
        assert pb.name == "paper"


# ══════════════════════════════════════════════════════════════
#  Test PaperBroker – Edge Cases
# ══════════════════════════════════════════════════════════════


class TestPaperBrokerEdgeCases:
    def test_insufficient_funds(self):
        pb = _make_paper_broker(initial_capital=1_000, price=200.0)
        with pytest.raises(InsufficientFundsError):
            pb.buy("AAPL", 10)  # 10 * 200 = 2000 > 1000

    def test_sell_without_position(self):
        pb = _make_paper_broker()
        with pytest.raises(OrderValidationError, match="Ingen position"):
            pb.sell("AAPL", 10)

    def test_sell_more_than_owned(self):
        pb = _make_paper_broker(price=100.0)
        pb.buy("AAPL", 5)
        with pytest.raises(OrderValidationError, match="har kun 5"):
            pb.sell("AAPL", 10)

    def test_cancel_filled_order_fails(self):
        pb = _make_paper_broker(price=100.0)
        order = pb.buy("AAPL", 10)
        assert not pb.cancel_order(order.order_id)

    def test_get_nonexistent_order(self):
        pb = _make_paper_broker()
        with pytest.raises(BrokerError, match="findes ikke"):
            pb.get_order_status("fake-id")

    def test_cancel_nonexistent_order(self):
        pb = _make_paper_broker()
        with pytest.raises(BrokerError, match="findes ikke"):
            pb.cancel_order("fake-id")

    def test_duplicate_buy_same_symbol(self):
        pb = _make_paper_broker(price=100.0)
        pb.buy("AAPL", 10)
        with pytest.raises(Exception):
            pb.buy("AAPL", 5)  # PortfolioTracker rejects duplicates

    def test_market_data_failure(self):
        pb = _make_paper_broker()
        pb._market_data.get_latest_price.side_effect = Exception("API ned")
        with pytest.raises(BrokerError, match="Kunne ikke hente pris"):
            pb.buy("AAPL", 10)


# ══════════════════════════════════════════════════════════════
#  Test AlpacaBroker (mocked)
# ══════════════════════════════════════════════════════════════


class _FakeAlpacaOrder:
    """Simulerer et Alpaca-ordre-objekt."""

    def __init__(self, **kwargs):
        self.id = kwargs.get("id", "alp-123")
        self.symbol = kwargs.get("symbol", "AAPL")
        self.side = kwargs.get("side", "buy")
        self.type = kwargs.get("type", "market")
        self.qty = kwargs.get("qty", "10")
        self.status = kwargs.get("status", "filled")
        self.limit_price = kwargs.get("limit_price", None)
        self.filled_qty = kwargs.get("filled_qty", "10")
        self.filled_avg_price = kwargs.get("filled_avg_price", "150.00")
        self.submitted_at = kwargs.get("submitted_at", "2025-01-01T10:00:00Z")
        self.filled_at = kwargs.get("filled_at", "2025-01-01T10:00:01Z")


class _FakeAlpacaPosition:
    def __init__(self, symbol="AAPL", qty="10", avg_entry_price="150.0",
                 current_price="155.0"):
        self.symbol = symbol
        self.qty = qty
        self.avg_entry_price = avg_entry_price
        self.current_price = current_price


class _FakeAlpacaAccount:
    def __init__(self):
        self.id = "acc-456"
        self.cash = "50000.0"
        self.portfolio_value = "100000.0"
        self.buying_power = "50000.0"
        self.equity = "100000.0"
        self.currency = "USD"


class TestAlpacaBroker:
    @patch("src.broker.alpaca_broker.tradeapi.REST")
    def test_buy_calls_submit_order(self, mock_rest_cls):
        mock_api = MagicMock()
        mock_rest_cls.return_value = mock_api
        mock_api.submit_order.return_value = _FakeAlpacaOrder(side="buy")

        broker = AlpacaBroker(api_key="k", secret_key="s", base_url="http://fake")
        order = broker.buy("AAPL", 10)

        mock_api.submit_order.assert_called_once_with(
            symbol="AAPL", qty=10, side="buy",
            type="market", time_in_force="day", limit_price=None,
        )
        assert order.status == OrderStatus.FILLED
        assert order.symbol == "AAPL"

    @patch("src.broker.alpaca_broker.tradeapi.REST")
    def test_sell_calls_submit_order(self, mock_rest_cls):
        mock_api = MagicMock()
        mock_rest_cls.return_value = mock_api
        mock_api.submit_order.return_value = _FakeAlpacaOrder(side="sell")

        broker = AlpacaBroker(api_key="k", secret_key="s", base_url="http://fake")
        order = broker.sell("AAPL", 10)

        mock_api.submit_order.assert_called_once_with(
            symbol="AAPL", qty=10, side="sell",
            type="market", time_in_force="day", limit_price=None,
        )
        assert order.side == OrderSide.SELL

    @patch("src.broker.alpaca_broker.tradeapi.REST")
    def test_limit_order(self, mock_rest_cls):
        mock_api = MagicMock()
        mock_rest_cls.return_value = mock_api
        mock_api.submit_order.return_value = _FakeAlpacaOrder(
            type="limit", status="new", limit_price="140.0",
            filled_qty="0", filled_avg_price=None,
        )

        broker = AlpacaBroker(api_key="k", secret_key="s", base_url="http://fake")
        order = broker.buy("AAPL", 10, OrderType.LIMIT, limit_price=140.0)

        assert order.order_type == OrderType.LIMIT
        assert order.status == OrderStatus.SUBMITTED

    @patch("src.broker.alpaca_broker.tradeapi.REST")
    def test_get_positions(self, mock_rest_cls):
        mock_api = MagicMock()
        mock_rest_cls.return_value = mock_api
        mock_api.list_positions.return_value = [
            _FakeAlpacaPosition("AAPL", "10", "150.0", "155.0"),
            _FakeAlpacaPosition("MSFT", "5", "300.0", "310.0"),
        ]

        broker = AlpacaBroker(api_key="k", secret_key="s", base_url="http://fake")
        positions = broker.get_positions()

        assert len(positions) == 2
        assert positions[0].symbol == "AAPL"
        assert positions[0].qty == 10
        assert positions[1].symbol == "MSFT"

    @patch("src.broker.alpaca_broker.tradeapi.REST")
    def test_get_account(self, mock_rest_cls):
        mock_api = MagicMock()
        mock_rest_cls.return_value = mock_api
        mock_api.get_account.return_value = _FakeAlpacaAccount()

        broker = AlpacaBroker(api_key="k", secret_key="s", base_url="http://fake")
        acct = broker.get_account()

        assert acct.account_id == "acc-456"
        assert acct.cash == 50_000
        assert acct.equity == 100_000

    @patch("src.broker.alpaca_broker.tradeapi.REST")
    def test_cancel_order_success(self, mock_rest_cls):
        mock_api = MagicMock()
        mock_rest_cls.return_value = mock_api

        broker = AlpacaBroker(api_key="k", secret_key="s", base_url="http://fake")
        assert broker.cancel_order("order-1") is True

    @patch("src.broker.alpaca_broker.tradeapi.REST")
    def test_cancel_order_failure(self, mock_rest_cls):
        import alpaca_trade_api as tradeapi_mod
        mock_api = MagicMock()
        mock_rest_cls.return_value = mock_api
        mock_api.cancel_order.side_effect = tradeapi_mod.rest.APIError({"message": "not found"})

        broker = AlpacaBroker(api_key="k", secret_key="s", base_url="http://fake")
        assert broker.cancel_order("order-1") is False

    @patch("src.broker.alpaca_broker.tradeapi.REST")
    def test_name(self, mock_rest_cls):
        mock_rest_cls.return_value = MagicMock()
        broker = AlpacaBroker(api_key="k", secret_key="s", base_url="http://fake")
        assert broker.name == "alpaca"

    def test_status_mapping_completeness(self):
        expected = {
            "new", "accepted", "pending_new", "accepted_for_bidding",
            "filled", "partially_filled", "canceled", "expired",
            "rejected", "stopped", "suspended",
        }
        assert set(_ALPACA_STATUS_MAP.keys()) == expected


# ══════════════════════════════════════════════════════════════
#  Test AlpacaBroker – Retry
# ══════════════════════════════════════════════════════════════


class TestAlpacaBrokerRetry:
    @patch("src.broker.alpaca_broker.time.sleep")
    @patch("src.broker.alpaca_broker.tradeapi.REST")
    def test_retry_succeeds_on_second_attempt(self, mock_rest_cls, mock_sleep):
        mock_api = MagicMock()
        mock_rest_cls.return_value = mock_api
        mock_api.submit_order.side_effect = [
            ConnectionError("timeout"),
            _FakeAlpacaOrder(side="buy"),
        ]

        broker = AlpacaBroker(api_key="k", secret_key="s", base_url="http://fake")
        order = broker.buy("AAPL", 10)

        assert order.status == OrderStatus.FILLED
        assert mock_api.submit_order.call_count == 2
        mock_sleep.assert_called_once()

    @patch("src.broker.alpaca_broker.time.sleep")
    @patch("src.broker.alpaca_broker.tradeapi.REST")
    def test_retry_exhausted_raises(self, mock_rest_cls, mock_sleep):
        mock_api = MagicMock()
        mock_rest_cls.return_value = mock_api
        mock_api.submit_order.side_effect = ConnectionError("timeout")

        broker = AlpacaBroker(api_key="k", secret_key="s", base_url="http://fake")
        with pytest.raises(BrokerError, match="efter 3 forsøg"):
            broker.buy("AAPL", 10)

        assert mock_api.submit_order.call_count == 3


# ══════════════════════════════════════════════════════════════
#  Test Factory
# ══════════════════════════════════════════════════════════════


class TestBrokerFactory:
    def test_create_paper_broker(self):
        broker = create_broker("paper")
        assert isinstance(broker, PaperBroker)
        assert broker.name == "paper"

    @patch("src.broker.alpaca_broker.tradeapi.REST")
    def test_create_alpaca_broker(self, mock_rest_cls):
        mock_rest_cls.return_value = MagicMock()
        broker = create_broker("alpaca")
        assert isinstance(broker, AlpacaBroker)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Ukendt broker"):
            create_broker("robinhood")

    def test_create_paper_with_kwargs(self):
        broker = create_broker("paper", initial_capital=50_000)
        assert isinstance(broker, PaperBroker)
        assert broker.get_account().cash == 50_000
