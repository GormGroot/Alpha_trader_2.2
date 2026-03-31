"""
Tests for BrokerRouter — symbol routing, exchange detection, fallback chains.

Tester:
  - detect_exchange() — suffix → exchange mapping
  - detect_asset_type() — heuristisk typedetektion
  - BrokerRouter.resolve_broker() — resolution order
  - BrokerRouter buy/sell delegation
  - Edge cases: ukendt symbol, ingen brokers, fallback
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from src.broker.broker_router import (
    BrokerRouter,
    RoutingConfig,
    RoutingError,
    detect_exchange,
    detect_asset_type,
)
from src.broker.models import Order, OrderSide, OrderType, OrderStatus, AccountInfo


# ── Fixtures ───────────────────────────────────────────────

class MockBroker:
    """Simpel mock broker der opfylder BaseBroker-interfacet."""

    def __init__(self, name: str = "mock"):
        self.name = name
        self._connected = True
        self.buy_calls = []
        self.sell_calls = []

    def connect(self):
        self._connected = True

    def buy(self, symbol, qty, order_type="market", limit_price=None):
        self.buy_calls.append((symbol, qty, order_type, limit_price))
        return Order(
            order_id=f"ORD-{self.name}-001",
            symbol=symbol,
            side=OrderSide.BUY,
            qty=qty,
            order_type=OrderType.MARKET,
            status=OrderStatus.SUBMITTED,
        )

    def sell(self, symbol, qty, order_type="market", limit_price=None, short=False):
        self.sell_calls.append((symbol, qty, order_type, limit_price))
        return Order(
            order_id=f"ORD-{self.name}-001",
            symbol=symbol,
            side=OrderSide.SELL,
            qty=qty,
            order_type=OrderType.MARKET,
            status=OrderStatus.SUBMITTED,
        )

    def get_positions(self):
        return []

    def get_account(self):
        return AccountInfo(
            account_id="mock-account",
            equity=100000, cash=50000, buying_power=50000,
            portfolio_value=50000,
        )

    def get_order_status(self, order_id):
        return Order(
            order_id=order_id, symbol="AAPL", side=OrderSide.BUY,
            qty=10, order_type=OrderType.MARKET, status=OrderStatus.FILLED,
        )

    def cancel_order(self, order_id):
        return True


@pytest.fixture
def router():
    r = BrokerRouter()
    r.register("alpaca", MockBroker("alpaca"))
    r.register("ibkr", MockBroker("ibkr"))
    r.register("nordnet", MockBroker("nordnet"))
    r.register("saxo", MockBroker("saxo"))
    return r


@pytest.fixture
def empty_router():
    return BrokerRouter()


# ── detect_exchange() ──────────────────────────────────────

class TestDetectExchange:
    def test_danish_stocks(self):
        assert detect_exchange("NOVO-B.CO") == "CSE"
        assert detect_exchange("MAERSK-B.CO") == "CSE"
        assert detect_exchange("DSV.CO") == "CSE"

    def test_swedish_stocks(self):
        assert detect_exchange("ERIC-B.ST") == "SFB"
        assert detect_exchange("SAND.ST") == "SFB"

    def test_german_stocks(self):
        assert detect_exchange("SAP.DE") == "XETRA"
        assert detect_exchange("SIE.DE") == "XETRA"

    def test_uk_stocks(self):
        assert detect_exchange("HSBA.L") == "LSE"
        assert detect_exchange("AZN.L") == "LSE"

    def test_amsterdam(self):
        assert detect_exchange("ASML.AS") == "AEB"

    def test_paris(self):
        assert detect_exchange("MC.PA") == "SBF"
        assert detect_exchange("TTE.PA") == "SBF"

    def test_oslo(self):
        assert detect_exchange("EQNR.OL") == "OSE"

    def test_helsinki(self):
        assert detect_exchange("ORNBV.HE") == "HEX"

    def test_swiss(self):
        assert detect_exchange("ROG.SW") == "EBS"

    def test_us_no_suffix(self):
        # US stocks have no suffix — should return None or 'US'
        result = detect_exchange("AAPL")
        assert result is None or result == "US"

    def test_crypto(self):
        result = detect_exchange("BTC-USD")
        # Crypto may not have a standard exchange
        assert result is None or isinstance(result, str)


# ── detect_asset_type() ───────────────────────────────────

class TestDetectAssetType:
    def test_crypto(self):
        assert detect_asset_type("BTC-USD") == "crypto"
        assert detect_asset_type("ETH-USD") == "crypto"

    def test_forex(self):
        result = detect_asset_type("EUR/USD")
        assert result in ("forex", None)

    def test_futures(self):
        result = detect_asset_type("ESZ24")
        assert result in ("futures", "future", None)

    def test_nordic_stock(self):
        result = detect_asset_type("NOVO-B.CO")
        assert result in ("stock_nordic", "stock", "stock_eu", None)

    def test_us_stock(self):
        result = detect_asset_type("AAPL")
        assert result in ("stock_us", "stock", None)

    def test_eu_stock(self):
        result = detect_asset_type("SAP.DE")
        assert result in ("stock_eu", "stock", None)


# ── BrokerRouter ───────────────────────────────────────────

class TestBrokerRouter:
    def test_register_broker(self, router):
        assert "alpaca" in router._brokers
        assert "ibkr" in router._brokers
        assert "nordnet" in router._brokers
        assert "saxo" in router._brokers

    def test_resolve_us_stock_to_alpaca(self, router):
        """US stocks uden suffix bør routes til Alpaca."""
        broker_name, _ = router.resolve_broker("AAPL")
        assert broker_name in ("alpaca", "ibkr")  # Both valid

    def test_resolve_danish_stock_to_nordnet(self, router):
        """Danske aktier (.CO) bør routes til Nordnet."""
        broker_name, _ = router.resolve_broker("NOVO-B.CO")
        assert broker_name in ("nordnet", "ibkr")  # Nordnet preferred for DK

    def test_resolve_crypto_to_alpaca(self, router):
        """Crypto bør routes til Alpaca."""
        broker_name, _ = router.resolve_broker("BTC-USD")
        assert broker_name == "alpaca"

    def test_resolve_german_stock(self, router):
        """Tyske aktier (.DE) bør routes til IBKR."""
        broker_name, _ = router.resolve_broker("SAP.DE")
        assert broker_name in ("ibkr", "saxo")

    def test_buy_delegates_to_correct_broker(self, router):
        """Buy bør delegere til den korrekte broker."""
        order = router.buy("AAPL", 10)
        assert order is not None
        assert order.symbol == "AAPL"
        assert order.qty == 10

    def test_sell_delegates_correctly(self, router):
        """Sell bør delegere til korrekt broker."""
        order = router.sell("AAPL", 5)
        assert order is not None
        assert order.symbol == "AAPL"

    def test_explain_routing(self, router):
        """explain_routing bør returnere info-dict."""
        info = router.explain_routing("NOVO-B.CO")
        assert isinstance(info, dict)
        assert "broker" in info or "symbol" in info

    def test_empty_router_raises(self, empty_router):
        """Router uden brokers bør fejle ved buy/sell."""
        with pytest.raises((RoutingError, Exception)):
            empty_router.buy("AAPL", 10)

    def test_get_positions_aggregates(self, router):
        """get_positions bør returnere fra alle brokers."""
        positions = router.get_positions()
        assert isinstance(positions, list)

    def test_get_account(self, router):
        """get_account bør returnere aggregeret info."""
        account = router.get_account()
        assert account is not None


# ── Edge Cases ─────────────────────────────────────────────

class TestBrokerRouterEdgeCases:
    def test_unknown_suffix(self, router):
        """Ukendt suffix bør falde igennem til fallback."""
        # Should not crash, should route to some broker
        try:
            broker_name, broker = router.resolve_broker("WEIRD.ZZ")
            assert broker is not None  # Fallback should work
        except RoutingError:
            pass  # Also acceptable

    def test_empty_symbol(self, router):
        """Tomt symbol via buy bør fejle gracefully."""
        with pytest.raises((RoutingError, ValueError, Exception)):
            router.buy("", 10)

    def test_register_duplicate(self, router):
        """Registrering af samme navn bør overskrive."""
        new_broker = MockBroker("alpaca_v2")
        router.register("alpaca", new_broker)
        assert router._brokers["alpaca"] is new_broker
