"""
Trading Fee Calculator — realistic per-broker, per-exchange fee computation.

Loads fee schedules from config/trading_fees.yaml and computes the correct
commission, spread, and transaction taxes for any symbol/broker combination.

Usage:
    from src.fees.fee_calculator import FeeCalculator
    calc = FeeCalculator(broker="saxo")
    fee = calc.calculate("NOVO-B.CO", side="buy", qty=100, price=850.0)
    # fee.commission = max(100*850*0.001, 14.0) = 85.0 DKK
    # fee.spread_cost = 100*850*0.0004 = 3.4
    # fee.total = 88.4
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

# US-listed ETF symbols (no suffix, so need explicit check)
_US_ETF_SYMBOLS = frozenset({
    "SPY", "QQQ", "IWM", "VTI", "VEA", "VWO", "EWJ", "EWH", "INDA",
    "EWG", "EWQ", "EWU", "GLD", "SLV", "USO", "TLT", "HYG",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP",
    "XLB", "XLRE", "XLU",
})


@dataclass
class TradingFee:
    """Computed fee breakdown for a single order."""
    commission: float = 0.0
    spread_cost: float = 0.0
    stamp_duty: float = 0.0
    transaction_tax: float = 0.0
    fx_spread_cost: float = 0.0
    exchange_fee: float = 0.0
    currency: str = "USD"

    @property
    def total(self) -> float:
        return (self.commission + self.spread_cost + self.stamp_duty
                + self.transaction_tax + self.fx_spread_cost + self.exchange_fee)


def _load_fee_config() -> dict:
    config_path = Path(__file__).resolve().parents[2] / "config" / "trading_fees.yaml"
    if not config_path.exists():
        logger.warning(f"Fee config not found at {config_path}, using zero fees")
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


import threading as _threading

_FEE_CONFIG: dict | None = None
_FEE_CONFIG_LOCK = _threading.Lock()


def _get_config() -> dict:
    global _FEE_CONFIG
    if _FEE_CONFIG is None:
        with _FEE_CONFIG_LOCK:
            if _FEE_CONFIG is None:  # Double-check after lock
                _FEE_CONFIG = _load_fee_config()
    return _FEE_CONFIG


def get_exchange_for_symbol(symbol: str) -> str:
    """Determine exchange category from symbol suffix."""
    symbol = symbol.upper()
    config = _get_config()
    mapping = config.get("exchange_mapping", {})

    # Check crypto first (-USD suffix)
    if symbol.endswith("-USD"):
        return mapping.get("-USD", "crypto")

    # Check futures (=F suffix)
    if symbol.endswith("=F"):
        return mapping.get("=F", "futures")

    # Check VIX-style symbols
    if symbol.startswith("^"):
        return mapping.get("^", "us_stocks")

    # US-listed ETFs (no suffix)
    if symbol in _US_ETF_SYMBOLS:
        return "us_etfs"

    # Match by suffix
    for suffix, exchange in mapping.items():
        if suffix and suffix not in ("-USD", "=F", "^") and symbol.endswith(suffix):
            return exchange

    # Default: US stocks (no suffix)
    return "us_stocks"


class FeeCalculator:
    """
    Computes trading fees for a given broker using realistic fee schedules.

    Args:
        broker: Broker name ("alpaca", "ibkr", "saxo", "nordnet", "paper").
                Defaults to "paper".
    """

    def __init__(self, broker: str = "paper") -> None:
        self.broker = broker.lower()
        self._config = _get_config()
        self._broker_fees = self._config.get(self.broker, {})
        if not self._broker_fees:
            logger.warning(f"No fee schedule for broker '{broker}', using zero fees")

    def get_fee_schedule(self, symbol: str) -> dict:
        """Get the fee schedule dict for a specific symbol."""
        exchange = get_exchange_for_symbol(symbol)

        # Try exact exchange match first
        if exchange in self._broker_fees:
            return self._broker_fees[exchange]

        # Fallback: if broker has 'default', use that
        if "default" in self._broker_fees:
            return self._broker_fees["default"]

        # Fallback: try us_stocks as generic
        if "us_stocks" in self._broker_fees:
            return self._broker_fees["us_stocks"]

        return {}

    def calculate(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> TradingFee:
        """
        Calculate all fees for an order.

        Args:
            symbol: Trading symbol (e.g. "NOVO-B.CO", "AAPL", "BTC-USD").
            side: "buy" or "sell".
            qty: Number of shares/units.
            price: Price per share/unit.

        Returns:
            TradingFee with full breakdown.
        """
        schedule = self.get_fee_schedule(symbol)
        if not schedule:
            return TradingFee()

        trade_value = qty * price
        fee = TradingFee(currency=schedule.get("currency", "USD"))

        # Commission: per-share or percentage
        per_share = schedule.get("per_share", 0.0)
        if per_share > 0:
            fee.commission = qty * per_share
        else:
            fee.commission = trade_value * schedule.get("commission_pct", 0.0)

        # Apply minimum commission
        min_comm = schedule.get("commission_min", 0.0)
        if fee.commission > 0 or min_comm > 0:
            fee.commission = max(fee.commission, min_comm)

        # Apply maximum commission (as pct of trade value)
        max_comm = schedule.get("commission_max", 0.0)
        if max_comm > 0:
            fee.commission = min(fee.commission, trade_value * max_comm)

        # Spread cost
        spread_pct = schedule.get("spread_pct", 0.0)
        fee.spread_cost = trade_value * spread_pct

        # Stamp duty (UK, HK, Swiss — typically only on buys)
        stamp_pct = schedule.get("stamp_duty_pct", 0.0)
        if stamp_pct > 0 and side.lower() == "buy":
            fee.stamp_duty = trade_value * stamp_pct

        # Securities Transaction Tax (India STT — on sells)
        stt_pct = schedule.get("stt_pct", 0.0)
        if stt_pct > 0 and side.lower() == "sell":
            fee.transaction_tax = trade_value * stt_pct

        # FX spread (e.g. Nordnet DKK→USD conversion)
        fx_spread = schedule.get("fx_spread_pct", 0.0)
        if fx_spread > 0:
            fee.fx_spread_cost = trade_value * fx_spread

        # Per-contract exchange fees (futures)
        per_contract = schedule.get("per_contract", 0.0)
        exch_fee = schedule.get("exchange_fee", 0.0)
        if per_contract > 0 or exch_fee > 0:
            fee.exchange_fee = qty * (per_contract + exch_fee)

        return fee

    def get_spread_pct(self, symbol: str) -> float:
        """Get just the spread percentage for a symbol."""
        schedule = self.get_fee_schedule(symbol)
        return schedule.get("spread_pct", 0.0005)

    def get_commission_pct(self, symbol: str) -> float:
        """Get effective commission percentage for a symbol (approximate)."""
        schedule = self.get_fee_schedule(symbol)
        return schedule.get("commission_pct", 0.0)

    def summary(self) -> dict[str, dict]:
        """Return the full fee schedule for this broker."""
        return dict(self._broker_fees)
