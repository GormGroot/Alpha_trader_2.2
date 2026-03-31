"""
Centralt konfigurationssystem for Alpha Trading Platform.

Prioritetsrækkefølge (højeste først):
  1. Miljøvariabler / .env
  2. default_config.yaml
  3. Hardkodede defaults i denne fil
"""

from __future__ import annotations

from pathlib import Path
from functools import lru_cache

import yaml
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Paths ────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default_config.yaml"


def _load_yaml_defaults() -> dict:
    """Læs default_config.yaml og returnér som flat dict."""
    if not DEFAULT_CONFIG_PATH.exists():
        return {}
    with open(DEFAULT_CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


_yaml = _load_yaml_defaults()


# ── Sub-models ───────────────────────────────────────────────
class BrokerSettings(BaseSettings):
    provider: str = _yaml.get("broker", {}).get("provider", "alpaca")
    api_key: str = ""
    secret_key: str = ""
    base_url: str = _yaml.get("broker", {}).get("base_url", "https://paper-api.alpaca.markets")
    api_version: str = _yaml.get("broker", {}).get("api_version", "v2")

    model_config = SettingsConfigDict(
        env_prefix="ALPACA_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )


class MarketDataSettings(BaseSettings):
    provider: str = _yaml.get("market_data", {}).get("provider", "yfinance")
    alpha_vantage_key: str = ""
    fred_api_key: str = ""
    interval: str = _yaml.get("market_data", {}).get("interval", "1d")
    lookback_days: int = _yaml.get("market_data", {}).get("lookback_days", 365)
    cache_enabled: bool = _yaml.get("market_data", {}).get("cache_enabled", True)
    cache_dir: str = _yaml.get("market_data", {}).get("cache_dir", "data_cache")

    model_config = SettingsConfigDict(
        env_prefix="MARKET_DATA_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )


_trading_yaml = _yaml.get("trading", {})


class TradingSettings(BaseSettings):
    symbols: list[str] = Field(
        default=_trading_yaml.get("symbols", ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"])
    )
    market_open: str = _trading_yaml.get("schedule", {}).get("market_open", "09:30")
    market_close: str = _trading_yaml.get("schedule", {}).get("market_close", "16:00")
    timezone: str = _trading_yaml.get("schedule", {}).get("timezone", "US/Eastern")
    check_interval_seconds: int = _trading_yaml.get("schedule", {}).get("check_interval_seconds", 60)

    model_config = SettingsConfigDict(
        env_prefix="TRADING_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )


_strategy_yaml = _yaml.get("strategy", {})


class StrategySettings(BaseSettings):
    default: str = _strategy_yaml.get("default", "sma_crossover")
    sma_short_window: int = _strategy_yaml.get("sma_crossover", {}).get("short_window", 20)
    sma_long_window: int = _strategy_yaml.get("sma_crossover", {}).get("long_window", 50)
    rsi_period: int = _strategy_yaml.get("rsi", {}).get("period", 14)
    rsi_overbought: int = _strategy_yaml.get("rsi", {}).get("overbought", 70)
    rsi_oversold: int = _strategy_yaml.get("rsi", {}).get("oversold", 30)
    bb_window: int = _strategy_yaml.get("bollinger", {}).get("window", 20)
    bb_num_std: float = _strategy_yaml.get("bollinger", {}).get("num_std", 2.0)

    model_config = SettingsConfigDict(
        env_prefix="STRATEGY_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )


_risk_yaml = _yaml.get("risk", {})


class RiskSettings(BaseSettings):
    max_position_pct: float = _risk_yaml.get("max_position_pct", 0.02)
    max_daily_loss_pct: float = _risk_yaml.get("max_daily_loss_pct", 0.05)
    max_open_positions: int = _risk_yaml.get("max_open_positions", 10)
    stop_loss_pct: float = _risk_yaml.get("stop_loss_pct", 0.02)
    take_profit_pct: float = _risk_yaml.get("take_profit_pct", 0.05)
    trailing_stop_pct: float = _risk_yaml.get("trailing_stop_pct", 0.03)
    max_drawdown_pct: float = _risk_yaml.get("max_drawdown_pct", 0.10)

    @model_validator(mode="after")
    def validate_risk_ranges(self) -> "RiskSettings":
        """Ensure risk percentages are decimal fractions in sane ranges."""
        if not (0.001 <= self.max_position_pct <= 0.25):
            raise ValueError(
                f"max_position_pct={self.max_position_pct} is out of range — "
                f"must be between 0.001 (0.1%) and 0.25 (25%). "
                f"Use decimal notation, e.g. 0.15 for 15%."
            )
        if not (0.005 <= self.stop_loss_pct <= 0.10):
            raise ValueError(
                f"stop_loss_pct={self.stop_loss_pct} is out of range — "
                f"must be between 0.005 (0.5%) and 0.10 (10%). "
                f"Use decimal notation, e.g. 0.015 for 1.5%."
            )
        return self

    model_config = SettingsConfigDict(
        env_prefix="RISK_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )


_backtest_yaml = _yaml.get("backtest", {})


class BacktestSettings(BaseSettings):
    start_date: str = _backtest_yaml.get("start_date", "2024-01-01")
    end_date: str = _backtest_yaml.get("end_date", "2025-12-31")
    initial_capital: float = _backtest_yaml.get("initial_capital", 100000)
    commission_pct: float = _backtest_yaml.get("commission_pct", 0.0)

    model_config = SettingsConfigDict(
        env_prefix="BACKTEST_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )


_dashboard_yaml = _yaml.get("dashboard", {})


class DashboardSettings(BaseSettings):
    host: str = _dashboard_yaml.get("host", "127.0.0.1")
    port: int = _dashboard_yaml.get("port", 8050)
    debug: bool = _dashboard_yaml.get("debug", False)
    refresh_interval_seconds: int = _dashboard_yaml.get("refresh_interval_seconds", 30)

    model_config = SettingsConfigDict(
        env_prefix="DASHBOARD_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )


_universe_yaml = _yaml.get("universe", {})


class UniverseSettings(BaseSettings):
    categories: dict = Field(
        default=_universe_yaml.get("categories", {
            "us_stocks": True,
            "nordic_stocks": True,
            "etfs": True,
        })
    )
    regions: list[str] = Field(default=_universe_yaml.get("regions", []))
    min_market_cap: float = _universe_yaml.get("min_market_cap", 0)
    max_symbols_per_scan: int = _universe_yaml.get("max_symbols_per_scan", 200)
    parallel_workers: int = _universe_yaml.get("parallel_workers", 8)
    batch_size: int = _universe_yaml.get("batch_size", 50)
    watchlist: list[str] = Field(default=_universe_yaml.get("watchlist", []))

    model_config = SettingsConfigDict(
        env_prefix="UNIVERSE_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )


_tax_yaml = _yaml.get("tax", {})


class TaxSettings(BaseSettings):
    progression_limit: float = _tax_yaml.get("progression_limit", 61_000)
    low_rate: float = _tax_yaml.get("low_rate", 0.27)
    high_rate: float = _tax_yaml.get("high_rate", 0.42)
    carried_losses: float = _tax_yaml.get("carried_losses", 0.0)
    fallback_fx_rate: float = _tax_yaml.get("fallback_fx_rate", 6.90)
    currency: str = _tax_yaml.get("currency", "DKK")

    model_config = SettingsConfigDict(
        env_prefix="TAX_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )


_notify_yaml = _yaml.get("notifications", {})


class NotificationSettings(BaseSettings):
    enabled: bool = _notify_yaml.get("enabled", True)
    smtp_host: str = _notify_yaml.get("smtp_host", "smtp.gmail.com")
    smtp_port: int = _notify_yaml.get("smtp_port", 587)
    smtp_user: str = ""
    smtp_password: str = ""
    from_email: str = ""
    to_email: str = ""
    # Skat-alerts
    monthly_report: bool = _notify_yaml.get("monthly_report", True)
    progression_alert: bool = _notify_yaml.get("progression_alert", True)
    march_reminder: bool = _notify_yaml.get("march_reminder", True)
    # Handels-alerts
    on_trade_executed: bool = _notify_yaml.get("on_trade_executed", True)
    on_stop_loss: bool = _notify_yaml.get("on_stop_loss", True)
    on_trailing_stop: bool = _notify_yaml.get("on_trailing_stop", True)
    on_take_profit: bool = _notify_yaml.get("on_take_profit", True)
    on_daily_report: bool = _notify_yaml.get("on_daily_report", True)
    on_drawdown_warning: bool = _notify_yaml.get("on_drawdown_warning", True)
    drawdown_threshold_pct: float = _notify_yaml.get("drawdown_threshold_pct", 0.05)

    model_config = SettingsConfigDict(
        env_prefix="NOTIFY_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )


_logging_yaml = _yaml.get("logging", {})


class LoggingSettings(BaseSettings):
    level: str = _logging_yaml.get("level", "DEBUG")
    file: str = _logging_yaml.get("file", "logs/trading.log")
    rotation: str = _logging_yaml.get("rotation", "1 day")
    retention: str = _logging_yaml.get("retention", "30 days")

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=ROOT_DIR / ".env",
        extra="ignore",
    )


# ── Samlet Settings ──────────────────────────────────────────
class Settings:
    """Samler alle sub-settings i ét objekt."""

    def __init__(self) -> None:
        self.broker = BrokerSettings()
        self.market_data = MarketDataSettings()
        self.trading = TradingSettings()
        self.strategy = StrategySettings()
        self.risk = RiskSettings()
        self.backtest = BacktestSettings()
        self.dashboard = DashboardSettings()
        self.universe = UniverseSettings()
        self.tax = TaxSettings()
        self.notifications = NotificationSettings()
        self.logging = LoggingSettings()

    def print_summary(self) -> None:
        """Print en overskuelig oversigt (uden secrets)."""
        masked_key = self.broker.api_key[:4] + "***" if self.broker.api_key else "NOT SET"
        print(f"""
╔══════════════════════════════════════════╗
║   Alpha Trading Platform – Config        ║
╠══════════════════════════════════════════╣
║ Broker:     {self.broker.provider:<28}║
║ API Key:    {masked_key:<28}║
║ Base URL:   {self.broker.base_url:<28}║
╠══════════════════════════════════════════╣
║ Data:       {self.market_data.provider:<28}║
║ Interval:   {self.market_data.interval:<28}║
║ Lookback:   {self.market_data.lookback_days} days{'':<22}║
╠══════════════════════════════════════════╣
║ Symbols:    {', '.join(self.trading.symbols):<28}║
║ Strategy:   {self.strategy.default:<28}║
║ SMA:        {self.strategy.sma_short_window}/{self.strategy.sma_long_window:<26}║
╠══════════════════════════════════════════╣
║ Max pos:    {self.risk.max_position_pct:.0%}{'':<25}║
║ Max loss:   {self.risk.max_daily_loss_pct:.0%}{'':<25}║
║ Stop loss:  {self.risk.stop_loss_pct:.0%}{'':<25}║
╠══════════════════════════════════════════╣
║ Log level:  {self.logging.level:<28}║
╚══════════════════════════════════════════╝""")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returnér cached settings-instans."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the settings cache and return a fresh Settings instance.

    Use this when config files or environment variables have changed
    and you need the updated values.
    """
    get_settings.cache_clear()
    fresh = get_settings()
    # Update the module-level convenience reference
    global settings
    settings = fresh
    return fresh


# Convenience – importér direkte: from config.settings import settings
settings = get_settings()
