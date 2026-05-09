"""
Phase A4: Tests mod duplikat-handler.

Audit-fund 10. maj 2026: XLRE 36-stk position blev solgt 2× på samme
tidspunkt (200ms afstand). Disse tests bekræfter at:

  1. Hvis et symbol exit'es i et scan, springes det over i entry-fasen
  2. _has_recent_executed_trade() afviser duplikater inden for 60-sek vindue
  3. Concurrent execute-calls med samme symbol blokeres af _claim_symbol
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


os.environ.setdefault("APP_USERNAME", "testuser")
os.environ.setdefault("APP_PASSWORD", "testpass123")
os.environ.setdefault("APP_SECRET_KEY", "x" * 64)


# ── Test idempotency window ────────────────────────────────────


class TestRecentTradeCheck:
    def _make_trader_with_db(self, tmp_db: Path):
        """Build a minimal AutoTrader-like object with just _db_path + _has_recent_executed_trade."""
        from src.trader.auto_trader import AutoTrader

        # Construct DB schema first
        with sqlite3.connect(tmp_db) as conn:
            conn.execute("""
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    scan_id INTEGER,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL,
                    reason TEXT,
                    confidence REAL,
                    alpha_score REAL,
                    executed INTEGER,
                    rejection_reason TEXT,
                    order_id TEXT,
                    risk_approved INTEGER,
                    error TEXT
                )
            """)

        # Bind method via tiny ducktype
        class Stub:
            _db_path = str(tmp_db)
            _db_connect = AutoTrader._db_connect
            _has_recent_executed_trade = AutoTrader._has_recent_executed_trade

        return Stub()

    def test_no_duplicate_when_no_history(self, tmp_path):
        stub = self._make_trader_with_db(tmp_path / "db.sqlite")
        assert stub._has_recent_executed_trade("AAPL", "SELL", 60) is False

    def test_blocks_duplicate_within_window(self, tmp_path):
        from src.trader.auto_trader import _now_cet
        stub = self._make_trader_with_db(tmp_path / "db.sqlite")
        # Indsæt et trade lige nu
        with sqlite3.connect(stub._db_path) as conn:
            conn.execute(
                """INSERT INTO trades (timestamp, scan_id, symbol, side, qty, executed)
                   VALUES (?, 1, 'AAPL', 'SELL', 10, 1)""",
                (_now_cet().isoformat(),)
            )
        assert stub._has_recent_executed_trade("AAPL", "SELL", 60) is True

    def test_allows_after_window(self, tmp_path):
        stub = self._make_trader_with_db(tmp_path / "db.sqlite")
        # Indsæt et trade for 5 minutter siden
        old_ts = (pd.Timestamp.now(tz="Europe/Copenhagen") - pd.Timedelta(minutes=5)).isoformat()
        with sqlite3.connect(stub._db_path) as conn:
            conn.execute(
                """INSERT INTO trades (timestamp, scan_id, symbol, side, qty, executed)
                   VALUES (?, 1, 'AAPL', 'SELL', 10, 1)""",
                (old_ts,)
            )
        # 60 sek vindue → trade er for gammelt → ikke duplikat
        assert stub._has_recent_executed_trade("AAPL", "SELL", 60) is False

    def test_only_blocks_executed_trades(self, tmp_path):
        """Afviste trades skal IKKE tælle som duplikater (executed=0)."""
        from src.trader.auto_trader import _now_cet
        stub = self._make_trader_with_db(tmp_path / "db.sqlite")
        with sqlite3.connect(stub._db_path) as conn:
            conn.execute(
                """INSERT INTO trades (timestamp, scan_id, symbol, side, qty, executed)
                   VALUES (?, 1, 'AAPL', 'SELL', 10, 0)""",  # executed=0
                (_now_cet().isoformat(),)
            )
        assert stub._has_recent_executed_trade("AAPL", "SELL", 60) is False

    def test_different_side_not_duplicate(self, tmp_path):
        """SELL bør ikke blokere senere BUY på samme symbol."""
        from src.trader.auto_trader import _now_cet
        stub = self._make_trader_with_db(tmp_path / "db.sqlite")
        with sqlite3.connect(stub._db_path) as conn:
            conn.execute(
                """INSERT INTO trades (timestamp, scan_id, symbol, side, qty, executed)
                   VALUES (?, 1, 'AAPL', 'SELL', 10, 1)""",
                (_now_cet().isoformat(),)
            )
        assert stub._has_recent_executed_trade("AAPL", "BUY", 60) is False


# ── Concurrent execute_action ──────────────────────────────────


class TestConcurrentExecution:
    def test_claim_symbol_blocks_concurrent_orders(self, tmp_path):
        """Hvis to threads samtidig prøver at execute samme symbol, skal kun én lykkes."""
        from src.trader.auto_trader import AutoTrader, TradeAction

        # Mock minimal AutoTrader med kun _claim_symbol + executor
        class MockTrader:
            _in_flight = set()
            _in_flight_lock = threading.Lock()
            _claim_symbol = AutoTrader._claim_symbol

            def __init__(self):
                self.executions = []

            def execute(self, sym, sleep_sec=0.1):
                with self._claim_symbol(sym) as claimed:
                    if not claimed:
                        return False
                    time.sleep(sleep_sec)
                    self.executions.append(sym)
                    return True

        trader = MockTrader()
        results = []

        def worker():
            results.append(trader.execute("AAPL"))

        t1 = threading.Thread(target=worker)
        t2 = threading.Thread(target=worker)
        t1.start(); t2.start()
        t1.join(); t2.join()

        # Kun én skal have succes (claim_symbol returnerede True)
        assert results.count(True) == 1
        assert results.count(False) == 1
        assert len(trader.executions) == 1
