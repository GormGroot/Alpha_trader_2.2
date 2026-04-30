"""
Regression tests for Fase 1.1 database migration (2026-04-17).

Verifies that the three hot SQLite databases (signals, learning,
auto_trader_log) are opened with:

  * journal_mode = WAL           — concurrent reader/writer without locking
  * synchronous  = NORMAL        — safe with WAL, ~10x faster commits
  * busy_timeout ≥ 5000 ms       — absorbs scheduler/dashboard contention

and that the indexes the dashboard/prune path rely on actually exist.

These tests spin up fresh tmp DBs per class via the production
constructors. They do NOT touch the user's real data_cache/*.db files.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.strategy.signal_engine import SignalStore, SymbolSignal
from src.strategy.base_strategy import Signal
from src.learning.continuous_learner import ContinuousLearner


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
def _pragma(conn: sqlite3.Connection, name: str):
    cur = conn.execute(f"PRAGMA {name}")
    row = cur.fetchone()
    return row[0] if row else None


def _index_names(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA index_list({table})")
    return {row[1] for row in cur.fetchall()}


# ──────────────────────────────────────────────────────────
# SignalStore (signals.db)
# ──────────────────────────────────────────────────────────
class TestSignalStoreMigration:
    def _store(self, tmp_path: Path) -> SignalStore:
        return SignalStore(tmp_path / "signals.db")

    def test_wal_mode_enabled(self, tmp_path):
        store = self._store(tmp_path)
        with store._get_conn() as conn:
            assert _pragma(conn, "journal_mode").lower() == "wal"

    def test_synchronous_normal(self, tmp_path):
        store = self._store(tmp_path)
        with store._get_conn() as conn:
            # synchronous: 0=OFF, 1=NORMAL, 2=FULL
            assert int(_pragma(conn, "synchronous")) == 1

    def test_busy_timeout_set(self, tmp_path):
        store = self._store(tmp_path)
        with store._get_conn() as conn:
            assert int(_pragma(conn, "busy_timeout")) >= 5000

    def test_required_indexes_exist(self, tmp_path):
        store = self._store(tmp_path)
        with store._get_conn() as conn:
            idx = _index_names(conn, "signal_history")
        # Must have both symbol-ts (for per-symbol lookups) and ts-only
        # (for time-range dashboard queries and prune DELETE).
        assert "idx_signal_symbol_ts" in idx
        assert "idx_signal_ts" in idx

    def test_prune_default_is_90_days(self, tmp_path):
        """Default retention aligned with Fase 1 plan (was 14d, now 90d)."""
        import inspect

        sig = inspect.signature(SignalStore.prune)
        assert sig.parameters["keep_days"].default == 90

    def test_prune_respects_cutoff(self, tmp_path):
        """Rows older than cutoff get deleted; fresher rows survive."""
        import pandas as pd

        store = self._store(tmp_path)
        old_ts = (pd.Timestamp.now() - pd.Timedelta(days=200)).isoformat()
        new_ts = pd.Timestamp.now().isoformat()

        for ts in (old_ts, new_ts):
            store.save(
                SymbolSignal(
                    symbol="AAPL",
                    signal=Signal.BUY,
                    confidence=50.0,
                    position_size_usd=1000.0,
                    reason="test",
                    timestamp=ts,
                )
            )
        assert store.count() == 2

        store.prune(keep_days=90)
        remaining = store.count()
        assert remaining == 1, f"expected 1 row, got {remaining}"

    def test_vacuum_does_not_raise(self, tmp_path):
        """Monthly VACUUM hook should be callable without side-effects."""
        store = self._store(tmp_path)
        store.vacuum()  # must not raise


# ──────────────────────────────────────────────────────────
# ContinuousLearner (learning.db)
# ──────────────────────────────────────────────────────────
class TestContinuousLearnerMigration:
    def _learner(self, tmp_path: Path) -> ContinuousLearner:
        return ContinuousLearner(
            db_path=str(tmp_path / "learning.db"),
            trade_db_path=str(tmp_path / "trader.db"),
        )

    def test_wal_mode_enabled(self, tmp_path):
        learner = self._learner(tmp_path)
        with learner._connect() as conn:
            assert _pragma(conn, "journal_mode").lower() == "wal"

    def test_synchronous_normal(self, tmp_path):
        learner = self._learner(tmp_path)
        with learner._connect() as conn:
            assert int(_pragma(conn, "synchronous")) == 1

    def test_busy_timeout_set(self, tmp_path):
        learner = self._learner(tmp_path)
        with learner._connect() as conn:
            assert int(_pragma(conn, "busy_timeout")) >= 5000

    def test_timestamp_indexes_exist(self, tmp_path):
        """Prune DELETEs scan by timestamp. Without these indexes the
        prune path does a full table scan on every 5-table prune."""
        learner = self._learner(tmp_path)
        with learner._connect() as conn:
            for table, expected in (
                ("trade_outcomes", "idx_outcomes_ts"),
                ("model_scores", "idx_scores_ts"),
                ("drift_events", "idx_drift_ts"),
                ("learning_log", "idx_log_ts"),
            ):
                assert expected in _index_names(
                    conn, table
                ), f"{expected} missing on {table}"

    def test_prune_deletes_old_rows(self, tmp_path):
        """_prune_learning_db removes outcomes older than 90 days."""
        learner = self._learner(tmp_path)
        with learner._connect() as conn:
            # One fresh, one ancient
            conn.execute(
                "INSERT INTO trade_outcomes (timestamp, symbol, side) "
                "VALUES (datetime('now', '-200 days'), 'AAPL', 'BUY')"
            )
            conn.execute(
                "INSERT INTO trade_outcomes (timestamp, symbol, side) "
                "VALUES (datetime('now', '-1 days'), 'AAPL', 'BUY')"
            )
        learner._prune_learning_db()
        with learner._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM trade_outcomes").fetchone()[0]
        assert count == 1


# ──────────────────────────────────────────────────────────
# AutoTrader log DB (auto_trader_log.db)
# ──────────────────────────────────────────────────────────
class TestAutoTraderLogMigration:
    def test_wal_mode_enabled(self, tmp_path):
        """AutoTrader._db_connect opens auto_trader_log.db in WAL mode."""
        from src.trader.auto_trader import AutoTrader

        t = AutoTrader.__new__(AutoTrader)
        t._db_path = tmp_path / "auto_trader_log.db"
        t._init_db()
        with t._db_connect() as conn:
            assert _pragma(conn, "journal_mode").lower() == "wal"
            assert int(_pragma(conn, "synchronous")) == 1
            assert int(_pragma(conn, "busy_timeout")) >= 5000

    def test_indexes_created(self, tmp_path):
        from src.trader.auto_trader import AutoTrader

        t = AutoTrader.__new__(AutoTrader)
        t._db_path = tmp_path / "auto_trader_log.db"
        t._init_db()
        with t._db_connect() as conn:
            scan_idx = _index_names(conn, "scans")
            trade_idx = _index_names(conn, "trades")
        assert "idx_scans_ts" in scan_idx
        assert "idx_trades_ts" in trade_idx
        assert "idx_trades_symbol_ts" in trade_idx


# ──────────────────────────────────────────────────────────
# DailyScheduler db_maintenance hook (Fase 1.1)
# ──────────────────────────────────────────────────────────
class TestDbMaintenanceHook:
    """The Saturday 03:00 CET prune hook must be registered, no-op on
    non-Saturdays, and actually prune on Saturdays."""

    def test_hook_registered_on_default_tasks(self):
        from src.ops.daily_scheduler import DailyScheduler

        tasks = {t.name: t for t in DailyScheduler.DEFAULT_TASKS}
        assert "db_maintenance" in tasks, "db_maintenance task not registered"
        t = tasks["db_maintenance"]
        assert (t.hour, t.minute) == (3, 0), "must run at 03:00 CET"
        assert t.requires_market_day is False, "must run on weekends"

    def test_skips_on_non_saturday(self, monkeypatch):
        """Fires daily; no-op unless weekday()==5."""
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from src.ops import daily_scheduler as ds

        # Monday
        monkeypatch.setattr(
            ds,
            "_now_cet",
            lambda: datetime(2026, 4, 20, 3, 0, tzinfo=ZoneInfo("Europe/Copenhagen")),
        )
        result = ds._db_maintenance()
        assert result.get("skipped") is True
        assert result.get("reason") == "not_saturday"
