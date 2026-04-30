"""
Regression tests for Fase 1.5 — scheduler heartbeat (2026-04-17).

Verifies that:
  * DailyScheduler._run_loop writes the heartbeat file on every tick
  * is_scheduler_alive() returns True within the stale window
  * is_scheduler_alive() returns False when the file is missing or stale
  * The write is atomic (no torn reads from a watchdog racing the writer)
  * Env-var ALPHA_TRADER_HEARTBEAT overrides the default path

Without these tests an outage like "Flask is up but scheduler thread has
wedged" goes undetected for hours — which is exactly the failure mode
the heartbeat was added to catch.
"""

from __future__ import annotations

import importlib
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.ops import daily_scheduler as ds


# ──────────────────────────────────────────────────────────
# Module-level helpers
# ──────────────────────────────────────────────────────────
class TestHeartbeatFileHelpers:
    def test_write_heartbeat_creates_file(self, tmp_path, monkeypatch):
        hb = tmp_path / "heartbeat"
        monkeypatch.setattr(ds, "HEARTBEAT_FILE", hb)
        ds._write_heartbeat()
        assert hb.exists()
        # Content is a float epoch timestamp
        ts = float(hb.read_text().strip())
        assert ts > 0
        # Fresh — within last few seconds
        assert time.time() - ts < 5

    def test_is_scheduler_alive_true_when_fresh(self, tmp_path, monkeypatch):
        hb = tmp_path / "heartbeat"
        monkeypatch.setattr(ds, "HEARTBEAT_FILE", hb)
        ds._write_heartbeat()
        assert ds.is_scheduler_alive() is True

    def test_is_scheduler_alive_false_when_missing(self, tmp_path, monkeypatch):
        hb = tmp_path / "never_written"
        monkeypatch.setattr(ds, "HEARTBEAT_FILE", hb)
        assert ds.is_scheduler_alive() is False

    def test_is_scheduler_alive_false_when_stale(self, tmp_path, monkeypatch):
        hb = tmp_path / "heartbeat"
        monkeypatch.setattr(ds, "HEARTBEAT_FILE", hb)
        # Write a timestamp 10 minutes in the past
        hb.write_text(f"{time.time() - 600:.3f}\n")
        assert ds.is_scheduler_alive(max_age_seconds=180) is False

    def test_is_scheduler_alive_false_on_corrupted_file(self, tmp_path, monkeypatch):
        hb = tmp_path / "heartbeat"
        monkeypatch.setattr(ds, "HEARTBEAT_FILE", hb)
        hb.write_text("not-a-float\n")
        assert ds.is_scheduler_alive() is False

    def test_is_scheduler_alive_accepts_explicit_path(self, tmp_path):
        """Callers can point at a custom path without mutating module state."""
        hb = tmp_path / "alt_heartbeat"
        hb.write_text(f"{time.time():.3f}\n")
        assert ds.is_scheduler_alive(path=hb) is True
        assert ds.is_scheduler_alive(path=tmp_path / "missing") is False

    def test_write_heartbeat_is_atomic(self, tmp_path, monkeypatch):
        """The temp-file + replace dance must leave NO torn state visible
        to a watchdog reading in parallel."""
        hb = tmp_path / "heartbeat"
        monkeypatch.setattr(ds, "HEARTBEAT_FILE", hb)

        # Seed a valid initial value
        hb.write_text(f"{time.time():.3f}\n")

        stop = threading.Event()
        corrupted = []

        def watchdog() -> None:
            while not stop.is_set():
                try:
                    raw = hb.read_text().strip()
                    if raw:
                        float(raw)  # must parse
                except ValueError:
                    corrupted.append(True)
                    return
                except FileNotFoundError:
                    # Acceptable momentarily during rename on some OSes,
                    # but we still don't want corrupted content.
                    pass

        w = threading.Thread(target=watchdog, daemon=True)
        w.start()
        try:
            for _ in range(200):
                ds._write_heartbeat()
        finally:
            stop.set()
            w.join(timeout=1)

        assert not corrupted, "watchdog saw a torn/corrupted heartbeat"

    def test_env_var_override(self, tmp_path, monkeypatch):
        """ALPHA_TRADER_HEARTBEAT must control the default path at import
        time — required for Docker/systemd unit files to redirect it."""
        custom = tmp_path / "custom_heartbeat"
        monkeypatch.setenv("ALPHA_TRADER_HEARTBEAT", str(custom))
        # Reimport to re-evaluate the module-level Path()
        reloaded = importlib.reload(ds)
        try:
            assert reloaded.HEARTBEAT_FILE == custom
        finally:
            # Restore the canonical module for other tests in this run.
            monkeypatch.delenv("ALPHA_TRADER_HEARTBEAT", raising=False)
            importlib.reload(ds)


# ──────────────────────────────────────────────────────────
# _run_loop integration
# ──────────────────────────────────────────────────────────
class TestRunLoopHeartbeat:
    def test_run_loop_writes_heartbeat_on_tick(self, tmp_path, monkeypatch):
        """Start the scheduler, let the loop tick once, stop it, and verify
        the heartbeat file was written with a fresh timestamp."""
        hb = tmp_path / "heartbeat"
        monkeypatch.setattr(ds, "HEARTBEAT_FILE", hb)

        scheduler = ds.DailyScheduler(tasks=[])  # empty task list — no side effects

        # Run the loop in a thread, let it prime + hit the top of the while,
        # then stop it. The prime write happens unconditionally; we don't
        # need to wait 30s for the first real tick.
        t = threading.Thread(target=scheduler._run_loop, daemon=True)
        t.start()
        # Allow the prime write + first loop-entry heartbeat to land.
        for _ in range(50):
            if hb.exists():
                break
            time.sleep(0.01)
        scheduler._stop_event.set()
        t.join(timeout=5)

        assert hb.exists(), "heartbeat file not created by _run_loop"
        ts = float(hb.read_text().strip())
        assert time.time() - ts < 5, "heartbeat timestamp not fresh"

    def test_run_loop_heartbeat_uses_now_cet(self, tmp_path, monkeypatch):
        """The timestamp written is derived from _now_cet().timestamp(),
        which gives a unix-epoch that any watchdog can diff against
        time.time(). Regression: a previous draft used tz-aware datetime
        .isoformat() which would have broken Docker HEALTHCHECK."""
        hb = tmp_path / "heartbeat"
        monkeypatch.setattr(ds, "HEARTBEAT_FILE", hb)

        fixed = datetime(2026, 4, 17, 12, 0, 0, tzinfo=ZoneInfo("Europe/Copenhagen"))
        monkeypatch.setattr(ds, "_now_cet", lambda: fixed)
        ds._write_heartbeat()
        ts = float(hb.read_text().strip())
        assert ts == pytest.approx(fixed.timestamp(), abs=0.001)
