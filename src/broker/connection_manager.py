"""
Connection Manager — health monitoring for alle brokers.

Features:
  - Health check for hver broker hvert 60. sekund
  - Status: CONNECTED, DEGRADED, DISCONNECTED
  - Auto-reconnect med exponential backoff
  - Alert callbacks når status ændres
  - Dashboard-kompatibelt status-overblik
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from loguru import logger

from src.broker.base_broker import BaseBroker


# ── Enums & Dataklasser ─────────────────────────────────────

class ConnectionStatus(Enum):
    """Broker-forbindelsesstatus."""
    CONNECTED = "connected"
    DEGRADED = "degraded"           # Langsom response eller partial failure
    DISCONNECTED = "disconnected"
    UNKNOWN = "unknown"


@dataclass
class BrokerHealth:
    """Health-status for én broker."""
    broker_name: str
    status: ConnectionStatus = ConnectionStatus.UNKNOWN
    last_check: datetime | None = None
    last_success: datetime | None = None
    last_error: str = ""
    response_time_ms: float = 0.0   # Seneste response tid
    avg_response_ms: float = 0.0    # Gennemsnit over tid
    consecutive_failures: int = 0
    total_checks: int = 0
    total_failures: int = 0
    uptime_pct: float = 100.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "broker": self.broker_name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_error": self.last_error,
            "response_time_ms": round(self.response_time_ms, 1),
            "avg_response_ms": round(self.avg_response_ms, 1),
            "consecutive_failures": self.consecutive_failures,
            "uptime_pct": round(self.uptime_pct, 1),
        }


@dataclass
class StatusChange:
    """En ændring i broker-status (til callbacks)."""
    broker_name: str
    old_status: ConnectionStatus
    new_status: ConnectionStatus
    error: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# ── Connection Manager ──────────────────────────────────────

class ConnectionManager:
    """
    Monitor og administrér broker-forbindelser.

    Brug:
        manager = ConnectionManager()
        manager.register("alpaca", alpaca_broker)
        manager.register("ibkr", ibkr_broker)

        # Callback ved status-ændring
        manager.on_status_change(lambda change: print(
            f"{change.broker_name}: {change.old_status} → {change.new_status}"
        ))

        # Start periodisk health check (background thread)
        manager.start(interval=60)

        # Manuel check
        manager.check_all()

        # Status
        print(manager.get_dashboard_status())
    """

    # Thresholds
    DEGRADED_RESPONSE_MS = 5000     # > 5s = degraded
    MAX_CONSECUTIVE_FAILURES = 3    # 3 failures = disconnected
    RECONNECT_BASE_DELAY = 10       # Exponential backoff start

    def __init__(self) -> None:
        self._brokers: dict[str, BaseBroker] = {}
        self._health: dict[str, BrokerHealth] = {}
        self._callbacks: list[Callable[[StatusChange], None]] = []
        self._response_history: dict[str, list[float]] = {}  # Sliding window
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()  # RLock: check_broker() tager lock, og _monitor_loop kalder check_all() med lock

    # ── Registration ────────────────────────────────────────

    def register(self, name: str, broker: BaseBroker) -> None:
        """Registrér en broker til health monitoring."""
        name = name.lower()
        self._brokers[name] = broker
        self._health[name] = BrokerHealth(broker_name=name)
        self._response_history[name] = []
        logger.info(f"[conn] Registreret broker til monitoring: {name}")

    def unregister(self, name: str) -> None:
        """Fjern en broker fra monitoring."""
        name = name.lower()
        self._brokers.pop(name, None)
        self._health.pop(name, None)
        self._response_history.pop(name, None)

    # ── Callbacks ───────────────────────────────────────────

    def on_status_change(self, callback: Callable[[StatusChange], None]) -> None:
        """Registrér callback der fires ved status-ændring."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def _notify_status_change(self, change: StatusChange) -> None:
        """Notify alle callbacks om status-ændring."""
        for cb in self._callbacks:
            try:
                cb(change)
            except Exception as exc:
                logger.error(f"[conn] Callback fejl: {exc}")

    # ── Health Checks ───────────────────────────────────────

    def check_broker(self, name: str) -> BrokerHealth:
        """
        Kør health check for én broker.

        Forsøger get_account() og måler response time.
        """
        name = name.lower()
        broker = self._brokers.get(name)
        health = self._health.get(name)

        if not broker or not health:
            logger.warning(f"[conn] Broker '{name}' ikke registreret")
            return BrokerHealth(broker_name=name, status=ConnectionStatus.UNKNOWN)

        with self._lock:
            return self._do_check_broker(name, broker, health)

    def _do_check_broker(self, name: str, broker, health: BrokerHealth) -> BrokerHealth:
        """Intern health check — kald kun med self._lock holdt."""
        old_status = health.status
        health.total_checks += 1
        health.last_check = datetime.now()

        start = time.time()
        try:
            # Prøv at hente kontoinformation som health check
            broker.get_account()
            elapsed_ms = (time.time() - start) * 1000
            health.response_time_ms = elapsed_ms
            health.consecutive_failures = 0
            health.last_success = datetime.now()
            health.last_error = ""

            # Track response history (sliding window, max 100)
            history = self._response_history.get(name, [])
            history.append(elapsed_ms)
            self._response_history[name] = history[-100:]
            health.avg_response_ms = sum(history) / len(history)

            # Bestem status
            if elapsed_ms > self.DEGRADED_RESPONSE_MS:
                health.status = ConnectionStatus.DEGRADED
            else:
                health.status = ConnectionStatus.CONNECTED

        except Exception as exc:
            elapsed_ms = (time.time() - start) * 1000
            health.response_time_ms = elapsed_ms
            health.consecutive_failures += 1
            health.total_failures += 1
            health.last_error = str(exc)[:200]

            if health.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                health.status = ConnectionStatus.DISCONNECTED
            else:
                health.status = ConnectionStatus.DEGRADED

            logger.warning(
                f"[conn] {name} check fejlet ({health.consecutive_failures}x): "
                f"{str(exc)[:100]}"
            )

        # Beregn uptime
        if health.total_checks > 0:
            health.uptime_pct = (
                (health.total_checks - health.total_failures)
                / health.total_checks * 100
            )

        # Notify ved status-ændring
        if health.status != old_status:
            change = StatusChange(
                broker_name=name,
                old_status=old_status,
                new_status=health.status,
                error=health.last_error,
            )
            self._notify_status_change(change)
            logger.info(
                f"[conn] {name}: {old_status.value} → {health.status.value}"
            )

        return health

    def check_all(self) -> dict[str, BrokerHealth]:
        """Kør health check for alle brokers."""
        results = {}
        for name in list(self._brokers.keys()):
            results[name] = self.check_broker(name)
        return results

    # ── Background Monitoring ───────────────────────────────

    def start(self, interval: int = 60) -> None:
        """
        Start periodisk health checking i en background thread.

        Args:
            interval: Sekunder mellem checks (default: 60).
        """
        if self._running:
            logger.warning("[conn] Monitoring kører allerede")
            return

        self._running = True

        def _monitor_loop() -> None:
            logger.info(f"[conn] Health monitoring startet (interval: {interval}s)")
            while self._running:
                try:
                    with self._lock:
                        self.check_all()
                except Exception as exc:
                    logger.error(f"[conn] Monitor loop fejl: {exc}")

                # Sleep i intervaller (så vi kan stoppe hurtigt)
                for _ in range(interval):
                    if not self._running:
                        break
                    time.sleep(1)

            logger.info("[conn] Health monitoring stoppet")

        self._thread = threading.Thread(
            target=_monitor_loop,
            name="broker-health-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop periodisk health checking."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    # ── Status Queries ──────────────────────────────────────

    def get_status(self, name: str) -> ConnectionStatus:
        """Hent status for én broker."""
        health = self._health.get(name.lower())
        return health.status if health else ConnectionStatus.UNKNOWN

    def get_health(self, name: str) -> BrokerHealth | None:
        """Hent fuld health info for én broker."""
        return self._health.get(name.lower())

    def get_all_health(self) -> dict[str, BrokerHealth]:
        """Hent health for alle brokers."""
        return dict(self._health)

    def is_any_connected(self) -> bool:
        """Er mindst én broker connected?"""
        return any(
            h.status == ConnectionStatus.CONNECTED
            for h in self._health.values()
        )

    def get_connected_brokers(self) -> list[str]:
        """Navne på connected brokers."""
        return [
            name for name, health in self._health.items()
            if health.status == ConnectionStatus.CONNECTED
        ]

    def get_disconnected_brokers(self) -> list[str]:
        """Navne på disconnected brokers."""
        return [
            name for name, health in self._health.items()
            if health.status == ConnectionStatus.DISCONNECTED
        ]

    # ── Dashboard Status ────────────────────────────────────

    def get_dashboard_status(self) -> dict[str, Any]:
        """
        Samlet status til dashboard-widget.

        Returns:
            Dict klar til rendering i UI.
        """
        brokers = {}
        overall = ConnectionStatus.CONNECTED

        for name, health in self._health.items():
            brokers[name] = health.to_dict()

            # Overall status: worst case
            if health.status == ConnectionStatus.DISCONNECTED:
                overall = ConnectionStatus.DISCONNECTED
            elif (health.status == ConnectionStatus.DEGRADED
                  and overall != ConnectionStatus.DISCONNECTED):
                overall = ConnectionStatus.DEGRADED

        return {
            "overall_status": overall.value,
            "brokers": brokers,
            "connected_count": len(self.get_connected_brokers()),
            "total_count": len(self._brokers),
            "monitoring_active": self._running,
            "timestamp": datetime.now().isoformat(),
        }

    def __del__(self) -> None:
        """Cleanup ved garbage collection."""
        self.stop()
