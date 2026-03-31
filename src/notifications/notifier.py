"""
Notifikationssystem – sender advarsler via email, log eller callbacks.

Understøtter flere kanaler:
  - EmailChannel: SMTP-baseret email (kræver konfiguration)
  - LogChannel: Logger til fil via loguru (altid aktiv)
  - CallbackChannel: Custom handler (til dashboard, webhooks osv.)

Alle notifikationer gemmes i SQLite til historik.
"""

from __future__ import annotations

import html as _html
import smtplib
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Callable

from loguru import logger


# ── Kanaler ───────────────────────────────────────────────────


class NotificationChannel(ABC):
    """Abstrakt base for en notifikationskanal."""

    @abstractmethod
    def send(self, severity: str, title: str, message: str, category: str) -> bool:
        """Send en notifikation. Returnér True ved succes."""
        ...


class LogChannel(NotificationChannel):
    """Logger notifikationer til logfilen (altid aktiv)."""

    def send(self, severity: str, title: str, message: str, category: str) -> bool:
        log_func = {
            "CRITICAL": logger.error,
            "WARNING": logger.warning,
            "INFO": logger.info,
        }.get(severity, logger.info)

        log_func(f"[notification:{category}] {title}")
        for line in message.split("\n")[:5]:  # Max 5 linjer i loggen
            log_func(f"  {line}")
        return True


class EmailChannel(NotificationChannel):
    """
    Sender notifikationer via SMTP email.

    Konfigurér via miljøvariabler:
      NOTIFY_SMTP_HOST, NOTIFY_SMTP_PORT, NOTIFY_SMTP_USER,
      NOTIFY_SMTP_PASSWORD, NOTIFY_FROM_EMAIL, NOTIFY_TO_EMAIL
    """

    def __init__(
        self,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        from_email: str = "",
        to_email: str = "",
    ) -> None:
        self._host = smtp_host
        self._port = smtp_port
        self._user = smtp_user
        self._password = smtp_password
        self._from = from_email
        self._to = to_email
        self._configured = bool(smtp_user and smtp_password and to_email)
        self._send_cooldown: dict[str, float] = {}  # category → last send timestamp
        self._cooldown_seconds = 300  # 5 min mellem emails per category

    def __repr__(self) -> str:
        return f"EmailChannel(host={self._host}, user={self._user}, to={self._to})"

    @property
    def is_configured(self) -> bool:
        return self._configured

    def send(self, severity: str, title: str, message: str, category: str) -> bool:
        if not self._configured:
            logger.debug("[email] Email ikke konfigureret – springer over")
            return False

        # Rate limiting: max 1 email per 5 min per category
        import time
        now = time.time()
        last_sent = self._send_cooldown.get(category, 0)
        if now - last_sent < self._cooldown_seconds:
            logger.debug(f"[email] Rate limited: {category} (cooldown {self._cooldown_seconds}s)")
            return False
        self._send_cooldown[category] = now

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{severity}] {title}"
            msg["From"] = self._from or self._user
            msg["To"] = self._to

            # Tekst-version
            text_body = (
                f"{title}\n"
                f"{'=' * len(title)}\n\n"
                f"Severity: {severity}\n"
                f"Kategori: {category}\n"
                f"Tid: {datetime.now().isoformat()}\n\n"
                f"{message}\n\n"
                f"---\n"
                f"Alpha Trading Platform – Automatisk notifikation\n"
                f"⚠️ Vejledende – verificér med revisor"
            )
            msg.attach(MIMEText(text_body, "plain", "utf-8"))

            # HTML-version
            html_body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px;">
              <div style="max-width: 600px; margin: 0 auto; background: #16213e; border-radius: 12px; padding: 24px;">
                <h2 style="color: {'#ff4757' if severity == 'CRITICAL' else '#ffa502' if severity == 'WARNING' else '#00d4aa'};">
                  {_html.escape(title)}
                </h2>
                <p style="color: #888; font-size: 12px;">
                  {_html.escape(severity)} | {_html.escape(category)} | {datetime.now().strftime('%d-%m-%Y %H:%M')}
                </p>
                <div style="background: #0f3460; padding: 16px; border-radius: 8px; margin: 16px 0;">
                  <pre style="white-space: pre-wrap; color: #e0e0e0; margin: 0;">{_html.escape(message)}</pre>
                </div>
                <p style="color: #666; font-size: 11px; margin-top: 24px;">
                  Alpha Trading Platform – Automatisk notifikation<br/>
                  ⚠️ Vejledende beregning – verificér med revisor
                </p>
              </div>
            </body>
            </html>
            """
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            with smtplib.SMTP(self._host, self._port) as server:
                server.starttls()
                server.login(self._user, self._password)
                server.send_message(msg)

            logger.info(f"[email] Sendt: {title} → {self._to}")
            return True

        except Exception as exc:
            logger.error(f"[email] Fejl ved afsendelse: {exc}")
            return False


class CallbackChannel(NotificationChannel):
    """Kanal der kalder en custom funktion (til dashboard, webhooks osv.)."""

    def __init__(self, callback: Callable[[str, str, str, str], None]) -> None:
        self._callback = callback

    def send(self, severity: str, title: str, message: str, category: str) -> bool:
        try:
            self._callback(severity, title, message, category)
            return True
        except Exception as exc:
            logger.error(f"[callback] Fejl: {exc}")
            return False


# ── Notifier ──────────────────────────────────────────────────


class Notifier:
    """
    Central notifikationshub.

    Modtager alerts fra TaxAdvisor (og andre moduler) og sender
    dem via alle konfigurerede kanaler. Historik gemmes i SQLite.
    """

    def __init__(self, cache_dir: str = "data_cache") -> None:
        self._channels: list[NotificationChannel] = []
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._cache_dir / "notifications.db"
        self._init_db()

        # Altid log
        self.add_channel(LogChannel())

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS notification_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    category TEXT NOT NULL,
                    channels_sent INTEGER DEFAULT 0,
                    channels_failed INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_notif_ts
                ON notification_history (timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_notif_cat
                ON notification_history (category)
            """)

    def add_channel(self, channel: NotificationChannel) -> None:
        """Tilføj en notifikationskanal."""
        self._channels.append(channel)

    def send(
        self,
        severity: str,
        title: str,
        message: str,
        category: str = "general",
    ) -> int:
        """
        Send notifikation via alle kanaler.

        Returns:
            Antal kanaler der modtog notifikationen.
        """
        sent = 0
        failed = 0

        for ch in self._channels:
            try:
                if ch.send(severity, title, message, category):
                    sent += 1
                else:
                    failed += 1
            except Exception as exc:
                logger.error(f"[notifier] Kanalfejl: {exc}")
                failed += 1

        # Gem i historik
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO notification_history
                   (timestamp, severity, title, message, category,
                    channels_sent, channels_failed)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    severity, title, message, category,
                    sent, failed,
                ),
            )

        return sent

    def send_tax_alert(self, alert) -> int:
        """
        Send en TaxAlert.

        Args:
            alert: TaxAlert fra TaxAdvisor.

        Returns:
            Antal kanaler der modtog notifikationen.
        """
        return self.send(
            severity=alert.severity,
            title=alert.title,
            message=alert.message,
            category=alert.category,
        )

    def get_history(
        self,
        category: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """Hent notifikationshistorik."""
        query = "SELECT * FROM notification_history WHERE 1=1"
        params: list = []

        if category:
            query += " AND category = ?"
            params.append(category)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_unread_count(self, since: str | None = None) -> int:
        """Antal notifikationer siden en given dato."""
        if not since:
            since = datetime.now().replace(
                hour=0, minute=0, second=0
            ).isoformat()

        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM notification_history WHERE timestamp >= ?",
                (since,),
            ).fetchone()
            return row[0] if row else 0
