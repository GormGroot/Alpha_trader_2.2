"""
Backup Manager — daglig PostgreSQL dump, config backup, retention.

Features:
  - Daglig PostgreSQL dump (pg_dump)
  - Config-filer backup
  - SQLite databases backup
  - 30 dages retention med automatisk oprydning
  - Backup-integritet verificering (checksum)
  - Komprimering (gzip)
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from loguru import logger

TZ_CET = ZoneInfo("Europe/Copenhagen")


@dataclass
class BackupConfig:
    """Backup-konfiguration."""
    backup_dir: str = "backups"
    retention_days: int = 30
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "alpha_trader"
    pg_database: str = "alpha_trader"
    pg_password: str = ""
    compress: bool = True
    verify_checksum: bool = True
    # Directories/files to back up
    config_paths: list[str] = field(default_factory=lambda: [
        "config/",
        ".env",
        "docker-compose.yml",
    ])
    sqlite_paths: list[str] = field(default_factory=lambda: [
        "data_cache/orders.db",
        "data_cache/paper_portfolio.db",
        "data_cache/auto_trader_log.db",
        "data_cache/fifo_lots.db",
        "data_cache/tax_credit.db",
        "data_cache/mark_to_market.db",
        "data_cache/dividends.db",
        "data_cache/currency_pnl.db",
        "data_cache/learning.db",
        "data_cache/signal_log.db",
        "data_cache/audit_log.db",
    ])

    @classmethod
    def from_env(cls) -> BackupConfig:
        return cls(
            backup_dir=os.getenv("BACKUP_DIR", "backups"),
            retention_days=int(os.getenv("BACKUP_RETENTION_DAYS", "30")),
            pg_host=os.getenv("POSTGRES_HOST", "localhost"),
            pg_port=int(os.getenv("POSTGRES_PORT", "5432")),
            pg_user=os.getenv("POSTGRES_USER", "alpha_trader"),
            pg_database=os.getenv("POSTGRES_DB", "alpha_trader"),
            pg_password=os.getenv("POSTGRES_PASSWORD", ""),
        )


@dataclass
class BackupResult:
    """Resultat af backup-operation."""
    success: bool
    timestamp: datetime
    backup_path: str = ""
    size_bytes: int = 0
    checksum: str = ""
    duration_seconds: float = 0
    errors: list[str] = field(default_factory=list)
    components: dict[str, bool] = field(default_factory=dict)


class BackupManager:
    """
    Håndterer daglige backups med retention og integritet.

    Usage:
        bm = BackupManager()
        result = bm.run_daily_backup()
        bm.cleanup_old_backups()
        bm.verify_latest()
    """

    def __init__(self, config: BackupConfig | None = None):
        self._config = config or BackupConfig.from_env()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Opret backup-mapper."""
        os.makedirs(self._config.backup_dir, exist_ok=True)
        os.makedirs(os.path.join(self._config.backup_dir, "pg"), exist_ok=True)
        os.makedirs(os.path.join(self._config.backup_dir, "config"), exist_ok=True)
        os.makedirs(os.path.join(self._config.backup_dir, "sqlite"), exist_ok=True)

    def _timestamp_str(self) -> str:
        return datetime.now(TZ_CET).strftime("%Y%m%d_%H%M%S")

    def _compute_checksum(self, filepath: str) -> str:
        """SHA-256 checksum."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    # ── PostgreSQL Backup ──────────────────────────────────

    def backup_postgresql(self) -> tuple[bool, str]:
        """Kør pg_dump og gem komprimeret backup."""
        ts = self._timestamp_str()
        filename = f"pg_dump_{ts}.sql"
        filepath = os.path.join(self._config.backup_dir, "pg", filename)

        env = os.environ.copy()
        if self._config.pg_password:
            env["PGPASSWORD"] = self._config.pg_password

        try:
            cmd = [
                "pg_dump",
                "-h", self._config.pg_host,
                "-p", str(self._config.pg_port),
                "-U", self._config.pg_user,
                "-d", self._config.pg_database,
                "--format=plain",
                "--no-owner",
                "--no-acl",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
            )

            if result.returncode != 0:
                logger.warning(f"[backup] pg_dump failed: {result.stderr[:200]}")
                return False, f"pg_dump error: {result.stderr[:200]}"

            # Write and compress
            if self._config.compress:
                gz_path = filepath + ".gz"
                with gzip.open(gz_path, "wt", encoding="utf-8") as f:
                    f.write(result.stdout)
                filepath = gz_path
            else:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(result.stdout)

            size = os.path.getsize(filepath)
            logger.info(f"[backup] PostgreSQL dump: {filepath} ({size / 1024:.0f} KB)")
            return True, filepath

        except FileNotFoundError:
            logger.warning("[backup] pg_dump not found — PostgreSQL backup skipped")
            return False, "pg_dump not found"
        except subprocess.TimeoutExpired:
            return False, "pg_dump timeout (300s)"
        except Exception as e:
            return False, str(e)

    # ── Config Backup ──────────────────────────────────────

    def backup_configs(self) -> tuple[bool, str]:
        """Kopiér config-filer til backup."""
        ts = self._timestamp_str()
        target_dir = os.path.join(self._config.backup_dir, "config", ts)
        os.makedirs(target_dir, exist_ok=True)
        errors = []

        for path in self._config.config_paths:
            try:
                if os.path.isdir(path):
                    dest = os.path.join(target_dir, os.path.basename(path))
                    shutil.copytree(path, dest, dirs_exist_ok=True)
                elif os.path.isfile(path):
                    shutil.copy2(path, target_dir)
                else:
                    logger.debug(f"[backup] Config path not found: {path}")
            except Exception as e:
                errors.append(f"{path}: {e}")

        if errors:
            logger.warning(f"[backup] Config backup errors: {errors}")

        # Compress the directory
        if self._config.compress:
            archive = shutil.make_archive(target_dir, "gztar", target_dir)
            shutil.rmtree(target_dir, ignore_errors=True)
            logger.info(f"[backup] Config backup: {archive}")
            return len(errors) == 0, archive

        return len(errors) == 0, target_dir

    # ── SQLite Backup ──────────────────────────────────────

    def backup_sqlite(self) -> tuple[bool, str]:
        """Kopiér SQLite-filer med backup API (safe copy)."""
        ts = self._timestamp_str()
        target_dir = os.path.join(self._config.backup_dir, "sqlite", ts)
        os.makedirs(target_dir, exist_ok=True)
        errors = []
        copied = 0

        for db_path in self._config.sqlite_paths:
            if not os.path.isfile(db_path):
                continue

            try:
                import sqlite3
                dest = os.path.join(target_dir, os.path.basename(db_path))

                # Use SQLite backup API for consistent copy
                with sqlite3.connect(db_path) as src_conn, sqlite3.connect(dest) as dst_conn:
                    src_conn.backup(dst_conn)

                # Compress
                if self._config.compress:
                    with open(dest, "rb") as f_in:
                        with gzip.open(f"{dest}.gz", "wb") as f_out:
                            f_out.write(f_in.read())
                    os.remove(dest)

                copied += 1
            except Exception as e:
                errors.append(f"{db_path}: {e}")

        if errors:
            logger.warning(f"[backup] SQLite backup errors: {errors}")

        logger.info(f"[backup] SQLite backup: {copied} databases to {target_dir}")
        return len(errors) == 0, target_dir

    # ── Full Daily Backup ──────────────────────────────────

    def run_daily_backup(self) -> BackupResult:
        """Kør komplet daglig backup: PG + config + SQLite."""
        started = time.time()
        ts = datetime.now(TZ_CET)
        errors = []
        components = {}

        # PostgreSQL
        pg_ok, pg_path = self.backup_postgresql()
        components["postgresql"] = pg_ok
        if not pg_ok:
            errors.append(f"PostgreSQL: {pg_path}")

        # Config
        cfg_ok, cfg_path = self.backup_configs()
        components["config"] = cfg_ok
        if not cfg_ok:
            errors.append(f"Config: {cfg_path}")

        # SQLite
        sq_ok, sq_path = self.backup_sqlite()
        components["sqlite"] = sq_ok
        if not sq_ok:
            errors.append(f"SQLite: {sq_path}")

        duration = time.time() - started
        success = all(components.values())

        # Compute total backup size
        total_size = 0
        for root, _, files in os.walk(self._config.backup_dir):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    if os.path.getmtime(fp) > started - 10:
                        total_size += os.path.getsize(fp)
                except OSError:
                    pass

        result = BackupResult(
            success=success,
            timestamp=ts,
            backup_path=self._config.backup_dir,
            size_bytes=total_size,
            duration_seconds=duration,
            errors=errors,
            components=components,
        )

        log_level = "info" if success else "warning"
        getattr(logger, log_level)(
            f"[backup] Daily backup {'OK' if success else 'PARTIAL'}: "
            f"{total_size / 1024:.0f} KB in {duration:.1f}s"
        )

        # Cleanup old
        self.cleanup_old_backups()

        return result

    # ── Retention / Cleanup ────────────────────────────────

    def cleanup_old_backups(self) -> int:
        """Slet backups ældre end retention_days."""
        cutoff = time.time() - (self._config.retention_days * 86400)
        removed = 0

        for subdir in ["pg", "config", "sqlite"]:
            dir_path = os.path.join(self._config.backup_dir, subdir)
            if not os.path.isdir(dir_path):
                continue

            for entry in os.listdir(dir_path):
                full_path = os.path.join(dir_path, entry)
                try:
                    if os.path.getmtime(full_path) < cutoff:
                        if os.path.isdir(full_path):
                            shutil.rmtree(full_path)
                        else:
                            os.remove(full_path)
                        removed += 1
                except OSError as e:
                    logger.warning(f"[backup] Cleanup error: {full_path}: {e}")

        if removed:
            logger.info(f"[backup] Cleaned up {removed} old backup(s)")
        return removed

    # ── Verification ───────────────────────────────────────

    def verify_latest(self) -> dict:
        """Verificér seneste backup-integritet."""
        results = {}

        for subdir in ["pg", "config", "sqlite"]:
            dir_path = os.path.join(self._config.backup_dir, subdir)
            if not os.path.isdir(dir_path):
                results[subdir] = {"exists": False}
                continue

            entries = sorted(os.listdir(dir_path), reverse=True)
            if not entries:
                results[subdir] = {"exists": False}
                continue

            latest = os.path.join(dir_path, entries[0])

            if os.path.isfile(latest):
                size = os.path.getsize(latest)
                checksum = self._compute_checksum(latest)
                age_hours = (time.time() - os.path.getmtime(latest)) / 3600

                results[subdir] = {
                    "exists": True,
                    "path": latest,
                    "size_bytes": size,
                    "checksum_sha256": checksum,
                    "age_hours": round(age_hours, 1),
                    "valid": size > 0 and age_hours < 48,
                }
            else:
                results[subdir] = {
                    "exists": True,
                    "path": latest,
                    "is_directory": True,
                }

        return results

    # ── Status ─────────────────────────────────────────────

    def get_status(self) -> dict:
        """Returnér backup-status til dashboard."""
        total_size = 0
        file_count = 0

        for root, _, files in os.walk(self._config.backup_dir):
            for f in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, f))
                    file_count += 1
                except OSError:
                    pass

        return {
            "backup_dir": self._config.backup_dir,
            "retention_days": self._config.retention_days,
            "total_size_mb": round(total_size / (1024 * 1024), 1),
            "file_count": file_count,
            "latest_verification": self.verify_latest(),
        }


# ── Windows Task Scheduler Helper ──────────────────────────

def generate_windows_autostart_script(
    python_path: str = "python",
    platform_dir: str = ".",
    dashboard_port: int = 8050,
) -> str:
    """
    Generér PowerShell-script til Windows Task Scheduler.

    Konfigurér i Task Scheduler:
      - Trigger: At logon / At startup
      - Action: powershell.exe -File C:\\path\\to\\autostart.ps1
    """
    return f"""# Alpha Trading Platform — Autostart Script
# Konfigurér i Windows Task Scheduler som "At Logon" eller "At Startup"

$ErrorActionPreference = "Continue"
$LogFile = "{platform_dir}\\logs\\autostart.log"

function Write-Log {{
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp $Message" | Out-File -Append $LogFile
    Write-Host "$timestamp $Message"
}}

Write-Log "=== Alpha Trading Platform starting ==="

# 1. Start Docker (PostgreSQL, Redis)
Write-Log "Starting Docker containers..."
try {{
    Set-Location "{platform_dir}"
    docker-compose up -d
    Start-Sleep -Seconds 10
    Write-Log "Docker containers started"
}} catch {{
    Write-Log "WARNING: Docker start failed: $($_.Exception.Message)"
}}

# 2. Start Python platform (scheduler)
Write-Log "Starting Alpha Trader scheduler..."
$schedulerProcess = Start-Process -FilePath "{python_path}" -ArgumentList "-m src.ops.daily_scheduler" -WorkingDirectory "{platform_dir}" -PassThru -WindowStyle Hidden -RedirectStandardError "{platform_dir}\\logs\\scheduler_error.log"
Write-Log "Scheduler PID: $($schedulerProcess.Id)"

# 3. Start Dashboard
Write-Log "Starting Dashboard on port {dashboard_port}..."
$dashboardProcess = Start-Process -FilePath "{python_path}" -ArgumentList "-m src.dashboard.app" -WorkingDirectory "{platform_dir}" -PassThru -WindowStyle Hidden -RedirectStandardError "{platform_dir}\\logs\\dashboard_error.log"
Write-Log "Dashboard PID: $($dashboardProcess.Id)"

# 4. Monitor and restart on crash
Write-Log "Monitoring processes..."
while ($true) {{
    Start-Sleep -Seconds 60

    if ($schedulerProcess.HasExited) {{
        Write-Log "WARNING: Scheduler crashed (exit code: $($schedulerProcess.ExitCode)). Restarting..."
        $schedulerProcess = Start-Process -FilePath "{python_path}" -ArgumentList "-m src.ops.daily_scheduler" -WorkingDirectory "{platform_dir}" -PassThru -WindowStyle Hidden
        Write-Log "Scheduler restarted, PID: $($schedulerProcess.Id)"
    }}

    if ($dashboardProcess.HasExited) {{
        Write-Log "WARNING: Dashboard crashed (exit code: $($dashboardProcess.ExitCode)). Restarting..."
        $dashboardProcess = Start-Process -FilePath "{python_path}" -ArgumentList "-m src.dashboard.app" -WorkingDirectory "{platform_dir}" -PassThru -WindowStyle Hidden
        Write-Log "Dashboard restarted, PID: $($dashboardProcess.Id)"
    }}
}}
"""
