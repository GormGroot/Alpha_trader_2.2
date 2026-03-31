"""
Tests for ops modules — scheduler, backup, email.

Tester:
  - DailyScheduler task management
  - Holiday/weekend detection
  - BackupManager — backup & cleanup
  - EmailReportRunner — HTML generation
"""

from __future__ import annotations

import os
import tempfile
import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, patch


# ── Scheduler Tests ────────────────────────────────────────

class TestDailyScheduler:
    def test_import(self):
        from src.ops.daily_scheduler import DailyScheduler, ScheduledTask, TaskPriority
        assert DailyScheduler is not None

    def test_default_tasks(self):
        from src.ops.daily_scheduler import DailyScheduler
        scheduler = DailyScheduler()
        schedule = scheduler.get_schedule()
        assert len(schedule) >= 6  # Expanded to 24/7 global coverage
        names = [t["name"] for t in schedule]
        assert "morning_check" in names
        assert "eu_market_open" in names
        assert "us_market_close" in names
        assert "maintenance" in names

    def test_run_task_now(self):
        from src.ops.daily_scheduler import DailyScheduler
        scheduler = DailyScheduler()
        # Maintenance doesn't require market day — safest to test
        result = scheduler.run_task_now("maintenance")
        assert result is not None
        assert result.task_name == "maintenance"

    def test_run_nonexistent_task(self):
        from src.ops.daily_scheduler import DailyScheduler
        scheduler = DailyScheduler()
        result = scheduler.run_task_now("nonexistent")
        assert result is None

    def test_enable_disable_task(self):
        from src.ops.daily_scheduler import DailyScheduler
        scheduler = DailyScheduler()
        scheduler.enable_task("morning_check", False)
        schedule = scheduler.get_schedule()
        mc = next(t for t in schedule if t["name"] == "morning_check")
        assert mc["enabled"] is False

    def test_get_results_empty(self):
        from src.ops.daily_scheduler import DailyScheduler
        scheduler = DailyScheduler()
        results = scheduler.get_results()
        assert isinstance(results, list)


class TestMarketDayDetection:
    def test_weekday_is_market_day(self):
        from src.ops.daily_scheduler import is_market_day
        # 2026-03-16 is Monday
        assert is_market_day(date(2026, 3, 16)) is True

    def test_saturday_not_market_day(self):
        from src.ops.daily_scheduler import is_market_day
        assert is_market_day(date(2026, 3, 14)) is False  # Saturday

    def test_sunday_not_market_day(self):
        from src.ops.daily_scheduler import is_market_day
        assert is_market_day(date(2026, 3, 15)) is False  # Sunday

    def test_new_years_not_market_day(self):
        from src.ops.daily_scheduler import is_market_day
        assert is_market_day(date(2026, 1, 1)) is False

    def test_christmas_not_market_day(self):
        from src.ops.daily_scheduler import is_market_day
        assert is_market_day(date(2026, 12, 25)) is False


# ── Backup Tests ───────────────────────────────────────────

class TestBackupManager:
    @pytest.fixture
    def backup_dir(self):
        return tempfile.mkdtemp()

    def test_import(self):
        from src.ops.backup import BackupManager, BackupConfig, BackupResult
        assert BackupManager is not None

    def test_ensure_dirs(self, backup_dir):
        from src.ops.backup import BackupManager, BackupConfig
        config = BackupConfig(backup_dir=backup_dir)
        bm = BackupManager(config)

        assert os.path.isdir(os.path.join(backup_dir, "pg"))
        assert os.path.isdir(os.path.join(backup_dir, "config"))
        assert os.path.isdir(os.path.join(backup_dir, "sqlite"))

    def test_backup_configs(self, backup_dir):
        from src.ops.backup import BackupManager, BackupConfig
        config = BackupConfig(
            backup_dir=backup_dir,
            config_paths=[],  # No config paths — should succeed anyway
        )
        bm = BackupManager(config)
        ok, path = bm.backup_configs()
        assert ok is True

    def test_cleanup_old_backups(self, backup_dir):
        from src.ops.backup import BackupManager, BackupConfig
        config = BackupConfig(backup_dir=backup_dir, retention_days=0)
        bm = BackupManager(config)

        # Create a fake old backup file
        pg_dir = os.path.join(backup_dir, "pg")
        fake_file = os.path.join(pg_dir, "old_backup.sql.gz")
        with open(fake_file, "w") as f:
            f.write("test")
        # Set modification time to 100 days ago
        old_time = datetime.now().timestamp() - (100 * 86400)
        os.utime(fake_file, (old_time, old_time))

        removed = bm.cleanup_old_backups()
        assert removed >= 1
        assert not os.path.exists(fake_file)

    def test_verify_latest_empty(self, backup_dir):
        from src.ops.backup import BackupManager, BackupConfig
        config = BackupConfig(backup_dir=backup_dir)
        bm = BackupManager(config)
        result = bm.verify_latest()
        assert isinstance(result, dict)

    def test_get_status(self, backup_dir):
        from src.ops.backup import BackupManager, BackupConfig
        config = BackupConfig(backup_dir=backup_dir)
        bm = BackupManager(config)
        status = bm.get_status()
        assert "backup_dir" in status
        assert "total_size_mb" in status


# ── Email Report Tests ─────────────────────────────────────

class TestEmailReports:
    def test_import(self):
        from src.ops.email_reports import EmailSender, ReportGenerator, ReportData, AlarmManager
        assert EmailSender is not None

    def test_smtp_config_from_env(self):
        from src.ops.email_reports import SMTPConfig
        config = SMTPConfig.from_env()
        assert isinstance(config.host, str)
        assert isinstance(config.port, int)

    def test_smtp_not_configured(self):
        from src.ops.email_reports import SMTPConfig
        config = SMTPConfig()  # No credentials
        assert config.is_configured is False

    def test_morning_report_html(self):
        from src.ops.email_reports import ReportGenerator, ReportData
        data = ReportData(
            portfolio_value_dkk=1_500_000,
            daily_pnl_dkk=12_500,
            daily_pnl_pct=0.84,
            positions_count=42,
            broker_status={"alpaca": "connected", "nordnet": "connected"},
        )
        html = ReportGenerator.morning_report(data)
        assert "1,500,000" in html
        assert "Morgenrapport" in html

    def test_evening_report_html(self):
        from src.ops.email_reports import ReportGenerator, ReportData
        data = ReportData(
            portfolio_value_dkk=1_500_000,
            daily_pnl_dkk=-5_000,
            daily_pnl_pct=-0.33,
            ytd_pnl_dkk=85_000,
            tax_credit_balance=350_000,
            estimated_tax_ytd=18_700,
        )
        html = ReportGenerator.evening_report(data)
        assert "Aftenrapport" in html
        assert "1,500,000" in html

    def test_weekly_report_html(self):
        from src.ops.email_reports import ReportGenerator, ReportData
        data = ReportData(portfolio_value_dkk=1_500_000)
        html = ReportGenerator.weekly_report(data, week_pnl=25_000, week_pnl_pct=1.7)
        assert "Ugentlig" in html

    def test_alarm_cooldown(self):
        from src.ops.email_reports import AlarmManager, EmailSender, SMTPConfig
        config = SMTPConfig()  # Not configured — sends will fail but logic works
        sender = EmailSender(config)
        alarm = AlarmManager(sender)

        # First call should try to send (will fail because SMTP not configured)
        alarm._cooldown_minutes = 0  # Disable cooldown for test
        result = alarm.drawdown_alarm(5.5, 1_000_000)
        # Will be False because SMTP not configured, but shouldn't crash
        assert isinstance(result, bool)


# ── Windows Autostart Script ──────────────────────────────

class TestWindowsAutostart:
    def test_generate_script(self):
        from src.ops.backup import generate_windows_autostart_script
        script = generate_windows_autostart_script(
            python_path="python3",
            platform_dir="C:\\AlphaTrader",
            dashboard_port=8050,
        )
        assert "Alpha Trading Platform" in script
        assert "docker-compose" in script
        assert "C:\\AlphaTrader" in script
