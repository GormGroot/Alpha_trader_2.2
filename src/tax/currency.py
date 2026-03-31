"""
Valutaomregning USD → DKK med daglige kurser fra ECB/Nationalbanken.

Cacher kurser lokalt i SQLite for hurtig adgang og offline-brug.
Bruges til at beregne korrekt DKK-værdi for hver handel på handelsdatoen.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import httpx
from loguru import logger


class CurrencyConverter:
    """
    Henter og cacher USD/DKK-dagskurser.

    Primær kilde: ECB Statistical Data Warehouse (gratis, ingen API-nøgle).
    Fallback: fast kurs fra config.
    """

    _ECB_URL = (
        "https://data-api.ecb.europa.eu/service/data/EXR/"
        "D.USD.DKK.SP00.A?format=csvdata"
    )

    def __init__(
        self,
        cache_dir: str = "data_cache",
        fallback_rate: float = 6.90,
    ) -> None:
        self._fallback_rate = fallback_rate
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self._cache_dir / "currency_cache.db"
        self._memory: dict[str, float] = {}
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fx_rates (
                    date TEXT PRIMARY KEY,
                    usd_dkk REAL NOT NULL,
                    source TEXT DEFAULT 'ecb',
                    fetched_at TEXT
                )
            """)

    # ── Offentlig API ─────────────────────────────────────────

    def get_rate(self, date: str) -> float:
        """
        Hent USD/DKK-kursen for en given dato (YYYY-MM-DD).

        Returnerer cachet kurs hvis tilgængelig, ellers henter fra ECB.
        For weekender/helligdage bruges nærmeste foregående hverdag.
        """
        # Tjek memory-cache
        if date in self._memory:
            return self._memory[date]

        # Tjek SQLite-cache
        rate = self._load_from_db(date)
        if rate:
            self._memory[date] = rate
            return rate

        # Prøv at hente fra ECB
        try:
            self._fetch_ecb_rates(date)
            rate = self._load_from_db(date)
            if rate:
                self._memory[date] = rate
                return rate
        except Exception as exc:
            logger.warning(f"Kunne ikke hente valutakurs for {date}: {exc}")

        # Prøv nærmeste foregående hverdag (op til 7 dage tilbage)
        dt = datetime.strptime(date, "%Y-%m-%d")
        for offset in range(1, 8):
            prev = (dt - timedelta(days=offset)).strftime("%Y-%m-%d")
            rate = self._load_from_db(prev)
            if rate:
                logger.debug(f"Bruger kurs fra {prev} for {date}: {rate:.4f}")
                self._memory[date] = rate
                return rate

        # Fallback
        logger.warning(
            f"Ingen kurs fundet for {date}, bruger fallback: {self._fallback_rate}"
        )
        self._memory[date] = self._fallback_rate
        return self._fallback_rate

    def convert_usd_to_dkk(self, amount_usd: float, date: str) -> float:
        """Konvertér USD til DKK med kurs fra given dato."""
        rate = self.get_rate(date)
        return amount_usd * rate

    def bulk_fetch(self, year: int) -> int:
        """
        Hent alle kurser for et helt år. Returnerer antal hentede kurser.
        """
        try:
            return self._fetch_ecb_rates_for_year(year)
        except Exception as exc:
            logger.error(f"Kunne ikke hente kurser for {year}: {exc}")
            return 0

    # ── Interne metoder ───────────────────────────────────────

    def _load_from_db(self, date: str) -> float | None:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT usd_dkk FROM fx_rates WHERE date = ?", (date,)
            ).fetchone()
            return row[0] if row else None

    def _save_rate(self, date: str, rate: float, source: str = "ecb") -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO fx_rates (date, usd_dkk, source, fetched_at)
                   VALUES (?, ?, ?, ?)""",
                (date, rate, source, datetime.now().isoformat()),
            )

    def _fetch_ecb_rates(self, date: str) -> None:
        """Hent kurser fra ECB for en periode omkring den givne dato."""
        dt = datetime.strptime(date, "%Y-%m-%d")
        start = (dt - timedelta(days=10)).strftime("%Y-%m-%d")
        end = dt.strftime("%Y-%m-%d")

        url = self._ECB_URL + f"&startPeriod={start}&endPeriod={end}"
        self._fetch_and_parse_ecb(url)

    def _fetch_ecb_rates_for_year(self, year: int) -> int:
        """Hent alle dagskurser for et helt år fra ECB."""
        url = (
            self._ECB_URL
            + f"&startPeriod={year}-01-01&endPeriod={year}-12-31"
        )
        return self._fetch_and_parse_ecb(url)

    def _fetch_and_parse_ecb(self, url: str) -> int:
        """Hent og parse ECB CSV-data. Returnerer antal gemte kurser."""
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()

        count = 0
        date_idx = None
        value_idx = None
        for line in resp.text.splitlines():
            parts = line.split(",")
            # ECB CSV: kolonne med dato (TIME_PERIOD) og kurs (OBS_VALUE)
            # Find header-index
            if "TIME_PERIOD" in line:
                headers = parts
                try:
                    date_idx = headers.index("TIME_PERIOD")
                    value_idx = headers.index("OBS_VALUE")
                except ValueError:
                    # Prøv at finde i stripped headers
                    date_idx = next(
                        i for i, h in enumerate(headers) if "TIME_PERIOD" in h
                    )
                    value_idx = next(
                        i for i, h in enumerate(headers) if "OBS_VALUE" in h
                    )
                continue

            if date_idx is None or value_idx is None:
                continue
            try:
                date_val = parts[date_idx].strip()
                rate_val = float(parts[value_idx].strip())
                if date_val and rate_val > 0:
                    self._save_rate(date_val, rate_val, "ecb")
                    self._memory[date_val] = rate_val
                    count += 1
            except (IndexError, ValueError):
                continue

        logger.info(f"ECB: {count} USD/DKK-kurser hentet")
        return count
