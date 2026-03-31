"""
Lagerbeskatning (Mark-to-Market) Engine.

Dansk selskabsskat kræver lagerbeskatning på finansielle aktiver:
  - Ved regnskabsårets udgang beregnes urealiseret gevinst/tab
  - Skat beregnes af ÆNDRINGEN i værdi fra primo til ultimo
  - Primo = værdi ved årets start ELLER anskaffelsessum (nye positioner)
  - Ultimo = markedsværdi ved årets slut

Denne engine:
  1. Gemmer primo-værdier (year_start snapshots)
  2. Beregner daglig estimeret lagerbeskatning (YTD)
  3. Genererer årsafslutning (final mark-to-market)
  4. Producerer rapport til revisor
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


# ── Dataklasser ─────────────────────────────────────────────

@dataclass
class MTMPosition:
    """Mark-to-market beregning for én position."""
    symbol: str
    qty: float
    primo_price_dkk: float         # Pris ved årets start (DKK per enhed)
    primo_value_dkk: float         # Total primo-værdi
    ultimo_price_dkk: float        # Pris ved årets slut / nu
    ultimo_value_dkk: float        # Total ultimo-værdi
    mtm_pnl_dkk: float            # Mark-to-market P&L (ultimo - primo)
    tax_22pct_dkk: float           # 22% af P&L
    is_new_position: bool = False  # Tilgået i løbet af året
    acquisition_date: str = ""
    broker: str = ""
    currency: str = "DKK"
    fx_rate: float = 1.0


@dataclass
class MTMSummary:
    """Samlet mark-to-market opgørelse."""
    year: int
    snapshot_date: str               # Dato for beregningen
    is_final: bool = False           # True = årsafslutning

    # Positioner
    positions: list[MTMPosition] = field(default_factory=list)
    position_count: int = 0

    # Aggregater
    total_primo_dkk: float = 0.0
    total_ultimo_dkk: float = 0.0
    total_mtm_pnl_dkk: float = 0.0
    total_gains_dkk: float = 0.0    # Kun positive
    total_losses_dkk: float = 0.0   # Kun negative
    estimated_tax_dkk: float = 0.0

    # Per broker
    by_broker: dict[str, float] = field(default_factory=dict)
    # Per currency
    by_currency: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "date": self.snapshot_date,
            "is_final": self.is_final,
            "position_count": self.position_count,
            "total_primo": round(self.total_primo_dkk, 2),
            "total_ultimo": round(self.total_ultimo_dkk, 2),
            "total_mtm_pnl": round(self.total_mtm_pnl_dkk, 2),
            "gains": round(self.total_gains_dkk, 2),
            "losses": round(self.total_losses_dkk, 2),
            "estimated_tax": round(self.estimated_tax_dkk, 2),
            "by_broker": {
                k: round(v, 2) for k, v in self.by_broker.items()
            },
        }


# ── Mark-to-Market Engine ──────────────────────────────────

class MarkToMarketEngine:
    """
    Lagerbeskatnings-engine.

    Brug:
        engine = MarkToMarketEngine()

        # Gem primo-snapshot ved årets start (1. januar)
        engine.save_year_start_snapshot(2025, positions)

        # Daglig estimering
        ytd = engine.calculate_ytd(2025, current_positions)
        print(f"Estimeret lagerbeskatning: {ytd.estimated_tax_dkk:,.0f} DKK")

        # Årsafslutning (31. december)
        final = engine.year_end_calculation(2025, final_positions)
    """

    TAX_RATE = 0.22

    def __init__(self, db_path: str = "data_cache/mark_to_market.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            # Year-start snapshots
            conn.execute("""
                CREATE TABLE IF NOT EXISTS year_start_snapshots (
                    year INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price_dkk REAL NOT NULL,
                    value_dkk REAL NOT NULL,
                    broker TEXT DEFAULT '',
                    currency TEXT DEFAULT 'DKK',
                    snapshot_date TEXT NOT NULL,
                    PRIMARY KEY (year, symbol, broker)
                )
            """)
            # MTM calculations (historik)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mtm_calculations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    year INTEGER NOT NULL,
                    calculation_date TEXT NOT NULL,
                    is_final INTEGER DEFAULT 0,
                    total_primo REAL,
                    total_ultimo REAL,
                    total_mtm_pnl REAL,
                    estimated_tax REAL,
                    position_count INTEGER
                )
            """)

    # ── Year Start Snapshot ─────────────────────────────────

    def save_year_start_snapshot(
        self,
        year: int,
        positions: list[dict],
    ) -> int:
        """
        Gem primo-snapshot af alle positioner.

        Kald dette 1. januar (eller den dato regnskabsåret starter).

        Args:
            year: Regnskabsår.
            positions: Liste af position-dicts med symbol, qty,
                      current_price, market_value_dkk, broker_source.

        Returns:
            Antal positioner gemt.
        """
        snapshot_date = datetime.now().isoformat()
        count = 0

        with sqlite3.connect(self._db_path) as conn:
            # Ryd tidligere snapshot for samme år
            conn.execute(
                "DELETE FROM year_start_snapshots WHERE year = ?",
                (year,),
            )

            for pos in positions:
                symbol = pos.get("symbol", "")
                qty = float(pos.get("qty", 0))
                price = float(pos.get("current_price", 0))
                value = float(pos.get("market_value_dkk", qty * price))
                broker = pos.get("broker_source", "")
                currency = pos.get("currency", "DKK")

                if qty > 0 and symbol:
                    conn.execute(
                        "INSERT OR REPLACE INTO year_start_snapshots "
                        "(year, symbol, qty, price_dkk, value_dkk, broker, "
                        "currency, snapshot_date) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (year, symbol, qty, price, value, broker,
                         currency, snapshot_date),
                    )
                    count += 1

        logger.info(
            f"[mtm] Primo-snapshot {year}: {count} positioner gemt"
        )
        return count

    def get_year_start_prices(self, year: int) -> dict[str, float]:
        """Hent primo-priser for et år (symbol → pris DKK)."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT symbol, price_dkk FROM year_start_snapshots "
                "WHERE year = ?",
                (year,),
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_year_start_snapshot(self, year: int) -> list[dict]:
        """Hent fuld primo-snapshot."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT symbol, qty, price_dkk, value_dkk, broker, currency "
                "FROM year_start_snapshots WHERE year = ?",
                (year,),
            ).fetchall()
        return [
            {
                "symbol": r[0], "qty": r[1], "price_dkk": r[2],
                "value_dkk": r[3], "broker": r[4], "currency": r[5],
            }
            for r in rows
        ]

    # ── MTM Calculation ─────────────────────────────────────

    def calculate_ytd(
        self,
        year: int,
        current_positions: list[dict],
    ) -> MTMSummary:
        """
        Beregn year-to-date mark-to-market.

        Daglig estimering af lagerbeskatning baseret på nuværende priser.
        """
        return self._calculate(year, current_positions, is_final=False)

    def year_end_calculation(
        self,
        year: int,
        final_positions: list[dict],
    ) -> MTMSummary:
        """
        Årsafslutning: Final mark-to-market beregning.

        Kald dette 31. december (eller regnskabsårets slutning).
        """
        result = self._calculate(year, final_positions, is_final=True)

        # Gem calculation
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO mtm_calculations "
                "(year, calculation_date, is_final, total_primo, "
                "total_ultimo, total_mtm_pnl, estimated_tax, position_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    year, result.snapshot_date, 1,
                    result.total_primo_dkk, result.total_ultimo_dkk,
                    result.total_mtm_pnl_dkk, result.estimated_tax_dkk,
                    result.position_count,
                ),
            )

        logger.info(
            f"[mtm] Årsafslutning {year}: "
            f"P&L={result.total_mtm_pnl_dkk:,.0f} DKK, "
            f"skat={result.estimated_tax_dkk:,.0f} DKK"
        )

        return result

    def _calculate(
        self,
        year: int,
        positions: list[dict],
        is_final: bool,
    ) -> MTMSummary:
        """Core MTM beregning."""
        primo_prices = self.get_year_start_prices(year)

        mtm_positions: list[MTMPosition] = []
        total_primo = 0.0
        total_ultimo = 0.0
        total_pnl = 0.0
        total_gains = 0.0
        total_losses = 0.0
        by_broker: dict[str, float] = {}
        by_currency: dict[str, float] = {}

        for pos in positions:
            symbol = pos.get("symbol", "")
            qty = float(pos.get("qty", 0))
            current_price = float(pos.get("current_price", 0))
            value_dkk = float(pos.get("market_value_dkk", qty * current_price))
            broker = pos.get("broker_source", "")
            currency = pos.get("currency", "DKK")

            # Primo: årets start eller anskaffelse
            is_new = symbol not in primo_prices
            if is_new:
                # Ny position: primo = anskaffelsessum
                primo_price = float(pos.get("entry_price", current_price))
                primo_value = primo_price * qty
            else:
                primo_price = primo_prices[symbol]
                primo_value = primo_price * qty

            # MTM P&L
            pnl = value_dkk - primo_value
            tax = pnl * self.TAX_RATE

            mtm_positions.append(MTMPosition(
                symbol=symbol,
                qty=qty,
                primo_price_dkk=primo_price,
                primo_value_dkk=primo_value,
                ultimo_price_dkk=current_price,
                ultimo_value_dkk=value_dkk,
                mtm_pnl_dkk=round(pnl, 2),
                tax_22pct_dkk=round(tax, 2),
                is_new_position=is_new,
                broker=broker,
                currency=currency,
            ))

            total_primo += primo_value
            total_ultimo += value_dkk
            total_pnl += pnl
            if pnl > 0:
                total_gains += pnl
            else:
                total_losses += pnl

            by_broker[broker] = by_broker.get(broker, 0) + pnl
            by_currency[currency] = by_currency.get(currency, 0) + pnl

        # Sortér efter absolut P&L (størst impact først)
        mtm_positions.sort(key=lambda p: abs(p.mtm_pnl_dkk), reverse=True)

        return MTMSummary(
            year=year,
            snapshot_date=datetime.now().isoformat(),
            is_final=is_final,
            positions=mtm_positions,
            position_count=len(mtm_positions),
            total_primo_dkk=round(total_primo, 2),
            total_ultimo_dkk=round(total_ultimo, 2),
            total_mtm_pnl_dkk=round(total_pnl, 2),
            total_gains_dkk=round(total_gains, 2),
            total_losses_dkk=round(total_losses, 2),
            estimated_tax_dkk=round(
                total_pnl * self.TAX_RATE, 2  # Kan være negativ (tilgodehavende ved tab)
            ),
            by_broker={k: round(v, 2) for k, v in by_broker.items()},
            by_currency={k: round(v, 2) for k, v in by_currency.items()},
        )

    # ── History ─────────────────────────────────────────────

    def get_calculation_history(self, year: int | None = None) -> list[dict]:
        """Hent historik over MTM-beregninger."""
        with sqlite3.connect(self._db_path) as conn:
            if year:
                rows = conn.execute(
                    "SELECT year, calculation_date, is_final, total_primo, "
                    "total_ultimo, total_mtm_pnl, estimated_tax, position_count "
                    "FROM mtm_calculations WHERE year = ? "
                    "ORDER BY calculation_date DESC",
                    (year,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT year, calculation_date, is_final, total_primo, "
                    "total_ultimo, total_mtm_pnl, estimated_tax, position_count "
                    "FROM mtm_calculations ORDER BY calculation_date DESC"
                ).fetchall()

        return [
            {
                "year": r[0], "date": r[1], "is_final": bool(r[2]),
                "primo": r[3], "ultimo": r[4], "pnl": r[5],
                "tax": r[6], "positions": r[7],
            }
            for r in rows
        ]
