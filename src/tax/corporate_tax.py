"""
Selskabsskat-beregning for dansk ApS/A/S.

REGLER:
  - Selskabsskat: 22% (selskabsskatteloven)
  - Lagerbeskatning: Skat på urealiserede gevinster ved regnskabsårets udgang
  - FIFO: First In, First Out for positionsmatching
  - Skattetilgodehavende: Fremførsel af underskud (modregnes i fremtidige gevinster)
  - Alle beregninger i DKK

VIGTIGT: Denne beregning er vejledende. Konsultér altid en revisor.

Beskatningstyper per instrument:
  - Danske/udenlandske aktier: Lagerbeskatning (ABL §9)
  - ETF'er (aktie-/obligationsbaserede): Lagerbeskatning (ABL §19 / KGL)
  - Obligationer: Lagerbeskatning (Kursgevinstloven)
  - Crypto: Lagerbeskatning (Statsskatteloven §4-6)
  - Forex: Lagerbeskatning (Kursgevinstloven)
  - Options/Futures: Lagerbeskatning (KGL §29-33)
  - Udbytter DK: 22% (modregnes i skat)
  - Udbytter udenlandske: 22% minus kildeskat-kredit (DBO)
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


# ── Constants ───────────────────────────────────────────────

_raw_tax_rate = float(os.environ.get("COMPANY_TAX_RATE", "0.22"))
CORPORATE_TAX_RATE = min(1.0, max(0.0, _raw_tax_rate))  # Validér 0-100%
TAX_YEAR_END = os.environ.get("TAX_YEAR_END", "12-31")
DEFAULT_TAX_CREDIT = float(os.environ.get("TAX_CREDIT_INITIAL", "0"))


# ── Dataklasser ─────────────────────────────────────────────

@dataclass
class PositionTaxInfo:
    """Skattemæssig info for én position."""
    symbol: str
    qty: float
    primo_value_dkk: float        # Værdi ved årets start (eller anskaffelse)
    ultimo_value_dkk: float       # Værdi ved årets slut (eller nuværende)
    unrealized_pnl_dkk: float     # Urealiseret gevinst/tab
    tax_type: str = "lager"       # "lager" eller "realisation"
    currency: str = "DKK"
    exchange: str = ""
    broker: str = ""
    primo_price: float = 0.0
    ultimo_price: float = 0.0
    fx_rate_primo: float = 1.0
    fx_rate_ultimo: float = 1.0

    @property
    def estimated_tax_dkk(self) -> float:
        """Estimeret skat på denne position (kan være negativ = fradrag)."""
        if self.tax_type == "lager":
            return self.unrealized_pnl_dkk * CORPORATE_TAX_RATE
        return 0.0  # Realisationsbeskattet: kun skat ved salg


@dataclass
class RealizedTrade:
    """En realiseret handel med skattemæssig beregning."""
    symbol: str
    qty: float
    acquisition_price_dkk: float   # Anskaffelsespris i DKK (FIFO)
    disposal_price_dkk: float      # Salgspris i DKK
    acquisition_date: str
    disposal_date: str
    gain_dkk: float                # Realiseret gevinst/tab
    fx_gain_dkk: float = 0.0      # Valutakursgevinst
    broker: str = ""
    order_id: str = ""


@dataclass
class DividendEntry:
    """Udbyttebetaling med skatteberegning."""
    symbol: str
    pay_date: str
    gross_amount: float            # Brutto i original valuta
    currency: str = "DKK"
    fx_rate: float = 1.0
    gross_dkk: float = 0.0        # Brutto i DKK
    withholding_tax: float = 0.0   # Kildeskat betalt
    withholding_pct: float = 0.0   # Kildeskat-sats
    net_dkk: float = 0.0          # Netto i DKK
    country: str = "DK"
    reclaimable_dkk: float = 0.0  # Reclaimable kildeskat


@dataclass
class TaxResult:
    """
    Samlet skatteberegning for et regnskabsår.

    DISCLAIMER: Denne beregning er vejledende. Konsultér din revisor.
    """
    year: int

    # Lagerbeskatning
    unrealized_pnl_dkk: float = 0.0        # Urealiserede gevinster/tab
    unrealized_gains_dkk: float = 0.0      # Kun gevinster
    unrealized_losses_dkk: float = 0.0     # Kun tab

    # Realiseret
    realized_pnl_dkk: float = 0.0          # Realiserede gevinster/tab
    realized_gains_dkk: float = 0.0
    realized_losses_dkk: float = 0.0

    # Udbytter
    dividend_income_dkk: float = 0.0
    withholding_tax_paid_dkk: float = 0.0  # Kildeskat betalt
    withholding_tax_credit_dkk: float = 0.0  # Kredit i DK skat

    # Valuta
    fx_pnl_dkk: float = 0.0

    # Omkostninger (fradragsberettigede)
    deductible_costs_dkk: float = 0.0      # Kurtage, platform fees, etc.

    # Samlet
    taxable_income_dkk: float = 0.0        # Skattepligtig indkomst
    gross_tax_dkk: float = 0.0             # Brutto skat (22%)
    tax_credit_used_dkk: float = 0.0       # Modregnet tilgodehavende
    net_tax_dkk: float = 0.0              # Netto skat at betale
    remaining_tax_credit_dkk: float = 0.0  # Resterende tilgodehavende

    # Detaljer
    positions: list[PositionTaxInfo] = field(default_factory=list)
    trades: list[RealizedTrade] = field(default_factory=list)
    dividends: list[DividendEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "unrealized_pnl": round(self.unrealized_pnl_dkk, 2),
            "realized_pnl": round(self.realized_pnl_dkk, 2),
            "dividend_income": round(self.dividend_income_dkk, 2),
            "fx_pnl": round(self.fx_pnl_dkk, 2),
            "deductible_costs": round(self.deductible_costs_dkk, 2),
            "taxable_income": round(self.taxable_income_dkk, 2),
            "gross_tax_22pct": round(self.gross_tax_dkk, 2),
            "tax_credit_used": round(self.tax_credit_used_dkk, 2),
            "net_tax": round(self.net_tax_dkk, 2),
            "remaining_tax_credit": round(self.remaining_tax_credit_dkk, 2),
            "position_count": len(self.positions),
            "trade_count": len(self.trades),
            "dividend_count": len(self.dividends),
            "disclaimer": (
                "Denne beregning er vejledende. Konsultér din revisor."
            ),
        }


@dataclass
class TaxImpact:
    """Skattemæssig konsekvens af at sælge en position."""
    symbol: str
    qty_to_sell: float
    current_price_dkk: float
    acquisition_price_dkk: float       # FIFO-vægtet
    realized_gain_dkk: float
    tax_impact_dkk: float              # Effekt på skat
    tax_credit_impact_dkk: float       # Effekt på tilgodehavende
    recommendation: str = ""
    wash_sale_warning: bool = False
    wash_sale_expires: str = ""


@dataclass
class TaxSuggestion:
    """Skatteoptimerings-forslag."""
    symbol: str
    action: str                        # "sell_loss", "defer_gain", "harvest_credit", "wash_sale_warning"
    description: str
    estimated_impact_dkk: float        # Skattemæssig besparelse (positiv = godt)
    urgency: str = "low"              # "high", "medium", "low"
    details: dict = field(default_factory=dict)


# ── FIFO Lot Tracker ───────────────────────────────────────

class FIFOTracker:
    """
    FIFO-baseret position tracking.

    SKAT kræver FIFO for aktiehandel.
    Tracker anskaffelsespris per lot for korrekt skatteberegning.
    """

    def __init__(self, db_path: str = "data_cache/fifo_lots.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    qty REAL NOT NULL,
                    remaining_qty REAL NOT NULL,
                    price_dkk REAL NOT NULL,
                    fx_rate REAL DEFAULT 1.0,
                    acquired_at TEXT NOT NULL,
                    broker TEXT DEFAULT '',
                    order_id TEXT DEFAULT ''
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_lots_symbol "
                "ON lots(symbol)"
            )

    def add_lot(
        self,
        symbol: str,
        qty: float,
        price_dkk: float,
        acquired_at: str,
        fx_rate: float = 1.0,
        broker: str = "",
        order_id: str = "",
    ) -> None:
        """Tilføj en ny FIFO-lot (køb)."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO lots "
                "(symbol, qty, remaining_qty, price_dkk, fx_rate, "
                "acquired_at, broker, order_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (symbol.upper(), qty, qty, price_dkk, fx_rate,
                 acquired_at, broker, order_id),
            )

    def consume_lots(
        self,
        symbol: str,
        qty: float,
        disposal_price_dkk: float,
        disposal_date: str,
    ) -> list[RealizedTrade]:
        """
        Forbrug FIFO-lots ved salg.

        Returns:
            Liste af RealizedTrade for hver lot der forbruges.
        """
        trades: list[RealizedTrade] = []
        remaining = qty

        with sqlite3.connect(self._db_path) as conn:
            # Hent lots i FIFO rækkefølge (ældste først)
            lots = conn.execute(
                "SELECT id, symbol, remaining_qty, price_dkk, acquired_at, "
                "broker, order_id, fx_rate "
                "FROM lots WHERE symbol = ? AND remaining_qty > 0 "
                "ORDER BY acquired_at ASC, id ASC",
                (symbol.upper(),),
            ).fetchall()

            for lot in lots:
                if remaining <= 0:
                    break

                lot_id, sym, lot_qty, acq_price, acq_date, broker, oid, fx = lot
                consume = min(remaining, lot_qty)

                gain = (disposal_price_dkk - acq_price) * consume

                trades.append(RealizedTrade(
                    symbol=sym,
                    qty=consume,
                    acquisition_price_dkk=acq_price,
                    disposal_price_dkk=disposal_price_dkk,
                    acquisition_date=acq_date,
                    disposal_date=disposal_date,
                    gain_dkk=gain,
                    broker=broker,
                    order_id=oid,
                ))

                new_remaining = lot_qty - consume
                conn.execute(
                    "UPDATE lots SET remaining_qty = ? WHERE id = ?",
                    (new_remaining, lot_id),
                )

                remaining -= consume

            if remaining > 0:
                logger.warning(
                    f"[tax] FIFO: {remaining:.2f} enheder af {symbol} "
                    f"uden matchende lots"
                )

        return trades

    def get_lots(self, symbol: str) -> list[dict]:
        """Hent alle åbne lots for et symbol."""
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT symbol, remaining_qty, price_dkk, acquired_at, broker "
                "FROM lots WHERE symbol = ? AND remaining_qty > 0 "
                "ORDER BY acquired_at ASC",
                (symbol.upper(),),
            ).fetchall()

        return [
            {
                "symbol": r[0],
                "qty": r[1],
                "price_dkk": r[2],
                "acquired_at": r[3],
                "broker": r[4],
            }
            for r in rows
        ]

    def get_weighted_avg_price(self, symbol: str) -> float:
        """FIFO-vægtet gennemsnitlig anskaffelsespris."""
        lots = self.get_lots(symbol)
        total_qty = sum(l["qty"] for l in lots)
        if total_qty == 0:
            return 0.0
        total_value = sum(l["qty"] * l["price_dkk"] for l in lots)
        return total_value / total_qty


# ── Corporate Tax Calculator ───────────────────────────────

class CorporateTaxCalculator:
    """
    Selskabsskat-beregning for dansk ApS/A/S.

    Brug:
        calc = CorporateTaxCalculator(tax_credit=500_000)

        # Beregn urealiseret P&L (lagerbeskatning)
        positions = [...]  # fra AggregatedPortfolio
        unrealized = calc.calculate_unrealized_pnl(positions, year_end_prices)

        # Beregn realiseret P&L
        realized = calc.calculate_realized_pnl(trades, year=2025)

        # Fuld årsberegning
        result = calc.calculate_annual_tax(year=2025, positions, trades, dividends)

        # Simulér salg
        impact = calc.simulate_sale("AAPL", qty=50)

        # Optimeringsforslag
        suggestions = calc.suggest_tax_optimization(positions)
    """

    def __init__(
        self,
        tax_credit: float = DEFAULT_TAX_CREDIT,
        tax_rate: float = CORPORATE_TAX_RATE,
        fifo_tracker: FIFOTracker | None = None,
    ) -> None:
        self.tax_rate = tax_rate
        self.tax_credit = tax_credit
        self.fifo = fifo_tracker or FIFOTracker()

    # ── Urealiseret P&L (Lagerbeskatning) ───────────────────

    def calculate_unrealized_pnl(
        self,
        positions: list[dict],
        year_start_prices: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Beregn urealiseret P&L for lagerbeskatning.

        Args:
            positions: Liste af position-dicts med symbol, qty, entry_price,
                      current_price, currency, market_value_dkk.
            year_start_prices: Symbol → pris ved årets start (DKK).
                              Hvis None bruges entry_price (nye positioner).

        Returns:
            Dict med per-position og total urealiseret P&L.
        """
        year_start = year_start_prices or {}
        results: list[PositionTaxInfo] = []
        total_unrealized = 0.0
        total_gains = 0.0
        total_losses = 0.0

        for pos in positions:
            symbol = pos.get("symbol", "")
            qty = float(pos.get("qty", 0))
            current_dkk = float(pos.get("market_value_dkk", 0))

            # Primo: årets start-pris eller anskaffelsespris
            if symbol in year_start:
                primo_price = year_start[symbol]
                primo_value = primo_price * qty
            else:
                # Brug FIFO-vægtet anskaffelsespris
                fifo_price = self.fifo.get_weighted_avg_price(symbol)
                primo_price = fifo_price if fifo_price > 0 else float(
                    pos.get("entry_price", 0)
                )
                primo_value = primo_price * qty

            unrealized = current_dkk - primo_value

            info = PositionTaxInfo(
                symbol=symbol,
                qty=qty,
                primo_value_dkk=primo_value,
                ultimo_value_dkk=current_dkk,
                unrealized_pnl_dkk=unrealized,
                currency=pos.get("currency", "DKK"),
                broker=pos.get("broker_source", ""),
                primo_price=primo_price,
                ultimo_price=float(pos.get("current_price", 0)),
            )
            results.append(info)

            total_unrealized += unrealized
            if unrealized > 0:
                total_gains += unrealized
            else:
                total_losses += unrealized

        return {
            "positions": results,
            "total_unrealized_dkk": round(total_unrealized, 2),
            "total_gains_dkk": round(total_gains, 2),
            "total_losses_dkk": round(total_losses, 2),
            "estimated_tax_dkk": round(total_unrealized * self.tax_rate, 2),
            "position_count": len(results),
        }

    # ── Realiseret P&L ─────────────────────────────────────

    def calculate_realized_pnl(
        self,
        trades: list[RealizedTrade],
        year: int | None = None,
    ) -> dict[str, Any]:
        """
        Beregn realiserede gevinster/tab.

        Args:
            trades: Liste af RealizedTrade fra FIFO tracker.
            year: Filtrer på år (None = alle).

        Returns:
            Dict med total realiseret P&L og detaljer.
        """
        filtered = trades
        if year:
            filtered = [
                t for t in trades
                if t.disposal_date.startswith(str(year))
            ]

        total_gains = sum(t.gain_dkk for t in filtered if t.gain_dkk > 0)
        total_losses = sum(t.gain_dkk for t in filtered if t.gain_dkk < 0)
        total_pnl = total_gains + total_losses

        return {
            "trades": filtered,
            "total_realized_dkk": round(total_pnl, 2),
            "total_gains_dkk": round(total_gains, 2),
            "total_losses_dkk": round(total_losses, 2),
            "trade_count": len(filtered),
        }

    # ── Fuld Årsberegning ───────────────────────────────────

    def calculate_annual_tax(
        self,
        year: int,
        unrealized_pnl: float = 0.0,
        realized_pnl: float = 0.0,
        dividend_income: float = 0.0,
        withholding_tax_credit: float = 0.0,
        fx_pnl: float = 0.0,
        deductible_costs: float = 0.0,
        positions: list[PositionTaxInfo] | None = None,
        commit: bool = True,
        trades: list[RealizedTrade] | None = None,
        dividends: list[DividendEntry] | None = None,
    ) -> TaxResult:
        """
        Beregn selskabsskat for et regnskabsår.

        Formel:
          Skattepligtig indkomst = (urealiseret P&L + realiseret P&L
                                    + udbytter + FX P&L - omkostninger)
          Brutto skat = indkomst × 22%
          Netto skat = max(0, brutto skat - skattetilgodehavende)
        """
        # Samlet skattepligtig indkomst
        taxable = (
            unrealized_pnl
            + realized_pnl
            + dividend_income
            + fx_pnl
            - deductible_costs
        )

        # Brutto skat
        gross_tax = taxable * self.tax_rate if taxable > 0 else 0.0

        # Kildeskat-kredit reducerer brutto skat
        gross_tax = max(0, gross_tax - withholding_tax_credit)

        # Skattetilgodehavende modregning (brug lokal kopi for at undgå sideeffekter)
        working_credit = self.tax_credit
        credit_used = 0.0
        if gross_tax > 0 and working_credit > 0:
            credit_used = min(gross_tax, working_credit)
            working_credit -= credit_used

        # Hvis underskud → tilføj til tilgodehavende
        loss_credit = 0.0
        if taxable < 0:
            loss_credit = abs(taxable) * self.tax_rate
            working_credit += loss_credit
            logger.info(
                f"[tax] Underskud {taxable:,.0f} DKK → "
                f"tilgodehavende +{loss_credit:,.0f} DKK"
            )

        # Opdater self.tax_credit kun hvis commit=True (undgår sideeffekter ved simulering)
        if commit:
            self.tax_credit = working_credit

        net_tax = max(0, gross_tax - credit_used)

        result = TaxResult(
            year=year,
            unrealized_pnl_dkk=unrealized_pnl,
            unrealized_gains_dkk=max(0, unrealized_pnl),
            unrealized_losses_dkk=min(0, unrealized_pnl),
            realized_pnl_dkk=realized_pnl,
            realized_gains_dkk=max(0, realized_pnl),
            realized_losses_dkk=min(0, realized_pnl),
            dividend_income_dkk=dividend_income,
            withholding_tax_paid_dkk=withholding_tax_credit,
            withholding_tax_credit_dkk=withholding_tax_credit,
            fx_pnl_dkk=fx_pnl,
            deductible_costs_dkk=deductible_costs,
            taxable_income_dkk=round(taxable, 2),
            gross_tax_dkk=round(gross_tax + credit_used, 2),
            tax_credit_used_dkk=round(credit_used, 2),
            net_tax_dkk=round(net_tax, 2),
            remaining_tax_credit_dkk=round(working_credit, 2),
            positions=positions or [],
            trades=trades or [],
            dividends=dividends or [],
        )

        logger.info(
            f"[tax] Årsberegning {year}: "
            f"indkomst={taxable:,.0f} DKK, "
            f"skat={net_tax:,.0f} DKK, "
            f"tilgodehavende={self.tax_credit:,.0f} DKK"
        )

        return result

    # ── Simulér Salg ────────────────────────────────────────

    def simulate_sale(
        self,
        symbol: str,
        qty: float,
        current_price_dkk: float,
    ) -> TaxImpact:
        """
        Simulér skattemæssig konsekvens af at sælge en position.

        "Hvad sker der med min skat hvis jeg sælger X?"
        """
        # Hent FIFO-lots
        fifo_price = self.fifo.get_weighted_avg_price(symbol)
        if fifo_price == 0:
            fifo_price = current_price_dkk  # Ingen lots → ingen gevinst

        realized_gain = (current_price_dkk - fifo_price) * qty
        # Vis fuld skatteeffekt — negativ ved tab (tilgodehavende)
        tax_impact = realized_gain * self.tax_rate

        # Credit impact
        credit_impact = 0.0
        if realized_gain > 0 and self.tax_credit > 0:
            credit_impact = min(tax_impact, self.tax_credit)
        elif realized_gain < 0:
            credit_impact = abs(realized_gain) * self.tax_rate  # Tilføjer til credit

        # Recommendation
        if realized_gain < 0 and self.tax_credit < abs(realized_gain) * self.tax_rate:
            rec = "Overvej at sælge for at realisere tab og øge skattetilgodehavende"
        elif realized_gain > 0 and self.tax_credit >= tax_impact:
            rec = "Skattetilgodehavende dækker gevinsten — skattefrit salg"
        elif realized_gain > 0:
            rec = (
                f"Salget udløser {tax_impact:,.0f} DKK i skat "
                f"(tilgodehavende dækker {credit_impact:,.0f} DKK)"
            )
        else:
            rec = "Neutral skattemæssig effekt"

        return TaxImpact(
            symbol=symbol,
            qty_to_sell=qty,
            current_price_dkk=current_price_dkk,
            acquisition_price_dkk=fifo_price,
            realized_gain_dkk=round(realized_gain, 2),
            tax_impact_dkk=round(tax_impact, 2),
            tax_credit_impact_dkk=round(credit_impact, 2),
            recommendation=rec,
        )

    # ── Skatteoptimering ────────────────────────────────────

    def suggest_tax_optimization(
        self,
        positions: list[dict],
        recent_sales: list[dict] | None = None,
    ) -> list[TaxSuggestion]:
        """
        Analyser porteføljen for skatteoptimerings-muligheder.

        Forslag:
          a. Tab-realisering for at øge tilgodehavende
          b. Gevinst-realisering mens tilgodehavende dækker
          c. Wash sale warnings
          d. Year-end planning
        """
        suggestions: list[TaxSuggestion] = []
        recent_symbols = set()

        # Wash sale check: symboler solgt inden for 30 dage
        if recent_sales:
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=30)).isoformat()
            for sale in recent_sales:
                if sale.get("date", "") >= cutoff and sale.get("gain_dkk", 0) < 0:
                    recent_symbols.add(sale.get("symbol", "").upper())

        for pos in positions:
            symbol = pos.get("symbol", "")
            unrealized = float(pos.get("unrealized_pnl_dkk", 0))
            market_value = float(pos.get("market_value_dkk", 0))

            # a. Tab-realisering
            if unrealized < -1000:  # Min 1000 DKK tab
                tax_saving = abs(unrealized) * self.tax_rate
                suggestions.append(TaxSuggestion(
                    symbol=symbol,
                    action="sell_loss",
                    description=(
                        f"Sælg {symbol} for at realisere tab på "
                        f"{unrealized:,.0f} DKK og øg tilgodehavende "
                        f"med {tax_saving:,.0f} DKK"
                    ),
                    estimated_impact_dkk=round(tax_saving, 2),
                    urgency="medium",
                    details={
                        "unrealized_loss": unrealized,
                        "tax_saving": tax_saving,
                    },
                ))

            # b. Gevinst-realisering med tilgodehavende
            if unrealized > 5000 and self.tax_credit > 0:
                tax_on_gain = unrealized * self.tax_rate
                covered = min(tax_on_gain, self.tax_credit)
                if covered >= tax_on_gain * 0.8:  # 80%+ dækket
                    suggestions.append(TaxSuggestion(
                        symbol=symbol,
                        action="harvest_credit",
                        description=(
                            f"Realisér gevinst på {symbol} ({unrealized:,.0f} DKK) "
                            f"— tilgodehavende dækker {covered / tax_on_gain * 100:.0f}% "
                            f"af skatten"
                        ),
                        estimated_impact_dkk=round(covered, 2),
                        urgency="low",
                        details={
                            "unrealized_gain": unrealized,
                            "tax_covered": covered,
                        },
                    ))

            # c. Wash sale warning
            if symbol.upper() in recent_symbols:
                suggestions.append(TaxSuggestion(
                    symbol=symbol,
                    action="wash_sale_warning",
                    description=(
                        f"ADVARSEL: {symbol} blev solgt med tab inden for "
                        f"30 dage. Genkøb kan påvirke tab-fradraget."
                    ),
                    estimated_impact_dkk=0,
                    urgency="high",
                ))

        # Sortér: højeste impact først, derefter urgency
        urgency_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(
            key=lambda s: (urgency_order.get(s.urgency, 3), -s.estimated_impact_dkk)
        )

        return suggestions

    # ── Year-to-Date Estimat ────────────────────────────────

    def ytd_estimated_tax(
        self,
        positions: list[dict],
    ) -> dict[str, Any]:
        """
        Estimeret skat ved årsskifte baseret på nuværende priser.

        Dashboard widget.
        """
        unrealized = self.calculate_unrealized_pnl(positions)
        total_unrealized = unrealized["total_unrealized_dkk"]
        estimated_tax = total_unrealized * self.tax_rate if total_unrealized > 0 else 0
        net_after_credit = max(0, estimated_tax - self.tax_credit)

        return {
            "ytd_unrealized_pnl": round(total_unrealized, 2),
            "estimated_gross_tax": round(estimated_tax, 2),
            "tax_credit_available": round(self.tax_credit, 2),
            "estimated_net_tax": round(net_after_credit, 2),
            "fully_covered_by_credit": net_after_credit == 0,
            "disclaimer": (
                "Estimat baseret på nuværende priser. "
                "Faktisk skat beregnes ved regnskabsårets udgang."
            ),
        }
