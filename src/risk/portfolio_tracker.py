"""
PortfolioTracker – holder styr på positioner, P&L og portfolio-metrics.

v2: Tilføjet SQLite-persistens.
    - Alle positioner gemmes til DB ved open/close
    - Closed trades logges permanent
    - Cash balance og equity history overlever genstart
    - Ved startup: load fra DB i stedet for at nulstille til $100K
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger


# ── DB path ──────────────────────────────────────────────────

_DEFAULT_DB_PATH = "data_cache/paper_portfolio.db"


@dataclass
class Position:
    """En åben position i porteføljen."""
    symbol: str
    side: str                  # "long" eller "short"
    qty: float
    entry_price: float
    entry_time: str
    current_price: float = 0.0
    peak_price: float = 0.0    # højeste pris siden entry (til trailing stop for longs)
    trough_price: float = 0.0  # laveste pris siden entry (til trailing stop for shorts)

    def __post_init__(self) -> None:
        if self.current_price == 0.0:
            self.current_price = self.entry_price
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price
        if self.trough_price == 0.0:
            self.trough_price = self.entry_price

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.qty * self.entry_price

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.qty
        return (self.entry_price - self.current_price) * self.qty

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return self.unrealized_pnl / self.cost_basis

    @property
    def pct_from_peak(self) -> float:
        """Adverse move from best price (til trailing stop).

        For long positions: percentage drop from highest price (peak).
        For short positions: percentage rise from lowest price (trough).
        """
        if self.side == "short":
            if self.trough_price == 0:
                return 0.0
            return (self.current_price - self.trough_price) / self.trough_price
        else:
            if self.peak_price == 0:
                return 0.0
            return (self.peak_price - self.current_price) / self.peak_price

    def update_price(self, price: float) -> None:
        self.current_price = price
        if price > self.peak_price:
            self.peak_price = price
        if price < self.trough_price:
            self.trough_price = price


@dataclass
class ClosedTrade:
    """En afsluttet handel med realiseret P&L."""
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    exit_reason: str           # "stop_loss", "take_profit", "trailing_stop", "signal", "manual"

    @property
    def realized_pnl(self) -> float:
        if self.side == "long":
            return (self.exit_price - self.entry_price) * self.qty
        return (self.entry_price - self.exit_price) * self.qty

    @property
    def realized_pnl_pct(self) -> float:
        cost = self.entry_price * self.qty
        if cost == 0:
            return 0.0
        return self.realized_pnl / cost


# ── SQLite Persistence Layer ─────────────────────────────────

class PortfolioDB:
    """SQLite persistence for portfolio state."""

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_tables(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS portfolio_state (
                    key TEXT PRIMARY KEY,
                    value REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS open_positions (
                    symbol TEXT PRIMARY KEY,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    peak_price REAL NOT NULL,
                    trough_price REAL NOT NULL DEFAULT 0.0
                );
                CREATE TABLE IF NOT EXISTS closed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    exit_reason TEXT NOT NULL,
                    realized_pnl REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equity REAL NOT NULL,
                    timestamp TEXT NOT NULL
                );
            """)
            # Migration: add trough_price column if missing (existing DBs)
            try:
                conn.execute("ALTER TABLE open_positions ADD COLUMN trough_price REAL NOT NULL DEFAULT 0.0")
            except sqlite3.OperationalError:
                pass  # Column already exists

    # ── State (cash, peak, daily start) ───────────────────────

    def save_state(self, cash: float, peak_equity: float,
                   daily_start: float, initial_capital: float):
        now = datetime.now().isoformat()
        with self._conn() as conn:
            for key, val in [("cash", cash), ("peak_equity", peak_equity),
                             ("daily_start_equity", daily_start),
                             ("initial_capital", initial_capital)]:
                conn.execute(
                    "INSERT OR REPLACE INTO portfolio_state (key, value, updated_at) "
                    "VALUES (?, ?, ?)", (key, val, now)
                )

    def load_state(self) -> dict | None:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM portfolio_state").fetchall()
        if not rows:
            return None
        return {row[0]: row[1] for row in rows}

    # ── Positions ─────────────────────────────────────────────

    def save_position(self, pos: Position):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO open_positions "
                "(symbol, side, qty, entry_price, entry_time, current_price, peak_price, trough_price) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (pos.symbol, pos.side, pos.qty, pos.entry_price,
                 pos.entry_time, pos.current_price, pos.peak_price, pos.trough_price),
            )

    def delete_position(self, symbol: str):
        with self._conn() as conn:
            conn.execute("DELETE FROM open_positions WHERE symbol = ?", (symbol,))

    def load_positions(self) -> dict[str, Position]:
        positions = {}
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM open_positions").fetchall()
        for row in rows:
            pos = Position(
                symbol=row[0], side=row[1], qty=row[2],
                entry_price=row[3], entry_time=row[4],
                current_price=row[5], peak_price=row[6],
                trough_price=row[7] if len(row) > 7 else row[3],
            )
            positions[row[0]] = pos
        return positions

    # ── Closed Trades ─────────────────────────────────────────

    def save_trade(self, trade: ClosedTrade):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO closed_trades "
                "(symbol, side, qty, entry_price, exit_price, "
                "entry_time, exit_time, exit_reason, realized_pnl) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (trade.symbol, trade.side, trade.qty, trade.entry_price,
                 trade.exit_price, trade.entry_time, trade.exit_time,
                 trade.exit_reason, trade.realized_pnl),
            )

    def load_trades(self) -> list[ClosedTrade]:
        trades = []
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT symbol, side, qty, entry_price, exit_price, "
                "entry_time, exit_time, exit_reason FROM closed_trades "
                "ORDER BY id"
            ).fetchall()
        for row in rows:
            trades.append(ClosedTrade(
                symbol=row[0], side=row[1], qty=row[2],
                entry_price=row[3], exit_price=row[4],
                entry_time=row[5], exit_time=row[6], exit_reason=row[7],
            ))
        return trades

    def trade_count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM closed_trades").fetchone()
        return row[0] if row else 0

    # ── Equity History ────────────────────────────────────────

    def save_equity(self, equity: float):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO equity_history (equity, timestamp) VALUES (?, ?)",
                (equity, datetime.now().isoformat()),
            )

    def load_equity_history(self) -> list[float]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT equity FROM equity_history ORDER BY id"
            ).fetchall()
        return [row[0] for row in rows]

    # ── Reset ─────────────────────────────────────────────────

    def reset_all(self):
        """Nulstil hele databasen (til test)."""
        with self._conn() as conn:
            conn.executescript("""
                DELETE FROM portfolio_state;
                DELETE FROM open_positions;
                DELETE FROM closed_trades;
                DELETE FROM equity_history;
            """)
        logger.info("[db] Portfolio database nulstillet")


class PortfolioTracker:
    """
    Tracker for portefølje-positioner og historiske trades.

    Beregner real-time P&L, Sharpe ratio, max drawdown og win rate.
    Persisterer alt til SQLite så data overlever genstart.
    """

    def __init__(self, initial_capital: float = 100_000,
                 db_path: str = _DEFAULT_DB_PATH) -> None:
        self._db = PortfolioDB(db_path)

        # Prøv at loade fra DB
        saved = self._db.load_state()
        if saved and "cash" in saved:
            self.initial_capital = saved.get("initial_capital", initial_capital)
            self.cash = saved["cash"]
            self.positions = self._db.load_positions()
            self.closed_trades = self._db.load_trades()
            self._equity_history = self._db.load_equity_history() or [self.initial_capital]
            self._peak_equity = saved.get("peak_equity", self.initial_capital)
            self._daily_start_equity = saved.get("daily_start_equity", self.initial_capital)

            logger.info(
                f"[portfolio] Loaded from DB: cash=${self.cash:,.2f}, "
                f"{len(self.positions)} positions, "
                f"{len(self.closed_trades)} closed trades"
            )
        else:
            self.initial_capital = initial_capital
            self.cash = initial_capital
            self.positions: dict[str, Position] = {}
            self.closed_trades: list[ClosedTrade] = []
            self._equity_history: list[float] = [initial_capital]
            self._peak_equity: float = initial_capital
            self._daily_start_equity: float = initial_capital
            logger.info(f"[portfolio] Fresh start: ${initial_capital:,.2f}")

    def _save_state(self) -> None:
        """Persist current state to DB."""
        self._db.save_state(
            self.cash, self._peak_equity,
            self._daily_start_equity, self.initial_capital,
        )

    # ── Position management ──────────────────────────────────

    def open_position(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        timestamp: str | None = None,
    ) -> Position:
        """Åbn en ny position (long eller short).

        For long: cash reduceres med cost (vi køber aktier).
        For short: cash øges med proceeds (vi sælger lånte aktier).
        """
        if symbol in self.positions:
            raise ValueError(f"Position i {symbol} eksisterer allerede")

        cost = qty * price

        if side == "short":
            # Short: vi modtager proceeds, men reserverer margin (150% af cost)
            margin_requirement = cost * 1.5
            if margin_requirement > self.cash + cost:
                raise ValueError(
                    f"Ikke nok margin til short: kræver ${margin_requirement:,.2f}, "
                    f"har ${self.cash:,.2f} + ${cost:,.2f} proceeds"
                )
            self.cash += cost
            self.cash -= margin_requirement  # Reserver margin
        else:
            # Long: vi betaler for aktierne
            if cost > self.cash:
                raise ValueError(
                    f"Ikke nok kontanter: kræver ${cost:,.2f}, har ${self.cash:,.2f}"
                )
            self.cash -= cost

        ts = timestamp or datetime.now().isoformat()
        pos = Position(
            symbol=symbol, side=side, qty=qty,
            entry_price=price, entry_time=ts,
        )
        self.positions[symbol] = pos

        # ── Persist ──
        self._db.save_position(pos)
        self._save_state()

        logger.info(
            f"Åbnet {side} {qty} {symbol} @ ${price:.2f} "
            f"(${cost:,.2f}, cash=${self.cash:,.2f})"
        )
        return pos

    def close_position(
        self,
        symbol: str,
        price: float,
        reason: str = "manual",
        timestamp: str | None = None,
        qty: float | None = None,
    ) -> ClosedTrade:
        """Luk en eksisterende position (long eller short).

        Args:
            qty: Number of shares to sell. If None, sells the entire position.
                 If less than the position size, performs a partial close.

        For long: vi sælger aktier → cash += proceeds.
        For short: vi køber aktier tilbage → cash -= cost.
        """
        if symbol not in self.positions:
            raise ValueError(f"Ingen position i {symbol}")

        pos = self.positions[symbol]
        ts = timestamp or datetime.now().isoformat()

        sell_qty = qty if qty is not None else pos.qty
        if sell_qty > pos.qty:
            sell_qty = pos.qty  # cap at position size

        partial = sell_qty < pos.qty
        proceeds = sell_qty * price

        if pos.side == "short":
            # Frigiv margin-reservation (150% af entry cost) og betal buy-back
            margin_reserved = sell_qty * pos.entry_price * 1.5
            self.cash += margin_reserved  # Frigiv margin
            self.cash -= proceeds         # Betal buy-back cost
        else:
            self.cash += proceeds

        trade = ClosedTrade(
            symbol=pos.symbol, side=pos.side, qty=sell_qty,
            entry_price=pos.entry_price, exit_price=price,
            entry_time=pos.entry_time, exit_time=ts,
            exit_reason=reason,
        )
        self.closed_trades.append(trade)

        if partial:
            # Reduce position size, keep the rest
            pos.qty -= sell_qty
            self._db.save_position(pos)
        else:
            # Full close — remove position entirely
            self.positions.pop(symbol)
            self._db.delete_position(symbol)

        self._db.save_trade(trade)
        self._save_state()

        action = "Delvis lukket" if partial else "Lukket"
        logger.info(
            f"{action} {pos.side} {sell_qty} {symbol} @ ${price:.2f} "
            f"({reason}, P&L=${trade.realized_pnl:+,.2f} / "
            f"{trade.realized_pnl_pct:+.2%})"
            + (f" — {pos.qty} remaining" if partial else "")
        )
        return trade

    def update_prices(self, prices: dict[str, float]) -> None:
        """Opdatér markedspriser for alle åbne positioner."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

        # Track equity
        equity = self.total_equity
        self._equity_history.append(equity)
        if len(self._equity_history) > 5000:
            self._equity_history = self._equity_history[-5000:]
        if equity > self._peak_equity:
            self._peak_equity = equity

        # ── Persist equity snapshot (throttled: max once per call) ──
        self._db.save_equity(equity)

    # ── Portfolio metrics ────────────────────────────────────

    @property
    def _short_margin_reserved(self) -> float:
        """Total margin reserveret for åbne short-positioner."""
        return sum(
            p.qty * p.entry_price * 1.5
            for p in self.positions.values() if p.side == "short"
        )

    @property
    def available_cash(self) -> float:
        """Kontanter tilgængelige for nye handler (ekskl. margin-reservationer)."""
        return self.cash

    @property
    def total_equity(self) -> float:
        """Samlet porteføljeværdi: kontanter + long markedsværdi + short P&L.

        Long: vi ejer aktier → market_value er et aktiv.
        Short: margin er reserveret i cash, P&L afspejles via unrealized_pnl.
        Cash er allerede reduceret med margin ved short-åbning.
        """
        long_val = sum(p.market_value for p in self.positions.values() if p.side == "long")
        short_pnl = sum(p.unrealized_pnl for p in self.positions.values() if p.side == "short")
        # Cash inkluderer margin-reservationer. Long values er aktiver.
        # Short P&L justerer for urealiseret gevinst/tab på shorts.
        return self.cash + long_val + short_pnl

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_realized_pnl(self) -> float:
        return sum(t.realized_pnl for t in self.closed_trades)

    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    @property
    def daily_pnl(self) -> float:
        """P&L siden start af dagen."""
        return self.total_equity - self._daily_start_equity

    @property
    def daily_pnl_pct(self) -> float:
        if self._daily_start_equity == 0:
            return 0.0
        return self.daily_pnl / self._daily_start_equity

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.total_equity - self.initial_capital) / self.initial_capital

    @property
    def current_drawdown_pct(self) -> float:
        """Nuværende drawdown fra peak equity."""
        if self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - self.total_equity) / self._peak_equity

    @property
    def max_drawdown_pct(self) -> float:
        """Største drawdown i porteføljens levetid."""
        if len(self._equity_history) < 2:
            return 0.0

        peak = self._equity_history[0]
        max_dd = 0.0
        for equity in self._equity_history:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @property
    def win_rate(self) -> float:
        """Andel af profitable trades (0.0–1.0)."""
        if not self.closed_trades:
            return 0.0
        wins = sum(1 for t in self.closed_trades if t.realized_pnl > 0)
        return wins / len(self.closed_trades)

    @property
    def profit_factor(self) -> float:
        """Sum af gevinster / sum af tab. > 1.0 er profitabelt."""
        gains = sum(t.realized_pnl for t in self.closed_trades if t.realized_pnl > 0)
        losses = abs(sum(t.realized_pnl for t in self.closed_trades if t.realized_pnl < 0))
        if losses == 0:
            return float("inf") if gains > 0 else 0.0
        return gains / losses

    @property
    def sharpe_ratio(self) -> float:
        """
        Annualiseret Sharpe ratio baseret på daglige equity-ændringer.

        Forudsætter 252 handelsdage per år og risikofri rente = 0.
        """
        if len(self._equity_history) < 3:
            return 0.0

        eq_arr = np.array(self._equity_history[:-1])
        if np.any(eq_arr <= 0):
            return 0.0  # Undgå division by zero ved wipeout
        returns = np.diff(self._equity_history) / eq_arr
        std = np.std(returns, ddof=1)  # Sample std (ddof=1) for korrekt Sharpe
        if len(returns) < 2 or std == 0:
            return 0.0

        return float(np.mean(returns) / std * np.sqrt(252))

    def start_new_day(self) -> None:
        """Markér start af ny handelsdag (nulstil daglig P&L)."""
        self._daily_start_equity = self.total_equity
        self._save_state()
        logger.debug(f"Ny dag: equity=${self.total_equity:,.2f}")

    def get_position_weight(self, symbol: str) -> float:
        """Position som andel af total equity."""
        equity = self.total_equity
        if equity == 0 or symbol not in self.positions:
            return 0.0
        pos = self.positions[symbol]
        # Shorts er liabilities — brug abs for vægt, men behold fortegn
        val = pos.market_value if pos.side == "long" else -pos.market_value
        return val / equity

    def summary(self) -> dict:
        """Returnér en dict med alle portfolio-metrics."""
        return {
            "total_equity": self.total_equity,
            "cash": self.cash,
            "positions": self.open_position_count,
            "unrealized_pnl": self.total_unrealized_pnl,
            "realized_pnl": self.total_realized_pnl,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl_pct,
            "total_return_pct": self.total_return_pct,
            "current_drawdown_pct": self.current_drawdown_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "closed_trades": len(self.closed_trades),
        }
