"""
Performance Report PDF Generator
=================================
Generates a comprehensive PDF performance report including:
  - Portfolio summary (value, P&L, positions)
  - Performance vs benchmarks (S&P 500, MSCI World)
  - Trade statistics (win rate, Sharpe, drawdown)
  - Position breakdown with per-stock analysis
  - Market comparison over session period
"""

from __future__ import annotations

import io
import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fpdf import FPDF
from loguru import logger


# ── Paths ────────────────────────────────────────────────────

_PORTFOLIO_DB = Path("data_cache/paper_portfolio.db")
_TRADER_DB = Path("data_cache/auto_trader_log.db")
_SIGNAL_DB = Path("data_cache/signal_log.db")
_EXCHANGE_SL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "exchange_stop_loss.json"

_EXCHANGE_SHORT = {
    "crypto": "Crypto", "new_zealand": "NZ", "australia": "ASX",
    "japan": "TSE", "hong_kong": "HK", "india": "NSE",
    "denmark": "DK", "sweden": "SE", "norway": "NO", "finland": "FI",
    "germany": "DE", "france": "FR", "netherlands": "NL",
    "switzerland": "CH", "spain": "ES", "italy": "IT",
    "london": "LSE", "us_stocks": "US", "chicago": "CME", "etfs": "ETFs",
}


def _stop_loss_header_text() -> str:
    """Build compact stop-loss summary for the report header."""
    try:
        import json
        if _EXCHANGE_SL_PATH.exists():
            sl_map = json.loads(_EXCHANGE_SL_PATH.read_text())
            if sl_map:
                parts = [f"{_EXCHANGE_SHORT.get(k, k)} {v:.1f}%"
                         for k, v in sorted(sl_map.items())]
                return "Stop-loss: " + ", ".join(parts)
    except Exception:
        pass
    return ""


# ── Data Fetchers ────────────────────────────────────────────

def _load_equity_history() -> list[dict]:
    """Load equity snapshots from portfolio DB."""
    if not _PORTFOLIO_DB.exists():
        return []
    with sqlite3.connect(_PORTFOLIO_DB) as conn:
        rows = conn.execute(
            "SELECT equity, timestamp FROM equity_history ORDER BY id"
        ).fetchall()
    return [{"equity": r[0], "timestamp": r[1]} for r in rows]


def _load_closed_trades() -> list[dict]:
    """Load closed trades from portfolio DB."""
    if not _PORTFOLIO_DB.exists():
        return []
    with sqlite3.connect(_PORTFOLIO_DB) as conn:
        rows = conn.execute(
            "SELECT symbol, side, qty, entry_price, exit_price, "
            "entry_time, exit_time, exit_reason FROM closed_trades "
            "ORDER BY exit_time DESC"
        ).fetchall()
    cols = ["symbol", "side", "qty", "entry_price", "exit_price",
            "entry_time", "exit_time", "exit_reason"]
    return [dict(zip(cols, r)) for r in rows]


def _load_open_positions() -> list[dict]:
    """Load open positions from portfolio DB, then refresh current_price via yfinance."""
    if not _PORTFOLIO_DB.exists():
        return []
    with sqlite3.connect(_PORTFOLIO_DB) as conn:
        rows = conn.execute(
            "SELECT symbol, side, qty, entry_price, current_price, entry_time "
            "FROM open_positions ORDER BY symbol"
        ).fetchall()
    cols = ["symbol", "side", "qty", "entry_price", "current_price", "entry_time"]
    positions = [dict(zip(cols, r)) for r in rows]

    # Fetch live prices for all symbols in one batch
    if positions:
        try:
            import yfinance as yf
            symbols = [p["symbol"] for p in positions]
            # yfinance batch download (last 2 days to ensure we get a close)
            df = yf.download(symbols, period="2d", progress=False)
            if not df.empty:
                close = df["Close"]
                for p in positions:
                    sym = p["symbol"]
                    try:
                        if isinstance(close, pd.Series):
                            # single symbol returns a Series
                            price = float(close.dropna().iloc[-1])
                        else:
                            price = float(close[sym].dropna().iloc[-1])
                        if price > 0:
                            p["current_price"] = price
                    except (KeyError, IndexError, TypeError):
                        pass  # keep DB value as fallback
        except Exception as e:
            logger.debug(f"Live price refresh failed: {e}")

    return positions


def _load_scan_history() -> list[dict]:
    """Load scan log from auto_trader DB."""
    if not _TRADER_DB.exists():
        return []
    with sqlite3.connect(_TRADER_DB) as conn:
        rows = conn.execute(
            "SELECT timestamp, symbols_scanned, signals_generated, "
            "trades_executed, duration_sec FROM scans ORDER BY timestamp DESC LIMIT 500"
        ).fetchall()
    cols = ["timestamp", "symbols_scanned", "signals_generated",
            "trades_executed", "duration_sec"]
    return [dict(zip(cols, r)) for r in rows]


def _get_benchmark_return(symbol: str, days: int = 1) -> float | None:
    """Get benchmark return over N days using yfinance."""
    try:
        import yfinance as yf
        end = datetime.now()
        start = end - timedelta(days=max(days + 5, 10))
        df = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), progress=False)
        if df.empty or len(df) < 2:
            return None
        # Use last N trading days
        if len(df) > days:
            df = df.tail(days + 1)
        close = df["Close"].squeeze().dropna()  # handle MultiIndex + NaN
        if len(close) < 2:
            return None
        first = float(close.iloc[0])
        last = float(close.iloc[-1])
        if first == 0:
            return None
        return (last - first) / first
    except Exception as e:
        logger.debug(f"Benchmark fetch failed for {symbol}: {e}")
        return None


# ── PDF Generator ────────────────────────────────────────────

class PerformanceReportPDF(FPDF):
    """Custom PDF with dark-themed header/footer."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(0, 212, 170)  # accent green
        self.cell(0, 10, "Alpha Trading Platform", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, f"Performance Report  -  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                  new_x="LMARGIN", new_y="NEXT")
        sl_summary = _stop_loss_header_text()
        if sl_summary:
            self.set_font("Helvetica", "", 7)
            self.cell(0, 4, sl_summary, new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 10, f"Side {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(0, 0, 0)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def kpi_row(self, items: list[tuple[str, str]], max_per_row: int = 4):
        """Print KPI boxes in rows: [(label, value), ...].
        Splits into multiple rows if more than max_per_row items."""
        for start in range(0, len(items), max_per_row):
            chunk = items[start:start + max_per_row]
            col_w = (self.w - 20) / len(chunk)

            # Labels
            self.set_font("Helvetica", "", 8)
            self.set_text_color(100, 116, 139)
            y0 = self.get_y()
            for i, (label, _) in enumerate(chunk):
                self.set_xy(10 + i * col_w, y0)
                self.cell(col_w, 5, label)
            self.ln(5)

            # Values
            self.set_font("Helvetica", "B", 11)
            self.set_text_color(0, 0, 0)
            y1 = self.get_y()
            for i, (_, value) in enumerate(chunk):
                self.set_xy(10 + i * col_w, y1)
                # Truncate if too wide for column
                while self.get_string_width(value) > col_w - 2 and len(value) > 5:
                    value = value[:-2] + ".."
                self.cell(col_w, 7, value)
            self.ln(10)

    def table(self, headers: list[str], rows: list[list[str]],
              col_widths: list[float] | None = None):
        """Simple table with header row."""
        if not col_widths:
            col_widths = [(self.w - 20) / len(headers)] * len(headers)

        # Header
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(240, 240, 245)
        self.set_text_color(50, 50, 50)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True)
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 8)
        self.set_text_color(30, 30, 30)
        for row in rows:
            if self.get_y() > 265:
                self.add_page()
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1)
            self.ln()
        self.ln(4)


def generate_performance_report() -> bytes:
    """
    Generate a full performance PDF report.
    Returns the PDF as bytes (ready for download).
    """
    pdf = PerformanceReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # ── Load Data ────────────────────────────────────────────
    equity_history = _load_equity_history()
    closed_trades = _load_closed_trades()
    open_positions = _load_open_positions()
    scan_history = _load_scan_history()

    # ── Abbreviation Glossary ──────────────────────────────
    pdf.section_title("Abbreviations")
    abbrevs = [
        ("P&L", "Profit and Loss"),
        ("SL", "Stop Loss"),
        ("TP", "Take Profit"),
        ("YTD", "Year To Date"),
        ("MTD", "Month To Date"),
        ("MDD", "Maximum Drawdown"),
        ("DD", "Drawdown"),
        ("DKK", "Danish Krone"),
        ("USD", "US Dollar"),
        ("ROI", "Return on Investment"),
        ("KPI", "Key Performance Indicator"),
        ("RSI", "Relative Strength Index"),
        ("MACD", "Moving Average Convergence Divergence"),
        ("SMA", "Simple Moving Average"),
        ("EMA", "Exponential Moving Average"),
        ("ATR", "Average True Range"),
        ("ETF", "Exchange-Traded Fund"),
        ("S&P", "Standard & Poor's"),
        ("MSCI", "Morgan Stanley Capital International"),
    ]
    # Render as two-column table
    col_w = 88
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(26, 28, 36)
    pdf.set_text_color(0, 212, 170)
    pdf.cell(28, 6, "Abbrev.", border=0, fill=True)
    pdf.cell(col_w, 6, "Full Term", border=0, fill=True)
    pdf.cell(5, 6, "", border=0)
    pdf.cell(28, 6, "Abbrev.", border=0, fill=True)
    pdf.cell(col_w, 6, "Full Term", border=0, ln=True, fill=True)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(226, 232, 240)
    mid = (len(abbrevs) + 1) // 2
    for i in range(mid):
        a1, f1 = abbrevs[i]
        pdf.cell(28, 5, a1, border=0)
        pdf.cell(col_w, 5, f1, border=0)
        pdf.cell(5, 5, "", border=0)
        if i + mid < len(abbrevs):
            a2, f2 = abbrevs[i + mid]
            pdf.cell(28, 5, a2, border=0)
            pdf.cell(col_w, 5, f2, border=0, ln=True)
        else:
            pdf.ln()
    pdf.ln(4)

    # ── 1. Portfolio Summary ─────────────────────────────────
    pdf.section_title("1. Portfolio Summary")

    if equity_history:
        current_equity = equity_history[-1]["equity"]
        first_equity = equity_history[0]["equity"]
        total_return = (current_equity - first_equity) / first_equity if first_equity else 0
        peak = max(e["equity"] for e in equity_history)
        drawdown = (peak - current_equity) / peak if peak else 0

        # Calculate recent session (last 6-8 hours)
        now = datetime.now()
        session_start = now - timedelta(hours=8)
        session_points = [e for e in equity_history
                          if e["timestamp"] >= session_start.isoformat()]
        if session_points:
            session_start_eq = session_points[0]["equity"]
            session_return = (current_equity - session_start_eq) / session_start_eq
        else:
            session_start_eq = current_equity
            session_return = 0

        # Equity returns for Sharpe
        equities = [e["equity"] for e in equity_history]
        if len(equities) > 2:
            returns = np.diff(equities) / np.array(equities[:-1])
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
    else:
        current_equity = 100000
        total_return = 0
        drawdown = 0
        session_return = 0
        sharpe = 0

    # USD/DKK
    usd_dkk = 6.90
    try:
        import yfinance as yf
        fx = yf.Ticker("DKK=X")
        rate = getattr(fx.fast_info, "last_price", None)
        if rate and rate > 0:
            usd_dkk = rate
    except Exception:
        pass

    total_dkk = current_equity * usd_dkk
    initial_dkk = first_equity * usd_dkk if equity_history else 100_000 * usd_dkk
    net_gain_dkk = total_dkk - initial_dkk
    total_return_dkk = net_gain_dkk / initial_dkk if initial_dkk else 0

    pdf.kpi_row([
        ("Portfolio Value", f"{total_dkk:,.0f} DKK"),
        ("Net Gain (DKK)", f"{net_gain_dkk:+,.0f}"),
        ("Total Return", f"{total_return_dkk * 100:+.2f}%"),
        ("Session (8h)", f"{session_return * 100:+.2f}%"),
        ("Max Drawdown", f"{drawdown * 100:.2f}%"),
        ("Sharpe Ratio", f"{sharpe:.2f}"),
    ], max_per_row=3)

    # ── 2. Performance vs Market ─────────────────────────────
    pdf.section_title("2. Performance vs Market Benchmarks")

    benchmarks = {
        "S&P 500 (SPY)": "SPY",
        "MSCI World (URTH)": "URTH",
        "Nasdaq 100 (QQQ)": "QQQ",
        "OMXC25 (Denmark)": "NOVO-B.CO",  # proxy
    }

    bench_rows = []
    for name, symbol in benchmarks.items():
        # 1-day return
        ret_1d = _get_benchmark_return(symbol, days=1)
        # 30-day return
        ret_30d = _get_benchmark_return(symbol, days=30)

        bench_rows.append([
            name,
            f"{ret_1d * 100:+.2f}%" if ret_1d is not None else "N/A",
            f"{ret_30d * 100:+.2f}%" if ret_30d is not None else "N/A",
        ])

    # Add our portfolio
    bench_rows.insert(0, [
        "Alpha Trader (this portfolio)",
        f"{session_return * 100:+.2f}%",
        f"{total_return * 100:+.2f}%",
    ])

    pdf.table(
        headers=["Benchmark", "Last Session / 1D", "30-Day Return"],
        rows=bench_rows,
        col_widths=[70, 55, 55],
    )

    # Alpha calculation
    spy_30d = _get_benchmark_return("SPY", days=30)
    if spy_30d is not None:
        alpha = total_return - spy_30d
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(0, 150, 100) if alpha >= 0 else pdf.set_text_color(200, 50, 50)
        pdf.cell(0, 8,
                 f"Alpha vs S&P 500 (30d): {alpha * 100:+.2f}% "
                 f"({'outperforming' if alpha >= 0 else 'underperforming'})",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # ── 3. Trade Statistics ──────────────────────────────────
    pdf.section_title("3. Trade Statistics")

    if closed_trades:
        total_trades = len(closed_trades)
        winners = [t for t in closed_trades
                   if (t["exit_price"] - t["entry_price"]) * (1 if t["side"] == "long" else -1) > 0]
        losers = [t for t in closed_trades
                  if (t["exit_price"] - t["entry_price"]) * (1 if t["side"] == "long" else -1) <= 0]
        win_rate = len(winners) / total_trades if total_trades else 0

        total_pnl = sum(
            (t["exit_price"] - t["entry_price"]) * t["qty"] * (1 if t["side"] == "long" else -1)
            for t in closed_trades
        )
        avg_win = np.mean([
            (t["exit_price"] - t["entry_price"]) * t["qty"] * (1 if t["side"] == "long" else -1)
            for t in winners
        ]) if winners else 0
        avg_loss = np.mean([
            abs((t["exit_price"] - t["entry_price"]) * t["qty"] * (1 if t["side"] == "long" else -1))
            for t in losers
        ]) if losers else 0

        profit_factor = (sum(
            (t["exit_price"] - t["entry_price"]) * t["qty"] * (1 if t["side"] == "long" else -1)
            for t in winners
        ) / abs(sum(
            (t["exit_price"] - t["entry_price"]) * t["qty"] * (1 if t["side"] == "long" else -1)
            for t in losers
        ))) if losers and sum(
            (t["exit_price"] - t["entry_price"]) * t["qty"] * (1 if t["side"] == "long" else -1)
            for t in losers
        ) != 0 else float("inf")

        pdf.kpi_row([
            ("Total Trades", str(total_trades)),
            ("Win Rate", f"{win_rate * 100:.1f}%"),
            ("Total P&L", f"${total_pnl:+,.2f}"),
            ("Avg Win", f"${avg_win:+,.2f}"),
            ("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float("inf") else "N/A"),
        ])

        # Exit reason breakdown
        reasons = {}
        for t in closed_trades:
            r = t.get("exit_reason", "unknown")
            reasons[r] = reasons.get(r, 0) + 1

        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 6, "Exit Reasons: " + ", ".join(f"{k}: {v}" for k, v in reasons.items()),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

        # Recent trades table
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 7, "Last 20 Closed Trades:", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        trade_rows = []
        for t in closed_trades[:20]:
            pnl = (t["exit_price"] - t["entry_price"]) * t["qty"]
            if t["side"] == "short":
                pnl = -pnl
            pnl_pct = (t["exit_price"] / t["entry_price"] - 1) * 100 if t["entry_price"] else 0
            if t["side"] == "short":
                pnl_pct = -pnl_pct
            trade_rows.append([
                t["symbol"],
                t["side"].upper(),
                f"${t['entry_price']:.2f}",
                f"${t['exit_price']:.2f}",
                f"${pnl:+.2f}",
                f"{pnl_pct:+.1f}%",
                t.get("exit_reason", "")[:12],
            ])
        pdf.table(
            headers=["Symbol", "Side", "Entry", "Exit", "P&L", "P&L%", "Reason"],
            rows=trade_rows,
            col_widths=[28, 15, 25, 25, 28, 20, 30],
        )
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, "No closed trades yet.", new_x="LMARGIN", new_y="NEXT")

    # ── 4. Performance by Exchange / Region ──────────────────
    pdf.add_page()
    pdf.section_title("4. Performance by Exchange & Region")

    # Classify positions into exchange groups
    _EXCHANGE_GROUPS = {
        "US (NYSE/Nasdaq)":    {"suffixes": [], "no_dot": True, "benchmark": "SPY",       "bench_name": "S&P 500",
                                "desc": "World's largest equity market; mega-cap tech, financials, and healthcare."},
        "Chicago (CME/CBOT)":  {"suffixes": ["=F"],             "benchmark": "ES=F",      "bench_name": "S&P 500 futures",
                                "desc": "Futures exchange for commodities, indices, treasuries, and energy contracts."},
        "ETFs":                {"suffixes": ["XL"],             "benchmark": "SPY",       "bench_name": "SPY",
                                "desc": "Sector and thematic ETFs for broad market and sector-level exposure."},
        "Denmark (OMXC)":      {"suffixes": [".CO"],            "benchmark": "NOVO-B.CO", "bench_name": "OMXC25 proxy",
                                "desc": "Nordic exchange dominated by Novo Nordisk, Maersk, and green energy plays."},
        "Sweden (OMX Sthlm)":  {"suffixes": [".ST"],            "benchmark": "ERIC-B.ST", "bench_name": "OMX Sthlm proxy",
                                "desc": "Export-heavy market led by Ericsson, Atlas Copco, and industrials."},
        "Norway (Oslo)":       {"suffixes": [".OL"],            "benchmark": "EQNR.OL",  "bench_name": "OBX proxy",
                                "desc": "Oil, gas, seafood, and shipping; Equinor and Mowi dominate."},
        "Finland (Helsinki)":  {"suffixes": [".HE"],            "benchmark": "SAMPO.HE",  "bench_name": "OMX Hels. proxy",
                                "desc": "Compact Nordic market with forestry, insurance, and telecom equipment leaders."},
        "Germany (XETRA)":     {"suffixes": [".DE"],            "benchmark": "SAP.DE",    "bench_name": "DAX proxy",
                                "desc": "Europe's largest economy; autos, chemicals, enterprise software, and engineering."},
        "France (Euronext)":   {"suffixes": [".PA"],            "benchmark": "MC.PA",     "bench_name": "CAC 40 proxy",
                                "desc": "Luxury goods, energy, and banking; LVMH, TotalEnergies, and BNP Paribas."},
        "Netherlands (AMS)":   {"suffixes": [".AS"],            "benchmark": "ASML.AS",   "bench_name": "AEX proxy",
                                "desc": "Semiconductor lithography leader ASML, Philips, and Heineken."},
        "Switzerland (SIX)":   {"suffixes": [".SW"],            "benchmark": "NESN.SW",   "bench_name": "SMI proxy",
                                "desc": "Pharma and consumer giants; Nestle, Novartis, and Roche."},
        "Italy (MIL)":         {"suffixes": [".MI"],            "benchmark": "ENEL.MI",   "bench_name": "FTSE MIB proxy",
                                "desc": "Southern European exchange with utilities, luxury goods, and banking majors."},
        "Spain (BME)":         {"suffixes": [".MC"],            "benchmark": "ITX.MC",    "bench_name": "IBEX proxy",
                                "desc": "Iberian market anchored by Inditex, Santander, and renewable energy firms."},
        "UK (LSE)":            {"suffixes": [".L"],             "benchmark": "SHEL.L",    "bench_name": "FTSE 100 proxy",
                                "desc": "Global financial hub; oil majors, mining giants, pharma, and consumer staples."},
        "Japan (TSE)":         {"suffixes": [".T"],             "benchmark": "7203.T",    "bench_name": "Nikkei proxy",
                                "desc": "Toyota, Sony, and tech conglomerates; Asia's most mature equity market."},
        "Hong Kong (HKEX)":    {"suffixes": [".HK"],            "benchmark": "0005.HK",   "bench_name": "HSI proxy",
                                "desc": "Asia-Pacific gateway; Chinese tech, HK financials, and property conglomerates."},
        "India (NSE)":         {"suffixes": [".NS", ".BO"],     "benchmark": "RELIANCE.NS","bench_name": "Nifty 50 proxy",
                                "desc": "Fast-growing economy; Reliance, IT services giants, and banking leaders."},
        "Australia (ASX)":     {"suffixes": [".AX"],            "benchmark": "BHP.AX",    "bench_name": "ASX 200 proxy",
                                "desc": "Resource-heavy market with mining, banking, and biotech exposure."},
        "New Zealand (NZX)":   {"suffixes": [".NZ"],            "benchmark": "FPH.NZ",    "bench_name": "NZX 50 proxy",
                                "desc": "Small-cap market with dairy, healthcare, and utilities focus."},
        "Crypto":              {"suffixes": ["-USD"],            "benchmark": "BTC-USD",   "bench_name": "Bitcoin",
                                "desc": "24/7 digital asset market; Bitcoin, Ethereum, and major altcoins."},
    }

    def _classify_symbol(sym: str) -> str:
        # Check suffix-based groups first (futures, ETFs, international)
        for group, cfg in _EXCHANGE_GROUPS.items():
            if cfg.get("no_dot"):
                continue  # skip US catch-all on first pass
            for sfx in cfg.get("suffixes", []):
                if sfx in sym:
                    return group
        # US catch-all: plain symbols without dot/suffix
        if "." not in sym and "=" not in sym and not sym.startswith("XL") and not sym.endswith("-USD"):
            return "US (NYSE/Nasdaq)"
        return "Other"

    def _pos_pnl_pct(p: dict) -> float:
        if not p["entry_price"]:
            return 0.0
        pct = (p["current_price"] / p["entry_price"] - 1) * 100
        return -pct if p["side"] == "short" else pct

    # Load per-exchange stop-loss overrides
    _sl_map = {}
    try:
        if _EXCHANGE_SL_PATH.exists():
            _sl_map = json.loads(_EXCHANGE_SL_PATH.read_text())
    except Exception:
        pass

    # Map exchange group names → config keys for stop-loss lookup
    _GROUP_TO_SL_KEY = {
        "US (NYSE/Nasdaq)":    "us_stocks",
        "Chicago (CME/CBOT)":  "chicago",
        "ETFs":                "etfs",
        "Denmark (OMXC)":      "denmark",
        "Sweden (OMX Sthlm)":  "sweden",
        "Norway (Oslo)":       "norway",
        "Finland (Helsinki)":  "finland",
        "Germany (XETRA)":     "germany",
        "France (Euronext)":   "france",
        "Netherlands (AMS)":   "netherlands",
        "Switzerland (SIX)":   "switzerland",
        "Italy (MIL)":         "italy",
        "Spain (BME)":         "spain",
        "UK (LSE)":            "london",
        "Japan (TSE)":         "japan",
        "Hong Kong (HKEX)":    "hong_kong",
        "India (NSE)":         "india",
        "Australia (ASX)":     "australia",
        "New Zealand (NZX)":   "new_zealand",
        "Crypto":              "crypto",
    }

    # Read global default stop-loss from risk config
    _default_sl = 5.0
    try:
        from config.settings import settings
        _default_sl = settings.risk.stop_loss_pct * 100
    except Exception:
        pass
    try:
        _gsl_path = _EXCHANGE_SL_PATH.parent / "global_stop_loss.json"
        if _gsl_path.exists():
            _default_sl = json.loads(_gsl_path.read_text()).get("stop_loss_pct", _default_sl)
    except Exception:
        pass

    # Group positions by exchange
    groups: dict[str, list[dict]] = {}
    for p in open_positions:
        g = _classify_symbol(p["symbol"])
        groups.setdefault(g, []).append(p)

    total_unrealized = 0.0
    all_bench_returns = []  # collect all benchmark 7d returns

    def _pos_pnl_dollar(p: dict) -> float:
        mult = 1 if p["side"] == "long" else -1
        return (p["current_price"] - p["entry_price"]) * p["qty"] * mult

    # Iterate ALL configured exchanges (not just ones with positions)
    for group_name, cfg in _EXCHANGE_GROUPS.items():
        positions = groups.get(group_name, [])
        bench_sym = cfg.get("benchmark", "SPY")
        bench_name = cfg.get("bench_name", bench_sym)
        desc = cfg.get("desc", "")

        if positions:
            # Calculate group stats
            group_pnl = sum(
                (p["current_price"] - p["entry_price"]) * p["qty"] * (1 if p["side"] == "long" else -1)
                for p in positions
            )
            group_value = sum(p["current_price"] * p["qty"] for p in positions)
            total_unrealized += group_pnl

            # Sort by P&L% to find best and worst
            scored = [(p, _pos_pnl_pct(p)) for p in positions]
            scored.sort(key=lambda x: x[1], reverse=True)
            best_pos, best_pnl = scored[0]
            worst_pos, worst_pnl = scored[-1]

            bench_ret = _get_benchmark_return(bench_sym, days=7)
            if bench_ret is not None and not np.isnan(bench_ret):
                all_bench_returns.append(bench_ret * 100)
            avg_pnl_pct_val = float(np.mean([s[1] for s in scored]))

            # Group header with stop-loss
            sl_key = _GROUP_TO_SL_KEY.get(group_name)
            sl_pct = _sl_map.get(sl_key, _default_sl) if sl_key else _default_sl
            sl_label = f"SL {sl_pct:.1f}%"
            if sl_key and sl_key in _sl_map:
                sl_label += " (custom)"

            # Count news articles for this exchange's symbols
            _article_count = 0
            try:
                import sqlite3 as _sq3
                from pathlib import Path as _Pa
                _ndb = _Pa("data_cache/news_sentiment.db")
                if _ndb.exists():
                    with _sq3.connect(_ndb) as _nc:
                        _syms = [p["symbol"] for p in positions]
                        _ph = ",".join("?" * len(_syms))
                        _row = _nc.execute(
                            f"SELECT COALESCE(SUM(news_count), 0) FROM daily_sentiment WHERE symbol IN ({_ph})",
                            _syms
                        ).fetchone()
                        _article_count = _row[0] if _row else 0
            except Exception:
                pass

            _art_label = f", {_article_count:,} artikler" if _article_count > 0 else ""

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 7,
                     f"{group_name}  -  {len(positions)} positions, "
                     f"value ${group_value:,.0f}, P&L ${group_pnl:+,.0f}{_art_label}  |  {sl_label}",
                     new_x="LMARGIN", new_y="NEXT")

            # Exchange description
            if desc:
                pdf.set_font("Helvetica", "I", 7)
                pdf.set_text_color(100, 116, 139)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin, 3.5,
                               f"  {desc} Benchmark is the 7-day total return of {bench_name} "
                               f"representing the overall exchange performance.")

            # Benchmark comparison line
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "", 8)
            if bench_ret is not None and not np.isnan(bench_ret):
                bench_pct = bench_ret * 100
                trader_pct = avg_pnl_pct_val
                alpha = trader_pct / 100 - bench_ret
                bench_dir = "up" if bench_pct >= 0 else "down"
                trader_dir = "up" if trader_pct >= 0 else "down"
                if alpha >= 0:
                    pdf.set_text_color(0, 150, 100)
                    verdict = "outperforming"
                else:
                    pdf.set_text_color(200, 50, 50)
                    verdict = "underperforming"
                pdf.cell(0, 5,
                         f"  {bench_name} was {bench_dir} {bench_pct:+.2f}% (7d) | "
                         f"Trader was {trader_dir} {trader_pct:+.2f}% | "
                         f"Performance: {trader_pct:+.2f}/{bench_pct:+.2f} ({verdict})",
                         new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.set_text_color(100, 116, 139)
                pdf.cell(0, 5,
                         f"  {bench_name}: N/A  |  Trader avg: {avg_pnl_pct_val:+.2f}%",
                         new_x="LMARGIN", new_y="NEXT")

            # Best & Worst table
            best_pnl_d = _pos_pnl_dollar(best_pos)
            bw_rows = []
            bw_rows.append([
                "BEST", best_pos["symbol"], best_pos["side"].upper(),
                f"${best_pos['entry_price']:.2f}", f"${best_pos['current_price']:.2f}",
                f"${best_pnl_d:+,.2f}", f"{best_pnl:+.1f}%",
            ])
            if worst_pos["symbol"] != best_pos["symbol"]:
                worst_pnl_d = _pos_pnl_dollar(worst_pos)
                bw_rows.append([
                    "WORST", worst_pos["symbol"], worst_pos["side"].upper(),
                    f"${worst_pos['entry_price']:.2f}", f"${worst_pos['current_price']:.2f}",
                    f"${worst_pnl_d:+,.2f}", f"{worst_pnl:+.1f}%",
                ])
            pdf.set_text_color(30, 30, 30)
            pdf.table(
                headers=["Rank", "Symbol", "Side", "Entry", "Current", "P&L", "P&L%"],
                rows=bw_rows,
                col_widths=[14, 26, 14, 26, 26, 22, 18],
            )
        else:
            # No positions on this exchange — show it with benchmark only
            bench_ret = _get_benchmark_return(bench_sym, days=7)
            if bench_ret is not None and not np.isnan(bench_ret):
                all_bench_returns.append(bench_ret * 100)

            sl_key = _GROUP_TO_SL_KEY.get(group_name)
            sl_pct = _sl_map.get(sl_key, _default_sl) if sl_key else _default_sl
            sl_label = f"SL {sl_pct:.1f}%"
            if sl_key and sl_key in _sl_map:
                sl_label += " (custom)"

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(0, 7, f"{group_name}  -  No positions  |  {sl_label}",
                     new_x="LMARGIN", new_y="NEXT")

            if desc:
                pdf.set_font("Helvetica", "I", 7)
                pdf.set_text_color(100, 116, 139)
                pdf.cell(0, 4, f"  {desc}", new_x="LMARGIN", new_y="NEXT")

            pdf.set_x(pdf.l_margin)
            pdf.set_font("Helvetica", "", 8)
            if bench_ret is not None and not np.isnan(bench_ret):
                bench_pct = bench_ret * 100
                bench_dir = "up" if bench_pct >= 0 else "down"
                pdf.set_text_color(100, 116, 139)
                pdf.cell(0, 5,
                         f"  {bench_name} was {bench_dir} {bench_pct:+.2f}% (7d)",
                         new_x="LMARGIN", new_y="NEXT")
            else:
                pdf.set_text_color(100, 116, 139)
                pdf.cell(0, 5, f"  {bench_name}: N/A",
                         new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)

    # Grand total — compute portfolio 7d return from equity history
    portfolio_7d_ret = None
    if equity_history:
        now = datetime.now()
        cutoff = (now - timedelta(days=7)).isoformat()
        older = [e for e in equity_history if e["timestamp"] <= cutoff]
        if older:
            eq_7d_ago = older[-1]["equity"]
            eq_now = equity_history[-1]["equity"]
            if eq_7d_ago > 0:
                portfolio_7d_ret = (eq_now - eq_7d_ago) / eq_7d_ago * 100

    total_entry_value = sum(
        p["entry_price"] * p["qty"] for p in open_positions
    ) if open_positions else 0
    total_pnl_pct = (total_unrealized / total_entry_value * 100) if total_entry_value else 0

    if open_positions:
        pdf.set_font("Helvetica", "B", 11)
        if total_unrealized >= 0:
            pdf.set_text_color(0, 150, 100)
        else:
            pdf.set_text_color(200, 50, 50)
        pdf.cell(0, 8,
                 f"Total Unrealized P&L: ${total_unrealized:+,.2f} "
                 f"({total_unrealized * usd_dkk:+,.0f} DKK) / {total_pnl_pct:+.2f}% (since entry)",
                 new_x="LMARGIN", new_y="NEXT")

    # Portfolio 7d return + combined market performance
    if portfolio_7d_ret is not None or all_bench_returns:
        pdf.set_font("Helvetica", "B", 11)
        parts = []
        if portfolio_7d_ret is not None:
            port_dir = "up" if portfolio_7d_ret >= 0 else "down"
            parts.append(f"Portfolio (7d): {port_dir} {portfolio_7d_ret:+.2f}%")
        if all_bench_returns:
            avg_market = float(np.mean(all_bench_returns))
            market_dir = "up" if avg_market >= 0 else "down"
            parts.append(f"Market avg ({len(all_bench_returns)} exchanges, 7d): {market_dir} {avg_market:+.2f}%")
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 8, "  |  ".join(parts), new_x="LMARGIN", new_y="NEXT")

    if not open_positions:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, "No open positions.", new_x="LMARGIN", new_y="NEXT")

    # ── 5. Scanner Activity ──────────────────────────────────
    pdf.add_page()
    pdf.section_title("5. Scanner Activity (Last 24h)")

    if scan_history:
        now = datetime.now()
        recent = [s for s in scan_history
                  if s["timestamp"] >= (now - timedelta(hours=24)).isoformat()]
        total_scans = len(recent)
        total_signals = sum(s.get("signals_generated", 0) or 0 for s in recent)
        total_executed = sum(s.get("trades_executed", 0) or 0 for s in recent)
        avg_duration = np.mean([s.get("duration_sec", 0) or 0 for s in recent]) if recent else 0

        pdf.kpi_row([
            ("Scans (24h)", str(total_scans)),
            ("Signals Generated", str(total_signals)),
            ("Trades Executed", str(total_executed)),
            ("Avg Scan Time", f"{avg_duration:.1f}s"),
        ])
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 10, "No scan data available.", new_x="LMARGIN", new_y="NEXT")

    # ── 6. Feedback Loop Status & Actions ────────────────────
    pdf.ln(4)
    pdf.section_title("6. Feedback Loop Status & Recommendations")

    # 6a. Read actual feedback loop state from the learner DB
    feedback_active = False
    learner_summary = {}
    learner_analytics = {}
    try:
        from src.learning.continuous_learner import ContinuousLearner
        _learn_db = Path("data_cache/learning.db")
        _trade_db = Path("data_cache/auto_trader_log.db")
        if _learn_db.exists():
            learner = ContinuousLearner(
                db_path=str(_learn_db),
                trade_db_path=str(_trade_db),
            )
            learner_summary = learner.get_learning_summary()
            learner_analytics = learner.get_trade_analytics()
            feedback_active = True
    except Exception as e:
        logger.debug(f"Could not read learner state: {e}")

    # 6b. Feedback loop status box
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, "A. Feedback Loop Implementation Status", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)

    status_lines = []
    if feedback_active:
        total_lessons = learner_summary.get("total_lessons", 0)
        drift_status = learner_summary.get("drift_status", "OK")
        weights = learner_summary.get("ensemble_weights", {})
        model_perf = learner_summary.get("model_performance", {})

        status_lines.append(
            f"ACTIVE: ContinuousLearner is running. "
            f"{total_lessons} trade outcomes analyzed so far."
        )
        status_lines.append(
            f"Concept drift status: {drift_status}"
        )

        if weights:
            w_str = ", ".join(f"{k}: {v:.0%}" for k, v in weights.items())
            status_lines.append(f"Current ensemble weights: {w_str}")

        if model_perf:
            for mname, mdata in model_perf.items():
                acc = mdata.get("accuracy", 0)
                sigs = mdata.get("total_signals", 0)
                avg_p = mdata.get("avg_pnl", 0)
                status_lines.append(
                    f"  Model '{mname}': accuracy={acc:.0%}, "
                    f"signals={sigs}, avg_pnl={avg_p:+.2f}%"
                )
        else:
            status_lines.append(
                "  No per-model data yet (needs more closed trades to evaluate)."
            )

        # What the feedback loop adjusts
        status_lines.append("")
        status_lines.append("What the feedback loop controls:")
        status_lines.append(
            "  - min_confidence: raised when win rate drops (<40%), "
            "lowered when performing well (>60%)"
        )
        status_lines.append(
            "  - position_size_pct: reduced during poor performance or drift, "
            "increased (max 15%) during good streaks"
        )
        status_lines.append(
            "  - ensemble_weights: strategies that predict better get more weight"
        )
        status_lines.append(
            "  - concept drift: if accuracy drops >15%, "
            "thresholds tighten aggressively (emergency mode)"
        )
    else:
        status_lines.append(
            "INACTIVE: ContinuousLearner database not found. "
            "The feedback loop will activate after the first trading session."
        )
        status_lines.append(
            "Once active, it will automatically adjust buy/sell thresholds "
            "based on trade outcomes."
        )

    content_w = pdf.w - pdf.l_margin - pdf.r_margin
    for line in status_lines:
        if line == "":
            pdf.ln(2)
        else:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(content_w, 5, line)

    # 6c. Concrete recommendations based on current data
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, "B. Actionable Recommendations", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 30, 30)

    recommendations = []

    # Session performance
    if session_return < -0.005:
        recommendations.append(
            f"[SESSION LOSS {session_return*100:.2f}%] "
            f"The overnight session lost value. "
            f"The feedback loop {'HAS' if feedback_active else 'WILL'} "
            f"raise min_confidence to filter weaker signals. "
            f"ACTION: Review stop-loss levels on losing positions."
        )
    elif session_return > 0.005:
        recommendations.append(
            f"[SESSION GAIN {session_return*100:+.2f}%] "
            f"Positive session. Current strategy parameters are working."
        )
    else:
        recommendations.append(
            "[FLAT SESSION] Near-zero return. "
            "Likely low market hours or tight entry thresholds."
        )

    # vs benchmarks
    if spy_30d is not None:
        if total_return > spy_30d:
            recommendations.append(
                f"[OUTPERFORMING SPY] Portfolio return {total_return*100:+.2f}% "
                f"vs S&P 500 {spy_30d*100:+.2f}% (30d). "
                f"Alpha: {(total_return - spy_30d)*100:+.2f}%. Keep current strategy mix."
            )
        else:
            gap = (spy_30d - total_return) * 100
            recommendations.append(
                f"[UNDERPERFORMING SPY] Portfolio trails S&P 500 by {gap:.2f}% (30d). "
                f"The feedback loop {'IS adjusting' if feedback_active else 'WILL adjust'} "
                f"strategy weights to favor better-performing models. "
                f"ACTION: Consider reducing exposure to worst-performing exchange groups above."
            )

    # Win rate
    if closed_trades:
        wr = len([t for t in closed_trades
                  if (t["exit_price"] - t["entry_price"]) * (1 if t["side"] == "long" else -1) > 0
                  ]) / len(closed_trades)
        if wr < 0.40:
            recommendations.append(
                f"[LOW WIN RATE {wr*100:.0f}%] "
                f"The feedback loop {'HAS raised' if feedback_active else 'WILL raise'} "
                f"min_confidence from base 40 to ~50-55 and reduced position size by 30%. "
                f"ACTION: Signal quality needs improvement. Review which strategies "
                f"generate losing signals."
            )
        elif wr > 0.60:
            recommendations.append(
                f"[STRONG WIN RATE {wr*100:.0f}%] "
                f"The feedback loop {'HAS relaxed' if feedback_active else 'WILL relax'} "
                f"confidence thresholds slightly to capture more opportunities."
            )

    # Drawdown
    if drawdown > 0.10:
        recommendations.append(
            f"[HIGH DRAWDOWN {drawdown*100:.1f}%] CRITICAL: Approaching risk limit (10%). "
            f"ACTION: Consider closing worst performers and reducing new position sizing."
        )
    elif drawdown > 0.05:
        recommendations.append(
            f"[DRAWDOWN {drawdown*100:.1f}%] WARNING: Elevated drawdown. "
            f"The feedback loop {'IS reducing' if feedback_active else 'WILL reduce'} position sizes."
        )

    # Sharpe
    if sharpe != 0 and sharpe < 0.5:
        recommendations.append(
            f"[SHARPE {sharpe:.2f}] Below target (>1.0). "
            f"Returns are not compensating for risk taken. "
            f"ACTION: Tighten stop-losses, increase diversification."
        )
    elif sharpe >= 1.5:
        recommendations.append(
            f"[SHARPE {sharpe:.2f}] Excellent risk-adjusted returns."
        )

    # Drift
    if feedback_active:
        drift = learner_summary.get("drift_status", "OK")
        if drift == "CRITICAL":
            recommendations.append(
                "[CONCEPT DRIFT - CRITICAL] Market regime has changed significantly. "
                "The feedback loop HAS tightened all thresholds and halved position sizes. "
                "ACTION: Wait for drift to stabilize before manual intervention."
            )
        elif drift == "WARNING":
            recommendations.append(
                "[CONCEPT DRIFT - WARNING] Accuracy declining. "
                "The feedback loop HAS raised confidence threshold by 8 points."
            )

    # Per-exchange group recommendations
    if open_positions:
        for group_name in sorted(groups.keys()):
            positions = groups[group_name]
            cfg = _EXCHANGE_GROUPS.get(group_name, {})
            bench_sym = cfg.get("benchmark", "SPY")
            bench_ret = _get_benchmark_return(bench_sym, days=7)
            avg_pnl = np.mean([_pos_pnl_pct(p) for p in positions])
            if bench_ret is not None and avg_pnl / 100 < bench_ret - 0.02:
                recommendations.append(
                    f"[{group_name.upper()}] Underperforming regional benchmark by "
                    f"{(bench_ret - avg_pnl/100)*100:.1f}%. "
                    f"Consider reducing allocation or tightening stops on this exchange."
                )

    if not recommendations:
        recommendations.append("No specific actions needed. Portfolio is performing within parameters.")

    for i, rec in enumerate(recommendations, 1):
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(content_w, 5, f"{i}. {rec}")
        pdf.ln(2)

    return pdf.output()
