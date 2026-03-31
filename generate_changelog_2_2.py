#!/usr/bin/env python3
"""Generate PDF changelog v2.2 — incremental update since last changelog."""

from fpdf import FPDF
from datetime import datetime


class ChangelogPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(0, 160, 120)
        self.cell(0, 8, "Alpha Trading Platform - Changelog v2.2", align="L")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 160, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 40)
        self.ln(4)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 160, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def subsection(self, title):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(50, 50, 70)
        self.ln(2)
        self.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 4.5, text)
        self.ln(1)

    def bullet(self, text, indent=5):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(40, 40, 40)
        self.set_x(self.l_margin)
        self.multi_cell(0, 4.5, " " * indent + "- " + text)

    def file_entry(self, path, desc):
        self.set_font("Courier", "", 7.5)
        self.set_text_color(0, 100, 80)
        self.cell(0, 4.5, f"  {path}", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(80, 80, 80)
        self.cell(0, 4.5, f"     {desc}", new_x="LMARGIN", new_y="NEXT")

    def table_row(self, cols, widths, bold=False, bg=False):
        if bg:
            self.set_fill_color(240, 248, 245)
        self.set_font("Helvetica", "B" if bold else "", 8)
        self.set_text_color(30, 30, 40)
        for i, (col, w) in enumerate(zip(cols, widths)):
            self.cell(w, 5.5, col, border=0, fill=bg, align="L" if i == 0 else "C")
        self.ln()


def build_pdf():
    pdf = ChangelogPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── TITLE PAGE ──
    pdf.ln(20)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(0, 160, 120)
    pdf.cell(0, 15, "Alpha Trading Platform", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, "Changelog v2.2 - Incremental Update", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)
    pdf.set_draw_color(0, 160, 120)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.multi_cell(0, 6, (
        "This document describes the incremental changes and improvements made to the "
        "Alpha Trading Platform since the last changelog update (v2.1). It covers bug fixes, "
        "new features, performance optimizations, and UI improvements across the dashboard, "
        "broker routing, and trading functionality."
    ), align="C")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(0, 130, 100)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%d %B %Y')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Platform: Rock 5B (aarch64, Debian 12)", align="C", new_x="LMARGIN", new_y="NEXT")

    # ── SECTION 1: CRITICAL BUG FIXES ──
    pdf.add_page()
    pdf.section_title("1. Critical Bug Fixes")

    pdf.subsection("1.1 Sell Order Routing Fix")
    pdf.body_text(
        "The sell function on the Trading page was completely broken for non-US stocks. "
        "Orders for symbols like AMT.NZ would be routed to the wrong broker (e.g., Alpaca instead "
        "of PaperBroker where the position actually lived), causing silent failures."
    )
    pdf.bullet("Root cause: BrokerRouter.sell() used generic exchange-based routing for sells, "
               "which could pick a different broker than the one holding the position.")
    pdf.bullet("Fix: sell() now scans all registered brokers to find which one holds the position "
               "before routing. Falls back to standard routing only for short-sells or when no "
               "position is found.")
    pdf.bullet("Added .NZ (NZX) to suffix_to_exchange and exchange_to_broker routing config.")
    pdf.file_entry("src/broker/broker_router.py", "Smart sell routing + NZX mapping")

    pdf.subsection("1.2 Regime Page Data Fix")
    pdf.body_text(
        "The Regime detection page showed no data because it called a non-existent method "
        "get_historical_data() instead of get_historical(). The exception was silently swallowed."
    )
    pdf.bullet("Fixed method name to get_historical() with lookback_days=500.")
    pdf.bullet("Added yfinance direct fallback if MarketDataFetcher fails.")
    pdf.file_entry("src/dashboard/app.py", "Fixed _get_regime_data() method call")

    pdf.subsection("1.3 Portfolio Value Mismatch")
    pdf.body_text(
        "The Overblik page showed ~710k DKK while the Portfolio page showed ~665k DKK. "
        "This was because Overblik used backtest final_equity (historical simulation) while "
        "Portfolio used live PaperBroker equity."
    )
    pdf.bullet("Overblik now fetches live portfolio equity from BrokerRouter, matching the Portfolio page.")
    pdf.bullet("Falls back to backtest value only if the router is unavailable.")

    pdf.subsection("1.4 Exchange Label Fix (WOW.AX)")
    pdf.body_text(
        "Australian (.AX), New Zealand (.NZ), Japanese (.T), Hong Kong (.HK) and other non-US/EU "
        "stocks were incorrectly labelled as 'NYSE/NASDAQ' on the Portfolio page."
    )
    pdf.bullet("Added 10+ new suffix-to-exchange mappings: .AX->ASX, .NZ->NZX, .T->TSE, "
               ".HK->HKEX, .NS->NSE, .TO->TSX, .SW->SIX, .MI->MIL, .MC->BME, =F->CME.")
    pdf.file_entry("src/dashboard/pages/portfolio.py", "Extended _exchange_from_symbol()")

    # ── SECTION 2: CURRENCY STANDARDISATION ──
    pdf.section_title("2. Currency Standardisation")
    pdf.body_text(
        "Multiple pages displayed values in hardcoded USD ($) instead of the user's selected "
        "display currency. All monetary values now use format_value() which respects the "
        "currency setting (DKK, EUR, USD)."
    )

    pdf.subsection("2.1 Pages Fixed")
    w = [55, 135]
    pdf.table_row(["Page", "Values Converted"], w, bold=True)
    pdf.table_row(["Strategier", "Total P&L, Avg Win, Avg Loss (was $, now display currency)"], w, bg=True)
    pdf.table_row(["Risiko", "Commission, Avg Loss, P&L chart titles and labels"], w)
    pdf.table_row(["Stress Test", "Scenario table values, chart annotations, Monte Carlo metrics"], w, bg=True)
    pdf.table_row(["System Health", "Strategy P&L, Daily/Weekly/Monthly/YTD P&L"], w)
    pdf.table_row(["Smart Money", "Insider trade prices and values"], w, bg=True)
    pdf.table_row(["Options Flow", "Max Pain display values"], w)

    pdf.subsection("2.2 Strategy P&L Now Includes Unrealized Gains")
    pdf.body_text(
        "The Strategier page Total P&L previously only counted realized P&L from closed trades, "
        "while the equity curve included unrealized gains. This caused a confusing mismatch "
        "(e.g., equity curve +3% but P&L showing -$2,297)."
    )
    pdf.bullet("Total P&L now includes unrealized P&L from open positions, matching the equity curve.")

    # ── SECTION 3: TRADING PAGE IMPROVEMENTS ──
    pdf.section_title("3. Trading Page Improvements")

    pdf.subsection("3.1 Performance Optimisation")
    pdf.body_text(
        "Trading responses were very slow due to sequential yfinance API calls. Multiple "
        "optimisations were applied:"
    )
    pdf.bullet("FX rate caching with 5-minute TTL - eliminates repeated DKK/USD lookups.")
    pdf.bullet("Parallel price fetching for Quick Trade modal using ThreadPoolExecutor (5 workers).")
    pdf.bullet("Cash lookups now use the shared BrokerRouter instead of creating new PaperBroker instances.")

    pdf.subsection("3.2 Exchange-Closed Order Queuing")
    pdf.body_text(
        "When a stock exchange is closed, market orders are now automatically converted to limit "
        "orders at the last known price and queued as pending. They appear in the new open orders "
        "view and will execute when the exchange opens."
    )
    pdf.bullet("submit_trade() checks exchange open status via MarketCalendar.")
    pdf.bullet("Quick Trade (Kob/Saelg Top 10) also queues orders for closed exchanges.")
    pdf.bullet("Queued orders shown with orange hourglass icon and 'queued' label.")

    pdf.subsection("3.3 Live Open Orders View (Aabne Ordrer)")
    pdf.body_text(
        "Replaced the static 'Ingen aabne ordrer' placeholder with a live-updating view "
        "that polls all brokers every 10 seconds for pending/submitted orders."
    )
    pdf.bullet("Table shows: time, symbol, side, quantity, price, broker, cancel button.")
    pdf.bullet("Two-layer cancel confirmation: first 'Are you sure?', then 'Confirm final cancellation'.")
    pdf.bullet("Cancellation executes via broker.cancel_order() and refreshes the view.")

    pdf.subsection("3.4 Kob Top 10 / Saelg Top 10 Fallback")
    pdf.body_text(
        "The Quick Trade buttons showed 'no recommendations' when the scanner cache was empty. "
        "Added fallback logic:"
    )
    pdf.bullet("Buy: Falls back to AutoTrader watchlist, then active market symbols from MarketCalendar.")
    pdf.bullet("Sell: Falls back to building recommendations from current portfolio positions.")

    # ── SECTION 4: SMART MONEY & OPTIONS FLOW ──
    pdf.section_title("4. Smart Money & Options Flow Fixes")

    pdf.subsection("4.1 Smart Money Tracker")
    pdf.body_text(
        "The Smart Money page only populated 2 of 5 content views and required a button click. "
        "Now all views are populated and stock selection triggers automatically."
    )
    pdf.bullet("Added dropdown value change as callback trigger (instant analysis on selection).")
    pdf.bullet("Insider Trades Table: shows up to 20 trades with date, name, title, side, shares, price, value.")
    pdf.bullet("Institutional Holdings: shows fund names, shares, value, % held, changes.")
    pdf.bullet("Overall Assessment: synthesized summary from insider sentiment + short interest data.")

    pdf.subsection("4.2 Options Flow")
    pdf.body_text(
        "Same issues as Smart Money - missing assessment view and no dropdown trigger."
    )
    pdf.bullet("Added dropdown trigger for instant analysis on stock selection.")
    pdf.bullet("Added Options Assessment view: unusual activity bias, IV rank interpretation, "
               "PCR interpretation, max pain direction.")

    pdf.subsection("4.3 Alternative Data")
    pdf.body_text("Added dropdown trigger - selecting a stock now auto-analyses without needing button click.")

    # ── SECTION 5: SYSTEM-WIDE IMPROVEMENTS ──
    pdf.section_title("5. System-Wide Improvements")

    pdf.subsection("5.1 Loading Spinners")
    pdf.body_text(
        "All page transitions now show a loading spinner (circle type, accent colour). "
        "Wrapped the main page-content div in dcc.Loading. Also added specific spinners "
        "for the trade result and quick trade result on the Trading page."
    )

    pdf.subsection("5.2 Background Data Refresh (5-minute interval)")
    pdf.body_text(
        "Added a background thread that refreshes cached data every 5 minutes so pages "
        "load instantly even when not recently visited:"
    )
    pdf.bullet("Scanner data (Markedsoverblik, Trading quick trades)")
    pdf.bullet("Backtests (Overblik, Risiko, Strategier)")
    pdf.bullet("Benchmark data (Overblik equity chart)")
    pdf.bullet("First refresh starts 60 seconds after startup to avoid competing with initial preload.")

    pdf.subsection("5.3 Markedsoverblik Button Fix")
    pdf.body_text(
        "Buttons on the Markedsoverblik page were unresponsive when scanner data was loading. "
        "The page used dcc.Location(refresh=True) which caused a full-page reload loop. "
        "Fixed by adding hidden placeholder elements for callback outputs during loading state "
        "and triggering background scanner fetch if not already running."
    )

    # ── SECTION 6: REPORTS & SETTINGS ──
    pdf.section_title("6. Reports & Settings")

    pdf.subsection("6.1 Portfolio Report: Article Counts per Exchange")
    pdf.body_text(
        "The per-exchange section of the PDF portfolio report now shows how many news "
        "articles have been downloaded for each exchange's symbols."
    )
    pdf.bullet("Queries news_sentiment.db for article counts grouped by exchange symbols.")
    pdf.bullet("Displayed as 'X artikler' next to position count and value in the exchange header.")
    pdf.file_entry("src/dashboard/pages/performance_report.py", "Added article count query")

    pdf.subsection("6.2 Settings: Risk Limit Active/Inactive Indicators")
    pdf.body_text(
        "Each of the 5 risk limits on the Settings page now has a visual indicator badge:"
    )
    pdf.bullet("Green 'Aktiv' badge: shown when the save successfully applies the value to the running "
               "AutoTrader/RiskManager.")
    pdf.bullet("Gray 'Inaktiv' badge: shown when the trader isn't running or the apply failed.")
    pdf.bullet("Indicators update immediately when the 'Gem' button is pressed.")
    pdf.ln(1)
    w2 = [65, 125]
    pdf.table_row(["Risk Limit", "What It Controls"], w2, bold=True)
    pdf.table_row(["Max Positions", "Maximum number of open positions"], w2, bg=True)
    pdf.table_row(["Global Stop-Loss", "Portfolio-wide stop-loss percentage"], w2)
    pdf.table_row(["Position Size %", "Max % of portfolio per single position"], w2, bg=True)
    pdf.table_row(["Max DKK per Symbol", "Maximum DKK investment per symbol"], w2)
    pdf.table_row(["Max Exposure %", "Maximum total portfolio exposure (rest is cash)"], w2, bg=True)

    pdf.subsection("6.3 Strategy Decay Links")
    pdf.body_text(
        "When a strategy decay warning appears on the System Health page, each alert now "
        "includes a clickable 'Gaa til Strategier' link that navigates directly to the "
        "Strategier page where the issue can be investigated and corrected."
    )

    # ── SECTION 7: TRANSLATION UPDATES ──
    pdf.section_title("7. Translation Updates")
    pdf.body_text("Added new translation keys to Danish and English language files:")
    pdf.bullet("trading.cancel_order, confirm_cancel, confirm_cancel_final")
    pdf.bullet("trading.cancel_confirm_msg, cancel_confirm_msg2")
    pdf.bullet("trading.order_cancelled, cancel_failed")
    pdf.bullet("trading.exchange_closed_queued")
    pdf.file_entry("lang/dansk.lan", "9 new trading keys")
    pdf.file_entry("lang/eng.lan", "9 new trading keys")

    # ── SECTION 8: FILES MODIFIED ──
    pdf.section_title("8. Files Modified")
    pdf.body_text("Summary of all files changed in this update:")
    pdf.ln(1)

    files = [
        ("src/broker/broker_router.py", "Smart sell routing, .NZ/.NZX mapping"),
        ("src/dashboard/app.py", "Currency fixes, regime fix, live portfolio, loading spinners, "
         "background refresh, markedsoverblik fix, smart money/options/alt data fixes, "
         "risk indicators, strategy decay links"),
        ("src/dashboard/pages/trading.py", "FX caching, parallel price fetch, exchange-closed queuing, "
         "open orders view, cancel confirmation, loading spinners, Kob top 10 fallback"),
        ("src/dashboard/pages/portfolio.py", "Extended exchange label mappings"),
        ("src/dashboard/pages/performance_report.py", "Article counts per exchange in PDF report"),
        ("lang/dansk.lan", "New trading/cancel translation keys"),
        ("lang/eng.lan", "New trading/cancel translation keys"),
    ]

    for path, desc in files:
        pdf.file_entry(path, desc)

    # ── OUTPUT ──
    out = "/home/rock/reports/alpha_trading_changelog_2_2.pdf"
    pdf.output(out)
    print(f"PDF generated: {out}")
    return out


if __name__ == "__main__":
    build_pdf()
