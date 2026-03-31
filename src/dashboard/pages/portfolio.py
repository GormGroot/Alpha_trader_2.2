"""
Portfolio Overview — samlet portefølje-view på tværs af alle brokers.

Dashboard-side: / (index)

Features:
  - Total portfolio value i DKK
  - Day P&L, MTD, YTD
  - Breakdown pie charts (broker, asset type, valuta, sektor)
  - Positions table med skattekolonne
  - Performance vs benchmarks
  - Cash balance per broker
"""

from __future__ import annotations

import json
import dash
from dash import dcc, html, callback, Input, Output, State, no_update, ClientsideFunction
import dash_bootstrap_components as dbc
import base64
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

from loguru import logger
from src.dashboard.i18n import t
from src.dashboard.currency_service import (
    convert_from_usd, format_value, format_value_dkk,
    get_currency_label, get_currency_symbol, get_fx_rates, convert_from_dkk,
)

# ── Colors (matches existing dark theme) ────────────────────

COLORS = {
    "bg": "#0f1117",
    "card": "#1a1c24",
    "accent": "#00d4aa",
    "red": "#ff4757",
    "green": "#2ed573",
    "blue": "#3498db",
    "orange": "#ffa502",
    "purple": "#a855f7",
    "text": "#e2e8f0",
    "muted": "#64748b",
    "border": "#2d3748",
}


# ── Helper: Dark empty figure (prevents white flash) ──────

def _dark_empty_fig(height: int = 280) -> go.Figure:
    """Return a dark-themed empty figure to prevent white flash on load."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ── Helper: KPI Card ───────────────────────────────────────

def _kpi_card(title: str, value: str, change: str = "", positive: bool = True) -> dbc.Card:
    """Generér et KPI-kort til dashboard."""
    change_color = COLORS["green"] if positive else COLORS["red"]
    change_prefix = "+" if positive and change else ""

    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.85rem"}),
            html.H3(value, className="mb-0", style={"color": COLORS["text"], "fontSize": "2rem"}),
            html.Span(
                f"{change_prefix}{change}" if change else "",
                style={"color": change_color, "fontSize": "0.9rem"},
            ),
        ]),
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
        },
    )


# ── Layout ──────────────────────────────────────────────────

def portfolio_layout() -> html.Div:
    """Generér portfolio overview layout."""
    # Pre-render everything from current data so first paint is instant
    initial_data = _get_portfolio_data()
    # Pre-compute trading status once so count and table rows agree
    positions = initial_data.get("positions", [])
    _usd_dkk = initial_data.get("usd_dkk", 6.90)
    _pos_dicts = [_pos_to_dict(p, _usd_dkk) for p in positions]
    n_trading = sum(1 for d in _pos_dicts if d.get("trading"))
    n_closed = len(_pos_dicts) - n_trading
    initial_table = _make_positions_table(initial_data.get("positions", []), usd_dkk=_usd_dkk)

    # Build charts once at page load
    broker_pie = _make_pie(initial_data.get("by_broker", {}), t('portfolio.per_broker'))
    asset_pie = _make_pie(initial_data.get("by_asset_type", {}), t('portfolio.per_asset_type'))
    currency_pie = _make_pie(initial_data.get("by_currency", {}), t('portfolio.per_currency'))
    exchange_pie = _make_pie(initial_data.get("by_exchange", {}), t('portfolio.per_exchange'))

    perf_fig = go.Figure()
    perf_fig.update_layout(
        title=dict(text=t('portfolio.performance_vs_benchmarks'), font=dict(color=COLORS["text"], size=14)),
        paper_bgcolor=COLORS["card"], plot_bgcolor=COLORS["card"],
        xaxis=dict(gridcolor=COLORS["border"], color=COLORS["muted"]),
        yaxis=dict(gridcolor=COLORS["border"], color=COLORS["muted"]),
        height=300, margin=dict(l=40, r=20, t=40, b=30),
    )
    try:
        import sqlite3
        from pathlib import Path as _P
        _eq_db = _P("data_cache/paper_portfolio.db")
        if _eq_db.exists():
            with sqlite3.connect(_eq_db) as _conn:
                _rows = _conn.execute(
                    "SELECT timestamp, equity FROM equity_history ORDER BY id"
                ).fetchall()
            if _rows:
                # Normalize to % return from start
                first_eq = _rows[0][1]
                dates = [r[0][:16] for r in _rows]  # trim to minute
                returns_pct = [(r[1] / first_eq - 1) * 100 for r in _rows]
                perf_fig.add_trace(go.Scatter(
                    x=dates, y=returns_pct,
                    mode="lines", name=t('portfolio.portfolio_label'),
                    line=dict(color=COLORS["accent"], width=2),
                    hovertemplate="%{y:+.2f}%<extra>" + t('portfolio.portfolio_label') + "</extra>",
                ))
                perf_fig.update_layout(
                    yaxis=dict(title="Return %", gridcolor=COLORS["border"],
                               color=COLORS["muted"]),
                )
    except Exception:
        pass
    if not perf_fig.data:
        perf_fig.add_annotation(
            text=t('portfolio.no_performance_data'),
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color=COLORS["muted"], size=14),
        )

    return html.Div([
        # Auto-refresh: only fetches data into store — no visible outputs
        dcc.Interval(id="portfolio-refresh", interval=30_000, n_intervals=0),
        # Store for position data — clientside JS reads this for all updates
        dcc.Store(id="portfolio-prev-state", data=None),
        # Hidden dummy divs for clientside callback outputs (one per callback)
        html.Div(id="portfolio-js-dummy", style={"display": "none"}),
        html.Div(id="portfolio-toggle-dummy", style={"display": "none"}),

        # Header
        dbc.Row([
            dbc.Col(html.H2(
                t('portfolio.title'),
                style={"color": COLORS["text"]},
            ), width=6),
            dbc.Col(html.Div([
                html.Span(
                    t('common.updated') + " ",
                    className="text-muted",
                    style={"fontSize": "0.85rem"},
                ),
                html.Span(
                    id="portfolio-last-updated",
                    className="text-muted",
                    style={"fontSize": "0.85rem"},
                ),
            ], className="text-end", style={"paddingTop": "8px"}), width=6),
        ], className="mb-3"),
        # Hidden elements to keep callback outputs valid
        dcc.Download(id="download-report-pdf"),
        html.Div(id="btn-download-report", style={"display": "none"}),

        # KPI Row — each card has stable IDs, only inner values update
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P(t('portfolio.total_portfolio'), className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                html.H3(id="kpi-total-value", className="mb-0", style={"color": COLORS["text"], "fontSize": "2rem"}),
                html.P(t('portfolio.total_desc'), style={"color": COLORS["muted"], "fontSize": "0.7rem", "marginBottom": "0"}),
            ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P(t('portfolio.unrealized_pnl'), className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                html.H3(id="kpi-pnl-value", className="mb-0", style={"color": COLORS["text"], "fontSize": "2rem"}),
                html.Span(id="kpi-pnl-change", style={"fontSize": "0.9rem"}),
            ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P(t('portfolio.cash'), className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                html.H3(id="kpi-cash-value", className="mb-0", style={"color": COLORS["text"], "fontSize": "2rem"}),
                html.P(t('portfolio.cash_desc'), style={"color": COLORS["muted"], "fontSize": "0.7rem", "marginBottom": "0"}),
            ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P(t('portfolio.positions'), className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                html.H3(id="kpi-num-positions", className="mb-0", style={"color": COLORS["text"], "fontSize": "2rem"}),
                html.Span(id="kpi-equity-value", style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
            ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}), width=3),
        ], className="mb-4"),

        # Charts Row — static, rendered once at page load
        dbc.Row([
            dbc.Col(dcc.Graph(figure=broker_pie, config={"displayModeBar": False}), width=3),
            dbc.Col(dcc.Graph(figure=asset_pie, config={"displayModeBar": False}), width=3),
            dbc.Col(dcc.Graph(figure=currency_pie, config={"displayModeBar": False}), width=3),
            dbc.Col(dcc.Graph(figure=exchange_pie, config={"displayModeBar": False}), width=3),
        ], className="mb-4"),

        # Positions Table
        dbc.Card([
            dbc.CardHeader(
                dbc.Row([
                    dbc.Col(html.H5(t('portfolio.positions'), className="mb-0"), width="auto"),
                    dbc.Col(html.Span(
                        f"🟢 {n_trading} {t('portfolio.trading_count')}   "
                        f"🔴 {n_closed} {t('portfolio.closed_count')}",
                        style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                        id="pos-count-summary",
                    ), className="d-flex align-items-center"),
                    dbc.Col(
                        dbc.Button(
                            t('portfolio.show_all'),
                            id="btn-toggle-closed",
                            size="sm",
                            color="outline-secondary",
                            style={"fontSize": "0.8rem"},
                        ),
                        width="auto",
                    ),
                ], align="center"),
                style={"backgroundColor": COLORS["card"]},
            ),
            dbc.CardBody(
                html.Div(initial_table, id="portfolio-positions-table"),
                style={"backgroundColor": COLORS["card"], "overflowX": "auto",
                        "maxHeight": "500px", "overflowY": "auto"},
            ),
        ], style={
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
        }, className="mb-4"),

        # Performance vs Benchmarks
        dbc.Card([
            dbc.CardHeader(
                html.H5(t('portfolio.performance_vs_benchmarks'), className="mb-0"),
                style={"backgroundColor": COLORS["card"]},
            ),
            dbc.CardBody(
                dcc.Graph(figure=perf_fig, config={"displayModeBar": False}),
                style={"backgroundColor": COLORS["card"]},
            ),
        ], style={
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
        }),

        # Cash & Allocation Summary
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader([
                        t('portfolio.cash_per_broker'),
                        html.Span(
                            f" — {t('portfolio.cash_alloc_desc')}",
                            style={"color": COLORS["muted"], "fontSize": "0.75rem", "fontWeight": "normal"},
                        ),
                    ], style={"backgroundColor": COLORS["card"]}),
                    dbc.CardBody(
                        dbc.Row([
                            dbc.Col([
                                html.P(t('portfolio.cash'), style={"color": COLORS["muted"], "fontSize": "0.8rem", "marginBottom": "2px"}),
                                html.H5(id="cash-breakdown-value",
                                        style={"color": COLORS["text"], "fontWeight": "bold"}),
                            ], width=4),
                            dbc.Col([
                                html.P(t('portfolio.equity'), style={"color": COLORS["muted"], "fontSize": "0.8rem", "marginBottom": "2px"}),
                                html.H5(id="cash-invested-value",
                                        style={"color": COLORS["accent"], "fontWeight": "bold"}),
                            ], width=4),
                            dbc.Col([
                                html.P(t('portfolio.per_broker'), style={"color": COLORS["muted"], "fontSize": "0.8rem", "marginBottom": "2px"}),
                                html.H5(id="cash-alloc-value",
                                        style={"color": COLORS["orange"], "fontWeight": "bold"}),
                            ], width=4),
                        ]),
                        style={"backgroundColor": COLORS["card"]},
                    ),
                ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}),
                width=12,
            ),
        ], className="mt-4"),

    ], style={"padding": "20px", "backgroundColor": COLORS["bg"]})


# ── Callback Data Helpers ───────────────────────────────────

def _get_portfolio_data() -> dict:
    """Hent portfolio data fra BrokerRouter/PaperBroker."""
    result = {
        "total_value": 0,
        "cash": 0,
        "equity": 0,
        "unrealized_pnl": 0,
        "by_broker": {},
        "by_currency": {},
        "by_asset_type": {},
        "positions": [],
        "broker_accounts": {},
    }

    # Read directly from PaperBroker (primary path for paper trading)
    # AggregatedPortfolio is used when real brokers (Alpaca/IBKR/Saxo) are connected
    try:
        # Try to get the live PaperBroker from the running trader's router
        pb = None
        try:
            from src.broker.registry import get_router
            router = get_router()
            if router:
                pb = router.get_broker("paper") if hasattr(router, "get_broker") else None
                if pb is None:
                    for name, broker in getattr(router, "brokers", {}).items():
                        if "paper" in name.lower():
                            pb = broker
                            break
        except Exception:
            pass

        # Fallback: create new PaperBroker (reads from SQLite)
        if pb is None:
            from src.broker.paper_broker import PaperBroker
            pb = PaperBroker()

        acc = pb.get_account()
        positions = pb.get_positions()

        # Get USD/DKK rate
        usd_dkk = 6.90  # fallback
        try:
            import yfinance as yf
            fx = yf.Ticker("DKK=X")
            rate = getattr(fx.fast_info, "last_price", None)
            if rate and rate > 0:
                usd_dkk = rate
        except Exception:
            pass

        long_val = sum(p.market_value for p in positions if getattr(p, "side", "long") == "long")
        short_val = sum(p.market_value for p in positions if getattr(p, "side", "long") == "short")
        total_equity = long_val - short_val
        total_pnl = sum(p.unrealized_pnl for p in positions)

        # Convert to DKK
        total_value_dkk = acc.equity * usd_dkk
        cash_dkk = acc.cash * usd_dkk
        equity_dkk = total_equity * usd_dkk
        pnl_dkk = total_pnl * usd_dkk

        # Group by currency and exchange based on symbol suffix (values in DKK)
        by_currency = {"USD": 0.0}
        by_asset: dict[str, float] = {}
        by_exchange: dict[str, float] = {}
        for p in positions:
            val = p.market_value * usd_dkk  # all PaperBroker values are in USD
            cls = _asset_class_label(p.symbol)
            by_asset[cls] = by_asset.get(cls, 0) + val
            exch = _exchange_from_symbol(p.symbol)
            by_exchange[exch] = by_exchange.get(exch, 0) + val

            if ".CO" in p.symbol or ".ST" in p.symbol:
                by_currency["DKK/SEK"] = by_currency.get("DKK/SEK", 0) + val
            elif ".HK" in p.symbol:
                by_currency["HKD"] = by_currency.get("HKD", 0) + val
            elif ".AX" in p.symbol:
                by_currency["AUD"] = by_currency.get("AUD", 0) + val
            elif ".L" in p.symbol:
                by_currency["GBP"] = by_currency.get("GBP", 0) + val
            else:
                by_currency["USD"] = by_currency.get("USD", 0) + val

        # Add cash to asset type breakdown
        if cash_dkk > 0:
            by_asset["Cash"] = cash_dkk

        # Remove zero entries
        by_currency = {k: v for k, v in by_currency.items() if v > 0}
        by_asset = {k: v for k, v in by_asset.items() if v > 0}
        by_exchange = {k: v for k, v in by_exchange.items() if v > 0}

        result = {
            "total_value": total_value_dkk,
            "cash": cash_dkk,
            "equity": equity_dkk,
            "unrealized_pnl": pnl_dkk,
            "by_broker": {"Paper": total_value_dkk},
            "by_currency": by_currency,
            "by_asset_type": by_asset,
            "by_exchange": by_exchange,
            "positions": positions,
            "broker_accounts": {"Paper": {"cash": cash_dkk}},
            "usd_dkk": usd_dkk,
        }
    except Exception as exc:
        logger.debug(f"[dashboard] PaperBroker fallback failed: {exc}")

    return result


def _make_pie(data: dict, title: str) -> go.Figure:
    """Generér pie chart med dark theme."""
    if not data:
        # Empty state — show a dim placeholder
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=title, font=dict(color=COLORS["text"], size=14)),
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["card"],
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig.add_annotation(
            text=t('common.no_data'),
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color=COLORS["muted"], size=14),
        )
        return fig

    fig = go.Figure(data=[go.Pie(
        labels=list(data.keys()),
        values=list(data.values()),
        hole=0.5,
        textinfo="label+percent",
        textfont=dict(size=11, color=COLORS["text"]),
        marker=dict(colors=[
            COLORS["accent"], COLORS["blue"], COLORS["orange"],
            COLORS["purple"], COLORS["green"], COLORS["red"],
            "#e91e63", "#00bcd4",
        ]),
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS["text"], size=14)),
        showlegend=False,
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _get_open_markets() -> set[str]:
    """Return set of currently open market keys."""
    try:
        from src.ops.market_calendar import MarketCalendar
        cal = MarketCalendar(include_pre_market=True, include_post_market=True)
        return set(cal.get_open_markets())
    except Exception:
        return set()


def _symbol_market(sym: str) -> str:
    """Map a symbol to its market_calendar key."""
    from src.ops.market_calendar import MARKET_SYMBOLS
    for market, symbols in MARKET_SYMBOLS.items():
        if sym in symbols:
            return market
    # Suffix fallback
    _SUFFIX_MAP = {
        "-USD": "crypto", ".NZ": "new_zealand", ".AX": "australia",
        ".T": "japan", ".HK": "hong_kong", ".NS": "india",
        ".CO": "eu_nordic", ".ST": "eu_nordic", ".OL": "eu_nordic",
        ".HE": "eu_nordic", ".DE": "eu_nordic", ".PA": "eu_nordic",
        ".AS": "eu_nordic", ".SW": "eu_nordic", ".MC": "eu_nordic",
        ".MI": "eu_nordic", ".L": "london", "=F": "chicago",
    }
    for sfx, mkt in _SUFFIX_MAP.items():
        if sym.endswith(sfx):
            return mkt
    return "us_stocks"


# Cache open markets for the duration of one page render / data fetch
_cached_open_markets: set[str] = set()
_cached_open_markets_ts: float = 0


def _is_symbol_trading(sym: str) -> bool:
    """Check if a symbol's market is currently open."""
    global _cached_open_markets, _cached_open_markets_ts
    import time
    now = time.time()
    if now - _cached_open_markets_ts > 60:  # refresh every 60s
        _cached_open_markets = _get_open_markets()
        _cached_open_markets_ts = now
    return _symbol_market(sym) in _cached_open_markets


def _exchange_from_symbol(sym: str) -> str:
    """Derive exchange name from symbol suffix."""
    s = sym.upper()
    if s.endswith(".CO") or s.endswith(".CPH"):
        return "OMXC"
    if s.endswith(".ST"):
        return "OMXS"
    if s.endswith(".HE"):
        return "OMXH"
    if s.endswith(".OL"):
        return "OSE"
    if s.endswith(".DE") or s.endswith(".F"):
        return "XETRA"
    if s.endswith(".L"):
        return "LSE"
    if s.endswith(".PA") or s.endswith(".AS") or s.endswith(".BR"):
        return "Euronext"
    if s.endswith(".AX"):
        return "ASX"
    if s.endswith(".NZ"):
        return "NZX"
    if s.endswith(".T"):
        return "TSE"
    if s.endswith(".HK"):
        return "HKEX"
    if s.endswith(".NS"):
        return "NSE"
    if s.endswith(".TO"):
        return "TSX"
    if s.endswith(".SW"):
        return "SIX"
    if s.endswith(".MI"):
        return "MIL"
    if s.endswith(".MC"):
        return "BME"
    if "-USD" in s or s.startswith("BTC") or s.startswith("ETH"):
        return "Crypto"
    if s.endswith("=F") or s.startswith("/"):
        return "CME"
    return "NYSE/NASDAQ"


def _asset_class_label(sym: str) -> str:
    """Classify a symbol into: Stock, Crypto, Bond, or Commodity."""
    s = sym.upper()
    if "-USD" in s or s.startswith("BTC") or s.startswith("ETH") or s.startswith("SOL") or s.startswith("BNB"):
        return "Crypto"
    _bond_syms = {"TLT", "IEF", "SHY", "BND", "AGG", "LQD", "HYG", "GOVT", "VCIT", "VCSH", "BNDX", "EMB", "TIP", "TIPS"}
    base = s.split(".")[0]
    if base in _bond_syms:
        return "Bond"
    _commodity_syms = {"GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "PPLT", "PALL", "WEAT", "CORN", "SOYB"}
    if base in _commodity_syms or (s.endswith("=F") and s[:2] in {"GC", "SI", "CL", "NG", "HG", "ZC", "ZW", "ZS"}):
        return "Commodity"
    return "Stock"


def _pos_to_dict(pos, usd_dkk: float = 6.90) -> dict:
    """Serialize a position object to a comparable dict.

    All monetary values are stored in DKK.
    """
    pnl_usd = getattr(pos, "unrealized_pnl", 0) or 0
    pnl_dkk = pnl_usd * usd_dkk
    pnl_pct = getattr(pos, "unrealized_pnl_pct", 0) or 0
    mkt_val = getattr(pos, "market_value_dkk", None)
    if mkt_val is None:
        mkt_val = (getattr(pos, "market_value", 0) or 0) * usd_dkk
    sym = getattr(pos, "symbol", "")
    return {
        "symbol": sym,
        "asset_class": _asset_class_label(sym),
        "broker": getattr(pos, "broker_source", "") or "Paper",
        "exchange": getattr(pos, "exchange", "") or _exchange_from_symbol(sym),
        "qty": round(getattr(pos, "qty", 0), 2),
        "entry_price": round(getattr(pos, "entry_price", 0), 2),
        "current_price": round(getattr(pos, "current_price", 0), 2),
        "mkt_val": round(mkt_val, 0),
        "pnl": round(pnl_dkk, 0),
        "pnl_pct": round(pnl_pct * 100, 1),
        "est_tax": round(pnl_dkk * 0.22 if pnl_dkk > 0 else 0, 0),
        "trading": _is_symbol_trading(sym),
        "is_short": getattr(pos, "side", "long") == "short",
    }


def _make_position_row(d: dict) -> html.Tr:
    """Build a single table row. Mutable cells get id='pos-{field}-{symbol}'
    so the clientside JS can update them in-place without Dash re-rendering."""
    sym = d["symbol"]
    safe = sym.replace(".", "_").replace("-", "_").replace("=", "_")
    pnl_color = COLORS["green"] if d["pnl"] >= 0 else COLORS["red"]
    trading = d.get("trading", False)
    status_dot = "🟢" if trading else "🔴"
    row_opacity = "1" if trading else "0.5"
    css_class = "pos-row-trading" if trading else "pos-row-closed"
    _cls_colors = {"Stock": COLORS["blue"], "Crypto": COLORS["purple"],
                    "Bond": COLORS["green"], "Commodity": COLORS["orange"]}
    ac = d.get("asset_class", "Stock")
    return html.Tr(
        [
            html.Td([
                html.Span(status_dot, style={"fontSize": "0.6rem", "marginRight": "4px"}),
                html.Span(sym, style={"color": COLORS["accent"], "fontWeight": "bold"}),
            ]),
            html.Td(ac, style={"color": _cls_colors.get(ac, COLORS["muted"]), "fontSize": "0.8rem", "fontWeight": "600"}),
            html.Td(d.get("exchange", ""), style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
            html.Td(f"{d['qty']:.2f}"),
            html.Td(f"{d['entry_price']:,.2f}"),
            html.Td(f"{d['current_price']:,.2f}", id=f"pos-price-{safe}"),
            html.Td(f"{d['mkt_val']:,.0f}", id=f"pos-val-{safe}"),
            html.Td(f"{d['pnl']:+,.0f}", id=f"pos-pnl-{safe}",
                     style={"color": pnl_color}),
            html.Td(f"{d['pnl_pct']:+.1f}%", id=f"pos-pnlpct-{safe}",
                     style={"color": pnl_color}),
            html.Td(f"{d['est_tax']:,.0f}", id=f"pos-tax-{safe}",
                     style={"color": COLORS["muted"]}),
            html.Td(t('common.short') if d.get("is_short") else t('common.long'),
                     style={"color": COLORS["red"] if d.get("is_short") else COLORS["green"],
                             "fontWeight": "bold"}),
        ],
        className=css_class,
        style={
            "borderBottom": f"1px solid {COLORS['border']}",
            "opacity": row_opacity,
            "display": "" if trading else "none",  # hidden by default
        },
    )


def _pos_table_headers():
    """Return translated position table headers (evaluated at render time)."""
    ccy = get_currency_label()
    value_header = t('portfolio.value_dkk').replace("DKK", ccy)
    return [t('common.symbol'), t('common.type'), t('trading.exchange'), t('common.quantity'), t('portfolio.avg_cost'), t('common.price'),
            value_header, t('portfolio.pnl'), t('portfolio.pnl_pct'), t('portfolio.tax_est'), t('common.side')]


def _make_positions_table(positions: list, show_all: bool = False, usd_dkk: float = 6.90) -> html.Div:
    """Build the full positions table. Trading positions first, closed-market dimmed."""
    headers = _pos_table_headers()
    header_row = html.Thead(html.Tr([
        html.Th(h, style={
            "color": COLORS["muted"],
            "borderBottom": f"1px solid {COLORS['border']}",
            "padding": "8px",
            "fontSize": "0.85rem",
        }) for h in headers
    ]))

    pos_dicts = [_pos_to_dict(p, usd_dkk) for p in positions]
    # Sort: trading positions first, then by P&L descending
    pos_dicts.sort(key=lambda d: (not d.get("trading", False), -abs(d.get("pnl", 0))))

    n_trading = sum(1 for d in pos_dicts if d.get("trading"))
    n_closed = len(pos_dicts) - n_trading

    if pos_dicts:
        rows = [_make_position_row(d) for d in pos_dicts]
        body = html.Tbody(rows, id="portfolio-positions-tbody")
    else:
        body = html.Tbody([
            html.Tr([html.Td(
                t('portfolio.no_positions'),
                colSpan=len(headers),
                style={"color": COLORS["muted"], "textAlign": "center", "padding": "20px"},
            )])
        ], id="portfolio-positions-tbody")

    return html.Table([header_row, body], style={
        "width": "100%",
        "color": COLORS["text"],
        "fontSize": "0.9rem",
    })


# ── Callbacks ───────────────────────────────────────────────

def register_portfolio_callbacks(app: object) -> None:
    """Registrér Dash callbacks for portfolio page."""

    @app.callback(
        Output("download-report-pdf", "data"),
        Input("btn-download-report", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_report(n_clicks):
        """Generate PDF performance report: save to ~/reports/ AND send to browser."""
        if not n_clicks:
            return None
        try:
            from src.dashboard.pages.performance_report import generate_performance_report
            from pathlib import Path
            from datetime import datetime as dt

            pdf_bytes = generate_performance_report()
            filename = f"performance_report_{dt.now().strftime('%Y%m%d_%H%M')}.pdf"

            # Always save to ~/reports/ so it's accessible on the server
            reports_dir = Path.home() / "reports"
            reports_dir.mkdir(exist_ok=True)
            save_path = reports_dir / filename
            save_path.write_bytes(pdf_bytes)
            logger.info(f"[PDF report] Saved to {save_path} ({len(pdf_bytes)} bytes)")

            return dcc.send_bytes(pdf_bytes, filename)
        except Exception as exc:
            logger.error(f"[PDF report] Generation failed: {exc}")
            return None

    # ── Data refresh: fetch portfolio → store only ──────────
    # Server callback ONLY writes data to the store.
    # NO visible components are touched — zero React re-renders.
    # All UI updates happen in the clientside JS callback below.
    @app.callback(
        Output("portfolio-prev-state", "data"),
        Input("portfolio-refresh", "n_intervals"),
        State("portfolio-prev-state", "data"),
    )
    def update_portfolio_data(_n: int, prev_state: dict | None):
        data = _get_portfolio_data()

        # Values from _get_portfolio_data are already in DKK
        total_dkk = data["total_value"]
        pnl_dkk = data["unrealized_pnl"]
        pnl_pct = (pnl_dkk / total_dkk * 100) if total_dkk > 0 else 0

        # Convert to display currency
        total = convert_from_dkk(total_dkk)
        pnl = convert_from_dkk(pnl_dkk)
        cash = convert_from_dkk(data["cash"])
        equity = convert_from_dkk(data["equity"])

        _rate = data.get("usd_dkk", 6.90)
        cur_positions = [_pos_to_dict(p, _rate) for p in data["positions"]]
        # Convert position values from DKK to display currency
        for d in cur_positions:
            d["mkt_val"] = round(convert_from_dkk(d["mkt_val"]), 0)
            d["pnl"] = round(convert_from_dkk(d["pnl"]), 0)
            d["est_tax"] = round(convert_from_dkk(d["est_tax"]), 0)

        cur_kpi = {
            "total": round(total, 0),
            "pnl": round(pnl, 0),
            "pnl_pct": round(pnl_pct, 1),
            "cash": round(cash, 0),
            "equity": round(equity, 0),
            "num_positions": len(data["positions"]),
            "currency_label": get_currency_label(),
            "currency_symbol": get_currency_symbol(),
        }
        cur_pos_map = {d["symbol"]: d for d in cur_positions}
        cur_state = {"kpi": cur_kpi, "positions": cur_pos_map}

        if cur_state == prev_state:
            raise dash.exceptions.PreventUpdate

        return cur_state

    # ── Clientside: update KPIs, timestamp, position cells — all in JS, zero flicker ──
    app.clientside_callback(
        """
        function(curState) {
            if (!curState) return window.dash_clientside.no_update;

            var kpi = curState.kpi || {};
            var positions = curState.positions || {};
            var GREEN = '#2ed573';
            var RED = '#ff4757';

            // Format number with comma thousands (matches Python {:,.0f})
            function fmt(n) {
                return Math.round(n).toString().replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
            }
            function fmtSigned(n) {
                return (n >= 0 ? '+' : '') + fmt(n);
            }

            // Update KPI values — use display currency from server
            var ccy = kpi.currency_label || 'DKK';
            var sym = kpi.currency_symbol || 'kr';
            function fmtCcy(n) {
                var v = fmt(Math.abs(n));
                var sign = n < 0 ? '-' : '';
                if (sym === '$' || sym === '\u20ac') return sign + sym + v;
                return sign + v + ' ' + sym;
            }
            function fmtCcySigned(n) {
                var v = fmt(Math.abs(n));
                var sign = n >= 0 ? '+' : '-';
                if (sym === '$' || sym === '\u20ac') return sign + sym + v;
                return sign + v + ' ' + sym;
            }

            var el;
            el = document.getElementById('kpi-total-value');
            if (el) el.textContent = fmtCcy(kpi.total || 0);

            el = document.getElementById('kpi-pnl-value');
            if (el) el.textContent = fmtCcySigned(kpi.pnl || 0);

            el = document.getElementById('kpi-pnl-change');
            if (el) {
                var pct = kpi.pnl_pct || 0;
                el.textContent = (pct >= 0 ? '+' : '') + pct.toFixed(1) + '%';
                el.style.color = pct >= 0 ? GREEN : RED;
            }

            el = document.getElementById('kpi-cash-value');
            if (el) el.textContent = fmtCcy(kpi.cash || 0);

            el = document.getElementById('kpi-num-positions');
            if (el) el.textContent = (kpi.num_positions || 0).toString();

            el = document.getElementById('kpi-equity-value');
            if (el) el.textContent = fmtCcy(kpi.equity || 0);

            // Update cash & allocation
            el = document.getElementById('cash-breakdown-value');
            if (el) el.textContent = fmtCcy(kpi.cash || 0);

            el = document.getElementById('cash-invested-value');
            if (el) el.textContent = fmtCcy(kpi.equity || 0);

            el = document.getElementById('cash-alloc-value');
            if (el) {
                var total = kpi.total || 1;
                var invested_pct = ((kpi.equity || 0) / total * 100).toFixed(1);
                var cash_pct = ((kpi.cash || 0) / total * 100).toFixed(1);
                el.textContent = invested_pct + '% invested / ' + cash_pct + '% cash';
            }

            // Update timestamp
            el = document.getElementById('portfolio-last-updated');
            if (el) {
                var now = new Date();
                var hh = String(now.getHours()).padStart(2, '0');
                var mm = String(now.getMinutes()).padStart(2, '0');
                var ss = String(now.getSeconds()).padStart(2, '0');
                el.textContent = hh + ':' + mm + ':' + ss;
            }

            // Update position cells — only touch cells whose value changed
            // Also detect structural changes (new/removed positions)
            var tbody = document.getElementById('portfolio-positions-tbody');
            var needsRebuild = false;

            if (tbody) {
                // Check if any new symbols are missing from the DOM
                for (var sym in positions) {
                    var safe = sym.replace(/\\./g, '_').replace(/-/g, '_').replace(/=/g, '_');
                    if (!document.getElementById('pos-price-' + safe)) {
                        needsRebuild = true;
                        break;
                    }
                }
            } else {
                needsRebuild = true;
            }

            if (needsRebuild && tbody) {
                // Rebuild all rows in JS — no server round-trip
                var ACCENT = '#00d4aa';
                var MUTED = '#64748b';
                var BORDER = '#2d3748';
                var btn = document.getElementById('btn-toggle-closed');
                var showAll = btn && btn.textContent.trim() === 'Trading only';
                tbody.innerHTML = '';
                for (var sym in positions) {
                    var d = positions[sym];
                    var safe = sym.replace(/\\./g, '_').replace(/-/g, '_').replace(/=/g, '_');
                    var pnlColor = d.pnl >= 0 ? GREEN : RED;
                    var trading = d.trading;
                    var dot = trading ? '🟢' : '🔴';
                    var tr = document.createElement('tr');
                    tr.className = trading ? 'pos-row-trading' : 'pos-row-closed';
                    tr.style.borderBottom = '1px solid ' + BORDER;
                    tr.style.opacity = trading ? '1' : '0.5';
                    if (!trading && !showAll) tr.style.display = 'none';
                    var acColors = {'Stock':'#3498db','Crypto':'#a855f7','Bond':'#2ed573','Commodity':'#ffa502'};
                    var ac = d.asset_class || 'Stock';
                    var acColor = acColors[ac] || MUTED;
                    tr.innerHTML =
                        '<td><span style="font-size:0.6rem;margin-right:4px">' + dot + '</span><span style="color:' + ACCENT + ';font-weight:bold">' + sym + '</span></td>' +
                        '<td style="color:' + acColor + ';font-size:0.8rem;font-weight:600">' + ac + '</td>' +
                        '<td style="color:' + MUTED + ';font-size:0.8rem">' + (d.exchange || '') + '</td>' +
                        '<td>' + d.qty.toFixed(2) + '</td>' +
                        '<td>' + Number(d.entry_price).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) + '</td>' +
                        '<td id="pos-price-' + safe + '">' + Number(d.current_price).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2}) + '</td>' +
                        '<td id="pos-val-' + safe + '">' + fmt(d.mkt_val) + '</td>' +
                        '<td id="pos-pnl-' + safe + '" style="color:' + pnlColor + '">' + fmtSigned(d.pnl) + '</td>' +
                        '<td id="pos-pnlpct-' + safe + '" style="color:' + pnlColor + '">' + (d.pnl_pct >= 0 ? '+' : '') + d.pnl_pct.toFixed(1) + '%</td>' +
                        '<td id="pos-tax-' + safe + '" style="color:' + MUTED + '">' + fmt(d.est_tax) + '</td>';
                    tbody.appendChild(tr);
                }
            } else {
                // Incremental update — only changed cells
                for (var sym in positions) {
                    var d = positions[sym];
                    var safe = sym.replace(/\\./g, '_').replace(/-/g, '_').replace(/=/g, '_');

                    var priceEl = document.getElementById('pos-price-' + safe);
                    if (!priceEl) continue;

                    var newPrice = Number(d.current_price).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    if (priceEl.textContent === newPrice) continue;  // skip unchanged

                    priceEl.textContent = newPrice;

                    var valEl = document.getElementById('pos-val-' + safe);
                    if (valEl) valEl.textContent = fmt(d.mkt_val);

                    var pnlColor = d.pnl >= 0 ? GREEN : RED;

                    var pnlEl = document.getElementById('pos-pnl-' + safe);
                    if (pnlEl) {
                        pnlEl.textContent = fmtSigned(d.pnl);
                        pnlEl.style.color = pnlColor;
                    }

                    var pnlpctEl = document.getElementById('pos-pnlpct-' + safe);
                    if (pnlpctEl) {
                        pnlpctEl.textContent = (d.pnl_pct >= 0 ? '+' : '') + d.pnl_pct.toFixed(1) + '%';
                        pnlpctEl.style.color = pnlColor;
                    }

                    var taxEl = document.getElementById('pos-tax-' + safe);
                    if (taxEl) taxEl.textContent = fmt(d.est_tax);
                }
            }

            return window.dash_clientside.no_update;
        }
        """,
        Output("portfolio-js-dummy", "children"),
        Input("portfolio-prev-state", "data"),
    )

    # Charts are rendered once in the layout (see _build_initial_charts).
    # No auto-refresh — charts update when user navigates to the page.

    # ── Toggle show/hide closed-market positions ──────────────
    # Use n_clicks parity: odd = showing all, even = trading only
    _show_all_label = t('portfolio.show_all')
    _trading_only_label = t('portfolio.trading_only')
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!n_clicks) return window.dash_clientside.no_update;
            var btn = document.getElementById('btn-toggle-closed');
            var rows = document.querySelectorAll('.pos-row-closed');
            var showAll = (n_clicks % 2 === 1);
            for (var i = 0; i < rows.length; i++) {
                rows[i].style.display = showAll ? '' : 'none';
            }
            btn.textContent = showAll ? '""" + _trading_only_label + """' : '""" + _show_all_label + """';
            return window.dash_clientside.no_update;
        }
        """,
        Output("portfolio-toggle-dummy", "children"),
        Input("btn-toggle-closed", "n_clicks"),
    )

