"""
Alpha Trading Platform – Web Dashboard.

Professionelt Dash-dashboard med 4 sider:
  1. Overblik – portfolio, P&L, positioner, vs. S&P 500
  2. Aktieanalyse – kursudvikling, indikatorer, signaler
  3. Strategier – backtest-resultater, signalhistorik
  4. Risiko – drawdown, exposure, metrics

Start: python -m src.dashboard.app
URL:   http://localhost:8050

Security note (H-15): This dashboard is intended for local/trusted network use only.
All user-facing data is rendered through Dash components (dcc/dbc/html), which auto-escape
content and prevent XSS. Avoid using raw HTML strings (e.g. innerHTML, Markdown with
allow_dangerous_html) for any data originating from external sources (broker APIs, user input).
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pathlib import Path

import dash
from dash import Dash, dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

from loguru import logger
from config.settings import settings
from src.dashboard.i18n import t, set_language, get_language, get_available_languages, get_languages_config
from src.data.market_data import MarketDataFetcher
from src.data.indicators import add_all_indicators, WaveType
from src.strategy.sma_crossover import SMACrossoverStrategy
from src.strategy.rsi_strategy import RSIStrategy
from src.strategy.combined_strategy import CombinedStrategy
from src.backtest.backtester import Backtester
from src.dashboard.currency_service import format_value, get_currency_symbol

# Multi-broker dashboard pages
from src.dashboard.pages.portfolio import portfolio_layout, register_portfolio_callbacks
from src.dashboard.pages.trading import trading_layout, register_trading_callbacks
from src.dashboard.pages.tax_center import tax_center_layout, register_tax_callbacks
from src.dashboard.pages.broker_status import broker_status_layout, register_status_callbacks
from src.dashboard.pages.market_explorer import market_explorer_layout, register_market_callbacks
from src.dashboard.pages.performance_report import generate_performance_report

# ── Konfiguration ─────────────────────────────────────────────

SYMBOLS = settings.trading.symbols
DARK_TEMPLATE = "plotly_dark"
set_language("da")  # Force Danish on startup
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

# ── Data-hentning (cached with TTL + max size) ──────────────────

import gc as _gc
import time as _time
import threading as _threading

_cache: dict = {}
_cache_ts: dict[str, float] = {}   # key -> timestamp of last write
_cache_lock = _threading.Lock()
_CACHE_TTL = 600       # 10 min TTL for stock data
_CACHE_MAX_STOCKS = 20  # max symbols in cache (Rock 5B: 16 GB)

# Background loading state
_preload_done = _threading.Event()
_preload_error: str | None = None


def _evict_cache() -> None:
    """Remove expired and oldest entries to cap memory usage."""
    now = _time.time()
    # Evict ALL expired entries (not just stock_ prefixed ones)
    _KEEP_KEYS = {"mdf"}  # singletons that should never be evicted
    expired = [k for k in _cache if k not in _KEEP_KEYS
               and (now - _cache_ts.get(k, 0)) > _CACHE_TTL * 2]
    for k in expired:
        _cache.pop(k, None)
        _cache_ts.pop(k, None)

    # Enforce max size for stock entries
    stock_keys = [k for k in _cache if k.startswith("stock_")]
    if len(stock_keys) > _CACHE_MAX_STOCKS:
        stock_keys.sort(key=lambda k: _cache_ts.get(k, 0))
        to_remove = stock_keys[:len(stock_keys) - _CACHE_MAX_STOCKS]
        for k in to_remove:
            _cache.pop(k, None)
            _cache_ts.pop(k, None)

    # Clear yfinance global caches to prevent memory leak
    try:
        import yfinance.shared as _yfs
        _yfs._DFS.clear()
        _yfs._ERRORS.clear()
    except Exception:
        pass
    _gc.collect()


def _get_market_data() -> MarketDataFetcher:
    if "mdf" not in _cache:
        _cache["mdf"] = MarketDataFetcher()
    return _cache["mdf"]


def _get_stock_data(symbol: str) -> pd.DataFrame:
    key = f"stock_{symbol}"
    now = _time.time()
    with _cache_lock:
        # Check TTL
        if key in _cache and (now - _cache_ts.get(key, 0)) < _CACHE_TTL:
            return _cache[key]
    # Fetch fresh data (outside lock — network I/O)
    mdf = _get_market_data()
    df = mdf.get_historical(symbol, interval="1d", lookback_days=730)
    if not df.empty:
        df = add_all_indicators(df)
    with _cache_lock:
        _cache[key] = df
        _cache_ts[key] = now
        _evict_cache()
        return _cache[key]


def _get_benchmark() -> pd.DataFrame:
    return _get_stock_data("SPY")


def _run_backtests() -> dict:
    """Run backtests with RAM-optimized settings for Rock 5B (16GB).

    Optimizations vs original:
      - Top 15 liquid symbols instead of all 82
      - 1 year date range instead of 2
      - Sequential strategy runs with gc.collect() between each
    """
    with _cache_lock:
        if "backtests" in _cache:
            return _cache["backtests"]

    # Top 15 most liquid symbols — keeps RAM under ~200 MB
    _BT_SYMBOLS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META",
        "JPM", "V", "UNH", "NOVO-B.CO", "ASML", "BTC-USD", "SPY", "QQQ",
    ]
    mdf = _get_market_data()
    strategies = [
        SMACrossoverStrategy(short_window=20, long_window=50),
        RSIStrategy(oversold=30, overbought=70),
        CombinedStrategy(
            strategies=[
                (SMACrossoverStrategy(short_window=20, long_window=50), 0.6),
                (RSIStrategy(oversold=30, overbought=70), 0.4),
            ],
            min_agreement=1,
        ),
    ]

    results = {}
    for strat in strategies:
        bt = Backtester(
            strategy=strat,
            symbols=_BT_SYMBOLS,
            start="2025-03-01",
            end="2026-03-01",
            commission_pct=0.001,
            spread_pct=0.0005,
            market_data=mdf,
        )
        results[strat.name] = bt.run()
        _gc.collect()

    with _cache_lock:
        _cache["backtests"] = results
    return results


def _preload_worker() -> None:
    """Background thread: fetch benchmark + run backtests so GUI stays responsive."""
    global _preload_error
    try:
        logger.info("Background preload: fetching benchmark data...")
        _get_benchmark()
        logger.info("Background preload: running backtests...")
        _run_backtests()
        logger.info("Background preload: done")
    except Exception as exc:
        import traceback
        _preload_error = str(exc)
        logger.error(f"Background preload failed: {exc}\n{traceback.format_exc()}")
    finally:
        _preload_done.set()


def _start_preload() -> None:
    """Kick off background data loading (call once at import time)."""
    t = _threading.Thread(target=_preload_worker, daemon=True, name="preload")
    t.start()


# Start background preload immediately
_start_preload()


# ── Periodic background refresh (every 5 min) ─────────────
_BG_REFRESH_INTERVAL = 300  # seconds

def _background_refresh_worker():
    """Periodically refresh cached data so pages load instantly."""
    import time as _time
    _time.sleep(60)  # wait 1 min after startup before first refresh
    while True:
        try:
            # Refresh scanner data (used by markedsoverblik, trading quick trades)
            if not _scanner_state.get("loading"):
                try:
                    _fetch_scanner_data_sync()
                    logger.debug("[bg-refresh] Scanner data refreshed")
                except Exception:
                    pass

            # Refresh backtests (used by overblik, risiko, strategier)
            try:
                _cache.pop("backtests", None)
                _run_backtests()
                logger.debug("[bg-refresh] Backtests refreshed")
            except Exception:
                pass

            # Refresh benchmark (used by overblik equity chart)
            try:
                _cache.pop("benchmark", None)
                _get_benchmark()
                logger.debug("[bg-refresh] Benchmark refreshed")
            except Exception:
                pass

        except Exception as exc:
            logger.warning(f"[bg-refresh] Error: {exc}")

        _time.sleep(_BG_REFRESH_INTERVAL)


_bg_refresh_thread = _threading.Thread(target=_background_refresh_worker, daemon=True, name="bg-refresh")
_bg_refresh_thread.start()


# ── Plotly-hjælpere ───────────────────────────────────────────


def _fig_layout(fig: go.Figure, title: str = "", height: int = 400) -> go.Figure:
    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor=COLORS["card"],
        plot_bgcolor=COLORS["card"],
        font=dict(color=COLORS["text"], family="Inter, sans-serif"),
        title=dict(text=title, font_size=16, x=0.02),
        margin=dict(l=50, r=20, t=50, b=40),
        height=height,
        legend=dict(orientation="h", y=-0.15),
        xaxis=dict(gridcolor=COLORS["border"]),
        yaxis=dict(gridcolor=COLORS["border"]),
    )
    return fig


def _metric_card(label: str, value: str, delta: str = "", color: str = "") -> dbc.Card:
    delta_color = COLORS["green"] if delta.startswith("+") else COLORS["red"] if delta.startswith("-") else COLORS["muted"]
    return dbc.Card(
        dbc.CardBody([
            html.P(label, style={"color": COLORS["muted"], "fontSize": "0.8rem", "margin": 0}),
            html.H4(value, style={"color": color or COLORS["text"], "margin": "4px 0"}),
            html.Span(delta, style={"color": delta_color, "fontSize": "0.85rem"}) if delta else html.Span(),
        ]),
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "12px",
        },
    )


# Alias for backward compat
_kpi_card = _metric_card


# ── App-setup ─────────────────────────────────────────────────

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Alpha Trading Platform",
)

# Force dark background on body/html to prevent white flash during page transitions
app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body, html { background-color: #0f1117 !important; }
            #page-content { background-color: #0f1117 !important; }
            /* Dropdown & input text: dark gray for readability */
            .Select-value-label, .Select-placeholder,
            .Select-input input,
            .VirtualizedSelectOption,
            .Select-option,
            .Select-menu-outer,
            .dash-dropdown .Select-value-label,
            .dash-dropdown .Select-menu-outer .VirtualizedSelectOption {
                color: #2d3748 !important;
            }
            .Select-value { color: #2d3748 !important; }
            .Select--single > .Select-control .Select-value { color: #2d3748 !important; }
            .Select-control { background-color: #e2e8f0 !important; border-color: #cbd5e0 !important; }
            .Select-menu-outer { background-color: #e2e8f0 !important; }
            .VirtualizedSelectOption:hover,
            .VirtualizedSelectFocusedOption {
                background-color: #cbd5e0 !important;
                color: #1a202c !important;
            }
            /* All text inputs, number inputs, selects — same dark gray text */
            input.form-control, select.form-select,
            .form-control, .form-select,
            input[type="text"], input[type="number"],
            .dash-input input {
                color: #2d3748 !important;
                background-color: #e2e8f0 !important;
                border-color: #cbd5e0 !important;
            }
            input.form-control::placeholder,
            .form-control::placeholder {
                color: #718096 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>'''

# ── Sidebar / Navigation ─────────────────────────────────────

# Build language flag buttons from languages.json
_available_langs = get_available_languages()
_lang_default = get_languages_config().get("default", "da")

# Map language codes to country codes for flagcdn.com
_FLAG_COUNTRY = {
    "en": "gb", "da": "dk", "fr": "fr", "de": "de",
    "es": "es", "pt": "pt", "vl": "be",
}

def _make_flag_buttons():
    """Build flag button row for language selection."""
    buttons = []
    for lg in _available_langs:
        code = lg["code"]
        country = _FLAG_COUNTRY.get(code, code)
        buttons.append(
            html.Img(
                id={"type": "lang-flag", "code": code},
                src=f"https://flagcdn.com/w40/{country}.png",
                title=lg["name"],
                n_clicks=0,
                style={
                    "width": "24px", "height": "16px",
                    "cursor": "pointer", "borderRadius": "2px",
                    "border": "1px solid transparent",
                    "objectFit": "cover",
                },
                className="me-1",
            )
        )
    return buttons

sidebar = html.Div([
    html.Div([
        html.H5("ALPHA", style={"color": COLORS["accent"], "fontWeight": "800", "letterSpacing": "3px", "margin": 0}),
        html.P("Trading Platform", style={"color": COLORS["muted"], "fontSize": "0.75rem", "margin": 0}),
    ], style={"padding": "20px 16px 12px"}),
    # Language flags
    html.Div(
        _make_flag_buttons(),
        style={"padding": "0 16px 0", "display": "flex", "flexWrap": "wrap", "gap": "2px"},
    ),
    # Hidden store for selected language (drives callbacks)
    dcc.Input(id="lang-select", type="hidden", value=_lang_default),
    html.Hr(style={"borderColor": COLORS["border"], "margin": "8px 16px"}),
    # ── Multi-Broker Trading section ──
    html.Div(id="nav-trading-section"),
    # ── Research & Analysis section ──
    html.Div(id="nav-analysis-section"),
], style={
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "220px",
    "backgroundColor": COLORS["card"],
    "borderRight": f"1px solid {COLORS['border']}",
    "zIndex": 1000,
    "overflowY": "auto",
})

# ── Layout ────────────────────────────────────────────────────

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="lang-store", data=_lang_default, storage_type="local"),
    dcc.Store(id="currency-store", data="DKK", storage_type="local"),
    dcc.Interval(id="auto-refresh", interval=60 * 1000, n_intervals=0),
    dcc.Interval(id="fx-refresh", interval=30 * 60 * 1000, n_intervals=0),  # 30 min FX refresh
    sidebar,
    dcc.Loading(
        html.Div(id="page-content", style={
            "marginLeft": "220px",
            "padding": "24px",
            "backgroundColor": COLORS["bg"],
            "minHeight": "100vh",
            "transition": "none",
        }),
        type="circle",
        color=COLORS["accent"],
        fullscreen=False,
        style={"marginLeft": "220px"},
    ),

    # ── Weekend Crypto Rollover Approval Modal ──
    dcc.Interval(id="weekend-approval-poll", interval=15 * 1000, n_intervals=0),
    dbc.Modal([
        dbc.ModalHeader(
            dbc.ModalTitle([
                html.I(className="bi bi-exclamation-triangle-fill me-2",
                       style={"color": COLORS["orange"]}),
                "Weekend Crypto Rollover",
            ]),
            close_button=False,
            style={"backgroundColor": COLORS["card"], "borderBottom": f"1px solid {COLORS['border']}"},
        ),
        dbc.ModalBody(id="weekend-approval-body",
                      style={"backgroundColor": COLORS["card"]}),
        dbc.ModalFooter([
            dbc.Button([html.I(className="bi bi-x-circle me-2"), "Afvis"],
                       id="btn-weekend-reject", color="danger", outline=True, className="me-2"),
            dbc.Button([html.I(className="bi bi-check-circle me-2"), "Godkend rollover"],
                       id="btn-weekend-approve", color="success"),
        ], style={"backgroundColor": COLORS["card"], "borderTop": f"1px solid {COLORS['border']}"}),
    ], id="weekend-approval-modal", is_open=False, centered=True, size="lg",
       backdrop="static", keyboard=False),
    html.Div(id="weekend-approval-status", style={"display": "none"}),
], style={"backgroundColor": COLORS["bg"]})


# ══════════════════════════════════════════════════════════════
#  SIDE 1 – Overblik
# ══════════════════════════════════════════════════════════════


def _loading_placeholder(title: str) -> html.Div:
    """Placeholder shown while background data is loading."""
    return html.Div([
        html.H3(title, style={"color": COLORS["text"]}),
        dbc.Spinner(color="success", size="lg", spinner_style={"width": "3rem", "height": "3rem"}),
        html.P(t('common.loading') if t('common.loading') != 'common.loading' else "Loading data...",
               style={"color": COLORS["muted"], "marginTop": "16px"}),
    ], style={"textAlign": "center", "paddingTop": "80px"})


def page_overview():
    if not _preload_done.is_set():
        return _loading_placeholder(t('analysis.dashboard_overview'))

    backtests = _run_backtests()
    if not backtests:
        return html.Div([
            html.H3(t('analysis.dashboard_overview'), style={"color": COLORS["text"]}),
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                t('risk.backtests_disabled'),
            ], color="warning", className="mt-3"),
        ])
    best_name = max(backtests, key=lambda k: backtests[k].sharpe_ratio)
    best = backtests[best_name]
    spy = _get_benchmark()

    # Use LIVE portfolio value (not backtest) for the portfolio KPI card
    _live_equity_usd = best.final_equity  # fallback to backtest
    _live_return_pct = best.total_return_pct
    try:
        from src.broker.registry import get_router
        _router = get_router()
        if _router:
            _live_acc = _router.get_account()
            if _live_acc and _live_acc.equity > 0:
                _live_equity_usd = _live_acc.equity
                # Compute return % based on initial capital (typically 100k)
                _initial = getattr(_router, "_initial_capital", 100_000)
                # Try to get from paper broker
                for _bn, _bk in _router._brokers.items():
                    if hasattr(_bk, "_initial_capital"):
                        _initial = _bk._initial_capital
                        break
                if _initial > 0:
                    _live_return_pct = (_live_equity_usd / _initial - 1) * 100
    except Exception:
        pass

    # KPI-kort
    kpi_row = dbc.Row([
        dbc.Col(_metric_card(
            t('analysis.portfolio_value'),
            format_value(_live_equity_usd),
            f"{_live_return_pct:+.2f}%",
            COLORS["accent"],
        ), xs=6, md=3),
        dbc.Col(_metric_card(
            t('analysis.best_strategy'),
            best_name,
            f"Sharpe {best.sharpe_ratio:.2f}",
        ), xs=6, md=3),
        dbc.Col(_metric_card(
            t('analysis.number_of_trades'),
            str(best.num_trades),
            f"Win rate {best.win_rate:.0f}%",
        ), xs=6, md=3),
        dbc.Col(_metric_card(
            t('analysis.max_drawdown'),
            f"{best.max_drawdown_pct:.2f}%",
            "",
            COLORS["red"] if best.max_drawdown_pct > 5 else COLORS["green"],
        ), xs=6, md=3),
    ], className="g-3 mb-4")

    # Equity-kurve vs. S&P 500
    fig_eq = go.Figure()
    if not best.equity_curve.empty:
        eq_norm = best.equity_curve / best.equity_curve.iloc[0] * 100
        fig_eq.add_trace(go.Scatter(
            x=eq_norm.index, y=eq_norm.values, mode="lines", name=best_name,
            line=dict(color=COLORS["accent"], width=2),
        ))
    if not spy.empty:
        spy_norm = spy["Close"] / spy["Close"].iloc[0] * 100
        fig_eq.add_trace(go.Scatter(
            x=spy.index, y=spy_norm.values, mode="lines", name="S&P 500",
            line=dict(color=COLORS["muted"], width=1.5, dash="dot"),
        ))
    _fig_layout(fig_eq, t('charts.portfolio_vs_sp500'), 380)

    # Daglig P&L — aggregate trade P&L by exit date
    fig_pnl = go.Figure()
    if best.trades:
        trade_pnl: dict[str, float] = {}
        for tr in best.trades:
            trade_pnl[tr.exit_date] = trade_pnl.get(tr.exit_date, 0) + tr.net_pnl
        dates = sorted(trade_pnl.keys())
        values = [trade_pnl[d] for d in dates]
        colors = [COLORS["green"] if v >= 0 else COLORS["red"] for v in values]
        fig_pnl.add_trace(go.Bar(
            x=dates, y=values,
            marker_color=colors,
            name=t('charts.daily_pnl_pct'),
        ))
    _fig_layout(fig_pnl, t('charts.daily_returns_pct'), 280)

    # Handelshistorik tabel
    trade_rows = []
    for tr in sorted(best.trades, key=lambda x: x.exit_date, reverse=True)[:10]:
        pnl_color = COLORS["green"] if tr.net_pnl > 0 else COLORS["red"]
        trade_rows.append(html.Tr([
            html.Td(tr.symbol, style={"fontWeight": "600"}),
            html.Td(tr.entry_date),
            html.Td(tr.exit_date),
            html.Td(format_value(tr.entry_price, 2)),
            html.Td(format_value(tr.exit_price, 2)),
            html.Td(format_value(tr.net_pnl, 2), style={"color": pnl_color, "fontWeight": "600"}),
            html.Td(f"{tr.return_pct:+.1f}%", style={"color": pnl_color}),
        ]))

    trades_table = dbc.Table(
        [html.Thead(html.Tr([
            html.Th("Symbol"), html.Th("Entry"), html.Th("Exit"),
            html.Th(t('table.entry')), html.Th(t('table.exit')), html.Th("P&L"), html.Th(t('table.return')),
        ]))] + [html.Tbody(trade_rows)],
        bordered=False, hover=True, responsive=True, size="sm",
        style={"color": COLORS["text"], "backgroundColor": COLORS["card"]},
    )

    return html.Div([
        html.H3(t('analysis.dashboard_overview'), style={"color": COLORS["text"], "marginBottom": "20px"}),
        kpi_row,
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_eq, config={"displayModeBar": False}),
                             style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                                    "border": f"1px solid {COLORS['border']}"}), md=7),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_pnl, config={"displayModeBar": False}),
                             style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                                    "border": f"1px solid {COLORS['border']}"}), md=5),
        ], className="g-3 mb-4"),
        dbc.Card([
            dbc.CardHeader(t('analysis.latest_trades'), style={"backgroundColor": COLORS["card"],
                           "color": COLORS["text"], "borderBottom": f"1px solid {COLORS['border']}"}),
            dbc.CardBody(trades_table),
        ], style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                  "border": f"1px solid {COLORS['border']}"}),
    ])


# ══════════════════════════════════════════════════════════════
#  SIDE 2 – Aktieanalyse
# ══════════════════════════════════════════════════════════════


def page_analyse():
    indicator_toggles = dbc.Card([
        dbc.CardBody([
            html.H6(t('analysis.advanced_indicators'), className="text-light mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Checklist(
                        id="overlay-toggles",
                        options=[
                            {"label": " Ichimoku Cloud", "value": "ichimoku"},
                            {"label": " Fibonacci Niveauer", "value": "fibonacci"},
                            {"label": " Keltner Channels", "value": "keltner"},
                            {"label": " Donchian Channels", "value": "donchian"},
                            {"label": " Volume Profile (POC/VA)", "value": "vol_profile"},
                        ],
                        value=[],
                        inline=False,
                        className="text-light small",
                        input_style={"marginRight": "6px"},
                        label_style={"marginBottom": "4px"},
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Checklist(
                        id="momentum-toggles",
                        options=[
                            {"label": " Stochastic RSI", "value": "stoch_rsi"},
                            {"label": " Williams %R", "value": "williams_r"},
                            {"label": " MFI (Money Flow)", "value": "mfi"},
                            {"label": " CCI", "value": "cci"},
                            {"label": " ADX (Trendstyrke)", "value": "adx"},
                        ],
                        value=[],
                        inline=False,
                        className="text-light small",
                        input_style={"marginRight": "6px"},
                        label_style={"marginBottom": "4px"},
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Checklist(
                        id="volatility-toggles",
                        options=[
                            {"label": " ATR (Volatilitet)", "value": "atr"},
                            {"label": " Historical Volatility", "value": "hv"},
                            {"label": " Elliott Wave", "value": "elliott"},
                        ],
                        value=[],
                        inline=False,
                        className="text-light small",
                        input_style={"marginRight": "6px"},
                        label_style={"marginBottom": "4px"},
                    ),
                ], md=4),
            ]),
        ]),
    ], style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
              "borderRadius": "12px"})

    return html.Div([
        html.H3(t('analysis.stock_analysis'), style={"color": COLORS["text"], "marginBottom": "20px"}),
        dbc.Row([
            dbc.Col(dbc.Select(
                id="stock-select",
                options=[{"label": s, "value": s} for s in SYMBOLS],
                value=SYMBOLS[0],
                style={"backgroundColor": COLORS["card"], "color": COLORS["text"],
                       "border": f"1px solid {COLORS['border']}"},
            ), xs=6, md=3),
        ], className="mb-4"),
        indicator_toggles,
        html.Div(style={"height": "12px"}),
        dbc.Spinner([
            html.Div(id="stock-charts"),
        ], color=COLORS["accent"]),
    ])


@callback(
    Output("stock-charts", "children"),
    Input("stock-select", "value"),
    Input("overlay-toggles", "value"),
    Input("momentum-toggles", "value"),
    Input("volatility-toggles", "value"),
)
def update_stock_charts(symbol: str, overlays: list, momentum: list, volatility: list):
    from src.data.indicators import (
        add_ichimoku, get_ichimoku_signal, add_fibonacci, add_keltner_channels,
        add_donchian_channels, add_volume_profile, add_stochastic_rsi,
        add_williams_r, add_mfi, add_cci, add_adx, add_atr,
        add_historical_volatility, analyze_elliott_waves,
    )

    overlays = overlays or []
    momentum = momentum or []
    volatility = volatility or []

    if not symbol:
        return html.P(t('analysis.select_stock'), style={"color": COLORS["muted"]})

    df = _get_stock_data(symbol)
    if df.empty:
        return html.P(f"{t('analysis.no_data_for')} {symbol}", style={"color": COLORS["red"]})

    # Beregn avancerede indikatorer efter behov
    if "ichimoku" in overlays:
        add_ichimoku(df)
    if "fibonacci" in overlays:
        add_fibonacci(df)
    if "keltner" in overlays:
        add_keltner_channels(df)
    if "donchian" in overlays:
        add_donchian_channels(df)
    if "vol_profile" in overlays:
        add_volume_profile(df)
    if "stoch_rsi" in momentum:
        add_stochastic_rsi(df)
    if "williams_r" in momentum:
        add_williams_r(df)
    if "mfi" in momentum:
        add_mfi(df)
    if "cci" in momentum:
        add_cci(df)
    if "adx" in momentum:
        add_adx(df)
    if "atr" in volatility:
        add_atr(df)
    if "hv" in volatility:
        add_historical_volatility(df)

    # Sidste 252 dage (1 år)
    df_year = df.tail(252)

    # ── Candlestick + SMA ──
    fig_price = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=["", "Volume"],
    )
    fig_price.add_trace(go.Candlestick(
        x=df_year.index, open=df_year["Open"], high=df_year["High"],
        low=df_year["Low"], close=df_year["Close"], name="OHLC",
        increasing_line_color=COLORS["green"],
        decreasing_line_color=COLORS["red"],
    ), row=1, col=1)

    for sma_col, color in [("SMA_20", COLORS["blue"]), ("SMA_50", COLORS["orange"]), ("SMA_200", COLORS["purple"])]:
        if sma_col in df_year.columns:
            fig_price.add_trace(go.Scatter(
                x=df_year.index, y=df_year[sma_col], mode="lines",
                name=sma_col, line=dict(color=color, width=1),
            ), row=1, col=1)

    if "BB_Upper" in df_year.columns:
        fig_price.add_trace(go.Scatter(
            x=df_year.index, y=df_year["BB_Upper"], mode="lines",
            name="BB Upper", line=dict(color=COLORS["muted"], width=0.5, dash="dash"),
            showlegend=False,
        ), row=1, col=1)
        fig_price.add_trace(go.Scatter(
            x=df_year.index, y=df_year["BB_Lower"], mode="lines",
            name="BB Lower", line=dict(color=COLORS["muted"], width=0.5, dash="dash"),
            fill="tonexty", fillcolor="rgba(100,116,139,0.1)", showlegend=False,
        ), row=1, col=1)

    # ── Ichimoku Cloud overlay ──
    if "ichimoku" in overlays and "Ichimoku_SpanA" in df_year.columns:
        fig_price.add_trace(go.Scatter(
            x=df_year.index, y=df_year["Ichimoku_Tenkan"], mode="lines",
            name="Tenkan (9)", line=dict(color="#e74c3c", width=1),
        ), row=1, col=1)
        fig_price.add_trace(go.Scatter(
            x=df_year.index, y=df_year["Ichimoku_Kijun"], mode="lines",
            name="Kijun (26)", line=dict(color="#3498db", width=1),
        ), row=1, col=1)
        fig_price.add_trace(go.Scatter(
            x=df_year.index, y=df_year["Ichimoku_SpanA"], mode="lines",
            name="Span A", line=dict(color="#2ecc71", width=0.5), showlegend=False,
        ), row=1, col=1)
        fig_price.add_trace(go.Scatter(
            x=df_year.index, y=df_year["Ichimoku_SpanB"], mode="lines",
            name="Span B", line=dict(color="#e74c3c", width=0.5),
            fill="tonexty", fillcolor="rgba(46,204,113,0.08)", showlegend=False,
        ), row=1, col=1)

    # ── Fibonacci niveauer ──
    if "fibonacci" in overlays and "Fib_618" in df_year.columns:
        fib_colors = {"Fib_236": "#f1c40f", "Fib_382": "#e67e22", "Fib_500": "#3498db",
                      "Fib_618": "#e74c3c", "Fib_786": "#9b59b6"}
        for col, color in fib_colors.items():
            if col in df_year.columns and df_year[col].notna().any():
                val = df_year[col].iloc[-1]
                fig_price.add_hline(
                    y=val, line_dash="dot", line_color=color, opacity=0.6,
                    annotation_text=f"{col.replace('Fib_', '')}% ({val:.1f})",
                    row=1, col=1,
                )

    # ── Keltner Channels ──
    if "keltner" in overlays and "Keltner_Upper" in df_year.columns:
        fig_price.add_trace(go.Scatter(
            x=df_year.index, y=df_year["Keltner_Upper"], mode="lines",
            name="Keltner Upper", line=dict(color="#f39c12", width=0.7, dash="dot"),
            showlegend=False,
        ), row=1, col=1)
        fig_price.add_trace(go.Scatter(
            x=df_year.index, y=df_year["Keltner_Lower"], mode="lines",
            name="Keltner Lower", line=dict(color="#f39c12", width=0.7, dash="dot"),
            fill="tonexty", fillcolor="rgba(243,156,18,0.06)", showlegend=False,
        ), row=1, col=1)

    # ── Donchian Channels ──
    if "donchian" in overlays and "Donchian_Upper" in df_year.columns:
        fig_price.add_trace(go.Scatter(
            x=df_year.index, y=df_year["Donchian_Upper"], mode="lines",
            name="Donchian Upper", line=dict(color="#1abc9c", width=0.7, dash="dashdot"),
            showlegend=False,
        ), row=1, col=1)
        fig_price.add_trace(go.Scatter(
            x=df_year.index, y=df_year["Donchian_Lower"], mode="lines",
            name="Donchian Lower", line=dict(color="#1abc9c", width=0.7, dash="dashdot"),
            fill="tonexty", fillcolor="rgba(26,188,156,0.06)", showlegend=False,
        ), row=1, col=1)

    # ── Volume Profile (POC + VA) ──
    if "vol_profile" in overlays and "VP_POC" in df_year.columns:
        poc_val = df_year["VP_POC"].iloc[-1]
        va_high = df_year["VP_VA_High"].iloc[-1]
        va_low = df_year["VP_VA_Low"].iloc[-1]
        fig_price.add_hline(y=poc_val, line_dash="solid", line_color="#ff6b6b",
                            line_width=2, opacity=0.8,
                            annotation_text=f"POC ({poc_val:.1f})", row=1, col=1)
        fig_price.add_hrect(y0=va_low, y1=va_high, fillcolor="rgba(255,107,107,0.08)",
                            line_width=0, row=1, col=1,
                            annotation_text=f"Value Area ({va_low:.0f}–{va_high:.0f})")

    vol_colors = [COLORS["green"] if df_year["Close"].iloc[i] >= df_year["Open"].iloc[i]
                  else COLORS["red"] for i in range(len(df_year))]
    fig_price.add_trace(go.Bar(
        x=df_year.index, y=df_year["Volume"], name="Volume",
        marker_color=vol_colors, opacity=0.6, showlegend=False,
    ), row=2, col=1)

    _fig_layout(fig_price, f"{symbol} – {t('analysis.price_chart_title')}", 500)
    fig_price.update_xaxes(rangeslider_visible=False)

    # ── RSI ──
    fig_rsi = go.Figure()
    if "RSI" in df_year.columns:
        fig_rsi.add_trace(go.Scatter(
            x=df_year.index, y=df_year["RSI"], mode="lines",
            name="RSI", line=dict(color=COLORS["accent"], width=1.5),
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color=COLORS["red"], opacity=0.5,
                          annotation_text=f"{t('analysis.overbought')} (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], opacity=0.5,
                          annotation_text=f"{t('analysis.oversold')} (30)")
        fig_rsi.add_hrect(y0=30, y1=70, fillcolor=COLORS["muted"], opacity=0.05)
    _fig_layout(fig_rsi, t('indicators.rsi'), 220)
    fig_rsi.update_yaxes(range=[0, 100])

    # ── MACD ──
    fig_macd = go.Figure()
    if "MACD" in df_year.columns:
        fig_macd.add_trace(go.Scatter(
            x=df_year.index, y=df_year["MACD"], mode="lines",
            name="MACD", line=dict(color=COLORS["blue"], width=1.5),
        ))
        fig_macd.add_trace(go.Scatter(
            x=df_year.index, y=df_year["MACD_Signal"], mode="lines",
            name="Signal", line=dict(color=COLORS["orange"], width=1),
        ))
        hist_colors = [COLORS["green"] if v >= 0 else COLORS["red"]
                       for v in df_year["MACD_Hist"]]
        fig_macd.add_trace(go.Bar(
            x=df_year.index, y=df_year["MACD_Hist"], name="Histogram",
            marker_color=hist_colors, opacity=0.6,
        ))
    _fig_layout(fig_macd, t('indicators.macd'), 220)

    # ── Avancerede momentum-charts ──
    extra_charts = []

    if "stoch_rsi" in momentum and "StochRSI_K" in df_year.columns:
        fig_stoch = go.Figure()
        fig_stoch.add_trace(go.Scatter(
            x=df_year.index, y=df_year["StochRSI_K"], mode="lines",
            name="%K", line=dict(color=COLORS["accent"], width=1.5),
        ))
        fig_stoch.add_trace(go.Scatter(
            x=df_year.index, y=df_year["StochRSI_D"], mode="lines",
            name="%D", line=dict(color=COLORS["orange"], width=1),
        ))
        fig_stoch.add_hline(y=80, line_dash="dash", line_color=COLORS["red"], opacity=0.5)
        fig_stoch.add_hline(y=20, line_dash="dash", line_color=COLORS["green"], opacity=0.5)
        _fig_layout(fig_stoch, t('indicators.stoch_rsi'), 200)
        fig_stoch.update_yaxes(range=[0, 100])
        extra_charts.append(("Stochastic RSI", fig_stoch))

    if "williams_r" in momentum and "Williams_R" in df_year.columns:
        fig_wr = go.Figure()
        fig_wr.add_trace(go.Scatter(
            x=df_year.index, y=df_year["Williams_R"], mode="lines",
            name="Williams %R", line=dict(color="#9b59b6", width=1.5),
        ))
        fig_wr.add_hline(y=-20, line_dash="dash", line_color=COLORS["red"], opacity=0.5,
                         annotation_text=f"{t('analysis.overbought')} (-20)")
        fig_wr.add_hline(y=-80, line_dash="dash", line_color=COLORS["green"], opacity=0.5,
                         annotation_text=f"{t('analysis.oversold')} (-80)")
        _fig_layout(fig_wr, t('indicators.williams_r'), 200)
        fig_wr.update_yaxes(range=[-100, 0])
        extra_charts.append(("Williams %R", fig_wr))

    if "mfi" in momentum and "MFI" in df_year.columns:
        fig_mfi = go.Figure()
        fig_mfi.add_trace(go.Scatter(
            x=df_year.index, y=df_year["MFI"], mode="lines",
            name="MFI", line=dict(color="#e67e22", width=1.5),
        ))
        fig_mfi.add_hline(y=80, line_dash="dash", line_color=COLORS["red"], opacity=0.5)
        fig_mfi.add_hline(y=20, line_dash="dash", line_color=COLORS["green"], opacity=0.5)
        _fig_layout(fig_mfi, t('indicators.mfi'), 200)
        fig_mfi.update_yaxes(range=[0, 100])
        extra_charts.append(("MFI", fig_mfi))

    if "cci" in momentum and "CCI" in df_year.columns:
        fig_cci = go.Figure()
        fig_cci.add_trace(go.Scatter(
            x=df_year.index, y=df_year["CCI"], mode="lines",
            name="CCI", line=dict(color="#1abc9c", width=1.5),
        ))
        fig_cci.add_hline(y=100, line_dash="dash", line_color=COLORS["red"], opacity=0.5)
        fig_cci.add_hline(y=-100, line_dash="dash", line_color=COLORS["green"], opacity=0.5)
        fig_cci.add_hline(y=0, line_dash="solid", line_color=COLORS["muted"], opacity=0.3)
        _fig_layout(fig_cci, t('indicators.cci'), 200)
        extra_charts.append(("CCI", fig_cci))

    if "adx" in momentum and "ADX" in df_year.columns:
        fig_adx = go.Figure()
        fig_adx.add_trace(go.Scatter(
            x=df_year.index, y=df_year["ADX"], mode="lines",
            name="ADX", line=dict(color="#f1c40f", width=2),
        ))
        fig_adx.add_trace(go.Scatter(
            x=df_year.index, y=df_year["Plus_DI"], mode="lines",
            name="+DI", line=dict(color=COLORS["green"], width=1),
        ))
        fig_adx.add_trace(go.Scatter(
            x=df_year.index, y=df_year["Minus_DI"], mode="lines",
            name="-DI", line=dict(color=COLORS["red"], width=1),
        ))
        fig_adx.add_hline(y=25, line_dash="dash", line_color=COLORS["muted"], opacity=0.5,
                          annotation_text=f"{t('analysis.strong_trend')} (25)")
        _fig_layout(fig_adx, t('indicators.adx'), 200)
        extra_charts.append(("ADX", fig_adx))

    # ── Volatilitets-charts ──
    if "atr" in volatility and "ATR" in df_year.columns:
        fig_atr = go.Figure()
        fig_atr.add_trace(go.Scatter(
            x=df_year.index, y=df_year["ATR"], mode="lines",
            name="ATR", line=dict(color="#e74c3c", width=1.5),
        ))
        _fig_layout(fig_atr, t('indicators.atr'), 200)
        extra_charts.append(("ATR", fig_atr))

    if "hv" in volatility and "HV_20" in df_year.columns:
        fig_hv = go.Figure()
        fig_hv.add_trace(go.Scatter(
            x=df_year.index, y=df_year["HV_20"] * 100, mode="lines",
            name="HV 20d", line=dict(color="#3498db", width=1.5),
        ))
        _fig_layout(fig_hv, t('indicators.hist_vol'), 200)
        extra_charts.append(("HV", fig_hv))

    # ── Elliott Wave info ──
    elliott_card = None
    if "elliott" in volatility:
        try:
            ew = analyze_elliott_waves(df)
            ew_color = COLORS["green"] if ew.expected_direction == "up" else \
                       COLORS["red"] if ew.expected_direction == "down" else COLORS["muted"]
            wave_label = ew.wave_type.value.title() if ew.wave_type != WaveType.UNKNOWN else "Ukendt"
            elliott_card = dbc.Card(dbc.CardBody([
                html.H5([
                    html.I(className="bi bi-tsunami me-2"),
                    t('indicators.elliott_wave'),
                ], style={"color": COLORS["text"]}),
                dbc.Row([
                    dbc.Col([
                        html.P(t('indicators.pattern'), style={"color": COLORS["muted"], "fontSize": "0.8rem", "margin": 0}),
                        html.H5(wave_label, style={"color": COLORS["accent"]}),
                    ], md=3),
                    dbc.Col([
                        html.P(t('indicators.current_wave'), style={"color": COLORS["muted"], "fontSize": "0.8rem", "margin": 0}),
                        html.H5(str(ew.current_wave), style={"color": COLORS["text"]}),
                    ], md=3),
                    dbc.Col([
                        html.P(t('indicators.expected_direction'), style={"color": COLORS["muted"], "fontSize": "0.8rem", "margin": 0}),
                        html.H5(
                            "↑ Op" if ew.expected_direction == "up" else
                            "↓ Ned" if ew.expected_direction == "down" else "? Ukendt",
                            style={"color": ew_color},
                        ),
                    ], md=3),
                    dbc.Col([
                        html.P("Confidence", style={"color": COLORS["muted"], "fontSize": "0.8rem", "margin": 0}),
                        html.H5(f"{ew.confidence:.0f}%", style={"color": COLORS["text"]}),
                    ], md=3),
                ]),
                html.P(ew.description, style={"color": COLORS["muted"], "fontSize": "0.85rem", "marginTop": "8px"}),
                html.P([
                    html.I(className="bi bi-exclamation-triangle me-1"),
                    t('indicators.elliott_note'),
                ], style={"color": COLORS["muted"], "fontSize": "0.75rem"}),
            ]), style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                       "border": f"1px solid {COLORS['border']}", "marginBottom": "16px"})
        except Exception:
            pass

    # ── Strategisignaler ──
    strategies = [
        SMACrossoverStrategy(short_window=20, long_window=50),
        RSIStrategy(oversold=30, overbought=70),
    ]
    signal_cards = []
    for strat in strategies:
        try:
            result = strat.analyze(df)
            sig_color = COLORS["green"] if result.signal.value == "BUY" else \
                       COLORS["red"] if result.signal.value == "SELL" else COLORS["muted"]
            signal_cards.append(dbc.Col(dbc.Card(dbc.CardBody([
                html.H6(strat.name, style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                html.H4(result.signal.value, style={"color": sig_color, "fontWeight": "700"}),
                html.P(f"Confidence: {result.confidence:.0f}%",
                       style={"color": COLORS["text"], "fontSize": "0.85rem", "margin": 0}),
                html.P(result.reason, style={"color": COLORS["muted"], "fontSize": "0.75rem", "margin": 0}),
            ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
                       "borderRadius": "12px"}), xs=6, md=3))
        except Exception:
            pass

    # ── Ichimoku signal card ──
    ichimoku_card = None
    if "ichimoku" in overlays and "Ichimoku_Tenkan" in df_year.columns:
        try:
            ichi_sig = get_ichimoku_signal(df)
            ichi_color = COLORS["green"] if ichi_sig.overall == "bullish" else \
                        COLORS["red"] if ichi_sig.overall == "bearish" else COLORS["muted"]
            signal_cards.append(dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Ichimoku", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                html.H4(ichi_sig.overall.upper(), style={"color": ichi_color, "fontWeight": "700"}),
                html.P(f"Kurs: {ichi_sig.price_vs_cloud} cloud",
                       style={"color": COLORS["text"], "fontSize": "0.85rem", "margin": 0}),
                html.P(f"TK: {ichi_sig.tk_cross} | Twist: {'Ja' if ichi_sig.cloud_twist else 'Nej'}",
                       style={"color": COLORS["muted"], "fontSize": "0.75rem", "margin": 0}),
            ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
                       "borderRadius": "12px"}), xs=6, md=3))
        except Exception:
            pass

    # ── Build extra chart rows ──
    extra_rows = []
    for i in range(0, len(extra_charts), 2):
        cols = []
        for j in range(2):
            if i + j < len(extra_charts):
                name, fig = extra_charts[i + j]
                cols.append(dbc.Col(dbc.Card(
                    dcc.Graph(figure=fig, config={"displayModeBar": False}),
                    style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                           "border": f"1px solid {COLORS['border']}"},
                ), md=6))
        extra_rows.append(dbc.Row(cols, className="g-3 mt-3"))

    children = [
        dbc.Row(signal_cards, className="g-3 mb-4"),
    ]
    if elliott_card:
        children.append(elliott_card)
    children.append(
        dbc.Card(dcc.Graph(figure=fig_price, config={"displayModeBar": False}),
                 style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                        "border": f"1px solid {COLORS['border']}", "marginBottom": "16px"}),
    )
    children.append(
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_rsi, config={"displayModeBar": False}),
                             style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                                    "border": f"1px solid {COLORS['border']}"}), md=6),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_macd, config={"displayModeBar": False}),
                             style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                                    "border": f"1px solid {COLORS['border']}"}), md=6),
        ], className="g-3"),
    )
    children.extend(extra_rows)

    return html.Div(children)


# ══════════════════════════════════════════════════════════════
#  SIDE 3 – Strategier
# ══════════════════════════════════════════════════════════════


def page_strategies():
    import sqlite3
    from pathlib import Path as _P

    # ── Load live data ──────────────────────────────────────
    # Active strategies from AutoTrader
    active_strategies = []
    try:
        from src.broker.registry import get_auto_trader
        trader = get_auto_trader()
        if trader and hasattr(trader, "_engine"):
            for strat, weight in trader._engine._strategies:
                active_strategies.append({
                    "name": getattr(strat, "name", type(strat).__name__),
                    "weight": weight,
                    "enabled": True,
                })
    except Exception:
        pass
    if not active_strategies:
        # Fallback: show default strategies
        active_strategies = [
            {"name": "RSI Strategy", "weight": 0.30, "enabled": True},
            {"name": "SMA Crossover", "weight": 0.30, "enabled": True},
            {"name": "Combined Strategy", "weight": 0.40, "enabled": True},
        ]

    # Closed trades from portfolio DB
    closed_trades = []
    try:
        db = _P("data_cache/paper_portfolio.db")
        if db.exists():
            with sqlite3.connect(db) as conn:
                conn.row_factory = sqlite3.Row
                closed_trades = conn.execute(
                    "SELECT * FROM closed_trades ORDER BY exit_time DESC"
                ).fetchall()
    except Exception:
        pass

    # Equity history
    equity_data = []
    try:
        db = _P("data_cache/paper_portfolio.db")
        if db.exists():
            with sqlite3.connect(db) as conn:
                equity_data = conn.execute(
                    "SELECT timestamp, equity FROM equity_history ORDER BY id"
                ).fetchall()
    except Exception:
        pass

    # ── Trade statistics ────────────────────────────────────
    total_trades = len(closed_trades)
    winners = [t for t in closed_trades if t["realized_pnl"] > 0]
    losers = [t for t in closed_trades if t["realized_pnl"] <= 0]
    realized_pnl = sum(t["realized_pnl"] for t in closed_trades)
    win_rate = len(winners) / total_trades * 100 if total_trades else 0
    avg_win = sum(t["realized_pnl"] for t in winners) / len(winners) if winners else 0
    avg_loss = sum(abs(t["realized_pnl"]) for t in losers) / len(losers) if losers else 0
    gross_wins = sum(t["realized_pnl"] for t in winners)
    gross_losses = abs(sum(t["realized_pnl"] for t in losers))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Include unrealized P&L from open positions for total P&L
    unrealized_pnl = 0.0
    try:
        from src.broker.registry import get_router
        _router = get_router()
        if _router:
            for pos in _router.get_positions():
                unrealized_pnl += getattr(pos, "unrealized_pnl", 0) or 0
    except Exception:
        pass
    total_pnl = realized_pnl + unrealized_pnl

    # ── Strategy weights table ──────────────────────────────
    strat_rows = []
    for s in active_strategies:
        strat_rows.append(html.Tr([
            html.Td(s["name"], style={"fontWeight": "600", "color": COLORS["accent"]}),
            html.Td(f"{s['weight']*100:.0f}%"),
            html.Td("🟢 Active" if s["enabled"] else "🔴 Disabled",
                     style={"color": COLORS["green"] if s["enabled"] else COLORS["red"]}),
        ]))

    strat_table = dbc.Table(
        [html.Thead(html.Tr([
            html.Th(t('table.strategy')), html.Th(t('common.weight')), html.Th(t('common.status')),
        ]))] + [html.Tbody(strat_rows)],
        bordered=False, hover=True, size="sm",
        style={"color": COLORS["text"]},
    )

    # ── KPI cards ───────────────────────────────────────────
    pnl_color = COLORS["green"] if total_pnl >= 0 else COLORS["red"]
    # Format P&L values in the user's display currency
    _pnl_sign = "+" if total_pnl >= 0 else ""
    _win_sign = "+" if avg_win >= 0 else ""
    kpi_row = dbc.Row([
        dbc.Col(_kpi_card(t('analysis.number_of_trades'), str(total_trades)), width=2),
        dbc.Col(_kpi_card("Win Rate", f"{win_rate:.1f}%"), width=2),
        dbc.Col(_kpi_card("Total P&L", f"{_pnl_sign}{format_value(total_pnl)}"), width=2),
        dbc.Col(_kpi_card("Avg Win", f"{_win_sign}{format_value(avg_win)}"), width=2),
        dbc.Col(_kpi_card("Avg Loss", format_value(avg_loss)), width=2),
        dbc.Col(_kpi_card("Profit Factor", f"{profit_factor:.2f}" if profit_factor != float("inf") else "∞"), width=2),
    ], className="mb-4")

    # ── Equity curve ────────────────────────────────────────
    fig_eq = go.Figure()
    if equity_data:
        first_eq = equity_data[0][1]
        dates = [r[0][:16] for r in equity_data]
        returns = [(r[1] / first_eq - 1) * 100 for r in equity_data]
        fig_eq.add_trace(go.Scatter(
            x=dates, y=returns, mode="lines", name=t('charts.portfolio_return'),
            line=dict(color=COLORS["accent"], width=2),
            fill="tozeroy", fillcolor="rgba(0,212,170,0.1)",
        ))
        fig_eq.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], line_width=1)
    _fig_layout(fig_eq, "Equity Curve (% return)", 350)

    # ── P&L per trade (waterfall) ───────────────────────────
    fig_pnl = go.Figure()
    if closed_trades:
        recent = list(reversed(closed_trades[:30]))  # oldest first
        syms = [t["symbol"] for t in recent]
        pnls = [t["realized_pnl"] for t in recent]
        colors = [COLORS["green"] if p > 0 else COLORS["red"] for p in pnls]
        fig_pnl.add_trace(go.Bar(
            x=syms, y=pnls, marker_color=colors,
            text=[f"${p:+,.0f}" for p in pnls],
            textposition="outside", textfont=dict(size=9),
        ))
    _fig_layout(fig_pnl, "P&L per Trade (last 30)", 320)

    # ── Strategy weight pie ─────────────────────────────────
    fig_pie = go.Figure(data=[go.Pie(
        labels=[s["name"] for s in active_strategies],
        values=[s["weight"] for s in active_strategies],
        hole=0.5, textinfo="label+percent",
        textfont=dict(size=11, color=COLORS["text"]),
        marker=dict(colors=[COLORS["accent"], COLORS["blue"], COLORS["orange"],
                            COLORS["purple"], COLORS["green"]]),
    )])
    _fig_layout(fig_pie, t('analysis.strategy_weights'), 300)

    return html.Div([
        html.H3(f"{t('analysis.strategies')} — {t('analysis.live_performance')}", style={"color": COLORS["text"], "marginBottom": "20px"}),

        kpi_row,

        # Active strategies table
        dbc.Card([
            dbc.CardHeader(t('analysis.active_strategies_header'),
                           style={"backgroundColor": COLORS["card"], "color": COLORS["text"],
                                  "borderBottom": f"1px solid {COLORS['border']}"}),
            dbc.CardBody(strat_table),
        ], style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                  "border": f"1px solid {COLORS['border']}", "marginBottom": "20px"}),

        # Charts
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_eq, config={"displayModeBar": False}),
                             style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                                    "border": f"1px solid {COLORS['border']}"}), md=7),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_pie, config={"displayModeBar": False}),
                             style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                                    "border": f"1px solid {COLORS['border']}"}), md=5),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_pnl, config={"displayModeBar": False}),
                             style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                                    "border": f"1px solid {COLORS['border']}"}), md=12),
        ], className="g-3"),
    ])


# ══════════════════════════════════════════════════════════════
#  SIDE 4 – Risiko
# ══════════════════════════════════════════════════════════════


def page_risk():
    backtests = _run_backtests()
    if not backtests:
        return html.Div([
            html.H3(t('analysis.risk_management'), style={"color": COLORS["text"], "marginBottom": "20px"}),
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                t('risk.backtests_disabled'),
            ], color="warning"),
        ], style={"padding": "40px"})
    best_name = max(backtests, key=lambda k: backtests[k].sharpe_ratio)
    best = backtests[best_name]

    # KPI-kort
    kpi_row = dbc.Row([
        dbc.Col(_metric_card(t('analysis.max_drawdown'), f"{best.max_drawdown_pct:.2f}%", "",
                             COLORS["red"] if best.max_drawdown_pct > 5 else COLORS["green"]), xs=6, md=3),
        dbc.Col(_metric_card("Sharpe Ratio", f"{best.sharpe_ratio:.2f}", "",
                             COLORS["green"] if best.sharpe_ratio > 1 else COLORS["orange"]), xs=6, md=3),
        dbc.Col(_metric_card("Sortino Ratio", f"{best.sortino_ratio:.2f}", "",
                             COLORS["green"] if best.sortino_ratio > 1 else COLORS["orange"]), xs=6, md=3),
        dbc.Col(_metric_card("Calmar Ratio", f"{best.calmar_ratio:.2f}", "",
                             COLORS["green"] if best.calmar_ratio > 1 else COLORS["orange"]), xs=6, md=3),
    ], className="g-3 mb-4")

    # Drawdown-chart
    fig_dd = go.Figure()
    if not best.equity_curve.empty:
        peak = best.equity_curve.expanding().max()
        dd = ((best.equity_curve - peak) / peak) * 100
        fig_dd.add_trace(go.Scatter(
            y=dd.values, mode="lines", name="Drawdown",
            line=dict(color=COLORS["red"], width=1.5),
            fill="tozeroy", fillcolor="rgba(255,71,87,0.15)",
        ))
    _fig_layout(fig_dd, f"Drawdown (%) – {best_name}", 320)
    fig_dd.update_yaxes(title=t('charts.drawdown_pct'))

    # Risk per strategi
    risk_rows = []
    for name, r in backtests.items():
        dd_color = COLORS["green"] if r.max_drawdown_pct < 5 else COLORS["orange"] if r.max_drawdown_pct < 10 else COLORS["red"]
        risk_rows.append(html.Tr([
            html.Td(name, style={"fontWeight": "600"}),
            html.Td(f"{r.max_drawdown_pct:.2f}%", style={"color": dd_color}),
            html.Td(f"{r.sharpe_ratio:.2f}"),
            html.Td(f"{r.sortino_ratio:.2f}"),
            html.Td(f"{r.calmar_ratio:.2f}"),
            html.Td(format_value(r.total_commission, 2)),
            html.Td(format_value(r.avg_loss, 2) if r.avg_loss else "–"),
        ]))

    risk_table = dbc.Table(
        [html.Thead(html.Tr([
            html.Th(t('table.strategy')), html.Th("Max DD"), html.Th(t('table.sharpe')),
            html.Th(t('table.sortino')), html.Th(t('table.calmar')), html.Th(t('table.commission')),
            html.Th(t('table.avg_loss')),
        ]))] + [html.Tbody(risk_rows)],
        bordered=False, hover=True, responsive=True, size="sm",
        style={"color": COLORS["text"]},
    )

    # Trade P&L distribution
    fig_dist = go.Figure()
    if best.trades:
        pnls = [t.net_pnl for t in best.trades]
        fig_dist.add_trace(go.Histogram(
            x=pnls, nbinsx=20, name=t('charts.pnl_distribution'),
            marker_color=COLORS["accent"], opacity=0.7,
        ))
        fig_dist.add_vline(x=0, line_dash="dash", line_color=COLORS["muted"])
    _fig_layout(fig_dist, f"Trade P&L fordeling ({get_currency_symbol()})", 320)
    fig_dist.update_xaxes(title=t('charts.profit_loss'))
    fig_dist.update_yaxes(title=t('charts.trade_count'))

    # Position exposure over tid (per symbol)
    fig_exposure = go.Figure()
    if best.trades:
        for sym in SYMBOLS:
            sym_trades = [t for t in best.trades if t.symbol == sym]
            if sym_trades:
                pnl_total = sum(t.net_pnl for t in sym_trades)
                fig_exposure.add_trace(go.Bar(
                    x=[sym], y=[pnl_total],
                    name=sym,
                    marker_color=COLORS["green"] if pnl_total > 0 else COLORS["red"],
                    text=[format_value(pnl_total)],
                    textposition="outside",
                ))
    _fig_layout(fig_exposure, f"P&L per aktie ({get_currency_symbol()})", 320)
    fig_exposure.update_layout(showlegend=False)

    # Rolling Sharpe
    fig_rolling = go.Figure()
    if not best.daily_returns.empty and len(best.daily_returns) > 60:
        rolling_mean = best.daily_returns.rolling(60).mean()
        rolling_std = best.daily_returns.rolling(60).std()
        rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
        fig_rolling.add_trace(go.Scatter(
            y=rolling_sharpe.values, mode="lines", name=t('charts.rolling_sharpe'),
            line=dict(color=COLORS["purple"], width=1.5),
        ))
        fig_rolling.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"])
        fig_rolling.add_hline(y=1, line_dash="dot", line_color=COLORS["green"], opacity=0.5,
                              annotation_text="Sharpe = 1")
    _fig_layout(fig_rolling, "Rullende Sharpe Ratio (60 dage)", 320)

    return html.Div([
        html.H3(t('analysis.risk_management'), style={"color": COLORS["text"], "marginBottom": "20px"}),
        kpi_row,
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_dd, config={"displayModeBar": False}),
                             style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                                    "border": f"1px solid {COLORS['border']}"}), md=7),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_exposure, config={"displayModeBar": False}),
                             style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                                    "border": f"1px solid {COLORS['border']}"}), md=5),
        ], className="g-3 mb-4"),
        dbc.Card([
            dbc.CardHeader(t('analysis.risk_metrics_per_strategy'),
                           style={"backgroundColor": COLORS["card"], "color": COLORS["text"],
                                  "borderBottom": f"1px solid {COLORS['border']}"}),
            dbc.CardBody(risk_table),
        ], style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                  "border": f"1px solid {COLORS['border']}", "marginBottom": "20px"}),
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_dist, config={"displayModeBar": False}),
                             style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                                    "border": f"1px solid {COLORS['border']}"}), md=6),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_rolling, config={"displayModeBar": False}),
                             style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                                    "border": f"1px solid {COLORS['border']}"}), md=6),
        ], className="g-3 mb-4"),

        # Circuit Breakers sektion
        dbc.Card([
            dbc.CardHeader([
                html.I(className="bi bi-lightning-charge me-2"),
                t('analysis.circuit_breakers'),
            ], style={"backgroundColor": COLORS["card"], "color": COLORS["text"],
                       "borderBottom": f"1px solid {COLORS['border']}"}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6(t('analysis.circuit_breaker_levels'), className="text-muted mb-3"),
                        html.Div([
                            html.Div([
                                dbc.Badge("DAILY", color="warning", className="me-2",
                                          style={"width": "80px"}),
                                html.Span(
                                    "3% tab paa 1 dag → stop nye handler resten af dagen",
                                    style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                                ),
                            ], className="mb-2"),
                            html.Div([
                                dbc.Badge("WEEKLY", color="danger", className="me-2",
                                          style={"width": "80px"}),
                                html.Span(
                                    "7% tab paa 1 uge → stop alt i 48 timer + alert",
                                    style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                                ),
                            ], className="mb-2"),
                            html.Div([
                                dbc.Badge("CRITICAL", style={"backgroundColor": "#ff0000",
                                           "width": "80px"}, className="me-2"),
                                html.Span(
                                    "15% fra peak → STOP ALT, kraev manuel genstart",
                                    style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                                ),
                            ], className="mb-2"),
                        ]),
                    ], md=4),
                    dbc.Col([
                        html.H6(t('analysis.regime_adaptive_risk'), className="text-muted mb-3"),
                        dbc.Table([
                            html.Thead(html.Tr([
                                html.Th(t('table.parameter'), style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                                html.Th(t('table.bull'), style={"color": "#2ed573", "fontSize": "0.8rem"}),
                                html.Th(t('table.sideways'), style={"color": "#ffa502", "fontSize": "0.8rem"}),
                                html.Th(t('table.bear'), style={"color": "#ff4757", "fontSize": "0.8rem"}),
                                html.Th(t('table.crash'), style={"color": "#ff0000", "fontSize": "0.8rem"}),
                            ])),
                            html.Tbody([
                                html.Tr([html.Td(t('risk.max_position'), style={"fontSize": "0.8rem"}),
                                         html.Td("5%"), html.Td("3%"), html.Td("2%"), html.Td("1%")]),
                                html.Tr([html.Td(t('risk.max_daily_loss'), style={"fontSize": "0.8rem"}),
                                         html.Td("3%"), html.Td("2%"), html.Td("1%"), html.Td("0.5%")]),
                                html.Tr([html.Td(t('risk.max_positions'), style={"fontSize": "0.8rem"}),
                                         html.Td("15"), html.Td("10"), html.Td("5"), html.Td("2")]),
                                html.Tr([html.Td(t('risk.stop_loss'), style={"fontSize": "0.8rem"}),
                                         html.Td("8%"), html.Td("5%"), html.Td("3%"), html.Td("2%")]),
                                html.Tr([html.Td(t('risk.max_exposure'), style={"fontSize": "0.8rem"}),
                                         html.Td("95%"), html.Td("60%"), html.Td("30%"), html.Td("10%")]),
                                html.Tr([html.Td(t('risk.cash_minimum'), style={"fontSize": "0.8rem"}),
                                         html.Td("5%"), html.Td("40%"), html.Td("70%"), html.Td("90%")]),
                            ]),
                        ], bordered=False, hover=True, responsive=True, size="sm",
                           className="table-dark mb-0", style={"fontSize": "0.85rem"}),
                    ], md=5),
                    dbc.Col([
                        html.H6(t('analysis.current_status'), className="text-muted mb-3"),
                        html.Div([
                            html.Div([
                                html.I(className="bi bi-check-circle-fill me-2",
                                       style={"color": COLORS["green"]}),
                                html.Span("Circuit Breaker: ",
                                          style={"color": COLORS["text"], "fontWeight": "600"}),
                                dbc.Badge("INGEN AKTIVE", color="success"),
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="bi bi-arrow-left-right me-2",
                                       style={"color": COLORS["blue"]}),
                                html.Span(f"{t('risk.transition')}: ",
                                          style={"color": COLORS["text"], "fontWeight": "600"}),
                                html.Span(t('risk.gradual_transition'),
                                          style={"color": COLORS["muted"]}),
                            ], className="mb-2"),
                            html.Div([
                                html.I(className="bi bi-lightning me-2",
                                       style={"color": COLORS["red"]}),
                                html.Span(f"{t('risk.crash_label')}: ",
                                          style={"color": COLORS["text"], "fontWeight": "600"}),
                                html.Span(t('risk.immediate_transition'),
                                          style={"color": COLORS["muted"]}),
                            ], className="mb-2"),
                        ]),
                    ], md=3),
                ]),
            ]),
        ], style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                  "border": f"1px solid {COLORS['border']}"}),
    ])


# ══════════════════════════════════════════════════════════════
#  SIDE 5 – Skat
# ══════════════════════════════════════════════════════════════


def _build_tax_transactions(trades, fx_rate):
    """Konverter backtest-trades til transaktionsliste for skatteberegning."""
    transactions = []
    for tr in trades:
        entry_date = tr.entry_date[:10] if tr.entry_date else "2026-01-15"
        exit_date = tr.exit_date[:10] if tr.exit_date else "2026-06-15"
        entry_dkk = tr.entry_price * tr.qty * fx_rate
        exit_dkk = tr.exit_price * tr.qty * fx_rate
        pnl_dkk = exit_dkk - entry_dkk
        transactions.append({
            "symbol": tr.symbol,
            "qty": tr.qty,
            "entry_value_dkk": entry_dkk,
            "exit_value_dkk": exit_dkk,
            "entry_date": entry_date,
            "trade_date": exit_date,
            "realized_pnl_dkk": pnl_dkk,
        })
    return transactions


def _card(header, body_content):
    """Hjælper: opret et stylet dashboard-kort."""
    return dbc.Card([
        dbc.CardHeader(header,
                       style={"backgroundColor": COLORS["card"], "color": COLORS["text"],
                              "borderBottom": f"1px solid {COLORS['border']}"}),
        dbc.CardBody(body_content),
    ], style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
              "border": f"1px solid {COLORS['border']}"})


def page_tax():
    from src.tax.tax_calculator import DanishTaxCalculator
    from src.tax.tax_advisor import TaxAdvisor
    from datetime import datetime

    year = 2026
    progression_limit = settings.tax.progression_limit
    fx_rate = settings.tax.fallback_fx_rate

    # ── Hent backtest-data ──
    backtests = _run_backtests()
    if not backtests:
        return html.Div([
            html.H3(t('analysis.tax_reporting'), style={"color": COLORS["text"]}),
            dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                t('risk.backtests_disabled'),
            ], color="warning", className="mt-3"),
            html.A(f"{t('nav.tax_center')} →", href="/tax", style={"color": COLORS["accent"]}),
        ])
    best_name = max(backtests, key=lambda k: backtests[k].sharpe_ratio)
    best = backtests[best_name]

    # ── Opret calculator & advisor ──
    calculator = DanishTaxCalculator(
        progression_limit=progression_limit,
        carried_losses=settings.tax.carried_losses,
    )
    advisor = TaxAdvisor(
        calculator=calculator,
        progression_limit=progression_limit,
        carried_losses=settings.tax.carried_losses,
        fx_rate=fx_rate,
    )

    # ── Transaktioner ──
    transactions = _build_tax_transactions(best.trades, fx_rate)
    tax = calculator.calculate(transactions=transactions, year=year)

    # ── Kvartalsestimat ──
    q_est = advisor.quarterly_estimate(transactions, year=year)

    # ── Advarsler ──
    # Tax alerts contain Danish-specific SKAT deadlines/reminders — only show in Danish
    alerts = advisor.collect_pending_alerts(transactions, year=year)
    if get_language() != "da":
        alerts = [a for a in alerts if a.category != "deadline"]

    # ═══════════════════════════════════════════════════════════
    #  LAYOUT
    # ═══════════════════════════════════════════════════════════

    # Disclaimer-banner
    disclaimer = dbc.Alert([
        html.Strong(f"⚠️ {t('tax.disclaimer_banner')} – "),
        t('tax.disclaimer_text'),
    ], color="warning", style={"backgroundColor": "#332b00", "color": "#ffd700",
                                "border": "1px solid #665500", "borderRadius": "12px"})

    # ── Alert-banner (proaktive advarsler) ──
    alert_elements = []
    for alert in alerts:
        color_map = {"CRITICAL": "danger", "WARNING": "warning", "INFO": "info"}
        bg_map = {"CRITICAL": "#3d0000", "WARNING": "#332b00", "INFO": "#002233"}
        fg_map = {"CRITICAL": "#ff6b6b", "WARNING": "#ffd700", "INFO": "#74b9ff"}
        border_map = {"CRITICAL": "#660000", "WARNING": "#665500", "INFO": "#004466"}
        alert_elements.append(dbc.Alert([
            html.Strong(f"{'🔴' if alert.severity == 'CRITICAL' else '🟡' if alert.severity == 'WARNING' else '🔵'} {alert.title} – "),
            alert.message[:200] + ("..." if len(alert.message) > 200 else ""),
        ], color=color_map.get(alert.severity, "info"),
            style={"backgroundColor": bg_map.get(alert.severity, "#002233"),
                   "color": fg_map.get(alert.severity, "#74b9ff"),
                   "border": f"1px solid {border_map.get(alert.severity, '#004466')}",
                   "borderRadius": "12px", "fontSize": "0.9rem"}))

    # ── KPI-kort (udvidet med projektion) ──
    pct_of_limit = (tax.taxable_gain_dkk / progression_limit * 100) if progression_limit > 0 else 0
    near_limit = pct_of_limit > 75

    kpi_row = dbc.Row([
        dbc.Col(_metric_card(
            t('tax.net_gain_loss'),
            f"{tax.net_gain_dkk:+,.0f} DKK",
            f"{tax.net_gain_dkk / fx_rate:+,.0f} USD",
            COLORS["green"] if tax.net_gain_dkk > 0 else COLORS["red"],
        ), xs=6, md=3),
        dbc.Col(_metric_card(
            t('risk.estimated_tax'),
            f"{tax.total_tax_dkk:,.0f} DKK",
            f"Effektiv sats: {(tax.total_tax_dkk / tax.taxable_gain_dkk * 100):.1f}%" if tax.taxable_gain_dkk > 0 else "",
            COLORS["orange"],
        ), xs=6, md=3),
        dbc.Col(_metric_card(
            t('risk.progression_limit'),
            f"{pct_of_limit:.0f}% {t('tax.pct_used')}",
            ('⚠️ ' + t('tax.close_to_limit')) if near_limit else f"{progression_limit - tax.taxable_gain_dkk:,.0f} {t('tax.dkk_remaining')}",
            COLORS["red"] if near_limit else COLORS["green"],
        ), xs=6, md=3),
        dbc.Col(_metric_card(
            t('tax.expected_annual_tax'),
            f"{q_est.projected_annual_tax_dkk:,.0f} DKK",
            f"{t('tax.projected_gain')}: {q_est.projected_annual_gain_dkk:+,.0f} DKK",
            COLORS["accent"],
        ), xs=6, md=3),
    ], className="g-3 mb-4")

    # ── Rubrik-tabel ──
    rubrik_rows = [
        html.Tr([
            html.Td("Rubrik 66", style={"fontWeight": "600"}),
            html.Td(t('tax.rubrik_66_desc')),
            html.Td(f"{tax.rubrik_66:,.2f} DKK", style={"textAlign": "right",
                     "color": COLORS["green"] if tax.rubrik_66 > 0 else COLORS["text"]}),
        ]),
        html.Tr([
            html.Td("Rubrik 67", style={"fontWeight": "600"}),
            html.Td(t('tax.rubrik_67_desc')),
            html.Td(f"{tax.rubrik_67:,.2f} DKK", style={"textAlign": "right",
                     "color": COLORS["red"] if tax.rubrik_67 > 0 else COLORS["text"]}),
        ]),
        html.Tr([
            html.Td("Rubrik 68", style={"fontWeight": "600"}),
            html.Td(t('tax.rubrik_68_desc')),
            html.Td(f"{tax.rubrik_68:,.2f} DKK", style={"textAlign": "right"}),
        ]),
    ]
    rubrik_table = dbc.Table(
        [html.Thead(html.Tr([html.Th(t('tax.rubrik')), html.Th(t('tax.description')), html.Th(t('tax.amount'), style={"textAlign": "right"})]))]
        + [html.Tbody(rubrik_rows)],
        bordered=False, hover=True, size="sm",
        style={"color": COLORS["text"]},
    )

    # ── Skatteberegning detaljer ──
    tax_detail_rows = [
        html.Tr([html.Td(t('tax.total_gains')), html.Td(f"{tax.total_gains_dkk:+,.2f} DKK", style={"textAlign": "right", "color": COLORS["green"]})]),
        html.Tr([html.Td(t('tax.total_losses')), html.Td(f"{tax.total_losses_dkk:+,.2f} DKK", style={"textAlign": "right", "color": COLORS["red"]})]),
        html.Tr([html.Td(t('tax.net')), html.Td(f"{tax.net_gain_dkk:+,.2f} DKK", style={"textAlign": "right", "fontWeight": "700"})]),
        html.Tr([html.Td(html.Hr()), html.Td(html.Hr())]),
        html.Tr([html.Td(t('tax.tax_low_bracket')), html.Td(f"{tax.tax_low_bracket:,.2f} DKK", style={"textAlign": "right"})]),
        html.Tr([html.Td(t('tax.tax_high_bracket')), html.Td(f"{tax.tax_high_bracket:,.2f} DKK", style={"textAlign": "right"})]),
        html.Tr([html.Td(html.Strong(t('risk.estimated_tax'))), html.Td(html.Strong(f"{tax.total_tax_dkk:,.2f} DKK"), style={"textAlign": "right", "color": COLORS["orange"]})]),
    ]
    tax_detail_table = dbc.Table(
        [html.Tbody(tax_detail_rows)],
        bordered=False, size="sm",
        style={"color": COLORS["text"]},
    )

    # ── Projektion-chart (YTD vs. forventet) ──
    fig_proj = go.Figure()
    fig_proj.add_trace(go.Bar(
        x=[t('tax.realized_ytd'), t('tax.expected_full_year')],
        y=[q_est.net_ytd_dkk, q_est.projected_annual_gain_dkk],
        marker_color=[COLORS["accent"], COLORS["muted"]],
        text=[f"{q_est.net_ytd_dkk:+,.0f}", f"{q_est.projected_annual_gain_dkk:+,.0f}"],
        textposition="outside",
    ))
    # Progressionsgrænse linje
    fig_proj.add_hline(
        y=progression_limit, line_dash="dash",
        line_color=COLORS["orange"], line_width=2,
        annotation_text=f"{t('tax.limit_label')}: {progression_limit:,.0f} DKK",
        annotation_font_color=COLORS["orange"],
    )
    _fig_layout(fig_proj, t('tax.projection_title'), 300)
    fig_proj.update_layout(showlegend=False)

    # ── P&L per aktie chart ──
    fig_per_sym = go.Figure()
    if tax.per_symbol:
        syms = sorted(tax.per_symbol.keys())
        nets = [tax.per_symbol[s]["gains"] + tax.per_symbol[s]["losses"] for s in syms]
        bar_colors = [COLORS["green"] if n > 0 else COLORS["red"] for n in nets]
        fig_per_sym.add_trace(go.Bar(
            x=syms, y=nets, marker_color=bar_colors,
            text=[f"{n:+,.0f}" for n in nets], textposition="outside",
        ))
    _fig_layout(fig_per_sym, t('tax.pnl_per_stock'), 300)
    fig_per_sym.update_layout(showlegend=False)

    # ── Progressionsgrænse gauge ──
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=tax.taxable_gain_dkk,
        number={"suffix": " DKK", "font": {"color": COLORS["text"]}},
        delta={"reference": progression_limit, "relative": False},
        gauge={
            "axis": {"range": [0, progression_limit * 1.5], "tickcolor": COLORS["muted"]},
            "bar": {"color": COLORS["accent"] if not near_limit else COLORS["red"]},
            "bgcolor": COLORS["card"],
            "bordercolor": COLORS["border"],
            "steps": [
                {"range": [0, progression_limit], "color": "rgba(0,212,170,0.1)"},
                {"range": [progression_limit, progression_limit * 1.5], "color": "rgba(255,71,87,0.1)"},
            ],
            "threshold": {
                "line": {"color": COLORS["orange"], "width": 3},
                "thickness": 0.8,
                "value": progression_limit,
            },
        },
    ))
    _fig_layout(fig_gauge, t('tax.progression_gauge_title'), 280)

    # ── Kvartalsestimat-tabel ──
    q_rows = [
        html.Tr([html.Td(t('tax.quarter')), html.Td(f"Q{q_est.quarter} {year}", style={"textAlign": "right", "fontWeight": "600"})]),
        html.Tr([html.Td(t('tax.trades_ytd')), html.Td(f"{q_est.num_trades_ytd}", style={"textAlign": "right"})]),
        html.Tr([html.Td(t('tax.net_ytd')), html.Td(f"{q_est.net_ytd_dkk:+,.0f} DKK", style={"textAlign": "right", "color": COLORS["green"] if q_est.net_ytd_dkk > 0 else COLORS["red"]})]),
        html.Tr([html.Td(t('tax.tax_ytd')), html.Td(f"{q_est.tax_ytd_dkk:,.0f} DKK", style={"textAlign": "right", "color": COLORS["orange"]})]),
        html.Tr([html.Td(html.Hr()), html.Td(html.Hr())]),
        html.Tr([html.Td(html.Strong(t('risk.expected_gain'))), html.Td(html.Strong(f"{q_est.projected_annual_gain_dkk:+,.0f} DKK"), style={"textAlign": "right"})]),
        html.Tr([html.Td(html.Strong(t('risk.expected_tax'))), html.Td(html.Strong(f"{q_est.projected_annual_tax_dkk:,.0f} DKK"), style={"textAlign": "right", "color": COLORS["orange"]})]),
        html.Tr([html.Td(t('tax.effective_rate')), html.Td(f"{q_est.projected_effective_rate:.1f}%", style={"textAlign": "right"})]),
    ]
    if q_est.projected_hits_limit:
        q_rows.append(html.Tr([
            html.Td(html.Strong(f"⚠️ {t('risk.progression_limit')}"), style={"color": COLORS["red"]}),
            html.Td(html.Strong(
                f"{t('tax.hits_limit')}{' ca. ' + q_est.projected_limit_date if q_est.projected_limit_date else '!'}"),
                style={"textAlign": "right", "color": COLORS["red"]}),
        ]))

    q_table = dbc.Table([html.Tbody(q_rows)], bordered=False, size="sm",
                        style={"color": COLORS["text"]})

    # ── Deadlines (årsafslutning) ──
    deadlines_items = [
        html.Li(f"31. dec {year}: {t('tax.deadline_last_trade')} {year}",
                style={"marginBottom": "8px"}),
        html.Li(f"1. marts {year + 1}: {t('tax.deadline_tax_opens')}",
                style={"marginBottom": "8px"}),
        html.Li(f"1. maj {year + 1}: {t('tax.deadline_correction')}",
                style={"marginBottom": "8px"}),
        html.Li(f"1. juli {year + 1}: {t('tax.deadline_residual_tax')}",
                style={"marginBottom": "8px"}),
    ]
    deadlines_content = html.Ul(deadlines_items, style={"listStyle": "none", "padding": "0"})

    # ── Tip til skatteoptimering ──
    tips_items = []
    remaining_limit = max(progression_limit - tax.taxable_gain_dkk, 0)
    if near_limit:
        tips_items.append(html.Li([
            html.Strong(f"⚠️ {t('risk.progression_limit')}: ", style={"color": COLORS["red"]}),
            t('tax.progression_warning'),
        ], style={"marginBottom": "10px"}))
    elif tax.taxable_gain_dkk > 0:
        tips_items.append(html.Li([
            html.Strong(f"💡 {t('risk.progression_limit')}: ", style={"color": COLORS["green"]}),
            t('tax.progression_ok').replace('{remaining}', f'{remaining_limit:,.0f}'),
        ], style={"marginBottom": "10px"}))

    if tax.total_losses_dkk < 0:
        tips_items.append(html.Li([
            html.Strong(f"📉 {t('risk.realized_losses')}: ", style={"color": COLORS["accent"]}),
            t('tax.losses_offset').replace('{losses}', f'{tax.total_losses_dkk:+,.0f}'),
        ], style={"marginBottom": "10px"}))

    tips_items.append(html.Li([
        html.Strong(f"🇺🇸 {t('risk.us_dividends')}: ", style={"color": COLORS["accent"]}),
        t('tax.us_dividend_tip'),
    ], style={"marginBottom": "10px"}))

    tips_items.append(html.Li([
        html.Strong(f"⚠️ {t('risk.wash_sale')}: ", style={"color": COLORS["orange"]}),
        t('tax.wash_sale_tip'),
    ], style={"marginBottom": "10px"}))

    tips_content = html.Ul(tips_items, style={"listStyle": "none", "padding": "0"})

    # ═══════════════════════════════════════════════════════════
    #  SAMLET LAYOUT
    # ═══════════════════════════════════════════════════════════

    return html.Div([
        html.H3(t('analysis.tax_reporting'),
                style={"color": COLORS["text"], "marginBottom": "20px"}),
        disclaimer,
        *alert_elements,
        kpi_row,

        # Række 1: Rubrikker + Skatteberegning + Gauge
        dbc.Row([
            dbc.Col(_card(t('tax.rubrik_header'), rubrik_table), md=5),
            dbc.Col(_card(t('tax.tax_calc_header'), tax_detail_table), md=4),
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_gauge, config={"displayModeBar": False}),
                             style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                                    "border": f"1px solid {COLORS['border']}"}), md=3),
        ], className="g-3 mb-4"),

        # Række 2: Projektion + Kvartalsestimat
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_proj, config={"displayModeBar": False}),
                             style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                                    "border": f"1px solid {COLORS['border']}"}), md=6),
            dbc.Col(_card(
                f"📊 {t('tax.quarterly_estimate')} – Q{q_est.quarter} {year}",
                html.Div([
                    q_table,
                    html.P(
                        t('tax.projection_note'),
                        style={"color": COLORS["muted"], "fontSize": "0.8rem",
                               "marginTop": "8px"},
                    ),
                ]),
            ), md=6),
        ], className="g-3 mb-4"),

        # Række 3: P&L per aktie
        dbc.Row([
            dbc.Col(dbc.Card(dcc.Graph(figure=fig_per_sym, config={"displayModeBar": False}),
                             style={"backgroundColor": COLORS["card"], "borderRadius": "12px",
                                    "border": f"1px solid {COLORS['border']}"}), md=12),
        ], className="g-3 mb-4"),

        # Række 4: Skatteoptimering + Deadlines
        dbc.Row([
            dbc.Col(_card(f"💡 {t('tax.tax_optimization_header')}", tips_content), md=7),
            dbc.Col(_card(f"📅 {t('tax.important_deadlines')}", deadlines_content), md=5),
        ], className="g-3 mb-4"),

        # Footer disclaimer
        dbc.Alert([
            html.Small(
                f"⚠️ {t('tax.footer_disclaimer')}"
            ),
        ], color="secondary",
            style={"backgroundColor": "#1a1a2e", "color": COLORS["muted"],
                   "border": f"1px solid {COLORS['border']}", "borderRadius": "12px"}),
    ])


# ══════════════════════════════════════════════════════════════
#  SIDE 6 – Markedsoverblik
# ══════════════════════════════════════════════════════════════


_scanner_state = {"loading": False, "done": False}


def _fetch_scanner_data_sync():
    """Fetch scanner data — heavy work, called from background thread at startup."""
    _scanner_state["loading"] = True
    try:
        from src.strategy.market_scanner import (
            MarketScanner, SECTOR_ETF_MAP, VIX_SYMBOL, DXY_SYMBOL,
            GOLD_SYMBOL, OIL_SYMBOL, SP500_SYMBOL, YIELD_2Y, YIELD_10Y,
        )

        mdf = _get_market_data()
        scanner = MarketScanner()

        # Hent sektor-ETF data
        sector_data = {}
        for etf in SECTOR_ETF_MAP:
            try:
                df = mdf.get_historical(etf, interval="1d", lookback_days=365)
                if not df.empty:
                    df = add_all_indicators(df)
                sector_data[etf] = df
            except Exception:
                sector_data[etf] = pd.DataFrame()

        # Hent makro-data
        macro_symbols = [VIX_SYMBOL, DXY_SYMBOL, GOLD_SYMBOL, OIL_SYMBOL,
                         SP500_SYMBOL, YIELD_2Y, YIELD_10Y]
        macro_data = {}
        for sym in macro_symbols:
            try:
                macro_data[sym] = mdf.get_historical(sym, interval="1d", lookback_days=365)
            except Exception:
                macro_data[sym] = pd.DataFrame()

        # Score aktive symboler (begrænset til SYMBOLS for dashboard-hastighed)
        asset_data = {}
        for sym in SYMBOLS:
            try:
                df = _get_stock_data(sym)
                if not df.empty:
                    asset_data[sym] = df
            except Exception:
                pass

        # Tilføj sektor-ETFs som aktiver
        asset_data.update(sector_data)

        benchmark = macro_data.get(SP500_SYMBOL)
        result = scanner.full_scan(asset_data, sector_data, macro_data, benchmark=benchmark)
        _cache["scanner_result"] = result
        _scanner_state["done"] = True
        logger.info("[scanner] Background market overview data ready")
        return result
    except Exception as exc:
        logger.warning(f"[scanner] Background fetch failed: {exc}")
        _scanner_state["done"] = True
        raise
    finally:
        _scanner_state["loading"] = False


def _get_scanner_data():
    """Hent scanner-data (cached). Returns None if still loading."""
    if "scanner_result" in _cache:
        return _cache["scanner_result"]
    if _scanner_state["loading"]:
        return None  # still loading in background
    # Not cached and not loading — fetch synchronously (fallback)
    return _fetch_scanner_data_sync()


def page_market_overview():
    """Side 6 – Markedsoverblik: heatmap, top movers, makro, scanner-picks."""
    try:
        result = _get_scanner_data()
    except Exception as exc:
        return html.Div([
            html.H4(t('analysis.market_overview'), style={"color": COLORS["text"]}),
            dbc.Alert(f"{t('health.loading_error')}: {exc}", color="danger"),
        ])

    if result is None:
        # Data is still being fetched in background — show spinner.
        # Try to trigger background fetch if not started
        try:
            if not _scanner_state.get("loading"):
                import threading as _th
                _th.Thread(target=_fetch_scanner_data_sync, daemon=True).start()
        except Exception:
            pass
        return dbc.Container([
            html.H3([
                html.I(className="bi bi-globe2 me-2"),
                t('analysis.market_overview'),
            ], style={"color": COLORS["text"]}),
            html.Div([
                dbc.Spinner(color="primary", type="grow", style={"width": "3rem", "height": "3rem"}),
                html.H5(
                    t('common.loading'),
                    className="text-muted mt-4",
                ),
                html.P("Scanner data indlæses i baggrunden — siden opdateres automatisk...",
                       style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
            ], className="text-center", style={"padding": "100px 0"}),
            # Hidden elements needed by callbacks (prevents missing ID errors)
            html.Div(id="alloc-apply-result", style={"display": "none"}),
            html.Div(id="alloc-total-warn", style={"display": "none"}),
            html.Div(id="exlim-save-result", style={"display": "none"}),
            html.Div(id="exlim-total-warn", style={"display": "none"}),
            dcc.Graph(id="alloc-donut", style={"display": "none"}),
            dcc.Location(id="scanner-reload", refresh=True),
            dcc.Interval(id="scanner-poll", interval=5_000, max_intervals=1),
        ], fluid=True, className="p-4")

    macro = result.macro
    sectors = result.sector_performance

    # ── Header ──
    header = html.H4(
        [html.I(className="bi bi-globe2 me-2"), t('analysis.market_overview')],
        style={"color": COLORS["text"], "marginBottom": "20px"},
    )

    # ── Makro KPI-kort ──
    vix_color = COLORS["green"] if macro.vix_level == "low" else \
        COLORS["orange"] if macro.vix_level in ("normal", "elevated") else COLORS["red"]
    yc_color = COLORS["green"] if macro.yield_curve_status == "normal" else \
        COLORS["orange"] if macro.yield_curve_status == "flat" else COLORS["red"]

    macro_row = dbc.Row([
        dbc.Col(_metric_card(t('vix'), f"{macro.vix:.1f}",
                              f"{macro.vix_change:+.1f}%", vix_color), xs=6, md=2),
        dbc.Col(_metric_card(t('dxy'), f"{macro.dxy:.1f}",
                              f"{macro.dxy_change:+.1f}%"), xs=6, md=2),
        dbc.Col(_metric_card(t('gold'), f"${macro.gold_price:,.0f}",
                              f"{macro.gold_change_1m:+.1f}%/m"), xs=6, md=2),
        dbc.Col(_metric_card(t('oil'), f"${macro.oil_price:.0f}",
                              f"{macro.oil_change_1m:+.1f}%/m"), xs=6, md=2),
        dbc.Col(_metric_card(t('yield_spread'), f"{macro.yield_spread:+.2f}%",
                              macro.yield_curve_status.upper(), yc_color), xs=6, md=2),
        dbc.Col(_metric_card("S&P 500", f"{macro.sp500_change_1m:+.1f}%",
                              t('common.1_month')), xs=6, md=2),
    ], className="g-3 mb-4")

    # ── Sektor-heatmap ──
    # H-21: Heatmap uses live sector_performance data from scanner results.
    # If scanner data is unavailable, an empty figure is shown as fallback.
    # TODO: Add a visible "no data" message when sectors list is empty.
    if sectors:
        s_names = [s.name for s in sectors]
        s_1m = [s.change_1m for s in sectors]

        heatmap_fig = go.Figure(data=go.Heatmap(
            z=[s_1m],
            x=s_names,
            y=[f"{t('common.1_month')} %"],
            colorscale=[[0, COLORS["red"]], [0.5, "#333"], [1, COLORS["green"]]],
            zmid=0,
            text=[[f"{v:+.1f}%" for v in s_1m]],
            texttemplate="%{text}",
            textfont={"size": 13},
            showscale=False,
            hovertemplate="%{x}: %{text}<extra></extra>",
        ))
        _fig_layout(heatmap_fig, t('analysis.sector_heatmap_1m'), height=140)
        heatmap_fig.update_layout(margin=dict(l=80, r=20, t=50, b=10))
    else:
        heatmap_fig = go.Figure()
        _fig_layout(heatmap_fig, t('analysis.sector_heatmap'), height=140)

    heatmap_card = dbc.Card(
        dbc.CardBody(dcc.Graph(figure=heatmap_fig, config={"displayModeBar": False})),
        style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
               "borderRadius": "12px"},
    )

    # ── Sektor-tabel ──
    sector_rows = []
    for s in sectors:
        trend_badge = {"up": ("↑ Op", "success"), "down": ("↓ Ned", "danger"),
                       "neutral": ("→ Neutral", "secondary")}
        badge_text, badge_color = trend_badge.get(s.trend, ("→", "secondary"))

        def _color_val(v):
            c = COLORS["green"] if v > 0 else COLORS["red"] if v < 0 else COLORS["muted"]
            return html.Span(f"{v:+.1f}%", style={"color": c})

        sector_rows.append(html.Tr([
            html.Td(s.name, style={"color": COLORS["text"]}),
            html.Td(s.etf_symbol, style={"color": COLORS["muted"]}),
            html.Td(_color_val(s.change_1d)),
            html.Td(_color_val(s.change_1w)),
            html.Td(_color_val(s.change_1m)),
            html.Td(_color_val(s.change_3m)),
            html.Td(dbc.Badge(badge_text, color=badge_color, className="px-2")),
        ]))

    sector_table = dbc.Card(
        dbc.CardBody([
            html.H6(t('analysis.sector_performance'), style={"color": COLORS["text"], "marginBottom": "12px"}),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th(t('common.sector')), html.Th("ETF"), html.Th("1d"),
                    html.Th("1w"), html.Th("1m"), html.Th("3m"), html.Th(t('common.trend')),
                ], style={"color": COLORS["muted"]})),
                html.Tbody(sector_rows),
            ], bordered=False, hover=True, responsive=True, size="sm",
               style={"color": COLORS["text"]}),
        ]),
        style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
               "borderRadius": "12px"},
    )

    # ── Top Movers (Scanner-picks) ──
    def _sells_table(picks, title, icon):
        rows = []
        for i, a in enumerate(picks[:10], 1):
            reason = a.reasons[0] if a.reasons else ""
            c = COLORS["green"] if a.change_pct > 0 else COLORS["red"]
            rows.append(html.Tr([
                html.Td(str(i), style={"color": COLORS["muted"]}),
                html.Td(a.symbol, style={"color": COLORS["text"], "fontWeight": "600"}),
                html.Td(f"{a.score:.0f}", style={"color": COLORS["accent"]}),
                html.Td(f"{a.change_pct:+.1f}%", style={"color": c}),
                html.Td(reason, style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
            ]))

        return dbc.Card(
            dbc.CardBody([
                html.H6([icon, f" {title}"],
                         style={"color": COLORS["text"], "marginBottom": "12px"}),
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("#"), html.Th("Symbol"), html.Th("Score"),
                        html.Th(t('common.change')), html.Th(t('common.signal')),
                    ], style={"color": COLORS["muted"]})),
                    html.Tbody(rows if rows else [
                        html.Tr(html.Td(t('common.no_data'), colSpan=5,
                                        style={"color": COLORS["muted"]}))
                    ]),
                ], bordered=False, hover=True, responsive=True, size="sm",
                   style={"color": COLORS["text"]}),
            ]),
            style={"backgroundColor": COLORS["card"],
                   "border": f"1px solid {COLORS['border']}", "borderRadius": "12px"},
        )

    def _buys_table(picks, title, icon):
        rows = []
        for i, a in enumerate(picks[:10], 1):
            reason = a.reasons[0] if a.reasons else ""
            c = COLORS["green"] if a.change_pct > 0 else COLORS["red"]
            rows.append(html.Tr([
                html.Td(str(i), style={"color": COLORS["muted"]}),
                html.Td(a.symbol, style={"color": COLORS["text"], "fontWeight": "600"}),
                html.Td(f"{a.score:.0f}", style={"color": COLORS["accent"]}),
                html.Td(f"{a.change_pct:+.1f}%", style={"color": c}),
                html.Td(reason, style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
            ]))

        return dbc.Card(
            dbc.CardBody([
                html.H6([icon, f" {title}"],
                         style={"color": COLORS["text"], "marginBottom": "12px"}),
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("#"), html.Th("Symbol"), html.Th("Score"),
                        html.Th(t('common.change')), html.Th(t('common.signal')),
                    ], style={"color": COLORS["muted"]})),
                    html.Tbody(rows if rows else [
                        html.Tr(html.Td(t('common.no_data'), colSpan=5,
                                        style={"color": COLORS["muted"]}))
                    ]),
                ], bordered=False, hover=True, responsive=True, size="sm",
                   style={"color": COLORS["text"]}),
            ]),
            style={"backgroundColor": COLORS["card"],
                   "border": f"1px solid {COLORS['border']}", "borderRadius": "12px"},
        )

    buys_card = _buys_table(result.top_buys, t('top_candidates.buy'), "\U0001f7e2")
    sells_card = _sells_table(result.top_sells, t('top_candidates.sell'), "\U0001f534")

    # ── Alerts ──
    alert_items = []
    for alert in result.alerts[:8]:
        color_map = {"HIGH": "danger", "MEDIUM": "warning", "LOW": "info"}
        alert_items.append(
            dbc.Alert([
                html.Strong(f"[{alert.severity}] {alert.title}"),
                html.P(alert.message, className="mb-0 mt-1",
                       style={"fontSize": "0.85rem"}),
            ], color=color_map.get(alert.severity, "secondary"),
               className="py-2 mb-2",
               style={"borderRadius": "8px"}),
        )

    alerts_card = dbc.Card(
        dbc.CardBody([
            html.H6(t('analysis.opportunity_alerts'), style={"color": COLORS["text"], "marginBottom": "12px"}),
            html.Div(alert_items if alert_items else [
                html.P(t('analysis.no_alerts'),
                       style={"color": COLORS["muted"]})
            ]),
        ]),
        style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
               "borderRadius": "12px"},
    )

    # ── Anbefalet allokering ──
    alloc = result.allocation
    # Use saved allocation if available, otherwise scanner recommendation
    _saved_alloc = {}
    try:
        import json as _json_alloc
        _alloc_path = Path("config/allocation.json")
        if _alloc_path.exists():
            _saved_alloc = _json_alloc.loads(_alloc_path.read_text())
    except Exception:
        pass
    _a_stocks = _saved_alloc.get("stocks_pct", alloc.stocks_pct)
    _a_bonds = _saved_alloc.get("bonds_pct", alloc.bonds_pct)
    _a_commodities = _saved_alloc.get("commodities_pct", alloc.commodities_pct)
    _a_crypto = _saved_alloc.get("crypto_pct", alloc.crypto_pct)
    _a_cash = _saved_alloc.get("cash_pct", alloc.cash_pct)

    alloc_fig = go.Figure(data=[go.Pie(
        labels=[t('allocation.stocks'), t('allocation.bonds'), t('allocation.commodities'), t('allocation.crypto'), t('allocation.cash')],
        values=[_a_stocks, _a_bonds, _a_commodities, _a_crypto, _a_cash],
        hole=0.55,
        marker_colors=[COLORS["blue"], COLORS["green"], COLORS["orange"],
                       COLORS["purple"], COLORS["muted"]],
        textinfo="label+percent",
        textfont_size=12,
    )])
    _fig_layout(alloc_fig, t('allocation.recommended'), height=300)
    alloc_fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20))

    _input_style = {
        "width": "70px", "backgroundColor": COLORS["bg"], "color": COLORS["text"],
        "border": f"1px solid {COLORS['border']}", "borderRadius": "4px",
        "textAlign": "center", "fontSize": "0.85rem",
    }
    _alloc_fields = [
        ("alloc-stocks", t('allocation.stocks'), _a_stocks, COLORS["blue"]),
        ("alloc-bonds", t('allocation.bonds'), _a_bonds, COLORS["green"]),
        ("alloc-commodities", t('allocation.commodities'), _a_commodities, COLORS["orange"]),
        ("alloc-crypto", t('allocation.crypto'), _a_crypto, COLORS["purple"]),
        ("alloc-cash", t('allocation.cash'), _a_cash, COLORS["muted"]),
    ]

    alloc_inputs = []
    for fid, label, val, color in _alloc_fields:
        alloc_inputs.append(
            html.Div([
                html.Span(label, style={"color": color, "fontWeight": "600", "fontSize": "0.85rem",
                                         "width": "120px", "textAlign": "right", "paddingRight": "8px",
                                         "flexShrink": "0"}),
                dcc.Input(id=fid, type="number", min=0, max=100, step=1, value=int(val),
                          style=_input_style),
                html.Span(" %", style={"color": COLORS["muted"], "marginLeft": "4px"}),
            ], className="d-flex align-items-center", style={"marginBottom": "6px"}),
        )

    alloc_card = dbc.Card(
        dbc.CardBody([
            dcc.Graph(id="alloc-donut", figure=alloc_fig, config={"displayModeBar": False}),
            html.Hr(style={"borderColor": COLORS["border"]}),
            html.H6(t('allocation.recommended'), style={"color": COLORS["text"], "marginBottom": "12px", "fontSize": "0.9rem"}),
            html.Div(alloc_inputs),
            html.Div(id="alloc-total-warn", style={"marginTop": "4px"}),
            dbc.Button(
                [html.I(className="bi bi-arrow-repeat me-2"), t('common.apply')],
                id="alloc-apply-btn", color="success", outline=True, className="w-100 mt-2",
            ),
            html.Div(id="alloc-apply-result", className="mt-2"),
        ]),
        style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
               "borderRadius": "12px"},
    )

    # ── Makro-indikatorer med pile ──
    def _macro_indicator(name, value, change, level_text, icon):
        arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
        c = COLORS["green"] if change > 0 else COLORS["red"] if change < 0 else COLORS["muted"]
        return html.Div([
            html.Span(icon + " ", style={"fontSize": "1.2rem"}),
            html.Strong(name, style={"color": COLORS["text"]}),
            html.Span(f"  {value}", style={"color": COLORS["accent"], "marginLeft": "8px"}),
            html.Span(f"  {arrow} {change:+.1f}%", style={"color": c, "marginLeft": "4px"}),
            html.Span(f"  ({level_text})",
                       style={"color": COLORS["muted"], "fontSize": "0.85rem", "marginLeft": "4px"}),
        ], style={"padding": "8px 0", "borderBottom": f"1px solid {COLORS['border']}"})

    macro_indicators = dbc.Card(
        dbc.CardBody([
            html.H6(t('analysis.macro_indicators'), style={"color": COLORS["text"], "marginBottom": "12px"}),
            _macro_indicator("VIX", f"{macro.vix:.1f}", macro.vix_change,
                              macro.vix_level, "😰" if macro.vix > 25 else "😊"),
            _macro_indicator("Dollar", f"{macro.dxy:.1f}", macro.dxy_change,
                              macro.dxy_trend, "💲"),
            _macro_indicator("Guld", f"${macro.gold_price:,.0f}", macro.gold_change_1m,
                              "1 md", "🥇"),
            _macro_indicator("Olie", f"${macro.oil_price:.0f}", macro.oil_change_1m,
                              "1 md", "🛢️"),
            _macro_indicator("Yield Spread", f"{macro.yield_spread:+.2f}%", 0,
                              macro.yield_curve_status, "📈"),
        ]),
        style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
               "borderRadius": "12px"},
    )

    # ── Saml layout ──
    return html.Div([
        header,
        macro_row,
        html.Div(heatmap_card, className="mb-4"),
        dbc.Row([
            dbc.Col(buys_card, xs=12, md=6),
            dbc.Col(sells_card, xs=12, md=6),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(sector_table, xs=12, md=8),
            dbc.Col(macro_indicators, xs=12, md=4),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(alerts_card, xs=12, md=7),
            dbc.Col(alloc_card, xs=12, md=5),
        ], className="g-3 mb-4"),
        dbc.Row([dbc.Col(_build_exchange_limits_card(), xs=12)], className="g-3 mb-4"),
        html.Footer(
            html.Small(
                f"Scanning: {result.scan_duration_ms:.0f}ms | "
                f"{len(result.all_scored)} aktiver | "
                f"{len(result.alerts)} alerts",
                style={"color": COLORS["muted"]},
            ),
            className="text-center mt-3",
        ),
    ])



# ── Exchange Limits card ──
_EXLIM_LABELS = {
    "us_stocks": "NYSE / NASDAQ",
    "eu_nordic": "EU / Nordic",
    "london": "London (LSE)",
    "canada": "Toronto (TSX)",
    "japan": "Tokyo (TSE)",
    "hong_kong": "Hong Kong (HKEX)",
    "australia": "Australia (ASX)",
    "new_zealand": "New Zealand (NZX)",
    "india": "India (NSE)",
    "crypto": "Crypto",
    "chicago": "Chicago (CME)",
    "singapore": "Singapore (SGX)",
    "etfs": "ETFs",
    "korea": "Seoul (KRX)",
    "taiwan": "Taiwan (TWSE)",
    "shanghai": "Shanghai (SSE)",
    "shenzhen": "Shenzhen (SZSE)",
    "brazil": "São Paulo (B3)",
    "mexico": "Mexico (BMV)",
    "south_africa": "Johannesburg (JSE)",
    "saudi": "Saudi (Tadawul)",
    "indonesia": "Jakarta (IDX)",
}


def _load_exchange_limits() -> dict[str, int]:
    try:
        import json as _j
        p = Path("config/exchange_limits.json")
        if p.exists():
            return _j.loads(p.read_text())
    except Exception:
        pass
    return {}


def _build_exchange_limits_card():
    saved = _load_exchange_limits()
    _inp_style = {
        "width": "62px", "textAlign": "center", "fontSize": "0.82rem",
        "backgroundColor": "#e2e8f0", "color": "#2d3748",
        "border": "1px solid #cbd5e0", "borderRadius": "4px",
        "padding": "2px 4px",
    }
    _label_style = {
        "color": COLORS["text"], "fontSize": "0.82rem", "fontWeight": "600",
        "flex": "1", "textAlign": "right", "paddingRight": "4px",
        "whiteSpace": "nowrap",
    }

    # All 11 items in a single list, rendered as a table-like grid
    items = []
    for key in _EXLIM_LABELS:
        label = t(f'exchange_limits.{key}')
        val = saved.get(key, 0)
        items.append(
            html.Div([
                html.Span(label, style=_label_style),
                dcc.Input(id={"type": "exlim", "index": key}, type="number",
                          min=0, max=100, step=1, value=val, style=_inp_style, debounce=True),
                html.Span(" %", style={"color": COLORS["muted"], "marginLeft": "2px", "width": "16px", "fontSize": "0.82rem"}),
            ], className="d-flex align-items-center", style={"marginBottom": "4px"}),
        )

    # Top grid: 4 columns × 4 rows = 16 exchanges
    col1 = items[0:4]
    col2 = items[4:8]
    col3 = items[8:12]
    col4 = items[12:16]
    col_divs = [
        dbc.Col(col1, xs=12, md=3),
        dbc.Col(col2, xs=12, md=3),
        dbc.Col(col3, xs=12, md=3),
        dbc.Col(col4, xs=12, md=3),
    ]

    # Bottom rows: save button + 3 exchanges per row (6 total)
    bottom_row1 = items[16:19]
    bottom_row2 = items[19:22]

    return dbc.Card(
        dbc.CardBody([
            html.H6([html.I(className="bi bi-bank me-2"), t('exchange_limits.title')],
                     style={"color": COLORS["text"], "marginBottom": "12px"}),
            html.P(t('exchange_limits.desc'),
                   style={"color": COLORS["muted"], "fontSize": "0.8rem"}, className="mb-3"),
            dbc.Row(col_divs),
            html.Div(id="exlim-total-warn", className="mt-2"),
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        [html.I(className="bi bi-save me-2"), t('common.save')],
                        id="exlim-save-btn", color="success", outline=True,
                    ),
                    html.Div(id="exlim-save-result", className="mt-1"),
                ], xs=12, md=3, className="d-flex flex-column justify-content-center"),
                dbc.Col(bottom_row1, xs=12, md=3),
                dbc.Col(bottom_row2, xs=12, md=3),
            ], className="mt-2"),
        ]),
        style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
               "borderRadius": "12px"},
    )


@callback(
    Output("exlim-total-warn", "children"),
    [Input({"type": "exlim", "index": key}, "value") for key in _EXLIM_LABELS],
    prevent_initial_call=True,
)
def _check_exchange_total(*values):
    total = sum(int(v or 0) for v in values)
    if total > 100:
        return html.Small(f"Total: {total}% (over 100%)", style={"color": COLORS["red"]})
    if total > 0:
        return html.Small(f"Total: {total}%", style={"color": COLORS["muted"]})
    return ""


@callback(
    Output("exlim-save-result", "children"),
    Input("exlim-save-btn", "n_clicks"),
    [State({"type": "exlim", "index": key}, "value") for key in _EXLIM_LABELS],
    prevent_initial_call=True,
)
def _save_exchange_limits(n_clicks, *values):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    import json as _j
    data = {}
    for key, val in zip(_EXLIM_LABELS.keys(), values):
        data[key] = int(val or 0)
    total = sum(data.values())
    if total > 100:
        return html.Span(f"Total {total}% overstiger 100%", style={"color": COLORS["red"]})
    try:
        Path("config/exchange_limits.json").write_text(_j.dumps(data, indent=2))
        return html.Span("Gemt", style={"color": COLORS["green"]})
    except Exception as e:
        return html.Span(f"Fejl: {e}", style={"color": COLORS["red"]})


# ── Allocation: update donut when inputs change ──
@callback(
    Output("alloc-donut", "figure"),
    Output("alloc-total-warn", "children"),
    Input("alloc-stocks", "value"),
    Input("alloc-bonds", "value"),
    Input("alloc-commodities", "value"),
    Input("alloc-crypto", "value"),
    Input("alloc-cash", "value"),
    prevent_initial_call=True,
)
def _update_alloc_donut(stocks, bonds, commodities, crypto, cash):
    vals = [int(stocks or 0), int(bonds or 0), int(commodities or 0),
            int(crypto or 0), int(cash or 0)]
    total = sum(vals)
    labels = [t('allocation.stocks'), t('allocation.bonds'), t('allocation.commodities'),
              t('allocation.crypto'), t('allocation.cash')]
    fig = go.Figure(data=[go.Pie(
        labels=labels, values=vals, hole=0.55,
        marker_colors=[COLORS["blue"], COLORS["green"], COLORS["orange"],
                       COLORS["purple"], COLORS["muted"]],
        textinfo="label+percent", textfont_size=12,
    )])
    _fig_layout(fig, t('allocation.recommended'), height=300)
    fig.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20))
    warn = None
    if total != 100:
        warn = html.Small(f"Total: {total}% (skal være 100%)",
                          style={"color": COLORS["red"]})
    return fig, warn


# ── Allocation: classify a symbol into one of the 5 asset classes ──
def _asset_class(sym: str) -> str:
    """Map a symbol to: stocks, bonds, commodities, crypto, or cash."""
    s = sym.upper()
    # Crypto
    if "-USD" in s or s.startswith("BTC") or s.startswith("ETH") or s.startswith("SOL") or s.startswith("BNB"):
        return "crypto"
    # Bonds / fixed-income ETFs
    _bond_syms = {"TLT", "IEF", "SHY", "BND", "AGG", "LQD", "HYG", "GOVT", "VCIT", "VCSH", "BNDX", "EMB", "TIP"}
    if s in _bond_syms or s.endswith("=F") and s.startswith("ZN") or s.startswith("ZB"):
        return "bonds"
    # Commodities ETFs / futures
    _commodity_syms = {"GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "PPLT", "PALL", "WEAT", "CORN", "SOYB"}
    if s in _commodity_syms or (s.endswith("=F") and s[:2] in {"GC", "SI", "CL", "NG", "HG", "ZC", "ZW", "ZS"}):
        return "commodities"
    # Everything else is stocks
    return "stocks"


# ── Allocation: apply rebalancing ──
@callback(
    Output("alloc-apply-result", "children"),
    Input("alloc-apply-btn", "n_clicks"),
    State("alloc-stocks", "value"),
    State("alloc-bonds", "value"),
    State("alloc-commodities", "value"),
    State("alloc-crypto", "value"),
    State("alloc-cash", "value"),
    prevent_initial_call=True,
)
def _apply_allocation(n_clicks, stocks_pct, bonds_pct, commodities_pct, crypto_pct, cash_pct):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    target = {
        "stocks": int(stocks_pct or 0),
        "bonds": int(bonds_pct or 0),
        "commodities": int(commodities_pct or 0),
        "crypto": int(crypto_pct or 0),
        "cash": int(cash_pct or 0),
    }
    total_pct = sum(target.values())
    if total_pct != 100:
        return dbc.Alert(f"Total er {total_pct}% — skal være 100%",
                         color="danger", className="py-2 mb-0")

    # Save allocation to config
    try:
        import json as _json
        alloc_path = Path("config/allocation.json")
        alloc_path.write_text(_json.dumps(
            {f"{k}_pct": v for k, v in target.items()}, indent=2))
    except Exception:
        pass

    results = []
    try:
        scanner_result = _cache.get("scanner_result")
        if not scanner_result:
            return dbc.Alert("Ingen scanner-data — åbn Markedsoverblik først",
                             color="warning", className="py-2 mb-0")

        from src.broker.paper_broker import PaperBroker
        from src.broker.base_broker import OrderType
        pb = PaperBroker()
        acc = pb.get_account()

        router = pb
        try:
            from src.broker.registry import get_router
            r = get_router()
            if r:
                router = r
        except Exception:
            pass

        # FX rate
        _usd_dkk = 6.90
        try:
            import yfinance as _yf_alloc
            _fx = _yf_alloc.Ticker("DKK=X")
            _r = getattr(_fx.fast_info, "last_price", None)
            if _r and _r > 0:
                _usd_dkk = _r
        except Exception:
            pass

        # ── 1. Current allocation by asset class (in USD) ──
        positions = list(router.get_positions()) if hasattr(router, 'get_positions') else []
        current_usd = {"stocks": 0.0, "bonds": 0.0, "commodities": 0.0, "crypto": 0.0}
        pos_by_class: dict[str, list] = {"stocks": [], "bonds": [], "commodities": [], "crypto": []}
        for p in positions:
            sym = getattr(p, "symbol", "")
            cls = _asset_class(sym)
            mv = getattr(p, "market_value", 0)
            current_usd[cls] += mv
            pos_by_class[cls].append(p)

        total_equity_usd = acc.equity  # cash + positions
        cash_usd = acc.cash

        # ── 2. Target values (in USD) ──
        target_usd = {k: total_equity_usd * target[k] / 100 for k in target if k != "cash"}
        target_cash_usd = total_equity_usd * target["cash"] / 100

        # ── 3. Compute drift per class (positive = need to buy, negative = need to sell) ──
        drift_usd = {k: target_usd[k] - current_usd[k] for k in target_usd}

        # Show current vs target
        results.append(html.Li(
            html.Small("Aktuel → Mål:", style={"color": COLORS["muted"], "fontWeight": "600"}),
        ))
        for cls in ["stocks", "bonds", "commodities", "crypto"]:
            cur_pct = (current_usd[cls] / total_equity_usd * 100) if total_equity_usd > 0 else 0
            tgt_pct = target[cls]
            drift_dkk = drift_usd[cls] * _usd_dkk
            arrow = "\u2191" if drift_dkk > 0 else "\u2193" if drift_dkk < 0 else "="
            color = COLORS["green"] if drift_dkk > 0 else COLORS["red"] if drift_dkk < 0 else COLORS["muted"]
            results.append(html.Li([
                html.Span(f"  {cls.capitalize()}: {cur_pct:.0f}% → {tgt_pct}% ",
                          style={"color": COLORS["text"]}),
                html.Span(f"({arrow} {abs(drift_dkk):,.0f} kr)",
                          style={"color": color, "fontWeight": "600"}),
            ]))
        cur_cash_pct = (cash_usd / total_equity_usd * 100) if total_equity_usd > 0 else 0
        results.append(html.Li(
            html.Span(f"  Cash: {cur_cash_pct:.0f}% → {target['cash']}%",
                      style={"color": COLORS["text"]}),
        ))
        results.append(html.Li(html.Hr(style={"borderColor": COLORS["border"]})))

        bought = 0
        sold = 0

        # ── 4. SELL overweight classes — sell worst-scored positions first ──
        # Build a score map from scanner sells (worst = lowest score)
        sell_scores = {a.symbol: a.score for a in scanner_result.top_sells}

        for cls in ["stocks", "bonds", "commodities", "crypto"]:
            if drift_usd[cls] >= 0:
                continue  # underweight or on target — no selling needed
            to_sell_usd = abs(drift_usd[cls])
            # Sort positions: sell scanner-flagged worst first, then lowest market value
            class_positions = sorted(
                pos_by_class[cls],
                key=lambda p: (sell_scores.get(getattr(p, "symbol", ""), 999),
                               getattr(p, "market_value", 0)),
            )
            sold_usd = 0.0
            for p in class_positions:
                if sold_usd >= to_sell_usd:
                    break
                sym = getattr(p, "symbol", "")
                qty = getattr(p, "qty", 0)
                mv = getattr(p, "market_value", 0)
                if qty <= 0:
                    continue
                # Sell partial if selling all would overshoot
                remaining = to_sell_usd - sold_usd
                if mv > remaining * 1.5 and qty > 1:
                    price = getattr(p, "current_price", 0) or 1
                    sell_qty = max(1, int(remaining / price))
                else:
                    sell_qty = qty
                try:
                    router.sell(symbol=sym, qty=sell_qty, order_type=OrderType.MARKET)
                    price_dkk = (getattr(p, "current_price", 0) or 0) * _usd_dkk
                    results.append(html.Li(
                        f"\u2713 {t('common.sell')} {sell_qty:.0f} \u00d7 {sym} @ {price_dkk:,.0f} kr",
                        style={"color": COLORS["text"]}))
                    sold += 1
                    sold_usd += sell_qty * (getattr(p, "current_price", 0) or 0)
                except Exception as exc:
                    results.append(html.Li(f"\u2717 {sym}: {exc}",
                                           style={"color": COLORS["red"]}))

        # ── 5. BUY underweight classes — buy from top-scored candidates ──
        # Map scanner buys to asset classes
        buy_candidates_by_class: dict[str, list] = {"stocks": [], "bonds": [], "commodities": [], "crypto": []}
        for a in scanner_result.top_buys:
            cls = _asset_class(a.symbol)
            buy_candidates_by_class[cls].append(a)

        # Fallback symbols for classes the scanner doesn't cover
        from dataclasses import dataclass as _dc, field as _fld
        @_dc
        class _FallbackAsset:
            symbol: str
            score: float = 50.0
            reasons: list = _fld(default_factory=list)

        from src.data.universe import ETFS_BONDS, ETFS_COMMODITIES
        _fallbacks = {
            "bonds": [_FallbackAsset(s) for s in ETFS_BONDS[:5]],
            "commodities": [_FallbackAsset(s) for s in ETFS_COMMODITIES[:5]],
        }

        for cls in ["stocks", "bonds", "commodities", "crypto"]:
            if drift_usd[cls] <= 0:
                continue  # overweight or on target — no buying needed
            to_buy_usd = drift_usd[cls]
            candidates = buy_candidates_by_class.get(cls, [])
            # Fallback: use well-known ETFs if scanner has no candidates
            if not candidates:
                candidates = _fallbacks.get(cls, [])
            bought_usd = 0.0
            for a in candidates[:10]:
                if bought_usd >= to_buy_usd:
                    break
                try:
                    from src.data.market_data import MarketDataFetcher
                    mdf = MarketDataFetcher()
                    df = mdf.get_historical(a.symbol, interval="1d", lookback_days=5)
                    if df.empty:
                        continue
                    price = float(df["Close"].iloc[-1])
                    if price <= 0:
                        continue
                    remaining = to_buy_usd - bought_usd
                    qty = max(1, int(remaining / price))
                    price_dkk = price * _usd_dkk
                    router.buy(symbol=a.symbol, qty=qty, order_type=OrderType.MARKET)
                    results.append(html.Li(
                        f"\u2713 {t('common.buy')} {qty} \u00d7 {a.symbol} @ {price_dkk:,.0f} kr",
                        style={"color": COLORS["text"]}))
                    bought += 1
                    bought_usd += qty * price
                except Exception as exc:
                    results.append(html.Li(f"\u2717 {a.symbol}: {exc}",
                                           style={"color": COLORS["red"]}))

        return dbc.Alert([
            html.Strong(f"Rebalancering: {bought} købt, {sold} solgt",
                        style={"color": COLORS["text"]}),
            html.Ul(results, className="list-unstyled mt-2 mb-0"),
        ], color="dark", dismissable=True,
           style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}"})

    except Exception as exc:
        logger.error(f"Allocation rebalance failed: {exc}")
        return dbc.Alert(str(exc), color="danger", dismissable=True, className="py-2")


# ══════════════════════════════════════════════════════════════
#  Side 7 – Sentiment
# ══════════════════════════════════════════════════════════════


def _get_sentiment_data(symbol: str = "AAPL") -> dict:
    """Hent sentiment-data for et symbol (med keyword-baseret fallback)."""
    try:
        from src.sentiment import (
            SentimentAnalyzer, EventDetector, NewsFetcher,
            keyword_sentiment, NewsArticle,
        )

        analyzer = SentimentAnalyzer(use_finbert=False)  # Keyword for hastighed
        detector = EventDetector()

        # Simulerede nyheder (erstat med rigtige via NewsFetcher med API keys)
        sample_articles = [
            NewsArticle(title=f"{symbol} Q4 earnings beat expectations, revenue surges",
                        summary="The company reported strong quarterly results.",
                        url="https://example.com/1", source="Reuters",
                        published=datetime.now().isoformat(), symbols=[symbol], relevance=1.0),
            NewsArticle(title=f"Analysts upgrade {symbol} stock after strong quarter",
                        summary="Multiple Wall Street firms raised their price targets.",
                        url="https://example.com/2", source="CNBC",
                        published=datetime.now().isoformat(), symbols=[symbol], relevance=0.9),
            NewsArticle(title=f"{symbol} announces new product launch for 2026",
                        summary="New device expected to drive growth.",
                        url="https://example.com/3", source="Yahoo Finance",
                        published=datetime.now().isoformat(), symbols=[symbol], relevance=0.8),
        ]

        agg = analyzer.aggregate_sentiment(symbol, sample_articles)
        events = detector.detect_from_articles(sample_articles)

        return {
            "sentiment": agg,
            "events": events,
            "articles": sample_articles,
        }
    except Exception as exc:
        logger.error(f"[dashboard] Sentiment fejl: {exc}")
        return {"sentiment": None, "events": [], "articles": []}


def page_sentiment():
    """Sentiment-side med nyhedsfeed, events og trend."""
    try:
        data = _get_sentiment_data("AAPL")
    except Exception as exc:
        logger.warning(f"Sentiment data fejl: {exc}")
        data = {"sentiment": None, "events": [], "articles": []}
    agg = data.get("sentiment")
    events = data.get("events", [])
    articles = data.get("articles", [])

    # Sentiment KPI
    if agg and agg.article_count > 0:
        score_color = "#00d4aa" if agg.score > 0.1 else "#ff4757" if agg.score < -0.1 else "#ffa502"
        sentiment_card = dbc.Card([
            dbc.CardBody([
                html.H5(t('analysis.overall_sentiment'), className="text-muted"),
                html.H2(f"{agg.score:+.2f}", style={"color": score_color, "fontSize": "3rem"}),
                html.P(agg.label.upper(), className="mb-1",
                       style={"color": score_color, "fontWeight": "bold"}),
                html.Small(f"{agg.article_count} artikler analyseret"),
            ])
        ], className="bg-dark text-light text-center p-3")
    else:
        sentiment_card = dbc.Card(dbc.CardBody(
            html.P(t('analysis.no_sentiment_data'), className="text-muted text-center")
        ), className="bg-dark text-light")

    # Distribution donut
    if agg and agg.article_count > 0:
        fig_dist = go.Figure(data=[go.Pie(
            labels=[t('analysis.positive'), t('analysis.negative'), t('analysis.neutral')],
            values=[agg.positive_count, agg.negative_count, agg.neutral_count],
            marker=dict(colors=["#00d4aa", "#ff4757", "#ffa502"]),
            hole=0.6,
            textinfo="label+value",
        )])
        fig_dist.update_layout(
            template="plotly_dark", paper_bgcolor="#1a1c24",
            plot_bgcolor="#1a1c24", height=250,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        )
    else:
        fig_dist = go.Figure()
        fig_dist.update_layout(template="plotly_dark", height=250)

    # Nyhedsfeed
    news_items = []
    for a in articles[:10]:
        sentiment_text = "neutral"
        badge_color = "secondary"
        from src.sentiment import keyword_sentiment as _ks
        s = _ks(a.title)
        if s.score > 0.1:
            sentiment_text = "positiv"
            badge_color = "success"
        elif s.score < -0.1:
            sentiment_text = "negativ"
            badge_color = "danger"

        news_items.append(
            dbc.ListGroupItem([
                html.Div([
                    dbc.Badge(sentiment_text, color=badge_color, className="me-2"),
                    html.Small(a.source, className="text-muted me-2"),
                    html.A(a.title, href=a.url, target="_blank",
                           style={"color": "#e0e0e0", "textDecoration": "none"}),
                ]),
            ], className="bg-dark border-secondary")
        )

    # Events
    event_items = []
    for e in events[:5]:
        icon = "bi-check-circle" if e.sentiment.value == "bullish" else \
               "bi-x-circle" if e.sentiment.value == "bearish" else "bi-dash-circle"
        color = "#00d4aa" if e.sentiment.value == "bullish" else \
                "#ff4757" if e.sentiment.value == "bearish" else "#ffa502"
        event_items.append(
            html.Div([
                html.I(className=f"bi {icon} me-2", style={"color": color}),
                dbc.Badge(e.impact.value.upper(), color="light", className="me-2",
                          style={"fontSize": "0.7rem"}),
                html.Span(e.title, style={"color": "#e0e0e0"}),
            ], className="mb-2")
        )

    return dbc.Container([
        html.H3(t('analysis.sentiment_news'), className="text-light mb-4"),
        dbc.Row([
            dbc.Col(sentiment_card, md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5(t('analysis.sentiment_distribution'), className="text-muted"),
                dcc.Graph(figure=fig_dist, config={"displayModeBar": False}),
            ]), className="bg-dark text-light"), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5(t('analysis.detected_events'), className="text-muted"),
                html.Div(event_items if event_items else [
                    html.P(t('analysis.no_events_detected'), className="text-muted")
                ]),
            ]), className="bg-dark text-light"), md=4),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5(t('analysis.news_feed'), className="text-muted mb-3"),
                dbc.ListGroup(news_items if news_items else [
                    dbc.ListGroupItem(t('analysis.no_news'), className="bg-dark text-muted")
                ]),
            ]), className="bg-dark text-light"), md=12),
        ]),
        html.Div([
            html.Hr(className="border-secondary mt-4"),
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                t('analysis.sentiment_note'),
            ], className="text-muted small"),
        ]),
    ], fluid=True, className="p-4")


# ══════════════════════════════════════════════════════════════
#  Side 7 – Kalender (Earnings + Makro)
# ══════════════════════════════════════════════════════════════

def _get_calendar_data() -> dict:
    """Hent kalender-data (demo-data hvis ingen API-keys)."""
    from datetime import timedelta

    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # Demo earnings events
    earnings_events = [
        {"symbol": "AAPL", "date": (now + timedelta(days=2)).strftime("%Y-%m-%d"),
         "hour": "amc", "eps_estimate": 2.35, "days_until": 2, "type": "earnings"},
        {"symbol": "MSFT", "date": (now + timedelta(days=5)).strftime("%Y-%m-%d"),
         "hour": "bmo", "eps_estimate": 3.10, "days_until": 5, "type": "earnings"},
        {"symbol": "GOOGL", "date": (now + timedelta(days=8)).strftime("%Y-%m-%d"),
         "hour": "amc", "eps_estimate": 1.85, "days_until": 8, "type": "earnings"},
        {"symbol": "NVDA", "date": (now + timedelta(days=12)).strftime("%Y-%m-%d"),
         "hour": "amc", "eps_estimate": 0.82, "days_until": 12, "type": "earnings"},
        {"symbol": "TSLA", "date": (now + timedelta(days=15)).strftime("%Y-%m-%d"),
         "hour": "amc", "eps_estimate": 0.73, "days_until": 15, "type": "earnings"},
        {"symbol": "META", "date": (now + timedelta(days=18)).strftime("%Y-%m-%d"),
         "hour": "amc", "eps_estimate": 4.71, "days_until": 18, "type": "earnings"},
        # Nylige (rapporterede)
        {"symbol": "AMZN", "date": (now - timedelta(days=3)).strftime("%Y-%m-%d"),
         "hour": "amc", "eps_estimate": 1.14, "eps_actual": 1.29,
         "surprise_pct": 0.132, "days_until": -3, "type": "earnings"},
        {"symbol": "JPM", "date": (now - timedelta(days=5)).strftime("%Y-%m-%d"),
         "hour": "bmo", "eps_estimate": 4.01, "eps_actual": 3.76,
         "surprise_pct": -0.062, "days_until": -5, "type": "earnings"},
    ]

    # Demo makro events
    macro_events = [
        {"name": "FOMC Rate Decision", "date": (now + timedelta(days=4)).strftime("%Y-%m-%d"),
         "impact": "critical", "type": "macro", "event_type": "fomc", "days_until": 4},
        {"name": "Non-Farm Payrolls", "date": (now + timedelta(days=9)).strftime("%Y-%m-%d"),
         "impact": "critical", "type": "macro", "event_type": "nfp", "days_until": 9},
        {"name": "CPI (YoY)", "date": (now + timedelta(days=14)).strftime("%Y-%m-%d"),
         "impact": "high", "type": "macro", "event_type": "cpi",
         "estimate": 3.1, "days_until": 14},
        {"name": "GDP Growth Rate", "date": (now + timedelta(days=20)).strftime("%Y-%m-%d"),
         "impact": "high", "type": "macro", "event_type": "gdp",
         "estimate": 2.4, "days_until": 20},
        {"name": "ISM Manufacturing PMI", "date": (now + timedelta(days=6)).strftime("%Y-%m-%d"),
         "impact": "medium", "type": "macro", "event_type": "pmi",
         "estimate": 50.2, "days_until": 6},
        # Nyligt
        {"name": "PPI (MoM)", "date": (now - timedelta(days=2)).strftime("%Y-%m-%d"),
         "impact": "medium", "type": "macro", "event_type": "ppi",
         "estimate": 0.3, "actual": 0.4, "previous": 0.2, "days_until": -2},
    ]

    return {
        "earnings": earnings_events,
        "macro": macro_events,
        "today": today_str,
    }


def page_calendar():
    """Kalender-side med earnings og makro-events."""
    data = _get_calendar_data()
    now = datetime.now()

    # ── Countdown til næste store event ──
    all_upcoming = []
    for e in data["earnings"]:
        if e.get("days_until", 0) > 0:
            all_upcoming.append({"name": f"{e['symbol']} Earnings", "days": e["days_until"],
                                  "type": "earnings", "date": e["date"]})
    for m in data["macro"]:
        if m.get("days_until", 0) > 0:
            all_upcoming.append({"name": m["name"], "days": m["days_until"],
                                  "type": "macro", "date": m["date"]})

    all_upcoming.sort(key=lambda x: x["days"])
    next_event = all_upcoming[0] if all_upcoming else None

    # Countdown-kort
    if next_event:
        countdown_color = COLORS["red"] if next_event["days"] <= 1 else \
                          COLORS["orange"] if next_event["days"] <= 3 else COLORS["accent"]
        countdown_card = dbc.Card(dbc.CardBody([
            html.Div([
                html.I(className="bi bi-alarm-fill me-2",
                       style={"fontSize": "1.5rem", "color": countdown_color}),
                html.Span(t('calendar.countdown'), style={"color": COLORS["muted"],
                           "fontSize": "0.8rem", "letterSpacing": "2px"}),
            ], className="d-flex align-items-center mb-3"),
            html.H1(f"{next_event['days']}",
                     style={"fontSize": "4rem", "fontWeight": "800",
                             "color": countdown_color, "lineHeight": "1", "margin": 0}),
            html.P(t('analysis.days'), style={"color": COLORS["muted"], "fontSize": "1.2rem", "margin": 0}),
            html.Hr(style={"borderColor": COLORS["border"]}),
            html.P(next_event["name"], style={"color": COLORS["text"],
                    "fontWeight": "600", "fontSize": "1.1rem", "margin": 0}),
            html.P(next_event["date"], style={"color": COLORS["muted"], "fontSize": "0.9rem"}),
        ]), className="bg-dark text-light text-center",
            style={"border": f"1px solid {countdown_color}40"})
    else:
        countdown_card = dbc.Card(dbc.CardBody([
            html.P(t('analysis.no_upcoming_events'), className="text-muted text-center"),
        ]), className="bg-dark text-light")

    # ── Earnings-tabel ──
    upcoming_earnings = [e for e in data["earnings"] if e.get("days_until", 0) > 0]
    recent_earnings = [e for e in data["earnings"] if e.get("eps_actual") is not None]

    earnings_rows = []
    for e in upcoming_earnings:
        hour_label = t('calendar.before_market_open') if e["hour"] == "bmo" else \
                     t('calendar.after_market_close') if e["hour"] == "amc" else e["hour"]
        urgency_color = COLORS["red"] if e["days_until"] <= 1 else \
                        COLORS["orange"] if e["days_until"] <= 3 else \
                        COLORS["accent"] if e["days_until"] <= 7 else COLORS["muted"]
        earnings_rows.append(html.Tr([
            html.Td(html.Span(f"{e['days_until']}d", style={"color": urgency_color,
                     "fontWeight": "700"})),
            html.Td(html.Span(e["symbol"], style={"fontWeight": "600", "color": COLORS["text"]})),
            html.Td(e["date"], style={"color": COLORS["muted"]}),
            html.Td(hour_label, style={"color": COLORS["muted"]}),
            html.Td(f"${e['eps_estimate']:.2f}" if e.get("eps_estimate") else "–",
                     style={"color": COLORS["blue"]}),
        ]))

    earnings_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th(t('common.days'), style={"color": COLORS["muted"]}),
            html.Th(t('common.symbol'), style={"color": COLORS["muted"]}),
            html.Th(t('common.date'), style={"color": COLORS["muted"]}),
            html.Th(t('common.timing'), style={"color": COLORS["muted"]}),
            html.Th("EPS Est.", style={"color": COLORS["muted"]}),
        ])),
        html.Tbody(earnings_rows),
    ], bordered=False, hover=True, responsive=True, className="table-dark mb-0")

    # ── Nylige earnings-resultater ──
    recent_rows = []
    for e in recent_earnings:
        surprise = e.get("surprise_pct", 0)
        icon = "bi-check-circle-fill" if surprise > 0.02 else \
               "bi-x-circle-fill" if surprise < -0.02 else "bi-dash-circle"
        icon_color = COLORS["green"] if surprise > 0.02 else \
                     COLORS["red"] if surprise < -0.02 else COLORS["orange"]
        recent_rows.append(html.Tr([
            html.Td(html.I(className=f"bi {icon}", style={"color": icon_color})),
            html.Td(e["symbol"], style={"fontWeight": "600", "color": COLORS["text"]}),
            html.Td(e["date"], style={"color": COLORS["muted"]}),
            html.Td(f"${e.get('eps_actual', 0):.2f}", style={"color": COLORS["text"]}),
            html.Td(f"${e.get('eps_estimate', 0):.2f}", style={"color": COLORS["muted"]}),
            html.Td(f"{surprise:+.1%}", style={"color": icon_color, "fontWeight": "600"}),
        ]))

    recent_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("", style={"color": COLORS["muted"]}),
            html.Th(t('common.symbol'), style={"color": COLORS["muted"]}),
            html.Th(t('common.date'), style={"color": COLORS["muted"]}),
            html.Th("EPS Actual", style={"color": COLORS["muted"]}),
            html.Th("EPS Est.", style={"color": COLORS["muted"]}),
            html.Th("Surprise", style={"color": COLORS["muted"]}),
        ])),
        html.Tbody(recent_rows),
    ], bordered=False, hover=True, responsive=True, className="table-dark mb-0")

    # ── Makro-event-tabel ──
    upcoming_macro = [m for m in data["macro"] if m.get("days_until", 0) > 0]
    recent_macro = [m for m in data["macro"] if m.get("actual") is not None]

    impact_badge = {
        "critical": {"color": "danger", "icon": "bi-exclamation-triangle-fill"},
        "high": {"color": "warning", "icon": "bi-exclamation-circle-fill"},
        "medium": {"color": "info", "icon": "bi-info-circle-fill"},
        "low": {"color": "secondary", "icon": "bi-circle"},
    }

    macro_rows = []
    for m in upcoming_macro:
        badge = impact_badge.get(m["impact"], impact_badge["low"])
        urgency_color = COLORS["red"] if m["days_until"] <= 1 else \
                        COLORS["orange"] if m["days_until"] <= 3 else \
                        COLORS["accent"] if m["days_until"] <= 7 else COLORS["muted"]
        macro_rows.append(html.Tr([
            html.Td(html.Span(f"{m['days_until']}d", style={"color": urgency_color,
                     "fontWeight": "700"})),
            html.Td([
                html.I(className=f"bi {badge['icon']} me-1",
                       style={"color": COLORS["red"] if m["impact"] == "critical" else
                                        COLORS["orange"] if m["impact"] == "high" else
                                        COLORS["blue"]}),
                html.Span(m["name"], style={"color": COLORS["text"], "fontWeight": "500"}),
            ]),
            html.Td(m["date"], style={"color": COLORS["muted"]}),
            html.Td(dbc.Badge(m["impact"].upper(), color=badge["color"],
                     style={"fontSize": "0.7rem"})),
            html.Td(f"{m.get('estimate', '–')}" if m.get("estimate") else "–",
                     style={"color": COLORS["blue"]}),
        ]))

    macro_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th(t('common.days'), style={"color": COLORS["muted"]}),
            html.Th("Event", style={"color": COLORS["muted"]}),
            html.Th(t('common.date'), style={"color": COLORS["muted"]}),
            html.Th("Impact", style={"color": COLORS["muted"]}),
            html.Th(t('common.expected'), style={"color": COLORS["muted"]}),
        ])),
        html.Tbody(macro_rows),
    ], bordered=False, hover=True, responsive=True, className="table-dark mb-0")

    # ── Timeline-figur ──
    timeline_data = []
    for e in upcoming_earnings:
        timeline_data.append({
            "Dato": e["date"], "Event": f"{e['symbol']} Earnings",
            "Dage": e["days_until"], "Type": "Earnings",
            "Impact": 3,
        })
    for m in upcoming_macro:
        imp_val = {"critical": 5, "high": 4, "medium": 3, "low": 2}.get(m["impact"], 2)
        timeline_data.append({
            "Dato": m["date"], "Event": m["name"],
            "Dage": m["days_until"], "Type": "Makro",
            "Impact": imp_val,
        })

    if timeline_data:
        tl_df = pd.DataFrame(timeline_data)
        fig_timeline = px.scatter(
            tl_df, x="Dato", y="Impact", color="Type", size="Impact",
            hover_data=["Event", "Dage"],
            color_discrete_map={"Earnings": COLORS["blue"], "Makro": COLORS["orange"]},
            template=DARK_TEMPLATE,
        )
        fig_timeline.update_layout(
            plot_bgcolor="#1a1c24",
            paper_bgcolor="#1a1c24",
            margin=dict(l=20, r=20, t=30, b=20),
            height=200,
            yaxis_title=t('charts.importance'),
            xaxis_title="",
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=1.15, x=0.5, xanchor="center"),
        )
        fig_timeline.update_yaxes(showgrid=False, zeroline=False)
    else:
        fig_timeline = go.Figure()
        fig_timeline.update_layout(
            template=DARK_TEMPLATE,
            plot_bgcolor="#1a1c24",
            paper_bgcolor="#1a1c24",
            height=200,
        )

    # ── Risiko-advarsler ──
    risk_warnings = []
    for e in upcoming_earnings:
        if e["days_until"] <= 1:
            risk_warnings.append(
                dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle-fill me-2"),
                    f"⚠️ {e['symbol']} rapporterer ",
                    html.Strong(f"i morgen" if e["days_until"] == 1 else "i dag"),
                    f" ({e['hour'].upper()}) – Anbefalet: reducér position 50%",
                ], color="warning", className="mb-2")
            )
    for m in upcoming_macro:
        if m["days_until"] <= 1 and m["impact"] in ("critical", "high"):
            risk_warnings.append(
                dbc.Alert([
                    html.I(className="bi bi-exclamation-triangle-fill me-2"),
                    f"🔴 {m['name']} ",
                    html.Strong(f"i morgen" if m["days_until"] == 1 else "i dag"),
                    f" – Anbefalet: reducér eksponering {25 if m['impact'] == 'critical' else 15}%",
                ], color="danger", className="mb-2")
            )

    return dbc.Container([
        html.H3([
            html.I(className="bi bi-calendar-event me-2"),
            t('calendar.event_calendar'),
        ], className="text-light mb-4"),

        # Risiko-advarsler
        html.Div(risk_warnings) if risk_warnings else html.Div(),

        # Top row: countdown + timeline
        dbc.Row([
            dbc.Col(countdown_card, md=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6(t('analysis.event_timeline'), className="text-muted mb-2"),
                dcc.Graph(figure=fig_timeline, config={"displayModeBar": False}),
            ]), className="bg-dark text-light"), md=9),
        ], className="mb-4"),

        # Earnings-sektion
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5([
                        html.I(className="bi bi-bar-chart-line me-2",
                               style={"color": COLORS["blue"]}),
                        t('calendar.upcoming_earnings'),
                    ], className="text-light mb-3"),
                    earnings_table if earnings_rows else html.P(
                        t('analysis.no_upcoming_earnings'), className="text-muted"),
                ]), className="bg-dark text-light"),
            ], md=7),
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5([
                        html.I(className="bi bi-clipboard-data me-2",
                               style={"color": COLORS["accent"]}),
                        t('calendar.recent_results'),
                    ], className="text-light mb-3"),
                    recent_table if recent_rows else html.P(
                        t('analysis.no_recent_results'), className="text-muted"),
                ]), className="bg-dark text-light"),
            ], md=5),
        ], className="mb-4"),

        # Makro-sektion
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody([
                    html.H5([
                        html.I(className="bi bi-bank me-2",
                               style={"color": COLORS["orange"]}),
                        t('calendar.macro_events'),
                    ], className="text-light mb-3"),
                    macro_table if macro_rows else html.P(
                        t('analysis.no_macro_events'), className="text-muted"),
                ]), className="bg-dark text-light"),
            ], md=12),
        ], className="mb-4"),

        # Handelsregler
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5([
                    html.I(className="bi bi-shield-check me-2",
                           style={"color": COLORS["accent"]}),
                    t('calendar.active_trading_rules'),
                ], className="text-light mb-3"),
                html.Div([
                    html.Div([
                        html.I(className="bi bi-arrow-down-circle me-1",
                               style={"color": COLORS["orange"]}),
                        html.Strong(f"{t('calendar.earnings_rule')}: ", style={"color": COLORS["text"]}),
                        html.Span(
                            t('calendar.earnings_rule_desc'),
                            style={"color": COLORS["muted"]}),
                    ], className="mb-2"),
                    html.Div([
                        html.I(className="bi bi-arrow-down-circle me-1",
                               style={"color": COLORS["red"]}),
                        html.Strong(f"{t('calendar.fomc_rule')}: ", style={"color": COLORS["text"]}),
                        html.Span(
                            t('calendar.fomc_rule_desc'),
                            style={"color": COLORS["muted"]}),
                    ], className="mb-2"),
                    html.Div([
                        html.I(className="bi bi-arrow-down-circle me-1",
                               style={"color": COLORS["orange"]}),
                        html.Strong(f"{t('calendar.nfp_cpi_rule')}: ", style={"color": COLORS["text"]}),
                        html.Span(
                            t('calendar.nfp_cpi_rule_desc'),
                            style={"color": COLORS["muted"]}),
                    ], className="mb-2"),
                ]),
            ]), className="bg-dark text-light"), md=12),
        ]),

        html.Div([
            html.Hr(className="border-secondary mt-4"),
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                t('calendar.calendar_demo_note'),
            ], className="text-muted small"),
        ]),
    ], fluid=True, className="p-4")


# ══════════════════════════════════════════════════════════════
#  Side 8 – Regime-detektion
# ══════════════════════════════════════════════════════════════

def _get_regime_data() -> dict:
    """Beregn regime-data fra S&P 500 / porteføljesymbol."""
    from src.strategy.regime import (
        RegimeDetector, AdaptiveStrategy, MarketRegime, REGIME_INFO,
    )
    from src.strategy.base_strategy import StrategyResult, Signal

    spy_df = None
    try:
        fetcher = MarketDataFetcher()
        spy_df = fetcher.get_historical("SPY", lookback_days=500)
    except Exception:
        pass
    # Fallback: try yfinance directly
    if spy_df is None or spy_df.empty:
        try:
            import yfinance as _yf
            _tk = _yf.Ticker("SPY")
            spy_df = _tk.history(period="2y")
        except Exception:
            spy_df = None

    detector = RegimeDetector()
    adaptive = AdaptiveStrategy(detector=detector)

    if spy_df is not None and not spy_df.empty and len(spy_df) >= 50:
        result = adaptive.analyze(spy_df)
        regime_result = adaptive.last_regime_result

        # Historisk regime-sekvens
        hist_df = detector.get_regime_history(spy_df, step=5)
    else:
        # Demo-data
        regime_result = None
        hist_df = pd.DataFrame()
        result = StrategyResult(signal=Signal.HOLD, confidence=0, reason="Ingen data")

    summary = adaptive.get_regime_summary()

    return {
        "regime_result": regime_result,
        "strategy_result": result,
        "summary": summary,
        "history_df": hist_df,
        "spy_df": spy_df,
    }


def page_regime():
    """Regime-detektionsside med badge, chart og tilpasninger."""
    from src.strategy.regime import MarketRegime, REGIME_INFO

    data = _get_regime_data()
    summary = data["summary"]
    regime_result = data["regime_result"]
    hist_df = data["history_df"]
    spy_df = data["spy_df"]

    # ── Regime badge ──
    if regime_result is not None:
        regime = regime_result.regime
        info = REGIME_INFO[regime]
        badge_color = info["color"]
        badge_label = info["label"]
        badge_icon = info["icon"]
        confidence = regime_result.confidence
        composite = regime_result.composite_score
    else:
        badge_color = COLORS["muted"]
        badge_label = t('regime.unknown')
        badge_icon = "bi-question-circle"
        confidence = 0
        composite = 0

    regime_badge = dbc.Card(dbc.CardBody([
        html.Div([
            html.I(className=f"bi {badge_icon}",
                   style={"fontSize": "3rem", "color": badge_color}),
        ], className="text-center mb-2"),
        html.H2(badge_label, className="text-center mb-1",
                 style={"color": badge_color, "fontWeight": "800", "letterSpacing": "2px"}),
        html.Div([
            html.Span(f"Confidence: {confidence:.0f}%",
                       style={"color": COLORS["text"], "fontSize": "1.1rem"}),
        ], className="text-center mb-2"),
        html.Div([
            html.Span(f"Score: {composite:+.2f}",
                       style={"color": COLORS["muted"], "fontSize": "0.9rem"}),
        ], className="text-center"),
        # Confidence-bar
        dbc.Progress(
            value=confidence, max=100,
            color="success" if confidence > 70 else "warning" if confidence > 40 else "danger",
            className="mt-3",
            style={"height": "8px"},
        ),
    ]), className="bg-dark text-light",
        style={"border": f"2px solid {badge_color}"})

    # ── Signal-breakdown ──
    signal_cards = []
    for sig in summary.get("signals", []):
        val = sig["value"]
        color = COLORS["green"] if val > 0.2 else COLORS["red"] if val < -0.2 else COLORS["orange"]
        bar_pct = (val + 1) / 2 * 100  # Normaliser -1..1 til 0..100
        signal_cards.append(
            html.Div([
                html.Div([
                    html.Span(sig["name"].capitalize(),
                              style={"color": COLORS["text"], "fontWeight": "600", "width": "90px",
                                      "display": "inline-block"}),
                    html.Span(f"{val:+.2f}", style={"color": color, "fontWeight": "700",
                               "width": "60px", "display": "inline-block", "textAlign": "right"}),
                ], className="d-flex justify-content-between align-items-center mb-1"),
                dbc.Progress(
                    value=bar_pct, max=100,
                    color="success" if val > 0.2 else "danger" if val < -0.2 else "warning",
                    style={"height": "6px"},
                ),
                html.P(sig.get("detail", ""), className="text-muted mb-2",
                       style={"fontSize": "0.75rem"}),
            ], className="mb-1")
        )

    # ── Historisk regime-chart ──
    if hist_df is not None and not hist_df.empty:
        regime_colors = {
            "bull": "#2ed573", "bear": "#ff4757", "sideways": "#ffa502",
            "crash": "#ff0000", "recovery": "#3498db", "euphoria": "#a855f7",
        }
        fig_history = go.Figure()

        # Baggrund: farvede zoner per regime
        if "date" in hist_df.columns and "regime" in hist_df.columns:
            for regime_val, color in regime_colors.items():
                mask = hist_df["regime"] == regime_val
                if mask.any():
                    subset = hist_df[mask]
                    fig_history.add_trace(go.Scatter(
                        x=subset["date"], y=subset["composite_score"],
                        mode="markers", name=regime_val.upper(),
                        marker=dict(color=color, size=6),
                    ))

        fig_history.update_layout(
            template=DARK_TEMPLATE,
            plot_bgcolor="#1a1c24",
            paper_bgcolor="#1a1c24",
            margin=dict(l=20, r=20, t=30, b=20),
            height=300,
            yaxis_title=t('charts.composite_score'),
            xaxis_title="",
            legend=dict(orientation="h", yanchor="top", y=1.15, x=0.5, xanchor="center"),
        )
        fig_history.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], opacity=0.5)
        fig_history.add_hline(y=0.2, line_dash="dot", line_color="#2ed57380", opacity=0.3)
        fig_history.add_hline(y=-0.3, line_dash="dot", line_color="#ff475780", opacity=0.3)
    else:
        fig_history = go.Figure()
        fig_history.update_layout(
            template=DARK_TEMPLATE,
            plot_bgcolor="#1a1c24",
            paper_bgcolor="#1a1c24",
            height=300,
            annotations=[dict(text=t('regime.no_historical_data'), xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False, font=dict(color=COLORS["muted"]))],
        )

    # ── Strategi-tilpasninger ──
    adj_items = []
    if summary.get("regime") != "unknown":
        max_exp = summary.get("max_exposure", 0)
        strategies = summary.get("preferred_strategies", [])
        sectors = summary.get("preferred_sectors", [])
        avoid = summary.get("avoid_sectors", [])
        havens = summary.get("safe_havens", [])
        notes = summary.get("notes", [])

        adj_items.append(html.Div([
            html.Div([
                html.I(className="bi bi-pie-chart me-2", style={"color": COLORS["accent"]}),
                html.Strong(f"{t('risk.max_exposure_label')}: ", style={"color": COLORS["text"]}),
                html.Span(f"{max_exp:.0%}",
                          style={"color": badge_color, "fontWeight": "700", "fontSize": "1.2rem"}),
            ], className="mb-3"),
        ]))

        if strategies:
            adj_items.append(html.Div([
                html.I(className="bi bi-robot me-2", style={"color": COLORS["blue"]}),
                html.Strong(f"{t('risk.recommended_strategies')}: ", style={"color": COLORS["text"]}),
                html.Span(", ".join(strategies), style={"color": COLORS["muted"]}),
            ], className="mb-2"))

        if sectors:
            adj_items.append(html.Div([
                html.I(className="bi bi-building me-2", style={"color": COLORS["green"]}),
                html.Strong(f"{t('risk.favor_sectors')}: ", style={"color": COLORS["text"]}),
                html.Span(", ".join(sectors), style={"color": COLORS["muted"]}),
            ], className="mb-2"))

        if avoid:
            adj_items.append(html.Div([
                html.I(className="bi bi-x-circle me-2", style={"color": COLORS["red"]}),
                html.Strong(f"{t('risk.avoid_sectors')}: ", style={"color": COLORS["text"]}),
                html.Span(", ".join(avoid), style={"color": COLORS["muted"]}),
            ], className="mb-2"))

        if havens:
            adj_items.append(html.Div([
                html.I(className="bi bi-shield-check me-2", style={"color": COLORS["orange"]}),
                html.Strong(f"{t('risk.safe_havens')}: ", style={"color": COLORS["text"]}),
                html.Span(", ".join(havens), style={"color": COLORS["muted"]}),
            ], className="mb-2"))

        if notes:
            adj_items.append(html.Hr(style={"borderColor": COLORS["border"]}))
            for note in notes:
                adj_items.append(html.Div([
                    html.I(className="bi bi-arrow-right me-2", style={"color": COLORS["accent"]}),
                    html.Span(note, style={"color": COLORS["muted"]}),
                ], className="mb-1"))

    # ── Regime-skift log ──
    shift_items = []
    for sh in summary.get("shifts", [])[-5:]:
        shift_items.append(html.Div([
            html.Span(sh["timestamp"][:19], style={"color": COLORS["muted"],
                       "fontSize": "0.8rem", "marginRight": "8px"}),
            dbc.Badge(sh["from"].upper(),
                      color="success" if sh["from"] == "bull" else
                            "danger" if sh["from"] in ("bear", "crash") else "warning",
                      className="me-1"),
            html.I(className="bi bi-arrow-right mx-1", style={"color": COLORS["muted"]}),
            dbc.Badge(sh["to"].upper(),
                      color="success" if sh["to"] == "bull" else
                            "danger" if sh["to"] in ("bear", "crash") else "warning"),
            html.Span(f" ({sh.get('confidence', 0):.0f}%)",
                       style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
        ], className="mb-2"))

    return dbc.Container([
        html.H3([
            html.I(className="bi bi-activity me-2"),
            t('regime.detection_title'),
        ], className="text-light mb-4"),

        # Top row: regime badge + signaler
        dbc.Row([
            dbc.Col(regime_badge, md=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5(t('analysis.signal_breakdown'), className="text-muted mb-3"),
                html.Div(signal_cards if signal_cards else [
                    html.P(t('regime.no_signals'), className="text-muted"),
                ]),
            ]), className="bg-dark text-light"), md=5),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5(t('analysis.strategy_adjustments'), className="text-muted mb-3"),
                html.Div(adj_items if adj_items else [
                    html.P(t('regime.no_adjustments'), className="text-muted"),
                ]),
            ]), className="bg-dark text-light"), md=4),
        ], className="mb-4"),

        # Historisk regime chart
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5(t('analysis.historical_regime'), className="text-muted mb-2"),
                dcc.Graph(figure=fig_history, config={"displayModeBar": False}),
            ]), className="bg-dark text-light"), md=12),
        ], className="mb-4"),

        # Regime-skift log
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5([
                    html.I(className="bi bi-clock-history me-2"),
                    t('regime.shift_log'),
                ], className="text-muted mb-3"),
                html.Div(shift_items if shift_items else [
                    html.P(t('regime.no_shifts'), className="text-muted"),
                ]),
            ]), className="bg-dark text-light"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5([
                    html.I(className="bi bi-info-circle me-2"),
                    t('regime.types_title'),
                ], className="text-muted mb-3"),
                html.Div([
                    html.Div([
                        dbc.Badge("BULL", style={"backgroundColor": "#2ed573"}, className="me-2"),
                        html.Span(t('regime.bull_desc'),
                                  style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
                    ], className="mb-2"),
                    html.Div([
                        dbc.Badge("BEAR", style={"backgroundColor": "#ff4757"}, className="me-2"),
                        html.Span(t('regime.bear_desc'),
                                  style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
                    ], className="mb-2"),
                    html.Div([
                        dbc.Badge("SIDEWAYS", style={"backgroundColor": "#ffa502"}, className="me-2"),
                        html.Span(t('regime.sideways_desc'),
                                  style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
                    ], className="mb-2"),
                    html.Div([
                        dbc.Badge("CRASH", style={"backgroundColor": "#ff0000"}, className="me-2"),
                        html.Span(t('regime.crash_desc'),
                                  style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
                    ], className="mb-2"),
                    html.Div([
                        dbc.Badge("RECOVERY", style={"backgroundColor": "#3498db"}, className="me-2"),
                        html.Span(t('regime.recovery_desc'),
                                  style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
                    ], className="mb-2"),
                    html.Div([
                        dbc.Badge("EUPHORIA", style={"backgroundColor": "#a855f7"}, className="me-2"),
                        html.Span(t('regime.euphoria_desc'),
                                  style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
                    ], className="mb-2"),
                ]),
            ]), className="bg-dark text-light"), md=6),
        ]),

        html.Div([
            html.Hr(className="border-secondary mt-4"),
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                t('regime.regime_note'),
            ], className="text-muted small"),
        ]),
    ], fluid=True, className="p-4")


# ══════════════════════════════════════════════════════════════
#  SIDE 10 – Stress Test
# ══════════════════════════════════════════════════════════════


def _get_stress_test_data() -> dict:
    """Cached stress-test resultater."""
    key = "stress_test"
    if key in _cache:
        return _cache[key]

    from src.backtest.stress_test import (
        StressTester, HISTORICAL_CRISES, SYNTHETIC_SCENARIOS,
    )

    # Byg portfolio fra SYMBOLS
    weights = {s: 1.0 / len(SYMBOLS) for s in SYMBOLS}
    tester = StressTester(portfolio_weights=weights, initial_value=100_000)
    report = tester.run_all(include_monte_carlo=True, monte_carlo_runs=5000)

    _cache[key] = {
        "report": report,
        "tester": tester,
        "historical_keys": list(HISTORICAL_CRISES.keys()),
        "synthetic_keys": list(SYNTHETIC_SCENARIOS.keys()),
    }
    return _cache[key]


def page_stress_test():
    """Stress-test side med krisescenarier og Monte Carlo."""
    try:
        data = _get_stress_test_data()
        report = data["report"]
    except Exception as exc:
        logger.error(f"Stress-test fejl: {exc}")
        return dbc.Container([
            html.H3("Stress Test", style={"color": COLORS["text"]}),
            dbc.Alert(f"{t('health.loading_error')}: {exc}", color="danger"),
        ], fluid=True, className="p-4")

    # ── Rating badge ───────────────────────────────────────
    rating_colors = {
        "LAV": COLORS["green"], "MIDDEL": COLORS["orange"],
        "HØJ": COLORS["red"], "KRITISK": "#ff0000",
    }
    rating_color = rating_colors.get(report.overall_risk_rating, COLORS["muted"])

    # ── Scenarie-oversigtstabel ──────────────────────────────
    scenario_rows = []
    for r in report.scenario_results:
        sev_badge = {
            "EKSTREM": "danger", "ALVORLIG": "warning",
            "MODERAT": "info", "LET": "success",
        }.get(r.scenario.severity, "secondary")

        saved_color = COLORS["green"] if r.risk_mgmt_saved_pct > 0 else COLORS["muted"]

        scenario_rows.append(html.Tr([
            html.Td(r.scenario.name, style={"color": COLORS["text"], "fontSize": "0.85rem"}),
            html.Td(dbc.Badge(r.scenario.severity, color=sev_badge, className="px-2")),
            html.Td(f"{r.max_drawdown_pct:+.1f}%",
                     style={"color": COLORS["red"], "fontWeight": "bold"}),
            html.Td(f"{r.worst_day_pct:+.1f}%", style={"color": COLORS["red"]}),
            html.Td(f"{r.recovery_days}d", style={"color": COLORS["muted"]}),
            html.Td(format_value(r.with_risk_mgmt_end),
                     style={"color": COLORS["accent"]}),
            html.Td(format_value(r.without_risk_mgmt_end),
                     style={"color": COLORS["muted"]}),
            html.Td(f"+{r.risk_mgmt_saved_pct:.1f}%",
                     style={"color": saved_color, "fontWeight": "bold"}),
        ]))

    scenario_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th(t('health.scenario'), style={"color": COLORS["accent"]}),
            html.Th(t('health.severity'), style={"color": COLORS["accent"]}),
            html.Th(t('health.max_loss'), style={"color": COLORS["accent"]}),
            html.Th(t('health.worst_day'), style={"color": COLORS["accent"]}),
            html.Th(t('health.recovery'), style={"color": COLORS["accent"]}),
            html.Th(t('health.with_rm'), style={"color": COLORS["accent"]}),
            html.Th(t('health.without_rm'), style={"color": COLORS["accent"]}),
            html.Th(t('health.rm_saved'), style={"color": COLORS["accent"]}),
        ])),
        html.Tbody(scenario_rows),
    ], bordered=True, hover=True, responsive=True,
       style={"backgroundColor": COLORS["card"]}, className="text-light")

    # ── Krisegraf: dag-for-dag for hvert scenarie ─────────────
    fig_crisis = go.Figure()
    color_cycle = [COLORS["red"], COLORS["orange"], COLORS["blue"],
                   COLORS["accent"], COLORS["purple"], "#ff6b81", "#ffd93d"]

    for i, r in enumerate(report.scenario_results[:7]):  # Kun historiske
        days = list(range(len(r.daily_values)))
        pct_values = [(v / r.portfolio_value_start - 1) * 100 for v in r.daily_values]
        fig_crisis.add_trace(go.Scatter(
            x=days, y=pct_values,
            mode="lines", name=r.scenario.name[:20],
            line=dict(color=color_cycle[i % len(color_cycle)], width=2),
        ))

    fig_crisis.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"])
    fig_crisis.add_hline(y=-20, line_dash="dash", line_color=COLORS["red"],
                         annotation_text="Bear Market (-20%)")
    _fig_layout(fig_crisis, t('health.portfolio_during_crises'))
    fig_crisis.update_layout(
        xaxis_title=t('charts.trading_days_from_crisis'),
        yaxis_title=t('charts.return_pct'),
        legend=dict(font=dict(size=10)),
    )

    # ── Med vs. Uden risikostyring ────────────────────────────
    scenario_names = [r.scenario.name[:18] for r in report.scenario_results]
    with_rm = [r.with_risk_mgmt_end for r in report.scenario_results]
    without_rm = [r.without_risk_mgmt_end for r in report.scenario_results]

    fig_comparison = go.Figure()
    fig_comparison.add_trace(go.Bar(
        x=scenario_names, y=with_rm,
        name=t('charts.with_risk_mgmt'),
        marker_color=COLORS["accent"],
    ))
    fig_comparison.add_trace(go.Bar(
        x=scenario_names, y=without_rm,
        name=t('charts.without_risk_mgmt'),
        marker_color=COLORS["red"],
        opacity=0.7,
    ))
    fig_comparison.add_hline(
        y=report.initial_value, line_dash="dot",
        line_color=COLORS["muted"],
        annotation_text=f"Start: {format_value(report.initial_value)}",
    )
    _fig_layout(fig_comparison, t('health.final_value_comparison'))
    fig_comparison.update_layout(
        barmode="group", xaxis_tickangle=-45,
        yaxis_title=t('charts.portfolio_value'),
    )

    # ── Monte Carlo histogram ─────────────────────────────────
    mc = report.monte_carlo
    mc_section = []
    if mc:
        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=mc.final_values, nbinsx=80,
            marker_color=COLORS["accent"], opacity=0.75,
            name=t('charts.final_value_dist'),
        ))
        fig_mc.add_vline(x=mc.initial_value, line_dash="dash",
                         line_color=COLORS["orange"],
                         annotation_text=t('charts.start_value'))
        fig_mc.add_vline(x=mc.median, line_dash="dot",
                         line_color=COLORS["green"],
                         annotation_text=t('charts.median'))
        fig_mc.add_vline(x=mc.percentile_5, line_dash="dot",
                         line_color=COLORS["red"],
                         annotation_text=t('charts.pctl_5th'))
        _mc_title = t('charts.mc_title').replace('{simulations}', f'{mc.num_simulations:,}').replace('{days}', str(mc.horizon_days))
        _fig_layout(fig_mc, _mc_title)
        fig_mc.update_layout(
            xaxis_title=t('charts.portfolio_value'),
            yaxis_title=t('charts.simulations'),
        )

        mc_section = [
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_mc))], className="mb-4"),
            dbc.Row([
                dbc.Col(_metric_card(t('charts.median'), format_value(mc.median),
                                     f"{(mc.median / mc.initial_value - 1) * 100:+.1f}%",
                                     COLORS["accent"]), md=2),
                dbc.Col(_metric_card(t('charts.worst_1pct'), format_value(mc.worst_case),
                                     f"{(mc.worst_case / mc.initial_value - 1) * 100:+.1f}%",
                                     COLORS["red"]), md=2),
                dbc.Col(_metric_card(t('charts.best_99pct'), format_value(mc.best_case),
                                     f"{(mc.best_case / mc.initial_value - 1) * 100:+.1f}%",
                                     COLORS["green"]), md=2),
                dbc.Col(_metric_card("VaR (95%)", format_value(mc.var_95), "",
                                     COLORS["orange"]), md=2),
                dbc.Col(_metric_card(t('charts.prob_loss'), f"{mc.prob_loss_pct:.1f}%", "",
                                     COLORS["red"]), md=2),
                dbc.Col(_metric_card(t('charts.prob_loss_20'), f"{mc.prob_loss_20_pct:.1f}%", "",
                                     COLORS["red"] if mc.prob_loss_20_pct > 10 else COLORS["muted"]),
                        md=2),
            ], className="mb-4"),
        ]

    # ── Sårbarhedssektion ─────────────────────────────────────
    vuln_cards = []
    for v in report.vulnerabilities:
        sev_color = {"HØJ": COLORS["red"], "MIDDEL": COLORS["orange"],
                     "LAV": COLORS["green"]}.get(v.severity, COLORS["muted"])
        vuln_cards.append(dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                dbc.Badge(v.severity, style={"backgroundColor": sev_color},
                          className="me-2"),
                html.Strong(v.area, style={"color": COLORS["text"]}),
            ]),
            html.P(v.description,
                   style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                   className="mt-2 mb-1"),
            html.P([
                html.I(className="bi bi-arrow-right me-1"),
                v.recommendation,
            ], style={"color": COLORS["accent"], "fontSize": "0.85rem"}),
        ]), style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
        }), md=6, className="mb-3"))

    # ── Regime-aktioner under kriser ──────────────────────────
    regime_rows = []
    for r in report.scenario_results[:7]:  # Historiske
        actions_text = " → ".join(r.regime_actions[:3])
        if len(r.regime_actions) > 3:
            actions_text += f" (+{len(r.regime_actions) - 3} mere)"
        regime_rows.append(html.Tr([
            html.Td(r.scenario.name[:25], style={"color": COLORS["text"],
                                                   "fontSize": "0.85rem"}),
            html.Td(f"{r.risk_mgmt_saved_pct:+.1f}%",
                     style={"color": COLORS["green"] if r.risk_mgmt_saved_pct > 0
                            else COLORS["muted"], "fontWeight": "bold"}),
            html.Td(actions_text, style={"color": COLORS["muted"],
                                          "fontSize": "0.8rem"}),
        ]))

    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H3([
                    html.I(className="bi bi-lightning-charge me-2"),
                    "Stress Test",
                ], style={"color": COLORS["text"]}),
                html.P(t('analysis.test_portfolio'),
                       style={"color": COLORS["muted"]}),
            ], md=8),
            dbc.Col([
                html.Div([
                    html.Span(t('analysis.risk_rating'), style={
                        "color": COLORS["muted"], "fontSize": "0.75rem",
                        "display": "block",
                    }),
                    html.H2(report.overall_risk_rating, style={
                        "color": rating_color, "fontWeight": "800",
                        "margin": "4px 0",
                    }),
                ], style={"textAlign": "right"}),
            ], md=4),
        ], className="mb-4"),

        # KPI-kort
        dbc.Row([
            dbc.Col(_metric_card(t('health.scenarios_tested'),
                                 str(len(report.scenario_results)), "",
                                 COLORS["accent"]), md=3),
            dbc.Col(_metric_card(t('health.worst_scenario'),
                                 f"{min(r.max_drawdown_pct for r in report.scenario_results):.1f}%",
                                 "", COLORS["red"]), md=3),
            dbc.Col(_metric_card(t('health.avg_rm_savings'),
                                 f"{np.mean([r.risk_mgmt_saved_pct for r in report.scenario_results]):.1f}%",
                                 "", COLORS["green"]), md=3),
            dbc.Col(_metric_card(t('health.vulnerabilities_count'),
                                 str(len(report.vulnerabilities)),
                                 f"{sum(1 for v in report.vulnerabilities if v.severity == 'HØJ')} {t('health.high_risk')}",
                                 COLORS["orange"]), md=3),
        ], className="mb-4"),

        # Scenario-tabel
        dbc.Row([dbc.Col([
            html.H5(t('analysis.crisis_scenarios'), style={"color": COLORS["text"]}),
            scenario_table,
        ])], className="mb-4"),

        # Krisegraf
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_crisis)
            ), style={"backgroundColor": COLORS["card"],
                      "border": f"1px solid {COLORS['border']}"}), md=6),
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_comparison)
            ), style={"backgroundColor": COLORS["card"],
                      "border": f"1px solid {COLORS['border']}"}), md=6),
        ], className="mb-4"),

        # Monte Carlo
        html.H5(t('analysis.monte_carlo'), style={"color": COLORS["text"]},
                 className="mb-3"),
        *mc_section,

        # Regime-aktioner
        dbc.Row([dbc.Col(dbc.Card(dbc.CardBody([
            html.H5(t('analysis.regime_during_crises'),
                     style={"color": COLORS["text"]}),
            html.P(t('analysis.what_would_risk_do'),
                   style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Scenarie", style={"color": COLORS["accent"]}),
                    html.Th("RM besparelse", style={"color": COLORS["accent"]}),
                    html.Th("Aktioner", style={"color": COLORS["accent"]}),
                ])),
                html.Tbody(regime_rows),
            ], bordered=True, hover=True, responsive=True,
               style={"backgroundColor": COLORS["card"]},
               className="text-light"),
        ]), style={"backgroundColor": COLORS["card"],
                   "border": f"1px solid {COLORS['border']}"}))],
                className="mb-4"),

        # Sårbarheder
        html.H5(t('analysis.vulnerabilities'), style={"color": COLORS["text"]},
                 className="mb-3"),
        dbc.Row(vuln_cards if vuln_cards else [
            dbc.Col(dbc.Alert(t('health.no_critical_vulns'),
                              color="success")),
        ], className="mb-4"),

        # Footer
        html.Div([
            html.Hr(className="border-secondary mt-4"),
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                t('health.stress_note'),
            ], className="text-muted small"),
        ]),
    ], fluid=True, className="p-4")


# ══════════════════════════════════════════════════════════════
#  SIDE 11 – System Health
# ══════════════════════════════════════════════════════════════


def _get_health_data() -> dict:
    """Generér system health data."""
    from src.monitoring.health_monitor import HealthMonitor, HealthStatus
    from src.monitoring.performance_tracker import PerformanceTracker
    from src.monitoring.anomaly_detector import AnomalyDetector
    from src.monitoring.audit_log import AuditLog, AuditCategory

    monitor = HealthMonitor()
    health = monitor.check_all()

    # Simulér performance data fra backtests
    perf = PerformanceTracker(initial_equity=100_000)
    backtests = _run_backtests()
    for name, bt_result in backtests.items():
        for tr in bt_result.trades:
            perf.record_trade(
                symbol=tr.symbol, pnl=tr.net_pnl, strategy=name,
                return_pct=tr.return_pct, side=tr.side,
            )
    # Tilføj snapshots
    for name, bt_result in backtests.items():
        if not bt_result.equity_curve.empty:
            eq_vals = bt_result.equity_curve.values
            step = max(1, len(eq_vals) // 30)
            for v in eq_vals[::step]:
                perf.record_snapshot(equity=float(v))
        break  # Kun én kurve

    strat_perf = perf.strategy_performance()
    daily = perf.daily_report()
    decay_alerts = perf.detect_decay()

    # Anomaly detector
    detector = AnomalyDetector()
    for name, bt_result in backtests.items():
        for tr in bt_result.trades:
            detector.check_trade(tr.symbol, tr.net_pnl)

    anomalies = detector.get_active_anomalies()

    return {
        "health": health,
        "strategies": strat_perf,
        "daily": daily,
        "decay_alerts": decay_alerts,
        "anomalies": anomalies,
        "perf": perf,
    }


def page_health():
    """System Health dashboard-side."""
    try:
        data = _get_health_data()
        health = data["health"]
        strategies = data["strategies"]
        daily = data["daily"]
        decay_alerts = data["decay_alerts"]
        anomalies = data["anomalies"]
    except Exception as exc:
        logger.error(f"Health page fejl: {exc}")
        return dbc.Container([
            html.H3("System Health", style={"color": COLORS["text"]}),
            dbc.Alert(f"Fejl: {exc}", color="danger"),
        ], fluid=True, className="p-4")

    from src.monitoring.health_monitor import HealthStatus

    # ── Overall status ─────────────────────────────────────────
    overall = health.overall_status
    status_colors = {
        HealthStatus.HEALTHY: COLORS["green"],
        HealthStatus.DEGRADED: COLORS["orange"],
        HealthStatus.UNHEALTHY: COLORS["red"],
        HealthStatus.UNKNOWN: COLORS["muted"],
    }
    overall_color = status_colors.get(overall, COLORS["muted"])

    # ── Komponent-status kort ──────────────────────────────────
    component_cards = []
    for c in health.components:
        c_color = status_colors.get(c.status, COLORS["muted"])
        component_cards.append(dbc.Col(dbc.Card(dbc.CardBody([
            html.Div([
                html.Span(c.status_icon, style={"fontSize": "1.5rem"}),
                html.Strong(f" {c.name}",
                            style={"color": COLORS["text"], "fontSize": "0.9rem"}),
            ]),
            html.P(c.message,
                   style={"color": COLORS["muted"], "fontSize": "0.8rem",
                          "margin": "4px 0 0"}),
            html.Span(f"{c.response_time_ms:.0f}ms",
                       style={"color": c_color, "fontSize": "0.75rem"}),
        ]), style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {c_color}",
            "borderRadius": "10px",
        }), md=2, className="mb-3"))

    # ── Strategi-rangering tabel ────────────────────────────────
    strat_rows = []
    for s in strategies:
        status_color = {
            "STÆRK": COLORS["green"], "OK": COLORS["accent"],
            "SVAG": COLORS["orange"], "DECAY": COLORS["red"],
        }.get(s.status, COLORS["muted"])

        strat_rows.append(html.Tr([
            html.Td(f"#{s.rank}", style={"color": COLORS["accent"]}),
            html.Td(s.name, style={"color": COLORS["text"]}),
            html.Td(f"{s.sharpe_30d:+.2f}", style={
                "color": COLORS["green"] if s.sharpe_30d > 0.5 else COLORS["red"]}),
            html.Td(f"{s.win_rate:.1f}%", style={"color": COLORS["text"]}),
            html.Td(format_value(s.total_pnl), style={
                "color": COLORS["green"] if s.total_pnl > 0 else COLORS["red"]}),
            html.Td(f"{s.profit_factor:.2f}", style={"color": COLORS["text"]}),
            html.Td(f"{s.total_trades}", style={"color": COLORS["muted"]}),
            html.Td(dbc.Badge(s.status, style={"backgroundColor": status_color})),
        ]))

    strat_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("#", style={"color": COLORS["accent"]}),
            html.Th("Strategi", style={"color": COLORS["accent"]}),
            html.Th("Sharpe 30d", style={"color": COLORS["accent"]}),
            html.Th("Win Rate", style={"color": COLORS["accent"]}),
            html.Th("Total P&L", style={"color": COLORS["accent"]}),
            html.Th("PF", style={"color": COLORS["accent"]}),
            html.Th("Trades", style={"color": COLORS["accent"]}),
            html.Th("Status", style={"color": COLORS["accent"]}),
        ])),
        html.Tbody(strat_rows),
    ], bordered=True, hover=True, responsive=True,
       style={"backgroundColor": COLORS["card"]}, className="text-light")

    # ── Decay-advarsler ────────────────────────────────────────
    decay_section = []
    if decay_alerts:
        decay_items = []
        for a in decay_alerts:
            sev_color = COLORS["red"] if a.severity == "CRITICAL" else COLORS["orange"]
            decay_items.append(dbc.ListGroupItem([
                html.Div([
                    dbc.Badge(a.severity, style={"backgroundColor": sev_color},
                              className="me-2"),
                    html.Strong(a.strategy, style={"color": COLORS["text"]}),
                ]),
                html.P(a.reason,
                       style={"color": COLORS["muted"], "fontSize": "0.85rem",
                              "margin": "4px 0"}),
                html.Div([
                    html.Span(f"→ {a.recommendation}  ",
                              style={"color": COLORS["accent"], "fontSize": "0.85rem"}),
                    dcc.Link(
                        t('health.go_to_strategies') if t('health.go_to_strategies') != 'health.go_to_strategies' else "Gå til Strategier →",
                        href="/strategies",
                        style={"color": COLORS["blue"], "fontSize": "0.85rem",
                               "textDecoration": "underline"},
                    ),
                ], style={"margin": 0}),
            ], style={"backgroundColor": COLORS["card"],
                      "border": f"1px solid {COLORS['border']}"}))

        decay_section = [
            html.H5([
                html.I(className="bi bi-exclamation-triangle me-2"),
                "Strategy Decay Advarsler",
            ], style={"color": COLORS["orange"]}, className="mb-3"),
            dbc.ListGroup(decay_items, className="mb-4"),
        ]

    # ── Anomalier ──────────────────────────────────────────────
    anomaly_section = []
    if anomalies:
        anom_items = []
        for a in anomalies[:10]:
            anom_items.append(html.Tr([
                html.Td(a.severity_icon),
                html.Td(a.anomaly_type.value, style={"color": COLORS["text"],
                                                       "fontSize": "0.85rem"}),
                html.Td(a.title, style={"color": COLORS["text"],
                                         "fontSize": "0.85rem"}),
                html.Td(a.timestamp[:16], style={"color": COLORS["muted"],
                                                   "fontSize": "0.8rem"}),
            ]))
        anomaly_section = [
            html.H5([
                html.I(className="bi bi-bug me-2"),
                f"Anomalier ({len(anomalies)})",
            ], style={"color": COLORS["text"]}, className="mb-3"),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("", style={"color": COLORS["accent"]}),
                    html.Th("Type", style={"color": COLORS["accent"]}),
                    html.Th("Beskrivelse", style={"color": COLORS["accent"]}),
                    html.Th("Tidspunkt", style={"color": COLORS["accent"]}),
                ])),
                html.Tbody(anom_items),
            ], bordered=True, hover=True, responsive=True,
               style={"backgroundColor": COLORS["card"]},
               className="text-light mb-4"),
        ]

    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H3([
                    html.I(className="bi bi-heart-pulse me-2"),
                    "System Health",
                ], style={"color": COLORS["text"]}),
                html.P(t('analysis.monitor_components'),
                       style={"color": COLORS["muted"]}),
            ], md=8),
            dbc.Col([
                html.Div([
                    html.Span(t('analysis.system_status'), style={
                        "color": COLORS["muted"], "fontSize": "0.75rem",
                        "display": "block",
                    }),
                    html.H2(overall.value.upper(), style={
                        "color": overall_color, "fontWeight": "800",
                        "margin": "4px 0",
                    }),
                    html.Span(
                        f"Uptime: {health.uptime_seconds / 3600:.1f}t | "
                        f"Checks: {health.total_checks}",
                        style={"color": COLORS["muted"], "fontSize": "0.8rem"},
                    ),
                ], style={"textAlign": "right"}),
            ], md=4),
        ], className="mb-4"),

        # KPI
        dbc.Row([
            dbc.Col(_metric_card("Komponenter OK",
                                 f"{health.healthy_count}/{len(health.components)}",
                                 "", overall_color), md=2),
            dbc.Col(_metric_card("Strategier",
                                 str(len(strategies)), "",
                                 COLORS["accent"]), md=2),
            dbc.Col(_metric_card("P&L i dag",
                                 format_value(daily.pnl_today),
                                 f"{daily.return_today_pct:+.2f}%",
                                 COLORS["green"] if daily.pnl_today >= 0
                                 else COLORS["red"]), md=2),
            dbc.Col(_metric_card("Sharpe (30d)",
                                 f"{daily.sharpe_30d:.2f}", "",
                                 COLORS["green"] if daily.sharpe_30d > 0.5
                                 else COLORS["orange"]), md=2),
            dbc.Col(_metric_card("Win Rate",
                                 f"{daily.win_rate:.1f}%", "",
                                 COLORS["accent"]), md=2),
            dbc.Col(_metric_card("Anomalier",
                                 str(len(anomalies)),
                                 "aktive",
                                 COLORS["red"] if anomalies
                                 else COLORS["green"]), md=2),
        ], className="mb-4"),

        # Komponent-status
        html.H5(t('analysis.component_status'), style={"color": COLORS["text"]},
                 className="mb-3"),
        dbc.Row(component_cards, className="mb-4"),

        # Strategi-rangering
        html.H5([
            html.I(className="bi bi-trophy me-2"),
            t('health.strategy_performance'),
        ], style={"color": COLORS["text"]}, className="mb-3"),
        strat_table,

        # Decay-advarsler
        *decay_section,

        # Anomalier
        *anomaly_section,

        # P&L-oversigt
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5(t('analysis.performance_overview'), style={"color": COLORS["text"]}),
                dbc.Table([
                    html.Tbody([
                        html.Tr([
                            html.Td("P&L i dag", style={"color": COLORS["muted"]}),
                            html.Td(format_value(daily.pnl_today, 2),
                                    style={"color": COLORS["green"] if daily.pnl_today >= 0
                                           else COLORS["red"], "textAlign": "right"}),
                        ]),
                        html.Tr([
                            html.Td("P&L denne uge", style={"color": COLORS["muted"]}),
                            html.Td(format_value(daily.pnl_week, 2),
                                    style={"color": COLORS["green"] if daily.pnl_week >= 0
                                           else COLORS["red"], "textAlign": "right"}),
                        ]),
                        html.Tr([
                            html.Td(t('health.pnl_month'), style={"color": COLORS["muted"]}),
                            html.Td(format_value(daily.pnl_month, 2),
                                    style={"color": COLORS["green"] if daily.pnl_month >= 0
                                           else COLORS["red"], "textAlign": "right"}),
                        ]),
                        html.Tr([
                            html.Td("P&L YTD", style={"color": COLORS["muted"]}),
                            html.Td(format_value(daily.pnl_ytd, 2),
                                    style={"color": COLORS["green"] if daily.pnl_ytd >= 0
                                           else COLORS["red"], "textAlign": "right",
                                           "fontWeight": "bold"}),
                        ]),
                        html.Tr([
                            html.Td(t('health.sharpe_1y'), style={"color": COLORS["muted"]}),
                            html.Td(f"{daily.sharpe_1y:.2f}",
                                    style={"color": COLORS["text"], "textAlign": "right"}),
                        ]),
                        html.Tr([
                            html.Td("Profit Factor", style={"color": COLORS["muted"]}),
                            html.Td(f"{daily.profit_factor:.2f}",
                                    style={"color": COLORS["text"], "textAlign": "right"}),
                        ]),
                        html.Tr([
                            html.Td("Benchmark (S&P 500)", style={"color": COLORS["muted"]}),
                            html.Td(f"{daily.benchmark_return_pct:+.2f}%",
                                    style={"color": COLORS["text"], "textAlign": "right"}),
                        ]),
                        html.Tr([
                            html.Td("Alpha", style={"color": COLORS["muted"]}),
                            html.Td(f"{daily.alpha_pct:+.2f}%",
                                    style={"color": COLORS["green"] if daily.alpha_pct > 0
                                           else COLORS["red"], "textAlign": "right",
                                           "fontWeight": "bold"}),
                        ]),
                    ]),
                ], bordered=True, style={"backgroundColor": COLORS["card"]},
                   className="text-light"),
            ]), style={"backgroundColor": COLORS["card"],
                       "border": f"1px solid {COLORS['border']}"}), md=6),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5(t('analysis.system_info'), style={"color": COLORS["text"]}),
                dbc.Table([
                    html.Tbody([
                        html.Tr([
                            html.Td("Uptime", style={"color": COLORS["muted"]}),
                            html.Td(f"{health.uptime_seconds / 3600:.1f} timer",
                                    style={"color": COLORS["text"], "textAlign": "right"}),
                        ]),
                        html.Tr([
                            html.Td("Health checks", style={"color": COLORS["muted"]}),
                            html.Td(f"{health.total_checks}",
                                    style={"color": COLORS["text"], "textAlign": "right"}),
                        ]),
                        html.Tr([
                            html.Td("Fejlede checks", style={"color": COLORS["muted"]}),
                            html.Td(f"{health.failed_checks}",
                                    style={"color": COLORS["red"] if health.failed_checks > 0
                                           else COLORS["green"], "textAlign": "right"}),
                        ]),
                        html.Tr([
                            html.Td("Handler totalt", style={"color": COLORS["muted"]}),
                            html.Td(f"{daily.total_trades}",
                                    style={"color": COLORS["text"], "textAlign": "right"}),
                        ]),
                        html.Tr([
                            html.Td("Handler i dag", style={"color": COLORS["muted"]}),
                            html.Td(f"{daily.trades_today}",
                                    style={"color": COLORS["text"], "textAlign": "right"}),
                        ]),
                    ]),
                ], bordered=True, style={"backgroundColor": COLORS["card"]},
                   className="text-light"),
                html.Div([
                    html.Hr(className="border-secondary"),
                    html.P(
                        f"Sidst opdateret: {health.timestamp[:19]}",
                        style={"color": COLORS["muted"], "fontSize": "0.8rem"},
                    ),
                ]),
            ]), style={"backgroundColor": COLORS["card"],
                       "border": f"1px solid {COLORS['border']}"}), md=6),
        ], className="mb-4"),

        # Footer
        html.Div([
            html.Hr(className="border-secondary mt-4"),
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                t('health.health_note'),
            ], className="text-muted small"),
        ]),
    ], fluid=True, className="p-4")


# ══════════════════════════════════════════════════════════════
#  Side 12: Smart Money – Insider Tracking
# ══════════════════════════════════════════════════════════════


def page_smart_money():
    """Smart Money-side: insider trades, institutional holdings, short interest."""
    return dbc.Container([
        # Header
        html.H2([
            html.I(className="bi bi-bank me-2"),
            t('smart_money.title'),
        ], className="mb-1"),
        html.P(
            t('smart_money.subtitle'),
            className="text-muted mb-4",
        ),

        # Symbol selector
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Label(t('common.select_stock'), className="fw-bold text-light"),
                        dcc.Dropdown(
                            id="smart-money-symbol",
                            options=[{"label": s, "value": s} for s in SYMBOLS],
                            value=SYMBOLS[0] if SYMBOLS else "AAPL",
                            className="mb-3",
                            style={"backgroundColor": "#1a1c24", "color": "#fff"},
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-search me-2"), t('smart_money.analyze')],
                            id="smart-money-btn",
                            color="primary",
                            className="w-100",
                        ),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(t('smart_money.what_is'), className="text-light"),
                        html.P([
                            t('smart_money.what_is_desc'),
                        ], className="text-muted small mb-2"),
                        html.Ul([
                            html.Li([
                                html.Strong(t('smart_money.insider_buy'), className="text-success"),
                                f" {t('smart_money.insider_buy_desc')}",
                            ], className="text-muted small"),
                            html.Li([
                                html.Strong(t('smart_money.cluster_buying'), className="text-success"),
                                f" {t('smart_money.cluster_buying_desc')}",
                            ], className="text-muted small"),
                            html.Li([
                                html.Strong(t('smart_money.high_short'), className="text-danger"),
                                f" {t('smart_money.high_short_desc')}",
                            ], className="text-muted small"),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=8),
        ], className="mb-4"),

        # Insider Sentiment Overview
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-person-badge me-2"),
                            t('smart_money.insider_sentiment'),
                        ], className="text-light mb-3"),
                        html.Div(id="insider-sentiment-content", children=[
                            html.P(
                                t('smart_money.click_to_fetch_insider'),
                                className="text-muted",
                            ),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-bar-chart-line me-2"),
                            t('smart_money.short_interest'),
                        ], className="text-light mb-3"),
                        html.Div(id="short-interest-content", children=[
                            html.P(
                                t('smart_money.click_to_fetch_short'),
                                className="text-muted",
                            ),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
        ], className="mb-4"),

        # Insider Trades Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-table me-2"),
                            t('smart_money.latest_insider_trades'),
                        ], className="text-light mb-3"),
                        html.Div(id="insider-trades-table", children=[
                            html.P(
                                t('smart_money.insider_trades_here'),
                                className="text-muted",
                            ),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=12),
        ], className="mb-4"),

        # Institutional Holdings
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-building me-2"),
                            t('smart_money.institutional'),
                        ], className="text-light mb-3"),
                        html.P(
                            "Trackede fonde: Berkshire Hathaway, Bridgewater, Renaissance Technologies, "
                            "Citadel, Two Sigma, DE Shaw, BlackRock, Vanguard, ARK Invest, Soros Fund",
                            className="text-muted small mb-3",
                        ),
                        html.Div(id="institutional-holdings-content", children=[
                            html.P(
                                "Institutionelle holdings vises her efter analyse",
                                className="text-muted",
                            ),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=12),
        ], className="mb-4"),

        # Overall Assessment
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-clipboard-data me-2"),
                            t('smart_money.overall_assessment'),
                        ], className="text-light mb-3"),
                        html.Div(id="smart-money-assessment", children=[
                            html.P(
                                t('smart_money.assessment_placeholder'),
                                className="text-muted",
                            ),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=12),
        ]),

        # Footer
        html.Div([
            html.Hr(className="border-secondary mt-4"),
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                t('smart_money.footer_note'),
            ], className="text-muted small"),
        ]),
    ], fluid=True, className="p-4")


# ══════════════════════════════════════════════════════════════
#  Side 13: Options Flow
# ══════════════════════════════════════════════════════════════


def page_options_flow():
    """Options Flow-side: UOA, P/C ratio, max pain, IV analyse."""
    return dbc.Container([
        # Header
        html.H2([
            html.I(className="bi bi-bar-chart-steps me-2"),
            t('options.title'),
        ], className="mb-1"),
        html.P(
            t('options.subtitle'),
            className="text-muted mb-4",
        ),

        # Symbol selector
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Label(t('common.select_stock'), className="fw-bold text-light"),
                        dcc.Dropdown(
                            id="options-flow-symbol",
                            options=[{"label": s, "value": s} for s in SYMBOLS],
                            value=SYMBOLS[0] if SYMBOLS else "AAPL",
                            className="mb-3",
                            style={"backgroundColor": "#1a1c24", "color": "#fff"},
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-search me-2"), t('options.analyze')],
                            id="options-flow-btn",
                            color="primary",
                            className="w-100",
                        ),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(t('options.what_is'), className="text-light"),
                        html.P([
                            t('options.what_is_desc'),
                        ], className="text-muted small mb-2"),
                        html.Ul([
                            html.Li([
                                html.Strong(t('options.unusual_activity'), className="text-warning"),
                                f" — {t('options.unusual_activity_desc')}",
                            ], className="text-muted small"),
                            html.Li([
                                html.Strong(t('options.put_call_ratio'), className="text-info"),
                                f" — {t('options.put_call_desc')}",
                            ], className="text-muted small"),
                            html.Li([
                                html.Strong(t('options.max_pain'), className="text-primary"),
                                f" — {t('options.max_pain_desc')}",
                            ], className="text-muted small"),
                            html.Li([
                                html.Strong(t('options.iv_rank'), className="text-danger"),
                                f" — {t('options.iv_rank_desc')}",
                            ], className="text-muted small"),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=8),
        ], className="mb-4"),

        # Top row: UOA + IV
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-lightning-charge me-2 text-warning"),
                            t('options.unusual_activity'),
                        ], className="text-light mb-3"),
                        html.Div(id="options-uoa-content", children=[
                            html.P(
                                t('options.click_to_scan'),
                                className="text-muted",
                            ),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=7),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-graph-up-arrow me-2 text-danger"),
                            t('options.implied_volatility'),
                        ], className="text-light mb-3"),
                        html.Div(id="options-iv-content", children=[
                            html.P(
                                t('options.iv_shown_here'),
                                className="text-muted",
                            ),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=5),
        ], className="mb-4"),

        # Middle row: P/C Ratio + Max Pain
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-pie-chart me-2 text-info"),
                            t('options.put_call_ratio'),
                        ], className="text-light mb-3"),
                        html.Div(id="options-pcr-content", children=[
                            html.P(
                                "Put/Call ratio og contrarian-tolkning vises her",
                                className="text-muted",
                            ),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-crosshair me-2 text-primary"),
                            t('options.max_pain'),
                        ], className="text-light mb-3"),
                        html.Div(id="options-maxpain-content", children=[
                            html.P(
                                t('options.maxpain_placeholder'),
                                className="text-muted",
                            ),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
        ], className="mb-4"),

        # Overall Assessment
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-clipboard-data me-2"),
                            t('options.overall_signal'),
                        ], className="text-light mb-3"),
                        html.Div(id="options-assessment", children=[
                            html.P(
                                t('options.signal_placeholder'),
                                className="text-muted",
                            ),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=12),
        ]),

        # Footer
        html.Div([
            html.Hr(className="border-secondary mt-4"),
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                t('options.footer_note'),
            ], className="text-muted small"),
        ]),
    ], fluid=True, className="p-4")


# ══════════════════════════════════════════════════════════════
#  Side 14: Alternativ Data
# ══════════════════════════════════════════════════════════════


def page_alt_data():
    """Alternativ Data-side: Google Trends, GitHub, patenter, jobopslag."""
    return dbc.Container([
        # Header
        html.H2([
            html.I(className="bi bi-stars me-2"),
            t('alt_data.title'),
        ], className="mb-1"),
        html.P(
            t('alt_data.subtitle'),
            className="text-muted mb-4",
        ),

        # Symbol selector
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Label(t('common.select_stock'), className="fw-bold text-light"),
                        dcc.Dropdown(
                            id="alt-data-symbol",
                            options=[{"label": s, "value": s} for s in SYMBOLS],
                            value=SYMBOLS[0] if SYMBOLS else "AAPL",
                            className="mb-3",
                            style={"backgroundColor": "#1a1c24", "color": "#fff"},
                        ),
                        dbc.Button(
                            [html.I(className="bi bi-search me-2"), t('alt_data.analyze')],
                            id="alt-data-btn",
                            color="primary",
                            className="w-100",
                        ),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(t('alt_data.what_is'), className="text-light"),
                        html.P([
                            t('alt_data.what_is_desc'),
                        ], className="text-muted small mb-2"),
                        html.Ul([
                            html.Li([
                                html.Strong(t('alt_data.google_trends'), className="text-info"),
                                f" — {t('alt_data.google_trends_desc')}",
                            ], className="text-muted small"),
                            html.Li([
                                html.Strong(t('alt_data.job_postings'), className="text-success"),
                                f" — {t('alt_data.job_desc')}",
                            ], className="text-muted small"),
                            html.Li([
                                html.Strong(t('alt_data.patents'), className="text-warning"),
                                f" — {t('alt_data.patent_desc')}",
                            ], className="text-muted small"),
                            html.Li([
                                html.Strong(t('alt_data.github'), className="text-light"),
                                f" — {t('alt_data.github_desc')}",
                            ], className="text-muted small"),
                        ]),
                        html.P([
                            html.I(className="bi bi-info-circle me-1"),
                            t('alt_data.weight_note'),
                        ], className="text-muted small mb-0 mt-2"),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=8),
        ], className="mb-4"),

        # Data panels
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-google me-2 text-info"),
                            t('alt_data.google_trends'),
                        ], className="text-light mb-3"),
                        html.Div(id="alt-trends-content", children=[
                            html.P(t('alt_data.trends_here'), className="text-muted"),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-briefcase me-2 text-success"),
                            t('alt_data.job_postings'),
                        ], className="text-light mb-3"),
                        html.Div(id="alt-jobs-content", children=[
                            html.P(t('alt_data.jobs_here'), className="text-muted"),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
        ], className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-file-earmark-text me-2 text-warning"),
                            t('alt_data.patents'),
                        ], className="text-light mb-3"),
                        html.Div(id="alt-patents-content", children=[
                            html.P(t('alt_data.patents_here'), className="text-muted"),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-github me-2"),
                            t('alt_data.github'),
                        ], className="text-light mb-3"),
                        html.Div(id="alt-github-content", children=[
                            html.P(t('alt_data.github_here'), className="text-muted"),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
        ], className="mb-4"),

        # Overall score
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-clipboard-data me-2"),
                            t('alt_data.overall_score'),
                        ], className="text-light mb-3"),
                        html.Div(id="alt-score-content", children=[
                            html.P(
                                t('alt_data.score_shown_after'),
                                className="text-muted",
                            ),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=12),
        ]),

        # Footer
        html.Div([
            html.Hr(className="border-secondary mt-4"),
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                "Data fra Google Trends (pytrends), GitHub API og USPTO. ",
                "Alternative data is indirect and should be weighted low (max 10% of signal). ",
                "Web traffic and app rankings are estimates based on search interest.",
            ], className="text-muted small"),
        ]),
    ], fluid=True, className="p-4")


# ══════════════════════════════════════════════════════════════
#  Side 15: Økonomi – Makro-indikatorer
# ══════════════════════════════════════════════════════════════

def _economy_indicator_desc(key: str) -> str:
    """Get translated indicator description, fall back to FRED_SERIES."""
    from src.data.macro_indicators import FRED_SERIES
    translated = t(f'economy.ind_{key}')
    if translated != f'economy.ind_{key}':
        return translated
    return FRED_SERIES.get(key, {}).get("description", "")


# ── Background macro data fetch (runs once at startup) ────
_macro_report_cache: dict = {}  # {"report": MacroReport, "ts": float}


def _ensure_macro_data():
    """Fetch macro report if not cached. Called from background thread at startup."""
    import os, time as _t
    if "report" in _macro_report_cache:
        return _macro_report_cache["report"]
    try:
        from src.data.macro_indicators import MacroIndicatorTracker
        fred_key = settings.market_data.fred_api_key or os.environ.get("FRED_API_KEY", "")
        tracker = MacroIndicatorTracker(fred_api_key=fred_key)
        report = tracker.get_macro_report()
        _macro_report_cache["report"] = report
        _macro_report_cache["ts"] = _t.time()
        logger.info(f"[macro] Background fetch done: {len(report.indicators)} indicators")
        return report
    except Exception as exc:
        logger.warning(f"[macro] Background fetch failed: {exc}")
        return None


# Start background fetches at import time (non-blocking)
import threading as _threading
_threading.Thread(target=_ensure_macro_data, daemon=True, name="macro-prefetch").start()
_threading.Thread(target=_fetch_scanner_data_sync, daemon=True, name="scanner-prefetch").start()


def page_economy():
    """Makro-økonomi side: FRED-data, recession-sandsynlighed, heatmap."""
    import time as _t
    from src.data.macro_indicators import FRED_SERIES, CATEGORIES

    # ── Try to render from cache (instant) ──
    report = _macro_report_cache.get("report")
    cache_age = _t.time() - _macro_report_cache.get("ts", 0) if "ts" in _macro_report_cache else None

    if report:
        recession, surprise, signal, heatmap, cat_cols = _build_macro_ui(report, FRED_SERIES)
        n_found = len(report.indicators)
        age_str = f" ({int(cache_age)}s ago)" if cache_age else ""
        status = html.Span([
            html.I(className="bi bi-check-circle me-1"),
            f"{n_found}/{len(FRED_SERIES)}{age_str}",
        ], style={"color": COLORS["green"], "fontSize": "0.8rem"})
    else:
        # Still loading from background thread
        loading = html.Div([
            dbc.Spinner(size="sm", color="light"),
            html.Span(f" {t('common.loading')}", className="text-muted ms-2"),
        ])
        recession = loading
        surprise = html.Div(loading)
        signal = html.Div(loading)
        heatmap = html.Div(loading)
        cat_cols = []
        status = html.Span(t('common.loading'), style={"color": COLORS["muted"], "fontSize": "0.8rem"})

        # Build placeholder category cards
        category_labels = {
            "shipping": ("bi-truck", t('economy.shipping'), "text-info"),
            "housing": ("bi-house-door", t('economy.housing'), "text-warning"),
            "energy": ("bi-lightning-charge", t('economy.energy'), "text-danger"),
            "consumer": ("bi-people", t('economy.consumer'), "text-success"),
            "labor": ("bi-briefcase", t('economy.labor'), "text-primary"),
            "recession": ("bi-exclamation-triangle", t('economy.recession'), "text-danger"),
        }
        for cat_key, (icon, label, color) in category_labels.items():
            series_in_cat = [(k, v) for k, v in FRED_SERIES.items() if v["category"] == cat_key]
            indicators_list = html.Ul([
                html.Li([
                    html.Strong(v["name"], className="text-light"),
                    html.Span(f" — {_economy_indicator_desc(k)}", className="text-muted small"),
                ], className="mb-1") for k, v in series_in_cat
            ], className="list-unstyled mb-0")
            cat_cols.append(dbc.Col([
                dbc.Card([dbc.CardBody([
                    html.H5([html.I(className=f"bi {icon} me-2 {color}"), label], className="text-light mb-3"),
                    indicators_list,
                ])], style={"backgroundColor": COLORS["card"]}),
            ], md=6, className="mb-4"))

    return dbc.Container([
        # Header + Update button
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="bi bi-globe-americas me-2"),
                    t('economy.title'),
                ], className="mb-1"),
                html.P(t('economy.subtitle'), className="text-muted mb-0"),
            ], md=8),
            dbc.Col([
                dbc.Button(
                    [html.I(className="bi bi-arrow-clockwise me-2"), t('economy.update_btn')],
                    id="macro-update-btn",
                    color="primary",
                    className="mt-2",
                ),
                html.Div(id="macro-update-status", children=[status], className="mt-1"),
            ], md=4, className="text-end"),
        ], className="mb-4"),

        # Top row: Recession probability + Economic Surprise + Signal
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-exclamation-triangle me-2 text-danger"),
                            t('economy.recession_probability'),
                        ], className="text-light mb-3"),
                        html.Div(id="macro-recession-content", children=[recession]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-graph-up-arrow me-2 text-success"),
                            t('economy.economic_surprise'),
                        ], className="text-light mb-3"),
                        html.Div(id="macro-surprise-content", children=[surprise]),
                        html.Hr(className="border-secondary"),
                        html.H5([
                            html.I(className="bi bi-thermometer-half me-2 text-info"),
                            t('economy.overall_signal'),
                        ], className="text-light mb-3"),
                        html.Div(id="macro-signal-content", children=[signal]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
        ], className="mb-4"),

        # Heatmap — live data
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-grid-3x3-gap me-2 text-warning"),
                            t('economy.indicator_heatmap'),
                        ], className="text-light mb-3"),
                        html.P(t('economy.heatmap_desc'), className="text-muted small mb-3"),
                        html.Div(id="macro-heatmap-content", children=[heatmap]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=12),
        ], className="mb-4"),

        # Category detail cards
        html.H4(t('health.categories'), className="text-light mb-3"),
        dbc.Row(cat_cols, id="macro-category-cards"),

        # Footer
        html.Div([
            html.Hr(className="border-secondary mt-4"),
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                t('economy.economy_note'),
            ], className="text-muted small"),
        ]),
    ], fluid=True, className="p-4")


# ══════════════════════════════════════════════════════════════
#  Side 17: Teknisk Analyse – Mønstergenkendelse
# ══════════════════════════════════════════════════════════════

def page_technical_analysis():
    """Technical Analysis page: chart patterns, candlesticks, S/R, seasonal, divergence, MTF."""
    lang = get_language()
    # Build dropdown options
    _ta_options = [{"label": s, "value": s, "title": s} for s in SYMBOLS]

    # Get portfolio positions for portfolio dropdown
    # Read from all sources: live router + SQLite DB (catches all stocks, not just US)
    _portfolio_symbols = []
    # Source 1: Live broker router
    try:
        from src.broker.registry import get_router
        router = get_router()
        if router:
            for broker_name, broker in router.brokers.items():
                try:
                    positions = broker.get_positions() if hasattr(broker, "get_positions") else []
                    for pos in positions:
                        sym = getattr(pos, "symbol", "")
                        if sym and sym not in _portfolio_symbols:
                            _portfolio_symbols.append(sym)
                except Exception:
                    pass
    except Exception:
        pass
    # Source 2: SQLite portfolio DB (catches all stocks including non-US)
    try:
        import sqlite3 as _sql3
        from pathlib import Path as _Path
        _pdb = _Path("data_cache/paper_portfolio.db")
        if _pdb.exists():
            with _sql3.connect(str(_pdb)) as _conn:
                _rows = _conn.execute("SELECT symbol FROM open_positions").fetchall()
                for (_sym,) in _rows:
                    if _sym and _sym not in _portfolio_symbols:
                        _portfolio_symbols.append(_sym)
    except Exception:
        pass
    if not _portfolio_symbols:
        # Fallback: use watchlist symbols as portfolio placeholder
        _portfolio_symbols = SYMBOLS[:10] if SYMBOLS else []
    _portfolio_options = [{"label": s, "value": s, "title": s} for s in sorted(_portfolio_symbols)]

    return dbc.Container([
        # Header
        html.H2([
            html.I(className="bi bi-search me-2"),
            t("technical.title", lang),
        ], className="mb-1"),
        html.P(t("technical.subtitle", lang), className="text-muted mb-4"),

        # Row 1: [Vælg aktie] [Portfolio aktier] [Samlet Signal → far right]
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Label(t("common.select_stock", lang), className="fw-bold text-light"),
                        dcc.Dropdown(
                            id="ta-symbol",
                            options=_ta_options,
                            value=SYMBOLS[0] if SYMBOLS else "AAPL",
                            persistence=True,
                            persistence_type="local",
                        ),
                        html.Div(id="ta-symbol-info", className="mt-2"),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Label(t("technical.portfolio_stocks", lang), className="fw-bold text-light"),
                        dcc.Dropdown(
                            id="ta-portfolio-symbol",
                            options=_portfolio_options,
                            placeholder=t("technical.portfolio_placeholder", lang),
                            persistence=True,
                            persistence_type="local",
                        ),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(t("technical.overall_signal", lang), className="text-light"),
                        html.Div(id="ta-overall-signal", children=[
                            html.Span("—", className="display-4 fw-bold text-muted"),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
        ], className="mb-4"),

        # Row 2: [Chart Patterns] [Support/Resistance]
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-graph-up me-2 text-info"),
                            t("technical.chart_patterns", lang),
                        ], className="text-light mb-2"),
                        html.P(t("technical.chart_patterns_desc", lang),
                               className="text-muted small mb-2"),
                        html.Div(id="ta-chart-patterns", children=[
                            html.P(t("technical.no_patterns", lang), className="text-muted"),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-rulers me-2 text-success"),
                            t("technical.support_resistance", lang),
                        ], className="text-light mb-2"),
                        html.Div(id="ta-sr-levels", children=[
                            html.P(t("technical.levels_after_scan", lang), className="text-muted"),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
        ], className="mb-4"),

        # Row 3: [Candlestick] [Multi-Timeframe]
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-bar-chart me-2 text-warning"),
                            t("technical.candlestick_patterns", lang),
                        ], className="text-light mb-3"),
                        html.P(t("technical.candlestick_desc", lang),
                               className="text-muted small mb-2"),
                        html.Div(id="ta-candle-patterns", children=[
                            html.P(t("technical.no_candle_patterns", lang), className="text-muted"),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-layers me-2 text-info"),
                            t("technical.multi_timeframe", lang),
                        ], className="text-light mb-3"),
                        html.Div(id="ta-mtf-signal", children=[
                            html.P(t("technical.scan_for_mtf", lang), className="text-muted small"),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=6),
        ], className="mb-4"),

        # Row 4: [Breakouts & Divergenser]
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-arrow-up-right me-2 text-danger"),
                            t("technical.breakouts_divergences", lang),
                        ], className="text-light mb-3"),
                        html.Div(id="ta-breakouts-div", children=[
                            html.P(t("technical.breakouts_after_scan", lang), className="text-muted"),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=12),
        ], className="mb-4"),

        # Row 4: Seasonal (full width)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([
                            html.I(className="bi bi-calendar3 me-2 text-info"),
                            t("technical.seasonal", lang),
                        ], className="text-light mb-3"),
                        html.P(t("technical.seasonal_desc", lang),
                               className="text-muted small mb-2"),
                        html.Div(id="ta-seasonal", children=[
                            html.P(t("technical.requires_2yr", lang), className="text-muted"),
                        ]),
                    ]),
                ], style={"backgroundColor": COLORS["card"]}),
            ], md=12),
        ], className="mb-4"),

        # Footer
        html.Div([
            html.Hr(className="border-secondary mt-4"),
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                t("technical.footer_note", lang),
            ], className="text-muted small"),
        ]),
    ], fluid=True, className="p-4")


# ══════════════════════════════════════════════════════════════
#  Side 17: Krypto On-Chain
# ══════════════════════════════════════════════════════════════


def page_crypto_onchain():
    """Krypto on-chain analyse side."""
    try:
        from src.data.onchain import OnChainTracker, OnChainSignal, FearGreedLevel
        tracker = OnChainTracker()
        report = tracker.get_report("BTC-USD")
    except Exception:
        report = None

    # Signal badge farve
    def _signal_color(sig):
        if sig in ("bullish", "strong_bullish", "buy", "accumulating", "undervalued"):
            return "success"
        if sig in ("bearish", "strong_bearish", "sell", "distributing", "overvalued"):
            return "danger"
        return "warning"

    # Header
    header = dbc.Row([
        dbc.Col([
            html.H2([html.I(className="bi bi-currency-bitcoin me-2"), t('crypto.title')],
                     className="text-light mb-0"),
            html.P(t('crypto.subtitle'),
                   className="text-muted"),
        ], width=8),
        dbc.Col([
            html.Div([
                html.Span(f"{t('crypto.overall_signal')}: ", className="text-muted me-2"),
                dbc.Badge(
                    report.overall_signal.value.replace("_", " ").title() if report else "N/A",
                    color=_signal_color(report.overall_signal.value) if report else "secondary",
                    className="fs-6 px-3 py-2",
                ),
            ], className="text-end"),
            html.Div([
                html.Span(f"Confidence: {report.confidence:.0f}%" if report else "",
                          className="text-muted small"),
            ], className="text-end mt-1"),
        ], width=4),
    ], className="mb-4")

    # Fear & Greed gauge
    fg_card = dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-emoji-dizzy me-2"), t('crypto.fear_greed')]),
        dbc.CardBody([
            html.Div([
                html.H1(
                    str(report.fear_greed.value) if report and report.fear_greed else "–",
                    className="text-center display-3",
                    style={"color": (
                        "#e74c3c" if report and report.fear_greed and report.fear_greed.value > 70
                        else "#2ecc71" if report and report.fear_greed and report.fear_greed.value < 30
                        else "#f39c12"
                    )},
                ),
                html.P(
                    report.fear_greed.classification if report and report.fear_greed else "",
                    className="text-center text-muted",
                ),
                html.Hr(className="border-secondary"),
                html.P([
                    html.Strong(f"{t('crypto.contrarian')}: "),
                    dbc.Badge(
                        report.fear_greed.contrarian_signal.upper() if report and report.fear_greed else "–",
                        color=_signal_color(report.fear_greed.contrarian_signal) if report and report.fear_greed else "secondary",
                    ),
                ], className="mb-1"),
                html.P(
                    t('crypto.fear_greed_note'),
                    className="text-muted small",
                ),
            ]),
        ]),
    ], className="bg-dark border-secondary h-100")

    # BTC Dominance
    dom_card = dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-pie-chart me-2"), t('crypto.btc_dominance')]),
        dbc.CardBody([
            html.H2(
                f"{report.btc_dominance.dominance_pct:.1f}%" if report and report.btc_dominance else "–",
                className="text-center text-info",
            ),
            html.P(
                report.btc_dominance.description if report and report.btc_dominance else "",
                className="text-center text-muted",
            ),
            html.Hr(className="border-secondary"),
            html.P([
                html.Strong(f"{t('crypto.alt_season')}: "),
                dbc.Badge(
                    "JA" if report and report.btc_dominance and report.btc_dominance.alt_season else "NEJ",
                    color="success" if report and report.btc_dominance and report.btc_dominance.alt_season else "secondary",
                ),
            ]),
        ]),
    ], className="bg-dark border-secondary h-100")

    # NVT Ratio
    nvt_card = dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-graph-up-arrow me-2"), t('crypto.nvt_ratio')]),
        dbc.CardBody([
            html.H2(
                f"{report.nvt_ratio.nvt:.0f}" if report and report.nvt_ratio else "–",
                className="text-center",
                style={"color": (
                    "#e74c3c" if report and report.nvt_ratio and report.nvt_ratio.is_overvalued
                    else "#2ecc71" if report and report.nvt_ratio and report.nvt_ratio.is_undervalued
                    else "#f39c12"
                )},
            ),
            html.P(
                report.nvt_ratio.description if report and report.nvt_ratio else "Network Value to Transactions",
                className="text-center text-muted",
            ),
            html.Hr(className="border-secondary"),
            html.P("< 45 = undervalued | 45-95 = fair | > 95 = overvalued",
                   className="text-muted small text-center"),
        ]),
    ], className="bg-dark border-secondary h-100")

    # Exchange Flow
    flow_card = dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-arrow-left-right me-2"), t('crypto.exchange_flow')]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P(t('crypto.inflow'), className="text-muted mb-0 text-center"),
                    html.H4(
                        f"{report.exchange_flow.inflow_btc:,.0f} BTC"
                        if report and report.exchange_flow else "–",
                        className="text-danger text-center",
                    ),
                ]),
                dbc.Col([
                    html.P(t('crypto.outflow'), className="text-muted mb-0 text-center"),
                    html.H4(
                        f"{report.exchange_flow.outflow_btc:,.0f} BTC"
                        if report and report.exchange_flow else "–",
                        className="text-success text-center",
                    ),
                ]),
            ]),
            html.Hr(className="border-secondary"),
            html.P([
                html.Strong(f"{t('crypto.net_flow')}: "),
                html.Span(
                    f"{report.exchange_flow.net_flow:+,.0f} BTC"
                    if report and report.exchange_flow else "–",
                ),
            ]),
            html.P(
                report.exchange_flow.description if report and report.exchange_flow else "",
                className="text-muted small",
            ),
        ]),
    ], className="bg-dark border-secondary h-100")

    # Hash Rate
    hash_card = dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-cpu me-2"), t('crypto.hash_rate')]),
        dbc.CardBody([
            html.H3(
                f"{report.hash_rate.hash_rate:,.1f} TH/s" if report and report.hash_rate else "–",
                className="text-center text-info",
            ),
            html.P([
                html.Strong(f"{t('crypto.30d_change')}: "),
                html.Span(
                    f"{report.hash_rate.change_pct_30d:+.1f}%"
                    if report and report.hash_rate else "–",
                    className=(
                        "text-success" if report and report.hash_rate and report.hash_rate.change_pct_30d > 0
                        else "text-danger"
                    ),
                ),
            ], className="text-center"),
            html.P([
                dbc.Badge(
                    report.hash_rate.signal if report and report.hash_rate else "–",
                    color=_signal_color(report.hash_rate.signal) if report and report.hash_rate else "secondary",
                ),
            ], className="text-center"),
        ]),
    ], className="bg-dark border-secondary h-100")

    # Whale Activity
    whale_card = dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-tsunami me-2"), t('crypto.whale_activity')]),
        dbc.CardBody([
            html.H4(
                f"~{report.whale_activity.large_txs_24h:,} store txs"
                if report and report.whale_activity else "–",
                className="text-center text-warning",
            ),
            html.P([
                html.Strong(f"{t('crypto.sentiment')}: "),
                dbc.Badge(
                    report.whale_activity.whale_sentiment if report and report.whale_activity else "–",
                    color=_signal_color(
                        report.whale_activity.whale_sentiment) if report and report.whale_activity else "secondary",
                ),
            ], className="text-center"),
            html.Hr(className="border-secondary"),
            html.P(
                report.whale_activity.description if report and report.whale_activity else "",
                className="text-muted small",
            ),
        ]),
    ], className="bg-dark border-secondary h-100")

    # DeFi Metrics
    defi_card = dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-boxes me-2"), t('crypto.defi_metrics')]),
        dbc.CardBody([
            html.H3(
                f"${report.defi_metrics.total_tvl_usd / 1e9:,.1f}B TVL"
                if report and report.defi_metrics else "–",
                className="text-center text-info",
            ),
            dbc.Row([
                dbc.Col([
                    html.P(t('crypto.24h'), className="text-muted mb-0 text-center small"),
                    html.P(
                        f"{report.defi_metrics.tvl_change_24h_pct:+.1f}%"
                        if report and report.defi_metrics else "–",
                        className="text-center",
                    ),
                ]),
                dbc.Col([
                    html.P(t('crypto.7d'), className="text-muted mb-0 text-center small"),
                    html.P(
                        f"{report.defi_metrics.tvl_change_7d_pct:+.1f}%"
                        if report and report.defi_metrics else "–",
                        className="text-center",
                    ),
                ]),
            ]),
            html.Hr(className="border-secondary"),
            html.P([
                html.Strong(f"{t('crypto.stablecoin_mcap')}: "),
                html.Span(
                    f"${report.defi_metrics.stablecoin_mcap / 1e9:,.1f}B"
                    if report and report.defi_metrics and report.defi_metrics.stablecoin_mcap > 0
                    else "–",
                ),
            ], className="small"),
            # Top protocols
            *([html.P(html.Strong(f"{t('crypto.top_protocols')}:"), className="mb-1 small")] +
              [html.P(
                  f"  {p['name']}: ${p['tvl'] / 1e9:,.1f}B",
                  className="text-muted mb-0 small",
              ) for p in (report.defi_metrics.top_protocols[:3]
                         if report and report.defi_metrics else [])]),
        ]),
    ], className="bg-dark border-secondary h-100")

    # Active Addresses
    addr_card = dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-people me-2"), t('crypto.active_addresses')]),
        dbc.CardBody([
            html.H3(
                f"{report.active_addresses.count:,}"
                if report and report.active_addresses else "–",
                className="text-center text-info",
            ),
            html.P([
                html.Strong(f"{t('crypto.7d_change')}: "),
                html.Span(
                    f"{report.active_addresses.change_pct_7d:+.1f}%"
                    if report and report.active_addresses else "–",
                    className=(
                        "text-success" if report and report.active_addresses
                        and report.active_addresses.change_pct_7d > 0 else "text-danger"
                    ),
                ),
            ], className="text-center"),
            html.P([
                dbc.Badge(
                    report.active_addresses.signal if report and report.active_addresses else "–",
                    color=_signal_color(
                        report.active_addresses.signal) if report and report.active_addresses else "secondary",
                ),
            ], className="text-center"),
        ]),
    ], className="bg-dark border-secondary h-100")

    # Sammenfatning
    summary_card = dbc.Card([
        dbc.CardHeader([html.I(className="bi bi-clipboard-data me-2"), t('crypto.summary')]),
        dbc.CardBody([
            html.P(report.summary if report else t('common.no_data'),
                   className="text-light"),
            html.Hr(className="border-secondary"),
            html.Pre(
                tracker.explain(report) if report else "",
                className="text-muted small",
                style={"maxHeight": "300px", "overflowY": "auto", "whiteSpace": "pre-wrap"},
            ),
        ]),
    ], className="bg-dark border-secondary")

    return dbc.Container([
        header,

        # Række 1: Fear & Greed, BTC Dominance, NVT
        dbc.Row([
            dbc.Col(fg_card, md=4, className="mb-3"),
            dbc.Col(dom_card, md=4, className="mb-3"),
            dbc.Col(nvt_card, md=4, className="mb-3"),
        ]),

        # Række 2: Exchange Flow, Hash Rate, Whale Activity
        dbc.Row([
            dbc.Col(flow_card, md=4, className="mb-3"),
            dbc.Col(hash_card, md=4, className="mb-3"),
            dbc.Col(whale_card, md=4, className="mb-3"),
        ]),

        # Række 3: DeFi, Active Addresses
        dbc.Row([
            dbc.Col(defi_card, md=6, className="mb-3"),
            dbc.Col(addr_card, md=6, className="mb-3"),
        ]),

        # Sammenfatning
        dbc.Row([
            dbc.Col(summary_card, md=12, className="mb-3"),
        ]),

        # Footer
        html.Div([
            html.Hr(className="border-secondary mt-4"),
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                "On-chain data via alternative.me, CoinGecko, Blockchain.com og DeFi Llama (alle gratis). ",
                "Exchange flow er estimeret (30% inflow / 25% outflow af total BTC sendt). ",
                "NVT bruger cirkulerende udbud ≈ 19.5M BTC. ",
                "Fear & Greed bruger contrarian-logik.",
            ], className="text-muted small"),
        ]),
    ], fluid=True, className="p-4")


# ══════════════════════════════════════════════════════════════
#  Settings Page
# ══════════════════════════════════════════════════════════════


_EXCHANGE_LABELS = {
    "crypto": "Crypto (24/7)",
    "new_zealand": "New Zealand (NZX)",
    "australia": "Australia (ASX)",
    "japan": "Japan (TSE)",
    "hong_kong": "Hong Kong (HKEX)",
    "india": "India (NSE)",
    # Individual European / Nordic exchanges
    "denmark": "Denmark (CSE / Copenhagen)",
    "sweden": "Sweden (SFB / Stockholm)",
    "norway": "Norway (OSE / Oslo)",
    "finland": "Finland (HEX / Helsinki)",
    "germany": "Germany (XETRA / Frankfurt)",
    "france": "France (Euronext Paris)",
    "netherlands": "Netherlands (Euronext Amsterdam)",
    "switzerland": "Switzerland (SIX / Zurich)",
    "spain": "Spain (BME / Madrid)",
    "italy": "Italy (Borsa Italiana / Milan)",
    "london": "London (LSE)",
    "us_stocks": "US Stocks (NYSE/NASDAQ)",
    "chicago": "Chicago (CME/CBOT)",
    "etfs": "ETFs",
}


_EXCHANGE_SL_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "exchange_stop_loss.json"


def _load_exchange_stoploss() -> dict[str, float]:
    """Load per-exchange stop-loss from config file."""
    try:
        if _EXCHANGE_SL_PATH.exists():
            import json
            return json.loads(_EXCHANGE_SL_PATH.read_text())
    except Exception:
        pass
    return {}


def _save_exchange_stoploss(sl_map: dict[str, float]) -> None:
    """Persist per-exchange stop-loss to config file."""
    import json
    _EXCHANGE_SL_PATH.write_text(json.dumps(sl_map, indent=2))


def _get_exchange_stoploss_map() -> dict[str, float]:
    """Read per-exchange stop-loss % (from AutoTrader if running, else from file)."""
    try:
        from src.broker.registry import get_auto_trader
        trader = get_auto_trader()
        if trader:
            return dict(getattr(trader, "_exchange_stop_loss", {}))
    except Exception:
        pass
    return _load_exchange_stoploss()


def _build_exchange_stoploss_table(lang: str):
    """Render current per-exchange stop-loss overrides as a compact table + remove control."""
    sl_map = _get_exchange_stoploss_map()
    if not sl_map:
        return html.Div([
            html.Span(
                t("settings.stoploss_no_overrides", lang),
                style={"color": COLORS["muted"], "fontSize": "0.85rem"},
            ),
            # Hidden remove controls (needed so Dash doesn't error on missing IDs)
            dcc.Dropdown(id="settings-stoploss-remove-dropdown",
                         options=[], style={"display": "none"}),
            html.Div(id="settings-stoploss-remove-btn", style={"display": "none"}),
        ])
    rows = []
    for exch, pct in sorted(sl_map.items()):
        label = _EXCHANGE_LABELS.get(exch, exch)
        rows.append(html.Tr([
            html.Td(label, style={"color": COLORS["text"], "padding": "4px 12px"}),
            html.Td(f"{pct:.1f} %", style={"color": COLORS["orange"], "fontWeight": "bold", "padding": "4px 12px"}),
        ]))
    remove_options = [
        {"label": f"{_EXCHANGE_LABELS.get(k, k)} ({v:.1f}%)", "value": k}
        for k, v in sorted(sl_map.items())
    ]
    return html.Div([
        dbc.Table(
            [html.Thead(html.Tr([
                html.Th(t("settings.exchange_label", lang), style={"color": COLORS["muted"], "padding": "4px 12px"}),
                html.Th(t("settings.stoploss_label", lang), style={"color": COLORS["muted"], "padding": "4px 12px"}),
            ])),
             html.Tbody(rows)],
            bordered=False, hover=True, size="sm",
            color="dark",
            style={"maxWidth": "400px"},
        ),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id="settings-stoploss-remove-dropdown",
                    options=remove_options,
                    placeholder=t("settings.stoploss_remove_placeholder", lang),
                    style={"backgroundColor": COLORS["card"], "color": COLORS["text"]},
                ),
            ], width=5),
            dbc.Col([
                dbc.Button(
                    [html.I(className="bi bi-trash me-1"), t("settings.stoploss_remove", lang)],
                    id="settings-stoploss-remove-btn",
                    color="danger",
                    size="sm",
                ),
            ], width=3),
        ], className="mt-2"),
    ])


def _get_current_display_currency() -> str:
    """Read current display currency from currency_service."""
    try:
        from src.dashboard.currency_service import get_display_currency
        return get_display_currency()
    except Exception:
        return "DKK"


def page_settings():
    """Settings page — strategy toggles and platform configuration."""
    lang = get_language()

    # Read current states
    pattern_enabled = False
    crypto_enabled = True
    pattern_readiness = {"cached": 0, "total": 0, "ready_pct": 0, "min_bars": 50, "seasonal_bars": 504}
    try:
        from src.broker.registry import get_auto_trader
        trader = get_auto_trader()
        if trader:
            pattern_enabled = getattr(trader, "_use_pattern_strategy", False)
            crypto_enabled = getattr(trader, "_crypto_trading_enabled", True)
            ps = getattr(trader, "_pattern_strategy", None)
            if ps:
                # Total = symbols that have received market data
                total = len(ps._bg_data)
                # Cached = symbols with fresh pattern scan results
                cached = sum(1 for sym in ps._bg_data if sym in ps._cache)
                pattern_readiness["cached"] = cached
                pattern_readiness["total"] = total
                pattern_readiness["ready_pct"] = int(cached / total * 100) if total > 0 else 0
    except Exception:
        pass

    # Read current weekend rotation setting
    weekend_rotation_enabled = False
    weekend_crypto_alloc = 60
    weekend_close_stocks = True
    weekend_close_futures = True
    try:
        import json as _json
        _wr_path = _EXCHANGE_SL_PATH.parent / "weekend_rotation.json"
        if _wr_path.exists():
            _wr = _json.loads(_wr_path.read_text())
            weekend_rotation_enabled = _wr.get("enabled", False)
            weekend_crypto_alloc = _wr.get("crypto_allocation_pct", 60)
            weekend_close_stocks = _wr.get("close_stocks", True)
            weekend_close_futures = _wr.get("close_futures", True)
    except Exception:
        pass

    # Read current advanced feedback setting
    adv_feedback_enabled = False
    try:
        import json as _json
        _af_path = _EXCHANGE_SL_PATH.parent / "advanced_feedback.json"
        if _af_path.exists():
            adv_feedback_enabled = _json.loads(_af_path.read_text()).get("enabled", False)
    except Exception:
        pass

    # Read current IBKR data feed settings
    ibkr_enabled = False
    ibkr_host = "127.0.0.1"
    ibkr_port = 4002
    ibkr_client_id = 1
    try:
        import json as _json
        _ibkr_cfg_path = _EXCHANGE_SL_PATH.parent / "ibkr_datafeed.json"
        if _ibkr_cfg_path.exists():
            _icfg = _json.loads(_ibkr_cfg_path.read_text())
            ibkr_enabled = _icfg.get("enabled", False)
            ibkr_host = _icfg.get("host", "127.0.0.1")
            ibkr_port = _icfg.get("port", 4002)
            ibkr_client_id = _icfg.get("client_id", 1)
    except Exception:
        pass

    # Read current max positions
    current_max_positions = 30  # default from config
    try:
        from src.broker.registry import get_auto_trader
        trader = get_auto_trader()
        if trader:
            rm = getattr(trader, "_risk_manager", None)
            if rm:
                current_max_positions = getattr(rm, "max_open_positions", 30)
    except Exception:
        pass
    # Check persisted value
    try:
        import json as _json
        _mp_path = _EXCHANGE_SL_PATH.parent / "max_positions.json"
        if _mp_path.exists():
            current_max_positions = _json.loads(_mp_path.read_text()).get("max_open_positions", current_max_positions)
    except Exception:
        pass

    # Read current position sizing
    current_position_pct = 10.0  # default 10%
    current_max_exposure = 95.0  # default 95%
    current_max_dkk_per_symbol = 5000  # default 5000 DKK
    try:
        import json as _json
        _rs_path = _EXCHANGE_SL_PATH.parent / "risk_sizing.json"
        if _rs_path.exists():
            _rs = _json.loads(_rs_path.read_text())
            current_position_pct = _rs.get("max_position_pct", current_position_pct)
            current_max_exposure = _rs.get("max_exposure_pct", current_max_exposure)
            current_max_dkk_per_symbol = _rs.get("max_dkk_per_symbol", current_max_dkk_per_symbol)
    except Exception:
        pass

    # Read current global stop-loss
    current_global_sl = 5.0  # default
    try:
        from src.broker.registry import get_auto_trader
        trader = get_auto_trader()
        if trader:
            rm = getattr(trader, "_risk_manager", None)
            if rm:
                current_global_sl = round(getattr(rm, "stop_loss_pct", 0.05) * 100, 1)
    except Exception:
        pass
    try:
        import json as _json
        _gsl_path = _EXCHANGE_SL_PATH.parent / "global_stop_loss.json"
        if _gsl_path.exists():
            current_global_sl = _json.loads(_gsl_path.read_text()).get("stop_loss_pct", current_global_sl)
    except Exception:
        pass

    return dbc.Container([
        html.H2([
            html.I(className="bi bi-gear me-2"),
            t("settings.title", lang),
        ], className="mb-1"),
        html.P(t("settings.subtitle", lang), className="text-muted mb-4"),

        # Strategy Settings
        dbc.Card([
            dbc.CardHeader(
                html.H5(t("settings.strategy_section", lang), className="mb-0"),
                style={"backgroundColor": COLORS["card"]},
            ),
            dbc.CardBody([
                # Pattern Analysis Toggle + Readiness + Link
                dbc.Row([
                    # Left: description + readiness + page link
                    dbc.Col([
                        html.H6(t("settings.pattern_analysis_toggle", lang),
                                style={"color": COLORS["text"]}, className="mb-1"),
                        html.P(t("settings.pattern_analysis_desc", lang),
                               style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                               className="mb-2"),
                        # Readiness bar + link in a row
                        dbc.Row([
                            dbc.Col([
                                html.Div([
                                    html.Span(
                                        f"{t('settings.pattern_readiness', lang)}: ",
                                        style={"color": COLORS["muted"], "fontSize": "0.8rem"},
                                    ),
                                    html.Span(
                                        f"{pattern_readiness['cached']}/{pattern_readiness['total']}",
                                        style={
                                            "color": COLORS["green"] if pattern_readiness["ready_pct"] >= 80
                                            else COLORS["orange"] if pattern_readiness["ready_pct"] >= 30
                                            else COLORS["red"],
                                            "fontWeight": "bold",
                                        },
                                    ),
                                ]),
                                dbc.Progress(
                                    value=pattern_readiness["ready_pct"],
                                    color="success" if pattern_readiness["ready_pct"] >= 80
                                    else "warning" if pattern_readiness["ready_pct"] >= 30
                                    else "danger",
                                    style={"height": "6px"},
                                    className="mt-1 mb-1",
                                ),
                                html.Span(
                                    t("settings.pattern_data_note", lang),
                                    style={"color": COLORS["muted"], "fontSize": "0.75rem"},
                                ),
                            ], width=8),
                            dbc.Col([
                                dcc.Link(
                                    dbc.Button(
                                        [html.I(className="bi bi-search me-1"), t("settings.open_pattern_page", lang)],
                                        size="sm", color="outline-info", className="w-100",
                                    ),
                                    href="/teknisk",
                                ),
                            ], width=4, className="d-flex align-items-center"),
                        ]),
                    ], width=10),
                    # Right: label + toggle switch
                    dbc.Col([
                        html.Div([
                            html.Span(
                                id="settings-pattern-label",
                                children=t("settings.enabled", lang) if pattern_enabled else t("settings.disabled", lang),
                                style={"color": COLORS["text"], "fontSize": "0.85rem", "marginRight": "8px"},
                            ),
                            dbc.Switch(
                                id="settings-pattern-toggle",
                                value=pattern_enabled,
                                label="",
                                style={"display": "inline-block"},
                            ),
                        ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"}),
                    ], width=2, className="d-flex align-items-start justify-content-end"),
                ]),
                html.Div(id="settings-pattern-status", className="mt-2"),

                html.Hr(style={"borderColor": COLORS["border"], "margin": "16px 0"}),

                # Crypto Trading Toggle
                dbc.Row([
                    # Left: description
                    dbc.Col([
                        html.H6(t("settings.crypto_trading_toggle", lang),
                                style={"color": COLORS["text"]}, className="mb-1"),
                        html.P(t("settings.crypto_trading_desc", lang),
                               style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                               className="mb-0"),
                    ], width=10),
                    # Right: label + toggle switch
                    dbc.Col([
                        html.Div([
                            html.Span(
                                id="settings-crypto-label",
                                children=t("settings.enabled", lang) if crypto_enabled else t("settings.disabled", lang),
                                style={"color": COLORS["text"], "fontSize": "0.85rem", "marginRight": "8px"},
                            ),
                            dbc.Switch(
                                id="settings-crypto-toggle",
                                value=crypto_enabled,
                                label="",
                                style={"display": "inline-block"},
                            ),
                        ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"}),
                    ], width=2, className="d-flex align-items-start justify-content-end"),
                ]),
                html.Div(id="settings-crypto-status", className="mt-2"),

                html.Hr(style={"borderColor": COLORS["border"], "margin": "16px 0"}),

                # Advanced Feedback Loop Toggle
                dbc.Row([
                    dbc.Col([
                        html.H6("Advanced Feedback Loop",
                                style={"color": COLORS["text"]}, className="mb-1"),
                        html.P(
                            "Auto-apply performance report recommendations: "
                            "per-exchange stop-loss tightening, drawdown-based exposure reduction, "
                            "Sharpe-based stop-loss adjustment, benchmark-relative position sizing. "
                            "Runs every 20 scans (~20 min).",
                            style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                            className="mb-0",
                        ),
                    ], width=10),
                    dbc.Col([
                        html.Div([
                            html.Span(
                                id="settings-adv-feedback-label",
                                children="Enabled" if adv_feedback_enabled else "Disabled",
                                style={"color": COLORS["text"], "fontSize": "0.85rem", "marginRight": "8px"},
                            ),
                            dbc.Switch(
                                id="settings-adv-feedback-toggle",
                                value=adv_feedback_enabled,
                                label="",
                                style={"display": "inline-block"},
                            ),
                        ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"}),
                    ], width=2, className="d-flex align-items-start justify-content-end"),
                ]),
                html.Div(id="settings-adv-feedback-status", className="mt-2"),

                html.Hr(style={"borderColor": COLORS["border"], "margin": "16px 0"}),

                # Weekend Rotation Toggle
                dbc.Row([
                    dbc.Col([
                        html.H6(t("settings.weekend_rotation_toggle", lang),
                                style={"color": COLORS["text"]}, className="mb-1"),
                        html.P(t("settings.weekend_rotation_desc", lang),
                               style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                               className="mb-2"),
                        # Sub-settings row
                        dbc.Row([
                            dbc.Col([
                                html.Label(
                                    t("settings.weekend_crypto_alloc_label", lang),
                                    style={"color": COLORS["muted"], "fontSize": "0.8rem"},
                                ),
                                dbc.InputGroup([
                                    dbc.Input(
                                        id="settings-weekend-crypto-alloc",
                                        type="number",
                                        min=20, max=100, step=5,
                                        value=weekend_crypto_alloc,
                                        size="sm",
                                        style={
                                            "backgroundColor": COLORS["bg"],
                                            "color": COLORS["text"],
                                            "border": f"1px solid {COLORS['border']}",
                                        },
                                    ),
                                    dbc.InputGroupText(
                                        "%",
                                        style={
                                            "backgroundColor": COLORS["bg"],
                                            "color": COLORS["muted"],
                                            "border": f"1px solid {COLORS['border']}",
                                            "fontSize": "0.85rem",
                                        },
                                    ),
                                ], size="sm"),
                            ], width=3),
                            dbc.Col([
                                html.Label(
                                    t("settings.weekend_close_stocks_label", lang),
                                    style={"color": COLORS["muted"], "fontSize": "0.8rem"},
                                ),
                                dbc.Switch(
                                    id="settings-weekend-close-stocks",
                                    value=weekend_close_stocks,
                                    label="",
                                    style={"display": "inline-block"},
                                ),
                            ], width=3),
                            dbc.Col([
                                html.Label(
                                    t("settings.weekend_close_futures_label", lang),
                                    style={"color": COLORS["muted"], "fontSize": "0.8rem"},
                                ),
                                dbc.Switch(
                                    id="settings-weekend-close-futures",
                                    value=weekend_close_futures,
                                    label="",
                                    style={"display": "inline-block"},
                                ),
                            ], width=3),
                            dbc.Col([
                                html.Label("\u00a0", style={"fontSize": "0.8rem"}),
                                dbc.Button(
                                    [html.I(className="bi bi-save me-1"), t("common.save", lang)],
                                    id="settings-weekend-rotation-save-btn",
                                    size="sm",
                                    color="primary",
                                    className="w-100",
                                ),
                            ], width=3, className="d-flex flex-column"),
                        ], className="mb-1"),
                    ], width=10),
                    # Right: label + toggle switch
                    dbc.Col([
                        html.Div([
                            html.Span(
                                id="settings-weekend-rotation-label",
                                children=t("settings.enabled", lang) if weekend_rotation_enabled else t("settings.disabled", lang),
                                style={"color": COLORS["text"], "fontSize": "0.85rem", "marginRight": "8px"},
                            ),
                            dbc.Switch(
                                id="settings-weekend-rotation-toggle",
                                value=weekend_rotation_enabled,
                                label="",
                                style={"display": "inline-block"},
                            ),
                        ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"}),
                    ], width=2, className="d-flex align-items-start justify-content-end"),
                ]),
                html.Div(id="settings-weekend-rotation-status", className="mt-2"),
            ], style={"backgroundColor": COLORS["card"]}),
        ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
           className="mb-4"),

        # ── IBKR Data Feed ──────────────────────────────────────
        dbc.Card([
            dbc.CardHeader(
                html.H5(t("settings.ibkr_title", lang), className="mb-0"),
                style={"backgroundColor": COLORS["card"]},
            ),
            dbc.CardBody([
                # Toggle + description
                dbc.Row([
                    dbc.Col([
                        html.H6(t("settings.ibkr_subtitle", lang),
                                style={"color": COLORS["text"]}, className="mb-1"),
                        html.P(
                            t("settings.ibkr_desc", lang),
                            style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                            className="mb-0",
                        ),
                    ], width=10),
                    dbc.Col([
                        html.Div([
                            html.Span(
                                id="settings-ibkr-label",
                                children=t("settings.enabled", lang) if ibkr_enabled else t("settings.disabled", lang),
                                style={"color": COLORS["text"], "fontSize": "0.85rem", "marginRight": "8px"},
                            ),
                            dbc.Switch(
                                id="settings-ibkr-toggle",
                                value=ibkr_enabled,
                                label="",
                                style={"display": "inline-block"},
                            ),
                        ], style={"display": "flex", "alignItems": "center", "justifyContent": "flex-end"}),
                    ], width=2, className="d-flex align-items-start justify-content-end"),
                ]),

                html.Hr(style={"borderColor": COLORS["border"], "margin": "12px 0"}),

                # Connection settings
                dbc.Row([
                    dbc.Col([
                        html.Label("Host", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                        dbc.Input(
                            id="settings-ibkr-host",
                            type="text",
                            value=ibkr_host,
                            placeholder="127.0.0.1",
                            size="sm",
                            style={"backgroundColor": COLORS["bg"], "color": COLORS["text"],
                                   "border": f"1px solid {COLORS['border']}"},
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Label("Port", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                        dbc.Input(
                            id="settings-ibkr-port",
                            type="number",
                            value=ibkr_port,
                            placeholder="4002",
                            min=1, max=65535, step=1,
                            size="sm",
                            style={"backgroundColor": COLORS["bg"], "color": COLORS["text"],
                                   "border": f"1px solid {COLORS['border']}"},
                        ),
                        html.Span(t("settings.ibkr_port_help", lang),
                                  style={"color": COLORS["muted"], "fontSize": "0.7rem"}),
                    ], width=4),
                    dbc.Col([
                        html.Label("Client ID", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                        dbc.Input(
                            id="settings-ibkr-client-id",
                            type="number",
                            value=ibkr_client_id,
                            placeholder="1",
                            min=1, max=999, step=1,
                            size="sm",
                            style={"backgroundColor": COLORS["bg"], "color": COLORS["text"],
                                   "border": f"1px solid {COLORS['border']}"},
                        ),
                    ], width=2),
                    dbc.Col([
                        html.Label("\u00a0", style={"fontSize": "0.8rem"}),  # spacer
                        dbc.Button(
                            t("common.save", lang),
                            id="settings-ibkr-save-btn",
                            size="sm",
                            color="primary",
                            className="w-100",
                        ),
                    ], width=2, className="d-flex flex-column"),
                ], className="mb-2"),

                html.Div(id="settings-ibkr-status", className="mt-2"),
            ], style={"backgroundColor": COLORS["card"]}),
        ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
           className="mb-4"),

        # ── Risk Limits ──────────────────────────────────────────
        dbc.Card([
            dbc.CardHeader(
                html.H5(t("settings.risk_section", lang), className="mb-0"),
                style={"backgroundColor": COLORS["card"]},
            ),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6(t("settings.max_positions_label", lang),
                                    style={"color": COLORS["text"], "display": "inline"}, className="mb-1 me-2"),
                            html.Span(id="settings-max-positions-indicator"),
                        ]),
                        html.P(t("settings.max_positions_desc", lang),
                               style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                               className="mb-0"),
                    ], width=6),
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.Input(
                                id="settings-max-positions-input",
                                type="number",
                                min=1, max=200, step=1,
                                value=current_max_positions,
                                style={
                                    "backgroundColor": COLORS["card"],
                                    "color": COLORS["text"],
                                    "borderColor": COLORS["border"],
                                },
                            ),
                            dbc.InputGroupText(
                                t("settings.max_positions_unit", lang),
                                style={
                                    "backgroundColor": COLORS["card"],
                                    "color": COLORS["muted"],
                                    "borderColor": COLORS["border"],
                                },
                            ),
                            dbc.Button(
                                [html.I(className="bi bi-save me-1"), t("settings.stoploss_save", lang)],
                                id="settings-max-positions-save-btn",
                                color="primary",
                            ),
                        ]),
                    ], width=4),
                ]),
                html.Div(id="settings-max-positions-status", className="mt-2"),

                html.Hr(style={"borderColor": COLORS["border"], "margin": "16px 0"}),

                # Global Stop-Loss
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6(t("settings.global_stoploss_label", lang),
                                    style={"color": COLORS["text"], "display": "inline"}, className="mb-1 me-2"),
                            html.Span(id="settings-global-stoploss-indicator"),
                        ]),
                        html.P(t("settings.global_stoploss_desc", lang),
                               style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                               className="mb-0"),
                    ], width=6),
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.Input(
                                id="settings-global-stoploss-input",
                                type="number",
                                min=0.5, max=50, step=0.5,
                                value=current_global_sl,
                                style={
                                    "backgroundColor": COLORS["card"],
                                    "color": COLORS["text"],
                                    "borderColor": COLORS["border"],
                                },
                            ),
                            dbc.InputGroupText(
                                "%",
                                style={
                                    "backgroundColor": COLORS["card"],
                                    "color": COLORS["muted"],
                                    "borderColor": COLORS["border"],
                                },
                            ),
                            dbc.Button(
                                [html.I(className="bi bi-save me-1"), t("settings.stoploss_save", lang)],
                                id="settings-global-stoploss-save-btn",
                                color="primary",
                            ),
                        ]),
                    ], width=4),
                ]),
                html.Div(id="settings-global-stoploss-status", className="mt-2"),

                html.Hr(style={"borderColor": COLORS["border"], "margin": "16px 0"}),

                # Position Size %
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6(t("settings.position_pct_label", lang),
                                    style={"color": COLORS["text"], "display": "inline"}, className="mb-1 me-2"),
                            html.Span(id="settings-position-pct-indicator"),
                        ]),
                        html.P(t("settings.position_pct_desc", lang),
                               style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                               className="mb-0"),
                    ], width=6),
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.Input(
                                id="settings-position-pct-input",
                                type="number",
                                min=1, max=50, step=1,
                                value=current_position_pct,
                                style={
                                    "backgroundColor": COLORS["card"],
                                    "color": COLORS["text"],
                                    "borderColor": COLORS["border"],
                                },
                            ),
                            dbc.InputGroupText(
                                "%",
                                style={
                                    "backgroundColor": COLORS["card"],
                                    "color": COLORS["muted"],
                                    "borderColor": COLORS["border"],
                                },
                            ),
                            dbc.Button(
                                [html.I(className="bi bi-save me-1"), t("settings.stoploss_save", lang)],
                                id="settings-position-pct-save-btn",
                                color="primary",
                            ),
                        ]),
                    ], width=4),
                ]),
                html.Div(id="settings-position-pct-status", className="mt-2"),

                html.Hr(style={"borderColor": COLORS["border"], "margin": "16px 0"}),

                # Max DKK per Symbol
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6(t("settings.max_dkk_per_symbol_label", lang),
                                    style={"color": COLORS["text"], "display": "inline"}, className="mb-1 me-2"),
                            html.Span(id="settings-max-dkk-per-symbol-indicator"),
                        ]),
                        html.P(t("settings.max_dkk_per_symbol_desc", lang),
                               style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                               className="mb-0"),
                    ], width=6),
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.Input(
                                id="settings-max-dkk-per-symbol-input",
                                type="number",
                                min=100, max=1000000, step=500,
                                value=current_max_dkk_per_symbol,
                                style={
                                    "backgroundColor": COLORS["card"],
                                    "color": COLORS["text"],
                                    "borderColor": COLORS["border"],
                                },
                            ),
                            dbc.InputGroupText(
                                "DKK",
                                style={
                                    "backgroundColor": COLORS["card"],
                                    "color": COLORS["muted"],
                                    "borderColor": COLORS["border"],
                                },
                            ),
                            dbc.Button(
                                [html.I(className="bi bi-save me-1"), t("settings.stoploss_save", lang)],
                                id="settings-max-dkk-per-symbol-save-btn",
                                color="primary",
                            ),
                        ]),
                    ], width=4),
                ]),
                html.Div(id="settings-max-dkk-per-symbol-status", className="mt-2"),

                html.Hr(style={"borderColor": COLORS["border"], "margin": "16px 0"}),

                # Max Exposure %
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6(t("settings.max_exposure_label", lang),
                                    style={"color": COLORS["text"], "display": "inline"}, className="mb-1 me-2"),
                            html.Span(id="settings-max-exposure-indicator"),
                        ]),
                        html.P(t("settings.max_exposure_desc", lang),
                               style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                               className="mb-0"),
                    ], width=6),
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.Input(
                                id="settings-max-exposure-input",
                                type="number",
                                min=10, max=100, step=5,
                                value=current_max_exposure,
                                style={
                                    "backgroundColor": COLORS["card"],
                                    "color": COLORS["text"],
                                    "borderColor": COLORS["border"],
                                },
                            ),
                            dbc.InputGroupText(
                                "%",
                                style={
                                    "backgroundColor": COLORS["card"],
                                    "color": COLORS["muted"],
                                    "borderColor": COLORS["border"],
                                },
                            ),
                            dbc.Button(
                                [html.I(className="bi bi-save me-1"), t("settings.stoploss_save", lang)],
                                id="settings-max-exposure-save-btn",
                                color="primary",
                            ),
                        ]),
                    ], width=4),
                ]),
                html.Div(id="settings-max-exposure-status", className="mt-2"),
            ], style={"backgroundColor": COLORS["card"]}),
        ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
           className="mb-4"),

        # ── Per-Exchange Stop-Loss Settings ──────────────────────
        dbc.Card([
            dbc.CardHeader(
                html.H5(t("settings.exchange_stoploss_section", lang), className="mb-0"),
                style={"backgroundColor": COLORS["card"]},
            ),
            dbc.CardBody([
                html.P(
                    t("settings.exchange_stoploss_desc", lang),
                    style={"color": COLORS["muted"], "fontSize": "0.85rem"},
                    className="mb-3",
                ),
                dbc.Row([
                    dbc.Col([
                        dbc.Label(
                            t("settings.exchange_label", lang),
                            html_for="settings-exchange-dropdown",
                            style={"color": COLORS["text"], "fontSize": "0.85rem"},
                        ),
                        dcc.Dropdown(
                            id="settings-exchange-dropdown",
                            options=[
                                {"label": "Crypto (24/7)",       "value": "crypto"},
                                {"label": "New Zealand (NZX)",   "value": "new_zealand"},
                                {"label": "Australia (ASX)",     "value": "australia"},
                                {"label": "Japan (TSE)",         "value": "japan"},
                                {"label": "Hong Kong (HKEX)",    "value": "hong_kong"},
                                {"label": "India (NSE)",         "value": "india"},
                                {"label": "Denmark (CSE / Copenhagen)", "value": "denmark"},
                                {"label": "Sweden (SFB / Stockholm)",   "value": "sweden"},
                                {"label": "Norway (OSE / Oslo)",        "value": "norway"},
                                {"label": "Finland (HEX / Helsinki)",   "value": "finland"},
                                {"label": "Germany (XETRA / Frankfurt)","value": "germany"},
                                {"label": "France (Euronext Paris)",    "value": "france"},
                                {"label": "Netherlands (Euronext Amsterdam)", "value": "netherlands"},
                                {"label": "Switzerland (SIX / Zurich)", "value": "switzerland"},
                                {"label": "Spain (BME / Madrid)",       "value": "spain"},
                                {"label": "Italy (Borsa Italiana / Milan)", "value": "italy"},
                                {"label": "London (LSE)",        "value": "london"},
                                {"label": "US Stocks (NYSE/NASDAQ)", "value": "us_stocks"},
                                {"label": "Chicago (CME/CBOT)",  "value": "chicago"},
                                {"label": "ETFs",                "value": "etfs"},
                            ],
                            placeholder=t("settings.exchange_placeholder", lang),
                            style={
                                "backgroundColor": COLORS["card"],
                                "color": COLORS["text"],
                            },
                            className="mb-2",
                        ),
                    ], width=5),
                    dbc.Col([
                        dbc.Label(
                            t("settings.stoploss_label", lang),
                            html_for="settings-stoploss-input",
                            style={"color": COLORS["text"], "fontSize": "0.85rem"},
                        ),
                        dbc.InputGroup([
                            dbc.Input(
                                id="settings-stoploss-input",
                                type="number",
                                min=0.5, max=50, step=0.5,
                                placeholder="5.0",
                                style={
                                    "backgroundColor": COLORS["card"],
                                    "color": COLORS["text"],
                                    "borderColor": COLORS["border"],
                                },
                            ),
                            dbc.InputGroupText(
                                "%",
                                style={
                                    "backgroundColor": COLORS["card"],
                                    "color": COLORS["muted"],
                                    "borderColor": COLORS["border"],
                                },
                            ),
                        ], className="mb-2"),
                    ], width=3),
                    dbc.Col([
                        dbc.Label("\u00a0", style={"fontSize": "0.85rem"}),  # spacer
                        dbc.Button(
                            [html.I(className="bi bi-save me-1"), t("settings.stoploss_save", lang)],
                            id="settings-stoploss-save-btn",
                            color="primary",
                            size="md",
                            className="w-100",
                        ),
                    ], width=2, className="d-flex flex-column"),
                ]),
                html.Div(id="settings-stoploss-status", className="mt-2"),

                # Current exchange stop-loss table
                html.Div(id="settings-stoploss-table", className="mt-3",
                         children=_build_exchange_stoploss_table(lang)),
            ], style={"backgroundColor": COLORS["card"]}),
        ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
           className="mb-4"),

        # Display Currency
        dbc.Card([
            dbc.CardHeader(
                html.H5(t("settings.currency_section", lang), className="mb-0"),
                style={"backgroundColor": COLORS["card"]},
            ),
            dbc.CardBody([
                html.P(t("settings.currency_desc", lang),
                       style={"color": COLORS["muted"], "fontSize": "0.85rem"}, className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        dbc.Select(
                            id="settings-currency-select",
                            options=[
                                {"label": "DKK (kr) - Dansk Krone", "value": "DKK"},
                                {"label": "USD ($) - US Dollar", "value": "USD"},
                                {"label": "EUR (\u20ac) - Euro", "value": "EUR"},
                            ],
                            value=_get_current_display_currency(),
                        ),
                    ], width=4),
                    dbc.Col([
                        dbc.Button(
                            t("stoploss_save" if lang == "da" else "settings.stoploss_save", lang) if False else t("settings.stoploss_save", lang),
                            id="settings-currency-save-btn",
                            color="success",
                            size="md",
                        ),
                    ], width=2),
                    dbc.Col([
                        html.Div(id="settings-currency-rate",
                                 style={"color": COLORS["muted"], "fontSize": "0.85rem", "paddingTop": "8px"}),
                    ], width=6),
                ]),
                html.Div(id="settings-currency-status", className="mt-2"),
            ], style={"backgroundColor": COLORS["card"]}),
        ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
           className="mb-4"),

        # ── Sell Positions ──────────────────────────────────────────
        dbc.Card([
            dbc.CardHeader(
                html.H5([html.I(className="bi bi-cash-coin me-2"), t("sell.title", lang)], className="mb-0"),
                style={"backgroundColor": COLORS["card"]},
            ),
            dbc.CardBody([
                html.P(t("sell.subtitle", lang), style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
                dbc.Button(
                    [html.I(className="bi bi-box-arrow-right me-2"), t("sell.nav_btn", lang)],
                    href="/sell",
                    color="danger",
                    outline=True,
                    className="w-100",
                ),
            ], style={"backgroundColor": COLORS["card"]}),
        ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
           className="mb-4"),

        # Placeholder for future settings
        dbc.Card([
            dbc.CardBody([
                html.P([
                    html.I(className="bi bi-info-circle me-2"),
                    t("settings.more_coming", lang),
                ], style={"color": COLORS["muted"]}, className="mb-0"),
            ], style={"backgroundColor": COLORS["card"]}),
        ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}),

    ], fluid=True, className="p-4")


# ══════════════════════════════════════════════════════════════
#  Performance Report page
# ══════════════════════════════════════════════════════════════


def page_performance_report():
    lang = get_language()
    return dbc.Container([
        html.H3([html.I(className="bi bi-file-earmark-pdf me-2"), t("nav.reports")],
                 style={"color": COLORS["text"]}),
        html.P(t("portfolio.download_report"),
               style={"color": COLORS["muted"]}),
        html.Hr(style={"borderColor": COLORS["border"]}),

        # Report buttons
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6([html.I(className="bi bi-graph-up me-2"), "Portfolio"],
                         style={"color": COLORS["text"]}),
                html.P("Performance, positioner, P&L",
                       style={"color": COLORS["muted"], "fontSize": "0.85rem"}, className="mb-2"),
                dbc.Button(
                    [html.I(className="bi bi-download me-2"), "Download PDF"],
                    id="btn-generate-perf-report", color="success", outline=True, className="w-100",
                ),
            ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
                       "borderRadius": "8px"}), md=4, className="mb-3"),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6([html.I(className="bi bi-receipt me-2"), t("nav.tax")],
                         style={"color": COLORS["text"]}),
                html.P("CSV med handler, udbytte, skat",
                       style={"color": COLORS["muted"], "fontSize": "0.85rem"}, className="mb-2"),
                dbc.Button(
                    [html.I(className="bi bi-download me-2"), "Download CSV"],
                    id="btn-generate-tax-report", color="warning", outline=True, className="w-100",
                ),
            ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
                       "borderRadius": "8px"}), md=4, className="mb-3"),

            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6([html.I(className="bi bi-gear me-2"), t("nav.settings")],
                         style={"color": COLORS["text"]}),
                html.P("Aktuel konfiguration (JSON)",
                       style={"color": COLORS["muted"], "fontSize": "0.85rem"}, className="mb-2"),
                dbc.Button(
                    [html.I(className="bi bi-download me-2"), "Download JSON"],
                    id="btn-generate-settings-report", color="info", outline=True, className="w-100",
                ),
            ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
                       "borderRadius": "8px"}), md=4, className="mb-3"),
        ]),

        # ── Fee Report Section ──
        html.Hr(style={"borderColor": COLORS["border"]}),
        html.H4([html.I(className="bi bi-cash-coin me-2"), "Gebyrrapport (ugentlig)"],
                 style={"color": COLORS["text"]}, className="mt-3 mb-3"),
        html.P("Handelsgebyrer fordelt pr. uge, børs og mægler.",
               style={"color": COLORS["muted"]}),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6([html.I(className="bi bi-cash-coin me-2"), "Gebyrrapport"],
                         style={"color": COLORS["text"]}),
                html.P("Ugentlig gebyrrapport med alle omkostninger",
                       style={"color": COLORS["muted"], "fontSize": "0.85rem"}, className="mb-2"),
                dbc.Button(
                    [html.I(className="bi bi-download me-2"), "Download CSV"],
                    id="btn-generate-fee-report", color="danger", outline=True, className="w-100",
                ),
            ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
                       "borderRadius": "8px"}), md=4, className="mb-3"),
        ]),

        # Fee summary table (rendered on button click)
        html.Div(id="fee-report-table", className="mt-3"),

        html.Div(id="perf-report-status", className="mt-2"),
        dcc.Download(id="download-perf-report"),
        dcc.Download(id="download-tax-report"),
        dcc.Download(id="download-settings-report"),
        dcc.Download(id="download-fee-report"),
    ], fluid=True, className="p-4")


@callback(
    [Output("download-perf-report", "data"), Output("perf-report-status", "children")],
    Input("btn-generate-perf-report", "n_clicks"),
    prevent_initial_call=True,
)
def _download_perf_report(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update
    try:
        pdf_bytes = generate_performance_report()
        filename = f"alpha_performance_{datetime.now():%Y%m%d_%H%M}.pdf"
        return dcc.send_bytes(pdf_bytes, filename), html.Span(
            "Report generated successfully.", style={"color": COLORS["green"]}
        )
    except Exception as e:
        logger.warning(f"Performance report generation failed: {e}")
        return dash.no_update, html.Span(
            f"Report generation failed: {e}", style={"color": COLORS["red"]}
        )


# ── Tax report download ──
@callback(
    Output("download-tax-report", "data"),
    Output("perf-report-status", "children", allow_duplicate=True),
    Input("btn-generate-tax-report", "n_clicks"),
    prevent_initial_call=True,
)
def _download_tax_report(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update
    try:
        import csv, io
        from src.broker.paper_broker import PaperBroker
        pb = PaperBroker()
        trades = list(pb._portfolio.closed_trades) if hasattr(pb._portfolio, 'closed_trades') else []
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Date", "Symbol", "Side", "Qty", "Entry Price", "Exit Price", "P&L", "Commission"])
        for tr in trades:
            writer.writerow([
                getattr(tr, "exit_date", ""), getattr(tr, "symbol", ""),
                getattr(tr, "side", ""), getattr(tr, "qty", 0),
                f"{getattr(tr, 'entry_price', 0):.2f}", f"{getattr(tr, 'exit_price', 0):.2f}",
                f"{getattr(tr, 'net_pnl', 0):.2f}", f"{getattr(tr, 'commission_cost', 0):.2f}",
            ])
        filename = f"alpha_tax_report_{datetime.now():%Y%m%d_%H%M}.csv"
        return dcc.send_string(buf.getvalue(), filename), html.Span(
            f"Tax report: {len(trades)} handler eksporteret.", style={"color": COLORS["green"]})
    except Exception as e:
        return dash.no_update, html.Span(f"Tax report fejl: {e}", style={"color": COLORS["red"]})


# ── Settings export ──
@callback(
    Output("download-settings-report", "data"),
    Output("perf-report-status", "children", allow_duplicate=True),
    Input("btn-generate-settings-report", "n_clicks"),
    prevent_initial_call=True,
)
def _download_settings_report(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update
    try:
        import json as _json
        # Gather all config files into one dict
        config_dir = Path("config")
        exported = {}
        for f in sorted(config_dir.glob("*.json")):
            try:
                exported[f.name] = _json.loads(f.read_text())
            except Exception:
                exported[f.name] = "parse error"
        for f in sorted(config_dir.glob("*.yaml")):
            try:
                exported[f.name] = f.read_text()
            except Exception:
                pass
        content = _json.dumps(exported, indent=2, default=str)
        filename = f"alpha_settings_{datetime.now():%Y%m%d_%H%M}.json"
        return dcc.send_string(content, filename), html.Span(
            "Settings eksporteret.", style={"color": COLORS["green"]})
    except Exception as e:
        return dash.no_update, html.Span(f"Settings fejl: {e}", style={"color": COLORS["red"]})


# ── Fee report download + table ──
@callback(
    Output("download-fee-report", "data"),
    Output("fee-report-table", "children"),
    Output("perf-report-status", "children", allow_duplicate=True),
    Input("btn-generate-fee-report", "n_clicks"),
    prevent_initial_call=True,
)
def _download_fee_report(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update
    try:
        import csv, io, sqlite3
        from datetime import datetime as _dt, timedelta
        from collections import defaultdict
        from src.fees.fee_calculator import FeeCalculator, get_exchange_for_symbol

        # Load closed trades from portfolio DB
        db_path = Path("data_cache/paper_portfolio.db")
        if not db_path.exists():
            return dash.no_update, dash.no_update, html.Span(
                "Ingen portefølje-database fundet.", style={"color": COLORS["red"]})

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT symbol, side, qty, entry_price, exit_price, "
            "entry_time, exit_time, realized_pnl FROM closed_trades "
            "ORDER BY exit_time"
        ).fetchall()
        conn.close()

        if not rows:
            return dash.no_update, dash.no_update, html.Span(
                "Ingen lukkede handler fundet.", style={"color": COLORS["red"]})

        # Compute fees for each trade using FeeCalculator
        fee_calc = FeeCalculator(broker="paper")
        trade_fees = []
        for row in rows:
            symbol = row["symbol"]
            qty = row["qty"]
            entry_price = row["entry_price"]
            exit_price = row["exit_price"]
            exit_time = row["exit_time"] or ""
            entry_time = row["entry_time"] or ""
            pnl = row["realized_pnl"]

            # Calculate entry + exit fees
            entry_fee = fee_calc.calculate(symbol, "buy", qty, entry_price)
            exit_fee = fee_calc.calculate(symbol, "sell", qty, exit_price)
            total_fee = entry_fee.total + exit_fee.total

            exchange = get_exchange_for_symbol(symbol)

            trade_fees.append({
                "symbol": symbol,
                "exchange": exchange,
                "qty": qty,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "pnl": pnl,
                "entry_commission": entry_fee.commission,
                "exit_commission": exit_fee.commission,
                "spread_cost": entry_fee.spread_cost + exit_fee.spread_cost,
                "stamp_duty": entry_fee.stamp_duty + exit_fee.stamp_duty,
                "transaction_tax": entry_fee.transaction_tax + exit_fee.transaction_tax,
                "fx_spread": entry_fee.fx_spread_cost + exit_fee.fx_spread_cost,
                "exchange_fee": entry_fee.exchange_fee + exit_fee.exchange_fee,
                "total_fee": total_fee,
            })

        # ── Group by ISO week ──
        weekly = defaultdict(lambda: {
            "trades": 0, "volume": 0.0, "commission": 0.0,
            "spread": 0.0, "stamp_duty": 0.0, "tax": 0.0,
            "fx_spread": 0.0, "exchange_fee": 0.0, "total_fee": 0.0,
            "pnl": 0.0, "net_pnl": 0.0,
        })
        exchange_totals = defaultdict(lambda: {"trades": 0, "total_fee": 0.0, "volume": 0.0})

        for tf in trade_fees:
            # Parse exit_time for week grouping
            try:
                dt = _dt.fromisoformat(tf["exit_time"])
                iso = dt.isocalendar()
                week_key = f"{iso[0]}-W{iso[1]:02d}"
                week_start = _dt.fromisocalendar(iso[0], iso[1], 1).strftime("%d/%m")
                week_end = _dt.fromisocalendar(iso[0], iso[1], 7).strftime("%d/%m")
                week_label = f"{week_key} ({week_start}-{week_end})"
            except (ValueError, TypeError):
                week_label = "Ukendt"

            volume = tf["qty"] * tf["exit_price"]
            w = weekly[week_label]
            w["trades"] += 1
            w["volume"] += volume
            w["commission"] += tf["entry_commission"] + tf["exit_commission"]
            w["spread"] += tf["spread_cost"]
            w["stamp_duty"] += tf["stamp_duty"]
            w["tax"] += tf["transaction_tax"]
            w["fx_spread"] += tf["fx_spread"]
            w["exchange_fee"] += tf["exchange_fee"]
            w["total_fee"] += tf["total_fee"]
            w["pnl"] += tf["pnl"]
            w["net_pnl"] += tf["pnl"] - tf["total_fee"]

            ex = exchange_totals[tf["exchange"]]
            ex["trades"] += 1
            ex["total_fee"] += tf["total_fee"]
            ex["volume"] += volume

        # ── Build CSV ──
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "Uge", "Handler", "Volumen", "Kommission", "Spread",
            "Stempelafgift", "Transaktionsskat", "FX-spread",
            "Børsgebyr", "Gebyr i alt", "Brutto P&L", "Netto P&L",
        ])
        grand_total = 0.0
        for week_label in sorted(weekly.keys()):
            w = weekly[week_label]
            grand_total += w["total_fee"]
            writer.writerow([
                week_label, w["trades"], f"{w['volume']:.2f}",
                f"{w['commission']:.2f}", f"{w['spread']:.2f}",
                f"{w['stamp_duty']:.2f}", f"{w['tax']:.2f}",
                f"{w['fx_spread']:.2f}", f"{w['exchange_fee']:.2f}",
                f"{w['total_fee']:.2f}", f"{w['pnl']:.2f}", f"{w['net_pnl']:.2f}",
            ])
        writer.writerow([])
        writer.writerow(["--- Gebyrer pr. børs ---"])
        writer.writerow(["Børs", "Handler", "Volumen", "Gebyr i alt", "Gebyr %"])
        for exch in sorted(exchange_totals.keys()):
            ex = exchange_totals[exch]
            fee_pct = (ex["total_fee"] / ex["volume"] * 100) if ex["volume"] > 0 else 0
            writer.writerow([exch, ex["trades"], f"{ex['volume']:.2f}",
                             f"{ex['total_fee']:.2f}", f"{fee_pct:.4f}%"])

        filename = f"alpha_fee_report_{_dt.now():%Y%m%d_%H%M}.csv"

        # ── Build dashboard table ──
        # Weekly summary table
        weekly_rows = []
        for week_label in sorted(weekly.keys(), reverse=True)[:12]:
            w = weekly[week_label]
            fee_pct = (w["total_fee"] / w["volume"] * 100) if w["volume"] > 0 else 0
            weekly_rows.append(html.Tr([
                html.Td(week_label, style={"color": COLORS["text"]}),
                html.Td(str(w["trades"]), style={"color": COLORS["text"], "textAlign": "right"}),
                html.Td(f"${w['volume']:,.0f}", style={"color": COLORS["text"], "textAlign": "right"}),
                html.Td(f"${w['commission']:.2f}", style={"color": COLORS["orange"], "textAlign": "right"}),
                html.Td(f"${w['spread']:.2f}", style={"color": COLORS["muted"], "textAlign": "right"}),
                html.Td(f"${w['total_fee']:.2f}", style={
                    "color": COLORS["red"], "textAlign": "right", "fontWeight": "bold"}),
                html.Td(f"{fee_pct:.3f}%", style={"color": COLORS["muted"], "textAlign": "right"}),
                html.Td(f"${w['pnl']:.2f}", style={
                    "color": COLORS["green"] if w["pnl"] >= 0 else COLORS["red"], "textAlign": "right"}),
                html.Td(f"${w['net_pnl']:.2f}", style={
                    "color": COLORS["green"] if w["net_pnl"] >= 0 else COLORS["red"],
                    "textAlign": "right", "fontWeight": "bold"}),
            ]))

        weekly_table = dbc.Card(dbc.CardBody([
            html.H5([html.I(className="bi bi-calendar-week me-2"), "Ugentlig gebyroversigt (seneste 12 uger)"],
                     style={"color": COLORS["accent"]}, className="mb-3"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Uge", style={"color": COLORS["muted"]}),
                    html.Th("Handler", style={"color": COLORS["muted"], "textAlign": "right"}),
                    html.Th("Volumen", style={"color": COLORS["muted"], "textAlign": "right"}),
                    html.Th("Kommission", style={"color": COLORS["muted"], "textAlign": "right"}),
                    html.Th("Spread", style={"color": COLORS["muted"], "textAlign": "right"}),
                    html.Th("Gebyr i alt", style={"color": COLORS["muted"], "textAlign": "right"}),
                    html.Th("Gebyr %", style={"color": COLORS["muted"], "textAlign": "right"}),
                    html.Th("Brutto P&L", style={"color": COLORS["muted"], "textAlign": "right"}),
                    html.Th("Netto P&L", style={"color": COLORS["muted"], "textAlign": "right"}),
                ])),
                html.Tbody(weekly_rows),
            ], style={"width": "100%", "borderCollapse": "collapse"},
               className="table table-dark table-sm table-hover"),
        ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
                   "borderRadius": "8px"}, className="mb-3")

        # Exchange breakdown table
        exch_rows = []
        for exch in sorted(exchange_totals.keys()):
            ex = exchange_totals[exch]
            fee_pct = (ex["total_fee"] / ex["volume"] * 100) if ex["volume"] > 0 else 0
            exch_rows.append(html.Tr([
                html.Td(exch.replace("_", " ").title(), style={"color": COLORS["text"]}),
                html.Td(str(ex["trades"]), style={"color": COLORS["text"], "textAlign": "right"}),
                html.Td(f"${ex['volume']:,.0f}", style={"color": COLORS["text"], "textAlign": "right"}),
                html.Td(f"${ex['total_fee']:.2f}", style={
                    "color": COLORS["red"], "textAlign": "right", "fontWeight": "bold"}),
                html.Td(f"{fee_pct:.4f}%", style={"color": COLORS["muted"], "textAlign": "right"}),
            ]))

        exchange_table = dbc.Card(dbc.CardBody([
            html.H5([html.I(className="bi bi-globe me-2"), "Gebyrer pr. børs"],
                     style={"color": COLORS["accent"]}, className="mb-3"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Børs", style={"color": COLORS["muted"]}),
                    html.Th("Handler", style={"color": COLORS["muted"], "textAlign": "right"}),
                    html.Th("Volumen", style={"color": COLORS["muted"], "textAlign": "right"}),
                    html.Th("Gebyr i alt", style={"color": COLORS["muted"], "textAlign": "right"}),
                    html.Th("Gebyr %", style={"color": COLORS["muted"], "textAlign": "right"}),
                ])),
                html.Tbody(exch_rows),
            ], style={"width": "100%", "borderCollapse": "collapse"},
               className="table table-dark table-sm table-hover"),
        ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
                   "borderRadius": "8px"}, className="mb-3")

        # Grand total KPI
        total_trades = len(trade_fees)
        total_volume = sum(tf["qty"] * tf["exit_price"] for tf in trade_fees)
        total_pnl = sum(tf["pnl"] for tf in trade_fees)
        net_pnl = total_pnl - grand_total

        kpi_row = dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Totale gebyrer", style={"color": COLORS["muted"], "fontSize": "0.8rem"}, className="mb-1"),
                html.H4(f"${grand_total:,.2f}", style={"color": COLORS["red"]}),
            ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
                       "borderRadius": "8px", "textAlign": "center"}), md=3, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Handler i alt", style={"color": COLORS["muted"], "fontSize": "0.8rem"}, className="mb-1"),
                html.H4(str(total_trades), style={"color": COLORS["text"]}),
            ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
                       "borderRadius": "8px", "textAlign": "center"}), md=3, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Gns. gebyr/handel", style={"color": COLORS["muted"], "fontSize": "0.8rem"}, className="mb-1"),
                html.H4(f"${grand_total / total_trades:.2f}" if total_trades else "$0",
                         style={"color": COLORS["orange"]}),
            ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
                       "borderRadius": "8px", "textAlign": "center"}), md=3, className="mb-3"),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Netto P&L (efter gebyr)", style={"color": COLORS["muted"], "fontSize": "0.8rem"}, className="mb-1"),
                html.H4(f"${net_pnl:,.2f}",
                         style={"color": COLORS["green"] if net_pnl >= 0 else COLORS["red"]}),
            ]), style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}",
                       "borderRadius": "8px", "textAlign": "center"}), md=3, className="mb-3"),
        ], className="mb-3")

        table_content = html.Div([kpi_row, weekly_table, exchange_table])

        return (
            dcc.send_string(buf.getvalue(), filename),
            table_content,
            html.Span(
                f"Gebyrrapport: {total_trades} handler, ${grand_total:,.2f} i gebyrer.",
                style={"color": COLORS["green"]}),
        )
    except Exception as e:
        logger.warning(f"Fee report generation failed: {e}")
        import traceback
        logger.warning(traceback.format_exc())
        return dash.no_update, dash.no_update, html.Span(
            f"Gebyrrapport fejl: {e}", style={"color": COLORS["red"]})


# ══════════════════════════════════════════════════════════════
#  Weekend Crypto Rollover Approval Modal
# ══════════════════════════════════════════════════════════════


@callback(
    Output("weekend-approval-modal", "is_open"),
    Output("weekend-approval-body", "children"),
    Input("weekend-approval-poll", "n_intervals"),
    Input("btn-weekend-approve", "n_clicks"),
    Input("btn-weekend-reject", "n_clicks"),
    prevent_initial_call=True,
)
def _weekend_approval_handler(n_intervals, approve_clicks, reject_clicks):
    """Poll for pending weekend approval and handle accept/reject."""
    from dash import ctx
    from src.ops.weekend_approval import (
        get_approval_state, approve, reject, is_pending,
    )

    triggered = ctx.triggered_id

    if triggered == "btn-weekend-approve":
        approve()
        return False, dash.no_update

    if triggered == "btn-weekend-reject":
        reject()
        return False, dash.no_update

    # Polling — check if there's a pending approval
    if not is_pending():
        return False, dash.no_update

    state = get_approval_state()
    est_fees = state.get("estimated_fees", 0)
    crypto_alloc = state.get("crypto_allocation_pct", 60)
    positions = state.get("positions_to_close", [])
    crypto_symbols = state.get("crypto_symbols", [])
    reopen = state.get("reopen_info", "?")

    # Build position close table
    pos_rows = []
    close_fees_total = 0.0
    for p in positions:
        fee = p.get("close_fee", 0)
        close_fees_total += fee
        pos_rows.append(html.Tr([
            html.Td(p.get("symbol", ""), style={"color": COLORS["text"]}),
            html.Td(p.get("exchange", "").replace("_", " ").title(),
                     style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
            html.Td(f"{p.get('qty', 0):.1f}", style={"color": COLORS["text"], "textAlign": "right"}),
            html.Td(f"${p.get('price', 0):,.2f}", style={"color": COLORS["text"], "textAlign": "right"}),
            html.Td(f"${fee:.2f}", style={"color": COLORS["orange"], "textAlign": "right"}),
        ]))

    crypto_entry_fees = est_fees - close_fees_total
    crypto_exit_est = crypto_entry_fees  # symmetric estimate

    body = html.Div([
        # Warning banner
        dbc.Alert([
            html.I(className="bi bi-exclamation-triangle me-2"),
            html.Strong("Alle børser er lukket. "),
            "Weekend crypto-rotation kræver din godkendelse.",
        ], color="warning", className="mb-3"),

        # Fee summary KPIs
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Estimerede gebyrer", className="mb-1",
                       style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                html.H4(f"${est_fees:,.2f}", style={"color": COLORS["red"]}),
            ]), style={"backgroundColor": "#1e2130", "border": f"1px solid {COLORS['border']}",
                       "borderRadius": "8px", "textAlign": "center"}), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Crypto allokering", className="mb-1",
                       style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                html.H4(f"{crypto_alloc}%", style={"color": COLORS["accent"]}),
            ]), style={"backgroundColor": "#1e2130", "border": f"1px solid {COLORS['border']}",
                       "borderRadius": "8px", "textAlign": "center"}), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Positioner lukkes", className="mb-1",
                       style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                html.H4(str(len(positions)), style={"color": COLORS["orange"]}),
            ]), style={"backgroundColor": "#1e2130", "border": f"1px solid {COLORS['border']}",
                       "borderRadius": "8px", "textAlign": "center"}), md=4),
        ], className="mb-3"),

        # Fee breakdown
        html.H6("Gebyrfordeling", style={"color": COLORS["text"]}, className="mt-3"),
        html.Table([
            html.Tbody([
                html.Tr([
                    html.Td("Lukning af aktie/futures-positioner:",
                             style={"color": COLORS["muted"]}),
                    html.Td(f"${close_fees_total:,.2f}",
                             style={"color": COLORS["orange"], "textAlign": "right", "fontWeight": "bold"}),
                ]),
                html.Tr([
                    html.Td("Crypto entry (fredag):",
                             style={"color": COLORS["muted"]}),
                    html.Td(f"${crypto_entry_fees / 2:,.2f}",
                             style={"color": COLORS["orange"], "textAlign": "right", "fontWeight": "bold"}),
                ]),
                html.Tr([
                    html.Td("Crypto exit (mandag):",
                             style={"color": COLORS["muted"]}),
                    html.Td(f"${crypto_entry_fees / 2:,.2f}",
                             style={"color": COLORS["orange"], "textAlign": "right", "fontWeight": "bold"}),
                ]),
                html.Tr([
                    html.Td(html.Strong("I alt:", style={"color": COLORS["text"]})),
                    html.Td(html.Strong(f"${est_fees:,.2f}"),
                             style={"color": COLORS["red"], "textAlign": "right"}),
                ], style={"borderTop": f"1px solid {COLORS['border']}"}),
            ]),
        ], style={"width": "100%"}, className="mb-3"),

        # Positions to close
        html.H6(f"Positioner der lukkes ({len(positions)})", style={"color": COLORS["text"]},
                 className="mt-3") if positions else html.Div(),
        html.Div(
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Symbol", style={"color": COLORS["muted"]}),
                    html.Th("Børs", style={"color": COLORS["muted"]}),
                    html.Th("Antal", style={"color": COLORS["muted"], "textAlign": "right"}),
                    html.Th("Pris", style={"color": COLORS["muted"], "textAlign": "right"}),
                    html.Th("Gebyr", style={"color": COLORS["muted"], "textAlign": "right"}),
                ])),
                html.Tbody(pos_rows),
            ], className="table table-dark table-sm", style={"width": "100%"}),
            style={"maxHeight": "200px", "overflowY": "auto"},
        ) if positions else html.Div(),

        # Crypto targets
        html.H6("Crypto-mål", style={"color": COLORS["text"]}, className="mt-3"),
        html.P(
            ", ".join(crypto_symbols) if crypto_symbols else "BTC-USD, ETH-USD, SOL-USD, BNB-USD",
            style={"color": COLORS["accent"], "fontSize": "0.9rem"},
        ),

        # Reopen info
        html.P([
            html.I(className="bi bi-clock me-2"),
            f"Næste børsåbning: {reopen}",
        ], style={"color": COLORS["muted"], "fontSize": "0.85rem"}, className="mt-2 mb-0"),
    ])

    return True, body


# ══════════════════════════════════════════════════════════════
#  Sell Positions page
# ══════════════════════════════════════════════════════════════


def _get_positions_by_group():
    """Get all open positions grouped by type and exchange.

    The PaperBroker tracks everything in USD internally, but positions
    on foreign exchanges have prices in local currency (AUD, HKD, SEK, etc.).
    To get the correct DKK value we use cost_basis (in USD) + unrealized_pnl (USD)
    which the PaperBroker calculates correctly, then multiply by USD/DKK.
    """
    positions = []
    pb = None
    try:
        from src.broker.registry import get_router
        router = get_router()
        if router:
            positions = list(router.get_positions())
    except Exception:
        pass
    if not positions:
        try:
            from src.broker.paper_broker import PaperBroker
            pb = PaperBroker()
            positions = list(pb.get_positions())
        except Exception:
            pass

    # Use the same FX rate as the portfolio page
    usd_dkk = 6.90
    try:
        import yfinance as yf
        fx = yf.Ticker("DKK=X")
        rate = getattr(fx.fast_info, "last_price", None)
        if rate and rate > 0:
            usd_dkk = rate
    except Exception:
        pass

    from src.dashboard.pages.portfolio import _exchange_from_symbol

    # Get the total positions value in USD from the broker's internal tracking
    # This is accurate because the broker bought in USD and tracks P&L in USD
    total_positions_usd = 0.0
    try:
        # Always use PaperBroker for equity — it tracks everything in USD
        if pb is None:
            from src.broker.paper_broker import PaperBroker
            pb = PaperBroker()
        acc = pb.get_account()
        total_positions_usd = acc.equity - acc.cash
    except Exception:
        total_positions_usd = sum(getattr(p, "market_value", 0) for p in positions)

    # Calculate each position's share of total for proportional DKK values
    raw_total = sum(getattr(p, "market_value", 0) for p in positions) or 1.0
    total_dkk = total_positions_usd * usd_dkk

    result = {"all": [], "crypto": [], "stocks": [], "bonds": [], "commodities": [], "exchanges": {}}
    for p in positions:
        sym = getattr(p, "symbol", "")
        qty = getattr(p, "qty", 0)
        price = getattr(p, "current_price", 0)
        mv = getattr(p, "market_value", 0)
        # Proportional share of the correctly-tracked USD total
        share = mv / raw_total if raw_total > 0 else 0
        val_dkk = share * total_dkk
        exchange = _exchange_from_symbol(sym)
        pnl = getattr(p, "unrealized_pnl", 0)
        pnl_pct = getattr(p, "unrealized_pnl_pct", 0)
        pnl_dkk = pnl * usd_dkk
        entry = {"symbol": sym, "qty": qty, "price": price, "value_dkk": val_dkk,
                 "exchange": exchange, "pnl_dkk": pnl_dkk, "pnl_pct": pnl_pct}
        result["all"].append(entry)
        cls = _asset_class(sym)
        if cls == "crypto":
            result["crypto"].append(entry)
        elif cls == "bonds":
            result["bonds"].append(entry)
        elif cls == "commodities":
            result["commodities"].append(entry)
        else:
            result["stocks"].append(entry)
        result["exchanges"].setdefault(exchange, []).append(entry)
    return result


def _sell_card(title, desc, count, value_dkk, btn_id, color="danger"):
    from src.dashboard.currency_service import format_value_dkk
    return dbc.Card([
        dbc.CardBody([
            html.H5(title, className="text-light mb-2"),
            html.P(desc, style={"color": COLORS["muted"], "fontSize": "0.85rem"}, className="mb-2"),
            html.Div([
                html.Span(f"{count} {t('sell.positions_count')}", className="text-light fw-bold me-3"),
                html.Span(format_value_dkk(value_dkk), className="text-warning fw-bold"),
            ], className="mb-3") if count > 0 else html.P(t('sell.no_positions'), className="text-muted mb-3"),
            dbc.Button(
                [html.I(className="bi bi-exclamation-triangle me-2"), title],
                id=btn_id,
                color=color,
                outline=True,
                disabled=count == 0,
                className="w-100",
            ),
        ], style={"backgroundColor": COLORS["card"]}),
    ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"})


def _build_sell_cards():
    """Build the sell card grid from current positions. Used for initial render and refresh."""
    groups = _get_positions_by_group()
    exchanges = sorted(groups["exchanges"].keys())
    exchange_options = [{"label": f"{ex} ({len(groups['exchanges'][ex])} {t('sell.positions_count')})", "value": ex} for ex in exchanges]

    all_val = sum(p["value_dkk"] for p in groups["all"])
    crypto_val = sum(p["value_dkk"] for p in groups["crypto"])
    stocks_val = sum(p["value_dkk"] for p in groups["stocks"])
    bonds_val = sum(p["value_dkk"] for p in groups["bonds"])
    commodities_val = sum(p["value_dkk"] for p in groups["commodities"])

    return html.Div([
        # Row 1: Sell All + Sell Crypto
        dbc.Row([
            dbc.Col(_sell_card(
                t('sell.sell_all'), t('sell.sell_all_desc'),
                len(groups["all"]), all_val, "sell-all-btn",
            ), md=6, className="mb-4"),
            dbc.Col(_sell_card(
                t('sell.sell_crypto'), t('sell.sell_crypto_desc'),
                len(groups["crypto"]), crypto_val, "sell-crypto-btn",
            ), md=6, className="mb-4"),
        ]),

        # Row 2: Sell All Stocks + Sell All Bonds
        dbc.Row([
            dbc.Col(_sell_card(
                t('sell.sell_stocks'), t('sell.sell_stocks_desc'),
                len(groups["stocks"]), stocks_val, "sell-stocks-btn",
            ), md=6, className="mb-4"),
            dbc.Col(_sell_card(
                t('sell.sell_bonds'), t('sell.sell_bonds_desc'),
                len(groups["bonds"]), bonds_val, "sell-bonds-btn",
            ), md=6, className="mb-4"),
        ]),

        # Row 3: Sell All Commodities + Sell by Exchange
        dbc.Row([
            dbc.Col(_sell_card(
                t('sell.sell_commodities'), t('sell.sell_commodities_desc'),
                len(groups["commodities"]), commodities_val, "sell-commodities-btn",
            ), md=6, className="mb-4"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(t('sell.sell_exchange'), className="text-light mb-2"),
                        html.P(t('sell.sell_exchange_desc'), style={"color": COLORS["muted"], "fontSize": "0.85rem"}, className="mb-2"),
                        dcc.Dropdown(
                            id="sell-exchange-dropdown",
                            options=exchange_options,
                            placeholder=t('sell.select_exchange'),
                            style={"backgroundColor": COLORS["bg"], "color": COLORS["text"]},
                            className="mb-3",
                        ),
                        html.Div(id="sell-exchange-info", className="mb-3"),
                        dbc.Button(
                            [html.I(className="bi bi-exclamation-triangle me-2"), t('sell.sell_selected')],
                            id="sell-exchange-btn",
                            color="danger",
                            outline=True,
                            disabled=True,
                            className="w-100",
                        ),
                    ], style={"backgroundColor": COLORS["card"]}),
                ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}),
            ], md=6, className="mb-4"),
        ]),
    ])


def page_sell():
    """Sell positions page with confirmation dialogs."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="bi bi-cash-coin me-2"),
                    t('sell.title'),
                ], className="mb-1"),
                html.P(t('sell.subtitle'), className="text-muted mb-0"),
            ], md=8),
            dbc.Col([
                dbc.Button(
                    [html.I(className="bi bi-arrow-left me-2"), t('sell.back_to_settings')],
                    href="/settings",
                    color="secondary",
                    outline=True,
                    size="sm",
                    className="mt-2",
                ),
            ], md=4, className="text-end"),
        ], className="mb-4"),

        # Dynamic sell cards container — refreshed after each sell
        html.Div(_build_sell_cards(), id="sell-cards-container"),

        # Confirmation modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(t('sell.confirm_title')), close_button=True),
            dbc.ModalBody(id="sell-confirm-body"),
            dbc.ModalFooter([
                dbc.Button(t('sell.confirm_cancel'), id="sell-cancel-btn", color="secondary", className="me-2"),
                dbc.Button(t('sell.confirm_yes'), id="sell-confirm-btn", color="danger"),
            ]),
        ], id="sell-confirm-modal", is_open=False, centered=True),

        # Hidden store for what to sell
        dcc.Store(id="sell-action-store", data=None),

        # Result display
        html.Div(id="sell-result", className="mt-3"),

    ], fluid=True, className="p-4")


# ══════════════════════════════════════════════════════════════
#  Routing
# ══════════════════════════════════════════════════════════════


@callback(Output("page-content", "children"), Input("url", "pathname"), Input("auto-refresh", "n_intervals"), Input("lang-store", "data"), Input("currency-store", "data"))
def display_page(pathname: str, _n: int, _lang: str, _currency: str):
    set_language(_lang or "da")
    # Sync display currency from browser store
    try:
        from src.dashboard.currency_service import set_display_currency
        if _currency and _currency in ("USD", "EUR", "DKK"):
            set_display_currency(_currency, persist=False)
    except Exception:
        pass

    # Don't re-render pages with interactive state on auto-refresh ticks
    ctx = dash.callback_context
    if ctx.triggered and len(ctx.triggered) == 1:
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if trigger_id == "auto-refresh" and pathname in ("/sell", "/settings", "/trading", "/marked"):
            return dash.no_update

    def _error_div(route: str, e: Exception):
        logger.warning(f"Page error {route}: {e}")
        return html.Div([
            html.H4("⚠ Page loading error", style={"color": "#ff4757"}),
            html.P(str(e), style={"color": "#64748b"}),
        ], style={"padding": "40px"})

    # Route mapping
    routes = {
        "/": page_overview,
        # Multi-broker trading pages
        "/portfolio": portfolio_layout,
        "/trading": trading_layout,
        "/tax": tax_center_layout,
        "/markets": market_explorer_layout,
        "/status": broker_status_layout,
        # Research & analysis pages
        "/analyse": page_analyse,
        "/strategier": page_strategies,
        "/risiko": page_risk,
        "/skat": page_tax,
        "/marked": page_market_overview,
        "/sentiment": page_sentiment,
        "/kalender": page_calendar,
        "/regime": page_regime,
        "/stress-test": page_stress_test,
        "/health": page_health,
        "/smart-money": page_smart_money,
        "/options-flow": page_options_flow,
        "/alt-data": page_alt_data,
        "/okonomi": page_economy,
        "/teknisk": page_technical_analysis,
        "/krypto": page_crypto_onchain,
        "/settings": page_settings,
        "/performance": page_performance_report,
        "/sell": page_sell,
    }

    page_func = routes.get(pathname, page_overview)
    try:
        return page_func()
    except Exception as e:
        return _error_div(pathname or "/", e)


# ══════════════════════════════════════════════════════════════
#  Start
# ══════════════════════════════════════════════════════════════

# ── Language selector callbacks ────────────────────────────

@callback(
    [Output("lang-store", "data"), Output("lang-select", "value")],
    Input({"type": "lang-flag", "code": dash.ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def flag_clicked(n_clicks_list):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
    # Find which flag was clicked
    prop_id = ctx.triggered[0]["prop_id"]
    # prop_id looks like '{"code":"en","type":"lang-flag"}.n_clicks'
    import json as _json
    try:
        btn_id = _json.loads(prop_id.split(".")[0])
        code = btn_id["code"]
    except Exception:
        return dash.no_update, dash.no_update
    set_language(code)
    return code, code


@callback(
    [
        Output("nav-trading-section", "children"),
        Output("nav-analysis-section", "children"),
    ],
    [Input("lang-store", "data"), Input("lang-select", "value")],
)
def update_nav_language(stored_lang, selected_lang):
    lang = selected_lang or stored_lang or "da"
    set_language(lang)

    trading_nav = html.Div([
        html.P(t("nav.section_trading", lang), style={
            "color": COLORS["accent"], "fontSize": "0.7rem", "fontWeight": "700",
            "letterSpacing": "2px", "padding": "12px 16px 4px", "marginBottom": 0,
        }),
        dbc.Nav([
            dbc.NavLink([html.I(className="bi bi-pie-chart me-2"), t("nav.portfolio", lang)],
                         href="/portfolio", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-cart-check me-2"), t("nav.trading", lang)],
                         href="/trading", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-calculator me-2"), t("nav.tax_center", lang)],
                         href="/tax", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-globe2 me-2"), t("nav.markets", lang)],
                         href="/markets", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-wifi me-2"), t("nav.broker_status", lang)],
                         href="/status", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-file-earmark-pdf me-2"), t("nav.reports", lang)],
                         href="/performance", active="exact", className="nav-dark"),
        ], vertical=True, pills=True, style={"padding": "0 8px"}),
        html.Hr(style={"borderColor": COLORS["border"], "margin": "8px 16px"}),
    ])

    analysis_nav = html.Div([
        html.P(t("nav.section_analysis", lang), style={
            "color": COLORS["muted"], "fontSize": "0.7rem", "fontWeight": "700",
            "letterSpacing": "2px", "padding": "4px 16px 4px", "marginBottom": 0,
        }),
        dbc.Nav([
            dbc.NavLink([html.I(className="bi bi-speedometer2 me-2"), t("nav.overview", lang)],
                         href="/", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-graph-up me-2"), t("nav.stock_analysis", lang)],
                         href="/analyse", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-robot me-2"), t("nav.strategies", lang)],
                         href="/strategier", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-shield-check me-2"), t("nav.risk", lang)],
                         href="/risiko", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-calculator me-2"), t("nav.tax", lang)],
                         href="/skat", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-globe2 me-2"), t("nav.market_overview", lang)],
                         href="/marked", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-newspaper me-2"), t("nav.sentiment", lang)],
                         href="/sentiment", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-calendar-event me-2"), t("nav.calendar", lang)],
                         href="/kalender", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-activity me-2"), t("nav.regime", lang)],
                         href="/regime", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-lightning-charge me-2"), t("nav.stress_test", lang)],
                         href="/stress-test", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-heart-pulse me-2"), t("nav.system_health", lang)],
                         href="/health", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-bank me-2"), t("nav.smart_money", lang)],
                         href="/smart-money", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-bar-chart-steps me-2"), t("nav.options_flow", lang)],
                         href="/options-flow", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-stars me-2"), t("nav.alt_data", lang)],
                         href="/alt-data", active="exact", className="nav-dark"),
            dbc.NavLink([html.I(className="bi bi-globe-americas me-2"), t("nav.economy", lang)],
                         href="/okonomi", active="exact", className="nav-dark"),
        ], vertical=True, pills=True, style={"padding": "0 8px"}),
        html.Hr(style={"borderColor": COLORS["border"], "margin": "8px 16px"}),
        dbc.Nav([
            dbc.NavLink([html.I(className="bi bi-gear me-2"), t("nav.settings", lang)],
                         href="/settings", active="exact", className="nav-dark"),
        ], vertical=True, pills=True, style={"padding": "0 8px"}),
    ])

    return trading_nav, analysis_nav


# ── Settings callbacks ─────────────────────────────────────

@callback(
    [
        Output("settings-pattern-status", "children"),
        Output("settings-pattern-label", "children"),
    ],
    Input("settings-pattern-toggle", "value"),
    prevent_initial_call=True,
)
def toggle_pattern_strategy(enabled):
    lang = get_language()
    try:
        from src.broker.registry import get_auto_trader
        trader = get_auto_trader()
        if trader:
            trader.set_pattern_strategy(enabled)
            label = t("settings.enabled", lang) if enabled else t("settings.disabled", lang)
            status_color = COLORS["green"] if enabled else COLORS["muted"]
            status = html.Span(
                f"{t('settings.saved', lang)} — Pattern analysis {'ON' if enabled else 'OFF'}",
                style={"color": status_color, "fontSize": "0.85rem"},
            )
            return status, label
        else:
            label = t("settings.disabled", lang)
            return html.Span(
                "AutoTrader not running — setting will apply on next start",
                style={"color": COLORS["orange"], "fontSize": "0.85rem"},
            ), label
    except Exception as e:
        label = t("settings.disabled", lang)
        return html.Span(
            f"Error: {e}",
            style={"color": COLORS["red"], "fontSize": "0.85rem"},
        ), label


# ── Crypto trading toggle callback ─────────────────────────

@callback(
    [
        Output("settings-crypto-status", "children"),
        Output("settings-crypto-label", "children"),
    ],
    Input("settings-crypto-toggle", "value"),
    prevent_initial_call=True,
)
def toggle_crypto_trading(enabled):
    lang = get_language()
    try:
        from src.broker.registry import get_auto_trader
        trader = get_auto_trader()
        if trader:
            trader.set_crypto_trading(enabled)
            label = t("settings.enabled", lang) if enabled else t("settings.disabled", lang)
            status_color = COLORS["green"] if enabled else COLORS["muted"]
            status = html.Span(
                f"{t('settings.saved', lang)} — Crypto trading {'ON' if enabled else 'OFF'}",
                style={"color": status_color, "fontSize": "0.85rem"},
            )
            return status, label
        else:
            label = t("settings.disabled", lang)
            return html.Span(
                "AutoTrader not running — setting will apply on next start",
                style={"color": COLORS["orange"], "fontSize": "0.85rem"},
            ), label
    except Exception as e:
        label = t("settings.disabled", lang)
        return html.Span(
            f"Error: {e}",
            style={"color": COLORS["red"], "fontSize": "0.85rem"},
        ), label


# ── Advanced Feedback Loop callback ───────────────────────

@callback(
    [
        Output("settings-adv-feedback-status", "children"),
        Output("settings-adv-feedback-label", "children"),
    ],
    Input("settings-adv-feedback-toggle", "value"),
    prevent_initial_call=True,
)
def toggle_advanced_feedback(enabled):
    import json
    try:
        cfg_path = _EXCHANGE_SL_PATH.parent / "advanced_feedback.json"
        cfg_path.write_text(json.dumps({"enabled": enabled}, indent=2))

        # Apply to running AutoTrader
        label = "Enabled" if enabled else "Disabled"
        try:
            from src.broker.registry import get_auto_trader
            trader = get_auto_trader()
            if trader:
                trader._advanced_feedback_enabled = enabled
                status_msg = (
                    "Gemt — Advanced feedback ON. "
                    "Stop-losses, eksponering og position sizing justeres automatisk hvert 20. scan (~20 min)."
                    if enabled else
                    "Gemt — Advanced feedback OFF. Kun basis feedback loop (confidence + position size) er aktiv."
                )
                return html.Span(
                    status_msg,
                    style={"color": COLORS["green"] if enabled else COLORS["muted"], "fontSize": "0.85rem"},
                ), label
            else:
                return html.Span(
                    "AutoTrader not running — setting saved, will apply on next start",
                    style={"color": COLORS["orange"], "fontSize": "0.85rem"},
                ), label
        except Exception:
            return html.Span(
                f"Saved — will take effect on next startup ({'ON' if enabled else 'OFF'})",
                style={"color": COLORS["green"] if enabled else COLORS["muted"], "fontSize": "0.85rem"},
            ), label

    except Exception as e:
        return html.Span(
            f"Error: {e}",
            style={"color": COLORS["red"], "fontSize": "0.85rem"},
        ), "Disabled"


# ── Weekend Rotation callback ─────────────────────────────

@callback(
    [
        Output("settings-weekend-rotation-status", "children"),
        Output("settings-weekend-rotation-label", "children"),
    ],
    [
        Input("settings-weekend-rotation-toggle", "value"),
        Input("settings-weekend-rotation-save-btn", "n_clicks"),
    ],
    [
        State("settings-weekend-crypto-alloc", "value"),
        State("settings-weekend-close-stocks", "value"),
        State("settings-weekend-close-futures", "value"),
    ],
    prevent_initial_call=True,
)
def toggle_weekend_rotation(toggle_val, save_clicks, crypto_alloc, close_stocks, close_futures):
    import json
    lang = get_language()
    try:
        cfg_path = _EXCHANGE_SL_PATH.parent / "weekend_rotation.json"
        # Determine what triggered the callback
        ctx = dash.callback_context
        triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

        # Load existing config or default
        try:
            cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        except Exception:
            cfg = {}

        if "toggle" in triggered:
            cfg["enabled"] = toggle_val
        elif "save" in triggered:
            cfg["crypto_allocation_pct"] = int(crypto_alloc or 60)
            cfg["close_stocks"] = bool(close_stocks)
            cfg["close_futures"] = bool(close_futures)
        else:
            cfg["enabled"] = toggle_val

        cfg.setdefault("enabled", False)
        cfg.setdefault("crypto_allocation_pct", 60)
        cfg.setdefault("close_stocks", True)
        cfg.setdefault("close_futures", True)

        cfg_path.write_text(json.dumps(cfg, indent=2))

        enabled = cfg["enabled"]
        label = t("settings.enabled", lang) if enabled else t("settings.disabled", lang)
        status_color = COLORS["green"] if enabled else COLORS["muted"]

        if "save" in triggered:
            msg = (
                f"{t('settings.saved', lang)} — "
                f"crypto {cfg['crypto_allocation_pct']}%, "
                f"close stocks={'ON' if cfg['close_stocks'] else 'OFF'}, "
                f"close futures={'ON' if cfg['close_futures'] else 'OFF'}"
            )
        else:
            msg = f"{t('settings.saved', lang)} — Weekend rotation {'ON' if enabled else 'OFF'}"

        return (
            html.Span(msg, style={"color": status_color, "fontSize": "0.85rem"}),
            label,
        )

    except Exception as e:
        return (
            html.Span(f"Error: {e}", style={"color": COLORS["red"], "fontSize": "0.85rem"}),
            t("settings.disabled", lang),
        )


# ── IBKR Data Feed callback ───────────────────────────────

@callback(
    [
        Output("settings-ibkr-status", "children"),
        Output("settings-ibkr-label", "children"),
    ],
    [
        Input("settings-ibkr-toggle", "value"),
        Input("settings-ibkr-save-btn", "n_clicks"),
    ],
    [
        State("settings-ibkr-host", "value"),
        State("settings-ibkr-port", "value"),
        State("settings-ibkr-client-id", "value"),
    ],
    prevent_initial_call=True,
)
def save_ibkr_settings(enabled, n_clicks, host, port, client_id):
    import json
    from dash import ctx

    host = host or "127.0.0.1"
    port = int(port or 4002)
    client_id = int(client_id or 1)

    cfg = {
        "enabled": enabled,
        "host": host,
        "port": port,
        "client_id": client_id,
    }

    try:
        cfg_path = _EXCHANGE_SL_PATH.parent / "ibkr_datafeed.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))

        # If toggle was flipped or save clicked, try to connect/disconnect live
        label = "Enabled" if enabled else "Disabled"

        if enabled:
            # Attempt a test connection (ib_insync needs an event loop)
            conn_status = ""
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    from src.broker.ibkr_broker import IBKRBroker
                    test_broker = IBKRBroker(host=host, port=port, client_id=client_id)
                    result = test_broker.connect()
                    test_broker.disconnect()
                    conn_status = f" — Forbindelse OK ({result.get('accounts', '?')})"
                finally:
                    loop.close()
            except Exception as conn_err:
                conn_status = f" — Kan ikke forbinde: {conn_err}"

            status = html.Span(
                f"Gemt — IBKR data feed ON ({host}:{port}){conn_status}",
                style={"color": COLORS["green"], "fontSize": "0.85rem"},
            )
        else:
            status = html.Span(
                "Gemt — IBKR data feed OFF (bruger yfinance)",
                style={"color": COLORS["muted"], "fontSize": "0.85rem"},
            )

        # Update running AutoTrader if available
        try:
            from src.broker.registry import get_auto_trader
            trader = get_auto_trader()
            if trader and hasattr(trader, "_ibkr_datafeed_enabled"):
                trader._ibkr_datafeed_enabled = enabled
        except Exception:
            pass

        return status, label

    except Exception as e:
        return html.Span(
            f"Error: {e}",
            style={"color": COLORS["red"], "fontSize": "0.85rem"},
        ), "Disabled"


def _risk_indicator(active: bool) -> dbc.Badge:
    """Return a green 'Aktiv' or gray 'Inaktiv' badge for risk limit status."""
    if active:
        return dbc.Badge("Aktiv", color="success", className="ms-2", style={"fontSize": "0.7rem"})
    return dbc.Badge("Inaktiv", color="secondary", className="ms-2", style={"fontSize": "0.7rem"})


# ── Max Positions callback ────────────────────────────────

@callback(
    Output("settings-max-positions-status", "children"),
    Output("settings-max-positions-indicator", "children"),
    Input("settings-max-positions-save-btn", "n_clicks"),
    State("settings-max-positions-input", "value"),
    prevent_initial_call=True,
)
def save_max_positions(n_clicks, max_pos):
    lang = get_language()
    if max_pos is None or max_pos < 1:
        return html.Span(
            t("settings.max_positions_invalid", lang),
            style={"color": COLORS["orange"], "fontSize": "0.85rem"},
        ), _risk_indicator(False)
    try:
        import json
        max_pos = int(max_pos)

        # Persist to file
        mp_path = _EXCHANGE_SL_PATH.parent / "max_positions.json"
        mp_path.write_text(json.dumps({"max_open_positions": max_pos}, indent=2))

        # Apply to running AutoTrader/RiskManager
        applied = False
        try:
            from src.broker.registry import get_auto_trader
            trader = get_auto_trader()
            if trader:
                rm = getattr(trader, "_risk_manager", None)
                if rm:
                    rm.max_open_positions = max_pos
                    applied = True
        except Exception:
            pass

        return html.Span(
            f"{t('settings.saved', lang)} — max {max_pos} positions",
            style={"color": COLORS["green"], "fontSize": "0.85rem"},
        ), _risk_indicator(applied)
    except Exception as e:
        return html.Span(
            f"Error: {e}",
            style={"color": COLORS["red"], "fontSize": "0.85rem"},
        ), _risk_indicator(False)


# ── Position Size % callback ──────────────────────────────

@callback(
    Output("settings-position-pct-status", "children"),
    Output("settings-position-pct-indicator", "children"),
    Input("settings-position-pct-save-btn", "n_clicks"),
    State("settings-position-pct-input", "value"),
    prevent_initial_call=True,
)
def save_position_pct(n_clicks, pct):
    lang = get_language()
    if pct is None or pct < 1:
        return html.Span("Enter a valid percentage (1-50%).",
                         style={"color": COLORS["orange"], "fontSize": "0.85rem"}), _risk_indicator(False)
    try:
        import json
        pct = float(pct)
        rs_path = _EXCHANGE_SL_PATH.parent / "risk_sizing.json"
        rs = {}
        if rs_path.exists():
            rs = json.loads(rs_path.read_text())
        rs["max_position_pct"] = pct
        rs_path.write_text(json.dumps(rs, indent=2))

        applied = False
        try:
            from src.broker.registry import get_auto_trader
            trader = get_auto_trader()
            if trader:
                trader.position_size_pct = pct / 100.0
                trader._base_position_size_pct = pct / 100.0
                rm = getattr(trader, "_risk_manager", None)
                if rm:
                    rm.max_position_pct = pct / 100.0
                applied = True
        except Exception:
            pass

        return html.Span(
            f"{t('settings.saved', lang)} — max {pct:.0f}% per position",
            style={"color": COLORS["green"], "fontSize": "0.85rem"}), _risk_indicator(applied)
    except Exception as e:
        return html.Span(f"Error: {e}",
                         style={"color": COLORS["red"], "fontSize": "0.85rem"}), _risk_indicator(False)


# ── Max DKK per Symbol callback ──────────────────────────

@callback(
    Output("settings-max-dkk-per-symbol-status", "children"),
    Output("settings-max-dkk-per-symbol-indicator", "children"),
    Input("settings-max-dkk-per-symbol-save-btn", "n_clicks"),
    State("settings-max-dkk-per-symbol-input", "value"),
    prevent_initial_call=True,
)
def save_max_dkk_per_symbol(n_clicks, max_dkk):
    lang = get_language()
    if max_dkk is None or max_dkk < 100:
        return html.Span("Enter a valid amount (min 100 DKK).",
                         style={"color": COLORS["orange"], "fontSize": "0.85rem"}), _risk_indicator(False)
    try:
        import json
        max_dkk = float(max_dkk)
        rs_path = _EXCHANGE_SL_PATH.parent / "risk_sizing.json"
        rs = {}
        if rs_path.exists():
            rs = json.loads(rs_path.read_text())
        rs["max_dkk_per_symbol"] = max_dkk
        rs_path.write_text(json.dumps(rs, indent=2))

        applied = False
        try:
            from src.broker.registry import get_auto_trader
            trader = get_auto_trader()
            if trader:
                trader.max_dkk_per_symbol = max_dkk
                applied = True
        except Exception:
            pass

        return html.Span(
            f"{t('settings.saved', lang)} — max {max_dkk:,.0f} DKK per symbol",
            style={"color": COLORS["green"], "fontSize": "0.85rem"}), _risk_indicator(applied)
    except Exception as e:
        return html.Span(f"Error: {e}",
                         style={"color": COLORS["red"], "fontSize": "0.85rem"}), _risk_indicator(False)


# ── Max Exposure % callback ──────────────────────────────

@callback(
    Output("settings-max-exposure-status", "children"),
    Output("settings-max-exposure-indicator", "children"),
    Input("settings-max-exposure-save-btn", "n_clicks"),
    State("settings-max-exposure-input", "value"),
    prevent_initial_call=True,
)
def save_max_exposure(n_clicks, pct):
    lang = get_language()
    if pct is None or pct < 10:
        return html.Span("Enter a valid percentage (10-100%).",
                         style={"color": COLORS["orange"], "fontSize": "0.85rem"}), _risk_indicator(False)
    try:
        import json
        pct = float(pct)
        rs_path = _EXCHANGE_SL_PATH.parent / "risk_sizing.json"
        rs = {}
        if rs_path.exists():
            rs = json.loads(rs_path.read_text())
        rs["max_exposure_pct"] = pct
        rs_path.write_text(json.dumps(rs, indent=2))

        applied = False
        try:
            from src.broker.registry import get_auto_trader
            trader = get_auto_trader()
            if trader:
                drm = getattr(trader, "_dynamic_risk", None)
                if drm:
                    drm._current_params["max_exposure_pct"] = pct / 100.0
                    applied = True
        except Exception:
            pass

        cash_reserve = 100 - pct
        return html.Span(
            f"{t('settings.saved', lang)} — max {pct:.0f}% exposure ({cash_reserve:.0f}% cash reserve)",
            style={"color": COLORS["green"], "fontSize": "0.85rem"}), _risk_indicator(applied)
    except Exception as e:
        return html.Span(f"Error: {e}",
                         style={"color": COLORS["red"], "fontSize": "0.85rem"}), _risk_indicator(False)


# ── Global Stop-Loss callback ─────────────────────────────

@callback(
    Output("settings-global-stoploss-status", "children"),
    Output("settings-global-stoploss-indicator", "children"),
    Input("settings-global-stoploss-save-btn", "n_clicks"),
    State("settings-global-stoploss-input", "value"),
    prevent_initial_call=True,
)
def save_global_stoploss(n_clicks, sl_pct):
    lang = get_language()
    if sl_pct is None or sl_pct <= 0:
        return html.Span(
            t("settings.stoploss_enter_value", lang),
            style={"color": COLORS["orange"], "fontSize": "0.85rem"},
        ), _risk_indicator(False)
    try:
        import json
        sl_pct = float(sl_pct)

        gsl_path = _EXCHANGE_SL_PATH.parent / "global_stop_loss.json"
        gsl_path.write_text(json.dumps({"stop_loss_pct": sl_pct}, indent=2))

        applied = False
        try:
            from src.broker.registry import get_auto_trader
            trader = get_auto_trader()
            if trader:
                rm = getattr(trader, "_risk_manager", None)
                if rm:
                    rm.stop_loss_pct = sl_pct / 100.0
                    applied = True
        except Exception:
            pass

        return html.Span(
            f"{t('settings.saved', lang)} — global stop-loss: {sl_pct:.1f}%",
            style={"color": COLORS["green"], "fontSize": "0.85rem"},
        ), _risk_indicator(applied)
    except Exception as e:
        return html.Span(
            f"Error: {e}",
            style={"color": COLORS["red"], "fontSize": "0.85rem"},
        ), _risk_indicator(False)


# ── Per-Exchange Stop-Loss callback ───────────────────────

@callback(
    [
        Output("settings-stoploss-status", "children"),
        Output("settings-stoploss-table", "children"),
    ],
    Input("settings-stoploss-save-btn", "n_clicks"),
    [
        State("settings-exchange-dropdown", "value"),
        State("settings-stoploss-input", "value"),
    ],
    prevent_initial_call=True,
)
def save_exchange_stoploss(n_clicks, exchange, stoploss_pct):
    lang = get_language()
    if not exchange:
        return html.Span(
            t("settings.stoploss_select_exchange", lang),
            style={"color": COLORS["orange"], "fontSize": "0.85rem"},
        ), dash.no_update
    if stoploss_pct is None or stoploss_pct <= 0:
        return html.Span(
            t("settings.stoploss_enter_value", lang),
            style={"color": COLORS["orange"], "fontSize": "0.85rem"},
        ), dash.no_update
    try:
        pct = float(stoploss_pct)
        label = _EXCHANGE_LABELS.get(exchange, exchange)

        # Always persist to config file
        sl_map = _load_exchange_stoploss()
        sl_map[exchange] = pct
        _save_exchange_stoploss(sl_map)

        # Also push to AutoTrader if running
        try:
            from src.broker.registry import get_auto_trader
            trader = get_auto_trader()
            if trader:
                trader.set_exchange_stop_loss(exchange, pct)
        except Exception:
            pass

        status = html.Span(
            f"{t('settings.saved', lang)} — {label}: {pct:.1f}%",
            style={"color": COLORS["green"], "fontSize": "0.85rem"},
        )
        return status, _build_exchange_stoploss_table(lang)
    except Exception as e:
        return html.Span(
            f"Error: {e}",
            style={"color": COLORS["red"], "fontSize": "0.85rem"},
        ), dash.no_update


# ── Remove Exchange Stop-Loss callback ────────────────────

@callback(
    [
        Output("settings-stoploss-status", "children", allow_duplicate=True),
        Output("settings-stoploss-table", "children", allow_duplicate=True),
    ],
    Input("settings-stoploss-remove-btn", "n_clicks"),
    State("settings-stoploss-remove-dropdown", "value"),
    prevent_initial_call=True,
)
def remove_exchange_stoploss(n_clicks, exchange):
    lang = get_language()
    if not exchange:
        return html.Span(
            t("settings.stoploss_select_exchange", lang),
            style={"color": COLORS["orange"], "fontSize": "0.85rem"},
        ), dash.no_update
    try:
        label = _EXCHANGE_LABELS.get(exchange, exchange)

        # Remove from config file
        sl_map = _load_exchange_stoploss()
        sl_map.pop(exchange, None)
        _save_exchange_stoploss(sl_map)

        # Remove from AutoTrader if running
        try:
            from src.broker.registry import get_auto_trader
            trader = get_auto_trader()
            if trader:
                trader._exchange_stop_loss.pop(exchange, None)
        except Exception:
            pass

        status = html.Span(
            f"{t('settings.saved', lang)} — {label} {t('settings.stoploss_removed', lang)}",
            style={"color": COLORS["green"], "fontSize": "0.85rem"},
        )
        return status, _build_exchange_stoploss_table(lang)
    except Exception as e:
        return html.Span(
            f"Error: {e}",
            style={"color": COLORS["red"], "fontSize": "0.85rem"},
        ), dash.no_update


# ── Currency settings callbacks ────────────────────────────

@callback(
    [
        Output("settings-currency-status", "children"),
        Output("settings-currency-rate", "children"),
        Output("currency-store", "data"),
    ],
    Input("settings-currency-save-btn", "n_clicks"),
    State("settings-currency-select", "value"),
    prevent_initial_call=True,
)
def save_display_currency(n_clicks, currency_code):
    lang = get_language()
    if not currency_code:
        return dash.no_update, dash.no_update, dash.no_update
    try:
        from src.dashboard.currency_service import (
            set_display_currency, get_fx_rates, refresh_rates, get_currency_symbol,
        )
        set_display_currency(currency_code)
        refresh_rates()
        rates = get_fx_rates()
        sym = get_currency_symbol()
        rate_info = f"1 USD = {rates['usd_dkk']:.2f} DKK | 1 USD = {rates['usd_eur']:.4f} EUR | 1 EUR = {rates['eur_dkk']:.2f} DKK"
        return (
            html.Span(
                f"{t('settings.currency_saved', lang)} — {currency_code} ({sym})",
                style={"color": COLORS["green"], "fontSize": "0.85rem"},
            ),
            html.Span(
                f"{t('settings.currency_rate', lang)}: {rate_info}",
                style={"fontSize": "0.8rem"},
            ),
            currency_code,
        )
    except Exception as e:
        return (
            html.Span(f"Error: {e}", style={"color": COLORS["red"], "fontSize": "0.85rem"}),
            dash.no_update,
            dash.no_update,
        )


@callback(
    Output("currency-store", "data", allow_duplicate=True),
    Input("fx-refresh", "n_intervals"),
    prevent_initial_call=True,
)
def refresh_fx_rates(_n):
    """Refresh FX rates every 30 minutes in background."""
    try:
        from src.dashboard.currency_service import refresh_rates, get_display_currency
        refresh_rates()
        return get_display_currency()  # trigger re-render
    except Exception:
        return dash.no_update


# ── Smart Money button callback ────────────────────────────

@callback(
    [
        Output("insider-sentiment-content", "children"),
        Output("short-interest-content", "children"),
        Output("insider-trades-table", "children"),
        Output("institutional-holdings-content", "children"),
        Output("smart-money-assessment", "children"),
    ],
    Input("smart-money-btn", "n_clicks"),
    Input("smart-money-symbol", "value"),
    State("smart-money-symbol", "value"),
    prevent_initial_call=True,
)
def analyze_smart_money(_clicks, _symbol_change, symbol):
    if not symbol:
        return (dash.no_update,) * 5
    try:
        from src.data.insider_tracking import InsiderTracker
        tracker = InsiderTracker()
        report = tracker.get_smart_money_report(symbol)

        # ── 1. Insider sentiment ──
        sentiment = report.insider_sentiment
        if sentiment and sentiment.trades:
            _buy_label = t('common.buy')
            _sell_label = t('common.sell')
            trade_items = []
            for tr in sentiment.trades[:10]:
                color = COLORS["green"] if tr.is_purchase else COLORS["red"]
                trade_items.append(html.Div([
                    html.Span(f"{tr.filed_date}  ", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                    html.Strong(tr.insider_name, style={"color": COLORS["text"]}),
                    html.Span(f"  {tr.insider_title}", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                    html.Span(
                        f"  {_buy_label if tr.is_purchase else _sell_label} {tr.shares:,.0f} @ {format_value(tr.price_per_share, 2)}",
                        style={"color": color, "fontWeight": "bold"},
                    ),
                ], style={"padding": "4px 0", "borderBottom": f"1px solid {COLORS['border']}"}))
            score_color = COLORS["green"] if sentiment.score >= 60 else (
                COLORS["red"] if sentiment.score <= 40 else COLORS["orange"]
            )
            insider_content = html.Div([
                html.Div([
                    html.Span("Sentiment Score: ", style={"color": COLORS["muted"]}),
                    html.Span(f"{sentiment.score}/100", style={"color": score_color, "fontWeight": "bold", "fontSize": "1.2rem"}),
                    html.Span(f"  ({sentiment.sentiment.value})", style={"color": score_color}),
                ], className="mb-2"),
                html.Div(trade_items),
            ])
        else:
            insider_content = html.P(f"{t('smart_money.no_insider_trades')} {symbol}", className="text-muted")

        # ── 2. Short interest ──
        short_data = report.short_interest
        if short_data:
            si_color = COLORS["red"] if short_data.is_heavily_shorted else COLORS["green"]
            short_content = html.Div([
                html.Div([
                    html.Span("Short Interest: ", style={"color": COLORS["muted"]}),
                    html.Span(f"{short_data.short_pct_float:.1f}%", style={"color": si_color, "fontWeight": "bold", "fontSize": "1.2rem"}),
                ]),
                html.Div([
                    html.Span("Short Volume: ", style={"color": COLORS["muted"]}),
                    html.Span(f"{short_data.short_volume:,.0f}", style={"color": COLORS["text"]}),
                ]),
                html.Div([
                    html.Span("Days to Cover: ", style={"color": COLORS["muted"]}),
                    html.Span(f"{short_data.days_to_cover:.1f}", style={"color": COLORS["text"]}),
                ]),
                html.Div([
                    html.Span("Squeeze Potential: ", style={"color": COLORS["muted"]}),
                    html.Span(f"{short_data.squeeze_potential:.0f}%", style={
                        "color": COLORS["orange"] if short_data.squeeze_potential > 50 else COLORS["muted"],
                    }),
                ]),
            ])
        else:
            short_content = html.P(f"{t('smart_money.no_short_data')} {symbol}", className="text-muted")

        # ── 3. Insider trades table ──
        trades_table_content = html.P(f"Ingen insider-handler fundet for {symbol}", className="text-muted")
        if sentiment and sentiment.trades:
            rows = []
            for tr in sentiment.trades[:20]:
                side_color = COLORS["green"] if tr.is_purchase else COLORS["red"]
                rows.append(html.Tr([
                    html.Td(tr.filed_date, style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
                    html.Td(tr.insider_name, style={"color": COLORS["text"]}),
                    html.Td(tr.insider_title or "", style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
                    html.Td("KØB" if tr.is_purchase else "SÆLG", style={"color": side_color, "fontWeight": "bold"}),
                    html.Td(f"{tr.shares:,.0f}", style={"color": COLORS["text"]}),
                    html.Td(format_value(tr.price_per_share, 2), style={"color": COLORS["text"]}),
                    html.Td(format_value(tr.shares * tr.price_per_share), style={"color": COLORS["accent"]}),
                ]))
            trades_table_content = dbc.Table([
                html.Thead(html.Tr([
                    html.Th(h, style={"color": COLORS["accent"]})
                    for h in ["Dato", "Insider", "Titel", "Side", "Antal", "Pris", "Værdi"]
                ])),
                html.Tbody(rows),
            ], bordered=True, hover=True, responsive=True,
               style={"backgroundColor": COLORS["card"]}, className="text-light")

        # ── 4. Institutional holdings ──
        inst_content = html.P("Ingen institutionelle data tilgængelige", className="text-muted")
        institutional = getattr(report, "institutional_holdings", None)
        if institutional and hasattr(institutional, "holders") and institutional.holders:
            inst_rows = []
            for h in institutional.holders[:15]:
                change_color = COLORS["green"] if getattr(h, "change_pct", 0) > 0 else COLORS["red"]
                inst_rows.append(html.Tr([
                    html.Td(getattr(h, "name", ""), style={"color": COLORS["text"]}),
                    html.Td(f"{getattr(h, 'shares', 0):,.0f}", style={"color": COLORS["text"]}),
                    html.Td(format_value(getattr(h, "value", 0)), style={"color": COLORS["accent"]}),
                    html.Td(f"{getattr(h, 'pct_held', 0):.2f}%", style={"color": COLORS["text"]}),
                    html.Td(f"{getattr(h, 'change_pct', 0):+.1f}%", style={"color": change_color}),
                ]))
            inst_content = dbc.Table([
                html.Thead(html.Tr([
                    html.Th(h, style={"color": COLORS["accent"]})
                    for h in ["Fond/Institution", "Aktier", "Værdi", "% Holdt", "Ændring"]
                ])),
                html.Tbody(inst_rows),
            ], bordered=True, hover=True, responsive=True,
               style={"backgroundColor": COLORS["card"]}, className="text-light")
        elif institutional and hasattr(institutional, "total_institutional_pct"):
            inst_content = html.Div([
                html.Div([
                    html.Span("Institutionel ejerskab: ", style={"color": COLORS["muted"]}),
                    html.Span(f"{institutional.total_institutional_pct:.1f}%",
                              style={"color": COLORS["accent"], "fontWeight": "bold", "fontSize": "1.1rem"}),
                ]),
            ])

        # ── 5. Overall assessment ──
        assessment = getattr(report, "assessment", None) or getattr(report, "overall_signal", None)
        if assessment and isinstance(assessment, str):
            assess_content = html.P(assessment, style={"color": COLORS["text"], "whiteSpace": "pre-line"})
        elif assessment:
            signal = getattr(assessment, "signal", "NEUTRAL")
            reason = getattr(assessment, "reason", "")
            sig_color = COLORS["green"] if "BUY" in str(signal).upper() else (
                COLORS["red"] if "SELL" in str(signal).upper() else COLORS["orange"]
            )
            assess_content = html.Div([
                html.Div([
                    html.Span("Signal: ", style={"color": COLORS["muted"]}),
                    html.Span(str(signal), style={"color": sig_color, "fontWeight": "bold", "fontSize": "1.2rem"}),
                ], className="mb-2"),
                html.P(reason, style={"color": COLORS["text"]}),
            ])
        else:
            # Build a summary from what we have
            parts = []
            if sentiment:
                parts.append(f"Insider sentiment: {sentiment.score}/100 ({sentiment.sentiment.value})")
            if short_data:
                parts.append(f"Short interest: {short_data.short_pct_float:.1f}%")
                if short_data.is_heavily_shorted:
                    parts.append("Heavily shorted — potentiel short squeeze")
            assess_content = html.Div([
                html.P(f"Analyse for {symbol}:", style={"color": COLORS["accent"], "fontWeight": "bold"}),
                html.Ul([html.Li(p, style={"color": COLORS["text"]}) for p in parts])
                if parts else html.P("Ikke nok data til samlet vurdering", className="text-muted"),
            ])

        return insider_content, short_content, trades_table_content, inst_content, assess_content
    except Exception as e:
        logger.debug(f"Smart money analyse fejl: {e}")
        err = html.P(f"Kunne ikke hente data for {symbol}: {e}", className="text-danger small")
        return err, err, err, err, err


# ── Options Flow button callback ──────────────────────────

@callback(
    [
        Output("options-uoa-content", "children"),
        Output("options-iv-content", "children"),
        Output("options-pcr-content", "children"),
        Output("options-maxpain-content", "children"),
        Output("options-assessment", "children"),
    ],
    Input("options-flow-btn", "n_clicks"),
    Input("options-flow-symbol", "value"),
    State("options-flow-symbol", "value"),
    prevent_initial_call=True,
)
def analyze_options_flow(_clicks, _sym_change, symbol):
    if not symbol:
        return (dash.no_update,) * 5
    try:
        from src.data.options_flow import OptionsFlowTracker
        tracker = OptionsFlowTracker()
        summary = tracker.get_options_flow_summary(symbol)

        # UOA
        if summary.unusual_options:
            uoa_items = []
            for opt in summary.unusual_options[:8]:
                color = COLORS["green"] if opt.option_type == "CALL" else COLORS["red"]
                uoa_items.append(html.Div([
                    dbc.Badge(opt.option_type, color="success" if opt.option_type == "CALL" else "danger", className="me-2"),
                    html.Span(f"${opt.strike:.0f} ", style={"color": COLORS["text"]}),
                    html.Span(f"exp {opt.expiration} ", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                    html.Span(f"vol {opt.volume:,} ", style={"color": COLORS["orange"]}),
                    html.Span(f"(OI: {opt.open_interest:,})", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                ], style={"padding": "4px 0", "borderBottom": f"1px solid {COLORS['border']}"}))
            uoa_content = html.Div(uoa_items)
        else:
            uoa_content = html.P(t('options.no_unusual'), className="text-muted")

        # IV
        iv = summary.iv_analysis
        if iv:
            iv_content = html.Div([
                html.Div([
                    html.Span("IV Rank: ", style={"color": COLORS["muted"]}),
                    html.Span(f"{iv.iv_rank:.0f}%", style={"color": COLORS["orange"] if iv.iv_rank > 50 else COLORS["text"], "fontWeight": "bold", "fontSize": "1.2rem"}),
                ]),
                html.Div([
                    html.Span("Current IV: ", style={"color": COLORS["muted"]}),
                    html.Span(f"{iv.current_iv:.1%}", style={"color": COLORS["text"]}),
                ]),
                html.Div([
                    html.Span("IV Percentile: ", style={"color": COLORS["muted"]}),
                    html.Span(f"{iv.iv_percentile:.0f}%", style={"color": COLORS["text"]}),
                ]),
            ])
            if iv.alert_text:
                iv_content.children.append(
                    dbc.Alert(iv.alert_text, color="warning", className="mt-2 mb-0 py-1 small")
                )
        else:
            iv_content = html.P(t('options.iv_unavailable'), className="text-muted")

        # PCR
        pcr = summary.put_call_ratio
        if pcr:
            pcr_content = html.Div([
                html.Div([
                    html.Span("Put/Call Ratio: ", style={"color": COLORS["muted"]}),
                    html.Span(f"{pcr.ratio:.2f}", style={"color": COLORS["text"], "fontWeight": "bold", "fontSize": "1.2rem"}),
                ]),
                html.Div([
                    html.Span(f"{t('options.interpretation')}: ", style={"color": COLORS["muted"]}),
                    html.Span(pcr.interpretation, style={"color": COLORS["text"]}),
                ]),
                html.Div([
                    html.Span(f"Put vol: {pcr.put_volume:,} · Call vol: {pcr.call_volume:,}",
                              style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
                ]),
            ])
        else:
            pcr_content = html.P(t('options.pcr_unavailable'), className="text-muted")

        # Max Pain
        mp = summary.max_pain
        if mp:
            mp_content = html.Div([
                html.Div([
                    html.Span("Max Pain: ", style={"color": COLORS["muted"]}),
                    html.Span(f"${mp.max_pain_price:,.2f}", style={"color": COLORS["accent"], "fontWeight": "bold", "fontSize": "1.2rem"}),
                ]),
                html.Div([
                    html.Span("Current: ", style={"color": COLORS["muted"]}),
                    html.Span(f"${mp.current_price:,.2f}", style={"color": COLORS["text"]}),
                    html.Span(f"  ({mp.direction})", style={"color": COLORS["green"] if "above" in mp.direction.lower() else COLORS["red"]}),
                ]),
            ])
        else:
            mp_content = html.P(t('options.maxpain_unavailable'), className="text-muted")

        # Overall assessment
        parts = []
        if summary.unusual_options:
            call_count = sum(1 for o in summary.unusual_options if o.option_type == "CALL")
            put_count = len(summary.unusual_options) - call_count
            bias = "Bullish" if call_count > put_count else "Bearish" if put_count > call_count else "Neutral"
            parts.append(f"Unusual activity: {len(summary.unusual_options)} trades ({call_count} calls, {put_count} puts) → {bias}")
        if iv:
            parts.append(f"IV Rank: {iv.iv_rank:.0f}% — {'High (premium selling)' if iv.iv_rank > 70 else 'Low (cheap options)' if iv.iv_rank < 30 else 'Normal'}")
        if pcr:
            parts.append(f"Put/Call Ratio: {pcr.ratio:.2f} — {pcr.interpretation}")
        if mp:
            parts.append(f"Max Pain: {format_value(mp.max_pain_price, 2)} (current: {format_value(mp.current_price, 2)}, {mp.direction})")

        if parts:
            assess_content = html.Div([
                html.P(f"Options analyse for {symbol}:", style={"color": COLORS["accent"], "fontWeight": "bold"}),
                html.Ul([html.Li(p, style={"color": COLORS["text"]}) for p in parts]),
            ])
        else:
            assess_content = html.P("Ikke nok data til samlet vurdering", className="text-muted")

        return uoa_content, iv_content, pcr_content, mp_content, assess_content
    except Exception as e:
        logger.debug(f"Options flow analyse fejl: {e}")
        err = html.P(f"Kunne ikke hente options data for {symbol}: {e}", className="text-danger small")
        return err, err, err, err, err


# ── Alt Data button callback ──────────────────────────────

@callback(
    [
        Output("alt-trends-content", "children"),
        Output("alt-jobs-content", "children"),
        Output("alt-patents-content", "children"),
        Output("alt-github-content", "children"),
        Output("alt-score-content", "children"),
    ],
    Input("alt-data-btn", "n_clicks"),
    Input("alt-data-symbol", "value"),
    State("alt-data-symbol", "value"),
    prevent_initial_call=True,
)
def analyze_alt_data(_clicks, _sym_change, symbol):
    if not symbol:
        return (dash.no_update,) * 5
    try:
        from src.data.alternative_data import AlternativeDataTracker
        tracker = AlternativeDataTracker()
        score = tracker.calculate_alt_data_score(symbol)

        def _score_badge(val, label):
            color = "success" if val >= 60 else ("warning" if val >= 40 else "danger")
            return html.Div([
                html.Span(f"{label}: ", style={"color": COLORS["muted"]}),
                dbc.Badge(f"{val:.0f}/100", color=color),
            ])

        # Trends
        trends = score.google_trends
        if trends:
            trends_content = html.Div([
                _score_badge(trends.score, "Trend Score"),
                html.Div([
                    html.Span(f"Trend: {trends.trend.value}", style={"color": COLORS["text"]}),
                ], className="mt-1"),
                html.Div([
                    html.Span(f"{', '.join(trends.search_terms[:3])}", style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
                ], className="mt-1"),
            ])
        else:
            trends_content = html.P(t('alt_data.no_trends'), className="text-muted")

        # Jobs
        jobs = score.job_postings
        if jobs:
            jobs_content = html.Div([
                _score_badge(jobs.score, "Job Score"),
                html.Div([
                    html.Span(f"Aktive opslag: {jobs.current_postings:,}", style={"color": COLORS["text"]}),
                ], className="mt-1"),
            ])
        else:
            jobs_content = html.P(t('alt_data.no_jobs'), className="text-muted")

        # Patents
        patents = score.patents
        if patents:
            patents_content = html.Div([
                _score_badge(patents.score, "Patent Score"),
                html.Div([
                    html.Span(f"Seneste patenter: {patents.recent_count}", style={"color": COLORS["text"]}),
                ], className="mt-1"),
            ])
        else:
            patents_content = html.P(t('alt_data.no_patents'), className="text-muted")

        # GitHub
        github = score.github
        if github:
            github_content = html.Div([
                _score_badge(github.score, "GitHub Score"),
                html.Div([
                    html.Span(f"Stars: {github.stars:,} · Forks: {github.forks:,}", style={"color": COLORS["text"]}),
                ], className="mt-1"),
                html.Div([
                    html.Span(f"Commits (30d): {github.recent_commits}", style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
                ], className="mt-1"),
            ])
        else:
            github_content = html.P(t('alt_data.no_github'), className="text-muted")

        # Overall score
        overall_color = COLORS["green"] if score.overall_score >= 60 else (
            COLORS["orange"] if score.overall_score >= 40 else COLORS["red"]
        )
        score_content = html.Div([
            html.Div([
                html.Span(f"{t('alt_data.overall_score_label')}: ", style={"color": COLORS["muted"]}),
                html.Span(f"{score.overall_score:.0f}/100", style={
                    "color": overall_color, "fontWeight": "bold", "fontSize": "1.5rem",
                }),
            ]),
            html.Div([
                html.Span(f"Signal: {score.signal.value}", style={"color": overall_color}),
                html.Span(f"  ·  Confidence adjustment: {score.confidence_adjustment:+d}",
                          style={"color": COLORS["muted"], "fontSize": "0.85rem"}),
            ], className="mt-1"),
        ])

        return trends_content, jobs_content, patents_content, github_content, score_content
    except Exception as e:
        logger.debug(f"Alt data analyse fejl: {e}")
        err = html.P(f"Kunne ikke hente alt data for {symbol}: {e}", className="text-danger small")
        return err, err, err, err, err


# ── TA portfolio symbol → main symbol sync ────────────────

@callback(
    Output("ta-symbol", "value"),
    Input("ta-portfolio-symbol", "value"),
    prevent_initial_call=True,
)
def sync_portfolio_to_main(portfolio_sym):
    if portfolio_sym:
        return portfolio_sym
    return dash.no_update


# ── TA symbol info callback ───────────────────────────────

@callback(
    Output("ta-symbol-info", "children"),
    Input("ta-symbol", "value"),
)
def update_ta_symbol_info(symbol):
    if not symbol:
        return ""
    try:
        import yfinance as yf
        t_info = yf.Ticker(symbol)
        info = t_info.info or {}
        name = info.get("shortName") or info.get("longName") or symbol
        exchange = info.get("exchange", "")
        currency = info.get("currency", "")
        sector = info.get("sector", "")
        parts = [name]
        if exchange:
            parts.append(exchange)
        if currency:
            parts.append(currency)
        if sector:
            parts.append(sector)
        return html.Div([
            html.Span(name, style={"color": COLORS["accent"], "fontWeight": "bold", "fontSize": "0.85rem"}),
            html.Br(),
            html.Span(
                " · ".join([p for p in [exchange, currency, sector] if p]),
                style={"color": COLORS["muted"], "fontSize": "0.75rem"},
            ),
        ])
    except Exception:
        return html.Span(symbol, style={"color": COLORS["text"], "fontSize": "0.85rem"})


# ── TA Scan callback ─────────────────────────────────────

@callback(
    [
        Output("ta-overall-signal", "children"),
        Output("ta-mtf-signal", "children"),
        Output("ta-chart-patterns", "children"),
        Output("ta-candle-patterns", "children"),
        Output("ta-sr-levels", "children"),
        Output("ta-breakouts-div", "children"),
        Output("ta-seasonal", "children"),
    ],
    Input("ta-symbol", "value"),
)
def scan_ta_patterns(symbol):
    if not symbol:
        return (dash.no_update,) * 7
    try:
        from src.strategy.patterns import PatternScanner
        scanner = PatternScanner()

        df = _get_stock_data(symbol)
        if df.empty:
            err = html.P(f"{t('analysis.no_data_for')} {symbol}", className="text-danger")
            return err, err, err, err, err, err, err

        result = scanner.scan(df, symbol=symbol)

        # Overall signal
        sig_str = result.overall_signal.value if hasattr(result.overall_signal, "value") else str(result.overall_signal)
        sig_color = COLORS["green"] if "BUY" in sig_str.upper() else (
            COLORS["red"] if "SELL" in sig_str.upper() else COLORS["muted"]
        )
        # Count bullish vs bearish signals across all pattern types
        n_bull_chart = sum(1 for p in result.chart_patterns if p.direction.value == "bullish")
        n_bear_chart = sum(1 for p in result.chart_patterns if p.direction.value == "bearish")
        n_bull_candle = sum(1 for p in result.candlestick_patterns if p.direction.value == "bullish")
        n_bear_candle = sum(1 for p in result.candlestick_patterns if p.direction.value == "bearish")
        n_breakouts_up = sum(1 for b in result.breakouts if b.direction == "up")
        n_breakouts_down = sum(1 for b in result.breakouts if b.direction == "down")
        n_div_bull = sum(1 for d in result.divergences if d.divergence_type == "bullish")
        n_div_bear = sum(1 for d in result.divergences if d.divergence_type == "bearish")
        total_bull = n_bull_chart + n_bull_candle + n_breakouts_up + n_div_bull
        total_bear = n_bear_chart + n_bear_candle + n_breakouts_down + n_div_bear

        # Explain why HOLD if signals are mixed
        if sig_str == "HOLD" and (total_bull > 0 or total_bear > 0):
            explain = t("technical.mixed_signals")
        else:
            explain = ""

        overall = html.Div([
            html.Span(sig_str, className="display-4 fw-bold", style={"color": sig_color}),
            html.Div([
                html.Span(f"{t('common.confidence')}: {result.overall_confidence:.0f}%", style={"color": COLORS["muted"]}),
            ]),
            html.Div([
                html.Span(f"{total_bull} ", style={"color": COLORS["green"], "fontWeight": "bold"}),
                html.Span("bullish · ", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                html.Span(f"{total_bear} ", style={"color": COLORS["red"], "fontWeight": "bold"}),
                html.Span("bearish", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
            ], className="mt-1"),
            html.P(explain, style={"color": COLORS["orange"], "fontSize": "0.75rem"}, className="mt-1 mb-0") if explain else None,
        ])

        # Multi-timeframe
        mtf = result.multi_timeframe
        if mtf:
            mtf_items = []
            for tf_sig in mtf.signals:
                tf_val = tf_sig.signal.value if hasattr(tf_sig.signal, "value") else str(tf_sig.signal)
                tf_color = COLORS["green"] if "BUY" in tf_val.upper() else (
                    COLORS["red"] if "SELL" in tf_val.upper() else COLORS["muted"]
                )
                mtf_items.append(html.Div([
                    html.Span(f"{tf_sig.timeframe}: ", style={"color": COLORS["muted"]}),
                    html.Span(tf_val, style={"color": tf_color, "fontWeight": "bold"}),
                    html.Span(f" ({tf_sig.confidence:.0f}%)", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                ]))
            consensus_val = mtf.consensus.value if hasattr(mtf.consensus, "value") else str(mtf.consensus)
            aligned_icon = " ✓" if mtf.aligned else ""
            mtf_items.append(html.Hr(style={"borderColor": COLORS["border"], "margin": "4px 0"}))
            mtf_items.append(html.Div([
                html.Span(f"Consensus: ", style={"color": COLORS["muted"]}),
                html.Span(f"{consensus_val}{aligned_icon}", style={
                    "color": COLORS["green"] if "BUY" in consensus_val.upper() else (
                        COLORS["red"] if "SELL" in consensus_val.upper() else COLORS["muted"]
                    ),
                    "fontWeight": "bold",
                }),
            ]))
            mtf_content = html.Div(mtf_items)
        else:
            mtf_content = html.P(t("technical.mtf_unavailable"), className="text-muted small")

        # Chart patterns — deduplicate: keep highest confidence per pattern_type+direction
        if result.chart_patterns:
            best = {}
            for p in result.chart_patterns:
                key = (p.pattern_type.value, p.direction.value)
                if key not in best or p.confidence > best[key].confidence:
                    best[key] = p
            unique_patterns = sorted(best.values(), key=lambda x: x.confidence, reverse=True)

            cp_items = []
            for p in unique_patterns[:8]:
                dir_val = p.direction.value
                cp_items.append(html.Div([
                    dbc.Badge(dir_val, color="success" if dir_val == "bullish" else ("danger" if dir_val == "bearish" else "secondary"), className="me-2"),
                    html.Span(p.pattern_type.value.replace("_", " ").title(), style={"color": COLORS["text"]}),
                    html.Span(f"  ({t('technical.conf')}: {p.confidence:.0f}%)", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                    html.Span("  ✓ vol", style={"color": COLORS["accent"], "fontSize": "0.8rem"}) if p.volume_confirmed else None,
                ], style={"padding": "3px 0"}))
            chart_content = html.Div(cp_items)
        else:
            chart_content = html.P(t("technical.no_chart_patterns"), className="text-muted")

        # Candlestick patterns — deduplicate same way
        if result.candlestick_patterns:
            best_c = {}
            for p in result.candlestick_patterns:
                key = (p.pattern_type.value, p.direction.value)
                if key not in best_c or p.confidence > best_c[key].confidence:
                    best_c[key] = p
            unique_candles = sorted(best_c.values(), key=lambda x: x.confidence, reverse=True)

            cd_items = []
            for p in unique_candles[:6]:
                dir_val = p.direction.value
                cd_items.append(html.Div([
                    dbc.Badge(dir_val, color="success" if dir_val == "bullish" else ("danger" if dir_val == "bearish" else "secondary"), className="me-2"),
                    html.Span(p.pattern_type.value.replace("_", " ").title(), style={"color": COLORS["text"]}),
                ], style={"padding": "3px 0"}))
            candle_content = html.Div(cd_items)
        else:
            candle_content = html.P(t("technical.no_candlestick"), className="text-muted")

        # S/R levels — merge nearby levels, sort by strength, show top 5
        if result.support_resistance:
            current_price = float(df["Close"].iloc[-1])

            # Group nearby levels (within 2%) — keep the strongest
            merged = []
            used = set()
            sorted_levels = sorted(result.support_resistance, key=lambda l: l.strength, reverse=True)
            for level in sorted_levels:
                if id(level) in used:
                    continue
                # Find nearby levels and merge
                total_strength = level.strength
                for other in sorted_levels:
                    if id(other) != id(level) and id(other) not in used:
                        if abs(other.price - level.price) / level.price < 0.02:
                            total_strength += other.strength
                            used.add(id(other))
                used.add(id(level))
                merged.append((level, total_strength))

            # Sort: supports below price, resistances above price, strongest first
            supports = sorted(
                [(l, s) for l, s in merged if l.level_type == "support"],
                key=lambda x: x[1], reverse=True,
            )[:3]
            resistances = sorted(
                [(l, s) for l, s in merged if l.level_type == "resistance"],
                key=lambda x: x[1], reverse=True,
            )[:3]

            sr_items = []
            # Current price reference
            sr_items.append(html.Div([
                html.Span(f"  ${current_price:.2f}", style={"color": COLORS["accent"], "fontWeight": "bold"}),
                html.Span(f"  ← current", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
            ], style={"padding": "4px 0", "borderBottom": f"1px solid {COLORS['accent']}"}))

            for level, strength in resistances:
                pct_away = (level.price - current_price) / current_price * 100
                sr_items.insert(0, html.Div([
                    dbc.Badge("RESISTANCE", color="danger", className="me-2"),
                    html.Span(f"${level.price:.2f}", style={"color": COLORS["text"], "fontWeight": "bold"}),
                    html.Span(f"  ({t('technical.touches')}: {strength})", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                    html.Span(f"  +{pct_away:.1f}%", style={"color": COLORS["red"], "fontSize": "0.8rem"}),
                    html.Span(f"  {t('technical.strong')}", style={"color": COLORS["accent"]}) if strength >= 3 else None,
                ], style={"padding": "3px 0"}))

            for level, strength in supports:
                pct_away = (current_price - level.price) / current_price * 100
                sr_items.append(html.Div([
                    dbc.Badge("SUPPORT", color="success", className="me-2"),
                    html.Span(f"${level.price:.2f}", style={"color": COLORS["text"], "fontWeight": "bold"}),
                    html.Span(f"  ({t('technical.touches')}: {strength})", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                    html.Span(f"  -{pct_away:.1f}%", style={"color": COLORS["green"], "fontSize": "0.8rem"}),
                    html.Span(f"  {t('technical.strong')}", style={"color": COLORS["accent"]}) if strength >= 3 else None,
                ], style={"padding": "3px 0"}))

            # Add context line about nearest S/R
            nearest_res = resistances[0] if resistances else None
            nearest_sup = supports[0] if supports else None
            if nearest_res and nearest_sup:
                res_dist = (nearest_res[0].price - current_price) / current_price * 100
                sup_dist = (current_price - nearest_sup[0].price) / current_price * 100
                if sup_dist < res_dist:
                    sr_note = t("technical.sr_closer_support")
                else:
                    sr_note = t("technical.sr_closer_resistance")
            elif nearest_sup and not nearest_res:
                sr_note = t("technical.sr_no_resistance")
            elif nearest_res and not nearest_sup:
                sr_note = t("technical.sr_no_support")
            else:
                sr_note = ""

            if sr_note:
                sr_items.append(html.P(
                    sr_note,
                    style={"color": COLORS["orange"], "fontSize": "0.75rem"},
                    className="mt-2 mb-0",
                ))

            sr_content = html.Div(sr_items)
        else:
            sr_content = html.P(t("technical.no_sr_levels"), className="text-muted")

        # Breakouts — field is "breakout_price" and "volume_ratio"
        if result.breakouts:
            bo_items = []
            for b in result.breakouts[:4]:
                bo_items.append(html.Div([
                    dbc.Badge(b.direction.upper(), color="success" if b.direction == "up" else "danger", className="me-2"),
                    html.Span(f"${b.breakout_price:.2f}", style={"color": COLORS["text"]}),
                    html.Span(f"  {t('technical.vol_conf')}: {b.volume_ratio:.1f}x", style={"color": COLORS["muted"], "fontSize": "0.8rem"}),
                    html.Span(f"  {b.description}", style={"color": COLORS["muted"], "fontSize": "0.8rem"}) if b.description else None,
                ], style={"padding": "3px 0"}))
            breakout_content = html.Div(bo_items)
        else:
            breakout_content = html.P(t("technical.no_breakouts"), className="text-muted")

        # Seasonal — uses best_period/worst_period/sell_in_may_effect/santa_rally_avg/january_effect
        seasonal = result.seasonal
        if seasonal:
            items = []
            items.append(html.Div([
                html.Span(f"Best: ", style={"color": COLORS["muted"]}),
                html.Span(seasonal.best_period, style={"color": COLORS["green"], "fontWeight": "bold"}),
                html.Span(f"  |  Worst: ", style={"color": COLORS["muted"]}),
                html.Span(seasonal.worst_period, style={"color": COLORS["red"], "fontWeight": "bold"}),
            ]))
            if seasonal.sell_in_may_effect is not None:
                items.append(html.Div([
                    html.Span("Sell in May: ", style={"color": COLORS["muted"]}),
                    html.Span(f"{seasonal.sell_in_may_effect:+.1f}%", style={
                        "color": COLORS["red"] if seasonal.sell_in_may_effect < 0 else COLORS["green"],
                    }),
                ]))
            if seasonal.santa_rally_avg is not None:
                items.append(html.Div([
                    html.Span("Santa Rally: ", style={"color": COLORS["muted"]}),
                    html.Span(f"{seasonal.santa_rally_avg:+.1f}%", style={
                        "color": COLORS["green"] if seasonal.santa_rally_avg > 0 else COLORS["red"],
                    }),
                ]))
            if seasonal.january_effect is not None:
                items.append(html.Div([
                    html.Span("January Effect: ", style={"color": COLORS["muted"]}),
                    html.Span(f"{seasonal.january_effect:+.1f}%", style={
                        "color": COLORS["green"] if seasonal.january_effect > 0 else COLORS["red"],
                    }),
                ]))
            # Monthly breakdown
            if seasonal.data:
                month_items = []
                for month, ret in seasonal.data.items():
                    color = COLORS["green"] if ret >= 0 else COLORS["red"]
                    month_items.append(
                        html.Span(f"{month[:3]}: {ret:+.1f}%  ", style={"color": color, "fontSize": "0.8rem"})
                    )
                items.append(html.Div(month_items, className="mt-1"))
            seasonal_content = html.Div(items)
        else:
            seasonal_content = html.P(t("technical.seasonal_unavailable"), className="text-muted")

        return overall, mtf_content, chart_content, candle_content, sr_content, breakout_content, seasonal_content
    except Exception as e:
        logger.debug(f"TA scan error: {e}")
        err = html.P(f"{t('technical.scan_failed')} {symbol}: {e}", className="text-danger small")
        return err, err, err, err, err, err, err


# ── Economy/Macro callbacks ────────────────────────────────

def _build_macro_ui(report, FRED_SERIES):
    """Build all economy page UI components from a MacroReport."""
    # ── Recession probability ──
    rp = report.recession_probability
    rp_color = {"red": "text-danger", "orange": "text-warning", "green": "text-success"}.get(rp.color, "text-warning")
    recession_content = html.Div([
        html.Div([
            html.Span(f"{rp.probability:.0f}%", className=f"display-4 fw-bold {rp_color}"),
            html.Span(f"  {rp.level.upper()}", className=f"{rp_color} fw-bold",
                       style={"fontSize": "1.2rem"}),
        ]),
        html.P(t('economy.recession_desc'), className="text-muted small mt-2"),
        html.Div([
            html.Div([
                html.I(className="bi bi-exclamation-circle me-1 text-danger"),
                w,
            ], className="text-muted small mb-1")
            for w in rp.key_warnings
        ]) if rp.key_warnings else None,
        html.Div([
            html.Div([
                html.I(className="bi bi-check-circle me-1 text-success"),
                p,
            ], className="text-muted small mb-1")
            for p in rp.key_positives
        ]) if rp.key_positives else None,
    ])

    # ── Economic Surprise ──
    si = report.surprise_index
    si_color = "text-success" if si.value > 0 else "text-danger" if si.value < 0 else "text-info"
    surprise_content = html.Div([
        html.Span(f"{si.value:+.1f}", className=f"display-4 fw-bold {si_color}"),
        html.P(t('economy.surprise_desc'), className="text-muted small mt-2"),
        html.P(f"{si.beats} {t('economy.beats')} / {si.misses} {t('economy.misses')} ({si.total} total)", className="text-muted small"),
    ])

    # ── Overall Signal ──
    signal_map = {
        "expansion": ("text-success", t('economy.expansion')),
        "stable": ("text-info", t('economy.stable')),
        "slowdown": ("text-warning", t('economy.slowdown')),
        "recession_warning": ("text-danger", t('economy.recession_warning')),
    }
    sig_color, sig_label = signal_map.get(report.overall_signal.value, ("text-muted", "—"))
    signal_content = html.Div([
        html.H3(sig_label, className=f"fw-bold {sig_color}"),
        html.Div([
            dbc.Alert(a, color="warning" if "⚠" in a else "danger" if "🚨" in a else "info",
                      className="py-1 px-2 mb-1", style={"fontSize": "0.8rem"})
            for a in report.alerts
        ]) if report.alerts else None,
    ])

    # ── Heatmap with live data ──
    heatmap_rows = []
    for key, meta in FRED_SERIES.items():
        ind = report.indicators.get(key)
        desc = _economy_indicator_desc(key)
        if ind:
            sig = ind.signal
            sig_cls = "text-success" if sig == "bullish" else "text-danger" if sig == "bearish" else "text-muted"
            trend_arrow = ind.trend_arrow
            val_str = f"{ind.current_value:,.2f}"
            chg_str = f"{ind.change_pct:+.1f}%"
            chg_cls = "text-success" if ind.change_pct > 0 else "text-danger" if ind.change_pct < 0 else "text-muted"
        else:
            sig_cls = "text-muted"
            trend_arrow = "—"
            val_str = "—"
            chg_str = ""
            chg_cls = "text-muted"

        heatmap_rows.append(html.Tr([
            html.Td([
                html.Div(meta["name"], className="text-light", style={"fontSize": "0.85rem"}),
                html.Div(desc, className="text-muted", style={"fontSize": "0.7rem"}),
            ]),
            html.Td(val_str, className=sig_cls, style={"fontSize": "0.85rem", "fontWeight": "bold"}),
            html.Td(chg_str, className=chg_cls, style={"fontSize": "0.85rem"}),
            html.Td(trend_arrow, className=sig_cls, style={"fontSize": "1.1rem"}),
        ]))

    heatmap_table = dbc.Table([
        html.Thead([
            html.Tr([
                html.Th(t('economy.indicator'), className="text-light"),
                html.Th(t('economy.value'), className="text-light"),
                html.Th(t('economy.change'), className="text-light"),
                html.Th(t('economy.direction'), className="text-light"),
            ]),
        ]),
        html.Tbody(heatmap_rows),
    ], bordered=True, hover=True, size="sm", className="table-dark")

    # ── Category cards with live data ──
    category_icons = {
        "shipping": ("bi-truck", "text-info"),
        "housing": ("bi-house-door", "text-warning"),
        "energy": ("bi-lightning-charge", "text-danger"),
        "consumer": ("bi-people", "text-success"),
        "labor": ("bi-briefcase", "text-primary"),
        "recession": ("bi-exclamation-triangle", "text-danger"),
    }
    cat_cols = []
    from src.data.macro_indicators import CATEGORIES
    for cat_key in CATEGORIES:
        icon, color = category_icons.get(cat_key, ("bi-circle", "text-muted"))
        cat_label = t(f'economy.{cat_key}')
        series_in_cat = [(k, v) for k, v in FRED_SERIES.items() if v["category"] == cat_key]
        rows = []
        for k, v in series_in_cat:
            ind = report.indicators.get(k)
            if ind:
                sig_cls = "text-success" if ind.signal == "bullish" else "text-danger" if ind.signal == "bearish" else "text-muted"
                rows.append(html.Li([
                    html.Strong(v["name"], className="text-light"),
                    html.Span(f"  {ind.current_value:,.2f}", className=sig_cls, style={"fontWeight": "bold"}),
                    html.Span(f"  {ind.change_pct:+.1f}% {ind.trend_arrow}", className=sig_cls),
                    html.Div(_economy_indicator_desc(k), className="text-muted", style={"fontSize": "0.75rem"}),
                ], className="mb-2"))
            else:
                rows.append(html.Li([
                    html.Strong(v["name"], className="text-light"),
                    html.Span(f" — {t('economy.no_data')}", className="text-muted small"),
                ], className="mb-1"))

        cat_cols.append(dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5([
                        html.I(className=f"bi {icon} me-2 {color}"),
                        cat_label,
                    ], className="text-light mb-3"),
                    html.Ul(rows, className="list-unstyled mb-0"),
                ]),
            ], style={"backgroundColor": COLORS["card"]}),
        ], md=6, className="mb-4"))

    return recession_content, surprise_content, signal_content, heatmap_table, cat_cols


@callback(
    [
        Output("macro-recession-content", "children"),
        Output("macro-surprise-content", "children"),
        Output("macro-signal-content", "children"),
        Output("macro-heatmap-content", "children"),
        Output("macro-category-cards", "children"),
        Output("macro-update-status", "children"),
    ],
    Input("macro-update-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_macro_data(n_clicks):
    """Re-fetch macro data on explicit button click only."""
    if not n_clicks:
        return [dash.no_update] * 6
    try:
        import os, time as _t
        from src.data.macro_indicators import MacroIndicatorTracker, FRED_SERIES

        fred_key = settings.market_data.fred_api_key or os.environ.get("FRED_API_KEY", "")
        tracker = MacroIndicatorTracker(fred_api_key=fred_key)
        report = tracker.get_macro_report()

        # Update global cache
        _macro_report_cache["report"] = report
        _macro_report_cache["ts"] = _t.time()

        recession, surprise, signal, heatmap, cat_cols = _build_macro_ui(report, FRED_SERIES)

        status = html.Span([
            html.I(className="bi bi-check-circle me-1"),
            f"{len(report.indicators)}/{len(FRED_SERIES)}",
        ], style={"color": COLORS["green"], "fontSize": "0.8rem"})

        return recession, surprise, signal, heatmap, cat_cols, status

    except Exception as exc:
        logger.error(f"[macro] Update failed: {exc}")
        err = html.Span(f"Error: {exc}", style={"color": COLORS["red"], "fontSize": "0.8rem"})
        return [dash.no_update] * 5 + [err]


# ── Scanner loading retry ──────────────────────────────────

@callback(
    Output("scanner-reload", "href"),
    Input("scanner-poll", "n_intervals"),
    prevent_initial_call=True,
)
def scanner_retry_reload(_n):
    """After 5s, redirect to /marked to re-check if data is ready."""
    return "/marked"


# ── Sell page callbacks ────────────────────────────────────

@callback(
    Output("sell-exchange-info", "children"),
    Output("sell-exchange-btn", "disabled"),
    Input("sell-exchange-dropdown", "value"),
)
def update_exchange_info(exchange):
    if not exchange:
        return "", True
    from src.dashboard.currency_service import format_value_dkk
    groups = _get_positions_by_group()
    positions = groups["exchanges"].get(exchange, [])
    total = sum(p["value_dkk"] for p in positions)
    info = html.Div([
        html.Span(f"{len(positions)} {t('sell.positions_count')}", className="text-light fw-bold me-3"),
        html.Span(format_value_dkk(total), className="text-warning fw-bold"),
    ])
    return info, len(positions) == 0


@callback(
    Output("sell-confirm-modal", "is_open"),
    Output("sell-confirm-body", "children"),
    Output("sell-action-store", "data"),
    [
        Input("sell-all-btn", "n_clicks"),
        Input("sell-crypto-btn", "n_clicks"),
        Input("sell-stocks-btn", "n_clicks"),
        Input("sell-bonds-btn", "n_clicks"),
        Input("sell-commodities-btn", "n_clicks"),
        Input("sell-exchange-btn", "n_clicks"),
        Input("sell-cancel-btn", "n_clicks"),
        Input("sell-confirm-btn", "n_clicks"),
    ],
    [State("sell-exchange-dropdown", "value"), State("sell-action-store", "data")],
    prevent_initial_call=True,
)
def handle_sell_flow(all_c, crypto_c, stocks_c, bonds_c, commodities_c, exchange_c, cancel_c, confirm_c,
                     selected_exchange, pending_action):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    # Ignore spurious fires from component re-renders (n_clicks resets to None)
    click_map = {
        "sell-all-btn": all_c, "sell-crypto-btn": crypto_c,
        "sell-stocks-btn": stocks_c, "sell-bonds-btn": bonds_c,
        "sell-commodities-btn": commodities_c, "sell-exchange-btn": exchange_c,
        "sell-cancel-btn": cancel_c, "sell-confirm-btn": confirm_c,
    }
    if not click_map.get(trigger):
        return dash.no_update, dash.no_update, dash.no_update

    # Cancel
    if trigger == "sell-cancel-btn":
        return False, "", None

    # Confirm — execute the sell
    if trigger == "sell-confirm-btn" and pending_action:
        return False, "", pending_action  # modal closes, sell-result callback picks up

    # First-level button clicks → show confirmation
    groups = _get_positions_by_group()

    if trigger == "sell-all-btn":
        positions = groups["all"]
        action = {"type": "all", "symbols": [p["symbol"] for p in positions]}
    elif trigger == "sell-crypto-btn":
        positions = groups["crypto"]
        action = {"type": "crypto", "symbols": [p["symbol"] for p in positions]}
    elif trigger == "sell-stocks-btn":
        positions = groups["stocks"]
        action = {"type": "stocks", "symbols": [p["symbol"] for p in positions]}
    elif trigger == "sell-bonds-btn":
        positions = groups["bonds"]
        action = {"type": "bonds", "symbols": [p["symbol"] for p in positions]}
    elif trigger == "sell-commodities-btn":
        positions = groups["commodities"]
        action = {"type": "commodities", "symbols": [p["symbol"] for p in positions]}
    elif trigger == "sell-exchange-btn":
        positions = groups["exchanges"].get(selected_exchange, [])
        action = {"type": "exchange", "exchange": selected_exchange,
                  "symbols": [p["symbol"] for p in positions]}
    else:
        return dash.no_update, dash.no_update, dash.no_update

    if not positions:
        return False, "", None

    from src.dashboard.currency_service import format_value_dkk
    total = sum(p["value_dkk"] for p in positions)
    total_fmt = format_value_dkk(total)
    msg = t('sell.confirm_msg').replace("{count}", str(len(positions))).replace("{value}", total_fmt)

    # Show position list in confirmation with gain/loss
    pos_items = []
    for p in positions[:20]:
        pnl = p.get("pnl_dkk", 0)
        pnl_pct = p.get("pnl_pct", 0) * 100
        pnl_color = COLORS["green"] if pnl >= 0 else COLORS["red"]
        pnl_str = format_value_dkk(pnl, 2)
        pos_items.append(html.Li([
            html.Span(f"{p['symbol']}  \u00d7{p['qty']:.2f}  ({format_value_dkk(p['value_dkk'])})  "),
            html.Span(f"{pnl_str} ({pnl_pct:+.1f}%)",
                       style={"color": pnl_color, "fontWeight": "600"}),
        ], className="text-light", style={"fontSize": "0.9rem"}))
    pos_list = html.Ul(pos_items, className="list-unstyled mt-3")
    if len(positions) > 20:
        pos_list.children.append(html.Li(f"... +{len(positions) - 20} more", className="text-muted"))

    body = html.Div([
        html.P(msg, className="text-warning fw-bold", style={"fontSize": "1.1rem"}),
        pos_list,
    ])

    return True, body, action


@callback(
    Output("sell-result", "children"),
    Output("sell-cards-container", "children"),
    Input("sell-confirm-btn", "n_clicks"),
    State("sell-action-store", "data"),
    prevent_initial_call=True,
)
def execute_sell(n_clicks, action):
    if not n_clicks or not action:
        return dash.no_update, dash.no_update

    symbols = action.get("symbols", [])
    if not symbols:
        return dbc.Alert(t('sell.no_positions'), color="warning"), dash.no_update

    results = []
    try:
        router = None
        try:
            from src.broker.registry import get_router
            router = get_router()
        except Exception:
            pass
        if not router:
            from src.broker.paper_broker import PaperBroker
            router = PaperBroker()

        from src.broker.base_broker import OrderType
        positions = list(router.get_positions()) if hasattr(router, 'get_positions') else []
        pos_map = {getattr(p, "symbol", ""): p for p in positions}

        sold = 0
        for sym in symbols:
            pos = pos_map.get(sym)
            if not pos:
                continue
            qty = getattr(pos, "qty", 0)
            if qty <= 0:
                continue
            try:
                router.sell(symbol=sym, qty=qty, order_type=OrderType.MARKET)
                sold += 1
                results.append(html.Li(f"\u2713 {sym} \u00d7{qty:.2f}",
                                       style={"color": COLORS["text"]}))
            except Exception as exc:
                results.append(html.Li(f"\u2717 {sym}: {exc}",
                                       style={"color": COLORS["red"]}))

        msg = t('sell.sold_ok').replace("{count}", str(sold))
        alert = dbc.Alert([
            html.Strong(msg, style={"color": COLORS["text"]}),
            html.Ul(results, className="list-unstyled mt-2 mb-0"),
        ], color="dark", dismissable=True,
           style={"backgroundColor": COLORS["card"], "border": f"1px solid {COLORS['border']}"})

        # Refresh the sell cards with updated positions
        return alert, _build_sell_cards()

    except Exception as exc:
        logger.error(f"[sell] Execution failed: {exc}")
        return dbc.Alert(f"{t('sell.sold_fail')}: {exc}", color="danger", dismissable=True), dash.no_update


# ── Register multi-broker page callbacks ───────────────────
register_portfolio_callbacks(app)
register_trading_callbacks(app)
register_tax_callbacks(app)
register_status_callbacks(app)
register_market_callbacks(app)

server = app.server

# ── Auth + /healthz ───────────────────────────────────────
# HTTP Basic auth activates only when DASHBOARD_USER + DASHBOARD_PASS are
# set in the environment. /healthz is always reachable for Docker's
# HEALTHCHECK. See src/dashboard/auth.py for the threat model.
from src.dashboard.auth import install_auth  # noqa: E402 — must follow `app`
install_auth(server)

if __name__ == "__main__":
    app.run(
        host=settings.dashboard.host,
        port=settings.dashboard.port,
        debug=os.getenv("DASH_DEBUG", "false").lower() == "true",
    )
