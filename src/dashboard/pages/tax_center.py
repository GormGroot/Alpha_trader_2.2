"""
Tax Center — skatteøverblik og optimering.

Dashboard-side: /tax

Features:
  - Skattetilgodehavende gauge
  - Lagerbeskatning estimat YTD
  - Skatteoptimerings-forslag
  - Udbytteoversigt
  - Export til CSV/Excel
"""

from __future__ import annotations

from dash import dcc, html, callback, Input, Output, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime

from loguru import logger
from src.dashboard.i18n import t
from src.dashboard.currency_service import (
    format_value_dkk, get_currency_label, convert_from_dkk,
)

COLORS = {
    "bg": "#0f1117", "card": "#1a1c24", "accent": "#00d4aa",
    "red": "#ff4757", "green": "#2ed573", "blue": "#3498db",
    "orange": "#ffa502", "text": "#e2e8f0", "muted": "#64748b",
    "border": "#2d3748",
}


def _tax_kpi(title: str, value: str, subtitle: str = "", color: str = "") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.85rem"}),
            html.H3(value, className="mb-0", style={"color": color or COLORS["text"]}),
            html.Small(subtitle, style={"color": COLORS["muted"]}) if subtitle else None,
        ]),
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
        },
    )


# ── Layout ──────────────────────────────────────────────────

def tax_center_layout() -> html.Div:
    return html.Div([
        dcc.Interval(id="tax-refresh", interval=60_000, n_intervals=0),

        html.H2(t('tax.title'), style={"color": COLORS["text"]}, className="mb-2"),
        html.P(
            t('tax.subtitle'),
            style={"color": COLORS["muted"]},
            className="mb-4",
        ),

        # KPI Row
        dbc.Row(id="tax-kpi-row", className="mb-4"),

        # Main Content
        dbc.Row([
            # Skattetilgodehavende
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H5(t('tax.tax_credit'), className="mb-0"),
                        style={"backgroundColor": COLORS["card"]},
                    ),
                    dbc.CardBody([
                        dcc.Graph(
                            id="tax-credit-gauge",
                            config={"displayModeBar": False},
                            style={"backgroundColor": COLORS["card"]},
                        ),
                        html.Div(id="tax-credit-history"),
                    ], style={"backgroundColor": COLORS["card"]}),
                ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}),
            ], width=6),

            # Lagerbeskatning
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(
                        html.H5(t('tax.estimated_mtm'), className="mb-0"),
                        style={"backgroundColor": COLORS["card"]},
                    ),
                    dbc.CardBody([
                        dcc.Graph(
                            id="tax-mtm-chart",
                            config={"displayModeBar": False},
                            style={"backgroundColor": COLORS["card"]},
                        ),
                        html.Div(id="tax-mtm-details"),
                    ], style={"backgroundColor": COLORS["card"]}),
                ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"}),
            ], width=6),
        ], className="mb-4"),

        # Skatteoptimering
        dbc.Card([
            dbc.CardHeader(
                html.H5(t('tax.optimization'), className="mb-0"),
                style={"backgroundColor": COLORS["card"]},
            ),
            dbc.CardBody(
                html.Div(id="tax-optimization-list"),
                style={"backgroundColor": COLORS["card"]},
            ),
        ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
           className="mb-4"),

        # Udbytteoversigt
        dbc.Card([
            dbc.CardHeader(
                html.H5(t('tax.dividends_ytd'), className="mb-0"),
                style={"backgroundColor": COLORS["card"]},
            ),
            dbc.CardBody(
                html.Div(id="tax-dividends-table"),
                style={"backgroundColor": COLORS["card"]},
            ),
        ], style={"border": f"1px solid {COLORS['border']}", "borderRadius": "8px"},
           className="mb-4"),

        # Export
        dbc.Row([
            dbc.Col([
                dbc.Button(
                    t('tax.download_report'),
                    id="tax-export-btn",
                    color="secondary",
                    outline=True,
                    className="me-2",
                ),
                html.Span(id="tax-export-status"),
            ]),
        ]),

        # Disclaimer
        dbc.Alert(
            t('tax.disclaimer'),
            color="warning",
            className="mt-4",
            style={"fontSize": "0.85rem"},
        ),

    ], style={"padding": "20px", "backgroundColor": COLORS["bg"]})


# ── Callbacks ───────────────────────────────────────────────

def register_tax_callbacks(app: object) -> None:
    """Registrér tax center callbacks."""

    @app.callback(
        [
            Output("tax-kpi-row", "children"),
            Output("tax-credit-gauge", "figure"),
            Output("tax-optimization-list", "children"),
            Output("tax-credit-history", "children"),
            Output("tax-mtm-chart", "figure"),
            Output("tax-mtm-details", "children"),
            Output("tax-dividends-table", "children"),
        ],
        Input("tax-refresh", "n_intervals"),
    )
    def update_tax_center(_n: int) -> tuple:
        # Tax data
        credit_balance = 0.0
        ytd_estimated_tax = 0.0
        net_tax = 0.0

        # Fetch live positions from PaperBroker
        position_dicts = []
        usd_dkk = 6.90
        try:
            try:
                import yfinance as yf
                fx = yf.Ticker("DKK=X")
                rate = getattr(fx.fast_info, "last_price", None)
                if rate and rate > 0:
                    usd_dkk = rate
            except Exception:
                pass

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
            if pb is None:
                from src.broker.paper_broker import PaperBroker
                pb = PaperBroker()

            for p in pb.get_positions():
                mkt_val = getattr(p, "market_value", 0) or 0
                position_dicts.append({
                    "symbol": getattr(p, "symbol", ""),
                    "qty": getattr(p, "qty", 0),
                    "entry_price": getattr(p, "entry_price", 0),
                    "current_price": getattr(p, "current_price", 0),
                    "market_value_dkk": round(mkt_val * usd_dkk, 2),
                    "currency": "USD",
                })
        except Exception as exc:
            logger.debug(f"[tax] Could not fetch positions: {exc}")

        try:
            from src.tax.tax_credit_tracker import TaxCreditTracker
            tracker = TaxCreditTracker()
            credit_balance = tracker.balance
        except Exception:
            pass

        try:
            from src.tax.corporate_tax import CorporateTaxCalculator
            calc = CorporateTaxCalculator(tax_credit=credit_balance)
            ytd = calc.ytd_estimated_tax(position_dicts)
            ytd_estimated_tax = ytd.get("estimated_gross_tax", 0)
            net_tax = ytd.get("estimated_net_tax", 0)
        except Exception:
            pass

        # KPIs — convert to display currency
        ccy = get_currency_label()
        credit_color = COLORS["green"] if credit_balance > 0 else COLORS["red"]
        kpis = [
            dbc.Col(_tax_kpi(
                t('tax.tax_credit'),
                format_value_dkk(credit_balance),
                color=credit_color,
            ), width=3),
            dbc.Col(_tax_kpi(
                t('tax.estimated_tax_ytd'),
                format_value_dkk(ytd_estimated_tax),
                t('tax.gross'),
            ), width=3),
            dbc.Col(_tax_kpi(
                t('tax.net_tax'),
                format_value_dkk(net_tax),
                color=COLORS["green"] if net_tax == 0 else COLORS["orange"],
            ), width=3),
            dbc.Col(_tax_kpi(
                t('tax.corporate_tax_rate'),
                "22%",
                t('tax.mtm_taxation'),
            ), width=3),
        ]

        # Credit gauge
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_balance,
            number={"prefix": "", "suffix": f" {ccy}", "font": {"color": COLORS["text"]}},
            gauge={
                "axis": {"range": [0, max(credit_balance * 1.5, 100_000)]},
                "bar": {"color": COLORS["accent"]},
                "bgcolor": COLORS["card"],
                "bordercolor": COLORS["border"],
                "steps": [
                    {"range": [0, credit_balance * 0.2], "color": COLORS["red"]},
                    {"range": [credit_balance * 0.2, credit_balance * 0.6], "color": COLORS["orange"]},
                    {"range": [credit_balance * 0.6, max(credit_balance * 1.5, 100_000)], "color": "#1a3a2a"},
                ],
            },
            title={"text": t('tax.remaining_credit'), "font": {"color": COLORS["text"]}},
        ))
        gauge.update_layout(
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["card"],
            height=250,
            margin=dict(l=20, r=20, t=60, b=20),
            transition=dict(duration=300),
        )

        # Optimization suggestions
        suggestions = []
        try:
            from src.tax.corporate_tax import CorporateTaxCalculator
            calc = CorporateTaxCalculator(tax_credit=credit_balance)
            suggestions_raw = calc.suggest_tax_optimization(position_dicts)
            for s in suggestions_raw[:5]:
                urgency_color = {
                    "high": COLORS["red"],
                    "medium": COLORS["orange"],
                    "low": COLORS["muted"],
                }.get(s.urgency, COLORS["muted"])

                suggestions.append(dbc.ListGroupItem([
                    dbc.Badge(s.urgency.upper(), color="danger" if s.urgency == "high" else "warning",
                              className="me-2"),
                    html.Strong(s.symbol, style={"color": COLORS["accent"]}),
                    html.Span(f" — {s.description}", style={"color": COLORS["text"]}),
                    html.Span(
                        f" ({s.estimated_impact_dkk:+,.0f} DKK)",
                        style={"color": COLORS["green"], "fontWeight": "bold"},
                    ) if s.estimated_impact_dkk else None,
                ], style={
                    "backgroundColor": COLORS["bg"],
                    "border": f"1px solid {COLORS['border']}",
                }))
        except Exception:
            pass

        if not suggestions:
            suggestions = [html.Div(
                t('tax.no_optimization'),
                style={"color": COLORS["muted"], "textAlign": "center", "padding": "20px"},
            )]

        # Mark-to-Market chart
        mtm_fig = go.Figure()
        mtm_fig.update_layout(
            title=dict(text=t('tax.unrealized_gain_loss'), font=dict(color=COLORS["text"], size=14)),
            paper_bgcolor=COLORS["card"],
            plot_bgcolor=COLORS["card"],
            xaxis=dict(gridcolor=COLORS["border"], color=COLORS["muted"]),
            yaxis=dict(gridcolor=COLORS["border"], color=COLORS["muted"], ticksuffix=f" {ccy}"),
            height=250,
            margin=dict(l=60, r=20, t=40, b=30),
            transition=dict(duration=300),
        )
        mtm_details_content = html.Div()
        try:
            if position_dicts:
                from src.tax.mark_to_market import MarkToMarketEngine
                mtm = MarkToMarketEngine()
                mtm_result = mtm.calculate_ytd(datetime.now().year, position_dicts)
                if mtm_result and mtm_result.positions:
                    symbols = [p.symbol for p in mtm_result.positions]
                    unrealized = [p.mtm_pnl_dkk for p in mtm_result.positions]
                    colors = [COLORS["green"] if u >= 0 else COLORS["red"] for u in unrealized]
                    mtm_fig.add_trace(go.Bar(
                        x=symbols, y=unrealized,
                        marker_color=colors,
                        name=t('tax.unrealized_pnl'),
                    ))
                    total_unrealized = mtm_result.total_mtm_pnl_dkk
                    est_tax = mtm_result.estimated_tax_dkk
                    mtm_details_content = html.Div([
                        html.Span(t('tax.total_unrealized') + ": ", style={"color": COLORS["muted"]}),
                        html.Span(
                            format_value_dkk(total_unrealized),
                            style={"color": COLORS["green"] if total_unrealized >= 0 else COLORS["red"],
                                   "fontWeight": "bold"},
                        ),
                        html.Span(f"  |  {t('tax.est_tax')}: {format_value_dkk(est_tax)}", style={"color": COLORS["muted"]}),
                    ], style={"textAlign": "center", "padding": "10px"})
                else:
                    raise ValueError("No MTM positions")
            else:
                raise ValueError("No positions")
        except Exception:
            mtm_fig.add_annotation(
                text=t('tax.no_mtm_data'),
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(color=COLORS["muted"], size=14),
            )

        # Dividends table
        dividends_content = html.Div()
        try:
            from src.tax.dividend_tracker import DividendTracker
            tracker = DividendTracker()
            dividends = tracker.get_dividends(year=datetime.now().year)
            if dividends:
                header = html.Thead(html.Tr([
                    html.Th(h, style={"color": COLORS["muted"], "padding": "8px",
                                       "borderBottom": f"1px solid {COLORS['border']}"})
                    for h in [t('common.date'), t('common.symbol'), t('tax.gross_col'), t('tax.tax_col'), t('tax.net_col')]
                ]))
                rows = []
                for d in dividends[:20]:
                    rows.append(html.Tr([
                        html.Td(getattr(d, "pay_date", ""), style={"color": COLORS["muted"]}),
                        html.Td(getattr(d, "symbol", ""), style={"color": COLORS["accent"]}),
                        html.Td(f"{getattr(d, 'gross_dkk', 0):,.2f}", style={"color": COLORS["text"]}),
                        html.Td(f"{getattr(d, 'withholding_dkk', 0):,.2f}", style={"color": COLORS["red"]}),
                        html.Td(f"{getattr(d, 'net_dkk', 0):,.2f}", style={"color": COLORS["green"]}),
                    ], style={"borderBottom": f"1px solid {COLORS['border']}"}))
                dividends_content = html.Table(
                    [header, html.Tbody(rows)],
                    style={"width": "100%", "color": COLORS["text"], "fontSize": "0.9rem"},
                )
            else:
                dividends_content = html.Div(
                    t('tax.no_dividends'),
                    style={"color": COLORS["muted"], "textAlign": "center", "padding": "20px"},
                )
        except Exception:
            dividends_content = html.Div(
                t('tax.dividend_data_unavailable'),
                style={"color": COLORS["muted"], "textAlign": "center", "padding": "20px"},
            )

        # Credit history
        credit_history_content = html.Div(
            t('tax.credit_history_empty'),
            style={"color": COLORS["muted"], "textAlign": "center", "padding": "10px",
                   "fontSize": "0.85rem"},
        )
        try:
            from src.tax.tax_credit_tracker import TaxCreditTracker
            tracker = TaxCreditTracker()
            if hasattr(tracker, "get_history"):
                history = tracker.get_history()
                if history:
                    items = []
                    for entry in history[-5:]:
                        yr = getattr(entry, "year", entry.get("year", "") if isinstance(entry, dict) else "")
                        amt = getattr(entry, "amount_dkk", getattr(entry, "amount", entry.get("amount", 0) if isinstance(entry, dict) else 0))
                        items.append(html.Div([
                            html.Span(f"{yr}: ", style={"color": COLORS["muted"]}),
                            html.Span(f"{amt:+,.0f} DKK", style={
                                "color": COLORS["green"] if amt >= 0 else COLORS["red"],
                            }),
                        ], style={"padding": "2px 0", "fontSize": "0.85rem"}))
                    credit_history_content = html.Div(items)
        except Exception:
            pass

        return (kpis, gauge, dbc.ListGroup(suggestions),
                credit_history_content, mtm_fig, mtm_details_content, dividends_content)

    @app.callback(
        Output("tax-export-status", "children"),
        Input("tax-export-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def export_tax_report(_clicks):
        try:
            import csv
            import os
            from datetime import datetime as dt

            report_dir = "reports"
            os.makedirs(report_dir, exist_ok=True)
            filename = f"tax_report_{dt.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = os.path.join(report_dir, filename)

            rows = []

            # Gather trade data
            try:
                from src.tax.transaction_log import TransactionLog
                log = TransactionLog()
                transactions = log.get_transactions() if hasattr(log, "get_transactions") else []
                for txn in transactions:
                    rows.append({
                        "type": "trade",
                        "date": getattr(txn, "date", ""),
                        "symbol": getattr(txn, "symbol", ""),
                        "side": getattr(txn, "side", ""),
                        "qty": getattr(txn, "qty", 0),
                        "price": getattr(txn, "price", 0),
                        "pnl": getattr(txn, "realized_pnl", 0),
                        "currency": getattr(txn, "currency", ""),
                    })
            except Exception:
                pass

            # Gather dividend data
            try:
                from src.tax.dividend_tracker import DividendTracker
                tracker = DividendTracker()
                dividends = tracker.get_dividends(year=datetime.now().year)
                for d in (dividends or []):
                    rows.append({
                        "type": "dividend",
                        "date": getattr(d, "pay_date", ""),
                        "symbol": getattr(d, "symbol", ""),
                        "side": "DIVIDEND",
                        "qty": 0,
                        "price": 0,
                        "pnl": getattr(d, "net_dkk", 0),
                        "currency": getattr(d, "currency", "DKK"),
                    })
            except Exception:
                pass

            if not rows:
                rows.append({
                    "type": "info", "date": dt.now().isoformat(),
                    "symbol": "", "side": "", "qty": 0, "price": 0,
                    "pnl": 0, "currency": "Ingen data at eksportere",
                })

            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["type", "date", "symbol", "side", "qty", "price", "pnl", "currency"])
                writer.writeheader()
                writer.writerows(rows)

            return html.Span(
                f"{t('tax.saved')} {filepath} ({len(rows)} {t('tax.rows')})",
                style={"color": COLORS["green"], "fontSize": "0.85rem"},
            )
        except Exception as e:
            logger.debug(f"Tax export fejl: {e}")
            return html.Span(
                f"{t('tax.export_error')} {e}",
                style={"color": COLORS["red"], "fontSize": "0.85rem"},
            )
