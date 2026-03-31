"""
TradingNotifier – handelsspecifikke notifikationer.

Sender email-alerts ved:
  - Ny handel udført (køb/salg)
  - Stop-loss ramt
  - Daglig performance-rapport
  - Trailing stop ramt
  - Drawdown-advarsel

Bygger ovenpå den eksisterende Notifier med rige HTML-templates.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

from src.notifications.notifier import Notifier, EmailChannel

if TYPE_CHECKING:
    from src.risk.portfolio_tracker import ClosedTrade, Position, PortfolioTracker


# ── Event-typer ──────────────────────────────────────────────

TRADE_EXECUTED = "trade_executed"
STOP_LOSS_HIT = "stop_loss_hit"
TRAILING_STOP_HIT = "trailing_stop_hit"
DAILY_REPORT = "daily_report"
DRAWDOWN_WARNING = "drawdown_warning"
TAKE_PROFIT_HIT = "take_profit_hit"
REGIME_SHIFT = "regime_shift"
CIRCUIT_BREAKER = "circuit_breaker"
WEEKLY_SUMMARY = "weekly_summary"
STRATEGY_DECAY = "strategy_decay"
CORRELATION_WARNING = "correlation_warning"
SYSTEM_ERROR = "system_error"
TAX_WARNING = "tax_warning"


@dataclass
class TradingEventConfig:
    """Konfigurerbare event-triggers."""
    on_trade_executed: bool = True
    on_stop_loss: bool = True
    on_trailing_stop: bool = True
    on_take_profit: bool = True
    on_daily_report: bool = True
    on_drawdown_warning: bool = True
    on_regime_shift: bool = True
    on_circuit_breaker: bool = True
    on_weekly_summary: bool = True
    on_strategy_decay: bool = True
    on_correlation_warning: bool = True
    on_system_error: bool = True
    on_tax_warning: bool = True
    drawdown_threshold_pct: float = 0.05  # Alert ved 5% drawdown


# ── HTML Templates ───────────────────────────────────────────

_BASE_STYLE = """
<style>
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f1a; color: #e0e0e0; margin: 0; padding: 20px; }
  .container { max-width: 640px; margin: 0 auto; background: #16213e; border-radius: 16px; overflow: hidden; }
  .header { padding: 24px 32px; text-align: center; }
  .header h1 { margin: 0; font-size: 22px; }
  .content { padding: 24px 32px; }
  .metric-grid { display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }
  .metric-card { background: #0f3460; border-radius: 10px; padding: 16px; flex: 1 1 140px; min-width: 140px; }
  .metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
  .metric-value { font-size: 22px; font-weight: 700; }
  .positive { color: #00d4aa; }
  .negative { color: #ff4757; }
  .neutral { color: #ffa502; }
  .info { color: #3498db; }
  .trade-box { background: #0f3460; border-radius: 10px; padding: 20px; margin: 16px 0; border-left: 4px solid; }
  .trade-buy { border-color: #00d4aa; }
  .trade-sell { border-color: #ff4757; }
  .trade-stop { border-color: #ff4757; }
  .trade-tp { border-color: #00d4aa; }
  table { width: 100%; border-collapse: collapse; margin: 12px 0; }
  th { text-align: left; padding: 8px 12px; font-size: 11px; color: #888; text-transform: uppercase; border-bottom: 1px solid #1a3a5c; }
  td { padding: 8px 12px; font-size: 14px; border-bottom: 1px solid #0f3460; }
  .footer { padding: 16px 32px; text-align: center; font-size: 11px; color: #555; border-top: 1px solid #0f3460; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; }
  .badge-buy { background: rgba(0,212,170,0.15); color: #00d4aa; }
  .badge-sell { background: rgba(255,71,87,0.15); color: #ff4757; }
  .badge-warn { background: rgba(255,165,2,0.15); color: #ffa502; }
</style>
"""


def _header_html(title: str, severity_color: str) -> str:
    return f"""
    <div class="header" style="background: linear-gradient(135deg, {severity_color}22, #16213e);">
      <h1 style="color: {severity_color};">{title}</h1>
      <p style="color: #888; font-size: 12px; margin: 8px 0 0;">
        Alpha Trading Platform &middot; {datetime.now().strftime('%d-%m-%Y %H:%M')}
      </p>
    </div>
    """


def _footer_html() -> str:
    return """
    <div class="footer">
      Alpha Trading Platform – Automatisk handelsnotifikation<br/>
      Denne email er genereret automatisk. Verificér altid handler manuelt.
    </div>
    """


def _metric_card(label: str, value: str, css_class: str = "") -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{_html.escape(label)}</div>
      <div class="metric-value {css_class}">{_html.escape(value)}</div>
    </div>
    """


def _pnl_class(value: float) -> str:
    if value > 0:
        return "positive"
    elif value < 0:
        return "negative"
    return "neutral"


def _format_pnl(value: float) -> str:
    if value >= 0:
        return f"+${value:,.2f}"
    return f"-${abs(value):,.2f}"


def _format_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2%}"


# ── Trade Executed Template ──────────────────────────────────

def _trade_executed_html(
    trade: ClosedTrade | None = None,
    position: Position | None = None,
    action: str = "BUY",
    symbol: str = "",
    qty: float = 0,
    price: float = 0,
    portfolio_value: float = 0,
    cash: float = 0,
) -> str:
    """HTML for en udført handel."""
    color = "#00d4aa" if action.upper() == "BUY" else "#ff4757"
    badge = "badge-buy" if action.upper() == "BUY" else "badge-sell"
    box_class = "trade-buy" if action.upper() == "BUY" else "trade-sell"
    cost = qty * price

    return f"""
    <html><head>{_BASE_STYLE}</head><body>
    <div class="container">
      {_header_html("Handel Udført", color)}
      <div class="content">
        <div class="trade-box {box_class}">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
              <span class="badge {badge}">{action.upper()}</span>
              <span style="font-size: 24px; font-weight: 700; margin-left: 12px;">{symbol}</span>
            </div>
            <div style="text-align: right;">
              <div style="font-size: 20px; font-weight: 700;">${price:,.2f}</div>
              <div style="font-size: 12px; color: #888;">{qty:,.0f} aktier</div>
            </div>
          </div>
        </div>
        <div class="metric-grid">
          {_metric_card("Handelsværdi", f"${cost:,.2f}", "info")}
          {_metric_card("Porteføljeværdi", f"${portfolio_value:,.2f}", "info")}
          {_metric_card("Kontanter", f"${cash:,.2f}", "neutral")}
        </div>
      </div>
      {_footer_html()}
    </div>
    </body></html>
    """


# ── Stop-Loss Template ───────────────────────────────────────

def _stop_loss_html(
    trade: ClosedTrade,
    portfolio_value: float = 0,
    remaining_positions: int = 0,
) -> str:
    """HTML for stop-loss ramt."""
    pnl = trade.realized_pnl
    pnl_pct = trade.realized_pnl_pct

    return f"""
    <html><head>{_BASE_STYLE}</head><body>
    <div class="container">
      {_header_html("Stop-Loss Aktiveret", "#ff4757")}
      <div class="content">
        <div class="trade-box trade-stop">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
              <span class="badge badge-sell">STOP-LOSS</span>
              <span style="font-size: 24px; font-weight: 700; margin-left: 12px;">{trade.symbol}</span>
            </div>
            <div style="text-align: right;">
              <div class="negative" style="font-size: 20px; font-weight: 700;">{_format_pnl(pnl)}</div>
              <div style="font-size: 12px; color: #888;">{_format_pct(pnl_pct)}</div>
            </div>
          </div>
        </div>
        <table>
          <tr><th>Detalje</th><th style="text-align:right;">Værdi</th></tr>
          <tr><td>Entry-pris</td><td style="text-align:right;">${trade.entry_price:,.2f}</td></tr>
          <tr><td>Exit-pris</td><td style="text-align:right;">${trade.exit_price:,.2f}</td></tr>
          <tr><td>Antal</td><td style="text-align:right;">{trade.qty:,.0f}</td></tr>
          <tr><td>Entry-tid</td><td style="text-align:right;">{trade.entry_time}</td></tr>
          <tr><td>Exit-tid</td><td style="text-align:right;">{trade.exit_time}</td></tr>
          <tr><td>Realiseret P&L</td><td style="text-align:right;" class="{_pnl_class(pnl)}">{_format_pnl(pnl)}</td></tr>
        </table>
        <div class="metric-grid">
          {_metric_card("Porteføljeværdi", f"${portfolio_value:,.2f}", "info")}
          {_metric_card("Åbne positioner", str(remaining_positions), "neutral")}
        </div>
      </div>
      {_footer_html()}
    </div>
    </body></html>
    """


# ── Daily Report Template ────────────────────────────────────

def _daily_report_html(
    total_equity: float,
    cash: float,
    daily_pnl: float,
    daily_pnl_pct: float,
    total_return_pct: float,
    open_positions: int,
    unrealized_pnl: float,
    realized_pnl: float,
    win_rate: float,
    max_drawdown_pct: float,
    sharpe_ratio: float,
    trades_today: int,
    positions: list[dict] | None = None,
) -> str:
    """HTML for daglig performance-rapport."""
    pnl_color = "#00d4aa" if daily_pnl >= 0 else "#ff4757"

    # Position-tabel
    pos_rows = ""
    if positions:
        for p in positions:
            cls = _pnl_class(p.get("pnl", 0))
            pos_rows += f"""
            <tr>
              <td><strong>{p['symbol']}</strong></td>
              <td>{p.get('side', 'long')}</td>
              <td style="text-align:right;">{p.get('qty', 0):,.0f}</td>
              <td style="text-align:right;">${p.get('current_price', 0):,.2f}</td>
              <td style="text-align:right;" class="{cls}">{_format_pnl(p.get('pnl', 0))}</td>
              <td style="text-align:right;" class="{cls}">{_format_pct(p.get('pnl_pct', 0))}</td>
            </tr>
            """

    positions_section = ""
    if pos_rows:
        positions_section = f"""
        <h3 style="color: #e0e0e0; margin-top: 24px;">Åbne Positioner</h3>
        <table>
          <tr>
            <th>Symbol</th><th>Side</th><th style="text-align:right;">Antal</th>
            <th style="text-align:right;">Kurs</th><th style="text-align:right;">P&L</th>
            <th style="text-align:right;">P&L %</th>
          </tr>
          {pos_rows}
        </table>
        """

    return f"""
    <html><head>{_BASE_STYLE}</head><body>
    <div class="container">
      {_header_html("Daglig Performance-Rapport", pnl_color)}
      <div class="content">
        <div class="metric-grid">
          {_metric_card("Porteføljeværdi", f"${total_equity:,.2f}", "info")}
          {_metric_card("Daglig P&L", _format_pnl(daily_pnl), _pnl_class(daily_pnl))}
          {_metric_card("Daglig Afkast", _format_pct(daily_pnl_pct), _pnl_class(daily_pnl_pct))}
          {_metric_card("Totalt Afkast", _format_pct(total_return_pct), _pnl_class(total_return_pct))}
        </div>

        <h3 style="color: #e0e0e0; margin-top: 24px;">Nøgletal</h3>
        <div class="metric-grid">
          {_metric_card("Kontanter", f"${cash:,.2f}", "neutral")}
          {_metric_card("Urealiseret P&L", _format_pnl(unrealized_pnl), _pnl_class(unrealized_pnl))}
          {_metric_card("Realiseret P&L", _format_pnl(realized_pnl), _pnl_class(realized_pnl))}
          {_metric_card("Åbne positioner", str(open_positions), "info")}
          {_metric_card("Handler i dag", str(trades_today), "info")}
          {_metric_card("Win Rate", f"{win_rate:.0%}", "positive" if win_rate >= 0.5 else "negative")}
          {_metric_card("Max Drawdown", f"{max_drawdown_pct:.1%}", "negative" if max_drawdown_pct > 0.05 else "neutral")}
          {_metric_card("Sharpe Ratio", f"{sharpe_ratio:.2f}", "positive" if sharpe_ratio > 1 else "neutral")}
        </div>

        {positions_section}
      </div>
      {_footer_html()}
    </div>
    </body></html>
    """


# ── Drawdown Warning Template ────────────────────────────────

def _drawdown_warning_html(
    current_drawdown_pct: float,
    peak_equity: float,
    current_equity: float,
    threshold_pct: float,
) -> str:
    """HTML for drawdown-advarsel."""
    loss = peak_equity - current_equity

    return f"""
    <html><head>{_BASE_STYLE}</head><body>
    <div class="container">
      {_header_html("Drawdown-Advarsel", "#ffa502")}
      <div class="content">
        <div style="text-align: center; padding: 20px 0;">
          <div style="font-size: 48px; font-weight: 700; color: #ff4757;">{current_drawdown_pct:.1%}</div>
          <div style="font-size: 14px; color: #888;">Nuværende drawdown fra peak</div>
        </div>
        <div class="metric-grid">
          {_metric_card("Peak Equity", f"${peak_equity:,.2f}", "neutral")}
          {_metric_card("Nuværende Equity", f"${current_equity:,.2f}", "negative")}
          {_metric_card("Tab fra Peak", _format_pnl(-loss), "negative")}
          {_metric_card("Grænse", f"{threshold_pct:.1%}", "neutral")}
        </div>
        <div style="background: rgba(255,165,2,0.1); border: 1px solid #ffa502; border-radius: 8px; padding: 16px; margin-top: 16px;">
          <strong style="color: #ffa502;">Anbefaling:</strong>
          <p style="margin: 8px 0 0; color: #ccc;">
            Porteføljen er faldet {current_drawdown_pct:.1%} fra toppen.
            Overvej at reducere positionsstørrelser eller stramme stop-losses.
          </p>
        </div>
      </div>
      {_footer_html()}
    </div>
    </body></html>
    """


# ── Regime Shift Template ────────────────────────────────────

def _regime_shift_html(
    from_regime: str,
    to_regime: str,
    confidence: float,
    recommended_action: str = "",
) -> str:
    """HTML for regime-skift alert."""
    color_map = {
        "BULL": "#00d4aa", "BEAR": "#ff4757", "CRASH": "#ff4757",
        "SIDEWAYS": "#ffa502", "RECOVERY": "#3498db",
    }
    to_color = color_map.get(to_regime.upper(), "#ffa502")

    action_section = ""
    if recommended_action:
        action_section = f"""
        <div style="background: rgba(52,152,219,0.1); border: 1px solid #3498db;
                    border-radius: 8px; padding: 16px; margin-top: 16px;">
          <strong style="color: #3498db;">Anbefalet handling:</strong>
          <p style="margin: 8px 0 0; color: #ccc;">{recommended_action}</p>
        </div>
        """

    return f"""
    <html><head>{_BASE_STYLE}</head><body>
    <div class="container">
      {_header_html("Regime-Skift Detekteret", to_color)}
      <div class="content">
        <div style="text-align: center; padding: 20px 0;">
          <div style="display: flex; align-items: center; justify-content: center; gap: 16px;">
            <span style="font-size: 28px; font-weight: 700; color: #888;">{from_regime}</span>
            <span style="font-size: 24px; color: {to_color};">→</span>
            <span style="font-size: 28px; font-weight: 700; color: {to_color};">{to_regime}</span>
          </div>
          <div style="font-size: 14px; color: #888; margin-top: 8px;">
            Konfidens: {confidence:.0f}%
          </div>
        </div>
        <div class="metric-grid">
          {_metric_card("Fra", from_regime, "neutral")}
          {_metric_card("Til", to_regime, "info")}
          {_metric_card("Konfidens", f"{confidence:.0f}%", "positive" if confidence > 80 else "neutral")}
        </div>
        {action_section}
      </div>
      {_footer_html()}
    </div>
    </body></html>
    """


# ── Circuit Breaker Template ────────────────────────────────

def _circuit_breaker_html(
    level: str,
    reason: str,
    current_drawdown_pct: float = 0.0,
    actions_taken: list[str] | None = None,
) -> str:
    """HTML for circuit breaker alert."""
    level_colors = {
        "WARNING": "#ffa502", "HALT": "#ff4757", "EMERGENCY": "#ff0000",
    }
    color = level_colors.get(level.upper(), "#ff4757")

    actions_html = ""
    if actions_taken:
        items = "".join(f"<li style='margin: 4px 0;'>{a}</li>" for a in actions_taken)
        actions_html = f"""
        <div style="margin-top: 16px;">
          <strong style="color: #ccc;">Handlinger taget:</strong>
          <ul style="color: #ccc; margin: 8px 0; padding-left: 20px;">{items}</ul>
        </div>
        """

    dd_section = ""
    if current_drawdown_pct > 0:
        dd_section = _metric_card("Drawdown", f"{current_drawdown_pct:.1%}", "negative")

    return f"""
    <html><head>{_BASE_STYLE}</head><body>
    <div class="container">
      {_header_html("Circuit Breaker Aktiveret", color)}
      <div class="content">
        <div style="text-align: center; padding: 20px 0;">
          <div style="font-size: 48px;">🛑</div>
          <div style="font-size: 24px; font-weight: 700; color: {color}; margin-top: 8px;">
            {level.upper()}
          </div>
        </div>
        <div class="metric-grid">
          {_metric_card("Niveau", level.upper(), "negative")}
          {_metric_card("Årsag", reason, "neutral")}
          {dd_section}
        </div>
        {actions_html}
        <div style="background: rgba(255,71,87,0.1); border: 1px solid #ff4757;
                    border-radius: 8px; padding: 16px; margin-top: 16px;">
          <strong style="color: #ff4757;">VIGTIGT:</strong>
          <p style="margin: 8px 0 0; color: #ccc;">
            Al automatisk handel er stoppet. Manuel gennemgang kræves før genstart.
          </p>
        </div>
      </div>
      {_footer_html()}
    </div>
    </body></html>
    """


# ── Weekly Summary Template ─────────────────────────────────

def _weekly_summary_html(
    week_pnl: float,
    week_pnl_pct: float,
    total_equity: float,
    trades_count: int,
    win_rate: float,
    sharpe_ratio: float = 0.0,
    max_drawdown_pct: float = 0.0,
    regime: str = "",
) -> str:
    """HTML for ugentlig opsummering."""
    pnl_color = "#00d4aa" if week_pnl >= 0 else "#ff4757"

    regime_section = ""
    if regime:
        regime_section = _metric_card("Regime", regime, "info")

    return f"""
    <html><head>{_BASE_STYLE}</head><body>
    <div class="container">
      {_header_html("Ugentlig Opsummering", pnl_color)}
      <div class="content">
        <div style="text-align: center; padding: 20px 0;">
          <div style="font-size: 14px; color: #888;">Ugens resultat</div>
          <div style="font-size: 42px; font-weight: 700; color: {pnl_color};">{_format_pnl(week_pnl)}</div>
          <div style="font-size: 18px; color: {pnl_color};">{_format_pct(week_pnl_pct)}</div>
        </div>
        <div class="metric-grid">
          {_metric_card("Porteføljeværdi", f"${total_equity:,.2f}", "info")}
          {_metric_card("Handler", str(trades_count), "info")}
          {_metric_card("Win Rate", f"{win_rate:.0%}", "positive" if win_rate >= 0.5 else "negative")}
          {_metric_card("Sharpe", f"{sharpe_ratio:.2f}", "positive" if sharpe_ratio > 1 else "neutral")}
          {_metric_card("Max Drawdown", f"{max_drawdown_pct:.1%}", "negative" if max_drawdown_pct > 0.05 else "neutral")}
          {regime_section}
        </div>
      </div>
      {_footer_html()}
    </div>
    </body></html>
    """


# ── Strategy Decay Template ─────────────────────────────────

def _strategy_decay_html(
    strategy_name: str,
    current_sharpe: float,
    previous_sharpe: float,
    win_rate: float,
    previous_win_rate: float,
    recommendation: str = "",
) -> str:
    """HTML for strategi-forfald alert."""
    sharpe_change = current_sharpe - previous_sharpe
    wr_change = win_rate - previous_win_rate

    rec_section = ""
    if recommendation:
        rec_section = f"""
        <div style="background: rgba(255,165,2,0.1); border: 1px solid #ffa502;
                    border-radius: 8px; padding: 16px; margin-top: 16px;">
          <strong style="color: #ffa502;">Anbefaling:</strong>
          <p style="margin: 8px 0 0; color: #ccc;">{recommendation}</p>
        </div>
        """

    return f"""
    <html><head>{_BASE_STYLE}</head><body>
    <div class="container">
      {_header_html(f"Strategi-Forfald: {strategy_name}", "#ffa502")}
      <div class="content">
        <h3 style="color: #e0e0e0;">Sharpe Ratio</h3>
        <div class="metric-grid">
          {_metric_card("Tidligere", f"{previous_sharpe:.2f}", "neutral")}
          {_metric_card("Nuværende", f"{current_sharpe:.2f}", _pnl_class(current_sharpe))}
          {_metric_card("Ændring", f"{sharpe_change:+.2f}", _pnl_class(sharpe_change))}
        </div>
        <h3 style="color: #e0e0e0; margin-top: 20px;">Win Rate</h3>
        <div class="metric-grid">
          {_metric_card("Tidligere", f"{previous_win_rate:.0%}", "neutral")}
          {_metric_card("Nuværende", f"{win_rate:.0%}", _pnl_class(wr_change))}
          {_metric_card("Ændring", f"{wr_change:+.0%}", _pnl_class(wr_change))}
        </div>
        {rec_section}
      </div>
      {_footer_html()}
    </div>
    </body></html>
    """


# ── System Error Template ───────────────────────────────────

def _system_error_html(
    component: str,
    error_message: str,
    error_type: str = "",
    is_recoverable: bool = True,
) -> str:
    """HTML for systemfejl alert."""
    color = "#ffa502" if is_recoverable else "#ff4757"
    status_text = "Recoverable" if is_recoverable else "KRITISK – Kræver handling"
    status_color = "#ffa502" if is_recoverable else "#ff4757"

    type_section = ""
    if error_type:
        type_section = _metric_card("Fejltype", error_type, "neutral")

    return f"""
    <html><head>{_BASE_STYLE}</head><body>
    <div class="container">
      {_header_html(f"Systemfejl: {component}", color)}
      <div class="content">
        <div class="trade-box trade-stop">
          <div style="font-size: 16px; font-weight: 600; color: #ff4757;">
            {error_message}
          </div>
        </div>
        <div class="metric-grid">
          {_metric_card("Komponent", component, "info")}
          {type_section}
          {_metric_card("Status", status_text, "neutral" if is_recoverable else "negative")}
        </div>
        <div style="background: rgba({'255,165,2' if is_recoverable else '255,71,87'},0.1);
                    border: 1px solid {status_color}; border-radius: 8px;
                    padding: 16px; margin-top: 16px;">
          <strong style="color: {status_color};">
            {'Systemet forsøger automatisk genopretning.' if is_recoverable else 'Manuel indgriben påkrævet!'}
          </strong>
        </div>
      </div>
      {_footer_html()}
    </div>
    </body></html>
    """


# ── Tax Warning Template ────────────────────────────────────

def _tax_warning_html(
    warning_type: str,
    realized_gains: float,
    tax_threshold: float,
    estimated_tax: float,
) -> str:
    """HTML for skatteadvarsel."""
    over_threshold = realized_gains >= tax_threshold
    color = "#ff4757" if over_threshold else "#ffa502"
    pct_of_threshold = (realized_gains / tax_threshold * 100) if tax_threshold > 0 else 0

    return f"""
    <html><head>{_BASE_STYLE}</head><body>
    <div class="container">
      {_header_html(f"Skatteadvarsel: {warning_type}", color)}
      <div class="content">
        <div style="text-align: center; padding: 20px 0;">
          <div style="font-size: 14px; color: #888;">Realiserede gevinster</div>
          <div style="font-size: 36px; font-weight: 700; color: {color};">
            {realized_gains:,.0f} DKK
          </div>
          <div style="font-size: 14px; color: #888;">
            {pct_of_threshold:.0f}% af skattegrænse
          </div>
        </div>
        <div class="metric-grid">
          {_metric_card("Skattegrænse", f"{tax_threshold:,.0f} DKK", "neutral")}
          {_metric_card("Estimeret skat", f"{estimated_tax:,.0f} DKK", "negative")}
          {_metric_card("Status", "Over grænse!" if over_threshold else "Under grænse", "negative" if over_threshold else "positive")}
        </div>
        <div style="background: rgba(255,165,2,0.1); border: 1px solid #ffa502;
                    border-radius: 8px; padding: 16px; margin-top: 16px;">
          <strong style="color: #ffa502;">⚠️ Vejledende beregning</strong>
          <p style="margin: 8px 0 0; color: #ccc;">
            Denne skatteberegning er vejledende. Kontakt din revisor for præcis skatteberegning.
          </p>
        </div>
      </div>
      {_footer_html()}
    </div>
    </body></html>
    """


# ── TradingNotifier ──────────────────────────────────────────


class TradingNotifier:
    """
    Handelsspecifik notifikationshub.

    Wrapper omkring Notifier med metoder til:
      - send_trade_alert(): Ny handel udført
      - send_stop_loss_alert(): Stop-loss ramt
      - send_trailing_stop_alert(): Trailing stop ramt
      - send_take_profit_alert(): Take profit ramt
      - send_daily_report(): Daglig performance-rapport
      - send_drawdown_warning(): Drawdown-advarsel

    Konfigurérbare event-triggers via TradingEventConfig.
    """

    def __init__(
        self,
        notifier: Notifier | None = None,
        event_config: TradingEventConfig | None = None,
        cache_dir: str = "data_cache",
    ) -> None:
        self._notifier = notifier or Notifier(cache_dir=cache_dir)
        self._config = event_config or TradingEventConfig()
        self._last_drawdown_alert: float = 0.0  # Undgå spam

    @property
    def notifier(self) -> Notifier:
        """Adgang til den underliggende Notifier."""
        return self._notifier

    @property
    def config(self) -> TradingEventConfig:
        """Adgang til event-konfiguration."""
        return self._config

    # ── Trade Executed ────────────────────────────────────────

    def send_trade_alert(
        self,
        action: str,
        symbol: str,
        qty: float,
        price: float,
        portfolio_value: float = 0,
        cash: float = 0,
    ) -> int:
        """
        Send notifikation om udført handel.

        Args:
            action: "BUY" eller "SELL"
            symbol: Aktiesymbol
            qty: Antal aktier
            price: Handelspris
            portfolio_value: Samlet porteføljeværdi
            cash: Kontantbeholdning

        Returns:
            Antal kanaler der modtog notifikationen.
        """
        if not self._config.on_trade_executed:
            logger.debug(f"[trading_notifier] Trade alert deaktiveret – springer over")
            return 0

        cost = qty * price
        title = f"{action.upper()} {symbol}: {qty:,.0f} aktier @ ${price:,.2f}"
        message = (
            f"Handel udført: {action.upper()} {qty:,.0f} {symbol} @ ${price:,.2f}\n"
            f"Handelsværdi: ${cost:,.2f}\n"
            f"Porteføljeværdi: ${portfolio_value:,.2f}\n"
            f"Kontanter: ${cash:,.2f}"
        )

        return self._notifier.send(
            severity="INFO",
            title=title,
            message=message,
            category=TRADE_EXECUTED,
        )

    # ── Stop-Loss ─────────────────────────────────────────────

    def send_stop_loss_alert(
        self,
        trade: ClosedTrade,
        portfolio_value: float = 0,
        remaining_positions: int = 0,
    ) -> int:
        """
        Send notifikation om stop-loss ramt.

        Args:
            trade: Den lukkede trade med P&L-info.
            portfolio_value: Samlet porteføljeværdi efter lukning.
            remaining_positions: Antal tilbageværende positioner.

        Returns:
            Antal kanaler der modtog notifikationen.
        """
        if not self._config.on_stop_loss:
            logger.debug("[trading_notifier] Stop-loss alert deaktiveret")
            return 0

        title = f"STOP-LOSS: {trade.symbol} lukket @ ${trade.exit_price:,.2f}"
        message = (
            f"Stop-loss aktiveret for {trade.symbol}\n"
            f"Entry: ${trade.entry_price:,.2f} → Exit: ${trade.exit_price:,.2f}\n"
            f"Realiseret P&L: {_format_pnl(trade.realized_pnl)} ({_format_pct(trade.realized_pnl_pct)})\n"
            f"Porteføljeværdi: ${portfolio_value:,.2f}\n"
            f"Åbne positioner: {remaining_positions}"
        )

        return self._notifier.send(
            severity="WARNING",
            title=title,
            message=message,
            category=STOP_LOSS_HIT,
        )

    # ── Trailing Stop ─────────────────────────────────────────

    def send_trailing_stop_alert(
        self,
        trade: ClosedTrade,
        portfolio_value: float = 0,
        remaining_positions: int = 0,
    ) -> int:
        """Send notifikation om trailing stop ramt."""
        if not self._config.on_trailing_stop:
            logger.debug("[trading_notifier] Trailing stop alert deaktiveret")
            return 0

        title = f"TRAILING STOP: {trade.symbol} lukket @ ${trade.exit_price:,.2f}"
        message = (
            f"Trailing stop aktiveret for {trade.symbol}\n"
            f"Entry: ${trade.entry_price:,.2f} → Exit: ${trade.exit_price:,.2f}\n"
            f"Realiseret P&L: {_format_pnl(trade.realized_pnl)} ({_format_pct(trade.realized_pnl_pct)})"
        )

        return self._notifier.send(
            severity="INFO" if trade.realized_pnl >= 0 else "WARNING",
            title=title,
            message=message,
            category=TRAILING_STOP_HIT,
        )

    # ── Take Profit ───────────────────────────────────────────

    def send_take_profit_alert(
        self,
        trade: ClosedTrade,
        portfolio_value: float = 0,
    ) -> int:
        """Send notifikation om take profit ramt."""
        if not self._config.on_take_profit:
            logger.debug("[trading_notifier] Take profit alert deaktiveret")
            return 0

        title = f"TAKE PROFIT: {trade.symbol} lukket @ ${trade.exit_price:,.2f}"
        message = (
            f"Take profit nået for {trade.symbol}\n"
            f"Entry: ${trade.entry_price:,.2f} → Exit: ${trade.exit_price:,.2f}\n"
            f"Realiseret P&L: {_format_pnl(trade.realized_pnl)} ({_format_pct(trade.realized_pnl_pct)})"
        )

        return self._notifier.send(
            severity="INFO",
            title=title,
            message=message,
            category=TAKE_PROFIT_HIT,
        )

    # ── Daily Report ──────────────────────────────────────────

    def send_daily_report(
        self,
        tracker: PortfolioTracker,
        trades_today: int = 0,
    ) -> int:
        """
        Send daglig performance-rapport.

        Args:
            tracker: PortfolioTracker med aktuelle metrics.
            trades_today: Antal handler udført i dag.

        Returns:
            Antal kanaler der modtog notifikationen.
        """
        if not self._config.on_daily_report:
            logger.debug("[trading_notifier] Daily report deaktiveret")
            return 0

        summary = tracker.summary()

        # Byg position-liste
        positions = []
        for sym, pos in tracker.positions.items():
            positions.append({
                "symbol": sym,
                "side": pos.side,
                "qty": pos.qty,
                "current_price": pos.current_price,
                "pnl": pos.unrealized_pnl,
                "pnl_pct": pos.unrealized_pnl_pct,
            })

        daily_pnl = summary["daily_pnl"]
        title = (
            f"Daglig Rapport: {_format_pnl(daily_pnl)} "
            f"({_format_pct(summary['daily_pnl_pct'])})"
        )

        # Tekst-version
        lines = [
            f"Daglig Performance-Rapport – {datetime.now().strftime('%d-%m-%Y')}",
            f"{'=' * 50}",
            f"Porteføljeværdi: ${summary['total_equity']:,.2f}",
            f"Kontanter: ${summary['cash']:,.2f}",
            f"Daglig P&L: {_format_pnl(daily_pnl)} ({_format_pct(summary['daily_pnl_pct'])})",
            f"Totalt afkast: {_format_pct(summary['total_return_pct'])}",
            f"Urealiseret P&L: {_format_pnl(summary['unrealized_pnl'])}",
            f"Realiseret P&L: {_format_pnl(summary['realized_pnl'])}",
            f"Åbne positioner: {summary['positions']}",
            f"Handler i dag: {trades_today}",
            f"Win rate: {summary['win_rate']:.0%}",
            f"Max drawdown: {summary['max_drawdown_pct']:.1%}",
            f"Sharpe ratio: {summary['sharpe_ratio']:.2f}",
        ]

        if positions:
            lines.append(f"\nÅbne positioner:")
            for p in positions:
                lines.append(
                    f"  {p['symbol']}: {p['qty']:,.0f} @ ${p['current_price']:,.2f} "
                    f"({_format_pnl(p['pnl'])} / {_format_pct(p['pnl_pct'])})"
                )

        message = "\n".join(lines)

        severity = "INFO" if daily_pnl >= 0 else "WARNING"

        return self._notifier.send(
            severity=severity,
            title=title,
            message=message,
            category=DAILY_REPORT,
        )

    # ── Drawdown Warning ──────────────────────────────────────

    def send_drawdown_warning(
        self,
        current_drawdown_pct: float,
        peak_equity: float,
        current_equity: float,
    ) -> int:
        """
        Send drawdown-advarsel hvis over threshold.

        Sender kun én alert per drawdown-niveau (undgår spam).

        Args:
            current_drawdown_pct: Nuværende drawdown som decimal (0.05 = 5%).
            peak_equity: Højeste porteføljeværdi.
            current_equity: Nuværende porteføljeværdi.

        Returns:
            Antal kanaler der modtog notifikationen, eller 0 hvis deaktiveret/under threshold.
        """
        if not self._config.on_drawdown_warning:
            return 0

        threshold = self._config.drawdown_threshold_pct
        if current_drawdown_pct < threshold:
            return 0

        # Undgå gentagelser – kun alert ved nye 1%-trin
        level = int(current_drawdown_pct * 100)
        if level <= self._last_drawdown_alert:
            return 0

        self._last_drawdown_alert = level

        title = f"DRAWDOWN ADVARSEL: Porteføljen er faldet {current_drawdown_pct:.1%}"
        loss = peak_equity - current_equity
        message = (
            f"Portefølje-drawdown: {current_drawdown_pct:.1%}\n"
            f"Peak equity: ${peak_equity:,.2f}\n"
            f"Nuværende equity: ${current_equity:,.2f}\n"
            f"Tab fra peak: {_format_pnl(-loss)}\n"
            f"Grænse: {threshold:.1%}"
        )

        severity = "CRITICAL" if current_drawdown_pct >= threshold * 2 else "WARNING"

        return self._notifier.send(
            severity=severity,
            title=title,
            message=message,
            category=DRAWDOWN_WARNING,
        )

    def reset_drawdown_tracker(self) -> None:
        """Nulstil drawdown-spam-filter (f.eks. ved ny handelsdag)."""
        self._last_drawdown_alert = 0.0

    # ── Check Portfolio for Alerts ────────────────────────────

    def check_portfolio_alerts(self, tracker: PortfolioTracker) -> int:
        """
        Tjek portefølje og send relevante alerts.

        Kalder send_drawdown_warning() automatisk.

        Returns:
            Antal sendte notifikationer.
        """
        sent = 0
        dd = tracker.current_drawdown_pct
        if dd > 0:
            result = self.send_drawdown_warning(
                current_drawdown_pct=dd,
                peak_equity=tracker._peak_equity,
                current_equity=tracker.total_equity,
            )
            sent += result
        return sent

    # ── HTML getters (til EmailChannel custom integration) ────

    @staticmethod
    def get_trade_html(
        action: str, symbol: str, qty: float, price: float,
        portfolio_value: float = 0, cash: float = 0,
    ) -> str:
        """Returnér HTML for trade-alert (til custom email-integration)."""
        return _trade_executed_html(
            action=action, symbol=symbol, qty=qty, price=price,
            portfolio_value=portfolio_value, cash=cash,
        )

    @staticmethod
    def get_stop_loss_html(
        trade: ClosedTrade,
        portfolio_value: float = 0,
        remaining_positions: int = 0,
    ) -> str:
        """Returnér HTML for stop-loss alert."""
        return _stop_loss_html(trade, portfolio_value, remaining_positions)

    @staticmethod
    def get_daily_report_html(
        tracker: PortfolioTracker,
        trades_today: int = 0,
    ) -> str:
        """Returnér HTML for daglig rapport."""
        summary = tracker.summary()
        positions = []
        for sym, pos in tracker.positions.items():
            positions.append({
                "symbol": sym,
                "side": pos.side,
                "qty": pos.qty,
                "current_price": pos.current_price,
                "pnl": pos.unrealized_pnl,
                "pnl_pct": pos.unrealized_pnl_pct,
            })
        return _daily_report_html(
            total_equity=summary["total_equity"],
            cash=summary["cash"],
            daily_pnl=summary["daily_pnl"],
            daily_pnl_pct=summary["daily_pnl_pct"],
            total_return_pct=summary["total_return_pct"],
            open_positions=summary["positions"],
            unrealized_pnl=summary["unrealized_pnl"],
            realized_pnl=summary["realized_pnl"],
            win_rate=summary["win_rate"],
            max_drawdown_pct=summary["max_drawdown_pct"],
            sharpe_ratio=summary["sharpe_ratio"],
            trades_today=trades_today,
            positions=positions,
        )

    @staticmethod
    def get_drawdown_html(
        current_drawdown_pct: float,
        peak_equity: float,
        current_equity: float,
        threshold_pct: float = 0.05,
    ) -> str:
        """Returnér HTML for drawdown-advarsel."""
        return _drawdown_warning_html(
            current_drawdown_pct, peak_equity, current_equity, threshold_pct,
        )

    # ── Regime Shift ───────────────────────────────────────────

    def send_regime_shift_alert(
        self,
        from_regime: str,
        to_regime: str,
        confidence: float,
        recommended_action: str = "",
        details: dict | None = None,
    ) -> int:
        """
        Send notifikation om regime-skift.

        Args:
            from_regime: Tidligere regime (f.eks. "BULL").
            to_regime: Nyt regime (f.eks. "BEAR").
            confidence: Konfidens i procent (0-100).
            recommended_action: Anbefalet handling.
            details: Ekstra detaljer.

        Returns:
            Antal kanaler der modtog notifikationen.
        """
        if not self._config.on_regime_shift:
            logger.debug("[trading_notifier] Regime shift alert deaktiveret")
            return 0

        title = f"REGIME-SKIFT: {from_regime} → {to_regime} ({confidence:.0f}%)"
        lines = [
            f"Regime-skift detekteret",
            f"Fra: {from_regime} → Til: {to_regime}",
            f"Konfidens: {confidence:.0f}%",
        ]
        if recommended_action:
            lines.append(f"Anbefalet handling: {recommended_action}")
        if details:
            for k, v in details.items():
                lines.append(f"{k}: {v}")

        severity = "CRITICAL" if to_regime.upper() in ("CRASH", "BEAR") else "WARNING"

        return self._notifier.send(
            severity=severity,
            title=title,
            message="\n".join(lines),
            category=REGIME_SHIFT,
        )

    # ── Circuit Breaker ────────────────────────────────────────

    def send_circuit_breaker_alert(
        self,
        level: str,
        reason: str,
        current_drawdown_pct: float = 0.0,
        actions_taken: list[str] | None = None,
    ) -> int:
        """
        Send notifikation om circuit breaker aktivering.

        Args:
            level: Niveau ("WARNING", "HALT", "EMERGENCY").
            reason: Årsag til aktivering.
            current_drawdown_pct: Nuværende drawdown.
            actions_taken: Liste af handlinger taget.

        Returns:
            Antal kanaler der modtog notifikationen.
        """
        if not self._config.on_circuit_breaker:
            logger.debug("[trading_notifier] Circuit breaker alert deaktiveret")
            return 0

        title = f"CIRCUIT BREAKER: {level.upper()} – {reason}"
        lines = [
            f"Circuit breaker aktiveret!",
            f"Niveau: {level.upper()}",
            f"Årsag: {reason}",
        ]
        if current_drawdown_pct > 0:
            lines.append(f"Nuværende drawdown: {current_drawdown_pct:.1%}")
        if actions_taken:
            lines.append("\nHandlinger taget:")
            for a in actions_taken:
                lines.append(f"  • {a}")

        return self._notifier.send(
            severity="CRITICAL",
            title=title,
            message="\n".join(lines),
            category=CIRCUIT_BREAKER,
        )

    # ── Weekly Summary ─────────────────────────────────────────

    def send_weekly_summary(
        self,
        week_pnl: float,
        week_pnl_pct: float,
        total_equity: float,
        trades_count: int,
        win_rate: float,
        best_trade: str = "",
        worst_trade: str = "",
        sharpe_ratio: float = 0.0,
        max_drawdown_pct: float = 0.0,
        regime: str = "",
        positions_held: int = 0,
    ) -> int:
        """
        Send ugentlig opsummering (typisk søndag aften).

        Returns:
            Antal kanaler der modtog notifikationen.
        """
        if not self._config.on_weekly_summary:
            logger.debug("[trading_notifier] Weekly summary deaktiveret")
            return 0

        title = (
            f"Ugentlig Opsummering: {_format_pnl(week_pnl)} "
            f"({_format_pct(week_pnl_pct)})"
        )
        lines = [
            f"Ugentlig Performance-Rapport",
            f"{'=' * 45}",
            f"Uge-P&L: {_format_pnl(week_pnl)} ({_format_pct(week_pnl_pct)})",
            f"Porteføljeværdi: ${total_equity:,.2f}",
            f"Handler i ugen: {trades_count}",
            f"Win rate: {win_rate:.0%}",
            f"Sharpe ratio: {sharpe_ratio:.2f}",
            f"Max drawdown: {max_drawdown_pct:.1%}",
            f"Åbne positioner: {positions_held}",
        ]
        if regime:
            lines.append(f"Aktuelt regime: {regime}")
        if best_trade:
            lines.append(f"Bedste handel: {best_trade}")
        if worst_trade:
            lines.append(f"Dårligste handel: {worst_trade}")

        severity = "INFO" if week_pnl >= 0 else "WARNING"

        return self._notifier.send(
            severity=severity,
            title=title,
            message="\n".join(lines),
            category=WEEKLY_SUMMARY,
        )

    # ── Strategy Decay ─────────────────────────────────────────

    def send_strategy_decay_alert(
        self,
        strategy_name: str,
        current_sharpe: float,
        previous_sharpe: float,
        win_rate: float,
        previous_win_rate: float,
        decay_severity: str = "WARNING",
        recommendation: str = "",
    ) -> int:
        """
        Send notifikation om strategi-forfald (faldende Sharpe/win rate).

        Args:
            strategy_name: Navn på strategien.
            current_sharpe: Nuværende Sharpe ratio.
            previous_sharpe: Tidligere Sharpe ratio.
            win_rate: Nuværende win rate.
            previous_win_rate: Tidligere win rate.
            decay_severity: "WARNING" eller "CRITICAL".
            recommendation: Anbefalet handling.

        Returns:
            Antal kanaler der modtog notifikationen.
        """
        if not self._config.on_strategy_decay:
            logger.debug("[trading_notifier] Strategy decay alert deaktiveret")
            return 0

        sharpe_change = current_sharpe - previous_sharpe
        wr_change = win_rate - previous_win_rate

        title = f"STRATEGI-FORFALD: {strategy_name} (Sharpe {current_sharpe:.2f})"
        lines = [
            f"Strategi-forfald detekteret: {strategy_name}",
            f"Sharpe ratio: {previous_sharpe:.2f} → {current_sharpe:.2f} ({sharpe_change:+.2f})",
            f"Win rate: {previous_win_rate:.0%} → {win_rate:.0%} ({wr_change:+.0%})",
        ]
        if recommendation:
            lines.append(f"\nAnbefaling: {recommendation}")

        severity = decay_severity if decay_severity in ("WARNING", "CRITICAL") else "WARNING"

        return self._notifier.send(
            severity=severity,
            title=title,
            message="\n".join(lines),
            category=STRATEGY_DECAY,
        )

    # ── Correlation Warning ────────────────────────────────────

    def send_correlation_warning(
        self,
        symbol_a: str,
        symbol_b: str,
        correlation: float,
        threshold: float = 0.85,
        combined_exposure_pct: float = 0.0,
    ) -> int:
        """
        Send advarsel om høj korrelation mellem positioner.

        Args:
            symbol_a: Første symbol.
            symbol_b: Andet symbol.
            correlation: Korrelationskoefficient.
            threshold: Grænseværdi.
            combined_exposure_pct: Samlet eksponering som andel af portefølje.

        Returns:
            Antal kanaler der modtog notifikationen.
        """
        if not self._config.on_correlation_warning:
            logger.debug("[trading_notifier] Correlation warning deaktiveret")
            return 0

        title = f"KORRELATIONS-ADVARSEL: {symbol_a}/{symbol_b} ({correlation:.2f})"
        lines = [
            f"Høj korrelation detekteret!",
            f"Par: {symbol_a} / {symbol_b}",
            f"Korrelation: {correlation:.2f} (grænse: {threshold:.2f})",
        ]
        if combined_exposure_pct > 0:
            lines.append(f"Samlet eksponering: {combined_exposure_pct:.1%} af portefølje")
        lines.append(
            "\nAnbefaling: Overvej at reducere én af positionerne "
            "for at mindske koncentrationsrisiko."
        )

        return self._notifier.send(
            severity="WARNING",
            title=title,
            message="\n".join(lines),
            category=CORRELATION_WARNING,
        )

    # ── System Error ───────────────────────────────────────────

    def send_system_error_alert(
        self,
        component: str,
        error_message: str,
        error_type: str = "",
        stack_trace: str = "",
        is_recoverable: bool = True,
    ) -> int:
        """
        Send notifikation om systemfejl.

        Args:
            component: Komponent der fejlede (f.eks. "Data Pipeline").
            error_message: Fejlbesked.
            error_type: Type af fejl (f.eks. "ConnectionError").
            stack_trace: Stack trace (trunkeret).
            is_recoverable: Om fejlen er recoverable.

        Returns:
            Antal kanaler der modtog notifikationen.
        """
        if not self._config.on_system_error:
            logger.debug("[trading_notifier] System error alert deaktiveret")
            return 0

        title = f"SYSTEMFEJL: {component} – {error_message[:80]}"
        lines = [
            f"Systemfejl i: {component}",
            f"Fejl: {error_message}",
        ]
        if error_type:
            lines.append(f"Type: {error_type}")
        lines.append(f"Recoverable: {'Ja' if is_recoverable else 'NEJ'}")
        if stack_trace:
            lines.append(f"\nStack trace:\n{stack_trace[:500]}")

        severity = "WARNING" if is_recoverable else "CRITICAL"

        return self._notifier.send(
            severity=severity,
            title=title,
            message="\n".join(lines),
            category=SYSTEM_ERROR,
        )

    # ── Tax Warning ────────────────────────────────────────────

    def send_tax_warning(
        self,
        warning_type: str,
        realized_gains: float,
        tax_threshold: float,
        estimated_tax: float,
        message: str = "",
    ) -> int:
        """
        Send skatteadvarsel (progressionsgrænse, tab-fradrag osv.).

        Args:
            warning_type: Type advarsel (f.eks. "progressionsgrænse", "tab-fradrag").
            realized_gains: Realiserede gevinster i DKK.
            tax_threshold: Relevant skattegrænse.
            estimated_tax: Estimeret skattebetaling.
            message: Ekstra besked.

        Returns:
            Antal kanaler der modtog notifikationen.
        """
        if not self._config.on_tax_warning:
            logger.debug("[trading_notifier] Tax warning deaktiveret")
            return 0

        title = f"SKATTEADVARSEL: {warning_type}"
        lines = [
            f"Skatteadvarsel: {warning_type}",
            f"Realiserede gevinster: {realized_gains:,.0f} DKK",
            f"Skattegrænse: {tax_threshold:,.0f} DKK",
            f"Estimeret skat: {estimated_tax:,.0f} DKK",
        ]
        if message:
            lines.append(f"\n{message}")
        lines.append("\n⚠️ Vejledende beregning – verificér med revisor.")

        severity = "WARNING" if realized_gains < tax_threshold else "CRITICAL"

        return self._notifier.send(
            severity=severity,
            title=title,
            message="\n".join(lines),
            category=TAX_WARNING,
        )

    # ── HTML getters for new alerts ────────────────────────────

    @staticmethod
    def get_regime_shift_html(
        from_regime: str,
        to_regime: str,
        confidence: float,
        recommended_action: str = "",
    ) -> str:
        """Returnér HTML for regime-skift alert."""
        return _regime_shift_html(from_regime, to_regime, confidence, recommended_action)

    @staticmethod
    def get_circuit_breaker_html(
        level: str,
        reason: str,
        current_drawdown_pct: float = 0.0,
        actions_taken: list[str] | None = None,
    ) -> str:
        """Returnér HTML for circuit breaker alert."""
        return _circuit_breaker_html(level, reason, current_drawdown_pct, actions_taken)

    @staticmethod
    def get_weekly_summary_html(
        week_pnl: float,
        week_pnl_pct: float,
        total_equity: float,
        trades_count: int,
        win_rate: float,
        sharpe_ratio: float = 0.0,
        max_drawdown_pct: float = 0.0,
        regime: str = "",
    ) -> str:
        """Returnér HTML for ugentlig opsummering."""
        return _weekly_summary_html(
            week_pnl, week_pnl_pct, total_equity, trades_count,
            win_rate, sharpe_ratio, max_drawdown_pct, regime,
        )

    @staticmethod
    def get_strategy_decay_html(
        strategy_name: str,
        current_sharpe: float,
        previous_sharpe: float,
        win_rate: float,
        previous_win_rate: float,
        recommendation: str = "",
    ) -> str:
        """Returnér HTML for strategi-forfald alert."""
        return _strategy_decay_html(
            strategy_name, current_sharpe, previous_sharpe,
            win_rate, previous_win_rate, recommendation,
        )

    @staticmethod
    def get_system_error_html(
        component: str,
        error_message: str,
        error_type: str = "",
        is_recoverable: bool = True,
    ) -> str:
        """Returnér HTML for systemfejl alert."""
        return _system_error_html(component, error_message, error_type, is_recoverable)

    @staticmethod
    def get_tax_warning_html(
        warning_type: str,
        realized_gains: float,
        tax_threshold: float,
        estimated_tax: float,
    ) -> str:
        """Returnér HTML for skatteadvarsel."""
        return _tax_warning_html(warning_type, realized_gains, tax_threshold, estimated_tax)
