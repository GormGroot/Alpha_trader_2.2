#!/usr/bin/env python3
"""
Phase A7: Parameter sweep til at finde profitable strategi-konfigurationer.

Test 6 strategi-varianter på 30 store-cap symboler over 2 år.
Kriterier for "live-klar": profit_factor > 1.20 OG win_rate > 40%
                         OG max_drawdown < 15%.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.strategy.sma_crossover import SMACrossoverStrategy
from src.strategy.rsi_strategy import RSIStrategy
from src.strategy.combined_strategy import CombinedStrategy
from src.backtest.backtester import Backtester


# 30 store-cap US symboler + nogle ETF'er + krypto for diversitet
SYMBOLS = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Finance
    "JPM", "V", "MA", "BAC", "WFC",
    # Healthcare
    "UNH", "JNJ", "PFE", "MRK", "LLY",
    # Consumer
    "HD", "WMT", "KO", "PEP", "MCD",
    # Energy + industri
    "XOM", "CVX", "CAT", "BA", "GE",
    # ETF'er
    "SPY", "QQQ", "DIA",
]

START = "2024-03-01"
END = "2026-03-01"
INITIAL_CAPITAL = 100_000

# 6 strategi-konfigurationer
CONFIGS = [
    # Baseline
    ("RSI_baseline_30/70_p14",
        RSIStrategy(period=14, oversold=30, overbought=70)),
    ("RSI_aggressive_25/75_p14",
        RSIStrategy(period=14, oversold=25, overbought=75)),
    ("RSI_loose_35/65_p14",
        RSIStrategy(period=14, oversold=35, overbought=65)),
    ("RSI_short_30/70_p7",
        RSIStrategy(period=7, oversold=30, overbought=70)),
    ("SMA_20_50",
        SMACrossoverStrategy(short_window=20, long_window=50)),
    ("SMA_50_200",
        SMACrossoverStrategy(short_window=50, long_window=200)),
]


def run_one(name: str, strategy) -> dict:
    bt = Backtester(
        strategy=strategy,
        symbols=SYMBOLS,
        start=START,
        end=END,
        initial_capital=INITIAL_CAPITAL,
        commission_pct=0.001,
        spread_pct=0.0005,
    )
    res = bt.run()
    return {
        "name": name,
        "total_return_pct": res.total_return_pct,
        "sharpe": res.sharpe_ratio,
        "win_rate": res.win_rate,
        "profit_factor": res.profit_factor,
        "max_drawdown_pct": res.max_drawdown_pct,
        "num_trades": res.num_trades,
    }


def main() -> None:
    print(f"\n{'=' * 70}")
    print(f"  PARAMETER SWEEP — {len(CONFIGS)} strategier × {len(SYMBOLS)} symboler")
    print(f"  Periode: {START} → {END}  | Kapital: ${INITIAL_CAPITAL:,}")
    print(f"{'=' * 70}\n")

    results = []
    for name, strategy in CONFIGS:
        try:
            r = run_one(name, strategy)
            results.append(r)
        except Exception as exc:
            print(f"  ❌ {name}: {exc}")

    # Sortér efter profit factor
    results.sort(key=lambda r: r["profit_factor"], reverse=True)

    print(f"\n{'=' * 70}")
    print(f"  RESULTATER (sorteret efter profit factor)")
    print(f"{'=' * 70}\n")
    print(f"  {'Strategi':<32} {'Return':>9} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'MaxDD':>7} {'Trades':>7}")
    print(f"  {'-' * 90}")
    for r in results:
        marker = "✅" if (r["profit_factor"] > 1.2 and r["win_rate"] > 40 and r["max_drawdown_pct"] < 15) else "  "
        print(
            f"{marker} {r['name']:<32} "
            f"{r['total_return_pct']:>+8.2f}% "
            f"{r['sharpe']:>+6.2f} "
            f"{r['win_rate']:>5.1f}% "
            f"{r['profit_factor']:>5.2f} "
            f"{r['max_drawdown_pct']:>6.2f}% "
            f"{r['num_trades']:>7}"
        )

    # Sammendrag
    print(f"\n{'=' * 70}")
    winners = [r for r in results
               if r["profit_factor"] > 1.2
               and r["win_rate"] > 40
               and r["max_drawdown_pct"] < 15]
    if winners:
        print(f"  ✅ {len(winners)} strategier opfylder ALLE live-kriterier:")
        for w in winners:
            print(f"     {w['name']} → PF {w['profit_factor']:.2f}, "
                  f"WR {w['win_rate']:.1f}%, DD {w['max_drawdown_pct']:.1f}%")
    else:
        print(f"  ❌ INGEN strategi opfyldte alle live-kriterier")
        print(f"     Bedste: {results[0]['name']} → PF {results[0]['profit_factor']:.2f}")
        print(f"  Anbefaling: Vej B (træn ML på Gorms maskine)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
