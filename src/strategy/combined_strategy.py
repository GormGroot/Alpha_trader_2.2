"""
Combined strategi – aggregerer signaler fra flere sub-strategier.

Regler:
  - Kun handler hvis mindst `min_agreement` strategier er enige om retning.
  - Confidence = vægtet gennemsnit af de enige strategier.
  - Modstridende signaler → HOLD.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.strategy.base_strategy import BaseStrategy, Signal, StrategyResult


class CombinedStrategy(BaseStrategy):

    def __init__(
        self,
        strategies: list[tuple[BaseStrategy, float]],
        min_agreement: int = 2,
    ) -> None:
        """
        Args:
            strategies: Liste af (strategi, vægt) tuples.
                        Vægte normaliseres automatisk.
            min_agreement: Minimum antal strategier der skal pege
                           samme vej for at handle.
        """
        if len(strategies) < 2:
            raise ValueError("CombinedStrategy kræver mindst 2 sub-strategier")

        total_weight = sum(w for _, w in strategies)
        if total_weight == 0:
            raise ValueError("CombinedStrategy: sum af vægte er 0 — mindst én strategi skal have vægt > 0")
        self.strategies = [
            (s, w / total_weight) for s, w in strategies
        ]
        self.min_agreement = min_agreement

    @property
    def name(self) -> str:
        names = [s.name for s, _ in self.strategies]
        return f"Combined({', '.join(names)})"

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        # Kør alle sub-strategier
        results: list[tuple[StrategyResult, float]] = []
        for strategy, weight in self.strategies:
            result = strategy.analyze(df)
            results.append((result, weight))
            logger.debug(f"  {strategy.name}: {result}")

        # Tæl signaler
        buy_results = [(r, w) for r, w in results if r.signal == Signal.BUY]
        sell_results = [(r, w) for r, w in results if r.signal == Signal.SELL]

        buy_count = len(buy_results)
        sell_count = len(sell_results)

        # Tjek konsensus
        if buy_count >= self.min_agreement and buy_count > sell_count:
            confidence = self._weighted_confidence(buy_results)
            return StrategyResult(
                Signal.BUY, confidence,
                f"{buy_count}/{len(self.strategies)} strategier siger BUY",
            )

        if sell_count >= self.min_agreement and sell_count > buy_count:
            confidence = self._weighted_confidence(sell_results)
            return StrategyResult(
                Signal.SELL, confidence,
                f"{sell_count}/{len(self.strategies)} strategier siger SELL",
            )

        # Ingen konsensus
        return StrategyResult(
            Signal.HOLD, 0,
            f"Ingen konsensus: {buy_count} BUY, {sell_count} SELL, "
            f"{len(results) - buy_count - sell_count} HOLD "
            f"(kræver {self.min_agreement} enige)",
        )

    def _weighted_confidence(
        self, aligned_results: list[tuple[StrategyResult, float]],
    ) -> float:
        """Beregn vægtet gennemsnit af confidence for enige strategier."""
        total_weight = sum(w for _, w in aligned_results)
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(r.confidence * w for r, w in aligned_results)
        return weighted_sum / total_weight
