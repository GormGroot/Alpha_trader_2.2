"""
ML-baseret handelsstrategi – bruger scikit-learn til at forudsige prisretning.

Modellen:
  - Gradient Boosted Trees (HistGradientBoostingClassifier)
  - Features: 14 tekniske indikatorer (RSI, MACD, Bollinger, SMA, volumen, returns)
  - Target: stiger prisen mere end en threshold næste dag? (binær klassifikation)
  - Træning: 3 års data, validering: 6 måneders out-of-sample

Hvorfor HistGradientBoosting?
  - Håndterer NaN-værdier naturligt (ingen imputation nødvendig)
  - Hurtigt at træne, selv på store datasets
  - Robust mod overfitting med tidlig stopping
  - Giver kalibrerede sandsynligheder → confidence scores
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from src.data.indicators import add_all_indicators
from src.strategy.base_strategy import BaseStrategy, Signal, StrategyResult


# ── Feature Engineering ──────────────────────────────────────

# Alle features modellen bruger
FEATURE_COLUMNS = [
    # Momentum
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    # Trend
    "SMA_20_pct",       # Pris relativ til SMA20
    "SMA_50_pct",       # Pris relativ til SMA50
    "SMA_200_pct",      # Pris relativ til SMA200
    "SMA_cross",        # SMA20 - SMA50 som pct af pris
    # Volatilitet
    "BB_position",      # Hvor i Bollinger-båndet er prisen (0-1)
    "BB_Width",         # Bollinger-bredde (volatilitetsmål)
    # Volumen
    "Volume_Ratio",     # Volumen vs. gennemsnit
    "OBV_slope",        # OBV-retning (5 dage)
    # Returns
    "return_1d",        # 1-dags return
    "return_5d",        # 5-dags return
    "return_20d",       # 20-dags return
    "volatility_20d",   # 20-dags realiseret volatilitet
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Byg feature-matrix fra OHLCV-data.

    Tilføjer tekniske indikatorer og beregner afledte features.
    Returnerer DataFrame med FEATURE_COLUMNS som kolonner.
    """
    df = df.copy()

    # Tilføj alle standard-indikatorer
    if "RSI" not in df.columns:
        add_all_indicators(df)

    close = df["Close"]

    # Pris relativ til SMA'er (i procent)
    for w in [20, 50, 200]:
        col = f"SMA_{w}"
        if col in df.columns:
            df[f"{col}_pct"] = (close - df[col]) / df[col]
        else:
            df[f"{col}_pct"] = np.nan

    # SMA cross: afstand mellem kort og lang SMA
    if "SMA_20" in df.columns and "SMA_50" in df.columns:
        df["SMA_cross"] = (df["SMA_20"] - df["SMA_50"]) / close
    else:
        df["SMA_cross"] = np.nan

    # Bollinger position: 0 = nedre bånd, 1 = øvre bånd
    if "BB_Upper" in df.columns and "BB_Lower" in df.columns:
        bb_range = df["BB_Upper"] - df["BB_Lower"]
        df["BB_position"] = np.where(
            bb_range > 0,
            (close - df["BB_Lower"]) / bb_range,
            0.5,
        )
    else:
        df["BB_position"] = np.nan

    # OBV slope (normaliseret hældning over 5 dage)
    if "OBV" in df.columns:
        obv_pct = df["OBV"].pct_change(5)
        df["OBV_slope"] = obv_pct.clip(-1, 1)  # Begræns til [-1, 1]
    else:
        df["OBV_slope"] = np.nan

    # Returns
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)

    # Realiseret volatilitet (annualiseret)
    df["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    return df


def build_target(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
    """
    Byg binært target: stiger prisen mere end `threshold` over `horizon` dage?

    Args:
        df: DataFrame med "Close"-kolonne.
        horizon: Antal dage frem.
        threshold: Minimum stigning for at tælle som positiv (0.0 = enhver stigning).

    Returns:
        Series med 1 (stiger) eller 0 (falder/flad), NaN i slutningen.
    """
    future_return = df["Close"].pct_change(horizon).shift(-horizon)
    return (future_return > threshold).astype(float).where(future_return.notna())


# ── Model Metrics ────────────────────────────────────────────

@dataclass
class MLMetrics:
    """Resultat af model-evaluering."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    auc_roc: float = 0.0
    n_train: int = 0
    n_test: int = 0
    feature_importance: dict[str, float] = field(default_factory=dict)
    train_period: str = ""
    test_period: str = ""

    def __repr__(self) -> str:
        return (
            f"MLMetrics(accuracy={self.accuracy:.1%}, precision={self.precision:.1%}, "
            f"recall={self.recall:.1%}, f1={self.f1:.1%}, AUC={self.auc_roc:.3f}, "
            f"train={self.n_train}, test={self.n_test})"
        )


@dataclass
class BacktestComparison:
    """Sammenligning mellem ML og regelbaserede strategier."""
    ml_return: float = 0.0
    buy_hold_return: float = 0.0
    ml_trades: int = 0
    ml_win_rate: float = 0.0
    ml_sharpe: float = 0.0
    buy_hold_sharpe: float = 0.0
    test_period: str = ""


# ── ML Strategy ──────────────────────────────────────────────

class MLStrategy(BaseStrategy):
    """
    Machine Learning handelsstrategi.

    Bruger HistGradientBoostingClassifier til at forudsige om prisen
    stiger eller falder næste dag. Modellen trænes på tekniske indikatorer
    og returnerer et BUY/SELL/HOLD signal med confidence baseret på
    modellens sandsynlighedsestimat.

    Workflow:
        1. Kald train(df) med historisk data (mindst 250 rækker)
        2. Kald analyze(df) for at få signal for seneste data
        3. Valgfrit: evaluate(df) for out-of-sample metrics

    Parametre:
        threshold:       Min. forventet return for at handle (default 0.0)
        confidence_min:  Min. model-sandsynlighed for signal (default 0.55)
        horizon:         Forecast-horisont i dage (default 1)
        train_years:     Antal års data til træning (default 3)
        test_months:     Antal måneders out-of-sample validering (default 6)
        n_estimators:    Antal boosting-iterationer (default 200)
        max_depth:       Max dybde per træ (default 5)
        learning_rate:   Shrinkage (default 0.05)
    """

    def __init__(
        self,
        threshold: float = 0.0,
        confidence_min: float = 0.55,
        horizon: int = 1,
        train_years: int = 3,
        test_months: int = 6,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
    ) -> None:
        self.threshold = threshold
        self.confidence_min = confidence_min
        self.horizon = horizon
        self.train_years = train_years
        self.test_months = test_months

        # Model hyperparametre
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate

        # State
        self._model = None
        self._is_trained = False
        self._metrics: MLMetrics | None = None
        self._feature_columns = FEATURE_COLUMNS.copy()

    @property
    def name(self) -> str:
        return f"ML_GradientBoosting(h={self.horizon}, conf>{self.confidence_min:.0%})"

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def metrics(self) -> MLMetrics | None:
        return self._metrics

    # ── Træning ────────────────────────────────────────────────

    def train(self, df: pd.DataFrame) -> MLMetrics:
        """
        Træn modellen på historisk data med tidsopdelt train/test split.

        Args:
            df: OHLCV DataFrame med mindst (train_years + test_months) data.

        Returns:
            MLMetrics med in-sample og out-of-sample resultater.
        """
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score,
        )

        # Byg features og target
        feat_df = build_features(df)
        target = build_target(df, horizon=self.horizon, threshold=self.threshold)

        # Saml til ren matrix (drop rækker med NaN-target)
        valid_mask = target.notna()
        X = feat_df.loc[valid_mask, self._feature_columns]
        y = target.loc[valid_mask]

        if len(X) < 100:
            raise ValueError(f"For lidt data til træning: {len(X)} rækker (kræver mindst 100)")

        # Tidsopdelt split: de seneste test_months til test, resten til train
        n_test = min(int(self.test_months * 21), len(X) // 4)  # ~21 handelsdage/md
        n_train = len(X) - n_test

        X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

        logger.info(
            f"[ML] Træner model: {n_train} train / {n_test} test rækker, "
            f"{len(self._feature_columns)} features"
        )

        # Træn model
        self._model = HistGradientBoostingClassifier(
            max_iter=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )
        self._model.fit(X_train.values, y_train.values)
        self._is_trained = True

        # Evaluér på test-sæt
        y_pred = self._model.predict(X_test.values)
        y_proba = self._model.predict_proba(X_test.values)

        # Håndtér edge-case: model ser kun én klasse
        if y_proba.shape[1] == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = 0.5

        # Feature importance via permutation importance
        perm = permutation_importance(
            self._model, X_test.values, y_test.values,
            n_repeats=5, random_state=42, n_jobs=-1,
        )
        importances = dict(zip(
            self._feature_columns,
            [max(0.0, float(x)) for x in perm.importances_mean],
        ))

        # Datoer for perioder
        train_dates = X_train.index
        test_dates = X_test.index
        train_period = f"{train_dates[0].strftime('%Y-%m-%d')} → {train_dates[-1].strftime('%Y-%m-%d')}"
        test_period = f"{test_dates[0].strftime('%Y-%m-%d')} → {test_dates[-1].strftime('%Y-%m-%d')}"

        self._metrics = MLMetrics(
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1=float(f1_score(y_test, y_pred, zero_division=0)),
            auc_roc=float(auc),
            n_train=n_train,
            n_test=n_test,
            feature_importance=importances,
            train_period=train_period,
            test_period=test_period,
        )

        logger.info(f"[ML] Model trænet: {self._metrics}")
        return self._metrics

    # ── Analyse (prediktion) ──────────────────────────────────

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        """
        Analysér seneste data og returnér BUY/SELL/HOLD signal.

        Modellen forudsiger sandsynligheden for at prisen stiger.
        - P(stiger) > confidence_min → BUY
        - P(stiger) < (1 - confidence_min) → SELL
        - Ellers → HOLD

        Confidence skaleres fra [confidence_min, 1.0] → [50, 95].
        """
        if not self._is_trained:
            return StrategyResult(Signal.HOLD, 0, "Model ikke trænet – kald train() først")

        if not self.validate_data(df, 210):
            return StrategyResult(Signal.HOLD, 0, "Ikke nok data (kræver 210+ rækker)")

        # Byg features for seneste observation
        feat_df = build_features(df)
        latest = feat_df[self._feature_columns].iloc[[-1]]

        # Prediktion
        proba = self._model.predict_proba(latest.values)

        # Håndtér edge-case: model ser kun én klasse
        if proba.shape[1] < 2:
            return StrategyResult(Signal.HOLD, 0, "Model kan ikke skelne klasser")

        p_up = float(proba[0, 1])    # Sandsynlighed for stigning
        p_down = float(proba[0, 0])  # Sandsynlighed for fald

        # Bestem signal
        if p_up >= self.confidence_min:
            confidence = self._scale_confidence(p_up)
            return StrategyResult(
                Signal.BUY, confidence,
                f"ML: P(stiger)={p_up:.1%}, "
                f"features: RSI={feat_df['RSI'].iloc[-1]:.0f}, "
                f"MACD_Hist={feat_df['MACD_Hist'].iloc[-1]:.3f}",
            )

        if p_down >= self.confidence_min:
            confidence = self._scale_confidence(p_down)
            return StrategyResult(
                Signal.SELL, confidence,
                f"ML: P(falder)={p_down:.1%}, "
                f"features: RSI={feat_df['RSI'].iloc[-1]:.0f}, "
                f"MACD_Hist={feat_df['MACD_Hist'].iloc[-1]:.3f}",
            )

        return StrategyResult(
            Signal.HOLD, 0,
            f"ML: P(stiger)={p_up:.1%} – under confidence-grænsen ({self.confidence_min:.0%})",
        )

    def _scale_confidence(self, probability: float) -> float:
        """
        Skalér model-sandsynlighed til confidence 50–95.

        probability ∈ [confidence_min, 1.0] → confidence ∈ [50, 95]
        """
        # Lineær mapping
        p_range = 1.0 - self.confidence_min
        if p_range <= 0:
            return 50.0
        normalized = (probability - self.confidence_min) / p_range
        return 50.0 + normalized * 45.0

    # ── Evaluering & Backtest ─────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> BacktestComparison:
        """
        Simulér handel med ML-signaler og sammenlign med buy-and-hold.

        Bruger de seneste test_months som test-periode.
        Handler kun når confidence er over grænsen.

        Returns:
            BacktestComparison med afkast, win rate, Sharpe ratio.
        """
        if not self._is_trained:
            raise RuntimeError("Model ikke trænet – kald train() først")

        feat_df = build_features(df)
        target = build_target(df, horizon=self.horizon, threshold=self.threshold)

        valid_mask = target.notna()
        X = feat_df.loc[valid_mask, self._feature_columns]
        y = target.loc[valid_mask]
        closes = df.loc[valid_mask, "Close"]

        # Test-split (samme som train)
        n_test = min(int(self.test_months * 21), len(X) // 4)
        X_test = X.iloc[-n_test:]
        y_test = y.iloc[-n_test:]
        closes_test = closes.iloc[-n_test:]

        # ML-prædiktioner
        probas = self._model.predict_proba(X_test.values)
        if probas.shape[1] < 2:
            return BacktestComparison()

        p_ups = probas[:, 1]

        # Simulér ML-handel
        daily_returns = closes_test.pct_change().fillna(0).values
        ml_returns = []
        ml_positions = []  # 1=long, -1=short, 0=cash

        for i, p_up in enumerate(p_ups):
            if p_up >= self.confidence_min:
                ml_positions.append(1)   # Long
            elif (1 - p_up) >= self.confidence_min:
                ml_positions.append(-1)  # Short
            else:
                ml_positions.append(0)   # Cash

        # Beregn daglige returns (position fra i gælder for return i+1)
        ml_daily = []
        for i in range(1, len(ml_positions)):
            ml_daily.append(ml_positions[i - 1] * daily_returns[i])

        ml_daily = np.array(ml_daily)
        bh_daily = daily_returns[1:]  # Buy-and-hold

        # Metrics
        ml_total = float(np.prod(1 + ml_daily) - 1) if len(ml_daily) > 0 else 0.0
        bh_total = float(np.prod(1 + bh_daily) - 1) if len(bh_daily) > 0 else 0.0

        n_trades = sum(1 for i in range(1, len(ml_positions))
                       if ml_positions[i] != ml_positions[i - 1])
        ml_wins = sum(1 for r in ml_daily if r > 0)
        ml_active = sum(1 for r in ml_daily if r != 0)
        win_rate = ml_wins / ml_active if ml_active > 0 else 0.0

        ml_sharpe = self._calc_sharpe(ml_daily)
        bh_sharpe = self._calc_sharpe(bh_daily)

        test_dates = X_test.index
        test_period = f"{test_dates[0].strftime('%Y-%m-%d')} → {test_dates[-1].strftime('%Y-%m-%d')}"

        return BacktestComparison(
            ml_return=ml_total,
            buy_hold_return=bh_total,
            ml_trades=n_trades,
            ml_win_rate=win_rate,
            ml_sharpe=ml_sharpe,
            buy_hold_sharpe=bh_sharpe,
            test_period=test_period,
        )

    @staticmethod
    def _calc_sharpe(returns: np.ndarray) -> float:
        """Annualiseret Sharpe ratio."""
        if len(returns) < 2 or np.std(returns, ddof=1) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252))

    # ── Forklaring ────────────────────────────────────────────

    def explain(self) -> str:
        """
        Returnér en menneskelig forklaring af modellen.

        Inkluderer: metrics, top features, performance-sammenligning.
        """
        if not self._is_trained or self._metrics is None:
            return "Model ikke trænet endnu."

        m = self._metrics

        # Top 5 vigtigste features
        sorted_features = sorted(
            m.feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        top5 = sorted_features[:5]

        lines = [
            "=" * 60,
            "ML MODEL – FORKLARING",
            "=" * 60,
            "",
            "MODEL: HistGradientBoostingClassifier (gradient boosted trees)",
            f"  - Antal træer: {self._n_estimators} (med early stopping)",
            f"  - Max dybde: {self._max_depth}",
            f"  - Learning rate: {self._learning_rate}",
            f"  - Forecast-horisont: {self.horizon} dag(e)",
            "",
            "HVAD GØR MODELLEN?",
            "  Modellen forudsiger om en aktie stiger eller falder næste dag.",
            "  Den lærer mønstre fra 16 tekniske indikatorer:",
            "  - Momentum (RSI, MACD) → er aktien overkøbt/oversolgt?",
            "  - Trend (SMA-positioner) → er prisen over/under trend?",
            "  - Volatilitet (Bollinger Bands) → hvor usikker er markedet?",
            "  - Volumen (ratio, OBV) → er der købs-/salgspres?",
            "  - Returns (1d/5d/20d) → hvad er momentum?",
            "",
            "PERFORMANCE (out-of-sample):",
            f"  Træningsperiode:    {m.train_period}",
            f"  Testperiode:        {m.test_period}",
            f"  Træningsdata:       {m.n_train} dage",
            f"  Testdata:           {m.n_test} dage",
            "",
            f"  Accuracy:           {m.accuracy:.1%}",
            f"  Precision:          {m.precision:.1%}",
            f"  Recall:             {m.recall:.1%}",
            f"  F1-score:           {m.f1:.1%}",
            f"  AUC-ROC:            {m.auc_roc:.3f}",
            "",
            "TOP 5 VIGTIGSTE FEATURES:",
        ]

        for fname, importance in top5:
            bar = "█" * int(importance * 50)
            lines.append(f"  {fname:<20s} {importance:.3f}  {bar}")

        # Vurdering
        lines.append("")
        lines.append("VURDERING:")
        if m.auc_roc >= 0.60:
            lines.append("  ✅ Modellen viser lovende diskrimination (AUC > 0.60)")
        elif m.auc_roc >= 0.55:
            lines.append("  ⚠️  Modellen har svag, men positiv signal (AUC 0.55-0.60)")
        else:
            lines.append("  ❌ Modellen er tæt på tilfældig (AUC ≈ 0.50)")
            lines.append("     Overvej mere data, andre features eller ensemble-tilgang")

        if m.accuracy > 0.55:
            lines.append(f"  ✅ Accuracy over 55% ({m.accuracy:.1%})")
        else:
            lines.append(f"  ⚠️  Accuracy under 55% ({m.accuracy:.1%}) – brug med forsigtighed")

        lines.append("")
        lines.append("VIGTIGT:")
        lines.append("  - Historisk performance garanterer IKKE fremtidig performance")
        lines.append("  - Modellen bør bruges som ÉT signal blandt flere (se CombinedStrategy)")
        lines.append("  - Retræn regelmæssigt med nyeste data (walk-forward)")
        lines.append("=" * 60)

        return "\n".join(lines)

    def print_explanation(self) -> None:
        """Print forklaring til konsol."""
        print(self.explain())

    # ── Persistence ───────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """
        Gem trænet model til disk med joblib.

        Gemmer model + metadata (metrics, feature_columns, hyperparametre).
        Returnerer den faktiske sti filen er gemt på.

        Eksempel:
            path = model.save("models/ml_model_20260412.joblib")
        """
        import joblib

        if not self._is_trained:
            raise RuntimeError("Model ikke trænet – kald train() først")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model":            self._model,
            "feature_columns":  self._feature_columns,
            "metrics":          self._metrics,
            "horizon":          self.horizon,
            "threshold":        self.threshold,
            "confidence_min":   self.confidence_min,
            "n_estimators":     self._n_estimators,
            "max_depth":        self._max_depth,
            "learning_rate":    self._learning_rate,
            "saved_at":         datetime.utcnow().isoformat(),
            "version":          "MLStrategy_v1",
        }
        joblib.dump(payload, path, compress=3)
        logger.info(f"[ML] Model gemt: {path} ({path.stat().st_size / 1024:.0f} KB)")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MLStrategy":
        """
        Indlæs en gemt model fra disk.

        Returnerer en klar-til-brug MLStrategy instans.

        Eksempel:
            model = MLStrategy.load("models/ml_model_20260412.joblib")
            signal = model.analyze(df)
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model ikke fundet: {path}")

        payload = joblib.load(path)

        instance = cls(
            threshold=payload.get("threshold", 0.0),
            confidence_min=payload.get("confidence_min", 0.55),
            horizon=payload.get("horizon", 1),
            n_estimators=payload.get("n_estimators", 200),
            max_depth=payload.get("max_depth", 5),
            learning_rate=payload.get("learning_rate", 0.05),
        )
        instance._model = payload["model"]
        instance._feature_columns = payload["feature_columns"]
        instance._metrics = payload.get("metrics")
        instance._is_trained = True

        saved_at = payload.get("saved_at", "ukendt")
        logger.info(f"[ML] Model indlæst: {path} (gemt {saved_at})")
        return instance
