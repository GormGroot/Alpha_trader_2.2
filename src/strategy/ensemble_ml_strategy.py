"""
Ensemble ML-strategi – kombinerer 3 modeller for robuste handelssignaler.

Hvad gør den?
  Modellen forudsiger om en aktie stiger mere end 1% inden for 5 handelsdage.
  Den bruger 22 features (tekniske indikatorer + regime + ekstra momentum).
  Tre uafhængige modeller stemmer:
    1. Random Forest – fanger ikke-lineære mønstre
    2. XGBoost – state-of-the-art gradient boosting
    3. Logistic Regression – lineær baseline (med regularisering)
  Kun hvis mindst 2 af 3 modeller er enige, handles der.
  Det giver mere robuste signaler end én enkelt model.

Hvor pålidelig er den?
  - Typisk AUC 0.55–0.65 (bedre end tilfældig, men ikke perfekt)
  - En AUC på 0.60 betyder: i 60% af tilfældene rangerer modellen
    en stiger-aktie højere end en falder-aktie. Ikke fantastisk,
    men nok til at give en edge over mange handler.
  - Vigtigt: Historisk performance garanterer IKKE fremtidig performance.
  - Modellen bør bruges som ÉT input blandt flere (regime, risiko, regler).

Auto-retræning:
  - Retræn hver måned med nyeste data (walk-forward)
  - Track model-performance over tid i _performance_history
  - Advar automatisk hvis modellen forringes (Sharpe falder, accuracy dropper)

Brug:
    from src.strategy.ensemble_ml_strategy import EnsembleMLStrategy
    model = EnsembleMLStrategy()
    metrics = model.train(df)
    signal = model.analyze(df)
    print(model.explain())
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.data.indicators import add_all_indicators
from src.strategy.base_strategy import BaseStrategy, Signal, StrategyResult
from src.strategy.ml_strategy import (
    FEATURE_COLUMNS as BASE_FEATURE_COLUMNS,
    MLMetrics,
    BacktestComparison,
    build_features as base_build_features,
    build_target,
)


# ── Udvidede Features ────────────────────────────────────────

# Ekstra features ud over de 16 fra ml_strategy.py
EXTRA_FEATURE_COLUMNS = [
    # Regime-relateret
    "regime_score",       # Composite regime-score (-1 til +1)
    # Ekstra momentum
    "roc_10",             # Rate of Change 10 dage
    "stoch_k",            # Stochastic %K (14 dage)
    "stoch_d",            # Stochastic %D (3-dags SMA af %K)
    # Ekstra volatilitet
    "atr_pct",            # Average True Range som pct af pris
    "volatility_ratio",   # 5d vol / 20d vol (regime-skift indikator)
]

ENSEMBLE_FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS


def build_ensemble_features(
    df: pd.DataFrame,
    regime_score: float | None = None,
) -> pd.DataFrame:
    """
    Byg udvidet feature-matrix med 22 features.

    Inkluderer alle 16 basis-features plus:
      - regime_score: Markedets tilstand (-1=crash, +1=bull)
      - roc_10: 10-dages Rate of Change
      - stoch_k/d: Stochastic oscillator
      - atr_pct: Average True Range
      - volatility_ratio: Kort/lang volatilitet

    Args:
        df: OHLCV DataFrame.
        regime_score: Valgfri regime-score fra RegimeDetector.

    Returns:
        DataFrame med ENSEMBLE_FEATURE_COLUMNS.
    """
    df = base_build_features(df)
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Regime-score (sættes udefra eller default 0 = neutral)
    df["regime_score"] = regime_score if regime_score is not None else 0.0

    # Rate of Change (10 dage)
    df["roc_10"] = close.pct_change(10)

    # Stochastic Oscillator (%K, %D)
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    denom = high_14 - low_14
    df["stoch_k"] = np.where(denom > 0, (close - low_14) / denom * 100, 50.0)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # Average True Range (14 dage) som pct af pris
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    df["atr_pct"] = atr / close

    # Volatility ratio: kort vol / lang vol (> 1 = stigende volatilitet)
    vol_5 = close.pct_change().rolling(5).std()
    vol_20 = close.pct_change().rolling(20).std()
    df["volatility_ratio"] = np.where(vol_20 > 0, vol_5 / vol_20, 1.0)

    return df


# ── Dataklasser ──────────────────────────────────────────────

@dataclass
class EnsembleVote:
    """Stemme fra én model i ensemblet."""
    model_name: str
    prediction: int          # 1=stiger, 0=falder
    probability: float       # P(stiger)
    confidence: float        # Skaleret confidence


@dataclass
class EnsembleMetrics:
    """Samlede metrics for ensemble-modellen."""
    # Per-model metrics
    rf_metrics: MLMetrics = field(default_factory=MLMetrics)
    xgb_metrics: MLMetrics = field(default_factory=MLMetrics)
    lr_metrics: MLMetrics = field(default_factory=MLMetrics)
    # Ensemble metrics
    ensemble_accuracy: float = 0.0
    ensemble_precision: float = 0.0
    ensemble_recall: float = 0.0
    ensemble_f1: float = 0.0
    ensemble_auc: float = 0.0
    # Feature importance (gennemsnit over modeller)
    feature_importance: dict[str, float] = field(default_factory=dict)
    # Meta
    n_train: int = 0
    n_test: int = 0
    train_period: str = ""
    test_period: str = ""
    agreement_rate: float = 0.0  # Hvor ofte er 2/3 enige?

    def __repr__(self) -> str:
        return (
            f"EnsembleMetrics(acc={self.ensemble_accuracy:.1%}, "
            f"f1={self.ensemble_f1:.1%}, AUC={self.ensemble_auc:.3f}, "
            f"agreement={self.agreement_rate:.0%})"
        )


@dataclass
class ModelPerformanceRecord:
    """Registrering af model-performance over tid."""
    timestamp: str
    accuracy: float
    f1: float
    auc: float
    sharpe: float
    n_test: int
    train_period: str
    test_period: str


@dataclass
class RetrainResult:
    """Resultat af en model-retræning."""
    metrics: EnsembleMetrics
    improved: bool
    previous_auc: float
    new_auc: float
    retrain_reason: str
    timestamp: str


@dataclass
class ModelHealth:
    """Sundhedstjek for modellen."""
    is_healthy: bool
    days_since_train: int
    needs_retrain: bool
    accuracy_trend: str        # "stable", "improving", "degrading"
    sharpe_trend: str
    warnings: list[str]
    recommendation: str


@dataclass
class StrategyComparisonResult:
    """Sammenligning mellem ML-ensemble og regelbaserede strategier."""
    strategy_name: str
    total_return: float
    sharpe_ratio: float
    win_rate: float
    n_trades: int
    max_drawdown: float


@dataclass
class ComparisonReport:
    """Fuld sammenligning med regelbaserede strategier."""
    ensemble_result: StrategyComparisonResult
    rule_results: list[StrategyComparisonResult]
    test_period: str
    winner: str
    summary: str


# ── Ensemble ML Strategi ─────────────────────────────────────

class EnsembleMLStrategy(BaseStrategy):
    """
    Ensemble ML-strategi med 3 modeller og auto-retræning.

    Kombinerer:
      1. Random Forest – fanger ikke-lineære mønstre i data
      2. XGBoost – state-of-the-art gradient boosting
      3. Logistic Regression – stabil lineær baseline

    Kun handel hvis mindst 2 af 3 modeller er enige (majority voting).
    Confidence = gennemsnitlig sandsynlighed fra de enige modeller.

    Parametre:
        threshold:       Min. forventet stigning for target (default 0.01 = 1%)
        confidence_min:  Min. ensemble-sandsynlighed for signal (default 0.55)
        horizon:         Forecast-horisont i dage (default 5)
        train_years:     År til træning (default 3)
        test_months:     Måneder til OOS-validering (default 6)
        retrain_interval_days: Auto-retræn efter N dage (default 30)
        min_agreement:   Min. antal modeller der skal være enige (default 2)
    """

    def __init__(
        self,
        threshold: float = 0.01,
        confidence_min: float = 0.55,
        horizon: int = 5,
        train_years: int = 3,
        test_months: int = 6,
        retrain_interval_days: int = 30,
        min_agreement: int = 2,
        cache_dir: str = "data_cache",
    ) -> None:
        self.threshold = threshold
        self.confidence_min = confidence_min
        self.horizon = horizon
        self.train_years = train_years
        self.test_months = test_months
        self.retrain_interval_days = retrain_interval_days
        self.min_agreement = min_agreement
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Modeller
        self._rf_model = None    # Random Forest
        self._xgb_model = None   # XGBoost
        self._lr_model = None    # Logistic Regression
        self._lr_scaler = None   # StandardScaler for LR

        # State
        self._is_trained = False
        self._metrics: EnsembleMetrics | None = None
        self._train_date: datetime | None = None
        self._performance_history: list[ModelPerformanceRecord] = []
        self._feature_columns = ENSEMBLE_FEATURE_COLUMNS.copy()

    @property
    def name(self) -> str:
        return (
            f"Ensemble_ML(RF+XGB+LR, h={self.horizon}d, "
            f">{self.threshold:.0%}, agree≥{self.min_agreement})"
        )

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def metrics(self) -> EnsembleMetrics | None:
        return self._metrics

    @property
    def performance_history(self) -> list[ModelPerformanceRecord]:
        return self._performance_history.copy()

    @property
    def days_since_train(self) -> int:
        if self._train_date is None:
            return 999
        return (datetime.now() - self._train_date).days

    @property
    def needs_retrain(self) -> bool:
        return self.days_since_train >= self.retrain_interval_days

    # ── Træning ────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        regime_score: float | None = None,
    ) -> EnsembleMetrics:
        """
        Træn alle 3 modeller på historisk data med time-split.

        Args:
            df: OHLCV DataFrame (mindst 250 rækker).
            regime_score: Valgfri regime-score for feature-engineering.

        Returns:
            EnsembleMetrics med per-model og ensemble-resultater.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score,
        )
        from sklearn.preprocessing import StandardScaler

        try:
            from xgboost import XGBClassifier
            # Test at biblioteket rent faktisk virker (libomp kan mangle)
            has_xgb = True
        except (ImportError, OSError, Exception) as exc:
            has_xgb = False
            logger.warning(f"[Ensemble] XGBoost ikke tilgængelig ({exc}) – bruger ekstra RF")

        # Byg features og target
        feat_df = build_ensemble_features(df, regime_score=regime_score)
        target = build_target(df, horizon=self.horizon, threshold=self.threshold)

        # Saml til ren matrix
        valid_mask = target.notna()
        X = feat_df.loc[valid_mask, self._feature_columns].copy()
        y = target.loc[valid_mask].copy()

        if len(X) < 100:
            raise ValueError(
                f"For lidt data til træning: {len(X)} rækker (kræver mindst 100)"
            )

        # Tidsopdelt split
        n_test = min(int(self.test_months * 21), len(X) // 4)
        n_train = len(X) - n_test

        X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

        # Erstat NaN med median (robust for alle modeller)
        self._train_medians = X_train.median()
        X_train_clean = X_train.fillna(self._train_medians)
        X_test_clean = X_test.fillna(self._train_medians)

        logger.info(
            f"[Ensemble] Træner 3 modeller: {n_train} train / {n_test} test, "
            f"{len(self._feature_columns)} features, "
            f"target: >{self.threshold:.1%} over {self.horizon}d"
        )

        # ── Model 1: Random Forest ──
        self._rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self._rf_model.fit(X_train_clean.values, y_train.values)
        rf_pred = self._rf_model.predict(X_test_clean.values)
        rf_proba = self._rf_model.predict_proba(X_test_clean.values)

        # ── Model 2: XGBoost (eller fallback RF) ──
        if has_xgb:
            self._xgb_model = XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1.0,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._xgb_model.fit(X_train_clean.values, y_train.values)
            xgb_pred = self._xgb_model.predict(X_test_clean.values)
            xgb_proba = self._xgb_model.predict_proba(X_test_clean.values)
        else:
            # Fallback: anden RF med andre parametre
            self._xgb_model = RandomForestClassifier(
                n_estimators=150, max_depth=6,
                min_samples_leaf=30, random_state=99, n_jobs=-1,
            )
            self._xgb_model.fit(X_train_clean.values, y_train.values)
            xgb_pred = self._xgb_model.predict(X_test_clean.values)
            xgb_proba = self._xgb_model.predict_proba(X_test_clean.values)

        # ── Model 3: Logistic Regression ──
        self._lr_scaler = StandardScaler()
        X_train_scaled = self._lr_scaler.fit_transform(X_train_clean.values)
        X_test_scaled = self._lr_scaler.transform(X_test_clean.values)

        self._lr_model = LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self._lr_model.fit(X_train_scaled, y_train.values)
        lr_pred = self._lr_model.predict(X_test_scaled)
        lr_proba = self._lr_model.predict_proba(X_test_scaled)

        self._is_trained = True
        self._train_date = datetime.now()

        # ── Ensemble predictions (majority voting) ──
        ensemble_pred = self._majority_vote(rf_pred, xgb_pred, lr_pred)

        # Ensemble probability (average af enige modeller)
        ensemble_proba = np.zeros(len(X_test))
        for i in range(len(X_test)):
            votes = [rf_pred[i], xgb_pred[i], lr_pred[i]]
            probas = [
                rf_proba[i, 1] if rf_proba.shape[1] > 1 else 0.5,
                xgb_proba[i, 1] if xgb_proba.shape[1] > 1 else 0.5,
                lr_proba[i, 1] if lr_proba.shape[1] > 1 else 0.5,
            ]
            majority = ensemble_pred[i]
            aligned = [p for v, p in zip(votes, probas) if v == majority]
            ensemble_proba[i] = np.mean(aligned) if aligned else 0.5

        # ── Beregn metrics ──
        def _safe_auc(y_true, y_proba):
            if y_proba.shape[1] < 2:
                return 0.5
            try:
                return float(roc_auc_score(y_true, y_proba[:, 1]))
            except ValueError:
                return 0.5

        def _model_metrics(name, y_true, y_pred, y_proba, importance_dict, train_p, test_p):
            return MLMetrics(
                accuracy=float(accuracy_score(y_true, y_pred)),
                precision=float(precision_score(y_true, y_pred, zero_division=0)),
                recall=float(recall_score(y_true, y_pred, zero_division=0)),
                f1=float(f1_score(y_true, y_pred, zero_division=0)),
                auc_roc=_safe_auc(y_true, y_proba),
                n_train=n_train,
                n_test=n_test,
                feature_importance=importance_dict,
                train_period=train_p,
                test_period=test_p,
            )

        # Datoer
        train_dates = X_train.index
        test_dates = X_test.index
        train_period = f"{train_dates[0].strftime('%Y-%m-%d')} → {train_dates[-1].strftime('%Y-%m-%d')}"
        test_period = f"{test_dates[0].strftime('%Y-%m-%d')} → {test_dates[-1].strftime('%Y-%m-%d')}"

        # Feature importance per model
        rf_importance = dict(zip(
            self._feature_columns,
            [float(x) for x in self._rf_model.feature_importances_],
        ))

        if has_xgb:
            xgb_importance = dict(zip(
                self._feature_columns,
                [float(x) for x in self._xgb_model.feature_importances_],
            ))
        else:
            # XGBoost not available — zero out its feature importance
            xgb_importance = dict.fromkeys(self._feature_columns, 0.0)

        lr_importance = dict(zip(
            self._feature_columns,
            [abs(float(x)) for x in self._lr_model.coef_[0]],
        ))

        # Gennemsnitlig feature importance (normaliseret)
        avg_importance = {}
        for feat in self._feature_columns:
            vals = [rf_importance.get(feat, 0), xgb_importance.get(feat, 0), lr_importance.get(feat, 0)]
            avg_importance[feat] = float(np.mean(vals))
        # Normalisér
        total_imp = sum(avg_importance.values()) or 1.0
        avg_importance = {k: v / total_imp for k, v in avg_importance.items()}

        # Agreement rate (unanimous only: all 3 agree)
        agreements = sum(
            1 for i in range(len(rf_pred))
            if (rf_pred[i] + xgb_pred[i] + lr_pred[i]) in (0, 3)
        )
        agreement_rate = agreements / len(rf_pred) if len(rf_pred) > 0 else 0.0

        self._metrics = EnsembleMetrics(
            rf_metrics=_model_metrics("RF", y_test, rf_pred, rf_proba, rf_importance, train_period, test_period),
            xgb_metrics=_model_metrics("XGB", y_test, xgb_pred, xgb_proba, xgb_importance, train_period, test_period),
            lr_metrics=_model_metrics("LR", y_test, lr_pred, lr_proba, lr_importance, train_period, test_period),
            ensemble_accuracy=float(accuracy_score(y_test, ensemble_pred)),
            ensemble_precision=float(precision_score(y_test, ensemble_pred, zero_division=0)),
            ensemble_recall=float(recall_score(y_test, ensemble_pred, zero_division=0)),
            ensemble_f1=float(f1_score(y_test, ensemble_pred, zero_division=0)),
            ensemble_auc=_safe_auc(y_test, np.column_stack([1 - ensemble_proba, ensemble_proba])),
            feature_importance=avg_importance,
            n_train=n_train,
            n_test=n_test,
            train_period=train_period,
            test_period=test_period,
            agreement_rate=agreement_rate,
        )

        # Gem i performance-historik
        self._performance_history.append(ModelPerformanceRecord(
            timestamp=datetime.now().isoformat(),
            accuracy=self._metrics.ensemble_accuracy,
            f1=self._metrics.ensemble_f1,
            auc=self._metrics.ensemble_auc,
            sharpe=0.0,  # Opdateres efter evaluate()
            n_test=n_test,
            train_period=train_period,
            test_period=test_period,
        ))

        logger.info(f"[Ensemble] Modeller trænet: {self._metrics}")
        return self._metrics

    # ── Majority Voting ────────────────────────────────────────

    def _majority_vote(self, *predictions) -> np.ndarray:
        """Majority voting: returner 1 hvis ≥min_agreement modeller siger 1."""
        stacked = np.column_stack(predictions)
        return (stacked.sum(axis=1) >= self.min_agreement).astype(int)

    # ── Analyse (prediktion) ──────────────────────────────────

    def analyze(
        self,
        df: pd.DataFrame,
        regime_score: float | None = None,
    ) -> StrategyResult:
        """
        Analysér seneste data med ensemble voting.

        Alle 3 modeller stemmer. Kun hvis ≥ min_agreement er enige
        OG gennemsnitlig sandsynlighed > confidence_min, gives signal.

        Note: regime_score defaults to 0.0 when not provided, matching
        the default used during training. This avoids train/inference
        feature mismatch when SignalEngine calls analyze() without
        passing regime_score.

        Args:
            df: OHLCV DataFrame.
            regime_score: Valgfri regime-score (default 0.0 = neutral).

        Returns:
            StrategyResult med BUY/SELL/HOLD.
        """
        # Ensure regime_score is always a float to avoid train/inference mismatch
        # (SignalEngine does not pass regime_score, so it arrives as None)
        if regime_score is None:
            regime_score = 0.0

        if not self._is_trained:
            return StrategyResult(Signal.HOLD, 0, "Ensemble ikke trænet – kald train() først")

        if not self.validate_data(df, 210):
            return StrategyResult(Signal.HOLD, 0, "Ikke nok data (kræver 210+ rækker)")

        # Byg features
        feat_df = build_ensemble_features(df, regime_score=regime_score)
        latest = feat_df[self._feature_columns].iloc[[-1]].copy()
        latest_clean = latest.fillna(self._train_medians if hasattr(self, '_train_medians') else 0)

        # Hent votes fra alle modeller
        votes = self._get_votes(latest_clean)

        # Tæl stemmer
        buy_votes = sum(1 for v in votes if v.prediction == 1)
        sell_votes = len(votes) - buy_votes

        # Majority
        if buy_votes >= self.min_agreement:
            aligned = [v for v in votes if v.prediction == 1]
            avg_prob = np.mean([v.probability for v in aligned])

            if avg_prob >= self.confidence_min:
                confidence = self._scale_confidence(avg_prob)
                vote_str = ", ".join(f"{v.model_name}={v.probability:.0%}" for v in votes)
                return StrategyResult(
                    Signal.BUY, confidence,
                    f"Ensemble BUY ({buy_votes}/3 enige): "
                    f"P(>{self.threshold:.0%} over {self.horizon}d)={avg_prob:.1%} "
                    f"[{vote_str}]",
                )

        if sell_votes >= self.min_agreement:
            aligned = [v for v in votes if v.prediction == 0]
            avg_prob = np.mean([1 - v.probability for v in aligned])

            if avg_prob >= self.confidence_min:
                confidence = self._scale_confidence(avg_prob)
                vote_str = ", ".join(f"{v.model_name}={v.probability:.0%}" for v in votes)
                return StrategyResult(
                    Signal.SELL, confidence,
                    f"Ensemble SELL ({sell_votes}/3 enige): "
                    f"P(falder)={avg_prob:.1%} "
                    f"[{vote_str}]",
                )

        # Ingen enighed eller under confidence-grænse
        vote_str = ", ".join(f"{v.model_name}={'UP' if v.prediction == 1 else 'DOWN'}({v.probability:.0%})" for v in votes)
        return StrategyResult(
            Signal.HOLD, 0,
            f"Ensemble HOLD: utilstrækkelig enighed eller confidence [{vote_str}]",
        )

    def _get_votes(self, X: pd.DataFrame) -> list[EnsembleVote]:
        """Hent stemmer fra alle 3 modeller."""
        votes = []

        # Random Forest
        rf_proba = self._rf_model.predict_proba(X.values)
        rf_p_up = float(rf_proba[0, 1]) if rf_proba.shape[1] > 1 else 0.5
        votes.append(EnsembleVote(
            model_name="RF",
            prediction=1 if rf_p_up >= 0.5 else 0,
            probability=rf_p_up,
            confidence=self._scale_confidence(rf_p_up) if rf_p_up >= self.confidence_min else 0,
        ))

        # XGBoost
        xgb_proba = self._xgb_model.predict_proba(X.values)
        xgb_p_up = float(xgb_proba[0, 1]) if xgb_proba.shape[1] > 1 else 0.5
        votes.append(EnsembleVote(
            model_name="XGB",
            prediction=1 if xgb_p_up >= 0.5 else 0,
            probability=xgb_p_up,
            confidence=self._scale_confidence(xgb_p_up) if xgb_p_up >= self.confidence_min else 0,
        ))

        # Logistic Regression
        X_scaled = self._lr_scaler.transform(X.values)
        lr_proba = self._lr_model.predict_proba(X_scaled)
        lr_p_up = float(lr_proba[0, 1]) if lr_proba.shape[1] > 1 else 0.5
        votes.append(EnsembleVote(
            model_name="LR",
            prediction=1 if lr_p_up >= 0.5 else 0,
            probability=lr_p_up,
            confidence=self._scale_confidence(lr_p_up) if lr_p_up >= self.confidence_min else 0,
        ))

        return votes

    def _scale_confidence(self, probability: float) -> float:
        """Skalér sandsynlighed [confidence_min, 1.0] → [50, 95]."""
        p_range = 1.0 - self.confidence_min
        if p_range <= 0:
            return 50.0
        normalized = (probability - self.confidence_min) / p_range
        normalized = max(0.0, min(1.0, normalized))
        return 50.0 + normalized * 45.0

    # ── Auto-Retræning ─────────────────────────────────────────

    def retrain(
        self,
        df: pd.DataFrame,
        regime_score: float | None = None,
        force: bool = False,
    ) -> RetrainResult | None:
        """
        Retræn modellen hvis det er tid (eller force=True).

        Sammenligner ny performance med gammel. Logger i historik.

        Args:
            df: Nyeste OHLCV data.
            regime_score: Regime-score.
            force: Tving retræning uanset tid.

        Returns:
            RetrainResult eller None hvis retræning ikke var nødvendig.
        """
        if not force and not self.needs_retrain:
            logger.debug(
                f"[Ensemble] Retræning ikke nødvendig "
                f"({self.days_since_train}d siden sidst, grænse={self.retrain_interval_days}d)"
            )
            return None

        previous_auc = self._metrics.ensemble_auc if self._metrics else 0.0
        reason = "Planlagt månedlig retræning" if not force else "Manuel retræning"

        if self.needs_retrain and not force:
            reason = f"Auto-retræning ({self.days_since_train} dage siden sidst)"

        logger.info(f"[Ensemble] Starter retræning: {reason}")
        metrics = self.train(df, regime_score=regime_score)

        improved = metrics.ensemble_auc >= previous_auc

        result = RetrainResult(
            metrics=metrics,
            improved=improved,
            previous_auc=previous_auc,
            new_auc=metrics.ensemble_auc,
            retrain_reason=reason,
            timestamp=datetime.now().isoformat(),
        )

        if not improved:
            logger.warning(
                f"[Ensemble] Model forringet efter retræning: "
                f"AUC {previous_auc:.3f} → {metrics.ensemble_auc:.3f}"
            )

        return result

    # ── Model Health Check ─────────────────────────────────────

    def check_model_health(self) -> ModelHealth:
        """
        Tjek modellens sundhed baseret på performance-historik.

        Returnerer advarsler hvis:
          - Modellen er for gammel (> retrain_interval_days)
          - AUC er faldende over de seneste 3 retræninger
          - Accuracy er under 50% (værre end tilfældig)
        """
        warnings_list: list[str] = []
        is_healthy = True
        accuracy_trend = "stable"
        sharpe_trend = "stable"

        # Alder
        days = self.days_since_train
        if days > self.retrain_interval_days * 2:
            warnings_list.append(
                f"Model er {days} dage gammel (anbefalet max {self.retrain_interval_days})"
            )
            is_healthy = False
        elif days > self.retrain_interval_days:
            warnings_list.append(f"Model bør retrænes ({days} dage gammel)")

        # Trend-analyse (kræver mindst 3 datapunkter)
        if len(self._performance_history) >= 3:
            recent = self._performance_history[-3:]
            aucs = [r.auc for r in recent]
            accs = [r.accuracy for r in recent]

            # AUC trend
            if aucs[-1] < aucs[0] - 0.03:
                accuracy_trend = "degrading"
                warnings_list.append(
                    f"AUC faldende: {aucs[0]:.3f} → {aucs[-1]:.3f} over seneste 3 retræninger"
                )
                is_healthy = False
            elif aucs[-1] > aucs[0] + 0.02:
                accuracy_trend = "improving"

            # Accuracy under tilfældig
            if accs[-1] < 0.50:
                warnings_list.append(
                    f"Accuracy ({accs[-1]:.1%}) er under 50% – model er værre end tilfældig!"
                )
                is_healthy = False

        elif self._metrics is not None:
            if self._metrics.ensemble_accuracy < 0.50:
                warnings_list.append(
                    f"Accuracy ({self._metrics.ensemble_accuracy:.1%}) er under 50%"
                )
                is_healthy = False
            if self._metrics.ensemble_auc < 0.52:
                warnings_list.append(
                    f"AUC ({self._metrics.ensemble_auc:.3f}) er tæt på tilfældig"
                )

        if not self._is_trained:
            warnings_list.append("Model er ikke trænet!")
            is_healthy = False

        # Anbefaling
        if not is_healthy:
            recommendation = "Retræn modellen med nyeste data. Overvej feature-engineering eller mere data."
        elif self.needs_retrain:
            recommendation = f"Planlagt retræning anbefalet ({days} dage siden sidst)."
        else:
            recommendation = "Model er sund. Næste retræning om " \
                             f"{self.retrain_interval_days - days} dage."

        return ModelHealth(
            is_healthy=is_healthy,
            days_since_train=days,
            needs_retrain=self.needs_retrain,
            accuracy_trend=accuracy_trend,
            sharpe_trend=sharpe_trend,
            warnings=warnings_list,
            recommendation=recommendation,
        )

    # ── Sammenligning med Regelbaserede Strategier ─────────────

    def compare_with_rules(
        self,
        df: pd.DataFrame,
        rule_strategies: list[BaseStrategy] | None = None,
        regime_score: float | None = None,
    ) -> ComparisonReport:
        """
        Sammenlign ensemble-ML med regelbaserede strategier på OOS-data.

        Kører simple long-only backtests på test-perioden.

        Args:
            df: OHLCV data (bruges til at finde test-perioden).
            rule_strategies: Liste af BaseStrategy-instanser at sammenligne med.
            regime_score: Regime-score.

        Returns:
            ComparisonReport med resultater for alle strategier.
        """
        if not self._is_trained:
            raise RuntimeError("Model ikke trænet – kald train() først")

        # Byg test-data
        feat_df = build_ensemble_features(df, regime_score=regime_score)
        n_test = min(int(self.test_months * 21), len(df) // 4)
        test_df = df.iloc[-n_test:]
        test_feat = feat_df.iloc[-n_test:]
        closes = test_df["Close"].values
        daily_returns = np.diff(closes) / closes[:-1]

        # Test-periode
        test_start = test_df.index[0].strftime("%Y-%m-%d")
        test_end = test_df.index[-1].strftime("%Y-%m-%d")
        test_period = f"{test_start} → {test_end}"

        # ── Ensemble backtest ──
        ensemble_result = self._backtest_ensemble(test_feat, daily_returns, test_period)

        # ── Regelbaserede backtests ──
        rule_results = []
        if rule_strategies:
            for strategy in rule_strategies:
                result = self._backtest_rule_strategy(strategy, test_df, daily_returns, test_period)
                rule_results.append(result)

        # Buy-and-hold (altid inkluderet)
        bh_total = float(np.prod(1 + daily_returns) - 1)
        bh_sharpe = self._calc_sharpe(daily_returns)
        rule_results.append(StrategyComparisonResult(
            strategy_name="Buy & Hold",
            total_return=bh_total,
            sharpe_ratio=bh_sharpe,
            win_rate=float(np.mean(daily_returns > 0)) if len(daily_returns) > 0 else 0.0,
            n_trades=1,
            max_drawdown=self._calc_max_drawdown(daily_returns),
        ))

        # Find vinder
        all_results = [ensemble_result] + rule_results
        winner = max(all_results, key=lambda r: r.sharpe_ratio)

        # Opsummering
        summary_lines = [
            f"Ensemble ML Sharpe: {ensemble_result.sharpe_ratio:.2f} "
            f"(return: {ensemble_result.total_return:.1%})",
        ]
        for r in rule_results:
            summary_lines.append(
                f"{r.strategy_name} Sharpe: {r.sharpe_ratio:.2f} "
                f"(return: {r.total_return:.1%})"
            )
        summary_lines.append(f"Vinder: {winner.strategy_name}")

        return ComparisonReport(
            ensemble_result=ensemble_result,
            rule_results=rule_results,
            test_period=test_period,
            winner=winner.strategy_name,
            summary="\n".join(summary_lines),
        )

    def _backtest_ensemble(
        self,
        test_feat: pd.DataFrame,
        daily_returns: np.ndarray,
        test_period: str,
    ) -> StrategyComparisonResult:
        """Simpel backtest af ensemble-signaler."""
        positions = []

        for i in range(len(test_feat)):
            row = test_feat[self._feature_columns].iloc[[i]].copy().fillna(0)
            try:
                votes = self._get_votes(row)
                buy_votes = sum(1 for v in votes if v.prediction == 1)
                if buy_votes >= self.min_agreement:
                    avg_prob = np.mean([v.probability for v in votes if v.prediction == 1])
                    positions.append(1 if avg_prob >= self.confidence_min else 0)
                elif (3 - buy_votes) >= self.min_agreement:
                    avg_prob = np.mean([1 - v.probability for v in votes if v.prediction == 0])
                    positions.append(-1 if avg_prob >= self.confidence_min else 0)
                else:
                    positions.append(0)
            except Exception:
                positions.append(0)

        # Beregn returns (position[i] earns next-day return to avoid look-ahead bias)
        ml_daily = []
        for i in range(len(daily_returns) - 1):
            ml_daily.append(positions[i] * daily_returns[i + 1])

        ml_daily = np.array(ml_daily) if ml_daily else np.array([0.0])
        total_return = float(np.prod(1 + ml_daily) - 1)

        n_trades = sum(1 for i in range(1, len(positions)) if positions[i] != positions[i - 1])
        wins = sum(1 for r in ml_daily if r > 0)
        active = sum(1 for r in ml_daily if r != 0)
        win_rate = wins / active if active > 0 else 0.0

        return StrategyComparisonResult(
            strategy_name="Ensemble ML (RF+XGB+LR)",
            total_return=total_return,
            sharpe_ratio=self._calc_sharpe(ml_daily),
            win_rate=win_rate,
            n_trades=n_trades,
            max_drawdown=self._calc_max_drawdown(ml_daily),
        )

    def _backtest_rule_strategy(
        self,
        strategy: BaseStrategy,
        test_df: pd.DataFrame,
        daily_returns: np.ndarray,
        test_period: str,
    ) -> StrategyComparisonResult:
        """Simpel backtest af en regelbaseret strategi."""
        positions = []

        for i in range(60, len(test_df)):
            window = test_df.iloc[:i + 1]
            try:
                result = strategy.analyze(window)
                if result.signal == Signal.BUY and result.confidence > 0:
                    positions.append(1)
                elif result.signal == Signal.SELL and result.confidence > 0:
                    positions.append(-1)
                else:
                    positions.append(0)
            except Exception:
                positions.append(0)

        # Pad med 0 for de første 60 dage
        positions = [0] * 60 + positions

        rule_daily = []
        for i in range(len(daily_returns)):
            rule_daily.append(positions[i] * daily_returns[i])

        rule_daily = np.array(rule_daily) if rule_daily else np.array([0.0])
        total_return = float(np.prod(1 + rule_daily) - 1)

        n_trades = sum(1 for i in range(1, len(positions)) if positions[i] != positions[i - 1])
        wins = sum(1 for r in rule_daily if r > 0)
        active = sum(1 for r in rule_daily if r != 0)

        return StrategyComparisonResult(
            strategy_name=strategy.name,
            total_return=total_return,
            sharpe_ratio=self._calc_sharpe(rule_daily),
            win_rate=wins / active if active > 0 else 0.0,
            n_trades=n_trades,
            max_drawdown=self._calc_max_drawdown(rule_daily),
        )

    @staticmethod
    def _calc_sharpe(returns: np.ndarray) -> float:
        """Annualiseret Sharpe ratio."""
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        return float(np.mean(returns) / np.std(returns) * np.sqrt(252))

    @staticmethod
    def _calc_max_drawdown(returns: np.ndarray) -> float:
        """Max drawdown fra daglige returns."""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # ── Feature Importance ─────────────────────────────────────

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """
        Returnér top N vigtigste features sorteret efter importance.

        Returns:
            Liste af (feature_name, importance) tuples.
        """
        if self._metrics is None:
            return []
        return sorted(
            self._metrics.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]

    def feature_importance_by_model(self) -> dict[str, dict[str, float]]:
        """
        Returnér feature importance per model.

        Returns:
            Dict med model_name → {feature → importance}.
        """
        if self._metrics is None:
            return {}
        return {
            "Random Forest": self._metrics.rf_metrics.feature_importance,
            "XGBoost": self._metrics.xgb_metrics.feature_importance,
            "Logistic Regression": self._metrics.lr_metrics.feature_importance,
        }

    # ── Forklaring ────────────────────────────────────────────

    def explain(self) -> str:
        """
        Returnér en menneskelig forklaring af ensemblet.

        Forklarer:
          - Hvad modellen gør (i simple termer)
          - Performance per model og samlet
          - Vigtigste features
          - Model-sundhed
          - Anbefalinger
        """
        if not self._is_trained or self._metrics is None:
            return "Ensemble model ikke trænet endnu. Kald train(df) med mindst 250 rækker data."

        m = self._metrics
        health = self.check_model_health()

        # Top features
        top = self.top_features(7)

        lines = [
            "=" * 65,
            "ENSEMBLE ML MODEL – FORKLARING",
            "=" * 65,
            "",
            "HVAD GØR MODELLEN?",
            f"  Forudsiger om en aktie stiger mere end {self.threshold:.0%} "
            f"inden for {self.horizon} handelsdage.",
            "  Tre uafhængige modeller stemmer:",
            "    1. Random Forest   – fanger komplekse mønstre (200 træer)",
            "    2. XGBoost          – state-of-the-art gradient boosting",
            "    3. Logistic Regression – stabil lineær baseline",
            f"  Kun handel hvis mindst {self.min_agreement} af 3 er enige.",
            "",
            f"  Features: {len(self._feature_columns)} indikatorer:",
            "    - Momentum: RSI, MACD, Stochastic, Rate of Change",
            "    - Trend: SMA-positioner (20/50/200), kryds-signaler",
            "    - Volatilitet: Bollinger, ATR, vol-ratio",
            "    - Volumen: Volume Ratio, OBV slope",
            "    - Returns: 1d/5d/20d afkast, realiseret volatilitet",
            "    - Regime: Markedets tilstand (bull/bear/crash)",
            "",
            "PERFORMANCE (out-of-sample):",
            f"  Træning:   {m.train_period} ({m.n_train} dage)",
            f"  Test:      {m.test_period} ({m.n_test} dage)",
            "",
            "  Model           Accuracy   F1-score   AUC-ROC",
            "  ─────────────   ────────   ────────   ───────",
            f"  Random Forest   {m.rf_metrics.accuracy:>7.1%}   {m.rf_metrics.f1:>7.1%}   {m.rf_metrics.auc_roc:>6.3f}",
            f"  XGBoost         {m.xgb_metrics.accuracy:>7.1%}   {m.xgb_metrics.f1:>7.1%}   {m.xgb_metrics.auc_roc:>6.3f}",
            f"  Log. Regression {m.lr_metrics.accuracy:>7.1%}   {m.lr_metrics.f1:>7.1%}   {m.lr_metrics.auc_roc:>6.3f}",
            f"  ─────────────   ────────   ────────   ───────",
            f"  ENSEMBLE        {m.ensemble_accuracy:>7.1%}   {m.ensemble_f1:>7.1%}   {m.ensemble_auc:>6.3f}",
            "",
            f"  Model-enighed: {m.agreement_rate:.0%} af tiden er ≥2/3 modeller enige",
            "",
            "TOP FEATURES (vigtigste indikatorer):",
        ]

        for fname, importance in top:
            bar = "█" * int(importance * 100)
            lines.append(f"  {fname:<22s} {importance:.3f}  {bar}")

        # Model sundhed
        lines.append("")
        lines.append("MODEL-SUNDHED:")
        if health.is_healthy:
            lines.append(f"  ✅ Model er sund ({health.days_since_train} dage gammel)")
        else:
            lines.append(f"  ❌ Model kræver opmærksomhed!")
        for w in health.warnings:
            lines.append(f"  ⚠️  {w}")
        lines.append(f"  Anbefaling: {health.recommendation}")

        # Vurdering
        lines.append("")
        lines.append("VURDERING:")
        if m.ensemble_auc >= 0.60:
            lines.append("  ✅ Ensemblet viser lovende diskrimination (AUC ≥ 0.60)")
        elif m.ensemble_auc >= 0.55:
            lines.append("  ⚠️  Svag men positiv signal (AUC 0.55-0.60)")
        else:
            lines.append("  ❌ Ensemblet er tæt på tilfældig (AUC < 0.55)")

        if m.ensemble_accuracy > 0.55:
            lines.append(f"  ✅ Accuracy over 55% ({m.ensemble_accuracy:.1%})")
        elif m.ensemble_accuracy > 0.50:
            lines.append(f"  ⚠️  Accuracy kun marginalt over 50% ({m.ensemble_accuracy:.1%})")
        else:
            lines.append(f"  ❌ Accuracy under 50% ({m.ensemble_accuracy:.1%})")

        if m.agreement_rate > 0.80:
            lines.append(f"  ✅ Høj model-enighed ({m.agreement_rate:.0%})")
        elif m.agreement_rate > 0.60:
            lines.append(f"  ⚠️  Moderat enighed ({m.agreement_rate:.0%})")
        else:
            lines.append(f"  ❌ Lav enighed ({m.agreement_rate:.0%}) – modellerne er uenige")

        lines.extend([
            "",
            "VIGTIGT:",
            "  - Historisk performance garanterer IKKE fremtidig performance",
            "  - Brug modellen som ÉT signal blandt flere (regime, risiko, regler)",
            f"  - Retræn hver {self.retrain_interval_days} dage med nyeste data",
            "  - Overvåg model-sundhed løbende via check_model_health()",
            "=" * 65,
        ])

        return "\n".join(lines)

    def print_explanation(self) -> None:
        """Print forklaring til konsol."""
        print(self.explain())

    # ── Persistence ───────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """
        Gem trænet ensemble model til disk med joblib.

        Gemmer alle 3 delmodeller + metadata.
        Returnerer den faktiske sti filen er gemt på.
        """
        import joblib
        from pathlib import Path as _Path

        if not self._is_trained:
            raise RuntimeError("Model ikke trænet – kald train() først")

        path = _Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "rf_model":             self._rf_model,
            "xgb_model":            self._xgb_model,
            "lr_model":             self._lr_model,
            "feature_columns":      self._feature_columns,
            "metrics":              self._metrics,
            "performance_history":  self._performance_history,
            "threshold":            self.threshold,
            "confidence_min":       self.confidence_min,
            "horizon":              self.horizon,
            "min_agreement":        self.min_agreement,
            "retrain_interval_days": self.retrain_interval_days,
            "saved_at":             datetime.utcnow().isoformat(),
            "version":              "EnsembleMLStrategy_v1",
        }
        joblib.dump(payload, path, compress=3)
        logger.info(f"[Ensemble] Model gemt: {path} ({path.stat().st_size / 1024:.0f} KB)")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EnsembleMLStrategy":
        """
        Indlæs en gemt ensemble model fra disk.

        Returnerer en klar-til-brug EnsembleMLStrategy instans.
        """
        import joblib
        from pathlib import Path as _Path

        path = _Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model ikke fundet: {path}")

        payload = joblib.load(path)

        instance = cls(
            threshold=payload.get("threshold", 0.01),
            confidence_min=payload.get("confidence_min", 0.55),
            horizon=payload.get("horizon", 5),
            min_agreement=payload.get("min_agreement", 2),
            retrain_interval_days=payload.get("retrain_interval_days", 30),
        )
        instance._rf_model  = payload["rf_model"]
        instance._xgb_model = payload["xgb_model"]
        instance._lr_model  = payload["lr_model"]
        instance._feature_columns     = payload["feature_columns"]
        instance._metrics             = payload.get("metrics")
        instance._performance_history = payload.get("performance_history", [])
        instance._is_trained = True

        saved_at = payload.get("saved_at", "ukendt")
        logger.info(f"[Ensemble] Model indlæst: {path} (gemt {saved_at})")
        return instance
