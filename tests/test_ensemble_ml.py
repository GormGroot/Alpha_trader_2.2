"""
Tests for EnsembleMLStrategy – ensemble ML-baseret handelsstrategi.

Tester:
  - Feature engineering (22 features)
  - Ensemble træning (3 modeller)
  - Majority voting
  - Analyse/prediktion
  - Auto-retræning
  - Model health check
  - Sammenligning med regelbaserede strategier
  - Forklaring
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.base_strategy import BaseStrategy, Signal, StrategyResult
from src.strategy.ensemble_ml_strategy import (
    EnsembleMLStrategy,
    EnsembleMetrics,
    EnsembleVote,
    ModelPerformanceRecord,
    RetrainResult,
    ModelHealth,
    ComparisonReport,
    StrategyComparisonResult,
    build_ensemble_features,
    ENSEMBLE_FEATURE_COLUMNS,
    EXTRA_FEATURE_COLUMNS,
)
from src.strategy.ml_strategy import FEATURE_COLUMNS as BASE_FEATURE_COLUMNS


# ── Helpers ──────────────────────────────────────────────────

def _make_df(
    n: int = 800,
    trend: float = 0.0003,
    noise: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """Generér syntetisk OHLCV-data."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range(end="2026-03-15", periods=n, freq="D")

    log_returns = trend + noise * rng.randn(n)
    log_returns[0] = 0
    prices = 100 * np.exp(np.cumsum(log_returns))

    high = prices * (1 + rng.uniform(0, 0.02, n))
    low = prices * (1 - rng.uniform(0, 0.02, n))
    volume = rng.randint(1_000_000, 10_000_000, n).astype(float)

    return pd.DataFrame({
        "Open": prices * (1 + rng.uniform(-0.005, 0.005, n)),
        "High": high,
        "Low": low,
        "Close": prices,
        "Volume": volume,
    }, index=dates)


def _make_uptrend(n: int = 800) -> pd.DataFrame:
    return _make_df(n=n, trend=0.001, noise=0.01, seed=42)


def _make_downtrend(n: int = 800) -> pd.DataFrame:
    return _make_df(n=n, trend=-0.001, noise=0.01, seed=42)


def _trained_ensemble(df=None) -> EnsembleMLStrategy:
    """Returnér en trænet ensemble-model."""
    model = EnsembleMLStrategy(
        threshold=0.005,
        confidence_min=0.50,
        horizon=5,
    )
    model.train(df or _make_df(800))
    return model


# ── Test Feature Engineering ─────────────────────────────────

class TestBuildEnsembleFeatures:
    def test_returns_all_columns(self):
        df = _make_df()
        feat = build_ensemble_features(df)
        for col in ENSEMBLE_FEATURE_COLUMNS:
            assert col in feat.columns, f"Mangler kolonne: {col}"

    def test_feature_count(self):
        assert len(ENSEMBLE_FEATURE_COLUMNS) == 22
        assert len(EXTRA_FEATURE_COLUMNS) == 6

    def test_includes_base_features(self):
        for col in BASE_FEATURE_COLUMNS:
            assert col in ENSEMBLE_FEATURE_COLUMNS

    def test_regime_score_default_zero(self):
        df = _make_df()
        feat = build_ensemble_features(df)
        assert (feat["regime_score"] == 0.0).all()

    def test_regime_score_custom(self):
        df = _make_df()
        feat = build_ensemble_features(df, regime_score=0.8)
        assert (feat["regime_score"] == 0.8).all()

    def test_stochastic_bounded(self):
        df = _make_df()
        feat = build_ensemble_features(df)
        stoch = feat["stoch_k"].dropna()
        assert stoch.min() >= 0
        assert stoch.max() <= 100

    def test_atr_pct_positive(self):
        df = _make_df()
        feat = build_ensemble_features(df)
        atr = feat["atr_pct"].dropna()
        assert (atr >= 0).all()

    def test_roc_10_present(self):
        df = _make_df()
        feat = build_ensemble_features(df)
        assert not feat["roc_10"].dropna().empty

    def test_volatility_ratio_around_one(self):
        df = _make_df()
        feat = build_ensemble_features(df)
        vr = feat["volatility_ratio"].dropna()
        assert vr.median() > 0.5
        assert vr.median() < 2.0


# ── Test Ensemble Training ───────────────────────────────────

class TestEnsembleTraining:
    def test_train_returns_metrics(self):
        model = EnsembleMLStrategy(threshold=0.005, confidence_min=0.50)
        metrics = model.train(_make_df(800))
        assert isinstance(metrics, EnsembleMetrics)
        assert 0 <= metrics.ensemble_accuracy <= 1
        assert 0 <= metrics.ensemble_auc <= 1
        assert metrics.n_train > 0
        assert metrics.n_test > 0

    def test_is_trained_flag(self):
        model = EnsembleMLStrategy()
        assert model.is_trained is False
        model.train(_make_df(800))
        assert model.is_trained is True

    def test_three_models_trained(self):
        model = EnsembleMLStrategy()
        model.train(_make_df(800))
        assert model._rf_model is not None
        assert model._xgb_model is not None
        assert model._lr_model is not None
        assert model._lr_scaler is not None

    def test_per_model_metrics(self):
        model = EnsembleMLStrategy(threshold=0.005)
        metrics = model.train(_make_df(800))
        # Alle per-model metrics har værdier
        assert metrics.rf_metrics.n_train > 0
        assert metrics.xgb_metrics.n_train > 0
        assert metrics.lr_metrics.n_train > 0

    def test_feature_importance_complete(self):
        model = EnsembleMLStrategy()
        metrics = model.train(_make_df(800))
        assert len(metrics.feature_importance) == len(ENSEMBLE_FEATURE_COLUMNS)

    def test_agreement_rate(self):
        model = EnsembleMLStrategy()
        metrics = model.train(_make_df(800))
        assert 0 <= metrics.agreement_rate <= 1

    def test_performance_history_updated(self):
        model = EnsembleMLStrategy()
        model.train(_make_df(800))
        assert len(model.performance_history) == 1

    def test_too_little_data_raises(self):
        model = EnsembleMLStrategy()
        with pytest.raises(ValueError, match="For lidt data"):
            model.train(_make_df(50))

    def test_train_with_regime_score(self):
        model = EnsembleMLStrategy(threshold=0.005)
        metrics = model.train(_make_df(800), regime_score=0.5)
        assert metrics.n_train > 0

    def test_metrics_repr(self):
        model = EnsembleMLStrategy()
        metrics = model.train(_make_df(800))
        s = repr(metrics)
        assert "acc=" in s
        assert "agreement=" in s

    def test_train_periods_set(self):
        model = EnsembleMLStrategy()
        metrics = model.train(_make_df(800))
        assert "→" in metrics.train_period
        assert "→" in metrics.test_period


# ── Test Majority Voting ─────────────────────────────────────

class TestMajorityVoting:
    def _vote(self, *predictions):
        """Helper: kald _majority_vote via en instans med min_agreement=2."""
        model = EnsembleMLStrategy(min_agreement=2)
        return model._majority_vote(*predictions)

    def test_all_agree_buy(self):
        result = self._vote(
            np.array([1, 1, 1]),
            np.array([1, 1, 1]),
            np.array([1, 1, 1]),
        )
        assert (result == 1).all()

    def test_all_agree_sell(self):
        result = self._vote(
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
            np.array([0, 0, 0]),
        )
        assert (result == 0).all()

    def test_two_of_three_buy(self):
        result = self._vote(
            np.array([1]),
            np.array([1]),
            np.array([0]),
        )
        assert result[0] == 1

    def test_two_of_three_sell(self):
        result = self._vote(
            np.array([0]),
            np.array([0]),
            np.array([1]),
        )
        assert result[0] == 0

    def test_mixed_votes(self):
        result = self._vote(
            np.array([1, 0, 1, 0]),
            np.array([1, 0, 0, 1]),
            np.array([0, 1, 1, 0]),
        )
        assert result[0] == 1  # 2 buy
        assert result[1] == 0  # 2 sell
        assert result[2] == 1  # 2 buy
        assert result[3] == 0  # 2 sell


# ── Test Analyse ─────────────────────────────────────────────

class TestEnsembleAnalyze:
    def test_untrained_returns_hold(self):
        model = EnsembleMLStrategy()
        result = model.analyze(_make_df(300))
        assert result.signal == Signal.HOLD
        assert "ikke trænet" in result.reason.lower()

    def test_too_little_data_returns_hold(self):
        model = _trained_ensemble()
        result = model.analyze(_make_df(100))
        assert result.signal == Signal.HOLD

    def test_returns_strategy_result(self):
        model = _trained_ensemble()
        result = model.analyze(_make_df(400))
        assert isinstance(result, StrategyResult)
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_confidence_range(self):
        model = _trained_ensemble()
        result = model.analyze(_make_df(400))
        assert 0 <= result.confidence <= 100

    def test_reason_mentions_ensemble(self):
        model = _trained_ensemble()
        result = model.analyze(_make_df(400))
        assert "Ensemble" in result.reason or "ensemble" in result.reason.lower()

    def test_analyze_with_regime(self):
        model = _trained_ensemble()
        result = model.analyze(_make_df(400), regime_score=0.8)
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)

    def test_name_property(self):
        model = EnsembleMLStrategy(horizon=5, threshold=0.01)
        assert "Ensemble" in model.name
        assert "h=5d" in model.name
        assert "RF+XGB+LR" in model.name

    def test_inherits_base_strategy(self):
        model = EnsembleMLStrategy()
        assert isinstance(model, BaseStrategy)


# ── Test Scale Confidence ────────────────────────────────────

class TestScaleConfidence:
    def test_min_gives_50(self):
        model = EnsembleMLStrategy(confidence_min=0.55)
        assert model._scale_confidence(0.55) == pytest.approx(50.0)

    def test_max_gives_95(self):
        model = EnsembleMLStrategy(confidence_min=0.55)
        assert model._scale_confidence(1.0) == pytest.approx(95.0)

    def test_below_min_clamps(self):
        model = EnsembleMLStrategy(confidence_min=0.55)
        assert model._scale_confidence(0.40) == pytest.approx(50.0)

    def test_monotonic(self):
        model = EnsembleMLStrategy(confidence_min=0.50)
        prev = 0
        for p in [0.50, 0.60, 0.70, 0.80, 0.90, 1.0]:
            c = model._scale_confidence(p)
            assert c >= prev
            prev = c


# ── Test EnsembleVote ────────────────────────────────────────

class TestEnsembleVote:
    def test_get_votes(self):
        model = _trained_ensemble()
        feat_df = build_ensemble_features(_make_df(400))
        latest = feat_df[ENSEMBLE_FEATURE_COLUMNS].iloc[[-1]].fillna(0)
        votes = model._get_votes(latest)
        assert len(votes) == 3
        assert votes[0].model_name == "RF"
        assert votes[1].model_name == "XGB"
        assert votes[2].model_name == "LR"
        for v in votes:
            assert 0 <= v.probability <= 1
            assert v.prediction in (0, 1)


# ── Test Auto-Retræning ─────────────────────────────────────

class TestAutoRetrain:
    def test_needs_retrain_initially(self):
        model = EnsembleMLStrategy()
        assert model.needs_retrain is True

    def test_needs_retrain_false_after_train(self):
        model = _trained_ensemble()
        assert model.needs_retrain is False

    def test_days_since_train(self):
        model = _trained_ensemble()
        assert model.days_since_train == 0

    def test_retrain_not_needed(self):
        model = _trained_ensemble()
        result = model.retrain(_make_df(800))
        assert result is None  # Ikke nødvendig

    def test_force_retrain(self):
        model = _trained_ensemble()
        result = model.retrain(_make_df(800), force=True)
        assert isinstance(result, RetrainResult)
        assert result.timestamp != ""

    def test_retrain_tracks_history(self):
        model = _trained_ensemble()
        model.retrain(_make_df(800), force=True)
        assert len(model.performance_history) == 2  # Initial + retrain

    def test_retrain_result_has_comparison(self):
        model = _trained_ensemble()
        result = model.retrain(_make_df(800), force=True)
        assert result.previous_auc >= 0
        assert result.new_auc >= 0
        assert isinstance(result.improved, bool)

    def test_retrain_reason(self):
        model = _trained_ensemble()
        result = model.retrain(_make_df(800), force=True)
        assert "Manuel" in result.retrain_reason


# ── Test Model Health ────────────────────────────────────────

class TestModelHealth:
    def test_untrained_unhealthy(self):
        model = EnsembleMLStrategy()
        health = model.check_model_health()
        assert health.is_healthy is False
        all_warnings = " ".join(w.lower() for w in health.warnings)
        assert "ikke trænet" in all_warnings

    def test_freshly_trained_healthy(self):
        model = _trained_ensemble()
        health = model.check_model_health()
        # Kan være healthy eller ej afhængig af metrics
        assert isinstance(health.is_healthy, bool)
        assert health.days_since_train == 0
        assert health.needs_retrain is False

    def test_health_has_recommendation(self):
        model = _trained_ensemble()
        health = model.check_model_health()
        assert len(health.recommendation) > 0

    def test_trend_analysis_with_history(self):
        model = _trained_ensemble()
        # Simulér 3 retræninger
        for _ in range(2):
            model.retrain(_make_df(800, seed=np.random.randint(100)), force=True)
        health = model.check_model_health()
        assert health.accuracy_trend in ("stable", "improving", "degrading")


# ── Test Feature Importance ──────────────────────────────────

class TestFeatureImportance:
    def test_top_features(self):
        model = _trained_ensemble()
        top = model.top_features(5)
        assert len(top) == 5
        # Sorteret faldende
        for i in range(len(top) - 1):
            assert top[i][1] >= top[i + 1][1]

    def test_top_features_untrained(self):
        model = EnsembleMLStrategy()
        assert model.top_features() == []

    def test_feature_importance_by_model(self):
        model = _trained_ensemble()
        by_model = model.feature_importance_by_model()
        assert "Random Forest" in by_model
        assert "XGBoost" in by_model
        assert "Logistic Regression" in by_model
        for model_name, importances in by_model.items():
            assert len(importances) == len(ENSEMBLE_FEATURE_COLUMNS)


# ── Test Sammenligning ───────────────────────────────────────

class TestComparison:
    def test_compare_with_buy_hold(self):
        model = _trained_ensemble()
        report = model.compare_with_rules(_make_df(800))
        assert isinstance(report, ComparisonReport)
        assert report.test_period != ""
        assert report.winner != ""
        # Buy & Hold altid inkluderet
        bh = [r for r in report.rule_results if "Buy" in r.strategy_name]
        assert len(bh) == 1

    def test_compare_with_rule_strategy(self):
        from src.strategy.rsi_strategy import RSIStrategy
        model = _trained_ensemble()
        report = model.compare_with_rules(
            _make_df(800),
            rule_strategies=[RSIStrategy()],
        )
        assert len(report.rule_results) == 2  # RSI + Buy&Hold

    def test_ensemble_result_in_report(self):
        model = _trained_ensemble()
        report = model.compare_with_rules(_make_df(800))
        er = report.ensemble_result
        assert "Ensemble" in er.strategy_name
        assert isinstance(er.total_return, float)
        assert isinstance(er.sharpe_ratio, float)

    def test_comparison_untrained_raises(self):
        model = EnsembleMLStrategy()
        with pytest.raises(RuntimeError, match="ikke trænet"):
            model.compare_with_rules(_make_df(800))

    def test_summary_text(self):
        model = _trained_ensemble()
        report = model.compare_with_rules(_make_df(800))
        assert "Ensemble" in report.summary
        assert "Vinder" in report.summary


# ── Test Explain ─────────────────────────────────────────────

class TestExplain:
    def test_untrained_explain(self):
        model = EnsembleMLStrategy()
        text = model.explain()
        assert "ikke trænet" in text.lower()

    def test_trained_explain_sections(self):
        model = _trained_ensemble()
        text = model.explain()
        assert "HVAD GØR MODELLEN?" in text
        assert "PERFORMANCE" in text
        assert "Random Forest" in text
        assert "XGBoost" in text
        assert "Log. Regression" in text
        assert "TOP FEATURES" in text
        assert "MODEL-SUNDHED" in text
        assert "VURDERING" in text
        assert "VIGTIGT" in text

    def test_explain_has_metrics(self):
        model = _trained_ensemble()
        text = model.explain()
        assert "Accuracy" in text or "acc" in text.lower()
        assert "AUC" in text

    def test_print_explanation(self, capsys):
        model = _trained_ensemble()
        model.print_explanation()
        captured = capsys.readouterr()
        assert "ENSEMBLE ML MODEL" in captured.out


# ── Test Sharpe & Max Drawdown ───────────────────────────────

class TestMetricCalculations:
    def test_sharpe_positive_returns(self):
        rng = np.random.RandomState(42)
        returns = 0.001 + 0.01 * rng.randn(252)
        sharpe = EnsembleMLStrategy._calc_sharpe(returns)
        assert sharpe > 0

    def test_sharpe_empty(self):
        assert EnsembleMLStrategy._calc_sharpe(np.array([])) == 0.0

    def test_sharpe_constant(self):
        assert EnsembleMLStrategy._calc_sharpe(np.array([0.01, 0.01, 0.01])) == 0.0

    def test_max_drawdown_positive(self):
        returns = np.array([0.01, -0.05, -0.03, 0.02, 0.01])
        dd = EnsembleMLStrategy._calc_max_drawdown(returns)
        assert dd > 0

    def test_max_drawdown_empty(self):
        assert EnsembleMLStrategy._calc_max_drawdown(np.array([])) == 0.0

    def test_max_drawdown_all_positive(self):
        returns = np.array([0.01, 0.02, 0.01, 0.03])
        dd = EnsembleMLStrategy._calc_max_drawdown(returns)
        assert dd == 0.0  # Ingen drawdown


# ── Test Dataklasser ─────────────────────────────────────────

class TestDataclasses:
    def test_ensemble_vote(self):
        v = EnsembleVote("RF", 1, 0.65, 70.0)
        assert v.model_name == "RF"
        assert v.prediction == 1
        assert v.probability == 0.65

    def test_model_performance_record(self):
        r = ModelPerformanceRecord(
            timestamp="2026-03-15T10:00:00",
            accuracy=0.55, f1=0.52, auc=0.58,
            sharpe=1.2, n_test=126,
            train_period="2023-01-01 → 2025-09-15",
            test_period="2025-09-16 → 2026-03-15",
        )
        assert r.accuracy == 0.55
        assert r.n_test == 126

    def test_strategy_comparison_result(self):
        r = StrategyComparisonResult(
            strategy_name="Buy & Hold",
            total_return=0.15,
            sharpe_ratio=1.0,
            win_rate=0.53,
            n_trades=1,
            max_drawdown=0.08,
        )
        assert r.strategy_name == "Buy & Hold"

    def test_model_health_dataclass(self):
        h = ModelHealth(
            is_healthy=True,
            days_since_train=5,
            needs_retrain=False,
            accuracy_trend="stable",
            sharpe_trend="stable",
            warnings=[],
            recommendation="Model er sund.",
        )
        assert h.is_healthy is True


# ── Test Integration ─────────────────────────────────────────

class TestIntegration:
    def test_works_in_combined_strategy(self):
        """Ensemble kan bruges i CombinedStrategy."""
        from src.strategy.combined_strategy import CombinedStrategy
        from src.strategy.rsi_strategy import RSIStrategy

        ensemble = _trained_ensemble()
        combined = CombinedStrategy(
            strategies=[(ensemble, 1.0), (RSIStrategy(), 1.0)],
            min_agreement=1,
        )
        result = combined.analyze(_make_df(400))
        assert isinstance(result, StrategyResult)

    def test_position_sizing(self):
        """Position sizing fra BaseStrategy virker."""
        model = _trained_ensemble()
        result = model.analyze(_make_df(400))
        if result.signal != Signal.HOLD:
            size = model.get_position_size(result, 100_000)
            assert size > 0

    def test_full_workflow(self):
        """Fuld workflow: train → analyze → retrain → health check."""
        df = _make_df(800)
        model = EnsembleMLStrategy(
            threshold=0.005,
            confidence_min=0.50,
            horizon=5,
        )

        # 1. Træn
        metrics = model.train(df)
        assert model.is_trained

        # 2. Analysér
        result = model.analyze(df)
        assert result.signal in (Signal.BUY, Signal.SELL, Signal.HOLD)

        # 3. Retræn (force)
        retrain = model.retrain(df, force=True)
        assert retrain is not None

        # 4. Health check
        health = model.check_model_health()
        assert isinstance(health, ModelHealth)

        # 5. Top features
        top = model.top_features(5)
        assert len(top) == 5

        # 6. Forklaring
        text = model.explain()
        assert len(text) > 100

    def test_validate_data(self):
        model = EnsembleMLStrategy()
        assert model.validate_data(_make_df(300), 200) is True
        assert model.validate_data(_make_df(100), 200) is False
