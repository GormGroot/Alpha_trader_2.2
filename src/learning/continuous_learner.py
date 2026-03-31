"""
Continuous Learning Engine
==========================
24/7 baggrundsprocess der konstant forbedrer modellerne:

1. FEEDBACK LOOP: Kobler trade-resultater tilbage til modellen
2. HISTORICAL CRISIS ANALYSIS: Studerer crashes, recovery, corona etc.
3. CONCEPT DRIFT DETECTION: Opdager når markedet ændrer karakter
4. DYNAMIC ENSEMBLE WEIGHTING: Giver bedste model mest vægt
5. INCREMENTAL LEARNING: Opdaterer modeller med nye data løbende
"""

import sqlite3
import json
import time
import threading
from collections import deque
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ─── Data Classes ───────────────────────────────────────────

@dataclass
class TradeOutcome:
    """Resultat af en enkelt trade"""
    symbol: str
    side: str           # BUY / SELL / SHORT
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float
    holding_period_hours: float
    strategy_signals: Dict[str, str]  # {strategy_name: signal}
    confidence_at_entry: float
    regime_at_entry: str
    regime_at_exit: str
    indicators_at_entry: Dict[str, float]


@dataclass
class ModelPerformance:
    """Performance tracking for en model"""
    model_name: str
    total_signals: int = 0
    correct_signals: int = 0
    accuracy_7d: float = 0.0
    accuracy_30d: float = 0.0
    accuracy_90d: float = 0.0
    avg_pnl_per_signal: float = 0.0
    best_regime: str = ""
    worst_regime: str = ""
    weight: float = 1.0  # dynamisk vægt i ensemble
    last_updated: str = ""


@dataclass
class CrisisPattern:
    """Mønster identificeret fra historiske kriser"""
    crisis_name: str
    start_date: str
    end_date: str
    pre_crisis_indicators: Dict[str, float]   # indikatorer FØR krisen
    during_crisis_indicators: Dict[str, float]
    recovery_indicators: Dict[str, float]
    max_drawdown: float
    recovery_days: int
    warning_signals: List[str]   # hvad signalerede krisen
    similarity_threshold: float  # hvor tæt skal nutiden matche


# ─── Continuous Learning Engine ─────────────────────────────

class ContinuousLearner:
    """
    24/7 learning engine der konstant forbedrer platformen.

    Kører som baggrundstråd og:
    - Evaluerer alle afsluttede handler
    - Opdaterer model-vægte baseret på performance
    - Scanner for krise-mønstre i aktuel data
    - Trigger retrain når concept drift detekteres
    - Logger alt til SQLite for historisk analyse
    """

    def __init__(self, db_path: str = "data_cache/learning.db",
                 trade_db_path: str = "data_cache/auto_trader_log.db"):
        self.db_path = db_path
        self.trade_db_path = trade_db_path
        self._running = False
        self._thread = None

        # Model performance tracking
        self.model_performance: Dict[str, ModelPerformance] = {}

        # Ensemble weights (dynamisk)
        self.ensemble_weights: Dict[str, float] = {
            "random_forest": 0.33,
            "xgboost": 0.34,
            "logistic_regression": 0.33,
        }

        # Crisis patterns (lært fra historien)
        self.crisis_patterns: List[CrisisPattern] = []

        # Concept drift tracking
        self.drift_window: deque = deque(maxlen=50)  # seneste predictions accuracy
        self.drift_threshold = 0.15  # 15% drop = drift

        # Learning stats
        self.total_lessons_learned = 0
        self.last_retrain_trigger = None
        self.current_market_similarity: Dict[str, float] = {}

        self._init_db()
        self._load_crisis_patterns()

        logger.info("🧠 Continuous Learning Engine initialiseret")

    # ─── Database Setup ─────────────────────────────────────

    def _init_db(self):
        """Opret learning database"""
        with sqlite3.connect(self.db_path) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now')),
                    symbol TEXT,
                    side TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    pnl REAL,
                    pnl_pct REAL,
                    holding_period_hours REAL,
                    confidence_at_entry REAL,
                    regime_at_entry TEXT,
                    regime_at_exit TEXT,
                    strategy_signals TEXT,
                    indicators_at_entry TEXT,
                    lesson_learned TEXT
                );

                CREATE TABLE IF NOT EXISTS model_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now')),
                    model_name TEXT,
                    accuracy_7d REAL,
                    accuracy_30d REAL,
                    accuracy_90d REAL,
                    weight REAL,
                    total_signals INTEGER,
                    correct_signals INTEGER
                );

                CREATE TABLE IF NOT EXISTS drift_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now')),
                    drift_type TEXT,
                    severity REAL,
                    old_accuracy REAL,
                    new_accuracy REAL,
                    action_taken TEXT
                );

                CREATE TABLE IF NOT EXISTS crisis_similarities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now')),
                    crisis_name TEXT,
                    similarity_score REAL,
                    matching_indicators TEXT,
                    recommendation TEXT
                );

                CREATE TABLE IF NOT EXISTS learning_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT (datetime('now')),
                    event_type TEXT,
                    details TEXT,
                    impact TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_outcomes_symbol
                    ON trade_outcomes(symbol);
                CREATE INDEX IF NOT EXISTS idx_outcomes_regime
                    ON trade_outcomes(regime_at_entry);
                CREATE INDEX IF NOT EXISTS idx_scores_model
                    ON model_scores(model_name);
            """)
        logger.debug("Learning database klar")

    # ─── Feedback Loop ──────────────────────────────────────

    def record_trade_outcome(self, outcome: TradeOutcome):
        """
        Registrer resultatet af en trade og lær af det.
        Dette er FEEDBACK LOOPET — den vigtigste del.
        """
        # Analysér hvad vi kan lære
        lesson = self._analyze_trade(outcome)

        # Gem i database
        with sqlite3.connect(self.db_path) as db:
            db.execute("""
                INSERT INTO trade_outcomes
                (symbol, side, entry_price, exit_price, entry_time, exit_time,
                 pnl, pnl_pct, holding_period_hours, confidence_at_entry,
                 regime_at_entry, regime_at_exit, strategy_signals,
                 indicators_at_entry, lesson_learned)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                outcome.symbol, outcome.side, outcome.entry_price,
                outcome.exit_price, outcome.entry_time, outcome.exit_time,
                outcome.pnl, outcome.pnl_pct, outcome.holding_period_hours,
                outcome.confidence_at_entry, outcome.regime_at_entry,
                outcome.regime_at_exit,
                json.dumps(outcome.strategy_signals),
                json.dumps(outcome.indicators_at_entry),
                lesson
            ))

        # Opdater model performance
        self._update_model_scores(outcome)

        # Check for concept drift
        self._check_drift(outcome)

        self.total_lessons_learned += 1

        status = "✅ PROFIT" if outcome.pnl > 0 else "❌ TAB"
        logger.info(
            f"🧠 Lektion #{self.total_lessons_learned}: "
            f"{status} {outcome.symbol} {outcome.side} "
            f"PnL={outcome.pnl_pct:+.1f}% | {lesson}"
        )

    def _analyze_trade(self, outcome: TradeOutcome) -> str:
        """Analysér en trade og udled en lektion"""
        lessons = []

        # Regime-skift under trade?
        if outcome.regime_at_entry != outcome.regime_at_exit:
            lessons.append(
                f"Regime skiftede fra {outcome.regime_at_entry} "
                f"til {outcome.regime_at_exit} under trade"
            )

        # Høj confidence men tab?
        if outcome.confidence_at_entry > 70 and outcome.pnl < 0:
            lessons.append(
                f"Høj confidence ({outcome.confidence_at_entry:.0f}%) "
                f"men tab — overconfident signal"
            )

        # Lav confidence men profit?
        if outcome.confidence_at_entry < 40 and outcome.pnl > 0:
            lessons.append(
                f"Lav confidence ({outcome.confidence_at_entry:.0f}%) "
                f"men profit — model undervurderede"
            )

        # Holding period analyse
        if outcome.holding_period_hours < 1 and outcome.pnl < 0:
            lessons.append("Hurtig exit med tab — muligvis for tidlig entry")
        elif outcome.holding_period_hours > 168 and outcome.pnl < 0:  # > 1 uge
            lessons.append("Lang holding med tab — stop-loss for løs")

        # PnL størrelse
        if outcome.pnl_pct < -5:
            lessons.append(f"Stort tab ({outcome.pnl_pct:.1f}%) — risk management fejl")
        elif outcome.pnl_pct > 10:
            lessons.append(f"Stor gevinst ({outcome.pnl_pct:.1f}%) — hold positionen længere?")

        return " | ".join(lessons) if lessons else "Normal trade, ingen speciel lektion"

    # ─── Model Performance Tracking ─────────────────────────

    def _update_model_scores(self, outcome: TradeOutcome):
        """Opdater score for hver model baseret på trade outcome"""
        was_profitable = outcome.pnl > 0

        for model_name, signal in outcome.strategy_signals.items():
            if model_name not in self.model_performance:
                self.model_performance[model_name] = ModelPerformance(
                    model_name=model_name
                )

            perf = self.model_performance[model_name]
            perf.total_signals += 1

            # Var modellens signal korrekt?
            signal_was_buy = signal in ("BUY", "STRONG_BUY")
            signal_was_sell = signal in ("SELL", "STRONG_SELL", "SHORT")

            correct = (
                (signal_was_buy and was_profitable and outcome.side == "BUY") or
                (signal_was_sell and was_profitable and outcome.side in ("SELL", "SHORT"))
            )

            if correct:
                perf.correct_signals += 1

            # Opdater accuracy (NB: dette er all-time accuracy, ikke 7d)
            # TODO: implementer tidsvindue-baseret accuracy
            if perf.total_signals > 0:
                perf.accuracy_7d = perf.correct_signals / perf.total_signals  # all-time, misvisende navn

            perf.avg_pnl_per_signal = (
                (perf.avg_pnl_per_signal * (perf.total_signals - 1) + outcome.pnl_pct)
                / perf.total_signals
            )

            perf.last_updated = datetime.now().isoformat()

            # Cap model_performance dict at 50 entries (prune least active)
            if len(self.model_performance) > 50:
                sorted_models = sorted(
                    self.model_performance.items(),
                    key=lambda x: x[1].total_signals,
                )
                for name, _ in sorted_models[:len(self.model_performance) - 50]:
                    del self.model_performance[name]

            # Gem til database
            with sqlite3.connect(self.db_path) as db:
                db.execute("""
                    INSERT INTO model_scores
                    (model_name, accuracy_7d, accuracy_30d, accuracy_90d,
                     weight, total_signals, correct_signals)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_name, perf.accuracy_7d, perf.accuracy_30d,
                    perf.accuracy_90d, perf.weight,
                    perf.total_signals, perf.correct_signals
                ))

    def recalculate_ensemble_weights(self):
        """
        Dynamisk ensemble-vægtning baseret på faktisk performance.
        Bedre modeller får mere vægt.
        """
        if not self.model_performance:
            return self.ensemble_weights

        # Brug accuracy som basis for vægt
        accuracies = {}
        for name, perf in self.model_performance.items():
            if perf.total_signals >= 10:  # minimum 10 signals
                accuracies[name] = max(perf.accuracy_7d, 0.1)

        if not accuracies:
            return self.ensemble_weights

        # Normaliser til vægte der summer til 1.0
        total = sum(accuracies.values())
        new_weights = {name: acc / total for name, acc in accuracies.items()}

        # Log ændringer
        for name, new_w in new_weights.items():
            old_w = self.ensemble_weights.get(name, 0.33)
            if abs(new_w - old_w) > 0.05:
                logger.info(
                    f"🔄 Ensemble vægt {name}: "
                    f"{old_w:.2f} → {new_w:.2f} "
                    f"(accuracy: {accuracies[name]:.1%})"
                )

        self.ensemble_weights = new_weights

        # Log til database
        with sqlite3.connect(self.db_path) as db:
            db.execute("""
                INSERT INTO learning_log (event_type, details, impact)
                VALUES (?, ?, ?)
            """, (
                "WEIGHT_UPDATE",
                json.dumps(new_weights),
                f"Baseret på {sum(p.total_signals for p in self.model_performance.values())} signals"
            ))

        return new_weights

    # ─── Concept Drift Detection ────────────────────────────

    def _check_drift(self, outcome: TradeOutcome):
        """
        Detect concept drift — når markedet ændrer karakter
        og modellerne ikke længere passer.
        """
        was_correct = outcome.pnl > 0
        self.drift_window.append(1.0 if was_correct else 0.0)

        if len(self.drift_window) < 20:
            return  # For lidt data

        # Sammenlign seneste 10 med historisk
        recent_accuracy = np.mean(self.drift_window[-10:])
        historical_accuracy = np.mean(self.drift_window[:-10])

        drift_magnitude = historical_accuracy - recent_accuracy

        if drift_magnitude > self.drift_threshold:
            severity = "CRITICAL" if drift_magnitude > 0.3 else "WARNING"

            logger.warning(
                f"⚠️ CONCEPT DRIFT DETEKTERET [{severity}]: "
                f"Accuracy faldet fra {historical_accuracy:.1%} "
                f"til {recent_accuracy:.1%} "
                f"(drift: {drift_magnitude:.1%})"
            )

            # Log drift event
            with sqlite3.connect(self.db_path) as db:
                db.execute("""
                    INSERT INTO drift_events
                    (drift_type, severity, old_accuracy, new_accuracy, action_taken)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    "PERFORMANCE_DROP", drift_magnitude,
                    historical_accuracy, recent_accuracy,
                    "RETRAIN_TRIGGERED" if severity == "CRITICAL" else "MONITORING"
                ))

            if severity == "CRITICAL":
                self._trigger_emergency_retrain(
                    f"Concept drift: {drift_magnitude:.1%} accuracy drop"
                )

    def _trigger_emergency_retrain(self, reason: str):
        """Trigger en nød-retræning af alle modeller"""
        logger.warning(f"🔄 EMERGENCY RETRAIN: {reason}")
        self.last_retrain_trigger = {
            "time": datetime.now().isoformat(),
            "reason": reason,
            "type": "emergency"
        }

        with sqlite3.connect(self.db_path) as db:
            db.execute("""
                INSERT INTO learning_log (event_type, details, impact)
                VALUES (?, ?, ?)
            """, ("EMERGENCY_RETRAIN", reason, "Alle modeller retrænes"))

        # Behold historisk drift-data for post-retrain evaluering
        # (nulstil IKKE — vi skal kunne se om retrain faktisk hjalp)
        logger.info(f"[learner] Drift window beholdt ({len(self.drift_window)} entries) for post-retrain evaluering")

    # ─── Historical Crisis Analysis ─────────────────────────

    def _load_crisis_patterns(self):
        """
        Indlæs kendte krise-mønstre fra historien.
        Disse bruges til at scanne aktuel data for lighed.
        """
        self.crisis_patterns = [
            CrisisPattern(
                crisis_name="Dotcom Crash 2000",
                start_date="2000-03-10",
                end_date="2002-10-09",
                pre_crisis_indicators={
                    "pe_ratio_avg": 44.0,       # Ekstremt høje P/E
                    "rsi_monthly": 82.0,         # Overkøbt
                    "sma50_above_sma200": 1.0,   # Bull trend (falsk sikkerhed)
                    "volatility_percentile": 35,  # Lav vol FØR crash
                    "new_highs_pct": 0.45,       # Mange nye highs
                    "margin_debt_growth": 0.40,   # 40% margin debt vækst
                },
                during_crisis_indicators={
                    "rsi_monthly": 22.0,
                    "drawdown_pct": -78.0,
                    "volatility_percentile": 95,
                },
                recovery_indicators={
                    "rsi_monthly": 45.0,
                    "sma50_cross_sma200": 1.0,   # Golden cross
                },
                max_drawdown=-78.4,
                recovery_days=1827,
                warning_signals=[
                    "Ekstremt høje P/E ratios",
                    "IPO frenzy",
                    "Teknologi-aktier dominerer",
                    "Retail investor eufori",
                    "Stigende margin debt",
                ],
                similarity_threshold=0.60,
            ),
            CrisisPattern(
                crisis_name="Financial Crisis 2008",
                start_date="2007-10-09",
                end_date="2009-03-09",
                pre_crisis_indicators={
                    "yield_curve_spread": -0.20,  # Inverteret yield curve!
                    "vix_avg": 16.0,              # Lav VIX (complacency)
                    "credit_spread_widening": 0.30,
                    "housing_starts_decline": -0.15,
                    "bank_stock_weakness": -0.10,
                    "sma50_above_sma200": 1.0,
                },
                during_crisis_indicators={
                    "vix_peak": 89.53,
                    "drawdown_pct": -57.0,
                    "credit_spreads": 6.0,
                },
                recovery_indicators={
                    "vix_below": 30.0,
                    "fed_rate_cut": 1.0,
                },
                max_drawdown=-56.8,
                recovery_days=1485,
                warning_signals=[
                    "Inverteret yield curve (12-18 mdr før)",
                    "Stigende credit spreads",
                    "Faldende housing starts",
                    "Bank-aktier svage",
                    "Subprime defaults stiger",
                ],
                similarity_threshold=0.55,
            ),
            CrisisPattern(
                crisis_name="COVID Crash 2020",
                start_date="2020-02-19",
                end_date="2020-03-23",
                pre_crisis_indicators={
                    "rsi_monthly": 72.0,
                    "sma50_above_sma200": 1.0,
                    "volatility_percentile": 15,  # Ekstremt lav vol
                    "all_time_high": 1.0,
                    "global_uncertainty": 0.60,    # COVID nyheder
                },
                during_crisis_indicators={
                    "vix_peak": 82.69,
                    "drawdown_pct": -34.0,
                    "speed_of_decline": -12.0,  # % per uge
                },
                recovery_indicators={
                    "fed_stimulus": 1.0,
                    "fiscal_stimulus": 1.0,
                    "vix_declining": 1.0,
                },
                max_drawdown=-33.9,
                recovery_days=148,  # V-shaped recovery
                warning_signals=[
                    "Ekstremt lav volatilitet",
                    "All-time highs",
                    "Ekstern chok (pandemi)",
                    "Global supply chain disruption",
                    "Hurtigste decline i historien",
                ],
                similarity_threshold=0.50,
            ),
            CrisisPattern(
                crisis_name="2022 Rate Hike Selloff",
                start_date="2022-01-03",
                end_date="2022-10-12",
                pre_crisis_indicators={
                    "inflation_rate": 0.07,        # 7%+ inflation
                    "fed_rate_expectations": 0.50,  # Aggressive hike forventet
                    "growth_vs_value_ratio": 2.5,   # Growth ekstremt overkøbt
                    "pe_ratio_avg": 38.0,
                    "bond_yield_rising": 1.0,
                },
                during_crisis_indicators={
                    "nasdaq_drawdown": -33.0,
                    "bond_simultaneous_decline": 1.0,
                    "fed_rate_actual": 4.25,
                },
                recovery_indicators={
                    "inflation_peaking": 1.0,
                    "fed_pivot_signals": 1.0,
                },
                max_drawdown=-33.1,
                recovery_days=365,
                warning_signals=[
                    "Høj inflation",
                    "Fed hawkish retorik",
                    "Growth-aktier med høje P/E",
                    "Stigende obligationsrenter",
                    "Bond-aktie korrelation bryder",
                ],
                similarity_threshold=0.55,
            ),
        ]

        logger.info(f"📚 {len(self.crisis_patterns)} historiske krise-mønstre indlæst")

    def scan_for_crisis_similarity(self, current_indicators: Dict[str, float]) -> List[Dict]:
        """
        Sammenlign aktuelle markedsindikatorer med historiske kriser.
        Returnerer liste af matchende kriser med similarity score.
        """
        matches = []

        for pattern in self.crisis_patterns:
            similarity = self._calculate_similarity(
                current_indicators,
                pattern.pre_crisis_indicators
            )

            if similarity >= pattern.similarity_threshold * 0.7:  # alert threshold
                match = {
                    "crisis": pattern.crisis_name,
                    "similarity": similarity,
                    "threshold": pattern.similarity_threshold,
                    "is_warning": similarity >= pattern.similarity_threshold,
                    "matching_indicators": [],
                    "max_drawdown_if_similar": pattern.max_drawdown,
                    "recovery_days_if_similar": pattern.recovery_days,
                    "warning_signals": pattern.warning_signals,
                }

                # Find hvilke indikatorer matcher
                for ind_name, crisis_val in pattern.pre_crisis_indicators.items():
                    if ind_name in current_indicators:
                        current_val = current_indicators[ind_name]
                        # Inden for 30% = match
                        if crisis_val != 0 and abs(current_val - crisis_val) / abs(crisis_val) < 0.30:
                            match["matching_indicators"].append(
                                f"{ind_name}: nu={current_val:.2f} vs krise={crisis_val:.2f}"
                            )

                matches.append(match)

                if match["is_warning"]:
                    logger.warning(
                        f"⚠️ KRISE-LIGHED: {pattern.crisis_name} "
                        f"similarity={similarity:.1%} "
                        f"(threshold={pattern.similarity_threshold:.1%}) "
                        f"— Historisk max drawdown: {pattern.max_drawdown:.1f}%"
                    )

        # Gem til database
        with sqlite3.connect(self.db_path) as db:
            for m in matches:
                db.execute("""
                    INSERT INTO crisis_similarities
                    (crisis_name, similarity_score, matching_indicators, recommendation)
                    VALUES (?, ?, ?, ?)
                """, (
                    m["crisis"], m["similarity"],
                    json.dumps(m["matching_indicators"]),
                    "REDUCE_EXPOSURE" if m["is_warning"] else "MONITOR"
                ))

        self.current_market_similarity = {
            m["crisis"]: m["similarity"] for m in matches
        }

        return sorted(matches, key=lambda x: x["similarity"], reverse=True)

    def _calculate_similarity(self, current: Dict[str, float],
                               historical: Dict[str, float]) -> float:
        """Beregn cosine-lignende similarity mellem to indicator-sæt"""
        common_keys = set(current.keys()) & set(historical.keys())
        if not common_keys:
            return 0.0

        matches = 0
        total = len(common_keys)

        for key in common_keys:
            curr_val = current[key]
            hist_val = historical[key]

            if hist_val == 0:
                if curr_val == 0:
                    matches += 1
                continue

            # Procentvis afvigelse
            deviation = abs(curr_val - hist_val) / abs(hist_val)

            if deviation < 0.10:      # Inden for 10% = fuld match
                matches += 1.0
            elif deviation < 0.25:    # Inden for 25% = delvis match
                matches += 0.6
            elif deviation < 0.50:    # Inden for 50% = svag match
                matches += 0.3

        return matches / total if total > 0 else 0.0

    # ─── 24/7 Background Learning ───────────────────────────

    def start(self, interval_minutes: int = 5):
        """Start kontinuerlig learning i baggrunden"""
        if self._running:
            logger.warning("Learner kører allerede")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._learning_loop,
            args=(interval_minutes,),
            daemon=True,
            name="ContinuousLearner"
        )
        self._thread.start()
        logger.info(
            f"🧠 Continuous Learning startet — "
            f"kører hvert {interval_minutes} min"
        )

    def stop(self):
        """Stop learning loop"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("🧠 Continuous Learning stoppet")

    def __del__(self):
        """Sikkerhedsnet — stop tråden ved garbage collection."""
        if self._running:
            self.stop()

    def _learning_loop(self, interval_minutes: int):
        """Hovedloop der kører i baggrunden"""
        _cycle_count = 0
        while self._running:
            try:
                self._run_learning_cycle()
                _cycle_count += 1
                # Prune old DB entries every 50 cycles (~4 hours at 5 min interval)
                if _cycle_count % 50 == 0:
                    self._prune_learning_db()
            except Exception as e:
                logger.error(f"Learning cycle fejl: {e}")

            # Vent til næste cycle
            for _ in range(interval_minutes * 60):
                if not self._running:
                    return
                time.sleep(1)

    def _run_learning_cycle(self):
        """En enkelt learning-cyklus"""
        cycle_start = time.time()

        # 1. Hent nye trade outcomes fra trade-databasen
        new_outcomes = self._fetch_unprocessed_trades()
        for outcome in new_outcomes:
            self.record_trade_outcome(outcome)

        # 2. Recalkuler ensemble weights
        if self.total_lessons_learned > 0 and self.total_lessons_learned % 10 == 0:
            self.recalculate_ensemble_weights()

        # 3. Krise-scanning (hvert 10. cycle)
        # (kræver at vi har aktuelle indikatorer)

        # 4. Log cycle stats
        duration = time.time() - cycle_start

        if new_outcomes:
            logger.info(
                f"🧠 Learning cycle: {len(new_outcomes)} nye trades analyseret, "
                f"{self.total_lessons_learned} total lektioner, "
                f"{duration:.1f}s"
            )

    def _fetch_unprocessed_trades(self) -> List[TradeOutcome]:
        """Hent trades der endnu ikke er analyseret"""
        outcomes = []

        try:
            # Hent trades der er completed men ikke lært fra
            with sqlite3.connect(self.db_path) as learn_db:
                processed = set()
                for row in learn_db.execute(
                    "SELECT entry_time, symbol FROM trade_outcomes"
                ).fetchall():
                    processed.add((row[0], row[1]))

            # Hent alle executed trades
            with sqlite3.connect(self.trade_db_path) as trade_db:
                trade_db.row_factory = sqlite3.Row
                trades = trade_db.execute("""
                    SELECT * FROM trades
                    WHERE executed = 1
                    ORDER BY timestamp DESC
                    LIMIT 100
                """).fetchall()

                for t in trades:
                    key = (t["timestamp"], t["symbol"])
                    if key not in processed:
                        outcome = TradeOutcome(
                            symbol=t["symbol"],
                            side=t["side"],
                            entry_price=t.get("price", 0) or 0,
                            exit_price=t.get("price", 0) or 0,  # Mangler exit data
                            entry_time=t["timestamp"],
                            exit_time="",  # Endnu åben
                            pnl=0.0,
                            pnl_pct=0.0,
                            holding_period_hours=0,
                            confidence_at_entry=t.get("confidence", 50),
                            regime_at_entry="UNKNOWN",
                            regime_at_exit="UNKNOWN",
                            strategy_signals={t.get("reason", "unknown"): t["side"]},
                            indicators_at_entry={}
                        )
                        outcomes.append(outcome)

        except Exception as e:
            logger.debug(f"Kunne ikke hente trades: {e}")

        return outcomes

    # ─── Analytics & Reports ────────────────────────────────

    def get_learning_summary(self) -> Dict:
        """Hent samlet summary af hvad systemet har lært"""
        summary = {
            "total_lessons": self.total_lessons_learned,
            "model_performance": {},
            "ensemble_weights": self.ensemble_weights,
            "drift_status": "OK",
            "crisis_similarities": self.current_market_similarity,
            "last_retrain_trigger": self.last_retrain_trigger,
        }

        for name, perf in self.model_performance.items():
            summary["model_performance"][name] = {
                "accuracy": perf.accuracy_7d,
                "total_signals": perf.total_signals,
                "avg_pnl": perf.avg_pnl_per_signal,
                "weight": perf.weight,
            }

        # Check drift status
        if self.drift_window and len(self.drift_window) >= 10:
            recent = np.mean(self.drift_window[-10:])
            if recent < 0.4:
                summary["drift_status"] = "CRITICAL"
            elif recent < 0.5:
                summary["drift_status"] = "WARNING"

        return summary

    def get_trade_analytics(self) -> Dict:
        """Detaljeret analytics over alle lærte trades"""
        analytics = {
            "total_trades_analyzed": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "best_regime": "",
            "worst_regime": "",
            "by_regime": {},
            "by_side": {},
            "common_lessons": [],
        }

        with sqlite3.connect(self.db_path) as db:
            db.row_factory = sqlite3.Row
            trades = db.execute("SELECT * FROM trade_outcomes").fetchall()

        analytics["total_trades_analyzed"] = len(trades)

        if trades:
            pnls = [t["pnl_pct"] for t in trades]
            wins = [t for t in trades if t["pnl"] > 0]
            analytics["win_rate"] = len(wins) / len(trades)
            analytics["avg_pnl_pct"] = np.mean(pnls) if pnls else 0

            # By regime
            regime_pnls = {}
            for t in trades:
                regime = t["regime_at_entry"]
                if regime not in regime_pnls:
                    regime_pnls[regime] = []
                regime_pnls[regime].append(t["pnl_pct"])

            for regime, pnls in regime_pnls.items():
                analytics["by_regime"][regime] = {
                    "count": len(pnls),
                    "avg_pnl": np.mean(pnls),
                    "win_rate": len([p for p in pnls if p > 0]) / len(pnls)
                }

            # By side
            side_pnls = {}
            for t in trades:
                side = t["side"]
                if side not in side_pnls:
                    side_pnls[side] = []
                side_pnls[side].append(t["pnl_pct"])

            for side, pnls in side_pnls.items():
                analytics["by_side"][side] = {
                    "count": len(pnls),
                    "avg_pnl": np.mean(pnls),
                    "win_rate": len([p for p in pnls if p > 0]) / len(pnls)
                }

        return analytics

    def _prune_learning_db(self) -> None:
        """Remove old entries to prevent unbounded DB growth."""
        try:
            with sqlite3.connect(self.db_path) as db:
                db.execute(
                    "DELETE FROM trade_outcomes WHERE timestamp < datetime('now', '-90 days')"
                )
                db.execute(
                    "DELETE FROM model_scores WHERE timestamp < datetime('now', '-30 days')"
                )
                db.execute(
                    "DELETE FROM drift_events WHERE timestamp < datetime('now', '-30 days')"
                )
                db.execute(
                    "DELETE FROM learning_log WHERE timestamp < datetime('now', '-30 days')"
                )
                db.execute(
                    "DELETE FROM crisis_similarities WHERE timestamp < datetime('now', '-30 days')"
                )
                db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            logger.debug("[learning] Database pruned (outcomes 90d, others 30d)")
        except Exception as e:
            logger.debug(f"[learning] DB prune error: {e}")


# ─── Historical Data Researcher ─────────────────────────────

class HistoricalResearcher:
    """
    Studerer historisk data for at finde mønstre.
    Kører dyb analyse af hvad der skete FØR og EFTER:
    - Market crashes
    - Recovery perioder
    - Sektor-rotationer
    - Korrelationsændringer
    """

    def __init__(self):
        self.research_results = {}
        logger.info("📊 Historical Researcher initialiseret")

    def study_period(self, symbol: str, start: str, end: str,
                     label: str = "") -> Dict:
        """
        Studér en specifik historisk periode dybt.

        Args:
            symbol: Ticker symbol
            start: Start dato (YYYY-MM-DD)
            end: Slut dato (YYYY-MM-DD)
            label: Beskrivende label (f.eks. "Pre-COVID")
        """
        try:
            import yfinance as yf

            # Hent data med buffer før og efter
            start_dt = pd.Timestamp(start) - pd.Timedelta(days=90)
            end_dt = pd.Timestamp(end) + pd.Timedelta(days=90)

            data = yf.download(symbol, start=start_dt.strftime("%Y-%m-%d"),
                             end=end_dt.strftime("%Y-%m-%d"), progress=False)

            if data.empty:
                return {"error": f"Ingen data for {symbol}"}

            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Beregn indikatorer
            data["SMA_20"] = data["Close"].rolling(20).mean()
            data["SMA_50"] = data["Close"].rolling(50).mean()
            data["SMA_200"] = data["Close"].rolling(200).mean()
            data["RSI"] = self._calc_rsi(data["Close"])
            data["Daily_Return"] = data["Close"].pct_change()
            data["Volatility_20d"] = data["Daily_Return"].rolling(20).std() * np.sqrt(252)
            data["Volume_SMA"] = data["Volume"].rolling(20).mean()
            data["Volume_Ratio"] = data["Volume"] / data["Volume_SMA"]

            # Del op i perioder
            pre_period = data.loc[:start]
            during_period = data.loc[start:end]
            post_period = data.loc[end:]

            result = {
                "symbol": symbol,
                "label": label or f"{symbol} {start} to {end}",
                "pre_period": self._analyze_period(pre_period, "PRE"),
                "during_period": self._analyze_period(during_period, "DURING"),
                "post_period": self._analyze_period(post_period, "POST"),
                "max_drawdown": self._max_drawdown(during_period["Close"]),
                "recovery_analysis": self._analyze_recovery(post_period),
            }

            self.research_results[label] = result
            return result

        except Exception as e:
            logger.error(f"Research fejl: {e}")
            return {"error": str(e)}

    def _analyze_period(self, df: pd.DataFrame, label: str) -> Dict:
        """Detaljeret analyse af en periode"""
        if df.empty or len(df) < 5:
            return {"label": label, "data_points": 0}

        close = df["Close"]
        returns = df.get("Daily_Return", close.pct_change())

        return {
            "label": label,
            "data_points": len(df),
            "return_pct": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
            "volatility": float(returns.std() * np.sqrt(252) * 100) if len(returns.dropna()) > 0 else 0,
            "avg_rsi": float(df["RSI"].mean()) if "RSI" in df else 0,
            "avg_volume_ratio": float(df["Volume_Ratio"].mean()) if "Volume_Ratio" in df else 0,
            "max_daily_drop": float(returns.min() * 100) if len(returns.dropna()) > 0 else 0,
            "max_daily_gain": float(returns.max() * 100) if len(returns.dropna()) > 0 else 0,
            "sma50_above_sma200": bool(
                df["SMA_50"].iloc[-1] > df["SMA_200"].iloc[-1]
            ) if "SMA_50" in df and "SMA_200" in df and not df["SMA_50"].isna().all() else None,
        }

    def _calc_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Beregn RSI"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def _max_drawdown(self, prices: pd.Series) -> float:
        """Beregn maximum drawdown"""
        if prices.empty:
            return 0.0
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return float(drawdown.min() * 100)

    def _analyze_recovery(self, df: pd.DataFrame) -> Dict:
        """Analysér recovery-mønster"""
        if df.empty or len(df) < 5:
            return {"pattern": "INSUFFICIENT_DATA"}

        close = df["Close"]
        initial = close.iloc[0]

        # Find hvornår den nåede nye highs
        pre_crash_high = close.max()
        days_to_recover = 0

        for i, price in enumerate(close):
            if price >= initial * 1.0:  # Tilbage til start
                days_to_recover = i
                break

        # Bestem recovery-type
        returns = close.pct_change().dropna()
        if len(returns) < 10:
            pattern = "UNKNOWN"
        elif returns.iloc[:10].mean() > returns.iloc[-10:].mean():
            pattern = "V_SHAPE"
        elif returns.std() > returns.mean() * 3:
            pattern = "W_SHAPE"
        else:
            pattern = "GRADUAL"

        return {
            "pattern": pattern,
            "days_to_breakeven": days_to_recover,
            "total_recovery_pct": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
        }

    def run_full_crisis_study(self) -> Dict:
        """
        Kør komplet studie af alle historiske kriser
        for SPY (S&P 500 proxy).
        """
        logger.info("📊 Starter fuld historisk krise-analyse...")

        crises = {
            "Dotcom_Crash": ("2000-03-10", "2002-10-09"),
            "Financial_Crisis": ("2007-10-09", "2009-03-09"),
            "COVID_Crash": ("2020-02-19", "2020-03-23"),
            "Rate_Hike_2022": ("2022-01-03", "2022-10-12"),
            "COVID_Recovery": ("2020-03-23", "2021-01-01"),
            "Post_2008_Recovery": ("2009-03-09", "2010-03-09"),
        }

        results = {}
        for name, (start, end) in crises.items():
            logger.info(f"  Studerer {name}...")
            results[name] = self.study_period("SPY", start, end, name)

        logger.info(f"📊 Krise-studie komplet: {len(results)} perioder analyseret")
        return results
