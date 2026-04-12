"""
Alpha Trader — Remote Training Script
======================================
Kores paa Gorms Linux/GPU-maskine for at traene ML-modellerne.

Hvad goetr det?
  1. Downloader historisk data fra Yahoo Finance (5 aars OHLCV)
  2. Traener MLStrategy (gradient boosted trees)
  3. Traener EnsembleMLStrategy (Random Forest + XGBoost + Logistic Regression)
  4. Gemmer modellerne i models/  med dato-stempel
  5. Printer rapport over model-kvalitet

Brug:
  python train_remote.py                  # Standard: alle symboler
  python train_remote.py --symbols AAPL MSFT NVDA   # Kun udvalgte symboler
  python train_remote.py --years 7        # Traen paa 7 aars data (default: 5)
  python train_remote.py --output models/ # Gem modeller her (default: models/)
  python train_remote.py --skip-ensemble  # Spring ensemble over (hurtigere)
  python train_remote.py --jobs 8         # Antal CPU-kerner til paralleltraening

Output:
  models/ml_YYYYMMDD_HHMMSS.joblib        — Gradient Boosted Trees model
  models/ensemble_YYYYMMDD_HHMMSS.joblib  — Ensemble model (RF + XGB + LR)
  models/training_report_YYYYMMDD.txt     — Rapport med metrics

Synkronisering:
  Filen sync_to_gorm.sh paa Ole's Mac Mini sender data hertil og
  henter modeller tilbage automatisk.

Kraever:
  pip install yfinance scikit-learn xgboost pandas numpy loguru joblib
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── Konfiguration ─────────────────────────────────────────────

# Symboler der traenes paa (samme som Alpha Trader's universe)
DEFAULT_SYMBOLS = [
    # US Large Cap Stocks
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "UNH", "JNJ", "XOM", "PG", "MA",
    "HD", "CVX", "MRK", "ABBV", "PEP", "KO", "AVGO", "COST",
    "WMT", "BAC", "DIS", "CSCO", "ADBE", "CRM", "AMD", "INTC",
    "NFLX", "QCOM", "TXN", "IBM", "GE", "CAT", "BA", "GS",
    "MS", "C", "WFC", "AXP", "LMT", "RTX", "UPS", "FDX",
    "MCD", "SBUX", "NKE", "PFE", "TMO", "ABT", "MDT", "AMGN",
    "GILD", "BMY",
    # ETFs
    "SPY", "QQQ", "IWM", "VTI", "GLD", "TLT",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY",
    "XLP", "XLB", "XLRE", "XLU",
]

# ─── Logging ────────────────────────────────────────────────────

try:
    from loguru import logger
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("train_remote")
    logger.info = logger.info
    logger.warning = logger.warning
    logger.error = logger.error
    logger.success = lambda msg: logger.info(f"OK: {msg}")


# ─── Feature Engineering (standalone, ingen src/ import) ───────

FEATURE_COLUMNS = [
    "RSI", "MACD", "MACD_Signal", "MACD_Hist",
    "SMA_20_pct", "SMA_50_pct", "SMA_200_pct", "SMA_cross",
    "BB_position", "BB_Width",
    "Volume_Ratio", "OBV_slope",
    "return_1d", "return_5d", "return_20d", "volatility_20d",
]

ENSEMBLE_EXTRA_COLUMNS = [
    "roc_10", "stoch_k", "stoch_d", "atr_pct", "volatility_ratio",
]

ENSEMBLE_FEATURE_COLUMNS = FEATURE_COLUMNS + ENSEMBLE_EXTRA_COLUMNS


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def build_features(df: pd.DataFrame, ensemble: bool = False) -> pd.DataFrame:
    """Byg feature-matrix fra OHLCV-data."""
    df = df.copy()
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # RSI
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=13, adjust=False).mean()
    avg_l = loss.ewm(com=13, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = _ema(df["MACD"], 9)
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # SMAs
    for w in [20, 50, 200]:
        sma = close.rolling(w).mean()
        df[f"SMA_{w}"]     = sma
        df[f"SMA_{w}_pct"] = (close - sma) / sma

    df["SMA_cross"] = (df["SMA_20"] - df["SMA_50"]) / close

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20
    bb_range = df["BB_Upper"] - df["BB_Lower"]
    df["BB_position"] = np.where(
        bb_range > 0, (close - df["BB_Lower"]) / bb_range, 0.5
    )
    df["BB_Width"] = bb_range / sma20

    # Volume
    df["Volume_Ratio"] = vol / vol.rolling(20).mean()
    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    df["OBV_slope"] = obv.pct_change(5).clip(-1, 1)

    # Returns
    df["return_1d"]     = close.pct_change(1)
    df["return_5d"]     = close.pct_change(5)
    df["return_20d"]    = close.pct_change(20)
    df["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    if ensemble:
        # ROC 10
        df["roc_10"] = close.pct_change(10)

        # Stochastic
        low14  = low.rolling(14).min()
        high14 = high.rolling(14).max()
        denom  = (high14 - low14).replace(0, np.nan)
        df["stoch_k"] = (close - low14) / denom * 100
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # ATR %
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr_pct"] = tr.rolling(14).mean() / close

        # Volatility ratio
        df["volatility_ratio"] = (
            close.pct_change().rolling(5).std() /
            close.pct_change().rolling(20).std().replace(0, np.nan)
        )

    return df


def build_target(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
    """Binaert target: stiger prisen > threshold over horizon dage?"""
    future_return = df["Close"].pct_change(horizon).shift(-horizon)
    return (future_return > threshold).astype(float).where(future_return.notna())


# ─── Data Download ──────────────────────────────────────────────

def download_data(symbols: list[str], years: int = 5) -> dict[str, pd.DataFrame]:
    """Download historisk OHLCV data fra Yahoo Finance."""
    import yfinance as yf

    end   = datetime.today()
    start = end - timedelta(days=years * 365 + 30)

    logger.info(f"Downloader data: {len(symbols)} symboler, {years} aars historik...")
    logger.info(f"Periode: {start.strftime('%Y-%m-%d')} - {end.strftime('%Y-%m-%d')}")

    result: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    # Batch-download (hurtigere end enkeltvis)
    batch_size = 20
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        try:
            raw = yf.download(
                batch,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
                threads=True,
            )

            # Haandter enten multi-level eller single-level kolonner
            if isinstance(raw.columns, pd.MultiIndex):
                for sym in batch:
                    try:
                        df = raw.xs(sym, axis=1, level=1).dropna(how="all")
                        if len(df) >= 200:
                            result[sym] = df
                        else:
                            logger.warning(f"  {sym}: for lidt data ({len(df)} rækker) — springer over")
                    except Exception:
                        failed.append(sym)
            else:
                # Kun ét symbol i batch
                sym = batch[0]
                df  = raw.dropna(how="all")
                if len(df) >= 200:
                    result[sym] = df
                else:
                    failed.append(sym)

        except Exception as e:
            logger.error(f"  Batch fejl {batch}: {e}")
            failed.extend(batch)

        downloaded = len(result)
        logger.info(f"  {downloaded}/{len(symbols)} symboler downloadet...")
        time.sleep(0.5)   # Respekter Yahoo Finance rate limit

    if failed:
        logger.warning(f"Mangler data for {len(failed)} symboler: {failed[:10]}...")

    logger.info(f"Download komplet: {len(result)} symboler klar til traening")
    return result


# ─── Træning: MLStrategy ────────────────────────────────────────

def train_ml_strategy(
    data: dict[str, pd.DataFrame],
    jobs: int = -1,
    horizon: int = 1,
    train_years: int = 4,
    test_months: int = 6,
) -> tuple[object, dict]:
    """
    Traen en MLStrategy (HistGradientBoostingClassifier) paa alle symboler.

    Returnerer (model_payload, metrics_dict).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    logger.info("=" * 60)
    logger.info("TRAENER: MLStrategy (Gradient Boosted Trees)")
    logger.info("=" * 60)

    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series]    = []

    for sym, df in data.items():
        try:
            feat = build_features(df, ensemble=False)
            tgt  = build_target(df, horizon=horizon)
            mask = tgt.notna()
            x    = feat.loc[mask, FEATURE_COLUMNS]
            y    = tgt.loc[mask]
            if len(x) >= 100 and not x.isnull().all().all():
                all_X.append(x)
                all_y.append(y)
        except Exception as e:
            logger.warning(f"  {sym}: feature-fejl — {e}")

    if not all_X:
        raise RuntimeError("Ingen brugbar traeningsdata fundet!")

    X = pd.concat(all_X).reset_index(drop=True)
    y = pd.concat(all_y).reset_index(drop=True)

    # Tidsopdelt split: de seneste 15% til test
    n_test  = max(1000, int(len(X) * 0.15))
    n_train = len(X) - n_test

    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]

    logger.info(f"  Traendata:  {n_train:,} rækker")
    logger.info(f"  Testdata:   {n_test:,} rækker")
    logger.info(f"  Features:   {len(FEATURE_COLUMNS)}")

    # Hyperparameter grid search (simpel version)
    best_model  = None
    best_auc    = 0.0
    best_params: dict = {}

    param_grid = [
        {"max_iter": 300, "max_depth": 5,  "learning_rate": 0.05},
        {"max_iter": 500, "max_depth": 6,  "learning_rate": 0.03},
        {"max_iter": 200, "max_depth": 4,  "learning_rate": 0.10},
        {"max_iter": 400, "max_depth": 7,  "learning_rate": 0.03},
    ]

    for params in param_grid:
        logger.info(f"  Afproeving: {params}")
        model = HistGradientBoostingClassifier(
            **params,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=25,
            random_state=42,
        )
        model.fit(X_train.values, y_train.values)

        y_proba = model.predict_proba(X_test.values)
        if y_proba.shape[1] >= 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = 0.5

        logger.info(f"    AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc    = auc
            best_model  = model
            best_params = params

    logger.info(f"  Bedste model: AUC={best_auc:.4f}, params={best_params}")

    # Final metrics
    y_pred  = best_model.predict(X_test.values)
    y_proba = best_model.predict_proba(X_test.values)
    auc     = roc_auc_score(y_test, y_proba[:, 1]) if y_proba.shape[1] >= 2 else 0.5

    metrics = {
        "accuracy":       float(accuracy_score(y_test, y_pred)),
        "f1":             float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc":        float(auc),
        "n_train":        int(n_train),
        "n_test":         int(n_test),
        "n_symbols":      len(data),
        "best_params":    best_params,
        "trained_at":     datetime.utcnow().isoformat(),
    }

    logger.info(f"  Accuracy: {metrics['accuracy']:.1%}  F1: {metrics['f1']:.1%}  AUC: {metrics['auc_roc']:.4f}")

    payload = {
        "model":            best_model,
        "feature_columns":  FEATURE_COLUMNS,
        "metrics":          metrics,
        "horizon":          horizon,
        "threshold":        0.0,
        "confidence_min":   0.55,
        "n_estimators":     best_params.get("max_iter", 300),
        "max_depth":        best_params.get("max_depth", 5),
        "learning_rate":    best_params.get("learning_rate", 0.05),
        "saved_at":         datetime.utcnow().isoformat(),
        "version":          "MLStrategy_v1",
    }

    return payload, metrics


# ─── Træning: EnsembleMLStrategy ────────────────────────────────

def train_ensemble_strategy(
    data: dict[str, pd.DataFrame],
    jobs: int = -1,
    horizon: int = 5,
) -> tuple[object, dict]:
    """
    Traen en EnsembleMLStrategy (RF + XGBoost + LR) paa alle symboler.

    3 modeller stemmer — kun handel hvis mindst 2 er enige.
    Returnerer (model_payload, metrics_dict).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, roc_auc_score

    try:
        import xgboost as xgb
        has_xgb = True
    except ImportError:
        logger.warning("  XGBoost ikke installeret — bruger ExtraTreesClassifier som erstatning")
        from sklearn.ensemble import ExtraTreesClassifier
        has_xgb = False

    logger.info("=" * 60)
    logger.info("TRAENER: EnsembleMLStrategy (RF + XGB + LR)")
    logger.info("=" * 60)

    all_X: list[pd.DataFrame] = []
    all_y: list[pd.Series]    = []

    for sym, df in data.items():
        try:
            feat = build_features(df, ensemble=True)
            tgt  = build_target(df, horizon=horizon, threshold=0.01)
            mask = tgt.notna()
            x    = feat.loc[mask, ENSEMBLE_FEATURE_COLUMNS]
            y    = tgt.loc[mask]
            if len(x) >= 100 and not x.isnull().all().all():
                all_X.append(x)
                all_y.append(y)
        except Exception as e:
            logger.warning(f"  {sym}: feature-fejl — {e}")

    if not all_X:
        raise RuntimeError("Ingen brugbar traeningsdata fundet!")

    X = pd.concat(all_X).reset_index(drop=True)
    y = pd.concat(all_y).reset_index(drop=True)

    # Fyld NaN med kolonne-median (XGBoost klarer det selv, LR/RF goetr ikke)
    X = X.fillna(X.median())

    n_test  = max(1000, int(len(X) * 0.15))
    n_train = len(X) - n_test

    X_train, X_test = X.iloc[:n_train].values, X.iloc[n_train:].values
    y_train, y_test = y.iloc[:n_train].values, y.iloc[n_train:].values

    logger.info(f"  Traendata:  {n_train:,} rækker")
    logger.info(f"  Testdata:   {n_test:,} rækker")
    logger.info(f"  Features:   {len(ENSEMBLE_FEATURE_COLUMNS)}")

    results: dict[str, dict] = {}

    # 1. Random Forest
    logger.info("  [1/3] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        n_jobs=jobs,
        random_state=42,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)
    rf_auc   = roc_auc_score(y_test, rf_proba[:, 1]) if rf_proba.shape[1] >= 2 else 0.5
    rf_acc   = accuracy_score(y_test, rf.predict(X_test))
    logger.info(f"    AUC: {rf_auc:.4f}  Accuracy: {rf_acc:.1%}")
    results["rf"] = {"auc": rf_auc, "accuracy": rf_acc}

    # 2. XGBoost / ExtraTrees
    if has_xgb:
        logger.info("  [2/3] XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=jobs,
            random_state=42,
            verbosity=0,
        )
    else:
        logger.info("  [2/3] ExtraTrees (XGBoost erstatning)...")
        from sklearn.ensemble import ExtraTreesClassifier
        xgb_model = ExtraTreesClassifier(
            n_estimators=300, max_depth=8, n_jobs=jobs, random_state=42
        )

    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)
    xgb_auc   = roc_auc_score(y_test, xgb_proba[:, 1]) if xgb_proba.shape[1] >= 2 else 0.5
    xgb_acc   = accuracy_score(y_test, xgb_model.predict(X_test))
    logger.info(f"    AUC: {xgb_auc:.4f}  Accuracy: {xgb_acc:.1%}")
    results["xgb"] = {"auc": xgb_auc, "accuracy": xgb_acc}

    # 3. Logistic Regression
    logger.info("  [3/3] Logistic Regression...")
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(C=0.1, max_iter=1000, n_jobs=jobs, random_state=42)),
    ])
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)
    lr_auc   = roc_auc_score(y_test, lr_proba[:, 1]) if lr_proba.shape[1] >= 2 else 0.5
    lr_acc   = accuracy_score(y_test, lr.predict(X_test))
    logger.info(f"    AUC: {lr_auc:.4f}  Accuracy: {lr_acc:.1%}")
    results["lr"] = {"auc": lr_auc, "accuracy": lr_acc}

    # Ensemble (majority voting) metrics
    preds = np.stack([
        rf.predict(X_test),
        xgb_model.predict(X_test),
        lr.predict(X_test),
    ])
    ensemble_pred = (preds.sum(axis=0) >= 2).astype(int)  # mindst 2 af 3 enige
    ensemble_acc  = accuracy_score(y_test, ensemble_pred)
    logger.info(f"  Ensemble accuracy (maj. vote): {ensemble_acc:.1%}")

    metrics = {
        "rf_auc":          results["rf"]["auc"],
        "xgb_auc":         results["xgb"]["auc"],
        "lr_auc":          results["lr"]["auc"],
        "ensemble_acc":    float(ensemble_acc),
        "n_train":         int(n_train),
        "n_test":          int(n_test),
        "n_symbols":       len(data),
        "trained_at":      datetime.utcnow().isoformat(),
        "has_xgboost":     has_xgb,
    }

    payload = {
        "rf_model":              rf,
        "xgb_model":             xgb_model,
        "lr_model":              lr,
        "feature_columns":       ENSEMBLE_FEATURE_COLUMNS,
        "metrics":               metrics,
        "performance_history":   [],
        "threshold":             0.01,
        "confidence_min":        0.55,
        "horizon":               horizon,
        "min_agreement":         2,
        "retrain_interval_days": 30,
        "saved_at":              datetime.utcnow().isoformat(),
        "version":               "EnsembleMLStrategy_v1",
    }

    return payload, metrics


# ─── Rapport ────────────────────────────────────────────────────

def print_report(ml_metrics: dict | None, ens_metrics: dict | None, output_paths: dict) -> str:
    """Generer og print rapport over traeningsresultater."""
    lines = [
        "=" * 65,
        "ALPHA TRADER — REMOTE TRAINING RAPPORT",
        f"Dato:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
    ]

    if ml_metrics:
        lines += [
            "ML STRATEGY (Gradient Boosted Trees)",
            "-" * 40,
            f"  Symboler traenet:  {ml_metrics.get('n_symbols', '?')}",
            f"  Traendata:         {ml_metrics.get('n_train', 0):,} rækker",
            f"  Testdata:          {ml_metrics.get('n_test', 0):,} rækker",
            f"  Accuracy:          {ml_metrics.get('accuracy', 0):.1%}",
            f"  F1-score:          {ml_metrics.get('f1', 0):.1%}",
            f"  AUC-ROC:           {ml_metrics.get('auc_roc', 0):.4f}",
            f"  Bedste params:     {ml_metrics.get('best_params', {})}",
            "",
            "  VURDERING:",
        ]
        auc = ml_metrics.get("auc_roc", 0)
        if auc >= 0.60:
            lines.append("  OK  Lovende diskrimination (AUC > 0.60)")
        elif auc >= 0.55:
            lines.append("  OK  Svag men positiv signal (AUC 0.55-0.60)")
        else:
            lines.append("  !   Tæt paa tilfældig (AUC < 0.55) — overvej mere data")
        lines.append("")

    if ens_metrics:
        lines += [
            "ENSEMBLE STRATEGY (RF + XGB + LR)",
            "-" * 40,
            f"  Symboler traenet:  {ens_metrics.get('n_symbols', '?')}",
            f"  Traendata:         {ens_metrics.get('n_train', 0):,} rækker",
            f"  Testdata:          {ens_metrics.get('n_test', 0):,} rækker",
            f"  Random Forest AUC: {ens_metrics.get('rf_auc', 0):.4f}",
            f"  XGBoost AUC:       {ens_metrics.get('xgb_auc', 0):.4f}",
            f"  Log. Regression:   {ens_metrics.get('lr_auc', 0):.4f}",
            f"  Ensemble Accuracy: {ens_metrics.get('ensemble_acc', 0):.1%}",
            f"  XGBoost tilstede:  {'Ja' if ens_metrics.get('has_xgboost') else 'Nej (ExtraTrees)'}",
            "",
        ]

    lines += [
        "GEMTE MODELLER:",
        "-" * 40,
    ]
    for name, path in output_paths.items():
        if path:
            size_kb = Path(path).stat().st_size / 1024
            lines.append(f"  {name}: {path} ({size_kb:.0f} KB)")

    lines += [
        "",
        "NAESTE SKRIDT:",
        "  Koetr sync_to_gorm.sh paa Ole's Mac Mini for at hente modellerne:",
        "  ./scripts/sync_to_gorm.sh --pull-only",
        "=" * 65,
    ]

    report = "\n".join(lines)
    print(report)
    return report


# ─── Main ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Alpha Trader — Remote Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symboler at traene paa (default: alle ~70 fra universe)"
    )
    p.add_argument(
        "--years", type=int, default=5,
        help="Aars historik at traene paa (default: 5)"
    )
    p.add_argument(
        "--output", default="models",
        help="Mappe til gemte modeller (default: models/)"
    )
    p.add_argument(
        "--skip-ensemble", action="store_true",
        help="Spring EnsembleMLStrategy over (hurtigere, kun GradientBoosting)"
    )
    p.add_argument(
        "--jobs", type=int, default=-1,
        help="Antal CPU-kerner (-1 = alle, default: -1)"
    )
    p.add_argument(
        "--horizon-ml", type=int, default=1,
        help="Forecast-horisont for MLStrategy i dage (default: 1)"
    )
    p.add_argument(
        "--horizon-ensemble", type=int, default=5,
        help="Forecast-horisont for EnsembleMLStrategy i dage (default: 5)"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    symbols   = args.symbols or DEFAULT_SYMBOLS
    output    = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 65)
    logger.info("ALPHA TRADER — REMOTE TRAINING STARTER")
    logger.info(f"Symboler: {len(symbols)}")
    logger.info(f"Historik: {args.years} aar")
    logger.info(f"Output:   {output.resolve()}")
    logger.info(f"CPU-kerner: {args.jobs}")
    logger.info("=" * 65)

    # Kontroller dependencies
    missing = []
    for pkg in ["yfinance", "sklearn", "joblib", "numpy", "pandas"]:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            missing.append(pkg)
    if missing:
        logger.error(f"Manglende pakker: {missing}")
        logger.error("Koetr: pip install " + " ".join(missing))
        return 1

    import joblib

    start_total = time.time()
    output_paths: dict[str, str | None] = {"ml": None, "ensemble": None}

    # ── Download data ─────────────────────────────────────────
    data = download_data(symbols, years=args.years)
    if not data:
        logger.error("Ingen data downloadet — afbryder")
        return 1

    ml_metrics:  dict | None = None
    ens_metrics: dict | None = None

    # ── Traen MLStrategy ──────────────────────────────────────
    try:
        ml_payload, ml_metrics = train_ml_strategy(
            data, jobs=args.jobs, horizon=args.horizon_ml
        )
        ml_path = output / f"ml_{timestamp}.joblib"
        joblib.dump(ml_payload, ml_path, compress=3)
        output_paths["ml"] = str(ml_path)

        # Gem ogsaa som "latest" saa Ole's platform altid bruger nyeste
        latest_ml = output / "ml_latest.joblib"
        joblib.dump(ml_payload, latest_ml, compress=3)
        logger.info(f"Gemt: {ml_path} + models/ml_latest.joblib")

    except Exception as e:
        logger.error(f"MLStrategy traening fejlede: {e}")
        import traceback
        traceback.print_exc()

    # ── Traen EnsembleMLStrategy ──────────────────────────────
    if not args.skip_ensemble:
        try:
            ens_payload, ens_metrics = train_ensemble_strategy(
                data, jobs=args.jobs, horizon=args.horizon_ensemble
            )
            ens_path = output / f"ensemble_{timestamp}.joblib"
            joblib.dump(ens_payload, ens_path, compress=3)
            output_paths["ensemble"] = str(ens_path)

            latest_ens = output / "ensemble_latest.joblib"
            joblib.dump(ens_payload, latest_ens, compress=3)
            logger.info(f"Gemt: {ens_path} + models/ensemble_latest.joblib")

        except Exception as e:
            logger.error(f"EnsembleMLStrategy traening fejlede: {e}")
            import traceback
            traceback.print_exc()

    # ── Rapport ───────────────────────────────────────────────
    elapsed = time.time() - start_total
    logger.info(f"\nTotal traeningstid: {elapsed/60:.1f} minutter")

    report = print_report(ml_metrics, ens_metrics, output_paths)

    report_path = output / f"training_report_{datetime.now().strftime('%Y%m%d')}.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Rapport gemt: {report_path}")

    # Gem metrics som JSON (bruges af sync-script)
    metrics_path = output / "latest_metrics.json"
    metrics_path.write_text(
        json.dumps({"ml": ml_metrics, "ensemble": ens_metrics, "trained_at": timestamp}, indent=2),
        encoding="utf-8",
    )

    if ml_metrics or ens_metrics:
        logger.info("TRAENING KOMPLET!")
        return 0
    else:
        logger.error("Ingen modeller traenet!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
