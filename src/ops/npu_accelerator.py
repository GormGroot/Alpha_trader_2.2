"""
NPU Accelerator — Rock 5B RK3588 NPU integration for Alpha Trading Platform.

The RK3588 has a 6 TOPS NPU (Neural Processing Unit) that can accelerate
ML inference tasks significantly vs CPU-only.

What the NPU accelerates:
  1. FinBERT sentiment analysis  → real-time news sentiment (was too slow on CPU)
  2. ML model inference          → Random Forest, XGBoost, GBT via ONNX → RKNN
  3. Pattern recognition         → price chart pattern detection

Workflow:
  CPU side:   Train models (sklearn/pytorch) → Export to ONNX
  NPU side:   Convert ONNX → RKNN → Run inference on NPU

Requirements (install on Rock 5B):
  pip install rknn-toolkit-lite2  # NPU Python interface
  pip install onnx onnxruntime    # For model conversion

Note: Model CONVERSION (ONNX → RKNN) must be done on an x86_64 machine
      with rknn-toolkit2 installed. Inference runs on Rock 5B NPU.
      If no converted models exist, falls back to CPU inference automatically.

File layout:
  data_cache/npu_models/
    finbert_sentiment.rknn       ← Converted FinBERT model
    ml_strategy.rknn             ← Converted ML strategy model
    ensemble_rf.rknn             ← Converted Random Forest
"""

from __future__ import annotations

import os
import time
import numpy as np
from pathlib import Path
from typing import Any
from loguru import logger


# ── NPU availability check ─────────────────────────────────

def check_npu_available() -> bool:
    """Check if RK3588 NPU is available and working."""
    try:
        # Check kernel driver
        if os.path.exists("/sys/kernel/debug/rknpu/version"):
            with open("/sys/kernel/debug/rknpu/version") as f:
                version = f.read().strip()
            logger.info(f"[npu] RK3588 NPU driver found: {version}")
            return True
        # Try importing rknn lite
        from rknnlite.api import RKNNLite  # noqa
        return True
    except Exception:
        return False


def get_npu_load() -> str:
    """Get current NPU utilization."""
    try:
        if os.path.exists("/sys/kernel/debug/rknpu/load"):
            with open("/sys/kernel/debug/rknpu/load") as f:
                return f.read().strip()
    except Exception:
        pass
    return "unknown"


# ── RKNN Model wrapper ─────────────────────────────────────

class RKNNModel:
    """
    Wrapper around RKNN Lite for running models on RK3588 NPU.
    Falls back to ONNX Runtime on CPU if NPU not available.
    """

    def __init__(self, model_path: str, use_npu: bool = True):
        self.model_path = model_path
        self.use_npu = use_npu and check_npu_available()
        self._model = None
        self._loaded = False
        self._inference_count = 0
        self._total_inference_ms = 0.0

    def load(self) -> bool:
        """Load model onto NPU or CPU."""
        if not os.path.exists(self.model_path):
            logger.warning(f"[npu] Model not found: {self.model_path}")
            return False

        if self.use_npu and self.model_path.endswith(".rknn"):
            return self._load_rknn()
        else:
            return self._load_onnx()

    def _load_rknn(self) -> bool:
        """Load RKNN model onto NPU."""
        try:
            from rknnlite.api import RKNNLite
            self._model = RKNNLite()
            ret = self._model.load_rknn(self.model_path)
            if ret != 0:
                logger.error(f"[npu] Failed to load RKNN model: {self.model_path}")
                return False
            ret = self._model.init_runtime()
            if ret != 0:
                logger.error("[npu] Failed to init NPU runtime")
                return False
            self._loaded = True
            logger.info(f"[npu] ✅ Model loaded on NPU: {Path(self.model_path).name}")
            return True
        except Exception as e:
            logger.warning(f"[npu] RKNN load failed, falling back to CPU: {e}")
            self.use_npu = False
            return self._load_onnx()

    def _load_onnx(self) -> bool:
        """Load ONNX model on CPU as fallback."""
        onnx_path = self.model_path.replace(".rknn", ".onnx")
        if not os.path.exists(onnx_path):
            logger.warning(f"[npu] No ONNX fallback found: {onnx_path}")
            return False
        try:
            import onnxruntime as ort
            self._model = ort.InferenceSession(
                onnx_path,
                providers=["CPUExecutionProvider"]
            )
            self._loaded = True
            logger.info(f"[npu] Model loaded on CPU (ONNX): {Path(onnx_path).name}")
            return True
        except Exception as e:
            logger.error(f"[npu] ONNX load failed: {e}")
            return False

    def infer(self, inputs: list[np.ndarray]) -> list[np.ndarray] | None:
        """Run inference on NPU or CPU."""
        if not self._loaded:
            return None

        t0 = time.time()
        try:
            if self.use_npu:
                outputs = self._model.inference(inputs=inputs)
            else:
                # ONNX Runtime inference
                input_names = [i.name for i in self._model.get_inputs()]
                feed = {name: inp for name, inp in zip(input_names, inputs)}
                outputs = self._model.run(None, feed)

            elapsed_ms = (time.time() - t0) * 1000
            self._inference_count += 1
            self._total_inference_ms += elapsed_ms
            return outputs

        except Exception as e:
            logger.error(f"[npu] Inference failed: {e}")
            return None

    def release(self) -> None:
        """Release NPU resources."""
        if self._model and self.use_npu:
            try:
                self._model.release()
            except Exception:
                pass
        self._loaded = False

    @property
    def avg_inference_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return self._total_inference_ms / self._inference_count

    @property
    def device(self) -> str:
        return "NPU (RK3588)" if self.use_npu else "CPU"


# ── NPU Sentiment Analyzer ─────────────────────────────────

class NPUSentimentAnalyzer:
    """
    FinBERT sentiment analysis accelerated by RK3588 NPU.

    If RKNN model exists → run on NPU (~10-30x faster than CPU)
    If not → fall back to CPU keyword analysis or transformers

    To convert FinBERT to RKNN (run on x86_64 machine):
        from rknn.api import RKNN
        rknn = RKNN()
        rknn.config(target_platform='rk3588')
        rknn.load_onnx(model='finbert.onnx')
        rknn.build(do_quantization=True, dataset='calibration_texts.txt')
        rknn.export_rknn('data_cache/npu_models/finbert_sentiment.rknn')
    """

    MODEL_PATH = "data_cache/npu_models/finbert_sentiment.rknn"
    MAX_LENGTH = 128  # Token limit for NPU inference

    _instance = None

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Singleton — undgå at oprette 328 instanser (én per symbol)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_dir: str = "data_cache/npu_models"):
        if self._initialized:
            return
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._rknn_model: RKNNModel | None = None
        self._tokenizer = None
        self._cpu_analyzer = None
        self._use_npu = False

    def initialize(self) -> bool:
        """Initialize NPU model or fall back to CPU."""
        if self._initialized:
            return True
        model_path = self._model_dir / "finbert_sentiment.rknn"

        # Try NPU first
        if model_path.exists() and check_npu_available():
            self._rknn_model = RKNNModel(str(model_path), use_npu=True)
            if self._rknn_model.load():
                # Load tokenizer (CPU side)
                try:
                    from transformers import AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        "ProsusAI/finbert",
                        cache_dir=str(self._model_dir / "tokenizer_cache")
                    )
                    self._use_npu = True
                    self._initialized = True
                    logger.info("[npu] ✅ FinBERT running on NPU")
                    return True
                except Exception as e:
                    logger.warning(f"[npu] Tokenizer load failed: {e}")

        # Fall back to CPU transformers
        try:
            from src.sentiment.sentiment_analyzer import SentimentAnalyzer
            self._cpu_analyzer = SentimentAnalyzer(use_finbert=True)
            self._initialized = True
            logger.info("[npu] FinBERT running on CPU (NPU model not available)")
            return True
        except Exception:
            pass

        # Last resort: keyword
        try:
            from src.sentiment.sentiment_analyzer import SentimentAnalyzer
            self._cpu_analyzer = SentimentAnalyzer(use_finbert=False)
            self._initialized = True
            logger.info("[npu] Using keyword sentiment (CPU fallback)")
            return True
        except Exception as e:
            logger.error(f"[npu] Sentiment analyzer init failed: {e}")
            return False

    def analyze(self, text: str) -> dict:
        """
        Analyze sentiment of text.
        Returns: {"label": str, "score": float, "confidence": float, "device": str}
        """
        if not self._initialized:
            self.initialize()

        if self._use_npu and self._rknn_model and self._tokenizer:
            return self._analyze_npu(text)
        elif self._cpu_analyzer:
            result = self._cpu_analyzer.analyze_text(text)
            return {
                "label": result.label,
                "score": result.score,
                "confidence": result.confidence,
                "device": "CPU",
            }
        else:
            return {"label": "neutral", "score": 0.0, "confidence": 0.0, "device": "none"}

    def _analyze_npu(self, text: str) -> dict:
        """Run FinBERT inference on NPU."""
        try:
            # Tokenize (CPU)
            encoding = self._tokenizer(
                text,
                max_length=self.MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            input_ids      = encoding["input_ids"].astype(np.int32)
            attention_mask = encoding["attention_mask"].astype(np.int32)
            token_type_ids = encoding.get(
                "token_type_ids",
                np.zeros_like(input_ids)
            ).astype(np.int32)

            # NPU inference
            outputs = self._rknn_model.infer([
                input_ids, attention_mask, token_type_ids
            ])

            if outputs is None:
                return {"label": "neutral", "score": 0.0, "confidence": 0.0, "device": "NPU_ERROR"}

            # FinBERT outputs: [negative, neutral, positive]
            logits = outputs[0][0]
            probs  = self._softmax(logits)
            neg, neu, pos = probs[0], probs[1], probs[2]

            score      = float(pos - neg)  # -1 to +1
            confidence = float(max(neg, neu, pos))

            if pos > neg and pos > neu:
                label = "positive"
            elif neg > pos and neg > neu:
                label = "negative"
            else:
                label = "neutral"

            return {
                "label":      label,
                "score":      score,
                "confidence": confidence,
                "device":     "NPU",
                "latency_ms": self._rknn_model.avg_inference_ms,
            }

        except Exception as e:
            logger.error(f"[npu] NPU sentiment inference failed: {e}")
            # Fall back to CPU
            if self._cpu_analyzer:
                result = self._cpu_analyzer.analyze_text(text)
                return {
                    "label": result.label,
                    "score": result.score,
                    "confidence": result.confidence,
                    "device": "CPU_FALLBACK",
                }
            return {"label": "neutral", "score": 0.0, "confidence": 0.0, "device": "error"}

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """Analyze multiple texts (NPU processes sequentially, still faster than CPU)."""
        return [self.analyze(text) for text in texts]

    def get_stats(self) -> dict:
        """Return NPU performance statistics."""
        if self._rknn_model:
            return {
                "device":           self._rknn_model.device,
                "inference_count":  self._rknn_model._inference_count,
                "avg_latency_ms":   round(self._rknn_model.avg_inference_ms, 2),
                "npu_load":         get_npu_load(),
            }
        return {"device": "CPU", "npu_load": "N/A"}


# ── NPU ML Model Accelerator ───────────────────────────────

class NPUMLAccelerator:
    """
    Accelerates sklearn ML model inference using RK3588 NPU via ONNX → RKNN.

    Supported models:
      - MLStrategy (HistGradientBoosting)
      - EnsembleMLStrategy (RandomForest, XGBoost, LogisticRegression)

    Conversion workflow (run on x86_64 PC):
        # 1. Export sklearn model to ONNX
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        onnx_model = convert_sklearn(
            sklearn_model,
            initial_types=[("input", FloatTensorType([None, 22]))]
        )
        with open("ml_strategy.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())

        # 2. Convert ONNX to RKNN (on x86_64 with rknn-toolkit2)
        from rknn.api import RKNN
        rknn = RKNN()
        rknn.config(target_platform='rk3588', mean_values=[[0]*22], std_values=[[1]*22])
        rknn.load_onnx('ml_strategy.onnx')
        rknn.build(do_quantization=False)
        rknn.export_rknn('ml_strategy.rknn')
    """

    def __init__(self, model_dir: str = "data_cache/npu_models"):
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, RKNNModel] = {}
        self._npu_available = check_npu_available()

    def load_model(self, name: str) -> bool:
        """Load a named model (looks for {name}.rknn or {name}.onnx)."""
        rknn_path = self._model_dir / f"{name}.rknn"
        onnx_path = self._model_dir / f"{name}.onnx"

        if rknn_path.exists():
            model = RKNNModel(str(rknn_path), use_npu=self._npu_available)
        elif onnx_path.exists():
            model = RKNNModel(str(rknn_path), use_npu=False)
        else:
            logger.debug(f"[npu] No model file for '{name}' — will use sklearn CPU")
            return False

        if model.load():
            self._models[name] = model
            return True
        return False

    def predict(self, name: str, features: np.ndarray) -> np.ndarray | None:
        """
        Run inference for a named model.
        features: shape (n_samples, n_features) float32
        Returns: probabilities array or None if model not loaded
        """
        if name not in self._models:
            return None

        inputs = [features.astype(np.float32)]
        outputs = self._models[name].infer(inputs)

        if outputs is None:
            return None

        return outputs[0]  # Return probabilities

    def predict_proba_single(self, name: str, features: np.ndarray) -> np.ndarray | None:
        """
        Predict probability for a single sample.
        features: shape (n_features,) or (1, n_features)
        Returns: [p_class0, p_class1] or None
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        result = self.predict(name, features)
        if result is None:
            return None
        return result[0] if len(result.shape) > 1 else result

    def is_loaded(self, name: str) -> bool:
        return name in self._models

    def get_device(self, name: str) -> str:
        if name in self._models:
            return self._models[name].device
        return "not_loaded"

    def release_all(self) -> None:
        for model in self._models.values():
            model.release()
        self._models.clear()


# ── Model Exporter (run on Rock 5B to prepare for conversion) ─

class ModelExporter:
    """
    Exports trained sklearn models to ONNX format.
    Run this on the Rock 5B after training to get .onnx files,
    then transfer to x86_64 PC for RKNN conversion.
    """

    def __init__(self, output_dir: str = "data_cache/npu_models"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export_ml_strategy(self, model, n_features: int = 16,
                            name: str = "ml_strategy") -> str | None:
        """Export MLStrategy sklearn model to ONNX."""
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType

            onnx_model = convert_sklearn(
                model,
                name=name,
                initial_types=[("input", FloatTensorType([None, n_features]))]
            )
            out_path = self._output_dir / f"{name}.onnx"
            with open(out_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            logger.info(f"[npu] Exported {name} to ONNX: {out_path}")
            return str(out_path)

        except ImportError:
            logger.warning("[npu] skl2onnx not installed — run: pip install skl2onnx")
            return None
        except Exception as e:
            logger.error(f"[npu] ONNX export failed: {e}")
            return None

    def export_ensemble(self, rf_model, xgb_model, lr_model,
                        n_features: int = 22) -> dict[str, str]:
        """Export all ensemble models to ONNX."""
        results = {}
        for name, model in [
            ("ensemble_rf", rf_model),
            ("ensemble_xgb", xgb_model),
            ("ensemble_lr", lr_model),
        ]:
            path = self.export_ml_strategy(model, n_features, name)
            if path:
                results[name] = path
        return results

    def generate_conversion_script(self) -> str:
        """
        Generate the RKNN conversion script to run on x86_64 PC.
        Copy the .onnx files to x86_64 PC, run this script,
        then copy the .rknn files back to Rock 5B.
        """
        script = '''#!/usr/bin/env python3
"""
RKNN Conversion Script — run on x86_64 PC with rknn-toolkit2 installed.

Install: pip install rknn-toolkit2

Steps:
  1. Copy .onnx files from Rock 5B to this PC
  2. Run this script
  3. Copy .rknn files back to Rock 5B: data_cache/npu_models/
"""

from rknn.api import RKNN

MODELS = [
    ("ml_strategy.onnx",  "ml_strategy.rknn",  16),
    ("ensemble_rf.onnx",  "ensemble_rf.rknn",  22),
    ("ensemble_xgb.onnx", "ensemble_xgb.rknn", 22),
    ("ensemble_lr.onnx",  "ensemble_lr.rknn",  22),
]

for onnx_file, rknn_file, n_features in MODELS:
    print(f"Converting {onnx_file} → {rknn_file}...")
    rknn = RKNN(verbose=False)

    rknn.config(
        target_platform="rk3588",
        mean_values=[[0] * n_features],
        std_values=[[1] * n_features],
    )

    ret = rknn.load_onnx(model=onnx_file)
    assert ret == 0, f"Failed to load {onnx_file}"

    ret = rknn.build(do_quantization=False)
    assert ret == 0, "Build failed"

    ret = rknn.export_rknn(rknn_file)
    assert ret == 0, f"Export failed for {rknn_file}"

    rknn.release()
    print(f"  ✅ {rknn_file} ready")

print("\\nAll models converted! Copy .rknn files to Rock 5B:")
print("  data_cache/npu_models/")
'''
        script_path = self._output_dir / "convert_to_rknn.py"
        with open(script_path, "w") as f:
            f.write(script)
        logger.info(f"[npu] Conversion script saved: {script_path}")
        return str(script_path)


# ── NPU Manager (main entry point) ────────────────────────

class NPUManager:
    """
    Central manager for all NPU-accelerated tasks in Alpha Trading Platform.

    Usage:
        npu = NPUManager()
        npu.initialize()

        # Sentiment (NPU or CPU fallback)
        result = npu.analyze_sentiment("Apple beats earnings estimates")

        # ML inference (NPU or sklearn fallback)
        proba = npu.predict_ml("ml_strategy", features_array)

        # Status
        print(npu.status())
    """

    def __init__(self, model_dir: str = "data_cache/npu_models"):
        self._model_dir = model_dir
        self._npu_available   = False
        self._sentiment: NPUSentimentAnalyzer | None = None
        self._ml: NPUMLAccelerator | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all NPU components."""
        self._npu_available = check_npu_available()

        logger.info(
            f"[npu] Initializing — NPU {'AVAILABLE ✅' if self._npu_available else 'NOT AVAILABLE (CPU fallback)'}"
        )

        # Sentiment analyzer
        self._sentiment = NPUSentimentAnalyzer(self._model_dir)
        self._sentiment.initialize()

        # ML accelerator
        self._ml = NPUMLAccelerator(self._model_dir)
        for model_name in ["ml_strategy", "ensemble_rf", "ensemble_xgb", "ensemble_lr"]:
            loaded = self._ml.load_model(model_name)
            if loaded:
                logger.info(
                    f"[npu] ML model '{model_name}' loaded on {self._ml.get_device(model_name)}"
                )

        self._initialized = True
        logger.info("[npu] NPU Manager initialized")

    def analyze_sentiment(self, text: str) -> dict:
        """Analyze text sentiment using NPU or CPU fallback."""
        if not self._initialized:
            self.initialize()
        return self._sentiment.analyze(text)

    def analyze_sentiment_batch(self, texts: list[str]) -> list[dict]:
        """Analyze multiple texts."""
        if not self._initialized:
            self.initialize()
        return self._sentiment.analyze_batch(texts)

    def predict_ml(self, model_name: str, features: np.ndarray) -> np.ndarray | None:
        """Run ML model inference on NPU or CPU."""
        if not self._initialized:
            self.initialize()
        return self._ml.predict_proba_single(model_name, features)

    def export_models_for_conversion(self, ml_strategy_model=None,
                                      ensemble_models: dict = None) -> None:
        """
        Export trained sklearn models to ONNX and generate conversion script.
        Run this after training your models to prepare for NPU conversion.
        """
        exporter = ModelExporter(self._model_dir)

        if ml_strategy_model is not None:
            exporter.export_ml_strategy(ml_strategy_model, n_features=16)

        if ensemble_models:
            exporter.export_ensemble(
                rf_model=ensemble_models.get("rf"),
                xgb_model=ensemble_models.get("xgb"),
                lr_model=ensemble_models.get("lr"),
                n_features=22,
            )

        script_path = exporter.generate_conversion_script()
        logger.info(
            f"[npu] Models exported. Run {script_path} on x86_64 PC to convert to RKNN."
        )

    # ── Processed Data Block Integration ──────────────────

    def get_cached_prediction(self, symbol: str) -> dict | None:
        """
        Get pre-computed prediction from the processed data block.
        Instant lookup — no inference needed.

        Returns dict with ml_signal, ensemble_signal, confidence, etc.
        or None if no cached prediction exists.
        """
        try:
            from src.ops.data_processor import get_data_processor
            return get_data_processor().get_prediction(symbol)
        except Exception:
            return None

    def get_cached_features(self, symbol: str, days: int = 365) -> Any:
        """
        Get pre-computed ML features from the processed data block.
        Returns a DataFrame with 22 feature columns, or None.
        """
        try:
            from src.ops.data_processor import get_data_processor
            return get_data_processor().get_features(symbol, days)
        except Exception:
            return None

    def predict_ml_fast(self, model_name: str, symbol: str) -> dict | None:
        """
        Fast prediction path: check processed data cache first,
        fall back to live NPU/CPU inference if cache miss.

        Returns: {signal, probability, confidence, source} or None.
        """
        # Try cached prediction first (instant)
        cached = self.get_cached_prediction(symbol)
        if cached:
            if model_name == "ml_strategy":
                return {
                    "signal": cached.get("ml_signal", "HOLD"),
                    "probability": cached.get("ml_prob_up", 0.5),
                    "confidence": cached.get("ml_confidence", 0),
                    "source": "cache",
                    "device": cached.get("device", "unknown"),
                }
            else:
                return {
                    "signal": cached.get("ensemble_signal", "HOLD"),
                    "probability": cached.get("ensemble_prob_up", 0.5),
                    "confidence": cached.get("ensemble_confidence", 0),
                    "agree": cached.get("ensemble_agree", 0),
                    "source": "cache",
                    "device": cached.get("device", "unknown"),
                }

        # Cache miss — try live inference with cached features
        features_df = self.get_cached_features(symbol, days=30)
        if features_df is not None and len(features_df) > 0:
            last_row = features_df.iloc[-1]
            if model_name == "ml_strategy":
                feature_cols = list(features_df.columns[:16])
            else:
                feature_cols = list(features_df.columns[:22])
            features = np.array(
                [last_row[c] for c in feature_cols], dtype=np.float32,
            )
            features = np.nan_to_num(features, nan=0.0)

            proba = self.predict_ml(model_name, features)
            if proba is not None:
                p = float(proba[1]) if len(proba) > 1 else float(proba[0])
                conf = abs(p - 0.5) * 200
                signal = "BUY" if p > 0.55 else ("SELL" if p < 0.45 else "HOLD")
                return {
                    "signal": signal,
                    "probability": p,
                    "confidence": conf,
                    "source": "live_inference",
                    "device": self._ml.get_device(model_name) if self._ml else "cpu",
                }

        return None

    def get_processor_status(self) -> dict:
        """Get the data processor status (features, predictions, models)."""
        try:
            from src.ops.data_processor import get_data_processor
            return get_data_processor().get_status()
        except Exception as e:
            return {"error": str(e)}

    def status(self) -> dict:
        """Return NPU system status."""
        status = {
            "npu_available": self._npu_available,
            "npu_load":      get_npu_load(),
            "initialized":   self._initialized,
            "sentiment":     {},
            "ml_models":     {},
        }

        if self._sentiment:
            status["sentiment"] = self._sentiment.get_stats()

        if self._ml:
            for name in ["ml_strategy", "ensemble_rf", "ensemble_xgb", "ensemble_lr"]:
                status["ml_models"][name] = self._ml.get_device(name)

        # Include processed data block status
        proc_status = self.get_processor_status()
        if "error" not in proc_status:
            status["processed_data"] = {
                "feature_symbols": proc_status.get("feature_symbols", 0),
                "prediction_symbols": proc_status.get("prediction_symbols", 0),
                "db_size_mb": proc_status.get("db_size_mb", 0),
            }
            if "models" in proc_status:
                status["trained_models"] = proc_status["models"]

        return status

    def print_status(self) -> None:
        """Print NPU status to console."""
        s = self.status()
        print(f"\n{'='*55}")
        print(f"  Rock 5B NPU + Data Processor Status")
        print(f"{'='*55}")
        print(f"  NPU Available:  {'YES' if s['npu_available'] else 'NO (CPU fallback)'}")
        print(f"  NPU Load:       {s['npu_load']}")
        print(f"  Sentiment:      {s['sentiment'].get('device', 'N/A')}")
        if s['sentiment'].get('avg_latency_ms'):
            print(f"  Sentiment ms:   {s['sentiment']['avg_latency_ms']:.1f}ms avg")
        print(f"\n  NPU/RKNN Models:")
        for name, device in s['ml_models'].items():
            icon = "[NPU]" if "NPU" in device else "[CPU]"
            print(f"    {icon} {name:<20} {device}")

        if "processed_data" in s:
            pd_info = s["processed_data"]
            print(f"\n  Processed Data Block:")
            print(f"    Feature symbols:    {pd_info.get('feature_symbols', 0)}")
            print(f"    Prediction symbols: {pd_info.get('prediction_symbols', 0)}")
            print(f"    DB size:            {pd_info.get('db_size_mb', 0)} MB")

        if "trained_models" in s and s["trained_models"]:
            print(f"\n  Trained Models:")
            for name, info in s["trained_models"].items():
                acc = info.get("accuracy", 0) or 0
                auc = info.get("auc", 0) or 0
                print(f"    {name:<20} acc={acc:.3f} auc={auc:.3f} ({info.get('device', '?')})")

        print(f"{'='*55}\n")


# ── Singleton ──────────────────────────────────────────────

_npu_manager: NPUManager | None = None


def get_npu_manager() -> NPUManager:
    """Get or create the global NPU manager instance."""
    global _npu_manager
    if _npu_manager is None:
        _npu_manager = NPUManager()
        _npu_manager.initialize()
    return _npu_manager
