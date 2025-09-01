# model_predictor.py
import json
import os
from functools import lru_cache

import numpy as np
import xgboost as xgb
import asyncio

from utils.logging import get_logger
from utils.prediction_class import PredictionClass
from config import HIGH_CONF_BUY_OVERRIDE

logger = get_logger(__name__)

MODEL_PATH = "ml_model.json"
FEATURES_PATH = "features.json"
LABELS_PATH = "labels.json"
THRESHOLD_TUNE_PATH = "validation_thresholds.json"
CALIBRATION_PARAMS_PATH = "prob_calibration.json"

try:
    DEFAULT_LOG_FREQUENCY = int(os.getenv("PREDICT_SIGNAL_LOG_FREQ", "100"))
except ValueError:
    DEFAULT_LOG_FREQUENCY = 100

_prediction_counter = 0
_last_logged_class = None




def _load_model_from_disk():
    """Load the XGBoost model, its feature list, and label mapping from disk."""
    try:
        model = xgb.Booster()
        model.load_model(MODEL_PATH)
        model_features = list(model.feature_names or [])
    except ModuleNotFoundError as e:
        if e.name == "xgboost":
            logger.error(
                "❌ Missing dependency 'xgboost'. Install it with 'pip install xgboost' to load the ML model."
            )
        else:
            logger.error("❌ Required module not found: %s", e.name)
        return None, [], []
    except FileNotFoundError:
        logger.error("❌ Model file not found at %s", MODEL_PATH)
        return None, [], []
    except Exception as e:
        logger.error("❌ Failed to load model from %s: %s", MODEL_PATH, e)
        return None, [], []

    if not model_features:
        logger.error("❌ Model did not provide feature names; cannot validate inputs")
        return None, [], []

    # Optional cross-check against features.json for debugging only
    try:
        with open(FEATURES_PATH, "r") as f:
            file_features = json.load(f)
        if isinstance(file_features, (list, tuple)) and all(
            isinstance(f, str) for f in file_features
        ):
            if list(file_features) != model_features:
                logger.debug(
                    "ℹ️ features.json differs from model feature names. Model: %s File: %s",
                    model_features,
                    file_features,
                )
        else:
            logger.debug("ℹ️ features.json is malformed; ignoring")
    except FileNotFoundError:
        logger.debug("ℹ️ features.json not found; relying on model feature names")
    except json.JSONDecodeError as e:
        logger.debug("ℹ️ Failed to parse features.json (%s); ignoring", e)

    try:
        with open(LABELS_PATH, "r") as f:
            labels = json.load(f)
    except FileNotFoundError:
        logger.error("❌ Labels file not found at %s", LABELS_PATH)
        return None, [], []
    except json.JSONDecodeError as e:
        logger.error("❌ Failed to parse label mapping: %s", e)
        return None, [], []

    if not isinstance(labels, (list, tuple)) or not all(isinstance(lbl, int) for lbl in labels):
        logger.error("❌ Label mapping missing or malformed")
        return None, [], []

    logger.info(
        "🔄 Loaded ML model with %d features from model metadata: %s",
        len(model_features),
        model_features,
    )
    return model, model_features, list(labels)


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the model, expected feature list, and label mapping."""
    return _load_model_from_disk()


def reload_model():
    """Clear the cached model so it can be reloaded from disk."""
    load_model.cache_clear()


@lru_cache(maxsize=1)
def _load_threshold_overrides():
    """Load tuned threshold values from disk if available.

    The tuning is expected to be performed offline on validation trades and
    stored as a small JSON mapping.  The structure is::

        {
            "threshold": 0.65,
            "high_conf_buy_override": 0.9
        }

    If the file is missing or malformed the defaults provided by the caller
    and :mod:`config` are used.
    """

    try:
        with open(THRESHOLD_TUNE_PATH, "r") as f:
            data = json.load(f)
        overrides = {}
        if "threshold" in data:
            overrides["threshold"] = float(data["threshold"])
        if "high_conf_buy_override" in data:
            overrides["high_conf_buy_override"] = float(
                data["high_conf_buy_override"]
            )
        return overrides
    except FileNotFoundError:
        logger.debug("ℹ️ Tuned thresholds file not found at %s", THRESHOLD_TUNE_PATH)
    except Exception as e:
        logger.debug("ℹ️ Failed to load tuned thresholds: %s", e)
    return {}


@lru_cache(maxsize=1)
def _load_calibration_params():
    """Load probability calibration parameters from disk.

    The calibration parameters allow post-hoc calibration of the raw
    probabilities emitted by the model.  Two simple methods are supported:

    * ``"platt"`` – logistic calibration with coefficients ``A`` and ``B``.
    * ``"isotonic"`` – monotonic piecewise-linear mapping defined by ``x`` and
      ``y`` arrays.
    """

    try:
        with open(CALIBRATION_PARAMS_PATH, "r") as f:
            params = json.load(f)
        return params
    except FileNotFoundError:
        logger.debug(
            "ℹ️ Calibration params file not found at %s", CALIBRATION_PARAMS_PATH
        )
    except Exception as e:
        logger.debug("ℹ️ Failed to load calibration params: %s", e)
    return {}


def _calibrate_confidence(prob: float) -> float:
    """Apply probability calibration if parameters are available."""

    params = _load_calibration_params()
    if not params:
        return prob

    method = str(params.get("method", "platt")).lower()
    try:
        if method == "platt":
            A = float(params.get("A"))
            B = float(params.get("B"))
            calibrated = 1.0 / (1.0 + np.exp(A * prob + B))
            return float(np.clip(calibrated, 0.0, 1.0))
        elif method == "isotonic":
            x = params.get("x")
            y = params.get("y")
            if not x or not y or len(x) != len(y):
                return prob
            calibrated = np.interp(prob, x, y)
            return float(np.clip(calibrated, 0.0, 1.0))
    except Exception as e:
        logger.debug("ℹ️ Failed to calibrate probability: %s", e)
    return prob


# === Predict signal from latest row ===
def predict_signal(df, threshold, log_frequency=None):
    """Predict the trading signal for the latest row in ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature dataframe containing all expected model features.
    threshold : float
        Confidence threshold that the caller has already computed.
    log_frequency : int | None, optional
        How often to emit info-level prediction logs. ``None`` uses the value
        from the ``PREDICT_SIGNAL_LOG_FREQ`` environment variable. ``0`` or a
        negative value disables these logs entirely.
    """

    model, expected_features, label_mapping = load_model()
    if model is None or not expected_features or not label_mapping:
        logger.warning("⚠️ No valid model available, skipping prediction.")
        return None, 0.0, None

    # Allow thresholds tuned on validation data to override defaults
    overrides = _load_threshold_overrides()
    tuned_threshold = overrides.get("threshold")
    tuned_buy_override = overrides.get("high_conf_buy_override")
    if tuned_threshold is not None:
        threshold = tuned_threshold
    high_conf_buy = (
        tuned_buy_override if tuned_buy_override is not None else HIGH_CONF_BUY_OVERRIDE
    )

    four_hour_cols = ["SMA_4h", "MACD_4h", "Signal_4h", "Hist_4h"]
    missing = [f for f in expected_features if f not in df.columns]
    missing_non_optional = [m for m in missing if m not in four_hour_cols]
    if missing_non_optional:
        logger.warning("⚠️ Missing features in input: %s", missing_non_optional)
        return None, 0.0, None
    for col in set(missing) & set(four_hour_cols):
        logger.debug("Filling missing %s with default 0.0", col)
        df[col] = 0.0

    X = df[expected_features].tail(1).copy()
    for col in four_hour_cols:
        if col in X.columns and X[col].isna().any():
            logger.debug("Replacing NaN %s with 0.0 for prediction", col)
            X[col] = X[col].fillna(0.0)
    if X.isnull().any().any():
        logger.warning("⚠️ NaNs found in final feature row, skipping prediction.")
        return None, 0.0, None

    try:
        dmatrix = xgb.DMatrix(X)
        class_probs = model.predict(dmatrix)[0]
        pred_idx = int(np.argmax(class_probs))
        try:
            original_label = label_mapping[pred_idx]
            predicted_class = PredictionClass(int(original_label))
        except (IndexError, ValueError) as e:
            logger.error("❌ Invalid label mapping for index %d: %s", pred_idx, e)
            return None, 0.0, None
        raw_confidence = float(class_probs[pred_idx])
        confidence = _calibrate_confidence(raw_confidence)
        if abs(confidence - raw_confidence) > 1e-6:
            logger.debug(
                "🔧 Calibrated confidence from %.3f to %.3f",
                raw_confidence,
                confidence,
            )

        logger.debug(
            "🔍 Class probabilities: %s",
            {int(lbl): float(np.round(prob, 3)) for lbl, prob in zip(label_mapping, class_probs)},
        )
        global _prediction_counter, _last_logged_class
        _prediction_counter += 1

        logger.debug(
            "📊 Predicted class: %d with confidence %.2f",
            predicted_class.value,
            confidence,
        )

        if log_frequency is None:
            log_frequency = DEFAULT_LOG_FREQUENCY

        if log_frequency and log_frequency > 0:
            if (
                predicted_class != _last_logged_class
                or _prediction_counter % log_frequency == 0
            ):
                logger.info(
                    "📊 Predicted class: %d with confidence %.2f",
                    predicted_class.value,
                    confidence,
                )
                _last_logged_class = predicted_class

        # Logic overrides
        if predicted_class == PredictionClass.SMALL_LOSS and confidence < threshold:
            return "HOLD", confidence, predicted_class.value

        if (
            predicted_class in (PredictionClass.SMALL_GAIN, PredictionClass.BIG_GAIN)
            and confidence >= high_conf_buy
        ):
            # Logging handled at caller layer to avoid duplicate messages
            logger.debug("🔥 High Conviction BUY override active")
            return "BUY", confidence, predicted_class.value

        if predicted_class == PredictionClass.BIG_GAIN:
            return "BUY", confidence, predicted_class.value
        elif predicted_class == PredictionClass.SMALL_GAIN and confidence >= threshold:
            return "BUY", confidence, predicted_class.value
        elif (
            predicted_class in (PredictionClass.BIG_LOSS, PredictionClass.SMALL_LOSS)
            and confidence >= threshold
        ):
            return "SELL", confidence, predicted_class.value
        else:
            return "HOLD", confidence, predicted_class.value

    except Exception as e:
        logger.error("❌ ML prediction failed: %s", e)
        return None, 0.0, None


async def predict_signal_async(df, threshold, log_frequency=None):
    """Asynchronous helper for :func:`predict_signal`.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature dataframe containing all expected model features.
    threshold : float
        Confidence threshold that the caller has already computed.
    log_frequency : int | None, optional
        Passed through to :func:`predict_signal`.
    """

    return await asyncio.to_thread(predict_signal, df, threshold, log_frequency)




