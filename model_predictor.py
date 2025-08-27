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

try:
    DEFAULT_LOG_FREQUENCY = int(os.getenv("PREDICT_SIGNAL_LOG_FREQ", "100"))
except ValueError:
    DEFAULT_LOG_FREQUENCY = 100

_prediction_counter = 0
_last_logged_class = None




def _load_model_from_disk():
    """Load the XGBoost model and expected feature list from disk."""
    try:
        model = xgb.Booster()
        model.load_model(MODEL_PATH)
    except ModuleNotFoundError as e:
        if e.name == "xgboost":
            logger.error("❌ Missing dependency 'xgboost'. Install it with 'pip install xgboost' to load the ML model.")
        else:
            logger.error("❌ Required module not found: %s", e.name)
        return None, []
    except FileNotFoundError:
        logger.error("❌ Model file not found at %s", MODEL_PATH)
        return None, []
    except Exception as e:
        logger.error("❌ Failed to load model from %s: %s", MODEL_PATH, e)
        return None, []

    try:
        with open(FEATURES_PATH, "r") as f:
            expected_features = json.load(f)
    except FileNotFoundError:
        logger.error("❌ Feature list file not found at %s", FEATURES_PATH)
        return None, []
    except json.JSONDecodeError as e:
        logger.error("❌ Failed to parse feature list: %s", e)
        return None, []

    if not isinstance(expected_features, (list, tuple)) or not all(isinstance(f, str) for f in expected_features):
        logger.error("❌ Expected features missing or malformed in feature list")
        return None, []

    logger.info(
        "🔄 Loaded ML model expecting %d features: %s",
        len(expected_features),
        expected_features,
    )
    return model, list(expected_features)


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the model and expected feature list."""
    return _load_model_from_disk()


def reload_model():
    """Clear the cached model so it can be reloaded from disk."""
    load_model.cache_clear()


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

    model, expected_features = load_model()
    if model is None or not expected_features:
        logger.warning("⚠️ No valid model available, skipping prediction.")
        return None, 0.0, None

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
        predicted_class = PredictionClass(int(np.argmax(class_probs)))
        confidence = class_probs[predicted_class.value]

        logger.debug(
            "🔍 Class probabilities: %s",
            dict(enumerate(np.round(class_probs, 3))),
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

        if predicted_class in (PredictionClass.SMALL_GAIN, PredictionClass.BIG_GAIN) and confidence >= HIGH_CONF_BUY_OVERRIDE:
            # Logging handled at caller layer to avoid duplicate messages
            logger.debug("🔥 High Conviction BUY override active")
            return "BUY", confidence, predicted_class.value

        if predicted_class == PredictionClass.BIG_GAIN:
            return "BUY", confidence, predicted_class.value
        elif predicted_class == PredictionClass.SMALL_GAIN and confidence >= threshold:
            return "BUY", confidence, predicted_class.value
        elif predicted_class in (PredictionClass.BIG_LOSS, PredictionClass.SMALL_LOSS) and confidence >= threshold:
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




