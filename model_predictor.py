# model_predictor.py
import json
from functools import lru_cache

import numpy as np
import xgboost as xgb
import asyncio

from utils.logging import get_logger

logger = get_logger(__name__)

MODEL_PATH = "ml_model.json"
FEATURES_PATH = "features.json"




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
def predict_signal(df, threshold):
    """Predict the trading signal for the latest row in ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature dataframe containing all expected model features.
    threshold : float
        Confidence threshold that the caller has already computed.
    """

    model, expected_features = load_model()
    if model is None or not expected_features:
        logger.warning("⚠️ No valid model available, skipping prediction.")
        return None, 0.0, None

    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        logger.warning("⚠️ Missing features in input: %s", missing)
        return None, 0.0, None


    X = df[expected_features].tail(1)
    if X.isnull().any().any():
        logger.warning("⚠️ NaNs found in final feature row, skipping prediction.")
        return None, 0.0, None

    try:
        dmatrix = xgb.DMatrix(X)
        class_probs = model.predict(dmatrix)[0]
        predicted_class = int(np.argmax(class_probs))
        confidence = class_probs[predicted_class]

        logger.info(
            "🔍 Class probabilities: %s",
            dict(enumerate(np.round(class_probs, 3))),
        )
        logger.info(
            "📊 Predicted class: %d with confidence %.2f",
            predicted_class,
            confidence,
        )

        # Logic overrides
        if predicted_class == 1 and confidence < threshold:
            return "HOLD", confidence, predicted_class

        if predicted_class in [3, 4] and confidence >= 0.75:
            logger.info("🔥 High Conviction BUY override active")
            return "BUY", confidence, predicted_class

        if predicted_class == 4:
            return "BUY", confidence, predicted_class
        elif predicted_class == 3 and confidence >= threshold:
            return "BUY", confidence, predicted_class
        elif predicted_class in [0, 1] and confidence >= threshold:
            return "SELL", confidence, predicted_class
        else:
            return "HOLD", confidence, predicted_class

    except Exception as e:
        logger.error("❌ ML prediction failed: %s", e)
        return None, 0.0, None


async def predict_signal_async(df, threshold):
    """Asynchronous helper for :func:`predict_signal`."""
    return await asyncio.to_thread(predict_signal, df, threshold)




