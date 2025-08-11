# model_predictor.py
import json
import numpy as np
import xgboost as xgb

MODEL_PATH = "ml_model.json"
FEATURES_PATH = "features.json"




def _load_model_from_disk():
    """Load the XGBoost model and expected feature list from disk."""
    try:
        model = xgb.Booster()
        model.load_model(MODEL_PATH)
    except ModuleNotFoundError as e:
        if e.name == "xgboost":
            print("❌ Missing dependency 'xgboost'. Install it with 'pip install xgboost' to load the ML model.")
        else:
            print(f"❌ Required module not found: {e.name}")
        return None, []
    except FileNotFoundError:
        print(f"❌ Model file not found at {MODEL_PATH}")
        return None, []
    except Exception as e:
        print(f"❌ Failed to load model from {MODEL_PATH}: {e}")
        return None, []

    try:
        with open(FEATURES_PATH, "r") as f:
            expected_features = json.load(f)
    except FileNotFoundError:
        print(f"❌ Feature list file not found at {FEATURES_PATH}")
        return None, []
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse feature list: {e}")
        return None, []

    if not isinstance(expected_features, (list, tuple)) or not all(isinstance(f, str) for f in expected_features):
        print("❌ Expected features missing or malformed in feature list")
        return None, []

    print(f"🔄 Loaded ML model expecting {len(expected_features)} features: {expected_features}")
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
        print("⚠️ No valid model available, skipping prediction.")
        return None, 0.0, None

    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        print(f"⚠️ Missing features in input: {missing}")
        return None, 0.0, None


    X = df[expected_features].tail(1)
    if X.isnull().any().any():
        print("⚠️ NaNs found in final feature row, skipping prediction.")
        return None, 0.0, None

    try:
        dmatrix = xgb.DMatrix(X)
        class_probs = model.predict(dmatrix)[0]
        predicted_class = int(np.argmax(class_probs))
        confidence = class_probs[predicted_class]

        print(f"🔍 Class probabilities: {dict(enumerate(np.round(class_probs, 3)))}")
        print(f"📊 Predicted class: {predicted_class} with confidence {confidence:.2f}")

        # Logic overrides
        if predicted_class == 1 and confidence < threshold:
            return "HOLD", confidence, predicted_class

        if predicted_class in [3, 4] and confidence >= 0.75:
            print("🔥 High Conviction BUY override active")
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
        print(f"❌ ML prediction failed: {e}")
        return None, 0.0, None




