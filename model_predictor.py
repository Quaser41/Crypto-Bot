# model_predictor.py
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "ml_model.pkl"

# === Load model and expected features ===
def load_model():
    try:
        bundle = joblib.load(MODEL_PATH)
        print(f"DEBUG: Loaded model content type: {type(bundle)}")
    except ModuleNotFoundError as e:
        if e.name == "xgboost":
            print("❌ Missing dependency 'xgboost'. Install it with 'pip install xgboost' to load the ML model.")
        else:
            print(f"❌ Required module not found: {e.name}")
        return None, []
    except FileNotFoundError:
        print(f"❌ Model file not found at {MODEL_PATH}")
        return None, []
    except ValueError as e:
        print(f"❌ Failed to load model from {MODEL_PATH}: {e}")
        return None, []

    if isinstance(bundle, tuple) and len(bundle) == 2:
        model, expected_features = bundle
    elif isinstance(bundle, dict) and "model" in bundle and "features" in bundle:
        model = bundle["model"]
        expected_features = bundle["features"]
    else:
        print("❌ Unsupported model format in ml_model.pkl")
        return None, []

    if not isinstance(expected_features, (list, tuple)) or not all(isinstance(f, str) for f in expected_features):
        print("❌ Expected features missing or malformed in model bundle")
        return None, []

    print(f"🔄 Loaded ML model expecting {len(expected_features)} features: {expected_features}")
    return model, list(expected_features)

# === Predict signal from latest row ===
def predict_signal(df):
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
        class_probs = model.predict_proba(X)[0]
        predicted_class = int(np.argmax(class_probs))
        confidence = class_probs[predicted_class]

        print(f"🔍 Class probabilities: {dict(enumerate(np.round(class_probs, 3)))}")
        print(f"📊 Predicted class: {predicted_class} with confidence {confidence:.2f}")

        # Volatility dynamic threshold
        vol = df.get("Volatility_7d", pd.Series([0.0])).iloc[-1]
        threshold = 0.6 if vol > 0.2 else 0.7
        print(f"🧠 Dynamic threshold: {threshold:.2f} (7d vol={vol:.3f})")

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




