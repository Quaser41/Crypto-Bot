# train_real_model.py

import pandas as pd
from data_fetcher import fetch_ohlcv_smart
from feature_engineer import add_indicators
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import resample
from xgboost import XGBClassifier
import json
import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)

# === Feature configuration ===
DEFAULT_FEATURES = [
    "RSI", "MACD", "Signal", "Hist",
    "SMA_20", "SMA_50",
    "Return_1d", "Return_2d", "Return_3d", "Return_5d", "Return_7d",
    "Price_vs_SMA20", "Price_vs_SMA50",
    "Volatility_7d", "MACD_Hist_norm"
]


def load_feature_list():
    """Load feature column names from ``features.json`` if available.

    Falls back to :data:`DEFAULT_FEATURES` when the file is missing or
    malformed. This keeps the training pipeline in sync with the
    inference pipeline, which also relies on ``features.json``.
    """
    try:
        with open("features.json", "r") as f:
            features = json.load(f)
        if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
            raise ValueError("features.json is not a list of strings")
        logger.info("📄 Loaded %d features from features.json", len(features))
        return features
    except FileNotFoundError:
        logger.warning("⚠️ features.json not found; using default feature set")
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("⚠️ Could not parse features.json (%s); using default feature set", e)
    return DEFAULT_FEATURES

# === Label encoding function (tight, short-term focused) ===
def return_bucket(r):
    if r <= -0.06:
        return 0  # Big loss
    elif r <= -0.02:
        return 1  # Small loss
    elif r < 0.01:
        return 2  # Neutral
    elif r < 0.05:
        return 3  # Small gain
    else:
        return 4  # Big gain

def prepare_training_data(symbol, coin_id):
    logger.info("\n⏳ Preparing data for %s...", coin_id)
    df = fetch_ohlcv_smart(symbol=symbol, coin_id=coin_id, days=730)

    if df.empty:
        logger.warning("⚠️ No data for %s", coin_id)
        return None, None

    df = add_indicators(df)
    df = df.copy()
    if df.empty:
        logger.warning("⚠️ Failed to calculate indicators for %s", coin_id)
        return None, None

    df.loc[:, "Future_Close"] = df["Close"].shift(-3)  # 🔁 3-day ahead for short-term trading
    df.loc[:, "Return"] = (df["Future_Close"] - df["Close"]) / df["Close"]
    df = df[df["Return"].abs() > 0.005]
    df = df.dropna()

    df["Target"] = df["Return"].apply(return_bucket)

    feature_cols = load_feature_list()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("⚠️ Missing features in data: %s", missing)
    X = df[[c for c in feature_cols if c in df.columns]]
    y = df["Target"]
    return X, y

def train_model(X, y):
    logger.info("\n🚀 Training multi-class classifier...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_count = len(le.classes_)
    logger.info("✅ Model will use %d classes: %s", class_count, list(le.classes_))

    logger.info("📊 Original class distribution:")
    class_dist = pd.Series(y_encoded).value_counts().sort_index()
    logger.info("%s", class_dist)

    rare_classes = class_dist[class_dist < 5].index.tolist()
    if rare_classes:
        logger.warning("⚠️ Dropping rare classes: %s", rare_classes)
        keep_idx = ~pd.Series(y_encoded).isin(rare_classes).values
        X = X[keep_idx]
        y_encoded = y_encoded[keep_idx]
        class_dist = pd.Series(y_encoded).value_counts().sort_index()

    # === Balance classes via upsampling ===
    df_xy = X.copy()
    df_xy["Target"] = y_encoded
    max_size = df_xy["Target"].value_counts().max()
    balanced_parts = []
    for cls, part in df_xy.groupby("Target"):
        balanced_parts.append(resample(part, replace=True, n_samples=max_size, random_state=42))
    df_bal = pd.concat(balanced_parts).sample(frac=1, random_state=42)  # shuffle
    X_bal = df_bal.drop(columns=["Target"])
    y_bal = df_bal["Target"].astype(int)

    logger.info("📊 Balanced class distribution:")
    logger.info("%s", y_bal.value_counts().sort_index())

    # === Time-based train/test split ===
    split_idx = int(len(X_bal) * 0.8)
    X_train, X_test = X_bal.iloc[:split_idx], X_bal.iloc[split_idx:]
    y_train, y_test = y_bal.iloc[:split_idx], y_bal.iloc[split_idx:]

    present_classes = np.unique(y_train)
    if len(present_classes) < class_count:
        logger.warning(
            "⚠️ Not all classes present in training split: %s. Skipping sample weighting.",
            present_classes,
        )
        sample_weights = None
    else:
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        class_weights = (
            pd.Series(sample_weights, index=y_train)
            .groupby(level=0)
            .mean()
            .to_dict()
        )
        logger.info("⚖️ Class weights: %s", class_weights)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob',
        num_class=len(np.unique(y_encoded)),
        random_state=42,
        eval_metric='mlogloss'
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    preds = model.predict(X_test)

    logger.info("\n📊 Classification Report:")
    logger.info("%s", classification_report(y_test, preds, digits=3))
    logger.info("📊 Prediction distribution: %s", pd.Series(preds).value_counts().sort_index())

    return model

def main():
    coins = [
        ("btc", "bitcoin"),
        ("eth", "ethereum"),
        ("sol", "solana"),
        ("doge", "dogecoin"),
        ("pepe", "pepe"),
        ("bonk", "bonk"),
        ("floki", "floki"),
        ("avax", "avalanche-2"),
        ("link", "chainlink"),
        # Removed INJ due to failure
        # Added short-trading friendly alts
        ("ada", "cardano"),
        ("sui", "sui"),
        ("apt", "aptos"),
        ("arb", "arbitrum")
    ]

    X_list = []
    y_list = []

    for symbol, coin_id in coins:
        X, y = prepare_training_data(symbol, coin_id)
        if X is not None and y is not None:
            X_list.append(X)
            y_list.append(y)

    X_list = [x for x in X_list if x is not None and not x.empty]
    y_list = [y for y in y_list if y is not None and not y.empty]

    if not X_list:
        logger.warning("⚠️ No usable data, aborting.")
        return

    X_all = pd.concat(X_list)
    y_all = pd.concat(y_list)

    model = train_model(X_all, y_all)
    model.save_model("ml_model.json")
    with open("features.json", "w") as f:
        json.dump(X_all.columns.tolist(), f)
    logger.info("💾 Saved multi-class model and feature list")

if __name__ == "__main__":
    main()
