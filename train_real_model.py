# train_real_model.py

import pandas as pd
from data_fetcher import fetch_ohlcv_smart
from feature_engineer import add_indicators
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import json
import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)

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

    feature_cols = [
        "RSI", "MACD", "Signal", "Hist",
        "SMA_20", "SMA_50",
        "Return_1d", "Return_2d", "Return_3d", "Return_5d", "Return_7d",
        "Price_vs_SMA20", "Price_vs_SMA50",
        "Volatility_7d", "MACD_Hist_norm",
        "FearGreed_norm", "TxVolume_norm", "ActiveAddresses_norm"
    ]

    X = df[feature_cols]
    y = df["Target"]
    return X, y

def train_model(X, y):
    logger.info("\n🚀 Training multi-class classifier...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_count = len(le.classes_)
    logger.info("✅ Model will use %d classes: %s", class_count, list(le.classes_))

    logger.info("📊 Class distribution:")
    class_dist = pd.Series(y_encoded).value_counts().sort_index()
    logger.info("%s", class_dist)

    rare_classes = class_dist[class_dist < 5].index.tolist()
    if rare_classes:
        logger.warning("⚠️ Dropping rare classes: %s", rare_classes)
        keep_idx = ~pd.Series(y_encoded).isin(rare_classes).values
        X = X[keep_idx]
        y_encoded = y_encoded[keep_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    present_classes = np.unique(y_train)
    if len(present_classes) < class_count:
        logger.warning(
            "⚠️ Not all classes present in training split: %s. Skipping sample weighting.",
            present_classes,
        )
        sample_weights = None
    else:
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        logger.info(
            "⚖️ Class weights: %s",
            dict(zip(present_classes, compute_sample_weight(class_weight='balanced', y=y_train))),
        )

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
