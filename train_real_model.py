# train_real_model.py

import pandas as pd
from data_fetcher import fetch_ohlcv_smart
from feature_engineer import add_indicators
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import numpy as np

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
    print(f"\n⏳ Preparing data for {coin_id}...")
    df = fetch_ohlcv_smart(symbol=symbol, coin_id=coin_id, days=730)

    if df.empty:
        print(f"⚠️ No data for {coin_id}")
        return None, None

    df = add_indicators(df)
    df = df.copy()
    if df.empty:
        print(f"⚠️ Failed to calculate indicators for {coin_id}")
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
        "Volatility_7d", "MACD_Hist_norm"
    ]

    X = df[feature_cols]
    y = df["Target"]
    return X, y

def train_model(X, y):
    print("\n🚀 Training multi-class classifier...")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_count = len(le.classes_)
    print(f"✅ Model will use {class_count} classes: {list(le.classes_)}")

    print("📊 Class distribution:")
    class_dist = pd.Series(y_encoded).value_counts().sort_index()
    print(class_dist)

    rare_classes = class_dist[class_dist < 5].index.tolist()
    if rare_classes:
        print(f"⚠️ Dropping rare classes: {rare_classes}")
        keep_idx = ~pd.Series(y_encoded).isin(rare_classes).values
        X = X[keep_idx]
        y_encoded = y_encoded[keep_idx]

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    present_classes = np.unique(y_train)
    if len(present_classes) < class_count:
        print(f"⚠️ Not all classes present in training split: {present_classes}. Skipping sample weighting.")
        sample_weights = None
    else:
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        print(f"⚖️ Class weights: {dict(zip(present_classes, compute_sample_weight(class_weight='balanced', y=y_train)))}")

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

    print("\n📊 Classification Report:")
    print(classification_report(y_test, preds, digits=3))

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
        print("⚠️ No usable data, aborting.")
        return

    X_all = pd.concat(X_list)
    y_all = pd.concat(y_list)

    model = train_model(X_all, y_all)
    joblib.dump((model, X_all.columns.tolist()), "ml_model.pkl")
    print("💾 Saved multi-class model as 'ml_model.pkl'")

if __name__ == "__main__":
    main()
