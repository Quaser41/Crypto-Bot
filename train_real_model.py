# train_real_model.py

import os
import pandas as pd
from data_fetcher import fetch_ohlcv_smart
from feature_engineer import add_indicators
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
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
    "Volatility_7d", "MACD_Hist_norm",
    "MACD_4h", "Signal_4h", "Hist_4h", "SMA_4h"
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
    if r <= -0.05:
        return 0  # Big loss
    elif r <= -0.02:
        return 1  # Small loss
    elif r < 0.01:
        return 2  # Neutral
    elif r < 0.04:
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

    class_counts = df["Target"].value_counts()
    for cls in [0, 3]:
        cnt = class_counts.get(cls, 0)
        if cnt < 50:
            logger.info(
                "⚠️ Class %d has %d samples; augmenting to reach 50", cls, cnt
            )
            subset = df[df["Target"] == cls]
            if not subset.empty:
                augmented = resample(subset, replace=True, n_samples=50 - cnt, random_state=42)
                df = pd.concat([df, augmented], ignore_index=True)
            else:
                logger.warning("⚠️ No data available to augment class %d", cls)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(
        "📊 Class distribution after augmentation: %s", df["Target"].value_counts().sort_index()
    )

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

    # === Time-based train/test split (chronological) ===
    df_xy = X.copy()
    df_xy["Target"] = y_encoded
    split_idx = int(len(df_xy) * 0.8)
    train_df, test_df = df_xy.iloc[:split_idx], df_xy.iloc[split_idx:]

    # === No upsampling: rely on class weights for imbalance handling ===
    X_train = train_df.drop(columns=["Target"])
    y_train = train_df["Target"].astype(int)

    logger.info("📊 Training class distribution:")
    logger.info("%s", y_train.value_counts().sort_index())

    X_test = test_df.drop(columns=["Target"])
    y_test = test_df["Target"].astype(int)

    # === Walk-forward cross-validation ===
    try:
        n_splits = min(5, max(2, len(X_train) // 50))
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            fold_weights = compute_sample_weight(class_weight="balanced", y=y_tr)
            cv_model = XGBClassifier(
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
            cv_model.fit(X_tr, y_tr, sample_weight=fold_weights)
            cv_preds = cv_model.predict(X_val)
            logger.info(
                "📊 CV Fold %d classification report:\n%s",
                fold,
                classification_report(y_val, cv_preds, digits=3),
            )
    except Exception as e:
        logger.warning("⚠️ Time series cross-validation failed: %s", e)

    # === Final model training with class weights ===
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

    label_map = {
        0: "big_loss",
        1: "small_loss",
        2: "neutral",
        3: "small_gain",
        4: "big_gain",
    }
    labels_sorted = sorted(np.unique(np.concatenate([y_test.values, preds])))
    target_names = [label_map[int(lbl)] for lbl in labels_sorted]

    report = classification_report(
        y_test, preds, labels=labels_sorted, target_names=target_names, digits=3
    )
    cm = confusion_matrix(y_test, preds, labels=labels_sorted)

    logger.info("\n📊 Classification Report:\n%s", report)
    logger.info("🔁 Confusion Matrix:\n%s", cm)
    logger.info(
        "📊 Prediction distribution: %s", pd.Series(preds).value_counts().sort_index()
    )

    os.makedirs("analytics", exist_ok=True)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
        os.path.join("analytics", "confusion_matrix.csv"), index_label="actual"
    )
    with open(os.path.join("analytics", "classification_report.txt"), "w") as f:
        f.write(report)
    logger.info("📁 Saved diagnostics to analytics/")

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
