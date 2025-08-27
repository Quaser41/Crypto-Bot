# train_real_model.py

import os
import json
from typing import Optional

import numpy as np
import pandas as pd
from data_fetcher import fetch_ohlcv_smart
from feature_engineer import add_indicators
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import resample
from xgboost import XGBClassifier

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
    """Bucket returns into coarse performance classes.

    The thresholds are deliberately moderate to avoid creating extremely rare
    classes which complicate training.  Extreme buckets may be merged later if
    the dataset lacks examples.
    """
    if r <= -0.04:
        return 0  # Big loss
    elif r <= -0.015:
        return 1  # Loss
    elif r < 0.015:
        return 2  # Neutral
    elif r < 0.04:
        return 3  # Gain
    else:
        return 4  # Big gain


def prepare_training_data(symbol, coin_id, oversampler: Optional[str] = None):
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

    # Merge extreme buckets if completely absent
    class_counts = df["Target"].value_counts()
    if class_counts.get(0, 0) == 0 and class_counts.get(1, 0) > 0:
        logger.warning("⚠️ Merging small losses into big loss bucket for %s", coin_id)
        df.loc[df["Target"] == 1, "Target"] = 0
    if class_counts.get(4, 0) == 0 and class_counts.get(3, 0) > 0:
        logger.warning("⚠️ Merging small gains into big gain bucket for %s", coin_id)
        df.loc[df["Target"] == 3, "Target"] = 4
    class_counts = df["Target"].value_counts()

    # If after merging we still lack extreme classes, skip this coin
    if class_counts.get(0, 0) == 0 or class_counts.get(4, 0) == 0:
        logger.warning("⚠️ Missing extreme classes for %s; skipping", coin_id)
        return None, None

    feature_cols = load_feature_list()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning("⚠️ Missing features in data: %s", missing)
    X = df[[c for c in feature_cols if c in df.columns]]
    y = df["Target"]

    # Optional oversampling with SMOTE/ADASYN
    class_counts = y.value_counts()
    rare_classes = [cls for cls, cnt in class_counts.items() if cnt < 50]
    if oversampler in {"smote", "adasyn"} and rare_classes:
        try:
            from imblearn.over_sampling import SMOTE, ADASYN

            sampler_cls = SMOTE if oversampler == "smote" else ADASYN
            strategy = {cls: 50 for cls in rare_classes if class_counts[cls] > 1}
            if strategy:
                sampler = sampler_cls(random_state=42, sampling_strategy=strategy)
                X, y = sampler.fit_resample(X, y)
                class_counts = y.value_counts()
                logger.info("📈 Applied %s oversampling", oversampler.upper())
        except Exception as e:
            logger.warning("⚠️ %s oversampling failed: %s", oversampler, e)

    # Fallback simple resampling for remaining minority classes
    class_counts = y.value_counts()
    for cls, cnt in class_counts.items():
        if cnt < 50:
            logger.info(
                "⚠️ Class %d has %d samples; augmenting to reach 50", cls, cnt
            )
            idx = y[y == cls].index
            if len(idx) == 0:
                logger.warning(
                    "⚠️ No data available to augment class %d for %s; skipping", cls, coin_id
                )
                return None, None
            subset = pd.concat([X.loc[idx], y.loc[idx]], axis=1)
            augmented = resample(
                subset, replace=True, n_samples=50 - cnt, random_state=42
            )
            X = pd.concat([X, augmented.drop(columns=["Target"])], ignore_index=True)
            y = pd.concat([y, augmented["Target"]], ignore_index=True)

    df_aug = pd.concat([X, y], axis=1).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(
        "📊 Class distribution after augmentation: %s", df_aug["Target"].value_counts().sort_index()
    )

    X = df_aug.drop(columns=["Target"])
    y = df_aug["Target"]
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
        y = y[keep_idx]
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_count = len(le.classes_)
        logger.info("✅ Using %d classes after drop: %s", class_count, list(le.classes_))
        class_dist = pd.Series(y_encoded).value_counts().sort_index()

    original_label_names = {
        0: "big_loss",
        1: "small_loss",
        2: "neutral",
        3: "small_gain",
        4: "big_gain",
    }
    label_map = {i: original_label_names[cls] for i, cls in enumerate(le.classes_)}

    # === Time-based train/test split (chronological) ===
    df_xy = X.copy()
    df_xy["Target"] = y
    split_idx = int(len(df_xy) * 0.8)
    train_df, test_df = df_xy.iloc[:split_idx], df_xy.iloc[split_idx:]

    # === No upsampling: rely on class weights for imbalance handling ===
    X_train = train_df.drop(columns=["Target"])
    y_train_raw = train_df["Target"]

    y_train = le.transform(y_train_raw)

    logger.info("📊 Training class distribution:")
    logger.info("%s", pd.Series(y_train).value_counts().sort_index())

    X_test = test_df.drop(columns=["Target"])
    y_test_raw = test_df["Target"]
    y_test = le.transform(y_test_raw)

    # === Walk-forward cross-validation ===
    n_splits = min(5, max(2, len(X_train) // 50))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr_raw, y_val_raw = y_train_raw.iloc[tr_idx], y_train_raw.iloc[val_idx]

        fold_le = LabelEncoder()
        y_tr = fold_le.fit_transform(y_tr_raw)
        y_val = fold_le.transform(y_val_raw)

        present_classes = set(y_tr_raw.unique())
        missing_classes = set(le.classes_) - present_classes
        if missing_classes:
            logger.warning(
                "⚠️ Fold %d missing classes: %s", fold, list(missing_classes)
            )

        if len(fold_le.classes_) < 2 or len(np.unique(y_val)) < 2:
            logger.warning(
                "⚠️ Fold %d skipped due to insufficient class variety", fold
            )
            continue

        fold_weights = compute_sample_weight(class_weight="balanced", y=y_tr)
        cv_model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=len(fold_le.classes_),
            random_state=42,
            eval_metric='mlogloss'
        )
        try:
            cv_model.fit(X_tr, y_tr, sample_weight=fold_weights)
            cv_preds_enc = cv_model.predict(X_val)
            cv_preds = fold_le.inverse_transform(cv_preds_enc)
            fold_label_map = {
                cls: original_label_names[cls] for cls in fold_le.classes_
            }
            logger.info(
                "📊 CV Fold %d classification report:\n%s",
                fold,
                classification_report(
                    y_val_raw,
                    cv_preds,
                    labels=sorted(fold_le.classes_),
                    target_names=[fold_label_map[cls] for cls in sorted(fold_le.classes_)],
                    digits=3,
                ),
            )
        except Exception as e:
            logger.warning("⚠️ CV fold %d failed: %s", fold, e)

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
        num_class=len(le.classes_),
        random_state=42,
        eval_metric='mlogloss'
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)
    preds = model.predict(X_test)

    labels_sorted = sorted(np.unique(np.concatenate([y_test, preds])))
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
