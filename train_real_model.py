# train_real_model.py

import os
import json
import argparse
import sys
from typing import Optional

import numpy as np
import pandas as pd
import requests
import data_fetcher
from data_fetcher import fetch_ohlcv_smart
import symbol_resolver
from config import MIN_24H_VOLUME
from feature_engineer import add_indicators
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import resample
from xgboost import XGBClassifier
from analytics.calibration_utils import calibrate_and_analyze

import joblib
from joblib.externals.loky.process_executor import TerminatedWorkerError
BackendError = getattr(joblib.parallel, "BackendError", Exception)

from utils.logging import get_logger
from threshold_utils import compute_return_thresholds

logger = get_logger(__name__)

try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
except ImportError:  # pragma: no cover - handled gracefully at runtime
    SMOTE = ADASYN = BorderlineSMOTE = None
    logger.warning(
        "imbalanced-learn is not installed. Run 'pip install imbalanced-learn' to enable oversampling"
    )

# === Feature configuration ===
DEFAULT_FEATURES = [
    "RSI", "MACD", "Signal", "Hist",
    "SMA_20", "SMA_50",
    "Return_1d", "Return_2d", "Return_3d", "Return_5d", "Return_7d",
    "Price_vs_SMA20", "Price_vs_SMA50",
    "Volatility_7d", "MACD_Hist_norm",
    "MACD_4h", "Signal_4h", "Hist_4h", "SMA_4h",
    "BB_Upper", "BB_Middle", "BB_Lower",
    "EMA_9", "EMA_26",
    "OBV", "Volume_vs_SMA20", "RelStrength_BTC"
]

# Optional columns that may not be present for all assets (e.g., higher timeframe
# metrics). Missing these will be tolerated.
OPTIONAL_FEATURES = {"MACD_4h", "Signal_4h", "Hist_4h", "SMA_4h"}


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


def get_volume_ranked_symbols(limit: int | None = None):
    """Return Binance.US symbols ordered by 24h quote volume."""
    symbol_resolver.load_binance_us_symbols()
    try:
        r = requests.get("https://api.binance.us/api/v3/ticker/24hr", timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:  # pragma: no cover - network
        logger.error("❌ Failed to fetch Binance.US volume data: %s", e)
        return []

    volumes = []
    for item in data:
        sym = item.get("symbol", "")
        if sym.endswith("USDT"):
            base = sym[:-4].lower()
            try:
                vol = float(item.get("quoteVolume", 0))
            except (TypeError, ValueError):
                vol = 0
            volumes.append((base, vol))

    volumes.sort(key=lambda x: x[1], reverse=True)
    return volumes if limit is None else volumes[:limit]

# === Label encoding function (tight, short-term focused) ===
def return_bucket(r, thresholds):
    """Bucket returns into performance classes using dynamic thresholds.

    Parameters
    ----------
    r : float
        Return value to bucket.
    thresholds : dict
        Mapping with keys ``"big_loss"``, ``"loss"``, ``"gain"`` and
        ``"big_gain"`` defining the boundary points.
    """

    if r <= thresholds["big_loss"]:
        return 0  # Big loss
    elif r <= thresholds["loss"]:
        return 1  # Loss
    elif r < thresholds["gain"]:
        return 2  # Neutral
    elif r < thresholds["big_gain"]:
        return 3  # Gain
    else:
        return 4  # Big gain


def prepare_training_data(
    symbol,
    coin_id,
    oversampler: Optional[str] = None,
    min_unique_samples: int = 3,
    augment_target: int = 50,
):
    """Prepare feature matrix and labels for a single symbol.

    Parameters
    ----------
    symbol, coin_id: str
        Asset identifiers used for data fetching.
    oversampler: {"smote", "adasyn", "borderline"}, optional
        Technique for oversampling minority classes.  ``None`` disables
        oversampling.
    min_unique_samples: int, default ``3``
        Minimum number of unique rows required in a class before simple
        resampling. Classes with fewer unique samples are dropped to avoid

        training on duplicated data.
    augment_target: int, default ``50``
        Target size for minority classes during augmentation.

        Symbols with fewer than 60 historical rows after retrying all data
        sources are dropped before indicator computation.
    """
    logger.info("\n⏳ Preparing data for %s...", coin_id)
    effective_min_unique = min_unique_samples

    df = fetch_ohlcv_smart(symbol=symbol, coin_id=coin_id, days=730, limit=20000)

    if len(df) < 60:
        logger.warning(
            "⚠️ Only %d rows fetched for %s; attempting extended history",
            len(df),
            coin_id,
        )
        df_ext = fetch_ohlcv_smart(symbol=symbol, coin_id=coin_id, days=1460, limit=20000)
        if len(df_ext) > len(df):
            df = df_ext
        if len(df) < 60:
            original_sources = data_fetcher.DATA_SOURCES[:]
            for i in range(1, len(original_sources)):
                data_fetcher.DATA_SOURCES = (
                    original_sources[i:] + original_sources[:i]
                )
                df_retry = fetch_ohlcv_smart(
                    symbol=symbol, coin_id=coin_id, days=1460, limit=20000
                )
                if len(df_retry) > len(df):
                    df = df_retry
                if len(df) >= 60:
                    break
            data_fetcher.DATA_SOURCES = original_sources
        if len(df) < 60:
            logger.warning(
                "⚠️ %s has only %d rows after extended fetch; dropping symbol",
                coin_id,
                len(df),
            )
            return None, None

    if df.empty:
        logger.warning("⚠️ No data for %s", coin_id)
        return None, None

    if "Timestamp" in df.columns:
        span_days = (df["Timestamp"].max() - df["Timestamp"].min()).days
        logger.info(
            "📆 %s: fetched %d rows spanning ~%d days", coin_id, len(df), span_days
        )
    else:
        logger.info("📆 %s: fetched %d rows", coin_id, len(df))

    df = add_indicators(df)
    df = df.copy()
    if df.empty:
        logger.warning("⚠️ Failed to calculate indicators for %s", coin_id)
        return None, None

    df.loc[:, "Future_Close"] = df["Close"].shift(-3)  # 🔁 3-day ahead for short-term trading
    df.loc[:, "Return"] = (df["Future_Close"] - df["Close"]) / df["Close"]
    df = df[df["Return"].abs() > 0.005]
    df = df.dropna()

    thresholds = compute_return_thresholds(df["Return"])
    logger.info("📐 Thresholds for %s: %s", coin_id, thresholds)
    df["Target"] = df["Return"].apply(lambda r: return_bucket(r, thresholds))

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

    # Adjust oversampling target based on distribution
    augment_target = max(augment_target, int(class_counts.max() * 0.8))
    logger.info("🎯 Using augment_target=%d", augment_target)

    feature_cols = load_feature_list()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        optional_missing = [c for c in missing if c in OPTIONAL_FEATURES]
        missing_non_optional = [c for c in missing if c not in OPTIONAL_FEATURES]
        if missing_non_optional:
            msg = f"Missing required features in data: {missing_non_optional}"
            logger.error("❌ %s", msg)
            raise ValueError(msg)
        logger.warning("⚠️ Missing optional features in data: %s", optional_missing)
    X = df[[c for c in feature_cols if c in df.columns]]
    y = df["Target"]

    # Optional oversampling with SMOTE variants
    class_counts = y.value_counts()
    rare_classes = [cls for cls, cnt in class_counts.items() if cnt < augment_target]
    if oversampler in {"smote", "adasyn", "borderline"} and rare_classes:
        if SMOTE is None:
            logger.warning(
                "⚠️ imbalanced-learn is required for %s oversampling. Run 'pip install imbalanced-learn'",
                oversampler,
            )
        else:
            try:
                sampler_map = {
                    "smote": SMOTE,
                    "adasyn": ADASYN,
                    "borderline": BorderlineSMOTE,
                }
                sampler_cls = sampler_map[oversampler]
                strategy = {
                    cls: augment_target for cls in rare_classes if class_counts[cls] > 1
                }
                if strategy:
                    sampler = sampler_cls(
                        random_state=42, sampling_strategy=strategy
                    )
                    X, y = sampler.fit_resample(X, y)
                    class_counts = y.value_counts()
                    logger.info("📈 Applied %s oversampling", oversampler.upper())
            except Exception as e:
                logger.warning("⚠️ %s oversampling failed: %s", oversampler, e)

    # Fallback simple resampling for remaining minority classes
    class_counts = y.value_counts()
    for cls, cnt in class_counts.items():
        target = min(augment_target, 2 * cnt)
        if cnt < target:
            logger.info(
                "⚠️ Class %d has %d samples; augmenting to reach %d", cls, cnt, target
            )
            idx = y[y == cls].index
            if len(idx) == 0:
                logger.warning(
                    "⚠️ No data available to augment class %d for %s; skipping", cls, coin_id
                )
                return None, None
            subset = pd.concat([X.loc[idx], y.loc[idx]], axis=1)
            unique_rows = subset.drop_duplicates().shape[0]
            if unique_rows < effective_min_unique:
                logger.warning(
                    "⚠️ Class %d has only %d unique rows; dropping %s",
                    cls,
                    unique_rows,
                    coin_id,
                )
                return None, None
            augmented = resample(
                subset, replace=True, n_samples=target - cnt, random_state=42
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

def train_model(X, y, oversampler: Optional[str] = None, fast: bool = False):
    logger.info("\n🚀 Training multi-class classifier...")
    original_label_names = {
        0: "big_loss",
        1: "small_loss",
        2: "neutral",
        3: "small_gain",
        4: "big_gain",
    }

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_count = len(le.classes_)
    label_names = [original_label_names[cls] for cls in le.classes_]
    assert class_count == len(label_names), "Encoded class count mismatch"
    logger.info("✅ Model will use %d classes: %s", class_count, label_names)

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
        label_names = [original_label_names[cls] for cls in le.classes_]
        assert class_count == len(label_names), "Encoded class count mismatch"
        logger.info("✅ Using %d classes after drop: %s", class_count, label_names)
        class_dist = pd.Series(y_encoded).value_counts().sort_index()

    label_map = {i: original_label_names[cls] for i, cls in enumerate(le.classes_)}

    # === Time-based train/test split (chronological) ===
    df_xy = X.copy()
    df_xy["Target"] = y
    split_idx = int(len(df_xy) * 0.8)
    train_df, test_df = df_xy.iloc[:split_idx], df_xy.iloc[split_idx:]

    # === Training/validation split ===
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

        val_dist = pd.Series(y_val_raw).value_counts().sort_index()
        logger.info(
            "📊 Fold %d validation class distribution (unchanged): %s",
            fold,
            val_dist,
        )

        if oversampler in {"smote", "adasyn", "borderline"}:
            if SMOTE is None:
                logger.warning(
                    "⚠️ imbalanced-learn is required for %s oversampling. Run 'pip install imbalanced-learn'",
                    oversampler,
                )
            else:
                try:
                    sampler_map = {
                        "smote": SMOTE,
                        "adasyn": ADASYN,
                        "borderline": BorderlineSMOTE,
                    }
                    sampler_cls = sampler_map[oversampler]
                    sampler = sampler_cls(random_state=42)
                    X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)
                    logger.info(
                        "📈 Fold %d applied %s oversampling", fold, oversampler.upper()
                    )
                    logger.info(
                        "📊 Fold %d training distribution after resampling: %s",
                        fold,
                        pd.Series(y_tr).value_counts().sort_index(),
                    )
                except Exception as e:
                    logger.warning(
                        "⚠️ Fold %d %s oversampling failed: %s", fold, oversampler, e
                    )

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
                    zero_division=0,
                ),
            )
        except Exception as e:
            logger.warning("⚠️ CV fold %d failed: %s", fold, e)

    # === Final model training with class weights & hyperparameter search ===
    X_train_bal, y_train_bal = X_train, y_train
    if oversampler in {"smote", "adasyn", "borderline"}:
        if SMOTE is None:
            logger.warning(
                "⚠️ imbalanced-learn is required for %s oversampling. Run 'pip install imbalanced-learn'",
                oversampler,
            )
        else:
            try:
                sampler_map = {
                    "smote": SMOTE,
                    "adasyn": ADASYN,
                    "borderline": BorderlineSMOTE,
                }
                sampler_cls = sampler_map[oversampler]
                sampler = sampler_cls(random_state=42)
                X_train_bal, y_train_bal = sampler.fit_resample(X_train_bal, y_train_bal)
                logger.info(
                    "📈 Applied %s oversampling to full training set",
                    oversampler.upper(),
                )
                logger.info(
                    "📊 Post-oversampling class distribution: %s",
                    pd.Series(y_train_bal).value_counts().sort_index(),
                )
            except Exception as e:
                logger.warning("⚠️ %s oversampling failed: %s", oversampler, e)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_bal)
    class_weights = (
        pd.Series(sample_weights, index=y_train_bal)
        .groupby(level=0)
        .mean()
        .to_dict()
    )
    logger.info("⚖️ Class weights: %s", class_weights)

    fast = fast or os.environ.get("FAST")
    if fast:
        param_grid = {
            "n_estimators": [100],
            "max_depth": [3],
            "learning_rate": [0.1],
            "subsample": [1.0],
            "colsample_bytree": [1.0],
        }
    else:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }

    # Use a single worker process and single-threaded fits for Windows stability

    grid = GridSearchCV(
        XGBClassifier(
            objective="multi:softprob",
            num_class=len(le.classes_),
            random_state=42,
            eval_metric="mlogloss",

            n_jobs=1,  # use a single thread per fit for Windows stability (nthread for older versions)

        ),
        param_grid,
        scoring="f1_macro",
        cv=TimeSeriesSplit(n_splits=n_splits),
        n_jobs=1,  # single process to avoid Windows multiprocessing issues
        refit=True,
        verbose=1,
    )
    grid.fit(X_train_bal, y_train_bal, sample_weight=sample_weights)
    model = grid.best_estimator_
    logger.info(
        "🔍 Best params: %s (macro-F1=%.3f)", grid.best_params_, grid.best_score_
    )

    preds = model.predict(X_test)

    labels_sorted = sorted(np.unique(np.concatenate([y_test, preds])))
    target_names = [label_map[int(lbl)] for lbl in labels_sorted]

    report_dict = classification_report(
        y_test,
        preds,
        labels=labels_sorted,
        target_names=target_names,
        digits=3,
        zero_division=0,
        output_dict=True,
    )
    report = classification_report(
        y_test,
        preds,
        labels=labels_sorted,
        target_names=target_names,
        digits=3,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, preds, labels=labels_sorted)

    logger.info("\n📊 Classification Report:\n%s", report)
    logger.info("🔁 Confusion Matrix:\n%s", cm)
    logger.info(
        "📊 Prediction distribution: %s", pd.Series(preds).value_counts().sort_index()
    )

    metrics_summary = {
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "per_class_recall": {
            name: report_dict.get(name, {}).get("recall", 0.0)
            for name in target_names
        },
    }

    os.makedirs("analytics", exist_ok=True)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
        os.path.join("analytics", "confusion_matrix.csv"), index_label="actual"
    )
    with open(os.path.join("analytics", "classification_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join("analytics", "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info("📁 Saved diagnostics to analytics/")

    try:
        _, thresholds = calibrate_and_analyze(model, X_test, y_test, target_names)
        logger.info("📈 Recommended thresholds: %s", thresholds)
    except Exception as e:
        logger.warning("⚠️ Calibration/threshold analysis failed: %s", e)

    return model, le.classes_

def main():
    parser = argparse.ArgumentParser(description="Train crypto classifier")
    parser.add_argument(
        "--oversampler",
        choices=["smote", "adasyn", "borderline"],
        default="smote",
        help="Apply oversampling technique to minority classes",
    )
    parser.add_argument(
        "--min-unique-samples",
        type=int,
        default=3,
        help="Minimum unique rows required per class before resampling",
    )
    parser.add_argument(
        "--augment-target",
        type=int,
        default=50,
        help="Target sample size for minority classes during augmentation",
    )
    parser.add_argument(
        "--max-assets",
        type=int,
        default=10,
        help="Target number of symbols to include in training",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a smaller hyperparameter grid for quicker runs",
    )
    args = parser.parse_args()

    if args.oversampler in {"smote", "adasyn"} and SMOTE is None:
        logger.error(
            "imbalanced-learn is required for %s oversampling. Run 'pip install -r requirements.txt' to install it.",
            args.oversampler,
        )
        sys.exit(1)

    candidates = get_volume_ranked_symbols()
    X_list: list[pd.DataFrame] = []
    y_list: list[pd.Series] = []

    for symbol, volume in candidates:
        if volume < MIN_24H_VOLUME:
            logger.info(
                "⏭️ Skipping %s: volume %.0f below %s", symbol.upper(), volume, MIN_24H_VOLUME
            )
            continue

        coin_id = data_fetcher.resolve_coin_id(symbol, symbol)
        if not coin_id:
            logger.info("⏭️ Skipping %s: unable to resolve coin id", symbol.upper())
            continue

        df = fetch_ohlcv_smart(symbol, coin_id=coin_id, days=730)
        if len(df) < 60:
            logger.info(
                "⏭️ Skipping %s: only %d rows of data", symbol.upper(), len(df)
            )
            continue

        X, y = prepare_training_data(
            symbol,
            coin_id,
            oversampler=None,
            min_unique_samples=args.min_unique_samples,
            augment_target=args.augment_target,
        )
        if X is not None and y is not None:
            logger.info("✅ Selected %s for training", symbol.upper())
            X_list.append(X)
            y_list.append(y)
        if len(X_list) >= args.max_assets:
            break

    X_list = [x for x in X_list if x is not None and not x.empty]
    y_list = [y for y in y_list if y is not None and not y.empty]

    if not X_list:
        logger.warning("⚠️ No usable data, aborting.")
        return

    X_all = pd.concat(X_list)
    y_all = pd.concat(y_list)

    model, labels = train_model(
        X_all,
        y_all,
        oversampler=args.oversampler,
        fast=args.fast,
    )
    model.save_model("ml_model.json")
    with open("features.json", "w") as f:
        json.dump(X_all.columns.tolist(), f)
    with open("labels.json", "w") as f:
        json.dump([int(lbl) for lbl in labels], f)
    logger.info("💾 Saved multi-class model, feature list, and labels")

if __name__ == "__main__":
    main()
