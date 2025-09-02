# train_real_model.py

import os
import json
import argparse
import sys
import io
import contextlib
from typing import Optional, Iterable


import numpy as np
import pandas as pd
import requests
import data_fetcher
from data_fetcher import fetch_ohlcv_smart
import symbol_resolver
from config import MIN_24H_VOLUME
from feature_engineer import add_indicators
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import resample
from xgboost import XGBClassifier
from analytics.calibration_utils import calibrate_and_analyze

import joblib
from joblib.externals.loky.process_executor import TerminatedWorkerError
BackendError = getattr(joblib.parallel, "BackendError", Exception)

from scipy.stats import ttest_rel

from utils.logging import get_logger
from threshold_utils import compute_return_thresholds

logger = get_logger(__name__)

_SHORT_HISTORY_LOGGED: set[str] = set()

try:
    from imblearn.over_sampling import (
        SMOTE,
        ADASYN,
        BorderlineSMOTE,
        RandomOverSampler,
    )
except ImportError:  # pragma: no cover - handled gracefully at runtime
    SMOTE = ADASYN = BorderlineSMOTE = RandomOverSampler = None
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
    augment_ratio: float = 0.8,
    quantiles: Iterable[float] = (0.2, 0.4, 0.6, 0.8),
    min_threshold_gap: float | None = None,
    min_return: float = 0.005,

    horizon: int = 3,

    min_rows: int = 60,
    min_rows_ratio: float = 0.6,
):
    """Prepare feature matrix and labels for a single symbol.

    Parameters
    ----------
    symbol, coin_id: str
        Asset identifiers used for data fetching.
    oversampler: {"random", "smote", "adasyn", "borderline"}, optional
        Technique for oversampling minority classes.  ``None`` disables
        oversampling.
    min_unique_samples: int, default ``3``
        Minimum number of unique rows required in a class before simple
        resampling. Classes with fewer unique samples are dropped to avoid

        training on duplicated data.
    augment_target: int, default ``50``
        Target size for minority classes during augmentation.
    augment_ratio: float, default ``0.8``
        Maximum fraction of the majority class size used when deriving
        ``augment_target``. Lower values reduce oversampling.
    quantiles: iterable of float, optional
        Percentiles for :func:`compute_return_thresholds`. Adjust to
        influence how return buckets are defined.
    min_threshold_gap: float, optional
        Minimum separation between adjacent return thresholds. If provided,
        :func:`compute_return_thresholds` will widen the outer quantiles when
        thresholds are closer than this value.

    min_return : float, default ``0.005``
        Minimum absolute future return required for a row to be kept.
        Set to ``0`` to retain all rows.

    horizon : int, default ``3``
        Number of future bars (15m each) to look ahead when computing the
        target. ``3`` corresponds to 45 minutes, while ``288`` covers roughly
        3 days.

    min_rows : int, default ``60``
        Baseline minimum number of rows desired before indicator computation.
    min_rows_ratio : float, default ``0.6``
        Scale factor applied to the fetched history length to derive an
        adaptive minimum row requirement. The effective threshold becomes
        ``min(min_rows, len(df) * min_rows_ratio)``. Symbols with fewer rows
        than this value are dropped before indicator computation.
    """
    logger.info("\n⏳ Preparing data for %s...", coin_id)
    effective_min_unique = min_unique_samples

    ok, count = data_fetcher.has_min_history(
        symbol, min_bars=data_fetcher.MIN_HISTORY_BARS, interval="15m"
    )
    if not ok:
        if coin_id not in _SHORT_HISTORY_LOGGED:
            logger.info(
                "⏭️ Skipping %s (%s 15m candles)",
                coin_id,
                count if count is not None else "unknown",
            )
            _SHORT_HISTORY_LOGGED.add(coin_id)
        return None, None

    def required_rows(n):
        return int(min(min_rows, n * min_rows_ratio))

    df = fetch_ohlcv_smart(symbol=symbol, coin_id=coin_id, days=730, limit=20000)

    if len(df) < data_fetcher.MIN_HISTORY_BARS:
        if coin_id not in _SHORT_HISTORY_LOGGED:
            logger.info(
                "⏭️ Skipping %s (%d fetched rows)", coin_id, len(df)
            )
            _SHORT_HISTORY_LOGGED.add(coin_id)
        return None, None

    if len(df) < required_rows(len(df)):
        logger.warning(
            "⚠️ Only %d rows fetched for %s; attempting extended history",
            len(df),
            coin_id,
        )
        df_ext = fetch_ohlcv_smart(
            symbol=symbol, coin_id=coin_id, days=1460, limit=20000
        )
        if len(df_ext) > len(df):
            df = df_ext
        if len(df) < required_rows(len(df)):
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
                if len(df) >= required_rows(len(df)):
                    break
            data_fetcher.DATA_SOURCES = original_sources
        if len(df) < required_rows(len(df)):
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
        try:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
            interval = (
                df.sort_values("Timestamp")["Timestamp"].diff().median()
            )
            if pd.isna(interval) or interval <= pd.Timedelta(0):
                raise ValueError
            ratio = max(pd.Timedelta("4h") / interval, 1)
        except Exception:
            ratio = 16
        required_4h = int(26 * ratio)
        if len(df) < required_4h:
            logger.warning(
                "⚠️ %s: %d rows (<%d needed for 4h aggregates)",
                coin_id,
                len(df),
                required_4h,
            )
            return None, None
        span_days = (df["Timestamp"].max() - df["Timestamp"].min()).days
        logger.info(
            "📆 %s: fetched %d rows spanning ~%d days", coin_id, len(df), span_days
        )
    else:
        logger.info("📆 %s: fetched %d rows", coin_id, len(df))

    min_rows_effective = required_rows(len(df))
    df = add_indicators(df, min_rows=min_rows_effective, min_rows_ratio=min_rows_ratio)
    if df.empty:
        logger.warning(
            "⚠️ %s has insufficient data after indicator prerequisites; dropping symbol",
            coin_id,
        )
        return None, None
    df = df.copy()
    if len(df) < min_rows_effective:
        logger.warning(
            "⚠️ %s has only %d rows after indicators; attempting alternate sources",
            coin_id,
            len(df),
        )
        original_sources = data_fetcher.DATA_SOURCES[:]
        for i in range(1, len(original_sources)):
            data_fetcher.DATA_SOURCES = original_sources[i:] + original_sources[:i]
            df_retry = fetch_ohlcv_smart(
                symbol=symbol, coin_id=coin_id, days=1460, limit=20000
            )
            retry_min_rows = required_rows(len(df_retry))
            df_retry = add_indicators(
                df_retry,
                min_rows=retry_min_rows,
                min_rows_ratio=min_rows_ratio,
            )
            if len(df_retry) >= retry_min_rows:
                df = df_retry
                min_rows_effective = retry_min_rows
                break
        data_fetcher.DATA_SOURCES = original_sources
        if len(df) < min_rows_effective:
            logger.warning(
                "⚠️ %s remains below %d rows after indicator retry; dropping symbol",
                coin_id,
                min_rows_effective,
            )
            return None, None


    df.loc[:, "Future_Close"] = df["Close"].shift(-horizon)
    df.loc[:, "Return"] = (df["Future_Close"] - df["Close"]) / df["Close"]
    if len(df) > horizon:
        aligned = np.allclose(
            df["Future_Close"].values[:-horizon],
            df["Close"].values[horizon:],
        )
        if not aligned:
            logger.error(
                "❌ Future_Close misalignment for %s (horizon=%d)", coin_id, horizon
            )
            raise AssertionError("Future_Close shift misalignment")
        logger.debug(
            "✅ Verified Future_Close alignment for %s (horizon=%d)", coin_id, horizon
        )
    if min_return > 0:
        df = df[df["Return"].abs() > min_return]
    df = df.dropna()


    thresholds = compute_return_thresholds(
        df["Return"], quantiles=quantiles, min_gap=min_threshold_gap
    )
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
    augment_target = min(augment_target, int(class_counts.max() * augment_ratio))
    logger.info(
        "🎯 Using augment_target=%d (ratio=%.2f)", augment_target, augment_ratio
    )

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

    # Log original distribution before any resampling
    class_counts = y.value_counts().sort_index()
    logger.info("📊 Initial class distribution: %s", class_counts)

    # Optional oversampling with imbalanced-learn methods
    rare_classes = [cls for cls, cnt in class_counts.items() if cnt < augment_target]
    if oversampler in {"smote", "adasyn", "borderline", "random"} and rare_classes:
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
                    "random": RandomOverSampler,
                }
                sampler_cls = sampler_map[oversampler]
                strategy = {
                    cls: augment_target for cls in rare_classes if class_counts[cls] > 1
                }
                if strategy:
                    logger.info(
                        "📊 Class distribution before %s oversampling: %s",
                        oversampler,
                        class_counts,
                    )
                    sampler = sampler_cls(
                        random_state=42, sampling_strategy=strategy
                    )
                    X, y = sampler.fit_resample(X, y)
                    class_counts = y.value_counts().sort_index()
                    logger.info("📈 Applied %s oversampling", oversampler.upper())
                    logger.info(
                        "📊 Class distribution after %s oversampling: %s",
                        oversampler,
                        class_counts,
                    )
            except Exception as e:
                logger.warning("⚠️ %s oversampling failed: %s", oversampler, e)

    # Fallback simple resampling for remaining minority classes
    class_counts = y.value_counts().sort_index()
    logger.info("📊 Class distribution before simple resampling: %s", class_counts)
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
            class_counts = y.value_counts().sort_index()
            logger.info(
                "📊 Class distribution after augmenting class %d: %s", cls, class_counts
            )

    df_aug = pd.concat([X, y], axis=1).sample(frac=1, random_state=42).reset_index(drop=True)
    logger.info(
        "📊 Class distribution after augmentation: %s", df_aug["Target"].value_counts().sort_index()
    )

    X = df_aug.drop(columns=["Target"])
    y = df_aug["Target"]
    return X, y

def train_model(
    X,
    y,
    oversampler: Optional[str] = "random",
    search_scale: str = "medium",
    cv_splits: int = 3,
    verbose: int = 1,
    n_jobs: int = 1,
    class_weight: str = "balanced",
    random_search: bool = False,
    random_iter: int = 10,
):
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
    X_train_raw = train_df.drop(columns=["Target"])
    y_train_raw = train_df["Target"]

    y_train = le.transform(y_train_raw)

    logger.info("📊 Training class distribution:")
    logger.info("%s", pd.Series(y_train).value_counts().sort_index())

    X_test_raw = test_df.drop(columns=["Target"])
    y_test_raw = test_df["Target"]
    y_test = le.transform(y_test_raw)

    # === Walk-forward cross-validation ===
    cv_splits = max(2, cv_splits)
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    model_f1s: list[float] = []
    baseline_f1s: list[float] = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train_raw), start=1):
        X_tr_raw, X_val_raw = X_train_raw.iloc[tr_idx], X_train_raw.iloc[val_idx]
        y_tr_raw, y_val_raw = y_train_raw.iloc[tr_idx], y_train_raw.iloc[val_idx]

        fold_le = LabelEncoder()
        y_tr = fold_le.fit_transform(y_tr_raw)
        y_val = fold_le.transform(y_val_raw)
        fold_label_map = {cls: original_label_names[cls] for cls in fold_le.classes_}

        val_dist = pd.Series(y_val_raw).value_counts().sort_index()
        logger.info(
            "📊 Fold %d validation class distribution (unchanged): %s",
            fold,
            val_dist,
        )

        fold_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="multi:softprob",
                        num_class=len(fold_le.classes_),
                        random_state=42,
                        eval_metric="mlogloss",
                    ),
                ),
            ]
        )

        X_tr = pd.DataFrame(
            fold_pipeline.named_steps["scaler"].fit_transform(X_tr_raw),
            columns=X_tr_raw.columns,
            index=X_tr_raw.index,
        )
        X_val = pd.DataFrame(
            fold_pipeline.named_steps["scaler"].transform(X_val_raw),
            columns=X_val_raw.columns,
            index=X_val_raw.index,
        )

        if oversampler in {"smote", "adasyn", "borderline", "random"}:
            if SMOTE is None:
                logger.warning(
                    "⚠️ imbalanced-learn is required for %s oversampling. Run 'pip install imbalanced-learn'",
                    oversampler,
                )
            else:
                try:
                    pre_dist = pd.Series(y_tr).value_counts().sort_index()
                    logger.info(
                        "📊 Fold %d training distribution before resampling: %s",
                        fold,
                        pre_dist,
                    )
                    sampler_map = {
                        "smote": SMOTE,
                        "adasyn": ADASYN,
                        "borderline": BorderlineSMOTE,
                        "random": RandomOverSampler,
                    }
                    sampler_cls = sampler_map[oversampler]
                    sampler = sampler_cls(random_state=42)
                    X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)
                    logger.info(
                        "📈 Fold %d applied %s oversampling", fold, oversampler.upper()
                    )
                    post_dist = pd.Series(y_tr).value_counts().sort_index()
                    logger.info(
                        "📊 Fold %d training distribution after resampling: %s",
                        fold,
                        post_dist,
                    )
                    if len(post_dist) == 1:
                        logger.warning(
                            "⚠️ Fold %d only one class present after resampling", fold
                        )
                    elif post_dist.nunique() == 1:
                        logger.info("✅ Fold %d classes balanced", fold)
                    else:
                        logger.warning(
                            "⚠️ Fold %d imbalance remains after resampling: %s",
                            fold,
                            post_dist,
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

        fold_weights = (
            compute_sample_weight(class_weight=class_weight, y=y_tr)
            if class_weight != "none"
            else None
        )
        try:
            if fold_weights is not None:
                fold_pipeline.named_steps["model"].fit(
                    X_tr, y_tr, sample_weight=fold_weights
                )
            else:
                fold_pipeline.named_steps["model"].fit(X_tr, y_tr)
            cv_preds_enc = fold_pipeline.named_steps["model"].predict(X_val)
            cv_preds = fold_le.inverse_transform(cv_preds_enc)
            if len(np.unique(cv_preds)) < 2:
                logger.warning(
                    "⚠️ Fold %d predictions contain only one class: %s",
                    fold,
                    fold_label_map.get(cv_preds[0], cv_preds[0]),
                )
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
            model_f1 = f1_score(
                y_val_raw,
                cv_preds,
                labels=sorted(fold_le.classes_),
                average="macro",
                zero_division=0,
            )
            rng = np.random.default_rng(42 + fold)
            baseline_preds_enc = rng.integers(
                0, len(fold_le.classes_), size=len(y_val_raw)
            )
            baseline_preds = fold_le.inverse_transform(baseline_preds_enc)
            baseline_f1 = f1_score(
                y_val_raw,
                baseline_preds,
                labels=sorted(fold_le.classes_),
                average="macro",
                zero_division=0,
            )
            logger.info(
                "🎯 Fold %d macro F1: model=%.3f, random=%.3f",
                fold,
                model_f1,
                baseline_f1,
            )
            model_f1s.append(model_f1)
            baseline_f1s.append(baseline_f1)
            cm = confusion_matrix(
                y_val_raw,
                cv_preds,
                labels=sorted(fold_le.classes_),
            )
            cm_df = pd.DataFrame(
                cm,
                index=[fold_label_map[cls] for cls in sorted(fold_le.classes_)],
                columns=[fold_label_map[cls] for cls in sorted(fold_le.classes_)],
            )
            logger.info("🧮 CV Fold %d confusion matrix:\n%s", fold, cm_df)
        except Exception as e:
            logger.warning("⚠️ CV fold %d failed: %s", fold, e)

    if model_f1s:
        mean_model = float(np.mean(model_f1s))
        mean_baseline = float(np.mean(baseline_f1s))
        logger.info(
            "📈 Mean CV macro F1: model=%.3f, random=%.3f",
            mean_model,
            mean_baseline,
        )
        try:
            t_stat, p_val = ttest_rel(model_f1s, baseline_f1s)
            logger.info(
                "🔬 Paired t-test vs random baseline: t=%.3f, p=%.3f",
                float(t_stat),
                float(p_val),
            )
        except Exception as e:
            logger.warning("⚠️ Failed to compute t-test: %s", e)

    # === Final model training with class weights & hyperparameter search ===
    X_train_bal, y_train_bal = X_train_raw.copy(), y_train
    if oversampler in {"smote", "adasyn", "borderline", "random"}:
        if SMOTE is None:
            logger.warning(
                "⚠️ imbalanced-learn is required for %s oversampling. Run 'pip install imbalanced-learn'",
                oversampler,
            )
        else:
            try:
                pre_dist = pd.Series(y_train_bal).value_counts().sort_index()
                logger.info(
                    "📊 Class distribution before %s oversampling: %s",
                    oversampler,
                    pre_dist,
                )
                sampler_map = {
                    "smote": SMOTE,
                    "adasyn": ADASYN,
                    "borderline": BorderlineSMOTE,
                    "random": RandomOverSampler,
                }
                sampler_cls = sampler_map[oversampler]
                sampler = sampler_cls(random_state=42)
                X_train_bal, y_train_bal = sampler.fit_resample(X_train_bal, y_train_bal)
                logger.info(
                    "📈 Applied %s oversampling to full training set",
                    oversampler.upper(),
                )
                post_dist = pd.Series(y_train_bal).value_counts().sort_index()
                logger.info(
                    "📊 Class distribution after %s oversampling: %s",
                    oversampler,
                    post_dist,
                )
                if len(post_dist) == 1:
                    logger.warning("⚠️ Only one class present after resampling")
                elif post_dist.nunique() == 1:
                    logger.info("✅ Training set balanced after resampling")
                else:
                    logger.warning(
                        "⚠️ Imbalance remains after resampling: %s", post_dist
                    )
            except Exception as e:
                logger.warning("⚠️ %s oversampling failed: %s", oversampler, e)

    final_dist = pd.Series(y_train_bal).value_counts().sort_index()
    logger.info("📊 Final training class distribution: %s", final_dist)
    if len(final_dist) < 2:
        logger.warning(
            "⚠️ Training data contains only one class: %s", final_dist.index.tolist()
        )
    else:
        logger.info("✅ Training data contains %d classes", len(final_dist))

    sample_weights = (
        compute_sample_weight(class_weight=class_weight, y=y_train_bal)
        if class_weight != "none"
        else None
    )
    if sample_weights is not None:
        class_weights = (
            pd.Series(sample_weights, index=y_train_bal)
            .groupby(level=0)
            .mean()
            .to_dict()
        )
    else:
        class_weights = {cls: 1.0 for cls in np.unique(y_train_bal)}
    logger.info("⚖️ Class weights: %s", class_weights)
    weight_name_map = {label_map.get(int(k), k): v for k, v in class_weights.items()}
    logger.info("⚖️ Class weights by label: %s", weight_name_map)
    if set(class_weights.keys()) == set(label_map.keys()):
        logger.info("✅ Label encoding matches class weight keys")
    else:
        logger.warning(
            "⚠️ Label encoding and class weight keys mismatch: %s vs %s",
            list(class_weights.keys()),
            list(label_map.keys()),
        )

    final_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=len(le.classes_),
                    random_state=42,
                    eval_metric="mlogloss",
                    n_jobs=1,
                ),
            ),
        ]
    )

    scale = (search_scale or "medium").lower()
    if scale == "small":
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3],
            "model__learning_rate": [0.1],
            "model__subsample": [1.0],
            "model__colsample_bytree": [1.0],
            "model__min_child_weight": [1],
        }
    elif scale == "medium":
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 4],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__min_child_weight": [1, 3],
        }
    else:
        param_grid = {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0],
            "model__min_child_weight": [1, 3, 5],
        }

    # Use user-specified parallelism for cross-validation. The estimator itself
    # remains single-threaded to avoid nested parallelism.

    if random_search:
        search = RandomizedSearchCV(
            final_pipeline,
            param_grid,
            n_iter=random_iter,
            scoring="f1_macro",
            cv=TimeSeriesSplit(n_splits=cv_splits),
            n_jobs=n_jobs,
            refit=True,
            verbose=verbose,
            random_state=42,
        )
    else:
        search = GridSearchCV(
            final_pipeline,
            param_grid,
            scoring="f1_macro",
            cv=TimeSeriesSplit(n_splits=cv_splits),
            n_jobs=n_jobs,
            refit=True,
            verbose=verbose,
        )

    class _LogStream(io.TextIOBase):
        def write(self, buf):
            for line in buf.rstrip().splitlines():
                logger.info(line)
        def flush(self):
            pass

    with contextlib.redirect_stdout(_LogStream()):
        if sample_weights is not None:
            search.fit(
                X_train_bal,
                y_train_bal,
                model__sample_weight=sample_weights,
            )
        else:
            search.fit(X_train_bal, y_train_bal)
    model = search.best_estimator_
    logger.info(
        "🔍 Best params: %s (macro-F1=%.3f)", search.best_params_, search.best_score_
    )

    preds = model.predict(X_test_raw)
    if len(np.unique(preds)) < 2:
        logger.warning(
            "⚠️ Test predictions contain only one class: %s",
            label_map.get(int(preds[0]), int(preds[0])) if len(preds) else "n/a",
        )
    proba = model.predict_proba(X_test_raw)

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

    mean_proba = dict(zip([label_map[i] for i in range(proba.shape[1])], proba.mean(axis=0)))
    logger.info("🔍 Mean per-class predicted probabilities: %s", mean_proba)

    logger.info("\n📊 Classification Report:\n%s", report)
    logger.info("🔁 Confusion Matrix:\n%s", cm)
    logger.info(
        "📊 Prediction distribution: %s", pd.Series(preds).value_counts().sort_index()
    )
    if len(np.unique(preds)) == 1:
        logger.warning("⚠️ All predictions collapsed into a single class")

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
    pd.DataFrame(proba, columns=[label_map[i] for i in range(proba.shape[1])]).to_csv(
        os.path.join("analytics", "predicted_probabilities.csv"), index=False
    )
    with open(os.path.join("analytics", "classification_report.txt"), "w") as f:
        f.write(report)
    with open(os.path.join("analytics", "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    joblib.dump(model.named_steps["scaler"], os.path.join("analytics", "scaler.pkl"))
    logger.info("📁 Saved diagnostics to analytics/")

    try:
        _, thresholds = calibrate_and_analyze(model, X_test_raw, y_test, target_names)
        logger.info("📈 Recommended thresholds: %s", thresholds)
    except Exception as e:
        logger.warning("⚠️ Calibration/threshold analysis failed: %s", e)

    return model, le.classes_

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train crypto classifier and overwrite features.json with the "
            "feature names used during training"
        )
    )
    parser.add_argument(
        "--oversampler",
        choices=["random", "smote", "adasyn", "borderline", "none"],
        default="random",
        help=(
            "Resampling strategy: 'random' duplicates minority samples, 'none' relies on class weights"
        ),
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Class weight strategy during training",
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
        "--augment-ratio",
        type=float,
        default=0.8,
        help="Max fraction of majority class size used for oversampling",
    )

    parser.add_argument(
        "--quantiles",
        type=float,
        nargs=4,
        default=(0.2, 0.4, 0.6, 0.8),
        metavar=("Q1", "Q2", "Q3", "Q4"),
        help="Quantiles for return thresholds (four floats between 0 and 1)",
    )
    parser.add_argument(
        "--min-threshold-gap",
        type=float,
        default=None,
        help="Minimum separation enforced between return thresholds",
    )
    parser.add_argument(
        "--min-return",
        type=float,
        default=0.005,
        help="Minimum absolute return required to keep a row; 0 disables filtering",

    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Number of future 15m bars to look ahead when computing the target",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a small hyperparameter grid for quicker runs",
    )

    parser.add_argument(
        "--max-assets",
        type=int,
        default=10,
        help="Target number of symbols to include in training",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=None,
        help=(
            "Maximum number of volume-ranked symbols to scan for training candidates"
        ),
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help=(
            "Comma-separated list of symbols to train on (e.g., BTC,ETH)"
        ),
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=MIN_24H_VOLUME / 10,
        help="Minimum 24h quote volume required for a symbol to be considered",
    )
    parser.add_argument(
        "--ignore-volume",
        action="store_true",
        help="Include symbols regardless of their 24h volume",
    )
    
    parser.add_argument(
        "--search-scale",
        "--param-scale",
        choices=["small", "medium", "full"],
        default="medium",
        dest="search_scale",
        help="Size of hyperparameter grid search",
    )
    parser.add_argument(
        "--random-search",
        action="store_true",
        help="Use RandomizedSearchCV instead of GridSearchCV",
    )
    parser.add_argument(
        "--random-iter",
        type=int,
        default=20,
        help="Number of parameter settings sampled in randomized search",
    )
    parser.add_argument(
        "--gs-verbose",
        "--verbose",
        dest="gs_verbose",
        type=int,
        default=1,
        help="Grid search verbosity level",

    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for grid search (use -1 for all cores; non-Windows only)",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=3,
        help="Number of cross-validation splits",
    )
    args = parser.parse_args()
    if args.candidate_limit is None:
        args.candidate_limit = args.max_assets * 3
    min_volume = 0 if args.ignore_volume else args.min_volume

    if sys.platform.startswith("win") and args.n_jobs != 1:
        logger.warning("n_jobs != 1 is not supported on Windows; using 1")
    n_jobs = 1 if sys.platform.startswith("win") else args.n_jobs

    if args.fast:
        args.search_scale = "small"


    if args.oversampler in {"smote", "adasyn", "borderline", "random"} and SMOTE is None:
        logger.error(
            "imbalanced-learn is required for %s oversampling. Run 'pip install -r requirements.txt' to install it.",
            args.oversampler,
        )
        sys.exit(1)
    if args.symbols:
        symbol_list = [s.strip().lower() for s in args.symbols.split(",") if s.strip()]
        args.candidate_limit = len(symbol_list)
        candidates = [(s, float("inf")) for s in symbol_list]
    else:
        candidates = get_volume_ranked_symbols(limit=args.candidate_limit)
    X_list: list[pd.DataFrame] = []
    y_list: list[pd.Series] = []

    attempts = 0
    for symbol, volume in candidates:
        attempts += 1
        if volume < min_volume:
            logger.info(
                "⏭️ Skipping %s: volume %.0f below %s", symbol.upper(), volume, min_volume
            )
            continue

        coin_id = data_fetcher.resolve_coin_id(symbol, symbol)
        if not coin_id:
            logger.info("⏭️ Skipping %s: unable to resolve coin id", symbol.upper())
            continue

        df = fetch_ohlcv_smart(symbol, coin_id=coin_id, days=730)
        if len(df) < 416:
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
            augment_ratio=args.augment_ratio,
            quantiles=args.quantiles,
            min_threshold_gap=args.min_threshold_gap,
            min_return=args.min_return,
            horizon=args.horizon,

        )
        if X is not None and y is not None:
            logger.info("✅ Selected %s for training", symbol.upper())
            X_list.append(X)
            y_list.append(y)
        if len(X_list) >= args.max_assets or attempts >= args.candidate_limit:
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
        oversampler=None if args.oversampler == "none" else args.oversampler,
        search_scale=args.search_scale,
        cv_splits=args.cv_splits,
        verbose=args.gs_verbose,
        n_jobs=n_jobs,
        class_weight=args.class_weight,
        random_search=args.random_search,
        random_iter=args.random_iter,
    )
    feature_list = X_all.columns.tolist()
    with open("features.json", "w") as f:
        json.dump(feature_list, f, indent=2)
    model.named_steps["model"].save_model("ml_model.json")
    with open("labels.json", "w") as f:
        json.dump([int(lbl) for lbl in labels], f)
    logger.info("💾 Saved multi-class model, feature list, and labels")

if __name__ == "__main__":
    main()
