"""
feature_engineering.py - Leakage-safe encoding and scaling for Phase 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.config import (
    BASELINE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ENCODING_STRATEGY,
    NUMERIC_FEATURES,
    REGRESSION_TARGET,
    TARGET_COLUMN,
)

try:
    from category_encoders import TargetEncoder
    TARGET_ENCODER_AVAILABLE = True
except ImportError:
    TargetEncoder = None
    TARGET_ENCODER_AVAILABLE = False


@dataclass
class EncoderBundle:
    strategy: str
    encoder: object
    scaler: StandardScaler
    numeric_features: list[str]
    categorical_features: list[str]
    feature_names: list[str]


def _normalise_strategy(strategy: str) -> str:
    if strategy == "target" and not TARGET_ENCODER_AVAILABLE:
        print("TargetEncoder not available; falling back to onehot.")
        return "onehot"
    return strategy


def get_feature_lists(include_baseline_only: bool = False) -> tuple[list[str], list[str]]:
    numeric_features = BASELINE_NUMERIC_FEATURES if include_baseline_only else NUMERIC_FEATURES
    return numeric_features, CATEGORICAL_FEATURES


def fit_encoders(
    train_df: pd.DataFrame,
    strategy: str = ENCODING_STRATEGY,
    include_baseline_only: bool = False,
    numeric_features_override: list[str] | None = None,
) -> EncoderBundle:
    strategy = _normalise_strategy(strategy)
    numeric_features, categorical_features = get_feature_lists(include_baseline_only)
    if numeric_features_override is not None:
        numeric_features = numeric_features_override
    numeric_train = train_df[numeric_features].astype(float).copy()

    if strategy == "target":
        encoder = TargetEncoder(
            cols=categorical_features,
            smoothing=10.0,
            handle_unknown="value",
            handle_missing="value",
        )
        encoded_cat = encoder.fit_transform(
            train_df[categorical_features].astype(str),
            train_df[REGRESSION_TARGET].astype(float),
        )
        cat_names = [f"{col}_target" for col in categorical_features]
        encoded_cat.columns = cat_names
        encoded_values = encoded_cat.to_numpy(dtype=float)
    else:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
        encoded_values = encoder.fit_transform(train_df[categorical_features].astype(str))
        cat_names = list(encoder.get_feature_names_out(categorical_features))

    X_train_raw = np.hstack([numeric_train.to_numpy(dtype=float), encoded_values])
    scaler = StandardScaler()
    scaler.fit(X_train_raw)
    feature_names = numeric_features + cat_names
    return EncoderBundle(
        strategy=strategy,
        encoder=encoder,
        scaler=scaler,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        feature_names=feature_names,
    )


def transform_features(df: pd.DataFrame, bundle: EncoderBundle) -> np.ndarray:
    numeric_values = df[bundle.numeric_features].astype(float).to_numpy(dtype=float)
    cat_df = df[bundle.categorical_features].astype(str)

    if bundle.strategy == "target":
        encoded_df = bundle.encoder.transform(cat_df)
        encoded_values = encoded_df.to_numpy(dtype=float)
    else:
        encoded_values = bundle.encoder.transform(cat_df)

    X_raw = np.hstack([numeric_values, encoded_values])
    X_scaled = bundle.scaler.transform(X_raw)
    if np.isnan(X_scaled).any():
        raise ValueError("NaN detected in transformed feature matrix.")
    return X_scaled


def encode_target(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_df[TARGET_COLUMN].astype(str))
    y_val = label_encoder.transform(val_df[TARGET_COLUMN].astype(str))
    y_test = label_encoder.transform(test_df[TARGET_COLUMN].astype(str))
    return y_train, y_val, y_test, label_encoder


def encode_for_prediction(pred_df: pd.DataFrame, bundle: EncoderBundle) -> np.ndarray:
    return transform_features(pred_df, bundle)
