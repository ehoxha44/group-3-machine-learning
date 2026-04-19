"""
feature_engineering.py – Encode and scale the growth dataset for modelling.

Input columns (from build_growth_dataset):
  Numeric   : year, prev_turnover_log1p, total_turnover_log1p,
              num_businesses_log1p, growth_rate
  Categorical: municipality, primary_sector
  Target     : growth_class  (GROWING / STABLE / DECLINING)

Note: growth_rate is included as a feature because the model may use it
to distinguish classes at the boundary (e.g., 4 % vs 6 % growth).
The actual labelling threshold (±5 %) is fixed, so including the raw rate
gives the model a continuous signal rather than forcing it to re-discover
the threshold from the encoded label alone.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN


def encode_and_scale(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], dict, StandardScaler]:
    """
    Encode categorical features, encode target, and scale all features.

    Returns
    -------
    X_scaled     : (n_samples, n_features) float array, zero-mean unit-var
    y            : (n_samples,) integer class labels
    feature_cols : list of column names matching X columns
    encoders     : dict {col_name: fitted LabelEncoder}
    scaler       : fitted StandardScaler
    """
    encoders: dict[str, LabelEncoder] = {}
    df_enc = df.copy()

    # Encode categoricals
    for col in CATEGORICAL_FEATURES:
        if col in df_enc.columns:
            le = LabelEncoder()
            df_enc[f"{col}_enc"] = le.fit_transform(df_enc[col].astype(str))
            encoders[col] = le

    # Encode target
    le_target = LabelEncoder()
    y: np.ndarray = le_target.fit_transform(df_enc[TARGET_COLUMN].astype(str))
    encoders[TARGET_COLUMN] = le_target

    # Feature columns: numeric + encoded categoricals
    feature_cols: list[str] = (
        [c for c in NUMERIC_FEATURES if c in df_enc.columns]
        + [f"{c}_enc" for c in CATEGORICAL_FEATURES if c in df_enc.columns]
    )

    X_raw: np.ndarray = df_enc[feature_cols].values.astype(float)

    scaler = StandardScaler()
    X_scaled: np.ndarray = scaler.fit_transform(X_raw)

    classes = list(le_target.classes_)
    print(
        f"Feature matrix: {X_scaled.shape}  |  "
        f"Target classes: {classes}"
    )
    return X_scaled, y, feature_cols, encoders, scaler


def encode_for_prediction(
    pred_df: pd.DataFrame,
    feature_cols: list[str],
    encoders: dict,
    scaler,
) -> np.ndarray:
    """
    Apply the *already-fitted* encoders and scaler to a new prediction DataFrame.

    This is used for the 2026 market forecast: we do NOT refit anything —
    we reuse the encoders from training so that integer codes are consistent.

    Unknown categories (municipalities/sectors not seen in training) are
    mapped to 0 silently, which is safe because such groups would carry
    very low confidence anyway.
    """
    df_enc = pred_df.copy()

    for col in CATEGORICAL_FEATURES:
        if col in df_enc.columns and col in encoders:
            le = encoders[col]
            known = set(le.classes_)
            df_enc[f"{col}_enc"] = df_enc[col].apply(
                lambda v: le.transform([str(v)])[0] if str(v) in known else 0
            )

    X_raw = df_enc[feature_cols].values.astype(float)
    return scaler.transform(X_raw)
