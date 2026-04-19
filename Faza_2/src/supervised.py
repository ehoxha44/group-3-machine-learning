"""
supervised.py – All supervised learning models (classification + regression).

Classification algorithms
-------------------------
1. Random Forest Classifier
   • Ensemble of 300 decision trees, each trained on a bootstrap sample.
   • Robust to outliers in financial data; no linearity assumption.
   • class_weight="balanced" compensates for GROWING/DECLINING/STABLE imbalance.
   • Exposes feature_importances_ showing which features drive growth class.

2. XGBoost Classifier
   • Sequential gradient boosting: each tree corrects errors of the previous.
   • Built-in L1+L2 regularisation prevents overfitting on 3 956-row dataset.
   • Consistently outperforms single-tree ensembles on structured tabular data.

Regression algorithms
---------------------
3. Linear Regression
   • Baseline model: fits a hyperplane to predict continuous growth_rate.
   • Assumes a linear relationship between features and growth rate.
   • Coefficients are directly interpretable (e.g., each extra log-EUR of
     previous turnover adds X% to next-year growth rate).
   • Serves as the lower-bound benchmark — if tree models can't beat it,
     the non-linear signal is weak.

4. Random Forest Regressor
   • Same ensemble logic as the classifier, but predicts a continuous value
     (growth_rate) instead of a class label.
   • Naturally handles non-linear interactions between municipality, sector,
     and historical turnover without manual feature engineering.

5. XGBoost Regressor
   • Gradient-boosted trees for continuous output (objective=reg:squarederror).
   • Typically the best performer on structured economic tabular data.
   • Used as the primary model for the 2026 market predictions.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.config import (
    MODELS_DIR, RANDOM_STATE, RF_PARAMS, RF_REG_PARAMS,
    TEST_SIZE, XGB_PARAMS, XGB_REG_PARAMS,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE: bool = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor  # type: ignore
    XGB_AVAILABLE = False
    print("WARNING: xgboost not installed — using sklearn GradientBoosting fallback.")


# ── Data splitting ────────────────────────────────────────────────────────────


def split_data(X, y, stratify=True):
    """Stratified (classification) or plain (regression) train/test split."""
    if stratify:
        return train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


# ── Classification ────────────────────────────────────────────────────────────


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    print("Training Random Forest Classifier …")
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_train, y_train)
    print("  → Done")
    return clf


def train_xgboost(X_train, y_train):
    n_classes = int(np.unique(y_train).size)
    if XGB_AVAILABLE:
        print("Training XGBoost Classifier …")
        params = dict(XGB_PARAMS)
        if n_classes > 2:
            params["objective"] = "multi:softprob"
            params["num_class"] = n_classes
        clf = XGBClassifier(**params)
    else:
        print("Training GradientBoostingClassifier (XGBoost fallback) …")
        clf = GradientBoostingClassifier(  # type: ignore[name-defined]
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=RANDOM_STATE,
        )
    clf.fit(X_train, y_train)
    print(f"  → {'XGBoost' if XGB_AVAILABLE else 'GradientBoosting'} done")
    return clf


# ── Regression ────────────────────────────────────────────────────────────────


def train_linear_regression(X_train, y_train) -> LinearRegression:
    """Fit an OLS Linear Regression as baseline. Predicts growth_rate directly."""
    print("Training Linear Regression …")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("  → Done")
    return lr


def train_rf_regressor(X_train, y_train) -> RandomForestRegressor:
    """Fit a Random Forest Regressor to predict continuous growth_rate."""
    print("Training Random Forest Regressor …")
    rfr = RandomForestRegressor(**RF_REG_PARAMS)
    rfr.fit(X_train, y_train)
    print("  → Done")
    return rfr


def train_xgb_regressor(X_train, y_train):
    """Fit an XGBoost Regressor (objective=reg:squarederror) for growth_rate."""
    if XGB_AVAILABLE:
        print("Training XGBoost Regressor …")
        xgbr = XGBRegressor(**XGB_REG_PARAMS)
    else:
        print("Training GradientBoostingRegressor (XGBoost fallback) …")
        xgbr = GradientBoostingRegressor(  # type: ignore[name-defined]
            n_estimators=100, max_depth=5, learning_rate=0.05,
            random_state=RANDOM_STATE,
        )
    xgbr.fit(X_train, y_train)
    print(f"  → {'XGBoost' if XGB_AVAILABLE else 'GradientBoosting'} Regressor done")
    return xgbr


# ── Persistence ───────────────────────────────────────────────────────────────


def save_model(model, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"  → Saved: outputs/models/{name}.pkl")
    return path


def load_model(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model: {path}")
    return joblib.load(path)
