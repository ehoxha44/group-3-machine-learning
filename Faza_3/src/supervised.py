"""
supervised.py - Supervised models, tuning and persistence for Phase 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight

from src.config import (
    EARLY_STOPPING_ROUNDS,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    RF_PARAMS,
    RF_REG_PARAMS,
    TUNE_CV_FOLDS,
    TUNE_ITER,
    XGB_PARAMS,
    XGB_REG_PARAMS,
)
from src.time_split import time_series_cv

try:
    from xgboost import XGBClassifier, XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def build_rf_classifier(params: dict | None = None) -> RandomForestClassifier:
    return RandomForestClassifier(**(params or RF_PARAMS))


def build_rf_regressor(params: dict | None = None) -> RandomForestRegressor:
    return RandomForestRegressor(**(params or RF_REG_PARAMS))


def build_linear_regression() -> LinearRegression:
    return LinearRegression()


def build_xgb_classifier(params: dict | None = None, n_classes: int = 3):
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is required for XGBoost Phase 3 training.")
    config = dict(XGB_PARAMS)
    if params:
        config.update(params)
    if n_classes > 2:
        config["objective"] = "multi:softprob"
        config["num_class"] = n_classes
    else:
        config["objective"] = "binary:logistic"
    return XGBClassifier(**config)


def build_xgb_regressor(params: dict | None = None):
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is required for XGBoost Phase 3 training.")
    config = dict(XGB_REG_PARAMS)
    if params:
        config.update(params)
    config["objective"] = "reg:squarederror"
    return XGBRegressor(**config)


def tune_hyperparameters(
    estimator,
    param_distributions: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str = "classification",
    n_iter: int = TUNE_ITER,
    cv_folds: int = TUNE_CV_FOLDS,
    scoring: str | None = None,
    output_name: str | None = None,
) -> tuple[object, dict, pd.DataFrame]:
    if scoring is None:
        scoring = "f1_macro" if task == "classification" else "r2"

    cv = time_series_cv(n_splits=cv_folds)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    if output_name:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        cv_results.to_csv(METRICS_DIR / f"cv_tuning_{output_name}.csv", index=False)

    print(f"Best params for {output_name or estimator.__class__.__name__}: {search.best_params_}")
    return search.best_estimator_, search.best_params_, cv_results


def train_xgboost_with_early_stop(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
    task: str = "classification",
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
):
    """
    Fit XGBoost with early stopping on the validation slice.

    For classification we also pass balanced ``sample_weight`` so that the
    early-stopping criterion (mlogloss on val) does not silently undo the
    SMOTE rebalancing of the training set. Without sample weights the model
    converges back to the natural imbalanced distribution and the STABLE
    class collapses to ~0 recall (observed during Phase-3 audit).
    """
    if task == "classification":
        model = build_xgb_classifier(params=params, n_classes=int(np.unique(y_train).size))
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        val_sample_weight = compute_sample_weight(class_weight="balanced", y=y_val)
    else:
        model = build_xgb_regressor(params=params)
        sample_weight = None
        val_sample_weight = None

    model.set_params(early_stopping_rounds=early_stopping_rounds)
    fit_kwargs = {
        "eval_set": [(X_val, y_val)],
        "verbose": False,
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
        fit_kwargs["sample_weight_eval_set"] = [val_sample_weight]
    model.fit(X_train, y_train, **fit_kwargs)
    return model


def save_model(model, name: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"  -> Saved: outputs/models/{name}.pkl")
    return path


def load_model(name: str):
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model: {path}")
    return joblib.load(path)


def save_best_params(best_params: dict, filename: str = "best_hyperparams.json") -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / filename
    path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    print(f"  -> Saved: outputs/metrics/{filename}")
    return path


def load_best_params(filename: str = "best_hyperparams.json") -> dict:
    path = METRICS_DIR / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
