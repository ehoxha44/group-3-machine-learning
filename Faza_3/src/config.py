"""
config.py - Central configuration for Phase 3.
"""

from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parent.parent
FAZA1_OUTPUTS: Path = BASE_DIR.parent / "Faza 1" / "outputs"
PHASE2_DIR: Path = BASE_DIR.parent / "Faza_2"
PHASE2_METRICS_DIR: Path = PHASE2_DIR / "outputs" / "metrics"
CLEAN_DATA_PATH: Path = FAZA1_OUTPUTS / "clean" / "model_ready.csv"

OUTPUTS_DIR: Path = BASE_DIR / "outputs"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"

RANDOM_STATE: int = 42

GROWTH_UPPER: float = 0.05
GROWTH_LOWER: float = -0.05
TARGET_COLUMN: str = "growth_class"
REGRESSION_TARGET: str = "growth_rate"

CATEGORICAL_FEATURES: list[str] = [
    "municipality",
    "primary_sector",
]

BASELINE_NUMERIC_FEATURES: list[str] = [
    "year",
    "prev_turnover_log1p",
    "num_businesses_log1p",
]

NUMERIC_FEATURES: list[str] = [
    "year",
    "month",
    "prev_turnover_log1p",
    "lag1_turnover_log1p",
    "lag2_turnover_log1p",
    "lag3_turnover_log1p",
    "rolling3_mean_log1p",
    "rolling6_mean_log1p",
    "rolling3_std",
    "rolling6_std",
    "num_businesses_log1p",
    "market_concentration_log1p",
    "month_sin",
    "month_cos",
]

TRAIN_MAX_YEAR: int = 2023
VAL_YEAR: int = 2024
TEST_YEAR: int = 2025

FORECAST_YEAR: int = 2027
TUNE_ITER: int = 25
TUNE_CV_FOLDS: int = 4
EARLY_STOPPING_ROUNDS: int = 30

ENCODING_STRATEGY: str = "target"
# "auto": oversample every class except the majority. Crucial here — Phase-3
# audit showed that "minority" (only STABLE) was insufficient because
# DECLINING is also under-represented at monthly granularity, and the model
# kept collapsing predictions onto GROWING.
SMOTE_STRATEGY: str = "auto"
SMOTE_K_NEIGHBORS: int = 5
MIN_GROUP_MONTHS: int = 24

PRED_CLIP_LOWER: float = -1.0
PRED_CLIP_UPPER: float = 2.0
TARGET_CLIP_LOWER: float = -2.0
TARGET_CLIP_UPPER: float = 5.0

TOP_MUNICIPALITIES: int = 20
TOP_SECTORS: int = 15

RF_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RF_REG_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_REG_PARAMS: dict = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "eval_metric": "rmse",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RF_PARAM_DIST: dict = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [8, 10, 12, 16, None],
    "min_samples_split": [2, 4, 6, 8, 12],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", "log2", None],
    "class_weight": ["balanced", "balanced_subsample"],
}

XGB_PARAM_DIST: dict = {
    "n_estimators": [200, 300, 400, 500, 700],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.4],
    "reg_alpha": [0.0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}

RF_REG_PARAM_DIST: dict = {
    "n_estimators": [200, 300, 400, 500],
    "max_depth": [8, 10, 12, 16, None],
    "min_samples_split": [2, 4, 6, 8, 12],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", "log2", None],
}

XGB_REG_PARAM_DIST: dict = {
    "n_estimators": [200, 300, 400, 500, 700],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.4],
    "reg_alpha": [0.0, 0.01, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0, 5.0],
}

KMEANS_K: int = 4
N_CLUSTERS_RANGE: range = range(2, 9)
N_PCA_COMPONENTS: int = 2
