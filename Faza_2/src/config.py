"""
config.py – Centralised configuration for Phase 2.

Goal: predict and cluster economic growth per municipality / sector / year.

The unit of analysis is one (year, municipality, primary_sector) combination.
The target is 'growth_class' – whether turnover grew, was stable, or declined
year-over-year.  Features are derived from the previous year's aggregated
turnover and business count for that same municipality+sector pair.
"""

from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────

BASE_DIR: Path = Path(__file__).resolve().parent.parent
FAZA1_OUTPUTS: Path = BASE_DIR.parent / "Faza 1" / "outputs"
CLEAN_DATA_PATH: Path = FAZA1_OUTPUTS / "clean" / "model_ready.csv"

OUTPUTS_DIR: Path = BASE_DIR / "outputs"
MODELS_DIR: Path = OUTPUTS_DIR / "models"
PLOTS_DIR: Path = OUTPUTS_DIR / "plots"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"

# ── Reproducibility ───────────────────────────────────────────────────────────

RANDOM_STATE: int = 42

# ── Train / test split ────────────────────────────────────────────────────────

TEST_SIZE: float = 0.20

# ── Growth labelling thresholds ───────────────────────────────────────────────

# YoY growth rate:  > GROWTH_UPPER  → GROWING
#                   < GROWTH_LOWER  → DECLINING
#                   between         → STABLE
GROWTH_UPPER: float = 0.05    # +5 %
GROWTH_LOWER: float = -0.05   # −5 %

# ── Feature definitions (on the aggregated growth dataset) ───────────────────

# Numeric features derived after aggregation + lag engineering.
# IMPORTANT – only features known BEFORE the current year's outcome are included.
# Excluded (leakage):
#   growth_rate        → directly encodes the target (derived from current turnover)
#   total_turnover_log1p → current year result, not known at prediction time
NUMERIC_FEATURES: list[str] = [
    "year",
    "prev_turnover_log1p",   # log1p of previous year's total turnover (known)
    "num_businesses_log1p",  # log1p of taxpayer count in current year (census)
]

# Categorical features encoded at modelling time
CATEGORICAL_FEATURES: list[str] = [
    "municipality",
    "primary_sector",
]

# Supervised target produced by build_growth_dataset()
TARGET_COLUMN: str = "growth_class"   # GROWING / STABLE / DECLINING

# ── Supervised hyper-parameters ───────────────────────────────────────────────

RF_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 4,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ── Unsupervised hyper-parameters ─────────────────────────────────────────────

N_CLUSTERS_RANGE: range = range(2, 9)
KMEANS_K: int = 4             # four groups: high growth / stable / recovering / declining
N_PCA_COMPONENTS: int = 2

# ── Regression settings ───────────────────────────────────────────────────────

# Continuous target for regression models (raw YoY growth rate, clipped [-2, +5])
REGRESSION_TARGET: str = "growth_rate"

RF_REG_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_split": 4,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

XGB_REG_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "rmse",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ── Prediction settings ───────────────────────────────────────────────────────

FORECAST_YEAR: int = 2026

# ── Heatmap / analysis settings ───────────────────────────────────────────────

# Top-N municipalities and sectors shown in growth heatmaps
TOP_MUNICIPALITIES: int = 20
TOP_SECTORS: int = 15
