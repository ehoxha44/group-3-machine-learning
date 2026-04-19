"""
data_loader.py – Load raw Phase 1 data and build the growth analysis dataset.

Two-stage pipeline
------------------
Stage 1 – load_data()
    Read model_ready.csv produced by Phase 1.

Stage 2 – build_growth_dataset()
    Aggregate rows to one record per (year, municipality, primary_sector).
    Compute YoY growth rate and classify into GROWING / STABLE / DECLINING.
    The result is the dataset actually used for modelling.

The output of Stage 2 has one row per group-year combination and answers the
question: "Did this sector in this municipality grow / stay stable / decline
compared with the previous year?"  That is exactly the granularity needed to
compare how each industry performed in each city.
"""

import numpy as np
import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    CLEAN_DATA_PATH,
    GROWTH_LOWER,
    GROWTH_UPPER,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
)


# ── Stage 1: raw load ─────────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    """Read the cleaned Phase 1 CSV."""
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {CLEAN_DATA_PATH}\n"
            "Run the Phase 1 pipeline first."
        )
    print(f"Loading: {CLEAN_DATA_PATH.name}  ({CLEAN_DATA_PATH.stat().st_size / 1e6:.1f} MB)")
    df = pd.read_csv(CLEAN_DATA_PATH, low_memory=False)
    print(f"  → {len(df):,} rows × {len(df.columns)} columns")
    return df


# ── Stage 2: growth dataset builder ──────────────────────────────────────────


def build_growth_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to (year, municipality, primary_sector) and engineer growth features.

    Steps
    -----
    1. Aggregate: sum turnover_eur and num_taxpayers per group-year.
    2. Lag: compute previous year's turnover for each (municipality, sector) pair.
    3. Growth rate: (current − previous) / previous, clipped to [−2, +5].
    4. Log-transform: log1p on turnover and business count.
    5. Classify: GROWING (> +5 %) / STABLE (−5 % to +5 %) / DECLINING (< −5 %).
    6. Drop: first year for each group (no lag available) and any remaining nulls.

    Returns
    -------
    DataFrame with one row per (year, municipality, primary_sector) combination
    from year 2020 onward (first lag year is 2019 → 2020 growth).
    """
    required = ["year", "municipality", "primary_sector", "turnover_eur", "num_taxpayers"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in input: {missing}")

    # ── 1. Aggregate ──────────────────────────────────────────────────────────
    agg = (
        df.groupby(["year", "municipality", "primary_sector"], sort=True)
        .agg(
            total_turnover=("turnover_eur", "sum"),
            num_businesses=("num_taxpayers", "sum"),
        )
        .reset_index()
    )
    print(f"  → Aggregated to {len(agg):,} (year × municipality × sector) rows")

    # ── 2. Lag feature (previous year's turnover for same group) ──────────────
    agg = agg.sort_values(["municipality", "primary_sector", "year"])
    agg["prev_turnover"] = agg.groupby(["municipality", "primary_sector"])[
        "total_turnover"
    ].shift(1)

    # ── 3. YoY growth rate ────────────────────────────────────────────────────
    # Avoid division by zero; clip extreme values caused by near-zero bases
    agg["growth_rate"] = (
        (agg["total_turnover"] - agg["prev_turnover"])
        / agg["prev_turnover"].replace(0, np.nan)
    ).clip(lower=-2.0, upper=5.0)

    # ── 4. Log-transforms ─────────────────────────────────────────────────────
    agg["total_turnover_log1p"] = np.log1p(agg["total_turnover"].clip(lower=0))
    agg["prev_turnover_log1p"] = np.log1p(agg["prev_turnover"].clip(lower=0))
    agg["num_businesses_log1p"] = np.log1p(agg["num_businesses"].clip(lower=0))

    # ── 5. Growth classification ──────────────────────────────────────────────
    def classify(r: float) -> str:
        if r > GROWTH_UPPER:
            return "GROWING"
        if r < GROWTH_LOWER:
            return "DECLINING"
        return "STABLE"

    agg[TARGET_COLUMN] = agg["growth_rate"].apply(classify)

    # ── 6. Drop rows with no lag (first year per group) and remaining nulls ───
    before = len(agg)
    agg = agg.dropna(subset=["prev_turnover", "growth_rate"])
    print(
        f"  → Dropped {before - len(agg):,} first-year rows (no prior data); "
        f"{len(agg):,} rows remain"
    )

    # ── Class distribution ────────────────────────────────────────────────────
    dist = agg[TARGET_COLUMN].value_counts()
    print("  → Growth class distribution:")
    for cls, cnt in dist.items():
        print(f"       {cls:<12} {cnt:5,}  ({cnt / len(agg) * 100:.1f}%)")

    return agg.reset_index(drop=True)


# ── Helper: growth pivot matrix ────────────────────────────────────────────────


def build_prediction_input(
    growth_df: pd.DataFrame,
    forecast_year: int = 2026,
) -> pd.DataFrame:
    """
    Build the feature rows needed to predict growth for `forecast_year`.

    Logic
    -----
    Take the most recent year's aggregated data (e.g. 2025) and treat it as
    the "previous year" for the forecast:
      • prev_turnover       = total_turnover from the latest year
      • prev_turnover_log1p = log1p of the above
      • num_businesses      = same business count (assumed similar)
      • year                = forecast_year (e.g. 2026)

    Only (municipality, sector) pairs that existed in the latest year are
    included — we cannot forecast groups we have never seen.

    Returns the raw DataFrame (municipality, primary_sector, year,
    prev_turnover, prev_turnover_log1p, num_businesses, num_businesses_log1p,
    total_turnover).  The caller must encode it with the fitted encoders.
    """
    import numpy as np
    latest_year = int(growth_df["year"].max())
    latest = growth_df[growth_df["year"] == latest_year].copy()

    pred = latest[["municipality", "primary_sector",
                   "total_turnover", "num_businesses",
                   "total_turnover_log1p", "num_businesses_log1p"]].copy()

    pred["year"] = forecast_year
    pred["prev_turnover"] = pred["total_turnover"]
    pred["prev_turnover_log1p"] = pred["total_turnover_log1p"]

    print(
        f"  → Built {len(pred):,} prediction rows for {forecast_year} "
        f"(based on {latest_year} actuals)"
    )
    return pred.reset_index(drop=True)


def build_growth_pivot(
    growth_df: pd.DataFrame,
    group_col: str,                 # "municipality" or "primary_sector"
    value_col: str = "growth_rate",
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Pivot growth rates into a matrix: group × year.

    Each cell is the mean growth_rate for that group in that year.
    Used for unsupervised trajectory clustering and heatmap visualisation.

    Parameters
    ----------
    group_col : column to use as row index ("municipality" or "primary_sector")
    value_col : numeric column to aggregate (default: "growth_rate")
    top_n     : if given, keep only the top-n groups by total_turnover
    """
    if top_n:
        top_groups = (
            growth_df.groupby(group_col)["total_turnover"]
            .sum()
            .nlargest(top_n)
            .index
        )
        growth_df = growth_df[growth_df[group_col].isin(top_groups)]

    pivot = (
        growth_df.groupby([group_col, "year"])[value_col]
        .mean()
        .unstack(level="year")
        .fillna(0)
    )
    return pivot
