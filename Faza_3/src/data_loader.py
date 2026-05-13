"""
data_loader.py - Monthly aggregation and forecasting inputs for Phase 3.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.config import (
    CLEAN_DATA_PATH,
    FORECAST_YEAR,
    GROWTH_LOWER,
    GROWTH_UPPER,
    MIN_GROUP_MONTHS,
    TARGET_CLIP_LOWER,
    TARGET_CLIP_UPPER,
    TARGET_COLUMN,
)


def load_data() -> pd.DataFrame:
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {CLEAN_DATA_PATH}\nRun the Phase 1 pipeline first."
        )
    df = pd.read_csv(CLEAN_DATA_PATH, low_memory=False)
    print(f"Loading {CLEAN_DATA_PATH.name}: {len(df):,} rows x {len(df.columns)} columns")
    return df


def _classify_growth(value: float) -> str:
    if value > GROWTH_UPPER:
        return "GROWING"
    if value < GROWTH_LOWER:
        return "DECLINING"
    return "STABLE"


def _validate_input(df: pd.DataFrame) -> None:
    required = [
        "year",
        "month",
        "municipality",
        "primary_sector",
        "turnover_eur",
        "num_taxpayers",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in input: {missing}")


def build_growth_dataset(df: pd.DataFrame) -> pd.DataFrame:
    _validate_input(df)

    work = df.copy()
    work["year"] = work["year"].astype(int)
    work["month"] = work["month"].astype(int)

    monthly = (
        work.groupby(["year", "month", "municipality", "primary_sector"], sort=True)
        .agg(
            total_turnover=("turnover_eur", "sum"),
            num_businesses=("num_taxpayers", "sum"),
        )
        .reset_index()
    )
    print(
        f"  -> Aggregated to {len(monthly):,} "
        "(year x month x municipality x sector) rows"
    )

    market_conc = (
        work.groupby(["year", "month", "municipality"], sort=True)["num_taxpayers"]
        .sum()
        .rename("market_concentration")
        .reset_index()
    )
    monthly = monthly.merge(
        market_conc,
        on=["year", "month", "municipality"],
        how="left",
    )

    monthly["period"] = pd.PeriodIndex.from_fields(
        year=monthly["year"],
        month=monthly["month"],
        freq="M",
    )
    monthly = monthly.sort_values(["municipality", "primary_sector", "period"]).reset_index(drop=True)

    group_keys = ["municipality", "primary_sector"]
    grouped = monthly.groupby(group_keys, sort=False)
    monthly["history_months"] = grouped["period"].transform("count")
    monthly["lag1_turnover"] = grouped["total_turnover"].shift(1)
    monthly["lag2_turnover"] = grouped["total_turnover"].shift(2)
    monthly["lag3_turnover"] = grouped["total_turnover"].shift(3)
    monthly["lag12_turnover"] = grouped["total_turnover"].shift(12)
    monthly["prev_turnover"] = monthly["lag12_turnover"]

    lag1 = grouped["total_turnover"].shift(1)
    monthly["rolling3_mean"] = lag1.groupby([monthly["municipality"], monthly["primary_sector"]]).transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    monthly["rolling6_mean"] = lag1.groupby([monthly["municipality"], monthly["primary_sector"]]).transform(
        lambda s: s.rolling(6, min_periods=1).mean()
    )
    monthly["rolling3_std"] = lag1.groupby([monthly["municipality"], monthly["primary_sector"]]).transform(
        lambda s: s.rolling(3, min_periods=2).std()
    )
    monthly["rolling6_std"] = lag1.groupby([monthly["municipality"], monthly["primary_sector"]]).transform(
        lambda s: s.rolling(6, min_periods=2).std()
    )

    monthly["growth_rate"] = (
        (monthly["total_turnover"] - monthly["lag12_turnover"])
        / monthly["lag12_turnover"].replace(0, np.nan)
    ).clip(lower=TARGET_CLIP_LOWER, upper=TARGET_CLIP_UPPER)

    for column in [
        "prev_turnover",
        "lag1_turnover",
        "lag2_turnover",
        "lag3_turnover",
        "rolling3_mean",
        "rolling6_mean",
        "market_concentration",
        "total_turnover",
        "num_businesses",
    ]:
        monthly[f"{column}_log1p"] = np.log1p(monthly[column].clip(lower=0))

    monthly["month_sin"] = np.sin(2 * math.pi * monthly["month"] / 12.0)
    monthly["month_cos"] = np.cos(2 * math.pi * monthly["month"] / 12.0)
    monthly["growth_class"] = monthly["growth_rate"].apply(_classify_growth)

    before = len(monthly)
    monthly = monthly[monthly["history_months"] >= MIN_GROUP_MONTHS].copy()
    monthly = monthly.dropna(subset=["lag12_turnover", "growth_rate"])
    monthly["rolling3_std"] = monthly["rolling3_std"].fillna(0.0)
    monthly["rolling6_std"] = monthly["rolling6_std"].fillna(0.0)
    monthly = monthly.dropna(
        subset=[
            "lag1_turnover",
            "lag2_turnover",
            "lag3_turnover",
            "rolling3_mean",
            "rolling6_mean",
        ]
    )
    print(f"  -> Dropped {before - len(monthly):,} rows due to history / lag requirements")

    dist = monthly[TARGET_COLUMN].value_counts()
    print("  -> Growth class distribution:")
    for cls, cnt in dist.items():
        print(f"       {cls:<12} {cnt:6,} ({cnt / len(monthly) * 100:.1f}%)")

    return monthly.reset_index(drop=True)


def build_prediction_input(
    growth_df: pd.DataFrame,
    forecast_year: int = FORECAST_YEAR,
) -> pd.DataFrame:
    latest_year = int(growth_df["year"].max())
    base = growth_df[growth_df["year"] == latest_year].copy()
    pred = base[
        [
            "municipality",
            "primary_sector",
            "month",
            "prev_turnover",
            "prev_turnover_log1p",
            "lag1_turnover_log1p",
            "lag2_turnover_log1p",
            "lag3_turnover_log1p",
            "rolling3_mean_log1p",
            "rolling6_mean_log1p",
            "rolling3_std",
            "rolling6_std",
            "num_businesses",
            "num_businesses_log1p",
            "market_concentration_log1p",
            "month_sin",
            "month_cos",
        ]
    ].copy()
    pred["year"] = forecast_year
    pred["forecast_base_year"] = latest_year
    pred["turnover_reference_eur"] = base["prev_turnover"].values
    print(
        f"  -> Built {len(pred):,} static forecast rows for {forecast_year} "
        f"from {latest_year} monthly history"
    )
    return pred.reset_index(drop=True)


def build_growth_pivot(
    growth_df: pd.DataFrame,
    group_col: str,
    value_col: str = "growth_rate",
    top_n: int | None = None,
) -> pd.DataFrame:
    work = growth_df.copy()
    if top_n:
        top_groups = (
            work.groupby(group_col)["total_turnover"]
            .sum()
            .nlargest(top_n)
            .index
        )
        work = work[work[group_col].isin(top_groups)]

    pivot = (
        work.groupby([group_col, "period"])[value_col]
        .mean()
        .unstack(level="period")
        .fillna(0)
    )
    pivot.columns = [str(period) for period in pivot.columns]
    return pivot
