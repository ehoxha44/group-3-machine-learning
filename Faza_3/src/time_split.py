"""
time_split.py - Time-aware train/validation/test helpers.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import TEST_YEAR, TRAIN_MAX_YEAR, VAL_YEAR


def time_based_split(
    df: pd.DataFrame,
    train_max_year: int = TRAIN_MAX_YEAR,
    val_year: int = VAL_YEAR,
    test_year: int = TEST_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["year"] <= train_max_year].copy()
    val_df = df[df["year"] == val_year].copy()
    test_df = df[df["year"] == test_year].copy()

    if len(test_df) < 1000:
        print(
            f"  -> Test year {test_year} has only {len(test_df):,} rows; "
            "falling back to val=2023 and test=2024."
        )
        train_df = df[df["year"] <= 2022].copy()
        val_df = df[df["year"] == 2023].copy()
        test_df = df[df["year"] == 2024].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Chronological split produced an empty partition.")

    assert train_df["year"].max() < val_df["year"].min()
    assert val_df["year"].max() <= test_df["year"].min()

    print(
        "Chronological split:\n"
        f"  train <= {int(train_df['year'].max())}: {len(train_df):,}\n"
        f"  val   = {sorted(val_df['year'].unique().tolist())}: {len(val_df):,}\n"
        f"  test  = {sorted(test_df['year'].unique().tolist())}: {len(test_df):,}"
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    return TimeSeriesSplit(n_splits=n_splits)
