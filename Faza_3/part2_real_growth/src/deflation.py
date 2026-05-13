"""
deflation.py – Convert nominal growth to real growth using Fisher formula.

Formula (Fisher / exact)
------------------------
    real_growth = (1 + nominal_growth) / (1 + inflation) - 1

This is more accurate than the linear approximation `real ≈ nominal - inflation`
when inflation is large (which it was in 2022 for Kosovo at ~11.6%).

The classification thresholds for real_growth_class are TIGHTER than nominal:
±3% instead of ±5%, because real growth has lower variance once inflation
has been removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Tighter thresholds for real growth class (vs ±5% for nominal in Phase 3 Part 1)
REAL_GROWTH_UPPER: float = 0.03   # +3%
REAL_GROWTH_LOWER: float = -0.03  # -3%


def add_inflation_features(
    growth_df: pd.DataFrame,
    cpi_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join CPI features into the monthly growth dataset and compute real_growth_rate.

    Adds columns:
        cpi_yoy_pct           – inflation YoY for that (year, month)
        cpi_yoy               – same as decimal (0.116 instead of 11.6)
        real_growth_rate      – Fisher-deflated growth rate
        real_growth_class     – GROWING / STABLE / DECLINING with ±3% threshold
        inflation_shock       – binary: 1 if cpi_yoy_pct > 8%
    """
    df = growth_df.merge(cpi_df, on=["year", "month"], how="left")

    # Forward-fill CPI for any months missing (e.g. partial 2025)
    missing_before = df["cpi_yoy_pct"].isna().sum()
    if missing_before > 0:
        df["cpi_yoy_pct"] = df["cpi_yoy_pct"].ffill().bfill()
        print(f"  CPI: filled {missing_before:,} missing rows via forward/back fill")

    df["cpi_yoy"] = df["cpi_yoy_pct"] / 100.0

    # Fisher exact deflation
    df["real_growth_rate"] = (
        (1.0 + df["growth_rate"]) / (1.0 + df["cpi_yoy"]) - 1.0
    )
    # Clip to same range as nominal (Phase-3 used [-2, +5])
    df["real_growth_rate"] = df["real_growth_rate"].clip(lower=-2.0, upper=5.0)

    df["real_growth_class"] = df["real_growth_rate"].apply(_classify_real)

    df["inflation_shock"] = (df["cpi_yoy_pct"] > 8.0).astype(int)

    return df


def _classify_real(value: float) -> str:
    if value > REAL_GROWTH_UPPER:
        return "REAL_GROWING"
    if value < REAL_GROWTH_LOWER:
        return "REAL_DECLINING"
    return "REAL_STABLE"


def deflate_value(nominal_growth: float, inflation_yoy: float) -> float:
    """
    Apply Fisher formula to a single value.
    `inflation_yoy` is in decimal form (e.g. 0.025 for 2.5%).
    """
    return (1.0 + nominal_growth) / (1.0 + inflation_yoy) - 1.0


def reflate_value(real_growth: float, inflation_yoy: float) -> float:
    """
    Inverse: real → nominal (for projecting 2027 nominal from a real forecast).
    `inflation_yoy` is in decimal form.
    """
    return (1.0 + real_growth) * (1.0 + inflation_yoy) - 1.0
