"""
covid_features.py – Build COVID-19 indicator features.

Adds:
    is_covid_year        – 1 for 2020-2021, 0 otherwise
    covid_stringency     – 0-100 OxCGRT monthly index
    cases_per_100k       – monthly new infections per 100k population
    vaccination_rate     – % of population fully vaccinated
    months_since_lockdown – count of months since Apr 2020
    is_post_covid        – 1 for 2022+, 0 otherwise

Reference period: April 2020 = month 0 since lockdown.
"""

from __future__ import annotations

import pandas as pd

LOCKDOWN_YEAR = 2020
LOCKDOWN_MONTH = 4  # Kosovo's first national lockdown was April 2020
COVID_YEARS = {2020, 2021}


def add_covid_features(
    growth_df: pd.DataFrame,
    covid_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join COVID features into the monthly growth dataset.
    """
    df = growth_df.merge(covid_df, on=["year", "month"], how="left")

    # Fill any missing values with zero (e.g. 2019 had no COVID, after merge it should still be 0)
    for col in ["stringency_index", "cases_per_100k", "vaccination_rate"]:
        df[col] = df[col].fillna(0.0)

    df["is_covid_year"] = df["year"].isin(COVID_YEARS).astype(int)
    df["is_post_covid"] = (df["year"] >= 2022).astype(int)

    # Months since April 2020 (negative for pre-lockdown → clipped to 0)
    df["months_since_lockdown"] = (
        (df["year"] - LOCKDOWN_YEAR) * 12 + (df["month"] - LOCKDOWN_MONTH)
    ).clip(lower=0)

    # Rename to project conventions
    df = df.rename(columns={"stringency_index": "covid_stringency"})

    return df
