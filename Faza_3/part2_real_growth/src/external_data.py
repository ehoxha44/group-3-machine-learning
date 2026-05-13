"""
external_data.py – Load and validate external macro data for Part 2.

Sources
-------
* `data/kosovo_cpi_monthly.csv`         – Kosovo CPI YoY (%) from ASK
* `data/covid_stringency_kosovo.csv`    – COVID stringency, cases, vaccination
* `data/inflation_forecast_imf.csv`     – IMF World Economic Outlook forecast

Usage
-----
    from part2_real_growth.src.external_data import load_cpi, load_covid, load_imf_forecast

    cpi   = load_cpi()
    covid = load_covid()
    imf   = load_imf_forecast()
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PART2_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PART2_DIR / "data"

CPI_PATH: Path = DATA_DIR / "kosovo_cpi_monthly.csv"
COVID_PATH: Path = DATA_DIR / "covid_stringency_kosovo.csv"
IMF_PATH: Path = DATA_DIR / "inflation_forecast_imf.csv"


def load_cpi() -> pd.DataFrame:
    """Return DataFrame with year, month, cpi_yoy_pct (float)."""
    df = pd.read_csv(CPI_PATH)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["cpi_yoy_pct"] = df["cpi_yoy_pct"].astype(float)
    df = df[["year", "month", "cpi_yoy_pct"]].sort_values(["year", "month"]).reset_index(drop=True)
    print(f"  CPI loaded: {len(df)} monthly rows from {df['year'].min()} to {df['year'].max()}")
    return df


def load_covid() -> pd.DataFrame:
    """Return DataFrame with year, month, stringency_index, cases_per_100k, vaccination_rate."""
    df = pd.read_csv(COVID_PATH)
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    for col in ["stringency_index", "cases_per_100k", "vaccination_rate"]:
        df[col] = df[col].astype(float)
    df = df[["year", "month", "stringency_index", "cases_per_100k", "vaccination_rate"]]
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    print(f"  COVID loaded: {len(df)} monthly rows from {df['year'].min()} to {df['year'].max()}")
    return df


def load_imf_forecast() -> pd.DataFrame:
    """Return DataFrame with year, scenario, cpi_yoy_pct for 2026-2027."""
    df = pd.read_csv(IMF_PATH)
    df["year"] = df["year"].astype(int)
    df["cpi_yoy_pct"] = df["cpi_yoy_pct"].astype(float)
    print(f"  IMF forecast: {len(df)} scenario rows for {sorted(df['year'].unique())}")
    return df


def validate() -> dict:
    """
    Run sanity checks on external data. Returns a dict of summary statistics
    that can be printed or saved to outputs/metrics/.
    """
    cpi = load_cpi()
    covid = load_covid()
    imf = load_imf_forecast()

    # Sanity check 1: 2022 should have peak inflation ~11.6% average
    cpi_2022_mean = cpi[cpi["year"] == 2022]["cpi_yoy_pct"].mean()
    assert 8.0 < cpi_2022_mean < 14.0, f"2022 CPI mean unrealistic: {cpi_2022_mean}"

    # Sanity check 2: 2020 should be near 0% (COVID deflation)
    cpi_2020_mean = cpi[cpi["year"] == 2020]["cpi_yoy_pct"].mean()
    assert -1.0 < cpi_2020_mean < 2.0, f"2020 CPI mean unrealistic: {cpi_2020_mean}"

    # Sanity check 3: 2020 stringency should peak
    string_2020_peak = covid[covid["year"] == 2020]["stringency_index"].max()
    assert string_2020_peak >= 70, f"2020 stringency peak too low: {string_2020_peak}"

    # Sanity check 4: 2023+ stringency should be near zero
    string_2023_mean = covid[covid["year"] == 2023]["stringency_index"].mean()
    assert string_2023_mean < 5, f"2023 stringency mean too high: {string_2023_mean}"

    summary = {
        "cpi_years": int(cpi["year"].nunique()),
        "cpi_2020_mean_pct": round(cpi_2020_mean, 2),
        "cpi_2022_mean_pct": round(cpi_2022_mean, 2),
        "cpi_2023_mean_pct": round(cpi[cpi["year"] == 2023]["cpi_yoy_pct"].mean(), 2),
        "cpi_2024_mean_pct": round(cpi[cpi["year"] == 2024]["cpi_yoy_pct"].mean(), 2),
        "covid_max_stringency": float(covid["stringency_index"].max()),
        "covid_2020_mean_stringency": round(float(covid[covid["year"] == 2020]["stringency_index"].mean()), 2),
        "imf_scenarios": sorted(imf["scenario"].unique().tolist()),
    }
    print("\n  Validation summary:")
    for k, v in summary.items():
        print(f"    {k}: {v}")
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        validate()
