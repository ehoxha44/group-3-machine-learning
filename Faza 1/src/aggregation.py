from __future__ import annotations

import pandas as pd

from .config import PipelineConfig
from .io_utils import write_dataframe


def _aggregate(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    grouped = (
        df.groupby(group_column, dropna=False)
        .agg(
            turnover_sum=("turnover_eur", "sum"),
            turnover_mean=("turnover_eur", "mean"),
            turnover_median=("turnover_eur", "median"),
            turnover_min=("turnover_eur", "min"),
            turnover_max=("turnover_eur", "max"),
            taxpayers_sum=("num_taxpayers", "sum"),
            taxpayers_mean=("num_taxpayers", "mean"),
            row_count=("turnover_eur", "size"),
        )
        .reset_index()
        .sort_values("turnover_sum", ascending=False)
    )
    return grouped


def build_aggregations(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_year = _aggregate(df, "year")
    by_month = _aggregate(df, "month")
    by_municipality = _aggregate(df, "municipality")
    by_sector = _aggregate(df, "primary_sector")
    by_registration_status = _aggregate(df, "registration_status")
    return {
        "turnover_by_year": by_year,
        "turnover_by_month": by_month,
        "turnover_by_municipality": by_municipality,
        "turnover_by_sector": by_sector,
        "turnover_by_registration_status": by_registration_status,
    }


def run_aggregate(config: PipelineConfig) -> dict[str, str]:
    from .pipeline import load_cleaned_data

    config.ensure_output_dirs()
    df = load_cleaned_data(config, model_ready=False)
    outputs = {}
    for name, table in build_aggregations(df).items():
        path = config.aggregate_dir / f"{name}.csv"
        write_dataframe(table, path)
        outputs[name] = str(path)
    return outputs
