from __future__ import annotations

import json

import pandas as pd

from .config import PipelineConfig
from .io_utils import write_dataframe, write_text


def random_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    sample_size = min(sample_size, len(df))
    return df.sample(n=sample_size, random_state=seed)


def stratified_sample(
    df: pd.DataFrame, group_column: str, sample_size: int, seed: int
) -> pd.DataFrame:
    sample_size = min(sample_size, len(df))
    fraction = sample_size / max(len(df), 1)
    sampled = (
        df.groupby(group_column, group_keys=False, dropna=False)
        .apply(lambda group: group.sample(n=max(1, round(len(group) * fraction)), random_state=seed))
    )
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed)
    return sampled.reset_index(drop=True)


def equal_period_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    sample_size = min(sample_size, len(df))
    if "year_month" not in df.columns:
        raise ValueError("year_month column is required for equal period sampling.")

    groups = [(name, group.copy()) for name, group in df.groupby("year_month", dropna=False)]
    if not groups:
        return df.head(0).copy()

    group_count = len(groups)
    base_take = sample_size // group_count
    remainder = sample_size % group_count
    sampled_parts: list[pd.DataFrame] = []
    leftovers: list[pd.DataFrame] = []

    for index, (_, group) in enumerate(sorted(groups, key=lambda item: str(item[0]))):
        target_n = base_take + (1 if index < remainder else 0)
        take_n = min(len(group), target_n)
        if take_n > 0:
            sampled = group.sample(n=take_n, random_state=seed)
            sampled_parts.append(sampled)
            remaining = group.drop(index=sampled.index)
        else:
            remaining = group
        if not remaining.empty:
            leftovers.append(remaining)

    sample_df = pd.concat(sampled_parts, ignore_index=False) if sampled_parts else df.head(0).copy()
    shortfall = sample_size - len(sample_df)
    if shortfall > 0 and leftovers:
        leftover_pool = pd.concat(leftovers, ignore_index=False)
        filler = leftover_pool.sample(n=min(shortfall, len(leftover_pool)), random_state=seed)
        sample_df = pd.concat([sample_df, filler], ignore_index=False)

    return sample_df.sample(frac=1, random_state=seed).reset_index(drop=True)


def build_subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    subsets = {
        "sector_focus": df[["year", "month", "primary_sector", "turnover_eur", "num_taxpayers"]],
        "municipality_focus": df[
            ["year", "month", "municipality", "turnover_eur", "num_taxpayers"]
        ],
        "registration_status_focus": df[
            ["year", "month", "registration_status", "turnover_eur", "num_taxpayers"]
        ],
        "monthly_time_pattern": df[
            ["year", "month", "year_month", "turnover_eur", "num_taxpayers"]
        ],
    }
    return subsets


def run_sample(
    config: PipelineConfig,
    sample_size: int = 5000,
    stratify_by: str | None = None,
    equal_by_period: bool = False,
) -> dict[str, str]:
    from .pipeline import load_cleaned_data

    config.ensure_output_dirs()
    df = load_cleaned_data(config, model_ready=True)
    use_equal_by_period = equal_by_period or not stratify_by
    if use_equal_by_period:
        sample_df = equal_period_sample(df, sample_size, config.random_seed)
        method = "equal_year_month"
    elif stratify_by:
        sample_df = stratified_sample(df, stratify_by, sample_size, config.random_seed)
        method = "stratified"
    else:
        sample_df = random_sample(df, sample_size, config.random_seed)
        method = "random"

    sample_path = config.sample_dir / f"{method}_sample.csv"
    write_dataframe(sample_df, sample_path)
    metadata = {
        "method": method,
        "sample_size": int(len(sample_df)),
        "seed": config.random_seed,
        "stratify_by": stratify_by,
        "equal_by_period": use_equal_by_period,
    }
    metadata_path = config.sample_dir / f"{method}_sample_metadata.json"
    write_text(json.dumps(metadata, indent=2), metadata_path)

    outputs = {"sample": str(sample_path), "metadata": str(metadata_path)}
    for name, subset in build_subsets(sample_df).items():
        path = config.sample_dir / f"{name}.csv"
        write_dataframe(subset, path)
        outputs[name] = str(path)
    return outputs
