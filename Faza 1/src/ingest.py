from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .constants import IGNORED_RAW_COLUMNS, RAW_TO_CANONICAL
from .io_utils import write_dataframe


def read_raw_excel(config: PipelineConfig) -> pd.DataFrame:
    df = pd.read_excel(config.input_path, header=config.header_row_index)
    ignored_columns = [column for column in IGNORED_RAW_COLUMNS if column in df.columns]
    return df.drop(columns=ignored_columns)


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ignored_columns = [column for column in IGNORED_RAW_COLUMNS if column in df.columns]
    if ignored_columns:
        df = df.drop(columns=ignored_columns)
    empty_columns = [column for column in df.columns if df[column].isna().all()]
    df = df.drop(columns=empty_columns)
    df = df.rename(columns=RAW_TO_CANONICAL)
    return df


def run_ingest(config: PipelineConfig) -> dict[str, Path]:
    config.ensure_output_dirs()
    raw_df = read_raw_excel(config)
    normalized_df = normalize_schema(raw_df)

    raw_extract_path = config.ingest_dir / "raw_extract.csv"
    normalized_path = config.ingest_dir / "normalized.csv"
    write_dataframe(raw_df, raw_extract_path)
    write_dataframe(normalized_df, normalized_path)

    return {"raw_extract": raw_extract_path, "normalized": normalized_path}
