from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .constants import BUSINESS_COLUMNS, CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from .io_utils import write_dataframe


@dataclass(slots=True)
class CleaningResult:
    strict_clean_df: pd.DataFrame
    model_ready_df: pd.DataFrame
    cleaning_log: pd.DataFrame
    invalid_rows: pd.DataFrame


def _standardize_blank_strings(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    cleaned = df.copy()
    replacements = 0
    for column in cleaned.columns:
        if pd.api.types.is_object_dtype(cleaned[column]) or pd.api.types.is_string_dtype(
            cleaned[column]
        ):
            before_non_null = cleaned[column].isna().sum()
            cleaned[column] = cleaned[column].apply(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            cleaned[column] = cleaned[column].replace(r"^\s*$", np.nan, regex=True)
            after_non_null = cleaned[column].isna().sum()
            replacements += max(after_non_null - before_non_null, 0)
    return cleaned, replacements


def _coerce_numeric(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    cleaned = df.copy()
    issues: list[dict[str, object]] = []
    for column in NUMERIC_COLUMNS:
        original_non_null = cleaned[column].notna()
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        conversion_failures = int((original_non_null & cleaned[column].isna()).sum())
        issues.append(
            {"action": "coerce_numeric", "column": column, "count": conversion_failures}
        )
    return cleaned, issues


def _standardize_text_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    cleaned = df.copy()
    issues: list[dict[str, object]] = []
    for column in CATEGORICAL_COLUMNS:
        if column == "municipality":
            cleaned[column] = (
                cleaned[column]
                .astype("string")
                .str.normalize("NFKC")
                .str.strip()
                .str.upper()
            )
        elif column == "registration_status":
            cleaned[column] = (
                cleaned[column]
                .astype("string")
                .str.normalize("NFKC")
                .str.strip()
                .str.upper()
            )
        else:
            cleaned[column] = (
                cleaned[column]
                .astype("string")
                .str.normalize("NFKC")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
            )
        issues.append({"action": "standardize_text", "column": column, "count": len(cleaned)})
        cleaned[column] = cleaned[column].replace("<NA>", pd.NA)
    return cleaned, issues


def _drop_non_data_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    cleaned = df.copy()
    all_business_missing = cleaned[list(BUSINESS_COLUMNS)].isna().all(axis=1)
    repeated_header = cleaned["year"].astype("string").str.contains("Year", na=False)
    mask = all_business_missing | repeated_header
    removed_count = int(mask.sum())
    cleaned = cleaned.loc[~mask].copy()
    return cleaned, [{"action": "drop_empty_or_header_rows", "column": "*", "count": removed_count}]


def _validate_domains(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    invalid_reason = pd.Series("", index=df.index, dtype="string")

    year_invalid = df["year"].notna() & ~df["year"].between(config.year_min, config.year_max)
    month_invalid = df["month"].notna() & ~df["month"].between(1, 12)
    taxpayers_invalid = df["num_taxpayers"].notna() & (df["num_taxpayers"] < 0)
    turnover_invalid = df["turnover_eur"].notna() & (df["turnover_eur"] < 0)

    invalid_reason = invalid_reason.mask(year_invalid, invalid_reason + "invalid_year;")
    invalid_reason = invalid_reason.mask(month_invalid, invalid_reason + "invalid_month;")
    invalid_reason = invalid_reason.mask(
        taxpayers_invalid, invalid_reason + "invalid_num_taxpayers;"
    )
    invalid_reason = invalid_reason.mask(
        turnover_invalid, invalid_reason + "invalid_turnover;"
    )

    validated = df.copy()
    validated["invalid_reason"] = invalid_reason.str.rstrip(";")
    return validated


def _build_model_ready(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    model_ready = df.copy()
    for column in NUMERIC_COLUMNS:
        flag_column = f"{column}_was_imputed"
        model_ready[flag_column] = model_ready[column].isna()
        median_value = model_ready[column].median()
        if not math.isnan(median_value):
            model_ready[column] = model_ready[column].fillna(median_value)
    for column in CATEGORICAL_COLUMNS:
        flag_column = f"{column}_was_imputed"
        model_ready[flag_column] = model_ready[column].isna()
        model_ready[column] = model_ready[column].fillna(config.categorical_fill_value)
    model_ready["year_month"] = (
        model_ready["year"].astype("Int64").astype("string")
        + "-"
        + model_ready["month"].astype("Int64").astype("string").str.zfill(2)
    )
    strictly_positive = model_ready["turnover_eur"].where(model_ready["turnover_eur"] > 0)
    model_ready["turnover_eur_log1p"] = np.log1p(strictly_positive.fillna(0))
    return model_ready


def clean_dataframe(df: pd.DataFrame, config: PipelineConfig) -> CleaningResult:
    logs: list[dict[str, object]] = []

    cleaned, replacements = _standardize_blank_strings(df)
    logs.append({"action": "standardize_blank_strings", "column": "*", "count": replacements})

    cleaned, drop_logs = _drop_non_data_rows(cleaned)
    logs.extend(drop_logs)

    cleaned, numeric_logs = _coerce_numeric(cleaned)
    logs.extend(numeric_logs)

    cleaned, text_logs = _standardize_text_columns(cleaned)
    logs.extend(text_logs)

    duplicate_mask = cleaned.duplicated()
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        cleaned = cleaned.loc[~duplicate_mask].copy()
    logs.append({"action": "drop_duplicates", "column": "*", "count": duplicate_count})

    validated = _validate_domains(cleaned, config)
    invalid_rows = validated.loc[validated["invalid_reason"].notna() & (validated["invalid_reason"] != "")]
    strict_clean_df = validated.copy()
    model_ready_df = _build_model_ready(strict_clean_df, config)
    cleaning_log = pd.DataFrame(logs)

    return CleaningResult(
        strict_clean_df=strict_clean_df,
        model_ready_df=model_ready_df,
        cleaning_log=cleaning_log,
        invalid_rows=invalid_rows,
    )


def run_clean(config: PipelineConfig) -> dict[str, str]:
    from .pipeline import load_normalized_data

    config.ensure_output_dirs()
    df = load_normalized_data(config)
    result = clean_dataframe(df, config)
    strict_path = config.clean_dir / "strict_cleaned.csv"
    model_ready_path = config.clean_dir / "model_ready.csv"
    cleaning_log_path = config.clean_dir / "cleaning_log.csv"
    invalid_rows_path = config.clean_dir / "invalid_rows.csv"
    write_dataframe(result.strict_clean_df, strict_path)
    write_dataframe(result.model_ready_df, model_ready_path)
    write_dataframe(result.cleaning_log, cleaning_log_path)
    write_dataframe(result.invalid_rows, invalid_rows_path)
    return {
        "strict_cleaned": str(strict_path),
        "model_ready": str(model_ready_path),
        "cleaning_log": str(cleaning_log_path),
        "invalid_rows": str(invalid_rows_path),
    }
