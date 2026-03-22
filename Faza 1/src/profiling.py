from __future__ import annotations

import pandas as pd

from .config import PipelineConfig
from .constants import BUSINESS_COLUMNS, NUMERIC_COLUMNS, RAW_TO_CANONICAL
from .io_utils import write_dataframe


def infer_semantic_type(series: pd.Series) -> str:
    non_null = series.dropna()
    if non_null.empty:
        return "empty"
    if pd.api.types.is_numeric_dtype(non_null):
        return "numeric"
    if series.name == "year_month":
        return "date-like"
    unique_ratio = non_null.nunique() / max(len(non_null), 1)
    if unique_ratio > 0.95:
        return "identifier-like"
    if unique_ratio < 0.2:
        return "categorical"
    return "mixed"


def build_schema_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        null_rate = float(df[column].isna().mean())
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "semantic_type": infer_semantic_type(df[column]),
                "null_count": int(df[column].isna().sum()),
                "null_rate_decimal": round(null_rate, 8),
                "null_rate_percent": f"{null_rate * 100:.6f}%",
                "unique_count": int(df[column].nunique(dropna=True)),
                "sample_values": " | ".join(df[column].dropna().astype(str).head(3).tolist()),
            }
        )
    return pd.DataFrame(rows)


def build_quality_report(df: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_rows = len(df)
    duplicate_count = int(df.duplicated().sum())
    invalid_year = int(df["year"].notna().sum() - df["year"].between(config.year_min, config.year_max).sum())
    invalid_month = int(df["month"].notna().sum() - df["month"].between(1, 12).sum())
    invalid_num_taxpayers = int((df["num_taxpayers"] < 0).fillna(False).sum())
    invalid_turnover = int((df["turnover_eur"] < 0).fillna(False).sum())
    summary = pd.DataFrame(
        [
            {
                "metric": "total_rows",
                "value": total_rows,
            }
            ,
            {"metric": "total_columns", "value": df.shape[1]},
            {"metric": "complete_rows", "value": int(df.notna().all(axis=1).sum())},
            {"metric": "rows_with_any_null", "value": int(df.isna().any(axis=1).sum())},
            {"metric": "duplicate_rows", "value": duplicate_count},
            {"metric": "invalid_year_rows", "value": invalid_year},
            {"metric": "invalid_month_rows", "value": invalid_month},
            {"metric": "invalid_num_taxpayers_rows", "value": invalid_num_taxpayers},
            {"metric": "invalid_turnover_rows", "value": invalid_turnover},
        ]
    )
    per_column = pd.DataFrame(
        [
            {
                "column": column,
                "null_count": int(df[column].isna().sum()),
                "null_rate_decimal": round(float(df[column].isna().mean()), 8),
                "null_rate_percent": f"{float(df[column].isna().mean()) * 100:.6f}%",
                "non_null_count": int(df[column].notna().sum()),
            }
            for column in df.columns
        ]
    )
    return summary, per_column


def build_readability_status(summary_df: pd.DataFrame, *, phase: str) -> str:
    metrics = dict(zip(summary_df["metric"], summary_df["value"]))
    if metrics.get("invalid_month_rows", 0) or metrics.get("invalid_turnover_rows", 0):
        if phase == "before":
            return "Initial dataset is readable, but it contains issues that should be cleaned before modeling."
        return "Cleaned dataset is usable after cleaning; blocked fields were not detected."
    if metrics.get("rows_with_any_null", 0):
        if phase == "before":
            return "Initial dataset is usable for profiling, but it still needs normalization and cleaning."
        return "Dataset is usable as-is for profiling and usable after cleaning for modeling."
    return "Dataset is usable as-is."


def build_null_strategy_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column_role": "numeric",
                "strategy": "preserve nulls in strict dataset, median-impute only in model-ready dataset",
            },
            {
                "column_role": "categorical",
                "strategy": "preserve nulls in strict dataset, fill with Unknown only in model-ready dataset",
            },
            {
                "column_role": "row_filtering",
                "strategy": "drop only rows with all critical business fields missing or repeated header rows",
            },
        ]
    )


def _build_quality_input(df: pd.DataFrame) -> pd.DataFrame:
    if set(BUSINESS_COLUMNS).issubset(df.columns):
        quality_df = df[[column for column in BUSINESS_COLUMNS if column in df.columns]].copy()
    else:
        renamed = df.rename(columns=RAW_TO_CANONICAL)
        quality_df = renamed[[column for column in BUSINESS_COLUMNS if column in renamed.columns]].copy()
    for column in NUMERIC_COLUMNS:
        if column in quality_df.columns:
            quality_df[column] = pd.to_numeric(quality_df[column], errors="coerce")
    return quality_df


def build_profile_bundle(
    df: pd.DataFrame,
    config: PipelineConfig,
    *,
    phase: str,
    restrict_schema_to_business_columns: bool,
) -> dict[str, pd.DataFrame | str]:
    if restrict_schema_to_business_columns:
        profile_df = df[[column for column in BUSINESS_COLUMNS if column in df.columns]].copy()
    else:
        profile_df = df.copy()
    quality_df = _build_quality_input(df)
    schema_report = build_schema_report(profile_df)
    quality_summary, nulls_by_column = build_quality_report(quality_df, config)
    null_strategy = build_null_strategy_summary()
    readability = build_readability_status(quality_summary, phase=phase)
    return {
        "schema_report": schema_report,
        "quality_summary": quality_summary,
        "nulls_by_column": nulls_by_column,
        "null_strategy": null_strategy,
        "readability_status": readability,
    }


def _export_profile_bundle(bundle: dict[str, pd.DataFrame | str], output_dir) -> dict[str, str]:
    schema_path = output_dir / "schema_report.csv"
    quality_path = output_dir / "data_quality_report.csv"
    nulls_path = output_dir / "nulls_by_column.csv"
    strategy_path = output_dir / "null_strategy.csv"
    write_dataframe(bundle["schema_report"], schema_path)
    write_dataframe(bundle["quality_summary"], quality_path)
    write_dataframe(bundle["nulls_by_column"], nulls_path)
    write_dataframe(bundle["null_strategy"], strategy_path)
    (output_dir / "readability_status.txt").write_text(bundle["readability_status"], encoding="utf-8")
    return {
        "schema_report": str(schema_path),
        "quality_report": str(quality_path),
        "nulls_by_column": str(nulls_path),
        "null_strategy": str(strategy_path),
        "readability_status": str(output_dir / "readability_status.txt"),
    }


def run_profile(config: PipelineConfig) -> dict[str, str]:
    from .pipeline import load_cleaned_data

    config.ensure_output_dirs()
    df = load_cleaned_data(config, model_ready=False)
    bundle = build_profile_bundle(
        df,
        config,
        phase="after",
        restrict_schema_to_business_columns=True,
    )
    outputs = _export_profile_bundle(bundle, config.profile_after_dir)
    _export_profile_bundle(bundle, config.profile_dir)
    return outputs


def run_profile_raw(config: PipelineConfig) -> dict[str, str]:
    from .ingest import read_raw_excel

    config.ensure_output_dirs()
    df = read_raw_excel(config)
    bundle = build_profile_bundle(
        df,
        config,
        phase="before",
        restrict_schema_to_business_columns=False,
    )
    return _export_profile_bundle(bundle, config.profile_before_dir)
