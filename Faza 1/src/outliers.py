from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .io_utils import write_dataframe


def detect_outliers(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    result = df.copy()
    for column in ("turnover_eur", "num_taxpayers"):
        q1 = result[column].quantile(0.25)
        q3 = result[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - config.outlier_iqr_multiplier * iqr
        upper = q3 + config.outlier_iqr_multiplier * iqr
        result[f"{column}_outlier_iqr"] = result[column].between(lower, upper, inclusive="both").eq(False)

        std = result[column].std()
        if std and not np.isnan(std):
            zscore = (result[column] - result[column].mean()) / std
            result[f"{column}_outlier_zscore"] = zscore.abs() > config.outlier_zscore_threshold
        else:
            result[f"{column}_outlier_zscore"] = False

    positive_turnover = result["turnover_eur"].where(result["turnover_eur"] > 0)
    log_turnover = np.log1p(positive_turnover.fillna(0))
    q1_log = log_turnover.quantile(0.25)
    q3_log = log_turnover.quantile(0.75)
    iqr_log = q3_log - q1_log
    lower_log = q1_log - config.outlier_iqr_multiplier * iqr_log
    upper_log = q3_log + config.outlier_iqr_multiplier * iqr_log
    result["turnover_eur_outlier_log_iqr"] = log_turnover.between(
        lower_log, upper_log, inclusive="both"
    ).eq(False)

    result["is_any_outlier"] = result.filter(like="_outlier_").any(axis=1)
    return result


def build_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    outlier_columns = [column for column in df.columns if "_outlier_" in column]
    return pd.DataFrame(
        [{"metric": column, "count": int(df[column].sum())} for column in outlier_columns]
    )


def run_outliers(config: PipelineConfig) -> dict[str, str]:
    from .pipeline import load_cleaned_data

    config.ensure_output_dirs()
    df = load_cleaned_data(config, model_ready=False)
    flagged = detect_outliers(df, config)
    flagged_path = config.outliers_dir / "outlier_flags.csv"
    summary_path = config.outliers_dir / "outlier_summary.csv"
    write_dataframe(flagged, flagged_path)
    write_dataframe(build_outlier_summary(flagged), summary_path)
    return {"outlier_flags": str(flagged_path), "outlier_summary": str(summary_path)}
