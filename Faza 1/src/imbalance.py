from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE

from .config import PipelineConfig
from .constants import CANDIDATE_TARGET_COLUMNS
from .io_utils import write_dataframe


def analyze_class_distribution(df: pd.DataFrame, candidate_columns: list[str] | None = None) -> pd.DataFrame:
    candidate_columns = candidate_columns or CANDIDATE_TARGET_COLUMNS
    rows: list[dict[str, object]] = []
    for column in candidate_columns:
        if column not in df.columns:
            continue
        counts = df[column].fillna("Missing").value_counts(dropna=False)
        total = int(counts.sum())
        for label, count in counts.items():
            share = float(count / total if total else 0)
            rows.append(
                {
                    "target_column": column,
                    "class_label": label,
                    "count": int(count),
                    "share_decimal": round(share, 8),
                    "share_percent": f"{share * 100:.4f}%",
                }
            )
    return pd.DataFrame(rows)


def validate_resampling_inputs(df: pd.DataFrame, target_column: str | None) -> None:
    if not target_column:
        raise ValueError("Target column is required for SMOTE/ADASYN.")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")
    feature_columns = [column for column in df.columns if column != target_column]
    numeric_feature_columns = df[feature_columns].select_dtypes(include="number").columns.tolist()
    if not numeric_feature_columns:
        raise ValueError("Model-ready feature matrix must contain numeric feature columns.")
    class_counts = df[target_column].value_counts(dropna=False)
    if len(class_counts) < 2:
        raise ValueError("At least two target classes are required.")
    if class_counts.min() < 2:
        raise ValueError("Each class needs at least two rows before resampling.")


def run_imbalance(config: PipelineConfig) -> dict[str, str]:
    from .pipeline import load_cleaned_data

    config.ensure_output_dirs()
    df = load_cleaned_data(config, model_ready=False)
    distribution = analyze_class_distribution(df)
    output_path = config.imbalance_dir / "class_distribution_summary.csv"
    write_dataframe(distribution, output_path)
    return {"class_distribution_summary": str(output_path)}


def run_resample(
    config: PipelineConfig, target_column: str, algorithm: str = "smote"
) -> dict[str, str]:
    from .pipeline import load_cleaned_data

    config.ensure_output_dirs()
    df = load_cleaned_data(config, model_ready=True)
    validate_resampling_inputs(df, target_column)

    target = df[target_column].copy()
    features = df.drop(columns=[target_column])
    numeric_features = features.select_dtypes(include="number").copy()
    if numeric_features.empty:
        raise ValueError("Numeric features are required for resampling.")
    class_counts = target.value_counts(dropna=False)
    min_class_size = int(class_counts.min())

    algorithm_lower = algorithm.lower()
    if algorithm_lower == "smote":
        k_neighbors = max(1, min(5, min_class_size - 1))
        sampler = SMOTE(random_state=config.random_seed, k_neighbors=k_neighbors)
    elif algorithm_lower == "adasyn":
        n_neighbors = max(1, min(5, min_class_size - 1))
        sampler = ADASYN(random_state=config.random_seed, n_neighbors=n_neighbors)
    else:
        raise ValueError("Algorithm must be either 'smote' or 'adasyn'.")

    resampled_features, resampled_target = sampler.fit_resample(numeric_features, target)
    result = resampled_features.copy()
    result[target_column] = resampled_target

    before = target.value_counts(dropna=False).rename_axis("class_label").reset_index(name="count")
    before["dataset"] = "before"
    after = resampled_target.value_counts(dropna=False).rename_axis("class_label").reset_index(name="count")
    after["dataset"] = "after"
    summary = pd.concat([before, after], ignore_index=True)
    summary["share_decimal"] = (
        summary.groupby("dataset")["count"].transform(lambda values: (values / values.sum()).round(8))
    )
    summary["share_percent"] = summary["share_decimal"].map(lambda value: f"{value * 100:.4f}%")

    result_path = config.imbalance_dir / f"resampled_{algorithm_lower}_{target_column}.csv"
    summary_path = config.imbalance_dir / f"resampled_{algorithm_lower}_{target_column}_summary.csv"
    write_dataframe(result, result_path)
    write_dataframe(summary, summary_path)
    return {"resampled_dataset": str(result_path), "resample_summary": str(summary_path)}
