from __future__ import annotations

import os
from pathlib import Path

_CACHE_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "report" / ".cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .aggregation import build_aggregations
from .config import PipelineConfig
from .imbalance import analyze_class_distribution
from .io_utils import write_dataframe, write_text
from .outliers import build_outlier_summary, detect_outliers
from .profiling import build_profile_bundle


def _frame_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    headers = [str(column) for column in df.columns]
    separator = ["---"] * len(headers)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(separator) + " |"]
    for _, record in df.iterrows():
        rows.append("| " + " | ".join(str(record[column]) for column in df.columns) + " |")
    return "\n".join(rows)


def _format_human_number(value: float, axis_kind: str) -> str:
    abs_value = abs(value)
    if axis_kind == "currency":
        if abs_value >= 1_000_000_000:
            return f"{value / 1_000_000_000:.1f}B EUR"
        if abs_value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M EUR"
        if abs_value >= 1_000:
            return f"{value / 1_000:.1f}K EUR"
        return f"{value:.0f} EUR"
    if axis_kind == "percent":
        return f"{value * 100:.1f}%"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def _apply_axis_format(ax, axis: str, axis_kind: str) -> None:
    formatter = FuncFormatter(lambda value, _: _format_human_number(value, axis_kind))
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def _write_chart(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    plt.figure(figsize=(12, 6))
    chart_df = df.head(20)
    ax = plt.gca()
    ax.bar(chart_df[x].astype(str), chart_df[y])
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    axis_kind = "percent" if "share" in y or "percent" in y else "currency" if "turnover" in y else "count"
    _apply_axis_format(ax, "y", axis_kind)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def _write_horizontal_chart(df: pd.DataFrame, x: str, y: str, title: str, path: Path) -> None:
    plt.figure(figsize=(12, 8))
    chart_df = df.head(15).iloc[::-1]
    ax = plt.gca()
    ax.barh(chart_df[y].astype(str), chart_df[x])
    ax.set_title(title)
    axis_kind = "percent" if "share" in x or "percent" in x else "currency" if "turnover" in x else "count"
    _apply_axis_format(ax, "x", axis_kind)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def _write_smote_comparison_charts(summary_df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for dataset_label in ("before", "after"):
        subset = (
            summary_df.loc[summary_df["dataset"] == dataset_label]
            .sort_values("count", ascending=False)
            .head(15)
        )
        if subset.empty:
            continue
        count_path = output_dir / f"smote_{dataset_label}_top15_counts.png"
        percent_path = output_dir / f"smote_{dataset_label}_top15_percent.png"
        _write_horizontal_chart(
            subset,
            "count",
            "class_label",
            f"SMOTE {dataset_label.title()} Top 15 Classes by Count",
            count_path,
        )
        _write_horizontal_chart(
            subset.sort_values("share_decimal", ascending=False),
            "share_decimal",
            "class_label",
            f"SMOTE {dataset_label.title()} Top 15 Classes by Share",
            percent_path,
        )
        outputs[f"smote_{dataset_label}_counts"] = str(count_path)
        outputs[f"smote_{dataset_label}_percent"] = str(percent_path)
    return outputs


def generate_report(config: PipelineConfig) -> dict[str, str]:
    from .pipeline import load_cleaned_data

    config.ensure_output_dirs()
    strict_df = load_cleaned_data(config, model_ready=False)
    profile_bundle = build_profile_bundle(
        strict_df,
        config,
        phase="after",
        restrict_schema_to_business_columns=True,
    )
    aggregations = build_aggregations(strict_df)
    outlier_df = detect_outliers(strict_df, config)
    outlier_summary = build_outlier_summary(outlier_df)
    class_distribution = analyze_class_distribution(strict_df)

    chart_year_path = config.report_dir / "turnover_by_year.png"
    chart_month_path = config.report_dir / "turnover_by_month.png"
    chart_municipality_path = config.report_dir / "turnover_by_municipality_top20.png"
    chart_sector_path = config.report_dir / "turnover_by_sector_top15.png"
    chart_nulls_path = config.report_dir / "nulls_by_column.png"
    chart_outlier_path = config.report_dir / "outlier_summary.png"
    chart_registration_path = config.report_dir / "registration_status_distribution_top15.png"
    _write_chart(aggregations["turnover_by_year"], "year", "turnover_sum", "Turnover by Year", chart_year_path)
    _write_chart(
        aggregations["turnover_by_month"].sort_values("month"),
        "month",
        "turnover_sum",
        "Turnover by Month",
        chart_month_path,
    )
    _write_chart(
        aggregations["turnover_by_municipality"],
        "municipality",
        "turnover_sum",
        "Top Municipalities by Turnover",
        chart_municipality_path,
    )
    _write_horizontal_chart(
        aggregations["turnover_by_sector"].sort_values("turnover_sum", ascending=False),
        "turnover_sum",
        "primary_sector",
        "Top Sectors by Turnover",
        chart_sector_path,
    )
    _write_chart(
        profile_bundle["nulls_by_column"].sort_values("null_count", ascending=False),
        "column",
        "null_count",
        "Null Counts by Column",
        chart_nulls_path,
    )
    _write_chart(
        outlier_summary.sort_values("count", ascending=False),
        "metric",
        "count",
        "Outlier Flags Summary",
        chart_outlier_path,
    )
    registration_distribution = class_distribution.loc[
        class_distribution["target_column"] == "registration_status"
    ].sort_values("count", ascending=False)
    _write_horizontal_chart(
        registration_distribution,
        "count",
        "class_label",
        "Registration Status Distribution",
        chart_registration_path,
    )

    smote_summary_path = config.imbalance_dir / "resampled_smote_registration_status_summary.csv"
    smote_chart_outputs: dict[str, str] = {}
    if smote_summary_path.exists():
        smote_summary_df = pd.read_csv(smote_summary_path)
        smote_chart_outputs = _write_smote_comparison_charts(smote_summary_df, config.report_dir)

    report_lines = [
        "# Qarkullimi Data Cleaning Report",
        "",
        "## Data Types",
        _frame_to_markdown(profile_bundle["schema_report"]),
        "",
        "## Quality Status",
        _frame_to_markdown(profile_bundle["quality_summary"]),
        "",
        "Readable overall state:",
        profile_bundle["readability_status"],
        "",
        "## Complete vs Null Data",
        _frame_to_markdown(profile_bundle["nulls_by_column"]),
        "",
        "## Missing Value Strategy",
        _frame_to_markdown(profile_bundle["null_strategy"]),
        "",
        "## Outlier Findings",
        _frame_to_markdown(outlier_summary),
        "",
        "## Class Distribution Summary",
        _frame_to_markdown(class_distribution),
        "",
        "## Readiness for Modeling",
        "The strict cleaned dataset preserves nulls and flags invalid rows. The model-ready dataset imputes numeric columns with medians, fills categorical columns with Unknown, and adds imputation flags plus derived features such as year_month and turnover_eur_log1p.",
        "",
        "## Generated Graphs",
        "- turnover_by_year.png",
        "- turnover_by_month.png",
        "- turnover_by_municipality_top20.png",
        "- turnover_by_sector_top15.png",
        "- nulls_by_column.png",
        "- outlier_summary.png",
        "- registration_status_distribution_top15.png",
    ]
    if smote_chart_outputs:
        report_lines.extend(
            [
                "- smote_before_top15_counts.png",
                "- smote_before_top15_percent.png",
                "- smote_after_top15_counts.png",
                "- smote_after_top15_percent.png",
            ]
        )
    report_path = config.report_dir / "report.md"
    write_text("\n".join(report_lines), report_path)

    write_dataframe(class_distribution, config.report_dir / "class_distribution_summary.csv")
    write_dataframe(outlier_summary, config.report_dir / "outlier_summary.csv")
    for name, table in aggregations.items():
        write_dataframe(table, config.report_dir / f"{name}.csv")

    outputs = {
        "report_markdown": str(report_path),
        "chart_turnover_by_year": str(chart_year_path),
        "chart_turnover_by_month": str(chart_month_path),
        "chart_turnover_by_municipality": str(chart_municipality_path),
        "chart_turnover_by_sector": str(chart_sector_path),
        "chart_nulls_by_column": str(chart_nulls_path),
        "chart_outlier_summary": str(chart_outlier_path),
        "chart_registration_status_distribution": str(chart_registration_path),
    }
    outputs.update(smote_chart_outputs)
    return outputs
