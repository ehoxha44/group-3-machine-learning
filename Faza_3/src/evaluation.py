"""
evaluation.py - Metrics, plots and phase comparisons for Phase 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from src.config import (
    METRICS_DIR,
    PHASE2_METRICS_DIR,
    PLOTS_DIR,
    PRED_CLIP_LOWER,
    PRED_CLIP_UPPER,
    TOP_MUNICIPALITIES,
    TOP_SECTORS,
)

matplotlib.use("Agg")


def _dirs() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> Path:
    _dirs()
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Plot saved: outputs/plots/{filename}")
    return path


def evaluate_classifier(model, X_test, y_test, label_names, model_name) -> dict:
    _dirs()
    y_pred = model.predict(X_test)
    report_dict = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True
    )
    pd.DataFrame(report_dict).T.to_csv(
        METRICS_DIR / f"{model_name}_classification_report.csv"
    )
    kappa = cohen_kappa_score(y_test, y_pred)
    stable_recall = report_dict.get("STABLE", {}).get("recall", np.nan)
    print(
        f"{model_name}: accuracy={report_dict['accuracy']:.3f} "
        f"f1_macro={report_dict['macro avg']['f1-score']:.3f} "
        f"kappa={kappa:.3f} stable_recall={stable_recall:.3f}"
    )
    return {
        "y_pred": y_pred,
        "report": report_dict,
        "kappa": kappa,
        "stable_recall": stable_recall,
    }


def plot_confusion_matrix(model, X_test, y_test, label_names, model_name) -> None:
    cmatrix = confusion_matrix(y_test, model.predict(X_test), normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cmatrix, display_labels=label_names).plot(
        ax=ax, cmap="Blues", colorbar=True, values_format=".2f"
    )
    ax.set_title(f"{model_name} - Normalised Confusion Matrix")
    _save(fig, f"{model_name}_confusion_matrix.png")


def plot_roc_curves(model, X_test, y_test, label_names, model_name) -> None:
    if not hasattr(model, "predict_proba"):
        return
    y_score = model.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=list(range(len(label_names))))
    colors = cm.tab10(np.linspace(0, 1, len(label_names)))
    fig, ax = plt.subplots(figsize=(9, 7))
    for index, (label, color) in enumerate(zip(label_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, index], y_score[:, index])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"{model_name} - ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    _save(fig, f"{model_name}_roc_curves.png")


def plot_feature_importance(model, feature_names, model_name, top_n: int = 20) -> None:
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][: min(top_n, len(importances))]
    fig, ax = plt.subplots(figsize=(10, max(4, len(order) * 0.4)))
    values = importances[order]
    labels = [feature_names[index] for index in order]
    bars = ax.barh(range(len(order)), values, color="steelblue")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(f"{model_name} - Feature Importances")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    _save(fig, f"{model_name}_feature_importance.png")


def compare_models(results: dict) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        report = res["report"]
        rows.append(
            {
                "Model": name,
                "Accuracy": report["accuracy"],
                "Precision (macro)": report["macro avg"]["precision"],
                "Recall (macro)": report["macro avg"]["recall"],
                "F1-Score (macro)": report["macro avg"]["f1-score"],
                "Kappa": res["kappa"],
                "Stable Recall": res["stable_recall"],
            }
        )
    df = pd.DataFrame(rows).set_index("Model")
    df.to_csv(METRICS_DIR / "model_comparison.csv")

    metrics = ["Accuracy", "F1-Score (macro)", "Kappa", "Stable Recall"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, model_name in enumerate(df.index):
        values = [df.loc[model_name, metric] for metric in metrics]
        bars = ax.bar(x + idx * width, values, width, label=model_name)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.set_title("Phase 3 Classification Comparison")
    ax.legend()
    _save(fig, "model_comparison.png")
    return df


def evaluate_regressor(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = r2_score(y_test, y_pred)
    mask = np.abs(y_test) > 0.01
    mape = (
        np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask]))
        if mask.any()
        else float("nan")
    )
    metrics = {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE": mape}
    pd.DataFrame([{"Model": model_name, **metrics}]).to_csv(
        METRICS_DIR / f"{model_name}_regression_metrics.csv",
        index=False,
    )
    print(f"{model_name}: R²={r2:.3f} RMSE={rmse:.3f}")
    return {"y_pred": y_pred, "metrics": metrics}


def plot_actual_vs_predicted(y_test, y_pred, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, y_pred, alpha=0.35, s=12, color="steelblue")
    bounds = [
        min(float(np.min(y_test)), float(np.min(y_pred))) - 0.05,
        max(float(np.max(y_test)), float(np.max(y_pred))) + 0.05,
    ]
    ax.plot(bounds, bounds, "r--", lw=1.5)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_title(f"{model_name} - Actual vs Predicted Growth Rate")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    _save(fig, f"{model_name}_actual_vs_predicted.png")


def plot_residuals(y_test, y_pred, model_name: str) -> None:
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.35, s=12, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_title(f"{model_name} - Residuals vs Fitted")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")
    axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title(f"{model_name} - Residual Distribution")
    _save(fig, f"{model_name}_residuals.png")


def compare_regressors(results: dict) -> pd.DataFrame:
    rows = [{"Model": name, **res["metrics"]} for name, res in results.items()]
    df = pd.DataFrame(rows).set_index("Model")
    df.to_csv(METRICS_DIR / "regressor_comparison.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, metric, lower_better in zip(
        axes, ["MAE", "RMSE", "R²"], [True, True, False]
    ):
        bars = axis.bar(df.index, df[metric], color=cm.tab10(np.linspace(0, 1, len(df))))
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        axis.set_title(f"{metric} ({'lower' if lower_better else 'higher'} is better)")
        axis.tick_params(axis="x", rotation=15)
    _save(fig, "regressor_comparison.png")
    return df


def save_predictions(
    pred_df: pd.DataFrame,
    cls_models: dict,
    reg_models: dict,
    X_pred: np.ndarray,
    label_encoder,
    forecast_year: int,
) -> pd.DataFrame:
    out = pred_df[["municipality", "primary_sector", "month", "turnover_reference_eur"]].copy()
    out = out.rename(columns={"turnover_reference_eur": "turnover_reference_eur"})
    out["forecast_year"] = forecast_year

    for name, model in cls_models.items():
        raw = model.predict(X_pred)
        out[f"growth_class_{name}"] = label_encoder.inverse_transform(raw)

    for name, model in reg_models.items():
        rate = np.clip(model.predict(X_pred), PRED_CLIP_LOWER, PRED_CLIP_UPPER)
        out[f"growth_rate_{name}"] = rate
        out[f"est_turnover_{name}"] = out["turnover_reference_eur"] * (1.0 + rate)

    out = out.sort_values(["municipality", "primary_sector", "month"]).reset_index(drop=True)
    out.to_csv(METRICS_DIR / f"predictions_{forecast_year}.csv", index=False)
    print(f"  -> Saved: outputs/metrics/predictions_{forecast_year}.csv")
    return out


def plot_predictions_heatmap(
    pred_df: pd.DataFrame,
    rate_col: str,
    forecast_year: int,
    top_municipalities: int = TOP_MUNICIPALITIES,
    top_sectors: int = TOP_SECTORS,
) -> None:
    top_mun = (
        pred_df.groupby("municipality")["turnover_reference_eur"].sum()
        .nlargest(top_municipalities)
        .index
    )
    top_sec = (
        pred_df.groupby("primary_sector")["turnover_reference_eur"].sum()
        .nlargest(top_sectors)
        .index
    )
    subset = pred_df[
        pred_df["municipality"].isin(top_mun) &
        pred_df["primary_sector"].isin(top_sec)
    ]
    pivot = (
        subset.groupby(["primary_sector", "municipality"])[rate_col]
        .mean()
        .unstack("municipality")
        .reindex(columns=top_mun, fill_value=0)
        .fillna(0)
    )
    fig, ax = plt.subplots(
        figsize=(max(14, len(pivot.columns) * 0.85), max(8, len(pivot) * 0.55))
    )
    image = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=2.0)
    plt.colorbar(image, ax=ax, label="Predicted YoY Growth Rate")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title(f"Predicted Growth Rate {forecast_year} [{rate_col.replace('growth_rate_', '')}]")
    _save(fig, f"predicted_growth_{forecast_year}_{rate_col.replace('growth_rate_', '')}.png")


def plot_top_predictions(pred_df: pd.DataFrame, rate_col: str, forecast_year: int, top_n: int = 15) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for axis, column, title in [
        (axes[0], "municipality", f"Top {top_n} Municipalities"),
        (axes[1], "primary_sector", f"Top {top_n} Sectors"),
    ]:
        grouped = (
            pred_df.groupby(column)[rate_col]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
        )
        bars = axis.barh(range(len(grouped)), grouped.values, color="seagreen")
        axis.set_yticks(range(len(grouped)))
        axis.set_yticklabels(grouped.index, fontsize=8)
        axis.invert_yaxis()
        axis.set_title(f"{title} by Predicted Growth {forecast_year}")
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    _save(fig, f"predicted_top_{forecast_year}_{rate_col.replace('growth_rate_', '')}.png")


def plot_growth_heatmap(
    growth_df: pd.DataFrame,
    row_col: str,
    col_col: str = "year",
    value_col: str = "growth_rate",
    top_n_rows: int = 20,
    title: str = "",
    filename: str = "growth_heatmap.png",
) -> None:
    top_groups = (
        growth_df.groupby(row_col)["total_turnover"].sum()
        .nlargest(top_n_rows)
        .index
    )
    subset = growth_df[growth_df[row_col].isin(top_groups)]
    pivot = (
        subset.groupby([row_col, col_col])[value_col]
        .mean()
        .unstack(col_col)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2), max(6, len(pivot) * 0.45)))
    image = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    plt.colorbar(image, ax=ax, label="Mean YoY Growth Rate")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(title or f"Growth Rate: {row_col} x {col_col}")
    _save(fig, filename)
    pivot.to_csv(METRICS_DIR / filename.replace(".png", ".csv"))


def plot_sector_by_municipality_heatmap(
    growth_df: pd.DataFrame,
    top_municipalities: int = TOP_MUNICIPALITIES,
    top_sectors: int = TOP_SECTORS,
    year: int | None = None,
) -> None:
    subset = growth_df.copy()
    if year is not None:
        subset = subset[subset["year"] == year]
    top_mun = (
        subset.groupby("municipality")["total_turnover"]
        .sum()
        .nlargest(top_municipalities)
        .index
    )
    top_sec = (
        subset.groupby("primary_sector")["total_turnover"]
        .sum()
        .nlargest(top_sectors)
        .index
    )
    subset = subset[
        subset["municipality"].isin(top_mun) &
        subset["primary_sector"].isin(top_sec)
    ]
    pivot = (
        subset.groupby(["primary_sector", "municipality"])["growth_rate"]
        .mean()
        .unstack("municipality")
        .reindex(columns=top_mun, fill_value=0)
        .fillna(0)
    )
    suffix = "all" if year is None else str(float(year))
    fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 0.85), max(8, len(pivot) * 0.55)))
    image = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    plt.colorbar(image, ax=ax, label="Mean YoY Growth Rate")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title(f"Sector x Municipality Growth ({suffix})")
    _save(fig, f"sector_x_municipality_growth_{suffix}.png")
    pivot.to_csv(METRICS_DIR / f"sector_x_municipality_growth_{suffix}.csv")


def plot_learning_curve_xgb(model, filename: str = "learning_curve_xgb.png") -> None:
    if not hasattr(model, "evals_result"):
        return
    evals_result = model.evals_result()
    if not evals_result:
        return
    first_key = next(iter(evals_result))
    metric_name = next(iter(evals_result[first_key]))
    values = evals_result[first_key][metric_name]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(values, color="darkorange", lw=2)
    ax.set_title(f"XGBoost Validation Curve ({metric_name})")
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel(metric_name)
    _save(fig, filename)


def collect_phase3_metrics(classification_df: pd.DataFrame, regression_df: pd.DataFrame) -> dict:
    return {
        "RandomForest": {
            "Accuracy": float(classification_df.loc["RandomForest", "Accuracy"]),
            "F1-Score (macro)": float(classification_df.loc["RandomForest", "F1-Score (macro)"]),
            "Kappa": float(classification_df.loc["RandomForest", "Kappa"]),
            "Stable Recall": float(classification_df.loc["RandomForest", "Stable Recall"]),
        },
        "XGBoost": {
            "Accuracy": float(classification_df.loc["XGBoost", "Accuracy"]),
            "F1-Score (macro)": float(classification_df.loc["XGBoost", "F1-Score (macro)"]),
            "Kappa": float(classification_df.loc["XGBoost", "Kappa"]),
            "Stable Recall": float(classification_df.loc["XGBoost", "Stable Recall"]),
        },
        "RandomForestRegressor": {
            "R²": float(regression_df.loc["RandomForestRegressor", "R²"]),
        },
        "XGBoostRegressor": {
            "R²": float(regression_df.loc["XGBoostRegressor", "R²"]),
        },
    }


def _load_phase2_metrics(phase2_metrics_dir: Path = PHASE2_METRICS_DIR) -> dict:
    model_comparison = pd.read_csv(phase2_metrics_dir / "model_comparison.csv", index_col=0)
    rf_report = pd.read_csv(phase2_metrics_dir / "RandomForest_classification_report.csv", index_col=0)
    xgb_report = pd.read_csv(phase2_metrics_dir / "XGBoost_classification_report.csv", index_col=0)
    reg_comp = pd.read_csv(phase2_metrics_dir / "regressor_comparison.csv", index_col=0)
    verdict = pd.read_csv(phase2_metrics_dir / "algorithm_verdict.csv", index_col=0)

    def parse_value(text: str) -> float:
        return float(str(text).split()[0].split("±")[0])

    return {
        "RandomForest": {
            "Accuracy": float(model_comparison.loc["RandomForest", "Accuracy"]),
            "F1-Score (macro)": float(model_comparison.loc["RandomForest", "F1-Score (macro)"]),
            "Kappa": parse_value(verdict.loc["Cohen's Kappa (hold-out)", "RandomForest"]),
            "Stable Recall": float(rf_report.loc["STABLE", "recall"]),
        },
        "XGBoost": {
            "Accuracy": float(model_comparison.loc["XGBoost", "Accuracy"]),
            "F1-Score (macro)": float(model_comparison.loc["XGBoost", "F1-Score (macro)"]),
            "Kappa": parse_value(verdict.loc["Cohen's Kappa (hold-out)", "XGBoost"]),
            "Stable Recall": float(xgb_report.loc["STABLE", "recall"]),
        },
        "RandomForestRegressor": {
            "R²": float(reg_comp.loc["RandomForestRegressor", "R²"]),
        },
        "XGBoostRegressor": {
            "R²": float(reg_comp.loc["XGBoostRegressor", "R²"]),
        },
    }


def compare_phase2_phase3(
    phase2_metrics_dir: Path,
    phase3_results: dict,
    out_csv: Path,
    out_plot: Path,
) -> pd.DataFrame:
    phase2_results = _load_phase2_metrics(phase2_metrics_dir)
    rows = []
    for algorithm, metrics in phase3_results.items():
        for metric, phase3_value in metrics.items():
            phase2_value = phase2_results.get(algorithm, {}).get(metric, np.nan)
            delta = phase3_value - phase2_value if pd.notna(phase2_value) else np.nan
            rows.append(
                {
                    "algorithm": algorithm,
                    "metric": metric,
                    "phase2": phase2_value,
                    "phase3": phase3_value,
                    "delta": delta,
                    "improved": bool(delta > 0) if pd.notna(delta) else False,
                }
            )
    df = pd.DataFrame(rows).sort_values(["algorithm", "metric"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    plot_phase_comparison(df, out_plot)
    return df


def plot_phase_comparison(df: pd.DataFrame, out_path: Path) -> None:
    metrics = df["metric"].unique().tolist()
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, max(4, len(metrics) * 3.2)))
    if len(metrics) == 1:
        axes = [axes]
    for axis, metric in zip(axes, metrics):
        subset = df[df["metric"] == metric]
        algorithms = subset["algorithm"].unique().tolist()
        x = np.arange(len(algorithms))
        phase2_vals = [subset[subset["algorithm"] == algo]["phase2"].iloc[0] for algo in algorithms]
        phase3_vals = [subset[subset["algorithm"] == algo]["phase3"].iloc[0] for algo in algorithms]
        width = 0.35
        bars1 = axis.bar(x - width / 2, phase2_vals, width, label="Phase 2")
        bars2 = axis.bar(x + width / 2, phase3_vals, width, label="Phase 3")
        axis.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
        axis.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)
        axis.set_xticks(x)
        axis.set_xticklabels(algorithms, rotation=0)
        axis.set_title(metric)
        axis.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
