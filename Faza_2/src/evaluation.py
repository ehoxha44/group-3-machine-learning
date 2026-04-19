"""
evaluation.py – Step B: Metrics, visualisation, and cross-analysis.

Supervised metrics   : Accuracy, Precision, Recall, F1-Score, Cohen's Kappa
                       5-fold cross-validation (mean ± std) for both models
                       Head-to-head verdict table (winner per metric)
                       Confusion Matrix, ROC Curves (One-vs-Rest)
                       Feature Importance, CV score distribution boxplot

Unsupervised metrics : Inertia, Silhouette, Davies-Bouldin
                       Elbow / Silhouette / DB plots
                       PCA scatter (true growth labels vs K-Means clusters)
                       Growth trajectory heatmaps

Cross-analysis       : sector × municipality growth heatmap (any year range)
                       Municipality growth trajectories over time
                       Sector growth trajectories over time
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

matplotlib.use("Agg")

from pathlib import Path
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import label_binarize

from src.config import METRICS_DIR, PLOTS_DIR, RANDOM_STATE, TOP_MUNICIPALITIES, TOP_SECTORS


# ── Helpers ───────────────────────────────────────────────────────────────────


def _dirs() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> Path:
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → Plot saved: outputs/plots/{filename}")
    return path


# ── Supervised ────────────────────────────────────────────────────────────────


def evaluate_classifier(model, X_test, y_test, label_names, model_name) -> dict:
    _dirs()
    y_pred = model.predict(X_test)
    report_str = classification_report(y_test, y_pred, target_names=label_names)
    report_dict = classification_report(
        y_test, y_pred, target_names=label_names, output_dict=True
    )
    print(f"\n{'='*60}\n  {model_name} – Classification Report\n{'='*60}")
    print(report_str)
    pd.DataFrame(report_dict).T.to_csv(
        METRICS_DIR / f"{model_name}_classification_report.csv"
    )
    return {"y_pred": y_pred, "report": report_dict, "report_str": report_str}


def plot_confusion_matrix(model, X_test, y_test, label_names, model_name) -> None:
    _dirs()
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=label_names).plot(
        ax=ax, cmap="Blues", colorbar=True, values_format=".2f"
    )
    ax.set_title(f"{model_name} – Normalised Confusion Matrix\n(Growth Classification)")
    plt.tight_layout()
    _save(fig, f"{model_name}_confusion_matrix.png")


def plot_roc_curves(model, X_test, y_test, label_names, model_name) -> None:
    _dirs()
    if not hasattr(model, "predict_proba"):
        return
    n = len(label_names)
    y_bin = label_binarize(y_test, classes=list(range(n)))
    y_score = model.predict_proba(X_test)
    colors = cm.tab10(np.linspace(0, 1, n))
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (name, color) in enumerate(zip(label_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlim=[0, 1], ylim=[0, 1.05],
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title=f"{model_name} – ROC Curves (One-vs-Rest, Growth Classes)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save(fig, f"{model_name}_roc_curves.png")


def plot_feature_importance(model, feature_names, model_name, top_n=20) -> None:
    _dirs()
    if not hasattr(model, "feature_importances_"):
        return
    top_n = min(top_n, len(feature_names))
    idx = np.argsort(model.feature_importances_)[::-1][:top_n]
    vals = model.feature_importances_[idx]
    names = [feature_names[i] for i in idx]
    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.4)))
    bars = ax.barh(range(top_n), vals, color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set(xlabel="Importance", title=f"{model_name} – Feature Importances")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    plt.tight_layout()
    _save(fig, f"{model_name}_feature_importance.png")


def compare_models(results: dict) -> pd.DataFrame:
    _dirs()
    rows = []
    for name, res in results.items():
        r = res["report"]
        rows.append({
            "Model": name,
            "Accuracy": r["accuracy"],
            "Precision (macro)": r["macro avg"]["precision"],
            "Recall (macro)": r["macro avg"]["recall"],
            "F1-Score (macro)": r["macro avg"]["f1-score"],
        })
    df = pd.DataFrame(rows).set_index("Model")
    df.to_csv(METRICS_DIR / "model_comparison.csv")
    print("\n=== Model Comparison ===")
    print(df.to_string())

    metrics = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-Score (macro)"]
    x = np.arange(len(metrics))
    width = 0.8 / len(results)
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (name, _) in enumerate(results.items()):
        vals = [df.loc[name, m] for m in metrics]
        rects = ax.bar(x + i * width, vals, width, label=name)
        ax.bar_label(rects, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set(ylim=[0, 1.15], ylabel="Score",
           title="Model Comparison – Growth Class Prediction")
    ax.legend()
    plt.tight_layout()
    _save(fig, "model_comparison.png")
    return df


# ── Cross-validation & head-to-head verdict ───────────────────────────────────


def cross_validate_models(
    models: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> pd.DataFrame:
    """
    Run stratified k-fold cross-validation on every model and return a
    DataFrame with mean ± std for Accuracy, Precision, Recall, F1, and Kappa.

    A single 80/20 split can be misleading due to variance in the random state.
    Cross-validation across 5 stratified folds gives a robust estimate of
    each model's true generalisation performance on the growth dataset.

    Parameters
    ----------
    models : dict {model_name: fitted_or_unfitted_estimator}
    X      : full feature matrix (train + test combined)
    y      : full target vector

    Returns
    -------
    DataFrame indexed by model name with columns:
        accuracy_mean, accuracy_std, precision_mean, precision_std,
        recall_mean, recall_std, f1_mean, f1_std, kappa_mean, kappa_std
    """
    _dirs()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy":  "accuracy",
        "precision": "precision_macro",
        "recall":    "recall_macro",
        "f1":        "f1_macro",
    }

    rows = []
    cv_raw: dict[str, dict[str, list]] = {}   # store raw fold scores for boxplot

    for name, model in models.items():
        print(f"  Cross-validating {name} ({n_splits} folds) …")
        cv_scores = cross_validate(
            model, X, y, cv=cv, scoring=scoring,
            return_train_score=False, n_jobs=-1,
        )

        # Kappa requires predict, not a scorer string — compute manually per fold
        kappas = []
        for train_idx, val_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            y_pred = model.predict(X[val_idx])
            kappas.append(cohen_kappa_score(y[val_idx], y_pred))

        row = {"Model": name}
        raw_fold: dict[str, list] = {}
        for metric in ("accuracy", "precision", "recall", "f1"):
            vals = cv_scores[f"test_{metric}"]
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_std"]  = vals.std()
            raw_fold[metric] = list(vals)
        row["kappa_mean"] = np.mean(kappas)
        row["kappa_std"]  = np.std(kappas)
        raw_fold["kappa"] = kappas

        rows.append(row)
        cv_raw[name] = raw_fold
        print(
            f"    accuracy={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f}  "
            f"f1={row['f1_mean']:.3f}±{row['f1_std']:.3f}  "
            f"kappa={row['kappa_mean']:.3f}±{row['kappa_std']:.3f}"
        )

    df = pd.DataFrame(rows).set_index("Model")
    df.to_csv(METRICS_DIR / "cross_validation_results.csv")
    print(f"  → CV results saved: outputs/metrics/cross_validation_results.csv")

    # Boxplot of fold scores
    _plot_cv_boxplot(cv_raw, n_splits)
    return df


def _plot_cv_boxplot(cv_raw: dict, n_splits: int) -> None:
    """Box-and-whisker plot comparing fold scores for each metric across models."""
    metrics = ["accuracy", "precision", "recall", "f1", "kappa"]
    metric_labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)",
                     "F1-Score\n(macro)", "Cohen's\nKappa"]
    model_names = list(cv_raw.keys())
    colors = ["steelblue", "darkorange"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), sharey=False)
    for ax, metric, label in zip(axes, metrics, metric_labels):
        data = [cv_raw[m][metric] for m in model_names]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops={"color": "black", "lw": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(model_names, fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"{n_splits}-Fold Cross-Validation Score Distribution\n"
        "(box = IQR, line = median, whiskers = min/max)",
        fontsize=11,
    )
    plt.tight_layout()
    _save(fig, "cv_score_boxplot.png")


def build_verdict_table(
    hold_out_results: dict,
    cv_df: pd.DataFrame,
    y_test: np.ndarray,
    predictions: dict,
) -> pd.DataFrame:
    """
    Produce a head-to-head verdict table showing the winner for every metric.

    Combines hold-out test metrics, CV means, and Cohen's Kappa into one
    summary.  A ✓ marks the better model for each metric row.

    Parameters
    ----------
    hold_out_results : output of evaluate_classifier calls  {name: {report: ...}}
    cv_df            : DataFrame from cross_validate_models
    y_test           : true labels on the hold-out test set
    predictions      : {model_name: y_pred array} on the hold-out test set
    """
    _dirs()
    model_names = list(hold_out_results.keys())

    rows = []
    metrics_hold = {
        "Accuracy (hold-out)":         lambda n: hold_out_results[n]["report"]["accuracy"],
        "Precision macro (hold-out)":  lambda n: hold_out_results[n]["report"]["macro avg"]["precision"],
        "Recall macro (hold-out)":     lambda n: hold_out_results[n]["report"]["macro avg"]["recall"],
        "F1-Score macro (hold-out)":   lambda n: hold_out_results[n]["report"]["macro avg"]["f1-score"],
        "Cohen's Kappa (hold-out)":    lambda n: cohen_kappa_score(y_test, predictions[n]),
    }
    metrics_cv = {
        "Accuracy (CV mean)":    "accuracy_mean",
        "Precision macro (CV)":  "precision_mean",
        "Recall macro (CV)":     "recall_mean",
        "F1-Score macro (CV)":   "f1_mean",
        "Cohen's Kappa (CV)":    "kappa_mean",
    }

    for label, fn in metrics_hold.items():
        vals = {n: fn(n) for n in model_names}
        best = max(vals, key=vals.get)
        row = {"Metric": label}
        for n in model_names:
            row[n] = f"{vals[n]:.4f} {'✓' if n == best else ''}"
        rows.append(row)

    for label, col in metrics_cv.items():
        if col not in cv_df.columns:
            continue
        vals = {n: cv_df.loc[n, col] for n in model_names if n in cv_df.index}
        best = max(vals, key=vals.get)
        row = {"Metric": label}
        std_col = col.replace("_mean", "_std")
        for n in model_names:
            std = cv_df.loc[n, std_col] if std_col in cv_df.columns else 0
            row[n] = f"{vals[n]:.4f}±{std:.4f} {'✓' if n == best else ''}"
        rows.append(row)

    verdict_df = pd.DataFrame(rows).set_index("Metric")
    verdict_df.to_csv(METRICS_DIR / "algorithm_verdict.csv")

    # Count wins per model
    wins = {n: sum(1 for r in rows if "✓" in str(r.get(n, ""))) for n in model_names}
    winner = max(wins, key=wins.get)

    print("\n" + "=" * 60)
    print("  ALGORITHM VERDICT")
    print("=" * 60)
    print(verdict_df.to_string())
    print(f"\n  Wins: " + "  |  ".join(f"{n}: {w}" for n, w in wins.items()))
    print(f"  → Overall best algorithm: {winner}")
    print("=" * 60)

    return verdict_df


# ── Regression evaluation ─────────────────────────────────────────────────────


def evaluate_regressor(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """
    Compute and save regression metrics: MAE, RMSE, R², MAPE.

    MAE  – average absolute prediction error (in growth-rate units, e.g. 0.12 = 12 pp)
    RMSE – penalises large errors more than MAE; sensitive to outlier years
    R²   – proportion of variance explained (1.0 = perfect, 0 = predicts mean)
    MAPE – percentage error relative to actual; undefined when actual ≈ 0, so
           rows with |actual| < 0.01 are excluded from MAPE only
    """
    _dirs()
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    # MAPE: skip near-zero actuals to avoid division explosion
    mask = np.abs(y_test) > 0.01
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) if mask.any() else float("nan")

    metrics = {"MAE": mae, "RMSE": rmse, "R²": r2, "MAPE": mape}
    print(
        f"\n{'='*60}\n  {model_name} – Regression Metrics\n{'='*60}\n"
        f"  MAE  = {mae:.4f}  ({mae*100:.2f} pp average error)\n"
        f"  RMSE = {rmse:.4f}\n"
        f"  R²   = {r2:.4f}  ({r2*100:.1f}% variance explained)\n"
        f"  MAPE = {mape:.2%}"
    )

    pd.DataFrame([{"Model": model_name, **metrics}]).to_csv(
        METRICS_DIR / f"{model_name}_regression_metrics.csv", index=False
    )
    return {"y_pred": y_pred, "metrics": metrics}


def plot_actual_vs_predicted(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Scatter plot of actual vs predicted growth_rate with identity line."""
    _dirs()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, y_pred, alpha=0.35, s=14, color="steelblue", edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()) - 0.05,
            max(y_test.max(), y_pred.max()) + 0.05]
    ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
    ax.set(xlabel="Actual Growth Rate", ylabel="Predicted Growth Rate",
           title=f"{model_name} – Actual vs Predicted Growth Rate",
           xlim=lims, ylim=lims)
    ax.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax.axvline(0, color="gray", lw=0.8, linestyle=":")
    ax.legend()
    plt.tight_layout()
    _save(fig, f"{model_name}_actual_vs_predicted.png")


def plot_residuals(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> None:
    """Residual plot: predicted value vs residual (y_test - y_pred)."""
    _dirs()
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs fitted
    axes[0].scatter(y_pred, residuals, alpha=0.35, s=14, color="steelblue")
    axes[0].axhline(0, color="red", lw=1.5, linestyle="--")
    axes[0].set(xlabel="Predicted Growth Rate", ylabel="Residual",
                title=f"{model_name} – Residuals vs Fitted")
    axes[0].grid(True, alpha=0.3)

    # Residual histogram
    axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", lw=1.5, linestyle="--")
    axes[1].set(xlabel="Residual", ylabel="Count",
                title=f"{model_name} – Residual Distribution")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, f"{model_name}_residuals.png")


def compare_regressors(results: dict) -> pd.DataFrame:
    """
    Bar chart and CSV comparing MAE / RMSE / R² across all regression models.

    Lower MAE and RMSE = better.  Higher R² = better.
    """
    _dirs()
    rows = [{"Model": n, **r["metrics"]} for n, r in results.items()]
    df = pd.DataFrame(rows).set_index("Model")
    df.to_csv(METRICS_DIR / "regressor_comparison.csv")

    print("\n=== Regressor Comparison ===")
    print(df.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = cm.tab10(np.linspace(0, 1, len(df)))

    for ax, metric, lower_better in zip(
        axes, ["MAE", "RMSE", "R²"], [True, True, False]
    ):
        bars = ax.bar(df.index, df[metric], color=colors, alpha=0.85)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        best = df[metric].idxmin() if lower_better else df[metric].idxmax()
        ax.set_title(f"{metric}  ({'↓ lower' if lower_better else '↑ higher'} is better)")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, alpha=0.3, axis="y")
        # Highlight winner
        for bar, name in zip(bars, df.index):
            if name == best:
                bar.set_edgecolor("black")
                bar.set_linewidth(2.5)

    fig.suptitle("Regression Model Comparison – Growth Rate Prediction", fontsize=13)
    plt.tight_layout()
    _save(fig, "regressor_comparison.png")
    return df


# ── 2026 Predictions output ───────────────────────────────────────────────────


def save_predictions(
    pred_df: pd.DataFrame,
    cls_models: dict,
    reg_models: dict,
    X_pred: np.ndarray,
    encoders: dict,
    label_names: list[str],
    forecast_year: int,
) -> pd.DataFrame:
    """
    Run all models on the 2026 input matrix and produce a consolidated
    prediction table with one row per (municipality, sector).

    Columns produced
    ----------------
    municipality, primary_sector,
    prev_turnover_eur (2025 actual),
    growth_class_{model}         – GROWING / STABLE / DECLINING
    growth_rate_{model}          – predicted continuous growth rate
    est_turnover_{model}         – estimated EUR turnover in forecast_year
    """
    _dirs()
    out = pred_df[["municipality", "primary_sector", "total_turnover"]].copy()
    out = out.rename(columns={"total_turnover": "turnover_2025_eur"})
    out["forecast_year"] = forecast_year

    # Classification predictions
    for name, model in cls_models.items():
        raw = model.predict(X_pred)
        decoded = encoders["growth_class"].inverse_transform(raw)
        out[f"growth_class_{name}"] = decoded

    # Regression predictions + estimated turnover
    for name, model in reg_models.items():
        rate = model.predict(X_pred)
        out[f"growth_rate_{name}"] = rate
        out[f"est_turnover_{name}"] = out["turnover_2025_eur"] * (1 + rate)

    out = out.sort_values(["municipality", "primary_sector"]).reset_index(drop=True)
    out.to_csv(METRICS_DIR / f"predictions_{forecast_year}.csv", index=False)
    print(f"  → {forecast_year} predictions saved: outputs/metrics/predictions_{forecast_year}.csv")
    return out


def plot_predictions_heatmap(
    pred_df: pd.DataFrame,
    rate_col: str,
    forecast_year: int,
    top_municipalities: int = 20,
    top_sectors: int = 15,
) -> None:
    """Heatmap of predicted growth rate for forecast_year (sector × municipality)."""
    _dirs()
    top_mun = (
        pred_df.groupby("municipality")["turnover_2025_eur"].sum()
        .nlargest(top_municipalities).index
    )
    top_sec = (
        pred_df.groupby("primary_sector")["turnover_2025_eur"].sum()
        .nlargest(top_sectors).index
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
    model_label = rate_col.replace("growth_rate_", "")
    fig, ax = plt.subplots(
        figsize=(max(14, len(pivot.columns) * 0.85),
                 max(8, len(pivot) * 0.55))
    )
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax, label="Predicted YoY Growth Rate")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title(
        f"Predicted Growth Rate {forecast_year} [{model_label}]\n"
        f"(green = growing, red = declining)",
        fontsize=11,
    )
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.0%}",
                    ha="center", va="center", fontsize=5,
                    color="black" if abs(pivot.values[i, j]) < 0.35 else "white")
    plt.tight_layout()
    fname = f"predicted_growth_{forecast_year}_{model_label}.png"
    _save(fig, fname)


def plot_top_predictions(
    pred_df: pd.DataFrame,
    rate_col: str,
    forecast_year: int,
    top_n: int = 15,
) -> None:
    """
    Two bar charts: top-N growing municipalities and top-N growing sectors
    by predicted growth rate.
    """
    _dirs()
    model_label = rate_col.replace("growth_rate_", "")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, group_col, title in [
        (axes[0], "municipality", f"Top {top_n} Municipalities by Predicted Growth {forecast_year}"),
        (axes[1], "primary_sector", f"Top {top_n} Sectors by Predicted Growth {forecast_year}"),
    ]:
        grouped = (
            pred_df.groupby(group_col)[rate_col]
            .mean()
            .sort_values(ascending=False)
            .head(top_n)
        )
        colors = ["green" if v > 0 else "red" for v in grouped.values]
        bars = ax.barh(range(len(grouped)), grouped.values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(grouped)))
        ax.set_yticklabels(grouped.index, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.8)
        ax.set(xlabel="Predicted Growth Rate", title=title)
        ax.bar_label(bars, fmt="%.1%", padding=3, fontsize=7)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle(f"Market Predictions {forecast_year} [{model_label}]", fontsize=13)
    plt.tight_layout()
    _save(fig, f"predicted_top_{forecast_year}_{model_label}.png")


# ── Unsupervised ──────────────────────────────────────────────────────────────


def plot_elbow_silhouette(km_results: dict) -> None:
    _dirs()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(km_results["k"], km_results["inertia"], "bo-", lw=2, ms=6)
    axes[0].set(xlabel="k", ylabel="Inertia", title="K-Means Elbow Curve")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(km_results["k"], km_results["silhouette"], "rs-", lw=2, ms=6)
    axes[1].set(xlabel="k", ylabel="Silhouette Score (↑)", title="Silhouette Score vs k")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(km_results["k"], km_results["davies_bouldin"], "g^-", lw=2, ms=6)
    axes[2].set(xlabel="k", ylabel="Davies-Bouldin Index (↓)", title="Davies-Bouldin vs k")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle("K-Means Cluster Quality Metrics", fontsize=14, y=1.02)
    plt.tight_layout()
    _save(fig, "kmeans_elbow_silhouette.png")


def plot_pca_scatter(X_pca, labels, label_names, title, filename,
                     alpha=0.35, s=12) -> None:
    _dirs()
    unique = np.unique(labels)
    colors = cm.tab10(np.linspace(0, 1, len(unique)))
    fig, ax = plt.subplots(figsize=(11, 8))
    for cls, color in zip(unique, colors):
        mask = labels == cls
        name = label_names[int(cls)] if int(cls) < len(label_names) else f"Cluster {cls}"
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color],
                   alpha=alpha, s=s, label=name)
    ax.set(xlabel="PC1", ylabel="PC2", title=title)
    ax.legend(loc="best", fontsize=9, markerscale=3)
    plt.tight_layout()
    _save(fig, filename)


def plot_pca_variance(pca) -> None:
    _dirs()
    ratios = pca.explained_variance_ratio_
    n_show = min(30, len(ratios))
    cumvar = np.cumsum(ratios)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(1, n_show + 1), ratios[:n_show], alpha=0.6,
           color="steelblue", label="Individual")
    ax.step(range(1, n_show + 1), cumvar[:n_show], where="mid",
            color="red", lw=2, label="Cumulative")
    ax.axhline(0.95, color="green", linestyle="--", lw=1.5, label="95% threshold")
    ax.set(xlabel="PC index", ylabel="Explained Variance Ratio",
           title="PCA – Explained Variance", ylim=[0, 1.05])
    ax.legend()
    plt.tight_layout()
    _save(fig, "pca_variance_explained.png")


def save_unsupervised_metrics(km_results, km_model, silhouette,
                               davies_bouldin, pca_full) -> None:
    _dirs()
    pd.DataFrame(km_results).to_csv(METRICS_DIR / "kmeans_sweep_metrics.csv", index=False)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n95 = int(np.searchsorted(cumvar, 0.95)) + 1
    pd.DataFrame({
        "metric": ["kmeans_inertia", "kmeans_silhouette", "kmeans_davies_bouldin",
                   "pca_variance_2d", "pca_components_95pct"],
        "value": [km_model.inertia_, silhouette, davies_bouldin,
                  float(pca_full.explained_variance_ratio_[:2].sum()), n95],
    }).to_csv(METRICS_DIR / "unsupervised_metrics.csv", index=False)
    print("  → Unsupervised metrics saved")


# ── Growth cross-analysis (the core economic analysis) ────────────────────────


def plot_growth_heatmap(
    growth_df: pd.DataFrame,
    row_col: str,
    col_col: str = "year",
    value_col: str = "growth_rate",
    top_n_rows: int = 20,
    title: str = "",
    filename: str = "growth_heatmap.png",
) -> None:
    """
    Heatmap of mean growth_rate for (row_col × col_col).

    Example: row_col='municipality', col_col='year'
             → rows are cities, columns are years, colour is avg growth rate.
    This answers "which city grew fastest in which year?"
    """
    _dirs()
    top_groups = (
        growth_df.groupby(row_col)["total_turnover"].sum()
        .nlargest(top_n_rows).index
    )
    subset = growth_df[growth_df[row_col].isin(top_groups)]
    pivot = (
        subset.groupby([row_col, col_col])[value_col]
        .mean()
        .unstack(col_col)
        .fillna(0)
    )

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2),
                                    max(6, len(pivot) * 0.45)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                   vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax, label="Mean YoY Growth Rate")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set(title=title or f"Growth Rate: {row_col} × {col_col}",
           xlabel=col_col, ylabel=row_col)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.0%}",
                    ha="center", va="center", fontsize=6,
                    color="black" if abs(pivot.values[i, j]) < 0.3 else "white")
    plt.tight_layout()
    _save(fig, filename)
    # Save data
    pivot.to_csv(METRICS_DIR / filename.replace(".png", ".csv"))


def plot_sector_by_municipality_heatmap(
    growth_df: pd.DataFrame,
    top_municipalities: int = TOP_MUNICIPALITIES,
    top_sectors: int = TOP_SECTORS,
    year: int | None = None,
) -> None:
    """
    Heatmap: top sectors (rows) × top municipalities (columns).
    Colour = mean YoY growth rate.
    Optionally filter to a single year.

    This is the core cross-analysis: "how did each industry perform
    in each city?" — one cell = avg growth rate of that sector in that city.
    """
    _dirs()
    df = growth_df.copy()
    if year is not None:
        df = df[df["year"] == year]

    top_mun = (
        df.groupby("municipality")["total_turnover"].sum()
        .nlargest(top_municipalities).index
    )
    top_sec = (
        df.groupby("primary_sector")["total_turnover"].sum()
        .nlargest(top_sectors).index
    )
    subset = df[df["municipality"].isin(top_mun) & df["primary_sector"].isin(top_sec)]

    pivot = (
        subset.groupby(["primary_sector", "municipality"])["growth_rate"]
        .mean()
        .unstack("municipality")
        .reindex(columns=top_mun, fill_value=0)
        .fillna(0)
    )

    year_str = f" ({year})" if year else " (all years)"
    fig, ax = plt.subplots(figsize=(max(14, len(pivot.columns) * 0.8),
                                    max(8, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=0.5)
    plt.colorbar(im, ax=ax, label="Mean YoY Growth Rate")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title(f"Sector × Municipality Growth Rate{year_str}\n"
                 f"(green = growing, red = declining)", fontsize=11)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.0%}",
                    ha="center", va="center", fontsize=5,
                    color="black" if abs(pivot.values[i, j]) < 0.3 else "white")
    plt.tight_layout()
    suffix = f"_{year}" if year else "_all"
    fname = f"sector_x_municipality_growth{suffix}.png"
    _save(fig, fname)
    pivot.to_csv(METRICS_DIR / fname.replace(".png", ".csv"))


def plot_trajectory_clusters(
    pivot: pd.DataFrame,
    labels: np.ndarray,
    group_label: str = "municipality",
) -> None:
    """
    Line chart of growth trajectories coloured by cluster assignment.

    Each line = one municipality (or sector), x = year, y = growth_rate.
    Colour = K-Means cluster.  Shows which groups share similar growth trends.
    """
    _dirs()
    n_clusters = len(np.unique(labels))
    colors = cm.tab10(np.linspace(0, 1, n_clusters))
    cluster_colors = {i: colors[i] for i in range(n_clusters)}

    fig, ax = plt.subplots(figsize=(12, 7))
    for i, (group, row) in enumerate(pivot.iterrows()):
        c = cluster_colors[labels[i]]
        ax.plot(pivot.columns, row.values, color=c, alpha=0.55, lw=1.5)

    # Legend: one patch per cluster
    from matplotlib.patches import Patch
    handles = [
        Patch(color=cluster_colors[k], label=f"Cluster {k}")
        for k in range(n_clusters)
    ]
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set(xlabel="Year", ylabel="YoY Growth Rate",
           title=f"{group_label.capitalize()} Growth Trajectories by Cluster")
    ax.legend(handles=handles, loc="upper right")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )
    plt.tight_layout()
    _save(fig, f"{group_label}_trajectory_clusters.png")
