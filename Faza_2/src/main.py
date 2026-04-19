#!/usr/bin/env python3
"""
main.py – Phase 2 entry point: Economic Growth Analysis & Market Predictions.

Algorithms used (7 total)
--------------------------
Supervised – Classification:
  1. Random Forest Classifier
  2. XGBoost Classifier

Supervised – Regression:
  3. Linear Regression       (baseline / interpretable)
  4. Random Forest Regressor
  5. XGBoost Regressor       (primary for 2026 forecast)

Unsupervised:
  6. K-Means Clustering      (trajectory clustering of municipalities & sectors)
  7. PCA                     (dimensionality reduction & visualisation)

Steps
-----
  supervised   – train & evaluate classifiers (hold-out + 5-fold CV + verdict)
  regression   – train & evaluate regressors  (MAE, RMSE, R²)
  unsupervised – K-Means sweep + trajectory clustering + PCA
  analysis     – growth heatmaps (sector × municipality, per year)
  predictions  – 2026 market forecasts using all 5 supervised models
  all          – run every step above

Usage
-----
    python -m src.main --step all
    python -m src.main --step predictions
    python -m src.main --step regression
"""

import sys
import argparse
import numpy as np
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.config import (
    FORECAST_YEAR, KMEANS_K, N_CLUSTERS_RANGE, OUTPUTS_DIR,
    RANDOM_STATE, REGRESSION_TARGET, TARGET_COLUMN,
    TOP_MUNICIPALITIES, TOP_SECTORS,
)
from src.data_loader import (
    build_growth_dataset, build_growth_pivot,
    build_prediction_input, load_data,
)
from src.evaluation import (
    build_verdict_table,
    compare_models,
    compare_regressors,
    cross_validate_models,
    evaluate_classifier,
    evaluate_regressor,
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_elbow_silhouette,
    plot_feature_importance,
    plot_growth_heatmap,
    plot_pca_scatter,
    plot_pca_variance,
    plot_predictions_heatmap,
    plot_residuals,
    plot_roc_curves,
    plot_sector_by_municipality_heatmap,
    plot_top_predictions,
    plot_trajectory_clusters,
    save_predictions,
    save_unsupervised_metrics,
)
from src.feature_engineering import encode_and_scale, encode_for_prediction
from src.supervised import (
    XGB_AVAILABLE,
    save_model,
    split_data,
    train_linear_regression,
    train_random_forest,
    train_rf_regressor,
    train_xgb_regressor,
    train_xgboost,
)
from src.unsupervised import (
    apply_pca, apply_pca_full, cluster_growth_trajectories,
    find_optimal_k, train_kmeans,
)

XGB_NAME = "XGBoost" if XGB_AVAILABLE else "GradientBoosting"


# ── Step A: Supervised classification ────────────────────────────────────────


def run_supervised(X_scaled, y, feature_cols, encoders) -> tuple:
    label_names = list(encoders[TARGET_COLUMN].classes_)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y, stratify=True)
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    rf  = train_random_forest(X_train, y_train)
    xgb = train_xgboost(X_train, y_train)

    save_model(rf,  "random_forest")
    save_model(xgb, XGB_NAME.lower())

    hold_out_results, predictions = {}, {}
    for model, name in [(rf, "RandomForest"), (xgb, XGB_NAME)]:
        res = evaluate_classifier(model, X_test, y_test, label_names, name)
        plot_confusion_matrix(model, X_test, y_test, label_names, name)
        plot_roc_curves(model, X_test, y_test, label_names, name)
        plot_feature_importance(model, feature_cols, name)
        hold_out_results[name] = res
        predictions[name] = res["y_pred"]

    compare_models(hold_out_results)

    print("\n--- 5-Fold Cross-Validation ---")
    cv_models = {
        "RandomForest": rf.__class__(**rf.get_params()),
        XGB_NAME: xgb.__class__(**xgb.get_params()),
    }
    cv_df = cross_validate_models(cv_models, X_scaled, y)

    print("\n--- Algorithm Verdict ---")
    build_verdict_table(hold_out_results, cv_df, y_test, predictions)

    return rf, xgb, X_train, X_test, y_train, y_test


# ── Step B: Supervised regression ────────────────────────────────────────────


def run_regression(X_scaled, growth_df, feature_cols) -> tuple:
    """Train Linear Regression, RF Regressor, XGB Regressor on growth_rate."""
    y_reg = growth_df[REGRESSION_TARGET].values
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_reg, stratify=False)
    print(f"Regression — Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    lr   = train_linear_regression(X_train, y_train)
    rfr  = train_rf_regressor(X_train, y_train)
    xgbr = train_xgb_regressor(X_train, y_train)

    save_model(lr,   "linear_regression")
    save_model(rfr,  "rf_regressor")
    save_model(xgbr, "xgb_regressor")

    reg_results = {}
    for model, name in [
        (lr,   "LinearRegression"),
        (rfr,  "RandomForestRegressor"),
        (xgbr, f"{XGB_NAME}Regressor"),
    ]:
        res = evaluate_regressor(model, X_test, y_test, name)
        plot_actual_vs_predicted(y_test, res["y_pred"], name)
        plot_residuals(y_test, res["y_pred"], name)
        plot_feature_importance(model, feature_cols, name)
        reg_results[name] = res

    compare_regressors(reg_results)
    return lr, rfr, xgbr


# ── Step C: Unsupervised ──────────────────────────────────────────────────────


def run_unsupervised(X_scaled, y, encoders, growth_df) -> None:
    label_names = list(encoders[TARGET_COLUMN].classes_)

    print("\n--- K-Means sweep ---")
    km_results = find_optimal_k(X_scaled, k_range=N_CLUSTERS_RANGE)
    plot_elbow_silhouette(km_results)

    km = train_kmeans(X_scaled, k=KMEANS_K)
    km_labels = km.labels_
    sil = silhouette_score(X_scaled, km_labels,
                           sample_size=min(5_000, len(X_scaled)),
                           random_state=RANDOM_STATE)
    db = davies_bouldin_score(X_scaled, km_labels)
    print(f"\nFinal K-Means (k={KMEANS_K}): Silhouette={sil:.4f}  DB={db:.4f}")

    print("\n--- PCA ---")
    X_pca, _ = apply_pca(X_scaled, n_components=2)
    plot_pca_scatter(X_pca, y, label_names,
                     "PCA – Coloured by Growth Class", "pca_true_labels.png")
    plot_pca_scatter(X_pca, km_labels,
                     [f"Cluster {i}" for i in range(KMEANS_K)],
                     f"PCA – K-Means Clusters (k={KMEANS_K})", "pca_kmeans_clusters.png")
    pca_full = apply_pca_full(X_scaled)
    plot_pca_variance(pca_full)
    save_unsupervised_metrics(km_results, km, sil, db, pca_full)

    print("\n--- Municipality trajectory clustering ---")
    mun_pivot = build_growth_pivot(growth_df, "municipality", top_n=TOP_MUNICIPALITIES)
    _, mun_labels, _ = cluster_growth_trajectories(mun_pivot, k=KMEANS_K)
    plot_trajectory_clusters(mun_pivot, mun_labels, "municipality")

    print("\n--- Sector trajectory clustering ---")
    sec_pivot = build_growth_pivot(growth_df, "primary_sector", top_n=TOP_SECTORS)
    _, sec_labels, _ = cluster_growth_trajectories(sec_pivot, k=KMEANS_K)
    plot_trajectory_clusters(sec_pivot, sec_labels, "primary_sector")


# ── Step D: Growth cross-analysis heatmaps ───────────────────────────────────


def run_analysis(growth_df) -> None:
    print("\n--- Municipality × Year ---")
    plot_growth_heatmap(growth_df, "municipality", top_n_rows=TOP_MUNICIPALITIES,
                        title=f"Top {TOP_MUNICIPALITIES} Municipalities – YoY Growth by Year",
                        filename="municipality_growth_by_year.png")

    print("\n--- Sector × Year ---")
    plot_growth_heatmap(growth_df, "primary_sector", top_n_rows=TOP_SECTORS,
                        title=f"Top {TOP_SECTORS} Sectors – YoY Growth by Year",
                        filename="sector_growth_by_year.png")

    print("\n--- Sector × Municipality (all years) ---")
    plot_sector_by_municipality_heatmap(growth_df, TOP_MUNICIPALITIES, TOP_SECTORS)

    for yr in sorted(growth_df["year"].unique())[-3:]:
        print(f"\n--- Sector × Municipality ({yr}) ---")
        plot_sector_by_municipality_heatmap(growth_df, TOP_MUNICIPALITIES, TOP_SECTORS,
                                            year=yr)


# ── Step E: 2026 Market predictions ──────────────────────────────────────────


def run_predictions(
    growth_df, rf, xgb, lr, rfr, xgbr,
    feature_cols, encoders, scaler,
) -> None:
    print(f"\nBuilding {FORECAST_YEAR} prediction input …")
    pred_df = build_prediction_input(growth_df, FORECAST_YEAR)
    X_pred  = encode_for_prediction(pred_df, feature_cols, encoders, scaler)

    print(f"Running all 5 models on {len(pred_df):,} municipality×sector pairs …")
    out = save_predictions(
        pred_df,
        cls_models={"RandomForest": rf, XGB_NAME: xgb},
        reg_models={
            "LinearRegression": lr,
            "RandomForestRegressor": rfr,
            f"{XGB_NAME}Regressor": xgbr,
        },
        X_pred=X_pred,
        encoders=encoders,
        label_names=list(encoders[TARGET_COLUMN].classes_),
        forecast_year=FORECAST_YEAR,
    )

    # Visualise predictions from the best regressor (XGBoost)
    best_rate_col = f"growth_rate_{XGB_NAME}Regressor"
    print(f"\n--- {FORECAST_YEAR} Prediction Heatmap ({XGB_NAME} Regressor) ---")
    plot_predictions_heatmap(out, best_rate_col, FORECAST_YEAR,
                             TOP_MUNICIPALITIES, TOP_SECTORS)
    plot_top_predictions(out, best_rate_col, FORECAST_YEAR)

    # Show top 10 growth opportunities
    print(f"\nTop 10 predicted growth opportunities in {FORECAST_YEAR}:")
    top10 = (
        out[["municipality", "primary_sector", best_rate_col,
             f"est_turnover_{XGB_NAME}Regressor"]]
        .sort_values(best_rate_col, ascending=False)
        .head(10)
    )
    for _, row in top10.iterrows():
        print(
            f"  {row['municipality']:<25} | {row['primary_sector'][:35]:<35} "
            f"| {row[best_rate_col]:+.1%} "
            f"| Est. €{row[f'est_turnover_{XGB_NAME}Regressor']:,.0f}"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Phase 2 – Economic Growth Analysis")
    parser.add_argument(
        "--step",
        choices=["all", "supervised", "regression", "unsupervised",
                 "analysis", "predictions"],
        default="all",
    )
    args = parser.parse_args(argv)

    print("=" * 60)
    print("  PHASE 2 – Economic Growth Analysis & Market Predictions")
    print(f"  Algorithms: RF Clf | XGB Clf | Linear Reg | RF Reg | XGB Reg")
    print(f"              K-Means | PCA")
    print("=" * 60)

    raw_df    = load_data()
    growth_df = build_growth_dataset(raw_df)
    X_scaled, y, feature_cols, encoders, scaler = encode_and_scale(growth_df)

    # Placeholders — populated by whichever steps run
    rf = xgb = lr = rfr = xgbr = None

    if args.step in ("all", "supervised"):
        print("\n" + "=" * 60)
        print("  STEP A – Classification: GROWING / STABLE / DECLINING")
        print("=" * 60)
        rf, xgb, *_ = run_supervised(X_scaled, y, feature_cols, encoders)

    if args.step in ("all", "regression"):
        print("\n" + "=" * 60)
        print("  STEP B – Regression: predict continuous growth_rate")
        print("=" * 60)
        lr, rfr, xgbr = run_regression(X_scaled, growth_df, feature_cols)

    if args.step in ("all", "unsupervised"):
        print("\n" + "=" * 60)
        print("  STEP C – Unsupervised: K-Means + PCA trajectory clustering")
        print("=" * 60)
        run_unsupervised(X_scaled, y, encoders, growth_df)

    if args.step in ("all", "analysis"):
        print("\n" + "=" * 60)
        print("  STEP D – Growth cross-analysis heatmaps")
        print("=" * 60)
        run_analysis(growth_df)

    if args.step in ("all", "predictions"):
        # Need all supervised models — load from disk if not already trained
        if rf is None:
            from src.supervised import load_model
            rf   = load_model("random_forest")
            xgb  = load_model(XGB_NAME.lower())
        if lr is None:
            from src.supervised import load_model
            lr   = load_model("linear_regression")
            rfr  = load_model("rf_regressor")
            xgbr = load_model("xgb_regressor")
        print("\n" + "=" * 60)
        print(f"  STEP E – {FORECAST_YEAR} Market Predictions")
        print("=" * 60)
        run_predictions(growth_df, rf, xgb, lr, rfr, xgbr,
                        feature_cols, encoders, scaler)

    print("\n" + "=" * 60)
    print(f"  Done.  Outputs → {OUTPUTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1:])
