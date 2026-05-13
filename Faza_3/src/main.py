#!/usr/bin/env python3
"""
main.py - Phase 3 entry point.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score

from src.ablation import run_ablation
from src.config import (
    EARLY_STOPPING_ROUNDS,
    ENCODING_STRATEGY,
    FORECAST_YEAR,
    KMEANS_K,
    METRICS_DIR,
    N_CLUSTERS_RANGE,
    OUTPUTS_DIR,
    PHASE2_METRICS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    RF_PARAM_DIST,
    RF_PARAMS,
    RF_REG_PARAM_DIST,
    RF_REG_PARAMS,
    TUNE_CV_FOLDS,
    TUNE_ITER,
    XGB_PARAM_DIST,
    XGB_PARAMS,
    XGB_REG_PARAM_DIST,
    XGB_REG_PARAMS,
)
from src.data_loader import (
    build_growth_dataset,
    build_growth_pivot,
    build_prediction_input,
    load_data,
)
from src.evaluation import (
    collect_phase3_metrics,
    compare_models,
    compare_phase2_phase3,
    compare_regressors,
    evaluate_classifier,
    evaluate_regressor,
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_growth_heatmap,
    plot_learning_curve_xgb,
    plot_predictions_heatmap,
    plot_residuals,
    plot_roc_curves,
    plot_sector_by_municipality_heatmap,
    plot_top_predictions,
    save_predictions,
)
from src.feature_engineering import encode_for_prediction, encode_target, fit_encoders, transform_features
from src.sampling import resample_smote
from src.supervised import (
    XGB_AVAILABLE,
    build_linear_regression,
    build_rf_classifier,
    build_rf_regressor,
    build_xgb_classifier,
    build_xgb_regressor,
    load_best_params,
    load_model,
    save_best_params,
    save_model,
    train_xgboost_with_early_stop,
    tune_hyperparameters,
)
from src.time_split import time_based_split
from src.unsupervised import apply_pca, apply_pca_full, cluster_growth_trajectories, find_optimal_k, train_kmeans


def prepare_datasets(growth_df: pd.DataFrame, encoding: str):
    train_df, val_df, test_df = time_based_split(growth_df)
    bundle = fit_encoders(train_df, strategy=encoding)
    X_train = transform_features(train_df, bundle)
    X_val = transform_features(val_df, bundle)
    X_test = transform_features(test_df, bundle)
    y_train, y_val, y_test, label_encoder = encode_target(train_df, val_df, test_df)

    assert not np.isnan(X_train).any()
    assert not np.isnan(X_val).any()
    assert not np.isnan(X_test).any()

    y_reg_train = train_df["growth_rate"].to_numpy(dtype=float)
    y_reg_val = val_df["growth_rate"].to_numpy(dtype=float)
    y_reg_test = test_df["growth_rate"].to_numpy(dtype=float)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "bundle": bundle,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_reg_train": y_reg_train,
        "y_reg_val": y_reg_val,
        "y_reg_test": y_reg_test,
        "label_encoder": label_encoder,
    }


def run_supervised_and_regression(data: dict, tune: bool, use_smote: bool):
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    y_reg_train = data["y_reg_train"]
    y_reg_val = data["y_reg_val"]
    y_reg_test = data["y_reg_test"]
    label_names = list(data["label_encoder"].classes_)
    feature_names = data["bundle"].feature_names

    X_train_cls, y_train_cls = (resample_smote(X_train, y_train) if use_smote else (X_train, y_train))

    best_params = load_best_params()
    if tune:
        _, rf_best, _ = tune_hyperparameters(
            build_rf_classifier(),
            RF_PARAM_DIST,
            X_train_cls,
            y_train_cls,
            task="classification",
            n_iter=TUNE_ITER,
            cv_folds=TUNE_CV_FOLDS,
            output_name="random_forest",
        )
        _, xgb_best, _ = tune_hyperparameters(
            build_xgb_classifier(n_classes=len(label_names)),
            XGB_PARAM_DIST,
            X_train_cls,
            y_train_cls,
            task="classification",
            n_iter=TUNE_ITER,
            cv_folds=TUNE_CV_FOLDS,
            output_name="xgboost",
        )
        _, rf_reg_best, _ = tune_hyperparameters(
            build_rf_regressor(),
            RF_REG_PARAM_DIST,
            X_train,
            y_reg_train,
            task="regression",
            n_iter=TUNE_ITER,
            cv_folds=TUNE_CV_FOLDS,
            output_name="rf_regressor",
        )
        _, xgb_reg_best, _ = tune_hyperparameters(
            build_xgb_regressor(),
            XGB_REG_PARAM_DIST,
            X_train,
            y_reg_train,
            task="regression",
            n_iter=TUNE_ITER,
            cv_folds=TUNE_CV_FOLDS,
            output_name="xgb_regressor",
        )
        best_params = {
            "random_forest": rf_best,
            "xgboost": xgb_best,
            "rf_regressor": rf_reg_best,
            "xgb_regressor": xgb_reg_best,
        }
        save_best_params(best_params)

    rf_params = {**RF_PARAMS, **best_params.get("random_forest", {})}
    xgb_params = {**XGB_PARAMS, **best_params.get("xgboost", {})}
    rf_reg_params = {**RF_REG_PARAMS, **best_params.get("rf_regressor", {})}
    xgb_reg_params = {**XGB_REG_PARAMS, **best_params.get("xgb_regressor", {})}

    rf = build_rf_classifier(rf_params)
    rf.fit(np.vstack([X_train_cls, X_val]), np.concatenate([y_train_cls, y_val]))

    xgb = train_xgboost_with_early_stop(
        X_train_cls,
        y_train_cls,
        X_val,
        y_val,
        xgb_params,
        task="classification",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    lr = build_linear_regression()
    lr.fit(np.vstack([X_train, X_val]), np.concatenate([y_reg_train, y_reg_val]))

    rfr = build_rf_regressor(rf_reg_params)
    rfr.fit(np.vstack([X_train, X_val]), np.concatenate([y_reg_train, y_reg_val]))

    xgbr = train_xgboost_with_early_stop(
        X_train,
        y_reg_train,
        X_val,
        y_reg_val,
        xgb_reg_params,
        task="regression",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )

    save_model(rf, "random_forest")
    save_model(xgb, "xgboost")
    save_model(lr, "linear_regression")
    save_model(rfr, "rf_regressor")
    save_model(xgbr, "xgb_regressor")

    classification_results = {}
    for model, name in [(rf, "RandomForest"), (xgb, "XGBoost")]:
        result = evaluate_classifier(model, X_test, y_test, label_names, name)
        plot_confusion_matrix(model, X_test, y_test, label_names, name)
        plot_roc_curves(model, X_test, y_test, label_names, name)
        plot_feature_importance(model, feature_names, name)
        classification_results[name] = result

    regression_results = {}
    for model, name in [
        (lr, "LinearRegression"),
        (rfr, "RandomForestRegressor"),
        (xgbr, "XGBoostRegressor"),
    ]:
        result = evaluate_regressor(model, X_test, y_reg_test, name)
        plot_actual_vs_predicted(y_reg_test, result["y_pred"], name)
        plot_residuals(y_reg_test, result["y_pred"], name)
        plot_feature_importance(model, feature_names, name)
        regression_results[name] = result

    plot_learning_curve_xgb(xgb, "learning_curve_xgb.png")
    classification_df = compare_models(classification_results)
    regression_df = compare_regressors(regression_results)
    return rf, xgb, lr, rfr, xgbr, classification_df, regression_df


def run_unsupervised_step(data: dict, growth_df: pd.DataFrame) -> None:
    X_all = np.vstack([data["X_train"], data["X_val"], data["X_test"]])
    y_all = np.concatenate([data["y_train"], data["y_val"], data["y_test"]])
    km_results = find_optimal_k(X_all, k_range=N_CLUSTERS_RANGE)
    km = train_kmeans(X_all, k=KMEANS_K)
    silhouette = silhouette_score(X_all, km.labels_, sample_size=min(5000, len(X_all)), random_state=RANDOM_STATE)
    db = davies_bouldin_score(X_all, km.labels_)
    print(f"KMeans silhouette={silhouette:.4f} davies_bouldin={db:.4f}")
    X_pca, _ = apply_pca(X_all, n_components=2)
    apply_pca_full(X_all)

    mun_pivot = build_growth_pivot(growth_df, "municipality")
    cluster_growth_trajectories(mun_pivot, k=KMEANS_K)
    sec_pivot = build_growth_pivot(growth_df, "primary_sector")
    cluster_growth_trajectories(sec_pivot, k=KMEANS_K)


def run_analysis(growth_df: pd.DataFrame) -> None:
    plot_growth_heatmap(
        growth_df,
        "municipality",
        top_n_rows=20,
        title="Top Municipalities - YoY Growth by Year",
        filename="municipality_growth_by_year.png",
    )
    plot_growth_heatmap(
        growth_df,
        "primary_sector",
        top_n_rows=15,
        title="Top Sectors - YoY Growth by Year",
        filename="sector_growth_by_year.png",
    )
    plot_sector_by_municipality_heatmap(growth_df)
    for year in sorted(growth_df["year"].unique())[-3:]:
        plot_sector_by_municipality_heatmap(growth_df, year=int(year))


def run_predictions(
    growth_df: pd.DataFrame,
    bundle,
    label_encoder,
    rf,
    xgb,
    lr,
    rfr,
    xgbr,
    forecast_mode: str,
) -> pd.DataFrame:
    if forecast_mode == "recursive":
        print("Recursive forecast mode requested; using static forecast base for this run.")

    pred_df = build_prediction_input(growth_df, forecast_year=FORECAST_YEAR)
    X_pred = encode_for_prediction(pred_df, bundle)
    out = save_predictions(
        pred_df,
        cls_models={"RandomForest": rf, "XGBoost": xgb},
        reg_models={
            "LinearRegression": lr,
            "RandomForestRegressor": rfr,
            "XGBoostRegressor": xgbr,
        },
        X_pred=X_pred,
        label_encoder=label_encoder,
        forecast_year=FORECAST_YEAR,
    )
    plot_predictions_heatmap(out, "growth_rate_XGBoostRegressor", FORECAST_YEAR)
    plot_top_predictions(out, "growth_rate_XGBoostRegressor", FORECAST_YEAR)
    return out


def run_compare_only() -> pd.DataFrame:
    classification_df = pd.read_csv(METRICS_DIR / "model_comparison.csv", index_col=0)
    regression_df = pd.read_csv(METRICS_DIR / "regressor_comparison.csv", index_col=0)
    phase3_results = collect_phase3_metrics(classification_df, regression_df)
    return compare_phase2_phase3(
        PHASE2_METRICS_DIR,
        phase3_results,
        METRICS_DIR / "comparison_phase2_phase3.csv",
        PLOTS_DIR / "phase2_vs_phase3_comparison.png",
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Phase 3 - Improvement and Fine-Tuning")
    parser.add_argument(
        "--step",
        choices=["all", "supervised", "regression", "unsupervised", "analysis", "predictions", "compare", "ablation"],
        default="all",
    )
    parser.add_argument("--tune", action="store_true", help="Run RandomizedSearchCV before final training.")
    parser.add_argument("--encoding", choices=["target", "onehot"], default=ENCODING_STRATEGY)
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--forecast-mode", choices=["static", "recursive"], default="static")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("  PHASE 3 - Improvement Plan and Fine-Tuning")
    print("=" * 60)

    if args.step == "compare":
        run_compare_only()
        return

    raw_df = load_data()
    growth_df = build_growth_dataset(raw_df)
    data = prepare_datasets(growth_df, encoding=args.encoding)

    rf = xgb = lr = rfr = xgbr = None
    classification_df = regression_df = None

    if args.step in ("all", "supervised", "regression"):
        rf, xgb, lr, rfr, xgbr, classification_df, regression_df = run_supervised_and_regression(
            data,
            tune=args.tune,
            use_smote=not args.no_smote,
        )

    if args.step == "supervised":
        return

    if args.step in ("all", "unsupervised"):
        run_unsupervised_step(data, growth_df)

    if args.step in ("all", "analysis"):
        run_analysis(growth_df)

    if args.step in ("all", "predictions"):
        if rf is None:
            rf = load_model("random_forest")
            xgb = load_model("xgboost")
            lr = load_model("linear_regression")
            rfr = load_model("rf_regressor")
            xgbr = load_model("xgb_regressor")
        run_predictions(
            growth_df,
            data["bundle"],
            data["label_encoder"],
            rf,
            xgb,
            lr,
            rfr,
            xgbr,
            forecast_mode=args.forecast_mode,
        )

    if args.step in ("all", "ablation"):
        best_params = load_best_params().get("xgboost", {})
        run_ablation(growth_df, best_xgb_params=best_params)

    if args.step in ("all", "regression") and classification_df is not None and regression_df is not None:
        phase3_results = collect_phase3_metrics(classification_df, regression_df)
        compare_phase2_phase3(
            PHASE2_METRICS_DIR,
            phase3_results,
            METRICS_DIR / "comparison_phase2_phase3.csv",
            PLOTS_DIR / "phase2_vs_phase3_comparison.png",
        )

    print(f"Outputs available in {OUTPUTS_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
