"""
ablation.py - Incremental contribution analysis for Phase 3 improvements.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, f1_score

from src.config import BASELINE_NUMERIC_FEATURES, METRICS_DIR, NUMERIC_FEATURES, PLOTS_DIR, XGB_PARAMS
from src.feature_engineering import encode_target, fit_encoders, transform_features
from src.sampling import resample_smote
from src.supervised import build_xgb_classifier
from src.time_split import time_based_split

matplotlib.use("Agg")


def _evaluate_level(name: str, model, X_test: np.ndarray, y_test: np.ndarray, label_names: list[str]) -> dict:
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
    return {
        "level": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "kappa": cohen_kappa_score(y_test, y_pred),
        "stable_recall": report.get("STABLE", {}).get("recall", np.nan),
    }


def run_ablation(
    growth_df: pd.DataFrame,
    best_xgb_params: dict | None = None,
) -> pd.DataFrame:
    train_df, val_df, test_df = time_based_split(growth_df)
    _, _, y_test, label_encoder = encode_target(train_df, val_df, test_df)
    y_train, y_val, _, _ = encode_target(train_df, val_df, test_df)
    label_names = list(label_encoder.classes_)

    a1_features = BASELINE_NUMERIC_FEATURES + [
        "month",
        "lag1_turnover_log1p",
        "lag2_turnover_log1p",
        "lag3_turnover_log1p",
        "rolling3_mean_log1p",
        "rolling6_mean_log1p",
        "rolling3_std",
        "rolling6_std",
    ]
    a2_features = a1_features + ["month_sin", "month_cos"]
    a3_features = a2_features + ["market_concentration_log1p"]

    levels = [
        ("A0", {"features": BASELINE_NUMERIC_FEATURES, "strategy": "onehot", "smote": False, "params": XGB_PARAMS}),
        ("A1", {"features": a1_features, "strategy": "onehot", "smote": False, "params": XGB_PARAMS}),
        ("A2", {"features": a2_features, "strategy": "onehot", "smote": False, "params": XGB_PARAMS}),
        ("A3", {"features": a3_features, "strategy": "onehot", "smote": False, "params": XGB_PARAMS}),
        ("A4", {"features": a3_features, "strategy": "target", "smote": False, "params": XGB_PARAMS}),
        ("A5", {"features": a3_features, "strategy": "target", "smote": True, "params": XGB_PARAMS}),
        ("A6", {"features": NUMERIC_FEATURES, "strategy": "target", "smote": True, "params": (best_xgb_params or XGB_PARAMS)}),
    ]

    rows = []
    for level_name, options in levels:
        bundle = fit_encoders(
            train_df,
            strategy=options["strategy"],
            numeric_features_override=options["features"],
        )
        X_train = transform_features(train_df, bundle)
        X_test = transform_features(test_df, bundle)

        if options["smote"]:
            X_train_fit, y_train_fit = resample_smote(X_train, y_train)
        else:
            X_train_fit, y_train_fit = X_train, y_train

        model = build_xgb_classifier(params=options["params"], n_classes=len(label_names))
        model.fit(X_train_fit, y_train_fit)
        rows.append(_evaluate_level(level_name, model, X_test, y_test, label_names))

    results = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(METRICS_DIR / "ablation_results.csv", index=False)

    deltas = results["f1_macro"].diff().fillna(results["f1_macro"])
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(results["level"], deltas, color=["steelblue" if value >= 0 else "indianred" for value in deltas])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_title("Ablation Contribution to F1-Macro")
    ax.set_ylabel("Incremental Delta")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ablation_contribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results
