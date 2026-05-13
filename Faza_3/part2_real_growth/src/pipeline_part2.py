"""
pipeline_part2.py – Orchestrator for Part 2 (Real Growth Analysis).

Pipeline
--------
1. Load Phase 3 Part 1 growth_df (monthly aggregation, 39k rows).
2. Augment with CPI (inflation) and COVID indicators.
3. Build three parallel models:
     - M_nominal      : target = growth_class       (Part 1 reproduction)
     - M_real         : target = real_growth_class  (Fisher-deflated)
     - M_real_no_covid: trained on 2019+2022-2024, used for counterfactual
4. Forecast 2027 with three inflation scenarios (1.5%, 2.5%, 4%).
5. Compare nominal vs real top-10 opportunities.
6. Run COVID counterfactual: predict 2020-2021 with M_real_no_covid,
   compare with actuals to quantify pandemic damage per sector.

Usage
-----
    python -m part2_real_growth.src.pipeline_part2 --step all
    python -m part2_real_growth.src.pipeline_part2 --step deflate
    python -m part2_real_growth.src.pipeline_part2 --step train
    python -m part2_real_growth.src.pipeline_part2 --step counterfactual
    python -m part2_real_growth.src.pipeline_part2 --step forecast
    python -m part2_real_growth.src.pipeline_part2 --step compare
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder

# Import Part 1 (Phase 3) helpers
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # group-3-machine-learning/
sys.path.insert(0, str(PROJECT_ROOT / "Faza_3"))

from src.data_loader import build_growth_dataset, load_data       # noqa: E402
from src.feature_engineering import EncoderBundle, fit_encoders, transform_features  # noqa: E402
from src.time_split import time_based_split                       # noqa: E402

from part2_real_growth.src.covid_features import add_covid_features  # noqa: E402
from part2_real_growth.src.deflation import (
    REAL_GROWTH_LOWER,
    REAL_GROWTH_UPPER,
    add_inflation_features,
    reflate_value,
)  # noqa: E402
from part2_real_growth.src.external_data import (
    load_cpi,
    load_covid,
    load_imf_forecast,
)  # noqa: E402

try:
    from xgboost import XGBClassifier, XGBRegressor

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


PART2_DIR = Path(__file__).resolve().parent.parent
METRICS_DIR = PART2_DIR / "outputs" / "metrics"
PLOTS_DIR = PART2_DIR / "outputs" / "plots"
MODELS_DIR = PART2_DIR / "outputs" / "models"

for d in [METRICS_DIR, PLOTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# Numeric features for Part 2 — Phase 3 features + new CPI/COVID columns
NUMERIC_FEATURES_PART2 = [
    "year",
    "month",
    "prev_turnover_log1p",
    "lag1_turnover_log1p",
    "lag2_turnover_log1p",
    "lag3_turnover_log1p",
    "rolling3_mean_log1p",
    "rolling6_mean_log1p",
    "rolling3_std",
    "rolling6_std",
    "num_businesses_log1p",
    "market_concentration_log1p",
    "month_sin",
    "month_cos",
    # NEW for Part 2:
    "cpi_yoy_pct",
    "inflation_shock",
    "covid_stringency",
    "cases_per_100k",
    "vaccination_rate",
    "is_covid_year",
    "is_post_covid",
    "months_since_lockdown",
]

CATEGORICAL_FEATURES = ["municipality", "primary_sector"]


# ── Step 1: Build augmented growth dataset ────────────────────────────────────


def build_augmented_dataset() -> pd.DataFrame:
    """Load Part 1 growth_df, augment with CPI + COVID."""
    print("=" * 60)
    print("  PART 2 — Real Growth Analysis (Inflation + COVID)")
    print("=" * 60)
    print("\n[1/6] Loading Phase 1 cleaned data + building monthly growth dataset…")
    raw = load_data()
    growth_df = build_growth_dataset(raw)
    print(f"  Phase 3 Part 1 monthly dataset: {len(growth_df):,} rows")

    print("\n[2/6] Loading and joining CPI (inflation) …")
    cpi = load_cpi()
    growth_df = add_inflation_features(growth_df, cpi)

    print("\n[3/6] Loading and joining COVID indicators …")
    covid = load_covid()
    growth_df = add_covid_features(growth_df, covid)

    # Print class distribution: nominal vs real
    nominal_counts = growth_df["growth_class"].value_counts()
    real_counts = growth_df["real_growth_class"].value_counts()
    print("\n  Class distribution (nominal vs real):")
    print(f"    Nominal: {nominal_counts.to_dict()}")
    print(f"    Real:    {real_counts.to_dict()}")

    return growth_df


# ── Step 2: Feature engineering wrapper that includes Part 2 features ─────────


def _fit_encoders_part2(train_df: pd.DataFrame, target_col: str) -> EncoderBundle:
    """Wrap Part 1 fit_encoders with Part 2 feature list and per-target choice."""
    # Patch the REGRESSION_TARGET globally for TargetEncoder fit
    # We'll use target_col as the regression target it fits on.
    import src.config as p3_config  # type: ignore
    original = p3_config.REGRESSION_TARGET
    if target_col == "real_growth_class":
        p3_config.REGRESSION_TARGET = "real_growth_rate"
    else:
        p3_config.REGRESSION_TARGET = "growth_rate"
    try:
        bundle = fit_encoders(
            train_df,
            strategy="target",
            numeric_features_override=NUMERIC_FEATURES_PART2,
        )
    finally:
        p3_config.REGRESSION_TARGET = original
    return bundle


# ── Step 3: Train one model "track" (M_nominal / M_real / M_real_no_covid) ────


def train_track(
    growth_df: pd.DataFrame,
    target_class_col: str,
    target_reg_col: str,
    track_name: str,
    train_filter=None,
) -> dict:
    """
    Train a full track (classifier + regressor on chosen target).

    Returns dict with:
        bundle, label_encoder, models (rf_cls, xgb_cls, lr, rf_reg, xgb_reg),
        metrics (classification + regression), and predictions on test 2025.
    """
    print(f"\n--- Training track: {track_name} | target_class={target_class_col} | target_reg={target_reg_col} ---")

    df = growth_df.copy()
    if train_filter is not None:
        df = df[train_filter(df)].copy()
        print(f"  Filter applied: {len(df):,} rows remaining")

    train_df, val_df, test_df = time_based_split(df)

    bundle = _fit_encoders_part2(train_df, target_class_col)
    X_train = transform_features(train_df, bundle)
    X_val = transform_features(val_df, bundle)
    X_test = transform_features(test_df, bundle)

    label_enc = LabelEncoder()
    y_train = label_enc.fit_transform(train_df[target_class_col].astype(str))
    y_val = label_enc.transform(val_df[target_class_col].astype(str))
    y_test = label_enc.transform(test_df[target_class_col].astype(str))

    y_reg_train = train_df[target_reg_col].to_numpy(dtype=float)
    y_reg_val = val_df[target_reg_col].to_numpy(dtype=float)
    y_reg_test = test_df[target_reg_col].to_numpy(dtype=float)

    # Concatenate train+val for final RF fits (no early stopping)
    X_tv = np.vstack([X_train, X_val])
    y_tv_cls = np.concatenate([y_train, y_val])
    y_tv_reg = np.concatenate([y_reg_train, y_reg_val])

    # Build classifiers
    rf_cls = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_cls.fit(X_tv, y_tv_cls)

    xgb_cls = None
    if XGB_AVAILABLE:
        n_classes = int(np.unique(y_train).size)
        xgb_cls = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            num_class=n_classes if n_classes > 2 else None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mlogloss",
        )
        # Balanced sample weights to compensate for early stopping on val
        from sklearn.utils.class_weight import compute_sample_weight
        sw_train = compute_sample_weight("balanced", y_train)
        sw_val = compute_sample_weight("balanced", y_val)
        xgb_cls.set_params(early_stopping_rounds=30)
        xgb_cls.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sw_train,
            sample_weight_eval_set=[sw_val],
            verbose=False,
        )

    # Regressors
    lr = LinearRegression()
    lr.fit(X_tv, y_tv_reg)

    rf_reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_reg.fit(X_tv, y_tv_reg)

    xgb_reg = None
    if XGB_AVAILABLE:
        xgb_reg = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="rmse",
        )
        xgb_reg.set_params(early_stopping_rounds=30)
        xgb_reg.fit(
            X_train, y_reg_train,
            eval_set=[(X_val, y_reg_val)],
            verbose=False,
        )

    # Metrics
    cls_metrics = {}
    for name, model in [("RandomForest", rf_cls), ("XGBoost", xgb_cls)]:
        if model is None:
            continue
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test, y_pred, target_names=label_enc.classes_, output_dict=True, zero_division=0
        )
        cls_metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
            "kappa": cohen_kappa_score(y_test, y_pred),
            "report": report,
        }

    reg_metrics = {}
    for name, model in [("LinearRegression", lr), ("RandomForestRegressor", rf_reg), ("XGBoostRegressor", xgb_reg)]:
        if model is None:
            continue
        y_pred = model.predict(X_test)
        reg_metrics[name] = {
            "MAE": float(mean_absolute_error(y_reg_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_reg_test, y_pred))),
            "R²": float(r2_score(y_reg_test, y_pred)),
        }

    print(f"  Classifiers: {[(n, round(m['accuracy'], 3), round(m['f1_macro'], 3)) for n, m in cls_metrics.items()]}")
    print(f"  Regressors:  {[(n, round(m['R²'], 3)) for n, m in reg_metrics.items()]}")

    return {
        "track_name": track_name,
        "bundle": bundle,
        "label_encoder": label_enc,
        "rf_cls": rf_cls,
        "xgb_cls": xgb_cls,
        "lr": lr,
        "rf_reg": rf_reg,
        "xgb_reg": xgb_reg,
        "cls_metrics": cls_metrics,
        "reg_metrics": reg_metrics,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_test": X_test,
        "y_test": y_test,
        "y_reg_test": y_reg_test,
    }


# ── Step 4: COVID Counterfactual ──────────────────────────────────────────────


def run_counterfactual(growth_df: pd.DataFrame, track_no_covid: dict) -> pd.DataFrame:
    """
    Use the no-COVID-trained model to predict what 2020-2021 would have been
    without the pandemic. Compare with actuals → quantify COVID damage per sector.
    """
    print("\n[4/6] COVID counterfactual analysis (no-COVID model predicts 2020-2021)…")

    covid_period = growth_df[growth_df["year"].isin([2020, 2021])].copy()
    if len(covid_period) == 0:
        print("  No COVID-period rows found.")
        return pd.DataFrame()

    bundle = track_no_covid["bundle"]
    rf_reg = track_no_covid["rf_reg"]
    X_cf = transform_features(covid_period, bundle)

    covid_period["counterfactual_real_growth"] = rf_reg.predict(X_cf)
    covid_period["covid_damage_pp"] = (
        covid_period["real_growth_rate"] - covid_period["counterfactual_real_growth"]
    )

    # Aggregate damage by sector (mean over 2020-2021)
    sector_damage = (
        covid_period.groupby("primary_sector")["covid_damage_pp"]
        .agg(["mean", "median", "count"])
        .sort_values("mean")
        .reset_index()
        .rename(columns={"mean": "mean_damage_pp", "median": "median_damage_pp", "count": "n_rows"})
    )
    sector_damage.to_csv(METRICS_DIR / "covid_impact_by_sector.csv", index=False)
    print(f"  -> Saved: outputs/metrics/covid_impact_by_sector.csv ({len(sector_damage)} sectors)")

    # Plot top-10 most damaged sectors
    top10 = sector_damage.head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["indianred" if v < 0 else "seagreen" for v in top10["mean_damage_pp"]]
    bars = ax.barh(range(len(top10)), top10["mean_damage_pp"], color=colors)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10["primary_sector"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean COVID damage (real growth pp, actual − counterfactual)")
    ax.set_title("Top 10 Sectors Most Damaged by COVID-19 (2020-2021)")
    ax.bar_label(bars, fmt="%+.1%", padding=3, fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "covid_impact_by_sector.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved: outputs/plots/covid_impact_by_sector.png")

    # Counterfactual full table
    full_cf = covid_period[[
        "year", "month", "municipality", "primary_sector",
        "real_growth_rate", "counterfactual_real_growth", "covid_damage_pp",
    ]].sort_values(["year", "month", "municipality", "primary_sector"])
    full_cf.to_csv(METRICS_DIR / "counterfactual_no_covid.csv", index=False)
    print(f"  -> Saved: outputs/metrics/counterfactual_no_covid.csv ({len(full_cf):,} rows)")

    return sector_damage


# ── Step 5: 2027 Forecast with 3 inflation scenarios ──────────────────────────


def forecast_2027(growth_df: pd.DataFrame, track_real: dict) -> pd.DataFrame:
    """
    Forecast 2027 using M_real, then convert real → nominal under 3 IMF scenarios.
    """
    print("\n[5/6] Forecasting 2027 (real growth + 3 inflation scenarios)…")
    imf = load_imf_forecast()

    # Build prediction input: take last available month/sector/mun and project to 2027
    latest_year = int(growth_df["year"].max())
    base = growth_df[growth_df["year"] == latest_year].copy()
    pred = base.copy()
    pred["year"] = 2027

    # For 2027, we use latest year's CPI as approximation for the lag features
    # (the CPI for 2027 itself comes from IMF scenarios below).
    bundle = track_real["bundle"]
    X_pred = transform_features(pred, bundle)

    rf_reg = track_real["rf_reg"]
    xgb_reg = track_real["xgb_reg"]
    real_pred_rf = np.clip(rf_reg.predict(X_pred), -1.0, 2.0)
    real_pred_xgb = np.clip(xgb_reg.predict(X_pred), -1.0, 2.0) if xgb_reg is not None else real_pred_rf

    out = pred[["year", "month", "municipality", "primary_sector"]].copy()
    out["real_growth_rate_rf"] = real_pred_rf
    out["real_growth_rate_xgb"] = real_pred_xgb

    # For each scenario, compute nominal_2027 = (1 + real) * (1 + inflation) - 1
    for _, row in imf[imf["year"] == 2027].iterrows():
        scenario = row["scenario"]
        infl = row["cpi_yoy_pct"] / 100.0
        out[f"nominal_growth_rate_{scenario}"] = [
            reflate_value(r, infl) for r in real_pred_xgb
        ]

    out.to_csv(METRICS_DIR / "real_growth_predictions_2027.csv", index=False)
    print(f"  -> Saved: outputs/metrics/real_growth_predictions_2027.csv ({len(out):,} rows)")

    # Top 10 real growth opportunities (averaged across months)
    top10 = (
        out.groupby(["municipality", "primary_sector"])["real_growth_rate_xgb"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top10.to_csv(METRICS_DIR / "real_growth_top10_2027.csv", index=False)
    print("\n  Top 10 REAL growth opportunities 2027 (XGB Regressor):")
    for _, row in top10.iterrows():
        print(f"    {row['municipality']:<20} | {row['primary_sector'][:40]:<40} | {row['real_growth_rate_xgb']:+.1%}")

    return out


# ── Step 6: Nominal vs Real comparison ────────────────────────────────────────


def compare_nominal_vs_real(growth_df: pd.DataFrame) -> pd.DataFrame:
    """Direction-change table: which (year, sector) flipped GROWING <-> DECLINING."""
    print("\n[6/6] Building nominal-vs-real direction-change comparison…")
    by_year_sec = (
        growth_df.groupby(["year", "primary_sector"])[["growth_rate", "real_growth_rate", "cpi_yoy_pct"]]
        .mean()
        .reset_index()
    )

    def label(r, lo=-0.05, hi=0.05):
        if r > hi:
            return "GROWING"
        if r < lo:
            return "DECLINING"
        return "STABLE"

    by_year_sec["nominal_label"] = by_year_sec["growth_rate"].apply(label)
    by_year_sec["real_label"] = by_year_sec["real_growth_rate"].apply(lambda r: label(r, -0.03, 0.03))
    by_year_sec["direction_change"] = (
        by_year_sec["nominal_label"] + " → " + by_year_sec["real_label"]
    )
    by_year_sec["flipped"] = (
        ((by_year_sec["nominal_label"] == "GROWING") & (by_year_sec["real_label"] == "DECLINING")) |
        ((by_year_sec["nominal_label"] == "DECLINING") & (by_year_sec["real_label"] == "GROWING"))
    )

    by_year_sec.to_csv(METRICS_DIR / "nominal_vs_real_comparison.csv", index=False)
    print(f"  -> Saved: outputs/metrics/nominal_vs_real_comparison.csv ({len(by_year_sec):,} rows)")

    flipped = by_year_sec[by_year_sec["flipped"]]
    print(f"  -> {len(flipped)} (year, sector) combinations flipped direction after deflation")
    if len(flipped) > 0:
        print("\n  Examples of flipped sectors (top 5 by inflation gap):")
        flipped_sorted = flipped.assign(
            gap=lambda d: (d["growth_rate"] - d["real_growth_rate"]).abs()
        ).sort_values("gap", ascending=False).head(5)
        for _, row in flipped_sorted.iterrows():
            print(
                f"    {int(row['year'])} | {row['primary_sector'][:40]:<40} "
                f"| nominal {row['growth_rate']:+.1%} → real {row['real_growth_rate']:+.1%} "
                f"({row['direction_change']})"
            )

    # Plot: nominal vs real for 2022 (highest inflation year)
    df_2022 = by_year_sec[by_year_sec["year"] == 2022].sort_values("growth_rate")
    if len(df_2022) > 0:
        fig, ax = plt.subplots(figsize=(13, 7))
        x = np.arange(len(df_2022))
        width = 0.4
        bars_nom = ax.barh(x - width / 2, df_2022["growth_rate"] * 100, width, color="steelblue", label="Nominal")
        bars_real = ax.barh(x + width / 2, df_2022["real_growth_rate"] * 100, width, color="orange", label="Real (deflated)")
        ax.set_yticks(x)
        ax.set_yticklabels(df_2022["primary_sector"], fontsize=8)
        ax.set_xlabel("Mean growth rate %")
        ax.set_title("Nominal vs Real Growth by Sector — 2022 (peak inflation ~11.6%)")
        ax.axvline(0, color="black", lw=0.8)
        ax.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "nominal_vs_real_2022.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  -> Saved: outputs/plots/nominal_vs_real_2022.png")

    return by_year_sec


# ── Save metrics summary ──────────────────────────────────────────────────────


def save_metrics_summary(tracks: dict) -> None:
    rows = []
    for track_name, track in tracks.items():
        for model_name, m in track["cls_metrics"].items():
            rows.append({
                "track": track_name,
                "model": model_name,
                "kind": "classifier",
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "kappa": m["kappa"],
                "metric_R2": None,
            })
        for model_name, m in track["reg_metrics"].items():
            rows.append({
                "track": track_name,
                "model": model_name,
                "kind": "regressor",
                "accuracy": None,
                "f1_macro": None,
                "kappa": None,
                "metric_R2": m["R²"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_DIR / "model_summary_part2.csv", index=False)
    print(f"\n  -> Saved: outputs/metrics/model_summary_part2.csv ({len(df)} rows)")


# ── Inflation timeline plot ───────────────────────────────────────────────────


def plot_inflation_timeline() -> None:
    cpi = load_cpi()
    cpi["date"] = pd.to_datetime(cpi[["year", "month"]].assign(day=1))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cpi["date"], cpi["cpi_yoy_pct"], color="darkred", lw=2)
    ax.axhline(0, color="black", lw=0.6)
    ax.axhspan(8, 16, alpha=0.15, color="red", label="Inflation shock (>8%)")
    ax.set_title("Kosovo CPI YoY — 2019 to 2025")
    ax.set_xlabel("Date")
    ax.set_ylabel("CPI YoY %")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "inflation_kosovo_2019_2025.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved: outputs/plots/inflation_kosovo_2019_2025.png")


def plot_covid_timeline() -> None:
    covid = load_covid()
    covid["date"] = pd.to_datetime(covid[["year", "month"]].assign(day=1))
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(covid["date"], covid["stringency_index"], color="purple", lw=2)
    axes[0].set_ylabel("Stringency 0-100")
    axes[0].set_title("Kosovo COVID-19 Indicators — 2019 to 2025")
    axes[1].plot(covid["date"], covid["cases_per_100k"], color="orange", lw=2)
    axes[1].set_ylabel("Cases per 100k")
    axes[2].plot(covid["date"], covid["vaccination_rate"], color="green", lw=2)
    axes[2].set_ylabel("Vaccination rate %")
    axes[2].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "covid_stringency_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  -> Saved: outputs/plots/covid_stringency_timeline.png")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(argv=None):
    parser = argparse.ArgumentParser(description="Part 2 — Real Growth (Inflation + COVID)")
    parser.add_argument(
        "--step",
        choices=["all", "deflate", "train", "counterfactual", "forecast", "compare", "plots"],
        default="all",
    )
    args = parser.parse_args(argv)

    # Always build augmented dataset
    growth_df = build_augmented_dataset()

    if args.step in ("all", "plots"):
        plot_inflation_timeline()
        plot_covid_timeline()

    if args.step == "deflate":
        print("\nDeflate step complete (growth_df augmented in memory).")
        return

    tracks = {}
    if args.step in ("all", "train"):
        # Track 1: Nominal (Part 1 reproduction)
        tracks["M_nominal"] = train_track(
            growth_df,
            target_class_col="growth_class",
            target_reg_col="growth_rate",
            track_name="M_nominal",
        )
        # Track 2: Real (Fisher deflated)
        tracks["M_real"] = train_track(
            growth_df,
            target_class_col="real_growth_class",
            target_reg_col="real_growth_rate",
            track_name="M_real",
        )
        # Track 3: Real excluding COVID years (for counterfactual)
        tracks["M_real_no_covid"] = train_track(
            growth_df,
            target_class_col="real_growth_class",
            target_reg_col="real_growth_rate",
            track_name="M_real_no_covid",
            train_filter=lambda d: ~d["year"].isin([2020, 2021]),
        )
        save_metrics_summary(tracks)

    if args.step in ("all", "counterfactual"):
        if "M_real_no_covid" not in tracks:
            print("Skipping counterfactual: M_real_no_covid not trained in this run.")
        else:
            run_counterfactual(growth_df, tracks["M_real_no_covid"])

    if args.step in ("all", "forecast"):
        if "M_real" not in tracks:
            print("Skipping forecast: M_real not trained in this run.")
        else:
            forecast_2027(growth_df, tracks["M_real"])

    if args.step in ("all", "compare"):
        compare_nominal_vs_real(growth_df)

    print("\n" + "=" * 60)
    print(f"  Part 2 complete. Outputs in {PART2_DIR / 'outputs'}")
    print("=" * 60)


if __name__ == "__main__":
    main(sys.argv[1:])
