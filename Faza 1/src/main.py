from __future__ import annotations

import argparse

from .config import DEFAULT_CONFIG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qarkullimi cleaning and preprocessing CLI")
    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "profile_raw",
            "ingest",
            "clean",
            "profile",
            "aggregate",
            "sample",
            "outliers",
            "imbalance",
            "resample",
            "report",
            "all",
        ],
    )
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--stratify-by", default=None)
    parser.add_argument("--equal-by-period", action="store_true")
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--algorithm", default="smote", choices=["smote", "adasyn"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = DEFAULT_CONFIG

    if args.step == "profile_raw":
        from .profiling import run_profile_raw

        print(run_profile_raw(config))
    elif args.step == "ingest":
        from .ingest import run_ingest

        print(run_ingest(config))
    elif args.step == "clean":
        from .cleaning import run_clean

        print(run_clean(config))
    elif args.step == "profile":
        from .profiling import run_profile

        print(run_profile(config))
    elif args.step == "aggregate":
        from .aggregation import run_aggregate

        print(run_aggregate(config))
    elif args.step == "sample":
        from .sampling import run_sample

        print(
            run_sample(
                config,
                sample_size=args.sample_size,
                stratify_by=args.stratify_by,
                equal_by_period=args.equal_by_period,
            )
        )
    elif args.step == "outliers":
        from .outliers import run_outliers

        print(run_outliers(config))
    elif args.step == "imbalance":
        from .imbalance import run_imbalance

        print(run_imbalance(config))
    elif args.step == "resample":
        from .imbalance import run_resample

        if not args.target_column:
            raise SystemExit("--target-column is required for --step resample")
        print(run_resample(config, target_column=args.target_column, algorithm=args.algorithm))
    elif args.step == "report":
        from .reporting import generate_report

        print(generate_report(config))
    elif args.step == "all":
        from .aggregation import run_aggregate
        from .cleaning import run_clean
        from .imbalance import run_imbalance
        from .ingest import run_ingest
        from .outliers import run_outliers
        from .profiling import run_profile, run_profile_raw
        from .reporting import generate_report
        from .sampling import run_sample

        print(run_profile_raw(config))
        print(run_ingest(config))
        print(run_clean(config))
        print(run_profile(config))
        print(run_aggregate(config))
        print(
            run_sample(
                config,
                sample_size=args.sample_size,
                stratify_by=args.stratify_by,
                equal_by_period=args.equal_by_period,
            )
        )
        print(run_outliers(config))
        print(run_imbalance(config))
        print(generate_report(config))


if __name__ == "__main__":
    main()
