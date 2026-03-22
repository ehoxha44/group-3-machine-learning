from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    input_path: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    ingest_dir: Path = field(init=False)
    profile_dir: Path = field(init=False)
    profile_before_dir: Path = field(init=False)
    profile_after_dir: Path = field(init=False)
    clean_dir: Path = field(init=False)
    aggregate_dir: Path = field(init=False)
    sample_dir: Path = field(init=False)
    outliers_dir: Path = field(init=False)
    imbalance_dir: Path = field(init=False)
    report_dir: Path = field(init=False)
    header_row_index: int = 8
    random_seed: int = 42
    year_min: int = 2019
    year_max: int = 2025
    outlier_iqr_multiplier: float = 1.5
    outlier_zscore_threshold: float = 3.0
    critical_columns: tuple[str, ...] = (
        "year",
        "month",
        "primary_sector",
        "municipality",
        "registration_status",
        "num_taxpayers",
        "turnover_eur",
    )
    categorical_fill_value: str = "Unknown"

    def __post_init__(self) -> None:
        self.input_path = self.project_root / "dataset" / "Qarkullimi.xlsx"
        self.outputs_dir = self.project_root / "outputs"
        self.ingest_dir = self.outputs_dir / "ingest"
        self.profile_dir = self.outputs_dir / "profile"
        self.profile_before_dir = self.profile_dir / "before"
        self.profile_after_dir = self.profile_dir / "after"
        self.clean_dir = self.outputs_dir / "clean"
        self.aggregate_dir = self.outputs_dir / "aggregate"
        self.sample_dir = self.outputs_dir / "sample"
        self.outliers_dir = self.outputs_dir / "outliers"
        self.imbalance_dir = self.outputs_dir / "imbalance"
        self.report_dir = self.outputs_dir / "report"

    def ensure_output_dirs(self) -> None:
        for path in (
            self.outputs_dir,
            self.ingest_dir,
            self.profile_dir,
            self.profile_before_dir,
            self.profile_after_dir,
            self.clean_dir,
            self.aggregate_dir,
            self.sample_dir,
            self.outliers_dir,
            self.imbalance_dir,
            self.report_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = PipelineConfig()
