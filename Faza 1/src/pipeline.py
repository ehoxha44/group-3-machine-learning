from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .ingest import run_ingest


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_normalized_data(config: PipelineConfig) -> pd.DataFrame:
    normalized_path = config.ingest_dir / "normalized.csv"
    if not normalized_path.exists():
        run_ingest(config)
    return _read_csv(normalized_path)


def load_cleaned_data(config: PipelineConfig, *, model_ready: bool) -> pd.DataFrame:
    filename = "model_ready.csv" if model_ready else "strict_cleaned.csv"
    path = config.clean_dir / filename
    if not path.exists():
        from .cleaning import run_clean

        run_clean(config)
    return _read_csv(path)
