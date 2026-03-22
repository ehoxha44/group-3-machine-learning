from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_dataframe(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=index)
    elif path.suffix.lower() == ".xlsx":
        df.to_excel(path, index=index)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=index)
    else:
        raise ValueError(f"Unsupported output format for {path}")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
