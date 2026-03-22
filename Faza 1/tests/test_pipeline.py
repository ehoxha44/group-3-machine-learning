from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.aggregation import build_aggregations
from src.cleaning import clean_dataframe
from src.config import PipelineConfig
from src.imbalance import validate_resampling_inputs
from src.ingest import normalize_schema, read_raw_excel
from src.outliers import detect_outliers
from src.profiling import build_profile_bundle
from src.sampling import equal_period_sample


RAW_COLUMNS = {
    "Unnamed: 0": [None, None, None],
    "Viti\nGodina\nYear": [2025, 2025, "Year"],
    "Muaji\nMesec\nMonth": [1, 13, "Month"],
    "Përshkrimi i Sektorit (Kryesor)\nOpis (Glavnog) Sektora\nDescription of (Primary) Sector": [
        " Bujqesia ",
        "Tregtia",
        "Description of (Primary) Sector",
    ],
    "Unnamed: 4": [None, None, None],
    "Unnamed: 5": [None, None, None],
    "Komuna\nOpština\nMunicipality ": ["Pejë", "Prishtinë", "Municipality"],
    "Statusi i Regjistrimit\nStatus Registracije\nRegistration Status": [
        "Sh.P.K.",
        "Biznes Individual",
        "Registration Status",
    ],
    "Unnamed: 8": [None, None, None],
    "Numri i Tatimpaguesve\nBroj Poreskih Obveznika \nNumber of Taxpayers": [5, -1, "Number of Taxpayers"],
    "Qarkullimi në Euro\nPromet u evrima\nTurnover in Euro ": [100.0, -50.0, "Turnover in Euro"],
}


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = PipelineConfig(project_root=Path.cwd())

    def test_normalize_schema_drops_empty_columns(self) -> None:
        raw_df = pd.DataFrame(RAW_COLUMNS)
        normalized = normalize_schema(raw_df)
        self.assertEqual(
            normalized.columns.tolist(),
            [
                "year",
                "month",
                "primary_sector",
                "municipality",
                "registration_status",
                "num_taxpayers",
                "turnover_eur",
            ],
        )

    def test_read_raw_excel_uses_header_row_nine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "dataset" / "Qarkullimi.xlsx"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            filler = pd.DataFrame([[""] * len(RAW_COLUMNS) for _ in range(8)])
            data = pd.DataFrame(RAW_COLUMNS)
            with pd.ExcelWriter(input_path, engine="openpyxl") as writer:
                filler.to_excel(writer, index=False, header=False)
                data.to_excel(writer, index=False, startrow=8)
            config = PipelineConfig(project_root=tmp_path)
            loaded = read_raw_excel(config)
            self.assertIn("Viti\nGodina\nYear", loaded.columns)
            self.assertNotIn("Unnamed: 0", loaded.columns)
            self.assertNotIn("Unnamed: 4", loaded.columns)
            self.assertNotIn("Unnamed: 5", loaded.columns)
            self.assertNotIn("Unnamed: 8", loaded.columns)
            self.assertEqual(len(loaded), 3)

    def test_raw_profile_skips_non_data_columns(self) -> None:
        raw_df = pd.DataFrame(RAW_COLUMNS).drop(columns=["Unnamed: 0", "Unnamed: 4", "Unnamed: 5", "Unnamed: 8"])
        bundle = build_profile_bundle(
            raw_df,
            self.config,
            phase="before",
            restrict_schema_to_business_columns=False,
        )
        schema_columns = bundle["schema_report"]["column"].tolist()
        self.assertNotIn("Unnamed: 0", schema_columns)
        self.assertNotIn("Unnamed: 4", schema_columns)
        self.assertNotIn("Unnamed: 5", schema_columns)
        self.assertNotIn("Unnamed: 8", schema_columns)

    def test_cleaning_flags_invalid_values_and_removes_repeated_header(self) -> None:
        raw_df = normalize_schema(pd.DataFrame(RAW_COLUMNS))
        result = clean_dataframe(raw_df, self.config)
        self.assertEqual(len(result.strict_clean_df), 2)
        self.assertEqual(result.strict_clean_df["municipality"].tolist(), ["PEJË", "PRISHTINË"])
        invalid_rows = result.invalid_rows
        self.assertEqual(len(invalid_rows), 1)
        self.assertIn("invalid_month", invalid_rows["invalid_reason"].iloc[0])
        self.assertIn("invalid_turnover", invalid_rows["invalid_reason"].iloc[0])

    def test_aggregation_outputs_consistent_sums(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2025, 2025, 2024],
                "month": [1, 2, 1],
                "primary_sector": ["A", "A", "B"],
                "municipality": ["X", "Y", "X"],
                "registration_status": ["SHPK", "SHPK", "BI"],
                "num_taxpayers": [1, 2, 3],
                "turnover_eur": [10.0, 20.0, 30.0],
            }
        )
        tables = build_aggregations(df)
        by_year = tables["turnover_by_year"].set_index("year")
        self.assertEqual(by_year.loc[2025, "turnover_sum"], 30.0)
        self.assertEqual(by_year.loc[2024, "turnover_sum"], 30.0)

    def test_outlier_flags_are_created(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2025] * 5,
                "month": [1] * 5,
                "primary_sector": ["A"] * 5,
                "municipality": ["X"] * 5,
                "registration_status": ["SHPK"] * 5,
                "num_taxpayers": [1, 1, 1, 1, 100],
                "turnover_eur": [10, 10, 11, 10, 1000],
            }
        )
        flagged = detect_outliers(df, self.config)
        self.assertIn("turnover_eur_outlier_iqr", flagged.columns)
        self.assertTrue(flagged["is_any_outlier"].iloc[-1])

    def test_resample_validator_requires_target(self) -> None:
        df = pd.DataFrame({"feature_a": [1.0, 2.0], "target": ["a", "b"]})
        with self.assertRaises(ValueError):
            validate_resampling_inputs(df, None)

    def test_equal_period_sample_balances_year_month_groups(self) -> None:
        df = pd.DataFrame(
            {
                "year": [2024] * 6 + [2025] * 6,
                "month": [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3,
                "year_month": ["2024-01"] * 3 + ["2024-02"] * 3 + ["2025-01"] * 3 + ["2025-02"] * 3,
                "turnover_eur": range(12),
                "num_taxpayers": [1] * 12,
                "municipality": ["X"] * 12,
                "primary_sector": ["A"] * 12,
                "registration_status": ["SHPK"] * 12,
            }
        )
        sampled = equal_period_sample(df, sample_size=8, seed=42)
        counts = sampled["year_month"].value_counts().to_dict()
        self.assertEqual(counts, {"2024-01": 2, "2024-02": 2, "2025-01": 2, "2025-02": 2})


if __name__ == "__main__":
    unittest.main()
