# Qarkullimi Data Cleaning Project

This project implements a script-first Python pipeline for cleaning and preprocessing `dataset/Qarkullimi.xlsx`.

## Dataset Context

- Main input: `dataset/Qarkullimi.xlsx`
- Header row in Excel: row `9`
- Canonical v1 scope: only the main dataset
- Future extension: yearly files in `2019-2025/`

## Project Structure

- `src/`: pipeline modules and CLI
- `tests/`: unit tests
- `outputs/`: generated artifacts
- `docs/steps/`: step-by-step implementation READMEs
- `data/`: reserved for future staged datasets
- `notebooks/`: reserved for optional presentation notebooks

## Pipeline Order

The pipeline is designed to be understood and executed in this order:

1. `profile_raw`
2. `ingest`
3. `clean`
4. `profile`
5. `aggregate`
6. `sample`
7. `outliers`
8. `imbalance`
9. `resample` (optional, only when a target column is chosen)
10. `report`

## Step-by-Step READMEs

- [Step 00: Raw Profile](docs/steps/00-profile-raw.md)
- [Step 01: Ingest](docs/steps/01-ingest.md)
- [Step 02: Clean](docs/steps/02-clean.md)
- [Step 03: Profile After Cleaning](docs/steps/03-profile.md)
- [Step 04: Aggregate](docs/steps/04-aggregate.md)
- [Step 05: Sample](docs/steps/05-sample.md)
- [Step 06: Outliers](docs/steps/06-outliers.md)
- [Step 07: Imbalance](docs/steps/07-imbalance.md)
- [Step 08: Resample](docs/steps/08-resample.md)
- [Step 09: Report](docs/steps/09-report.md)

## Main Commands

Run the full pipeline:

```bash
python3 -m src.main --step all
```

Run steps individually:

```bash
python3 -m src.main --step profile_raw
python3 -m src.main --step ingest
python3 -m src.main --step clean
python3 -m src.main --step profile
python3 -m src.main --step aggregate
python3 -m src.main --step sample --sample-size 1200
python3 -m src.main --step sample --sample-size 5000 --stratify-by municipality
python3 -m src.main --step outliers
python3 -m src.main --step imbalance
python3 -m src.main --step report
python3 -m src.main --step resample --target-column registration_status --algorithm smote
```

## Produced Outputs

- `outputs/profile/before/schema_report.csv`
- `outputs/profile/before/data_quality_report.csv`
- `outputs/profile/before/nulls_by_column.csv`
- `outputs/profile/before/null_strategy.csv`
- `outputs/profile/before/readability_status.txt`
- `outputs/ingest/raw_extract.csv`
- `outputs/ingest/normalized.csv`
- `outputs/clean/strict_cleaned.csv`
- `outputs/clean/model_ready.csv`
- `outputs/clean/cleaning_log.csv`
- `outputs/clean/invalid_rows.csv`
- `outputs/profile/after/schema_report.csv`
- `outputs/profile/after/data_quality_report.csv`
- `outputs/profile/after/nulls_by_column.csv`
- `outputs/profile/after/null_strategy.csv`
- `outputs/profile/after/readability_status.txt`
- `outputs/aggregate/*.csv`
- `outputs/sample/*.csv`
- `outputs/outliers/*.csv`
- `outputs/imbalance/*.csv`
- `outputs/report/report.md`
- `outputs/report/*.png`

## Notes

- The main canonical columns are `year`, `month`, `primary_sector`, `municipality`, `registration_status`, `num_taxpayers`, and `turnover_eur`.
- Raw profiling is executed first on the initial dataset before schema normalization and cleaning.
- Excel columns `A`, `E`, `F`, and `I` are skipped because they do not contain business data.
- Blank spacer columns from Excel are removed during ingestion.
- Sampling defaults to balanced selection across `year_month` unless `--stratify-by` is provided.
- SMOTE and ADASYN are optional downstream steps and do not modify the canonical cleaned dataset.
- The report step generates presentation charts for turnover trends, municipality/sector comparisons, null counts, outlier summary, and registration-status distribution.
