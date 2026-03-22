# Step 02: Clean

## Goal

Apply the main data cleaning rules, preserve a strict cleaned dataset, and also create a model-ready dataset with explicit imputations and derived fields.

## Command

```bash
python3 -m src.main --step clean
```

## Implementation Steps

1. Read `outputs/ingest/normalized.csv`.
2. Standardize blank strings and whitespace-only values to null.
3. Remove rows that are fully empty across business-critical columns.
4. Remove repeated header rows if they appear inside the data.
5. Coerce `year`, `month`, `num_taxpayers`, and `turnover_eur` to numeric values.
6. Standardize text columns:
   - municipality to uppercase
   - registration status to uppercase
   - sector text to trimmed normalized text
7. Remove duplicate rows.
8. Validate business rules:
   - `year` between `2019` and `2025`
   - `month` between `1` and `12`
   - `num_taxpayers >= 0`
   - `turnover_eur >= 0`
9. Store invalid row reasons in `invalid_reason`.
10. Export the strict cleaned dataset without forced imputation.
11. Build a model-ready dataset:
   - numeric nulls filled with median
   - categorical nulls filled with `Unknown`
   - imputation flag columns added
   - `year_month` added
   - `turnover_eur_log1p` added
12. Save a cleaning log with all major cleaning actions and counts.

## Source Code

- `src/cleaning.py`
- `src/pipeline.py`

## Output Files

- `outputs/clean/strict_cleaned.csv`
- `outputs/clean/model_ready.csv`
- `outputs/clean/cleaning_log.csv`
- `outputs/clean/invalid_rows.csv`

## Why This Step Comes Second

Profiling, aggregation, outlier detection, and modeling all need a cleaned and consistent dataset first.
