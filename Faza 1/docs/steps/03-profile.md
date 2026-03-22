# Step 03: Profile After Cleaning

## Goal

Measure data types, completeness, nulls, uniqueness, and quality indicators after ingestion and cleaning.

## Command

```bash
python3 -m src.main --step profile
```

## Implementation Steps

1. Read `outputs/clean/strict_cleaned.csv`.
2. Limit profiling to the business columns used in the canonical dataset.
3. Infer the semantic type of each column:
   - numeric
   - categorical
   - identifier-like
   - mixed
   - empty
4. Count nulls and non-nulls per column.
5. Count unique values per column.
6. Build overall quality metrics:
   - total rows
   - total columns
   - complete rows
   - rows with any null
   - duplicate rows
   - invalid values by rule
7. Build a readable status message for the whole dataset.
8. Document the missing-value strategy in a separate summary table.
9. Export all profile artifacts.

## Source Code

- `src/profiling.py`

## Output Files

- `outputs/profile/after/schema_report.csv`
- `outputs/profile/after/data_quality_report.csv`
- `outputs/profile/after/nulls_by_column.csv`
- `outputs/profile/after/null_strategy.csv`
- `outputs/profile/after/readability_status.txt`

## Why This Step Comes Third

Once the dataset is cleaned, profiling gives the evidence for data quality, readiness, and presentation.
