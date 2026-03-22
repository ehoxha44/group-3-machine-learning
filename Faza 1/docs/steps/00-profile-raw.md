# Step 00: Raw Profile

## Goal

Profile the initial dataset before schema normalization, ingestion outputs, and cleaning. This gives a true baseline of the raw Excel file as it was received.

## Command

```bash
python3 -m src.main --step profile_raw
```

## Implementation Steps

1. Read `dataset/Qarkullimi.xlsx` directly with header row `9`.
2. Skip Excel columns `A`, `E`, `F`, and `I` because they do not contain business data.
3. Keep the remaining raw column structure exactly as loaded.
4. Build a schema report on the raw dataset exactly as loaded.
5. Build a business-quality view by mapping the raw multilingual headers to canonical fields in memory.
6. Measure:
   - total rows
   - total columns
   - complete rows
   - rows with any null
   - duplicate rows
   - invalid year/month/taxpayer/turnover values
7. Build per-column null counts and null rates.
8. Save the baseline readability status for the raw dataset.
9. Export the raw-profile artifacts to `outputs/profile/before/`.

## Source Code

- `src/profiling.py`
- `src/ingest.py`
- `src/constants.py`

## Output Files

- `outputs/profile/before/schema_report.csv`
- `outputs/profile/before/data_quality_report.csv`
- `outputs/profile/before/nulls_by_column.csv`
- `outputs/profile/before/null_strategy.csv`
- `outputs/profile/before/readability_status.txt`

## Why This Step Comes First

This is the baseline snapshot of the dataset before any normalization, deduplication, cleaning, or feature engineering changes are applied.
