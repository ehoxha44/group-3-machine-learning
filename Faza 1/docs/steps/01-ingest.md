# Step 01: Ingest

## Goal

Read the Excel file correctly, start from header row `9`, remove empty helper columns, and create a normalized raw dataset for the rest of the pipeline.

## Command

```bash
python3 -m src.main --step ingest
```

## Implementation Steps

1. Load `dataset/Qarkullimi.xlsx` with `pandas.read_excel(..., header=8)`.
2. Treat Excel row `9` as the real header row.
3. Explicitly skip Excel columns `A`, `E`, `F`, and `I`.
4. Detect any remaining columns that are completely empty, such as spacer columns.
5. Drop those empty columns.
6. Rename multilingual Excel headers into canonical machine-readable names.
7. Export the extracted table to `outputs/ingest/raw_extract.csv`.
8. Export the normalized table to `outputs/ingest/normalized.csv`.

## Canonical Columns

- `year`
- `month`
- `primary_sector`
- `municipality`
- `registration_status`
- `num_taxpayers`
- `turnover_eur`

## Source Code

- `src/ingest.py`
- `src/constants.py`

## Output Files

- `outputs/ingest/raw_extract.csv`
- `outputs/ingest/normalized.csv`

## Why This Step Comes After Raw Profiling

Raw profiling should capture the initial dataset exactly as received. After that, ingestion creates the stable schema needed by all later steps.
