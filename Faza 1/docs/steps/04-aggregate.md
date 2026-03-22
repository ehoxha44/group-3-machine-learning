# Step 04: Aggregate

## Goal

Create grouped summary tables for presentation and higher-level analysis.

## Command

```bash
python3 -m src.main --step aggregate
```

## Implementation Steps

1. Read `outputs/clean/strict_cleaned.csv`.
2. Group by the required dimensions:
   - year
   - month
   - municipality
   - primary sector
   - registration status
3. Compute summary statistics for each group:
   - turnover sum
   - turnover mean
   - turnover median
   - turnover min
   - turnover max
   - taxpayers sum
   - taxpayers mean
   - row count
4. Sort the results by `turnover_sum` where useful.
5. Export each grouped table as a separate CSV.

## Source Code

- `src/aggregation.py`

## Output Files

- `outputs/aggregate/turnover_by_year.csv`
- `outputs/aggregate/turnover_by_month.csv`
- `outputs/aggregate/turnover_by_municipality.csv`
- `outputs/aggregate/turnover_by_sector.csv`
- `outputs/aggregate/turnover_by_registration_status.csv`

## Why This Step Comes Fourth

Aggregations depend on cleaned values and are used later in the presentation report.
