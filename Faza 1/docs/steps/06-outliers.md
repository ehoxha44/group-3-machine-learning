# Step 06: Outliers

## Goal

Detect extreme values without deleting them, so the dataset stays complete while still exposing risk areas.

## Command

```bash
python3 -m src.main --step outliers
```

## Implementation Steps

1. Read `outputs/clean/strict_cleaned.csv`.
2. Focus on the main numeric business columns:
   - `turnover_eur`
   - `num_taxpayers`
3. Compute IQR bounds for each numeric column.
4. Flag rows outside the IQR bounds.
5. Compute z-scores for each numeric column.
6. Flag rows above the configured z-score threshold.
7. Apply an additional log-IQR check for `turnover_eur` because financial data is often skewed.
8. Build a combined `is_any_outlier` flag.
9. Export the row-level outlier flags and a compact summary table.

## Source Code

- `src/outliers.py`

## Output Files

- `outputs/outliers/outlier_flags.csv`
- `outputs/outliers/outlier_summary.csv`

## Why This Step Comes Sixth

Outlier detection is meaningful only after cleaning and type normalization are already finished.
