# Step 07: Imbalance

## Goal

Measure class distribution for candidate categorical target columns before any synthetic balancing is attempted.

## Command

```bash
python3 -m src.main --step imbalance
```

## Implementation Steps

1. Read `outputs/clean/strict_cleaned.csv`.
2. Select candidate target columns:
   - `registration_status`
   - `municipality`
   - `primary_sector`
3. Count rows in each class.
4. Compute class share relative to the full column total.
5. Export one summary table that contains:
   - target column
   - class label
   - count
   - share

## Source Code

- `src/imbalance.py`

## Output Files

- `outputs/imbalance/class_distribution_summary.csv`

## Why This Step Comes Seventh

You must understand class imbalance before deciding whether SMOTE or ADASYN is justified.
