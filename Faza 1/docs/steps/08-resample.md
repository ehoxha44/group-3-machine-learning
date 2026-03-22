# Step 08: Resample

## Goal

Optionally generate a balanced training dataset with SMOTE or ADASYN after a target column has been explicitly chosen.

## Command

SMOTE example:

```bash
python3 -m src.main --step resample --target-column registration_status --algorithm smote
```

ADASYN example:

```bash
python3 -m src.main --step resample --target-column registration_status --algorithm adasyn
```

## Implementation Steps

1. Read `outputs/clean/model_ready.csv`.
2. Require an explicit `--target-column`.
3. Validate resampling conditions:
   - target column exists
   - at least two classes exist
   - each class has at least two rows
   - numeric feature matrix exists
4. Split the dataset into:
   - target vector
   - numeric feature matrix
5. Choose the resampling algorithm:
   - `SMOTE`
   - `ADASYN`
6. Fit the sampler and generate synthetic minority examples.
7. Rebuild the balanced dataset.
8. Compare class counts before and after resampling.
9. Export the balanced dataset and the before/after summary.

## Source Code

- `src/imbalance.py`

## Output Files

- `outputs/imbalance/resampled_<algorithm>_<target>.csv`
- `outputs/imbalance/resampled_<algorithm>_<target>_summary.csv`

## Why This Step Is Optional

This is a downstream modeling step, not part of the canonical cleaning pipeline. It should only be used after a target variable is decided.
