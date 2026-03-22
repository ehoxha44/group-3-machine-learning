# Step 05: Sample

## Goal

Create controlled samples and predefined subsets for exploration, experiments, and presentation.

## Command

Default sample:

```bash
python3 -m src.main --step sample --sample-size 1200
```

Stratified sample:

```bash
python3 -m src.main --step sample --sample-size 5000 --stratify-by municipality
```

Explicit equal sample from each year+month:

```bash
python3 -m src.main --step sample --sample-size 1200 --equal-by-period
```

## Implementation Steps

1. Read `outputs/clean/model_ready.csv`.
2. Choose the sampling mode:
   - equal by `year_month` by default
   - stratified if `--stratify-by` is provided
3. If stratified:
   - group by the selected column
   - sample proportionally from each group
   - cap the final result at the requested sample size
4. If equal by `year_month`:
   - group by the derived `year_month` field
   - assign roughly the same sample count to each period
   - if some periods have too few rows, fill the remaining sample from the leftover pool
5. Save sampling metadata:
   - method
   - sample size
   - random seed
   - stratification column
6. Build predefined subset views from the sample:
   - sector-focused
   - municipality-focused
   - registration-status-focused
   - monthly time-pattern
7. Export the sample and all subsets.

## Source Code

- `src/sampling.py`

## Output Files

- `outputs/sample/equal_year_month_sample.csv`
- `outputs/sample/equal_year_month_sample_metadata.json`
- `outputs/sample/sector_focus.csv`
- `outputs/sample/municipality_focus.csv`
- `outputs/sample/registration_status_focus.csv`
- `outputs/sample/monthly_time_pattern.csv`

## Why This Step Comes Fifth

Sampling and subset selection are secondary analytical views that depend on a clean, model-ready dataset.
