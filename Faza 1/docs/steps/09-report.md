# Step 09: Report

## Goal

Generate final presentation-ready outputs, including a narrative Markdown report and chart images.

## Command

```bash
python3 -m src.main --step report
```

## Implementation Steps

1. Read `outputs/clean/strict_cleaned.csv`.
2. Rebuild profile summaries for the report context.
3. Rebuild aggregation tables used by the report.
4. Recompute outlier summaries.
5. Recompute class-distribution summaries.
6. Create chart images:
   - turnover by year
   - top municipalities by turnover
7. Build a Markdown report that includes:
   - data types
   - quality metrics
   - null/completeness overview
   - missing-value strategy
   - outlier findings
   - class-distribution summary
   - readiness for modeling
8. Export the report and all chart-ready tables.

## Source Code

- `src/reporting.py`

## Output Files

- `outputs/report/report.md`
- `outputs/report/turnover_by_year.png`
- `outputs/report/turnover_by_municipality_top20.png`
- `outputs/report/class_distribution_summary.csv`
- `outputs/report/outlier_summary.csv`
- `outputs/report/turnover_by_year.csv`
- `outputs/report/turnover_by_month.csv`
- `outputs/report/turnover_by_municipality.csv`
- `outputs/report/turnover_by_sector.csv`
- `outputs/report/turnover_by_registration_status.csv`

## Why This Step Comes Last

The report depends on the outputs of cleaning, profiling, aggregation, imbalance analysis, and outlier detection.
