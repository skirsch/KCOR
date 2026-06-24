# Executables

This guide lists the main KCOR executable entry points and their usual inputs/outputs.

## Root Makefile

Path:

- `Makefile`

Typical commands:

```bash
make KCOR DATASET=Czech
make monte_carlo DATASET=Czech MC_ITERATIONS=200 MC_ENROLLMENT_DATE=2021_24
make KCOR_variable DATASET=Czech
```

Notes:

- Prefer WSL for Makefile targets because several recipes assume Unix tools.
- `DATASET` defaults to `Czech`.

## code/KCOR_CMR.py

Purpose:

- Builds cohort mortality-rate workbooks from standardized records.
- In Monte Carlo mode, creates bootstrap CMR input workbooks for KCOR.

Typical command:

```bash
python3 code/KCOR_CMR.py data/Czech/records.csv data/Czech/KCOR_CMR.xlsx
```

Monte Carlo command:

```bash
PYTHONIOENCODING=utf-8 MONTE_CARLO=1 MC_ITERATIONS=200 MC_THREADS=8 \
MC_ENROLLMENT_DATE=2021_24 MC_YOB_START=1940 MC_YOB_END=1949 \
MC_INCLUDE_AGGREGATES=0 DATASET=Czech \
python3 code/KCOR_CMR.py data/Czech/records.csv data/Czech/KCOR_CMR_MC.xlsx
```

Key inputs:

- `data/<DATASET>/records.csv`
- `data/<DATASET>/<DATASET>.yaml`

Key outputs:

- `data/<DATASET>/KCOR_CMR.xlsx`
- `data/<DATASET>/KCOR_CMR_MC.xlsx` in Monte Carlo mode

Important environment variables:

- `MONTE_CARLO=1`
- `MC_ITERATIONS`
- `MC_THREADS`
- `MC_ENROLLMENT_DATE`
- `MC_YOB_START`
- `MC_YOB_END`
- `MC_INCLUDE_AGGREGATES`
- `DATASET`

## code/KCOR.py

Purpose:

- Computes KCOR dose-pair curves and summaries from a CMR workbook.
- In Monte Carlo mode, refits normalization per replicate and summarizes KCOR distributions.

Typical command:

```bash
python3 code/KCOR.py data/Czech/KCOR_CMR.xlsx data/Czech/KCOR.xlsx "Primary Analysis" KCOR_summary.log
```

Monte Carlo command:

```bash
PYTHONIOENCODING=utf-8 MONTE_CARLO=1 MC_ITERATIONS=200 \
MC_ENROLLMENT_DATE=2021_24 MC_YOB_START=1940 MC_YOB_END=1949 \
MC_INCLUDE_AGGREGATES=0 DATASET=Czech \
python3 code/KCOR.py data/Czech/KCOR_CMR_MC.xlsx data/Czech/KCOR_MC.xlsx \
"Monte Carlo Analysis" KCOR_MC_summary.log
```

Key inputs:

- `data/<DATASET>/KCOR_CMR.xlsx` or `data/<DATASET>/KCOR_CMR_MC.xlsx`
- `data/<DATASET>/<DATASET>.yaml`

Key outputs:

- `data/<DATASET>/KCOR.xlsx`
- `data/<DATASET>/KCOR_summary.log`
- `data/<DATASET>/KCOR_summary.xlsx`
- `data/<DATASET>/KCOR_MC.xlsx` in Monte Carlo mode
- `data/<DATASET>/KCOR_MC_summary.log` in Monte Carlo mode

Important notes:

- Iteration `0` in Monte Carlo mode is the original-data point estimate.
- Bootstrap CI percentiles should generally be computed from resample iterations only.
- With default `AGE_RANGE=10`, raw birth years `1940-1949` are output as `YearOfBirth=1940`.

## code/KCOR_variable.py

Purpose:

- Builds variable-cohort aggregation outputs.

Typical command:

```bash
python3 code/KCOR_variable.py data/Czech/records.csv data/Czech/KCOR_variable.xlsx
```

Key outputs:

- `data/<DATASET>/KCOR_variable.xlsx`

## Mortality analysis scripts

Paths:

- `code/kcor_mortality.py`
- `code/kcor_mortality_stats.py`
- `code/kcor_mortality_plots.py`
- `code/kcor_mortality_sensitivity.py`
- `code/kcor_mortality_age_stratified.py`

Purpose:

- Run alternate mortality analyses, statistical summaries, plots, sensitivity grids, and age-stratified outputs.

Typical Makefile entry points:

```bash
make mortality
make mortality_stats
make mortality_plots
make mortality_sensitivity
make mortality_age
```

Key outputs:

- `data/Czech2/kcor_mortality_output`
