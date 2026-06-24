# Work Log

## 2026-06-01 - KCOR Monte Carlo CI for 2021_24 / 1940-1949

### What we did

- Updated the Monte Carlo pipeline so it can restrict the source cohort by raw birth-year range.
- Fixed Monte Carlo mode so `MC_ENROLLMENT_DATE` is not overwritten by YAML enrollment dates.
- Ran a 200-iteration full-pipeline bootstrap for enrollment `2021_24`, raw birth years `1940-1949`, aggregate rows off.
- KCOR refit the current KCOR7 gamma/theta normalization inside the Monte Carlo pipeline.

### Command / executable

```bash
PYTHONIOENCODING=utf-8 MONTE_CARLO=1 MC_ITERATIONS=200 MC_THREADS=8 \
MC_ENROLLMENT_DATE=2021_24 MC_YOB_START=1940 MC_YOB_END=1949 \
MC_INCLUDE_AGGREGATES=0 DATASET=Czech \
python3 code/KCOR_CMR.py data/Czech/records.csv data/Czech/KCOR_CMR_MC.xlsx

PYTHONIOENCODING=utf-8 MONTE_CARLO=1 MC_ITERATIONS=200 \
MC_ENROLLMENT_DATE=2021_24 MC_YOB_START=1940 MC_YOB_END=1949 \
MC_INCLUDE_AGGREGATES=0 DATASET=Czech \
python3 code/KCOR.py data/Czech/KCOR_CMR_MC.xlsx data/Czech/KCOR_MC.xlsx \
"Monte Carlo Analysis" KCOR_MC_summary.log
```

### Outputs

- `data/Czech/KCOR_CMR_MC.xlsx`
- `data/Czech/KCOR_MC.xlsx`
- `data/Czech/KCOR_MC_summary.log`
- `data/Czech/KCOR_summary.xlsx`

### Results

- Reporting date: `2022-06-13`.
- Output row `YearOfBirth=1940` represents raw birth years `1940-1949`.
- Focal `2 vs 0` result:
  - point estimate from original-data iteration `0`: `1.2234`;
  - percentile bootstrap CI from draws `1-199`: `[1.2104, 1.3748]`.
- The focal interval stayed above `1.0`.

### Next steps

- Use the workbooks/log above as the traceable source for this run.
- If a final submission needs higher precision, rerun with more bootstrap draws and record the command/results here.

## 2026-06-02 - Audited asymmetry of focal 2-vs-0 bootstrap CI

### What we did

- Re-extracted the `2021_24`, `YearOfBirth=1940`, `2 vs 0` reporting-date rows from `data/Czech/KCOR_MC.xlsx`.
- Checked where the original-data iteration `0` point estimate falls inside the bootstrap distribution from draws `1-199`.

### Command / executable

- Python extraction from `data/Czech/KCOR_MC.xlsx`, sheet `2021_24`.

### Outputs

- No new output artifact; results are from the existing Monte Carlo workbook.

### Results

- Reporting date: `2022-06-13`.
- Point estimate from iteration `0`: `1.2234055`.
- Bootstrap draws `1-199`: mean `1.28725`, median `1.28477`, percentile CI `[1.21037, 1.37480]`.
- The point estimate is at about the `6.5%` percentile of bootstrap draws, not literally at the `2.5%` lower bound, but low enough that the percentile interval looks presentation-sensitive.

### Next steps

- Before using this in a deck/submission, audit for point-estimate/bootstrap-path mismatch and consider a larger bootstrap run plus a bias-aware interval/reporting strategy.

## 2026-06-01 - KCOR MC CI asymmetry audit for 2021_24 / 1940-1949 / 2 vs 0

### What we did
- Added a Monte Carlo resampling hygiene fix in `code/KCOR_CMR.py`: iteration 0 and bootstrap samples now reset row indexes before downstream processing, avoiding duplicate-index artifacts after sampling with replacement.
- Re-ran focused full-pipeline MC audits for enrollment `2021_24`, raw birth years `1940-1949` (output bucket `YearOfBirth=1940`), aggregates disabled, at requested iteration counts 25, 100, 200, and 400.
- Compared percentile, basic bootstrap, and simple bias-corrected bootstrap intervals for the focal `2 vs 0` KCOR at `2022-06-13`.

### Command / executable
- CMR stage template:
  - `PYTHONIOENCODING=utf-8 MONTE_CARLO=1 MC_ITERATIONS=<N> MC_THREADS=8 MC_ENROLLMENT_DATE=2021_24 MC_YOB_START=1940 MC_YOB_END=1949 MC_INCLUDE_AGGREGATES=0 DATASET=Czech python3 code/KCOR_CMR.py data/Czech/records.csv data/Czech/KCOR_CMR_MC_audit_<N>.xlsx`
- KCOR stage template:
  - `PYTHONIOENCODING=utf-8 MONTE_CARLO=1 MC_ITERATIONS=<N> MC_ENROLLMENT_DATE=2021_24 MC_YOB_START=1940 MC_YOB_END=1949 MC_INCLUDE_AGGREGATES=0 DATASET=Czech python3 code/KCOR.py data/Czech/KCOR_CMR_MC_audit_<N>.xlsx data/Czech/KCOR_MC_audit_<N>.xlsx "Monte Carlo Audit n=<N>" KCOR_MC_audit_<N>_summary.log`

### Outputs
- `data/Czech/KCOR_CMR_MC_audit_25.xlsx`, `data/Czech/KCOR_MC_audit_25.xlsx`, `data/Czech/KCOR_MC_audit_25_summary.log`
- `data/Czech/KCOR_CMR_MC_audit_100.xlsx`, `data/Czech/KCOR_MC_audit_100.xlsx`, `data/Czech/KCOR_MC_audit_100_summary.log`
- `data/Czech/KCOR_CMR_MC_audit_200.xlsx`, `data/Czech/KCOR_MC_audit_200.xlsx`, `data/Czech/KCOR_MC_audit_200_summary.log`
- `data/Czech/KCOR_CMR_MC_audit_400.xlsx`, `data/Czech/KCOR_MC_audit_400.xlsx`, `data/Czech/KCOR_MC_audit_400_summary.log`
- `process/mc_audit_<N>_cmr.stdout.log` and `process/mc_audit_<N>_kcor.stdout.log`
- `process/kcor_mc_audit_2021_24_1940_2v0_counts.csv`
- `process/kcor_mc_audit_2021_24_1940_all_pairs_counts.csv`

### Results
- The asymmetry is stable across MC sizes and is not fixed by increasing from 25 to 400 iterations.
- Focal `2 vs 0` point estimate remained `1.223405504`; percentile CI moved from `[1.1997, 1.3952]` at 25 to `[1.1933, 1.3779]` at 400.
- In the 400-run, the bootstrap median was `1.283089477`, mean `1.285373827`, and the original-data point was at about the 9th percentile of bootstrap draws.
- `1 vs 0` shows the same upward bootstrap shift; `2 vs 1` is much more centered. This points to a dose-0 denominator/bootstrap-bias effect, not a one-off error isolated to `2 vs 0`.
- For the 400-run `2 vs 0`, alternative intervals were: percentile `[1.1933, 1.3779]`, basic bootstrap `[1.0689, 1.2535]`, simple BC `[1.1656, 1.2513]`.

### Next steps
- Do not present the percentile CI alone for the focal `2 vs 0` value without explaining the bootstrap shift; it makes the original estimate look suspiciously close to the lower bound.
- Prefer reporting a bias-aware interval (basic bootstrap or BC/BCa if jackknife acceleration is implemented) plus the bootstrap median/diagnostic note, or explicitly label the percentile CI as the percentile distribution interval rather than a centered CI around the original estimator.

## 2026-06-01 - KCOR MC summary reporting fix

### What we did
- Updated `code/KCOR.py` Monte Carlo summary reporting so iteration 0 is displayed as the original-data point estimate and excluded from bootstrap summary statistics.
- Added explicit `Bootstrap draws`, `Percentile 95% CI`, `Basic bootstrap 95% CI`, and `Point percentile in bootstrap draws` lines to the MC summary log.

### Command / executable
- Compile check: `python -m py_compile code\KCOR_CMR.py code\KCOR.py`
- Verification run: `PYTHONIOENCODING=utf-8 MONTE_CARLO=1 MC_ITERATIONS=25 MC_ENROLLMENT_DATE=2021_24 MC_YOB_START=1940 MC_YOB_END=1949 MC_INCLUDE_AGGREGATES=0 DATASET=Czech python3 code/KCOR.py data/Czech/KCOR_CMR_MC_audit_25.xlsx data/Czech/KCOR_MC_audit_25_summaryfix.xlsx "Monte Carlo Audit summary fix n=25" KCOR_MC_audit_25_summaryfix.log`

### Outputs
- Temporary summary-fix verification workbook/log were generated and removed after checking.
- `data/Czech/KCOR_summary.xlsx` was refreshed by the verification run.

### Results
- The focal 25-run `2 vs 0` summary now reports point estimate `1.2234`, bootstrap draws `24`, percentile CI `[1.1997, 1.3952]`, basic bootstrap CI `[1.0516, 1.2471]`, and point percentile `0.1250`.

### Next steps
- Future MC summary logs should use the new separated point/bootstrap reporting; use the basic or BC/BCa interval for deck/reporting if the percentile interval remains visually biased.
