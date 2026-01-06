Yes. Paste this as **one Cursor prompt** (top-level instruction) and let it run. 

It’s written to (1) find placeholders, (2) compute the missing Monte Carlo/bootstrapped quantities from existing artifacts, (3) fill the tables in `paper.md`, and (4) leave the build working.

Note that we use test/sim_grid for the code and results. there is a test/sim_grid/code directory, data directory, and out directory.


```text
You are working in the KCOR paper repo. Goal: complete ALL remaining “Claude round 3” tasks by filling any placeholder tables/values in the manuscript using already-generated simulation/MC artifacts (do not invent numbers). Do not refactor the method; only compute + insert missing values, add minimal clarifying text, and keep the build green.

FILES:
- Manuscript: documentation/preprint/paper.md (if different, locate the active paper.md used by `make paper`)
- Claude notes: /mnt/data/claude3.md (read and implement all remaining actionable items)
- There may be multiple paper.md; use the one actually built by make.

HIGH-LEVEL PLAN (execute fully):
1) Read claud3.md (in this directory) and list every remaining actionable item. Treat “placeholders in tables” and “fill in values” as mandatory.
2) In the manuscript, grep for placeholders / unfinished spots:
   - patterns: "[coverage", "TODO", "TBD", "xxx", "??", "PLACEHOLDER", "fill", "FIXME", empty table cells, or qualitative claims where a numeric table is referenced.
3) Identify each table that needs numeric values. For each one:
   - Determine the exact estimand and the data source already produced by the repo (simulation outputs, MC sheets, cached CSVs used for figures, etc.).
   - Compute the missing values by reading those artifacts (do not rerun full sims unless strictly necessary).
   - Insert the numbers into the table in paper.md, rounding reasonably (typically 3 decimals for proportions, 2–3 sig figs for ratios).
   - Ensure captions and in-text references remain consistent.

MONTE CARLO / BOOTSTRAP CI AGGREGATION (CZECH OR EMPIRICAL OUTPUTS):
- The repo already outputs one KCOR sheet per MC draw, with columns like:
  Date, ISOweekDied_x, KCOR, CI_lower, CI_upper, CH_num, hazard_num, hazard_adj_num, t_num, CH_den, hazard_den, hazard_adj_den, t_den, abnormal_fit, EnrollmentDate, YearOfBirth, Dose_num, Dose_den, ISOweekDied_y, KCOR_ns
- Standardize internal/paper-facing columns as:
  PUBLIC columns: ISOweekDied_x (rename to ISOweekDied), KCOR, CI_lower, CI_upper, EnrollmentDate, YearOfBirth, Dose_num, Dose_den
  INTERNAL columns (keep in artifacts, not shown in paper tables): mc_id, theta_num, theta_den, CH_num, CH_den, hazard_num, hazard_den, hazard_adj_num, hazard_adj_den, t (since t_num==t_den), abnormal_fit
  DROP: ISOweekDied_y, KCOR_ns, t_num, t_den, Date (if ISOweekDied used)
- If the current MC outputs already include CI_lower/CI_upper per draw, ignore those and recompute aggregate CI bands across draws:
  For each group (EnrollmentDate, YearOfBirth, Dose_num, Dose_den, ISOweekDied):
    KCOR_hat = median of KCOR across draws
    CI_lower = 2.5th percentile across draws
    CI_upper = 97.5th percentile across draws
- Produce (or update) a single aggregated CSV used for figures/tables, and ensure figures reference that artifact.

SIMULATION COVERAGE TABLE (BOOTSTRAP COVERAGE):
- Locate the simulation / bootstrap artifacts already generated for the paper (outputs feeding the negative controls, positive controls, sim grid, etc.).
- Compute empirical coverage for each scenario row that the manuscript claims:
  coverage = mean(true_value in [CI_lower, CI_upper])
- Use a scalar target that matches what the table intends (end-of-window KCOR or the same timepoint used in the relevant results table). Do not change estimands; match the existing text/figure conventions.
- Fill the coverage table fully with numeric values.

OPERATING CHARACTERISTICS TABLE:
- If the manuscript references “operating characteristics” without a filled numeric table, add or fill a compact table with:
  Scenario, True effect, Median KCOR (end-of-window), IQR (or RMSE if already computed), Drift/year (if already in outputs), Diagnostic pass rate (from abnormal_fit or existing diagnostic flag)
- Prefer using existing derived files (e.g., sim summary CSVs) rather than recomputing expensive sims.

CONSISTENCY / SANITY:
- Ensure t_num and t_den are treated as identical; collapse to a single t variable in any internal data handling.
- Ensure ISOweek columns are consistent: use ISOweekDied_x as canonical; rename to ISOweekDied.
- Verify that any filled table values match the figures/tables they support; if mismatch arises, make the paper consistent with the artifacts that drive the figures (single source of truth).
- After edits, run the repo’s paper build command (e.g., `make paper`) and fix any warnings/errors introduced (broken refs, table formatting, etc.). Keep changes minimal.

DELIVERABLES:
- Commit-ready edits to the manuscript table(s) with all placeholders removed.
- Any new/updated derived CSVs needed for the paper should be saved under the existing analysis/output convention (do not create random directories).
- Add minimal notes in the manuscript (1–2 sentences max) only if needed to define that CIs are pointwise bootstrap percentile bands, or to define coverage.
- Leave the repo in a state where `make paper` completes without undefined references.

Now do the work end-to-end.
```


