---
name: Wave summary VE outputs
overview: Period-aggregated wave and baseline summaries from weekly_metrics (summed person-weeks and events), wave_ve_summary and baseline_ve_summary (RR, VE, absolute diffs), expanded console logging for age_bin=all, sanity warnings with explicit negative-VE threshold, optional VE decomposition plot (four VE metrics), and required docstrings on why VE measures differ. Debug CSV dose/COVID column tweaks are unrelated housekeeping (appendix only).
todos:
  - id: metrics-period-summary
    content: "metrics.py: ISO period helper, build_period_aggregate_summary (incl. expected_weeks, total_rows, optional unique people), build_period_ve_summary (RR, VE, diff_*), warn_period_summary_sanity"
    status: completed
  - id: wire-run-cfr
    content: "run_cfr_analysis: write wave_summary, baseline_summary, wave_ve_summary, baseline_ve_summary; expanded log for wave age_bin=all; sanity calls; optional plot flag"
    status: completed
  - id: plot-wave-ve
    content: plots.py — plot_wave_ve_summary with ve_case_rate + three other VEs (four series)
    status: completed
  - id: tests-wave-summary
    content: pytest synthetic weekly — sums, rates, RR, VE, diff_*
    status: completed
  - id: yaml-optional-plot
    content: Document plots.save_wave_ve_plot in czech.yaml if added
    status: completed
isProject: false
---

# Wave / baseline aggregated summaries and VE outputs

## Context (current vs requested)

- [`test/CFR/code/metrics.py`](test/CFR/code/metrics.py) already builds [`weekly`](test/CFR/code/metrics.py) with `population_at_risk`, `cases`, `deaths_covid`, `deaths_all`, `deaths_non_covid`, `cfr_*`, and weekly `ve_implied_*` vs reference (via `_add_implied_ve_columns`).
- [`cohort_summary`](test/CFR/code/metrics.py) uses **means of weekly rates** in baseline/wave, not **sums**. This work adds **integrated** totals: `sum(population_at_risk)`, `sum(cases)`, `sum(deaths_*)`, then rates from those totals.
- Reference cohort: `reference_cohort` from `expected_deaths.reference_cohort` in [`run_cfr_analysis.py`](test/CFR/run_cfr_analysis.py), typically `dose0`.
- Do **not** assume `dose3` in `weekly` unless present; VE rows for every `cohort != reference_cohort`.

## Required interpretation note (docstrings / module comments)

**Copy verbatim into** `build_period_ve_summary` (and optionally `build_period_aggregate_summary`) docstring or adjacent module comment:

- `ve_case_rate`, `ve_cfr_covid`, and `ve_covid_death_rate` are **not expected to match**, because:
  - **Case-rate VE** (`ve_case_rate`) measures **infection incidence reduction** (per person-week).
  - **CFR VE** (`ve_cfr_covid`, `ve_cfr_allcause`) measures **severity conditional on infection** (deaths per case).
  - **COVID death-rate VE** (`ve_covid_death_rate`) **combines** incidence and severity (deaths per person-week).

Also keep the existing distinction: CFR-type rates answer severity given a positive test; all-cause death rate per person-week answers population mortality burden.

## 1. Core helpers in [`test/CFR/code/metrics.py`](test/CFR/code/metrics.py)

- **`_iso_weeks_in_period(period_start_iso, period_end_iso) -> set[str]`**  
  `iter_followup_mondays` + `monday_to_iso_week`.

- **`expected_weeks_in_period(period_start, period_end) -> int`**  
  `len(iter_followup_mondays(...))` — canonical expected week count for the YAML window.

- **`build_period_aggregate_summary(weekly, *, period_start, period_end, period_name, ...)`**
  - Filter rows to `iso_week ∈ _iso_weeks_in_period(...)`.
  - Per group `groupby(["cohort", "age_bin"])`:
    - **`expected_weeks_in_period`**: same constant for every row (from config strings).
    - **`weeks_in_period`**: `iso_week` **nunique** per group — compare to expected in sanity (warn if mismatch).
    - **`total_rows`**: count of weekly rows aggregated into that group (before groupby collapse = 1 per week per stratum; after groupby this is implicit; implement as **number of iso_weeks represented** = same as `weeks_in_period` **or** literal `len(sub)` pre-agg — document: e.g. `n_weekly_rows_in_period` = rows summed, equals `weeks_in_period` when one row per week).
    - **`total_person_weeks`**, **`total_cases`**, **`total_covid_deaths`**, **`total_allcause_deaths`**, **`total_noncovid_deaths`** (if column exists).
    - **`total_unique_people` (optional):** weekly rows do not carry `ID`. If cheap: accept optional **`df_model`** and compute `df_model` filtered by cohort mask + `age_bin` with `.groupby(["cohort","age_bin"])["ID"].nunique()` for the same strata, then merge onto summary; if `df_model` omitted, column `NaN` or omit. Prefer **one optional merge** from `run_cfr_analysis` after summary build to avoid bloating the core signature — either way, document in plan as implemented.
  - Constants: `period_start`, `period_end`, `period_name`.
  - Rates (safe div, `NaN` if denom ≤ 0): `case_rate_period`, `covid_death_rate_period`, `allcause_death_rate_period`, `cfr_covid_period`, `cfr_allcause_period`, optional `noncovid_death_rate_period`.
  - **CSV column names:** use `*_wave` / `*_baseline` in saved files via suffix parameter or rename at write time.

- **`build_period_ve_summary(period_summary, *, reference_cohort)`**
  - Join each non-reference `(cohort, age_bin)` to reference row.
  - **RR** columns (five + optional noncovid): `rr_case_rate`, …
  - **VE** = `1 - RR` (same set).
  - **Absolute differences** (cohort minus reference):  
    `diff_case_rate`, `diff_covid_death_rate`, `diff_allcause_death_rate`, `diff_cfr_covid`, `diff_cfr_allcause` (+ optional noncovid).
  - Wide numerators/denominators for ref and cohort (presentation-ready).

## 2. Wire [`test/CFR/run_cfr_analysis.py`](test/CFR/run_cfr_analysis.py)

1. `wave_summary` = `build_period_aggregate_summary(..., wave.start/end, "wave")`.
2. `baseline_summary` = same for `baseline.start/end`, `"baseline"`.
3. `wave_ve_summary` = `build_period_ve_summary(wave_summary, reference_cohort=ref_cohort)`.
4. **`baseline_ve_summary` = `build_period_ve_summary(baseline_summary, reference_cohort=ref_cohort)`** — **required** (cheap; supports “VE outside wave?” vs “baseline selection” interpretation).

**Write four CSVs:**

- [`test/CFR/out/wave_summary.csv`](test/CFR/out/wave_summary.csv)
- [`test/CFR/out/baseline_summary.csv`](test/CFR/out/baseline_summary.csv)
- [`test/CFR/out/wave_ve_summary.csv`](test/CFR/out/wave_ve_summary.csv)
- [`test/CFR/out/baseline_ve_summary.csv`](test/CFR/out/baseline_ve_summary.csv)

Existing outputs unchanged.

## 3. Console logging (explicit requirement)

**Wave period, `age_bin == "all"` only (minimum):** for each cohort present (`dose0`, `dose1`, `dose2`, …), print **in one block** (so debate use does not require opening CSVs):

- `total_cases`
- `total_covid_deaths`
- `total_allcause_deaths`
- `total_person_weeks`
- `cfr_covid_period` (or suffixed name in log)
- `covid_death_rate_period`
- `allcause_death_rate_period`

Then a compact line for **dose2 vs dose0** (or each non-ref vs ref): e.g. the three **VE** types above + case-rate VE if space allows.

**Optional:** repeat a shorter variant per `age_bin != "all"`.

Implement as **`log_period_aggregate_for_console(wave_summary, log, ...)`** in `metrics.py` or next to call site.

## 4. Sanity checks (`warn_period_summary_sanity`)

- Missing reference row per `age_bin` → **warning**.
- `total_person_weeks == 0` but cases/deaths &gt; 0 → **warning**.
- **VE &gt; 1** (finite) or **RR ≤ 0** (where defined) → **warning** (pathological).
- **Negative VE:** **allowed**; **do not warn by default.** Only warn if **strongly** negative, e.g. **`VE < -1`** for any of the main VE columns (tunable constant, document in code). Values in `(-1, 0]` are normal output, not suspicious.
- Integrated vs mean weekly mortality: keep relative-difference check with explanation (Jensen / varying denominators).

## 5. Optional plot [`plot_wave_ve_summary`](test/CFR/code/plots.py)

- Filter `cohort == compare_cohort` (default `dose2`).
- **Four** series by `age_bin`: **`ve_case_rate`**, **`ve_cfr_covid`**, **`ve_covid_death_rate`**, **`ve_allcause_death_rate`** — decomposition story requires case-rate VE on the figure.
- Style consistent with existing plots.
- Gate: `cfg.get("plots", {}).get("save_wave_ve_plot", True)`.

## 6. Tests

Synthetic `weekly` in **`test/CFR/test_wave_summary.py`**: verify aggregates, `expected_weeks_in_period`, RR, VE, **`diff_*`**.

## 7. Files touched

| File | Change |
|------|--------|
| [`test/CFR/code/metrics.py`](test/CFR/code/metrics.py) | Period summary + VE summary + diffs + sanity + logging helper |
| [`test/CFR/run_cfr_analysis.py`](test/CFR/run_cfr_analysis.py) | Four CSVs, logging, sanity, plot |
| [`test/CFR/code/plots.py`](test/CFR/code/plots.py) | Four-metric VE plot |
| [`test/CFR/config/czech.yaml`](test/CFR/config/czech.yaml) | Optional plot flag note |
| `test/CFR/test_wave_summary.py` | New |

**Not in this plan’s scope (housekeeping):** debug enrollment CSV numeric `dose` / `covid_deaths_in_week` — see **Appendix A**. Do not block core wave/VE work on it.

---

## Appendix A — Unrelated housekeeping (debug enrollment CSV)

*Separate from wave/VE deliverables.* Optional follow-up in [`qa_summary.py`](test/CFR/code/qa_summary.py) (`write_debug_birth_cohort_weekly_csv`): `dose` as int 0–3; add `covid_deaths_in_week` from `deaths_covid`. File remains CSV. Implement in a small PR or mini-plan so implementation focus stays on §1–7.
