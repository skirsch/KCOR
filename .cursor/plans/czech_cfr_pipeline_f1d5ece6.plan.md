---
name: Czech CFR pipeline
overview: Add a reproducible, YAML-driven pipeline under `test/CFR/` that loads Czech record-level CSVs (with a Czech→canonical column map matching existing KCOR code), builds fixed enrollment cohorts at ISO week 2021-24 with follow-up through 2022-20, computes weekly population-at-risk, cases, and deaths (all-cause vs COVID where fields allow), derives case rate / CFR (with lags) / mortality, wave-vs-baseline summaries, cohort ratios, arithmetic decomposition, age stratification, VE=90% expected-vs-observed (killer figure), plus required optional extensions (KM, time-since-dose, reinfection sensitivity).
todos:
  - id: yaml-config
    content: Add test/CFR/config/czech.yaml (paths, weeks, age_bins, cfr.lag_weeks, cohorts, VE, filters, output dir)
    status: completed
  - id: load-cohort
    content: Implement load_data.py + cohort_builder.py (53-col map, ISO week parsing, mRNA optional, enrollment/freeze rules, age bins)
    status: completed
  - id: metrics-core
    content: "Implement metrics.py: weekly pop/cases/deaths, rates, wave vs baseline, cohort ratios, decomposition columns (COVID vs all-cause tracks)"
    status: completed
  - id: simulate-plot
    content: Implement simulate_expected.py + plots.py + expected_vs_observed killer figure (weekly + cumulative)
    status: completed
  - id: analysis-entry
    content: Implement analysis.py + run_cfr_analysis.py; write all four CSVs; stability check logs
    status: completed
  - id: required-extras
    content: Age-stratified outputs, lagged CFR 0/2/3/4, cumulative case/death plots, quiet-period sanity prints
    status: completed
  - id: optional-km-tsd-reinf
    content: "Optional: KM post-infection plots, time-since-2nd-dose strata, reinfection YAML flag"
    status: completed
isProject: false
---

# Czech CFR / infection / mortality analysis pipeline

## Context from the repo

- **Input file**: Your spec uses `/data/Czech/records.csv`. In this repo the same file is expected at [`data/Czech/records.csv`](data/Czech/records.csv) (used by [`code/KCOR_ts.py`](code/KCOR_ts.py)); it is **not** committed. A small sample exists at [`data/Czech/records_10k.csv`](data/Czech/records_10k.csv) for development smoke tests. Default YAML path should be **repo-root-relative** (e.g. `data/Czech/records.csv`).
- **Schema**: Real exports match the **53-column** Czech layout already renamed to English in [`code/KCOR_ts.py`](code/KCOR_ts.py) (lines 122–132) and [`code/KCOR_variable.py`](code/KCOR_variable.py) (lines 79–89). Reuse the same mapping in [`test/CFR/code/load_data.py`](test/CFR/code/load_data.py) (or a tiny shared `column_map` constant) so behavior matches the rest of the pipeline.
- **Dates**: Doses and death (`DateOfDeath`, LPZ) are **ISO week strings** (`YYYY-WW`); parse to **Monday** via `%G-%V-%u` + `'-1'` (same as existing code). Infection onset: [`DateOfPositiveTest`](code/KCOR_ts.py) (same convention).
- **COVID vs all-cause death**: The English rename maps Czech `DatumUmrtiLPZ` → `DateOfDeath` (all-cause timing). `Date_COVID_death` is a separate column in the same mapping—**verify in your full extract** whether it is populated for COVID-attributed deaths; if sparse, document in logs and fall back to **all-cause** for primary mortality while still outputting COVID-specific columns when non-null. [`PrimaryCauseHospCOVID`](code/KCOR_variable.py) can support a secondary rule only if validated against the dictionary.
- **Dependencies**: [`requirements.txt`](requirements.txt) already includes `pandas`, `numpy`, `matplotlib`, `pyyaml`, and **`lifelines`** (use `KaplanMeierFitter` for optional KM—no Cox).
- **Optional mRNA filter**: Existing tools filter non-mRNA via [`code/mfg_codes.py`](code/mfg_codes.py). Expose `filters.non_mrna_only: true|false` in YAML (default `true` to match [`code/KCOR_ts.py`](code/KCOR_ts.py); set `false` for a fully raw comparison).

## Design choices (locked for transparency)

1. **Enrollment cohorts (frozen at 2021-24)**  
   - Define **enrollment instant** as the **Monday** of `enrollment_week` (configurable if needed).  
   - **dose0**: `Date_SecondDose` is missing **or** second-dose Monday **is after** enrollment Monday (not fully vaccinated as of enrollment).  
   - **dose2**: `Date_SecondDose` Monday **≤** enrollment Monday (completed primary series by enrollment).  
   - **dose3** (optional): `Date_ThirdDose` Monday **≤** enrollment Monday; mutually exclusive strata or nested flag—implement as **separate optional cohort list** in YAML (`cohorts: [dose0, dose2]` plus optional `dose3`) so tables stay clear.  
   - **No switching after enrollment**: person stays in initial cohort for all follow-up weeks (even if they later dose up—this matches “freeze membership” and avoids Cox-style adjustment).

2. **Follow-up and censoring**  
   - Follow-up weeks: inclusive from `followup_start` through `followup_end` (2021-24 … 2022-20).  
   - **Population at risk** at the **start** of week `w`: in cohort, **enrolled** (alive at enrollment—exclude those with `DateOfDeath` strictly before enrollment Monday), and **not dead before Monday of `w`**.  
   - **Death in week `w`**: death Monday equals Monday of `w` (same ISO-week convention as KCOR_variable).  
   - **New case in week `w`**: first positive test week equals `w`, with optional sensitivity for reinfections (see optional extensions).

3. **Age stratification (40–49, …, 70+)**  
   - `YearOfBirth` is often a **band** (e.g. `1970-1974`). Use the **first four digits** as band start (existing pattern in [`code/KCOR_ts.py`](code/KCOR_ts.py) lines 165–167), then **age at enrollment** ≈ `iso_year(enrollment) - band_start` (document that this is **left edge of band**, conservative for bins). Assign to YAML bins `[40,49]`, `[50,59]`, `[60,69]`, `[70,120]`; drop/exclude rows outside defined bins with logged counts.

4. **CFR and lags**  
   - **Same-week CFR (lag 0)**: `COVID_deaths_in_week_w_among_infections_in_week_w` / `cases_in_week_w`—implement as **person-level**: infections with onset in week `t`, deaths with COVID attribution in weeks `t` through `t+k` (k from `cfr.lag_weeks`). Use **distinct infection cohorts** for each lag column to avoid double-counting across lags in one row (store `cfr_lag0`, `cfr_lag2`, … in wide or long tidy output).  
   - **Mortality rate**: all-cause deaths / pop at risk (primary); also emit COVID death rate if column usable.

5. **Decomposition**  
   - Identity for each week: `mortality_rate ≈ case_rate × CFR` when CFR uses **all-cause deaths following infection** vs **infection-defined CFR**—spec’s “COVID_deaths / cases” is **not** identical to all-cause mortality unless you align definitions. **Implement two tracks in outputs**: (A) **COVID-CFR track** for VE narrative; (B) **all-cause mortality decomposition** using `cases × (all_cause_deaths_among_cases / cases)` over the same infection cohort, with captions explaining alignment. Plot “decomposition” as grouped lines or stacked contribution (case_rate × CFR vs observed mortality) **per track** to avoid misleading overlays.

6. **Wave vs baseline**  
   - Baseline: `2021-24`–`2021-40`; wave: `2021-41`–`2022-20` (from YAML).  
   - Summaries: ratio of mean weekly case rate (wave/baseline), mean CFR (wave, and baseline where defined), mean mortality (wave/baseline).  
   - **Critical ratios**: vaccinated / unvaccinated for case rate, CFR, mortality (per lag and age stratum).

## File layout (as requested)

| Path | Role |
|------|------|
| [`test/CFR/config/czech.yaml`](test/CFR/config/czech.yaml) | Paths, weeks, bins, lags, cohorts, VE scenario, plot style, filters |
| [`test/CFR/code/load_data.py`](test/CFR/code/load_data.py) | CSV read (reuse `_read_csv_flex` pattern from KCOR_ts or call shared logic), column rename, dtype cleaning, optional mRNA filter, infection≤1 default with reinfection override |
| [`test/CFR/code/cohort_builder.py`](test/CFR/code/cohort_builder.py) | Enrollment rules, age_bin assignment, per-person parsed Monday fields, eligibility flags |
| [`test/CFR/code/metrics.py`](test/CFR/code/metrics.py) | Weekly aggregates: pop, cases, deaths (split), case_rate, CFR variants, mortality, ratios, wave/baseline summaries |
| [`test/CFR/code/simulate_expected.py`](test/CFR/code/simulate_expected.py) | `expected_deaths_vax[w] = pop_vax[w] * case_rate_vax[w] * ( (1-VE) * CFR_unvax_ref[w] )` with configurable `VE` (default 0.9) and configurable **CFR reference** (same-week unvax vs baseline-period unvax—document in YAML) |
| [`test/CFR/code/analysis.py`](test/CFR/code/analysis.py) | Orchestrate pipelines: overall + age-stratified + cumulative series + stability checks |
| [`test/CFR/code/plots.py`](test/CFR/code/plots.py) | All figures to [`test/CFR/out/`](test/CFR/out/) |
| [`test/CFR/run_cfr_analysis.py`](test/CFR/run_cfr_analysis.py) | CLI entry: load YAML, run analysis, write CSVs + plots, logging |

## Outputs (CSVs and plots)

Under [`test/CFR/out/`](test/CFR/out/):

- `weekly_metrics.csv` — long format: `week`, `cohort`, `age_bin` (or `all`), pop, cases, deaths_all, deaths_covid, rates, CFR_lag*, cohort ratios vs reference.
- `cohort_summary.csv` — enrollment N, total cases/deaths, wave/baseline means and ratios.
- `age_stratified_metrics.csv` — same structure as weekly_metrics keyed by age bin.
- `expected_vs_observed.csv` — weekly (and optionally cumulative) observed vs expected vaccinated deaths under VE scenario + residuals.

Plots:

- `case_rate.png`, `cfr.png`, `mortality_rate.png` — time series by cohort (faceted by age in optional multi-panel or separate files if too busy; default: overall + save age-stratified variants as `case_rate_age.png` etc. or single PDF—pick one convention and YAML-toggle).
- `decomposition.png` — mortality vs case_rate×CFR (with definition subtitle).
- `expected_vs_observed.png` — **primary debate figure**: observed vaccinated deaths vs expected under 90% VE (weekly and **cumulative inset or second panel** for “slope during wave”). This satisfies the **killer plot** ask (same metric, presentation-maximized: clear legend, wave shading, residual text).

## Logging / success criteria

- Print cohort sizes at enrollment, exclusions, total cases/deaths, min/max pop, any negative counts (assert zero), weeks with zero pop.
- Echo YAML paths and definition strings (enrollment rule, CFR lag definition) so outputs are self-describing.

## Optional extensions (included in scope)

1. **Kaplan–Meier post-infection**: among first-infection cohort, time from infection Monday to death (all-cause and COVID if coded), censored at end of follow-up; curves by vaccination cohort and age; save `km_post_infection.png` (and CSV of survival table if useful). Use **lifelines**, not Cox.  
2. **Time-since-vaccination**: within `dose2`, bins of `(enrollment_monday - second_dose_monday)` in weeks (YAML-defined); repeat key tables/plots for largest stratum or small multiples.  
3. **Reinfection**: YAML flag to **disable** `Infection <= 1` filter or to use all positive dates if the schema allows multiple rows per person (document if one row per person only).

## Testing / runbook

- Smoke test on [`data/Czech/records_10k.csv`](data/Czech/records_10k.csv) via YAML override or CLI `--config` / `--input` to ensure the pipeline runs end-to-end.  
- Document in a short comment at top of `run_cfr_analysis.py`: full national file path expectation and approximate runtime.

## Non-goals

- No Cox models, no IPTW/PS adjustment—only arithmetic, ratios, KM descriptive curves.

```mermaid
flowchart LR
  yaml[czech.yaml] --> load[load_data]
  load --> cohort[cohort_builder]
  cohort --> metrics[metrics]
  metrics --> analysis[analysis]
  analysis --> sim[simulate_expected]
  analysis --> plots[plots]
  sim --> plots
  plots --> out[test/CFR/out]
```
