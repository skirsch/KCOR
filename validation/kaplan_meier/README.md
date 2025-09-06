### Kaplan-Meier Validation

This directory will contain a traditional Kaplan–Meier survival analysis to validate KCOR results.

- Purpose: Provide a non-parametric benchmark of survival/incidence over time.
- Inputs: Cohort-level or individual-level event/censor data (to be specified).
- Outputs: Survival curves, confidence intervals, and comparisons vs KCOR-derived metrics.

Planned contents:
- `km_analysis.py`: Core Kaplan–Meier fitting and plotting
- `examples/`: Example datasets and usage scripts
- `out/`: Generated figures and tables

Implementation to follow.

### Usage

Run with make (defaults: `SHEET=2021_24 START=1940 END=2000 GROUPA=0 GROUPB=1,2`):

```bash
make -C validation/kaplan_meier run
```

Notes:
- The `--groups` parameter defines two dose groups. For example, `"0"` and `"1,2"` means cohort 0 vs. combined cohorts 1 and 2.
- Survival is equalized to the vaccinated group's enrollment population so curves are comparable when cohorts are death-matched but have different sizes.

To override parameters:

```bash
make -C validation/kaplan_meier run SHEET=2021_24 START=1940 END=2000 GROUPA="0" GROUPB="1,2"
```

### Results

This directory (`validation/kaplan_meier/`) contains outputs under `out/`.

- Example result (YoB 1940–1995, sheet 2021_24):
  - `validation/kaplan_meier/out/KM_2021_24_1940_1995.png`

Observation: Using the naturally matched cohorts (equalized initial population at enrollment), the curves diverge with the unvaccinated cohort exhibiting lower mortality over time.


