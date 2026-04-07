# KCOR Simulation Grid

This directory contains the simulation grid for validating KCOR's operating characteristics and failure-mode diagnostics, as described in §3.4 of the KCOR methods paper.

## Purpose

The simulation grid demonstrates:

1. **Correct null behavior** under selection-induced curvature (gamma-frailty)
2. **Detection of injected effects** (hazard increase/decrease)
3. **Graceful failure** with explicit diagnostics under model misspecification

## Scenarios

| # | Scenario | Description | Expected KCOR |
|---|----------|-------------|---------------|
| 1 | Gamma-Frailty Null | θ_A=1.0, θ_B=0.3, no effect | ≈ 1.0 (±5%) |
| 2 | Injected Hazard Increase | r=1.2 during weeks 20-80 | > 1.05 |
| 3 | Injected Hazard Decrease | r=0.8 during weeks 20-80 | < 0.95 |
| 4 | Non-Gamma Frailty | Lognormal frailty, no effect | Degraded fit |
| 5 | Quiet-Window Contamination | External shock weeks 30-50 | Poor diagnostics |
| 6 | Sparse Events | n=1000, low baseline hazard | Weak identifiability |

## Diagnostics

For each scenario and cohort, we compute:

- **RMSE**: Cumulative-hazard fit error over quiet window
- **θ̂**: Fitted frailty variance
- **R²**: Post-normalization linearity metric

## Usage

**Recommended order from the repository root:** run the operating-characteristics bundle first, then the bootstrap coverage run (they are separate pipelines; this order matches how most people refresh paper artifacts).

1. **`make sim_grid`** — six-scenario grid, S7, Cox-bias, skip-weeks, cohort-size, rollout, and related figures (up to 6 parallel scenario workers by default).
2. **`make bootstrap`** — empirical bootstrap coverage Monte Carlo plus KCOR/theta figures under `out/` (`make bootstrap_coverage` is the same target; 20 workers by default). This step is long.

```bash
# From repository root
make sim_grid
make bootstrap

# Serial scenarios only (optional)
make sim_grid SIM_GRID_MAX_WORKERS=1

# Or build only the sim_grid bundle from this directory (no bootstrap)
make all
```

After `make bootstrap`, see `out/fig_bootstrap_coverage_*.png` (and `.pdf` from the plot step).

## Output Files

- `out/sim_grid_results.xlsx` - KCOR results per scenario
- `out/sim_grid_diagnostics.csv` - Per-cohort diagnostic metrics
- `out/fig_sim_grid_overview.png` - KCOR(t) trajectories
- `out/fig_sim_grid_diagnostics.png` - Diagnostic summaries

After **`make bootstrap_coverage`** (from repo root or this directory with `PYTHON` set):

- `out/bootstrap_coverage.csv`, `out/bootstrap_coverage_theta.csv`, replicate CSVs, `out/bootstrap_coverage_run.json`
- `out/fig_bootstrap_coverage_kcor.png` / `.pdf`, `fig_bootstrap_coverage_kcor_ciwidth.*`, and theta figures when theta CSV is present

## Acceptance Criteria

Per the specification:

- Null scenarios: median KCOR(t) within ±5% over weeks 20-100
- Effect scenarios: directional deviation >5% from unity
- Failure modes: elevated RMSE, reduced R²

## Time Units

All simulations use **event-time** (weeks since cohort entry), not calendar ISO-weeks.

- Time horizon: ~120 weeks
- Quiet window: weeks 20-80
- Effect window (scenarios 2-3): weeks 20-80

