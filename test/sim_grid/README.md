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

```bash
# From repository root
make sim_grid

# Or from this directory
make all
```

## Output Files

- `out/sim_grid_results.xlsx` - KCOR results per scenario
- `out/sim_grid_diagnostics.csv` - Per-cohort diagnostic metrics
- `out/fig_sim_grid_overview.png` - KCOR(t) trajectories
- `out/fig_sim_grid_diagnostics.png` - Diagnostic summaries

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

