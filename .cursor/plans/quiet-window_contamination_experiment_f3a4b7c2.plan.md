---
name: Quiet-window contamination experiment
overview: Add a new synthetic simulation under the same Gompertz-plus-gamma-frailty DGP as [test/theta0/theta0_nwave.py](test/theta0/theta0_nwave.py), calibrate baseline k to Czech KCOR_CMR weekly hazard (YoB ~1940–1960), inject additive non-frailty hazard only in rebased weeks 4–56 on the lower-frailty cohort, run production `fit_theta0_gompertz` + inversion + anchored KCOR (matching [test/sim_grid/code/generate_sim_grid.py](test/sim_grid/code/generate_sim_grid.py) dose-1/dose-0 ratio), sweep Types 1–2 plus Type 3 combined grid, aggregate 20 reps per cell, and write tables/figures, epsilon/h_ref scaling, and a summary block including the required combined sign-reversal line for the Grok rebuttal.
todos:
  - id: scaffold-script
    content: Add test/quiet_window_contamination/run_contamination_test.py with repo-root import path, argparse, out/ directory creation, and constants (N, weeks, theta0_A/B, epsilon grids, n_reps=20).
    status: completed
  - id: calibrate-k-czech
    content: Derive h_ref from data/Czech/KCOR_CMR.xlsx (canonical; present in full dev checkout). Support --cmr-xlsx override; if default path missing, clear error (run CMR aggregation per README). Set k from h_ref; log h_ref and epsilon/h_ref in outputs.
    status: completed
  - id: dgp-contamination
    content: Implement stochastic two-cohort simulator from theta0_nwave-style loop + quiet-window-only additive g(t); verify k read weeks 0–3 rebased are uncontaminated.
    status: completed
  - id: pipeline-kcor
    content: Wire hazard_eff, quiet_mask, fit_theta0_gompertz, invert_gamma_frailty, dose-1/dose-0 anchored KCOR matching generate_sim_grid norm_week; compute asymptote and slope.
    status: completed
  - id: aggregate-figures
    content: Run 1D grids for types 1–2 plus Type 3 full 4×4 positive outer product (16 cells × 20 reps = 320 runs for combined only); enforce 16/20 rep rule; CSVs, figures, plausibility warnings, full summary block.
    status: completed
  - id: paper-hook
    content: "After results: add §3.x subsection + figure/table pointers in paper.md/main.tex (user-driven text)."
    status: completed
isProject: false
---

# Quiet-window synthetic contamination experiment

## Goal

Empirically bound how much **quiet-window-only, non-frailty** additive hazard curvature can (i) bias $\hat\theta_{0}$, and (ii) move **anchored** KCOR away from the true null (1), including the smallest $\epsilon$ at which normalized KCOR crosses 1 in the “harm” direction ($\mathrm{KCOR}>1$ with cohort A = dose 1 / lower $\theta_0$, cohort B = dose 0 / higher $\theta_0$, same baseline $k$).

This complements existing positive controls (S4.3 / Figure S1 style: multiplicative shift in an **effect** window) by contaminating the **quiet** window itself.

## What to reuse from the repo (do not reimplement the estimator)

| Piece | Source |
|--------|--------|
| Gompertz + frailty **DGP** (individual $h^{\mathrm{eff}}$, cohort $h_{\mathrm{obs}}$, discrete $H_{0,\mathrm{eff}}$) | [test/theta0/theta0_nwave.py](test/theta0/theta0_nwave.py) (`simulate_cohort` loop: lines ~97–107) |
| Production $\theta_0$ fitter (v7.4 delta-iteration, hazard space) | [code/KCOR.py](code/KCOR.py) `fit_theta0_gompertz` |
| $\gamma$ and related knobs | `K.get_gompertz_theta_fit_config()` (same pattern as [test/alpha/code/estimate_alpha.py](test/alpha/code/estimate_alpha.py) ~498–515) |
| Skip / rebasing | `K.DYNAMIC_HVE_SKIP_WEEKS` (= 2) and `t_rebased = t_week - float(K.DYNAMIC_HVE_SKIP_WEEKS)` |
| Inversion | `K.invert_gamma_frailty` |
| Anchored KCOR convention | Match [test/sim_grid/code/generate_sim_grid.py](test/sim_grid/code/generate_sim_grid.py): ratio **Dose 1 / Dose 0** on $H_0$, then divide by raw ratio at `norm_week = skip_weeks + 4` (see ~892–906). |

**Verified call signature (`fit_theta0_gompertz`):** Matches [code/KCOR.py](code/KCOR.py) at lines 1094–1101 and [test/alpha/code/estimate_alpha.py](test/alpha/code/estimate_alpha.py) 508–514: positional `h_arr`, `t_rebased`, `quiet_mask`, `k_anchor_weeks`, `gamma_per_week`; optional `theta0_init`; **`deaths_arr`** keyword is correct.

**Quiet mask (critical):** Do **not** use `K._build_theta_quiet_mask` with synthetic data (calendar `theta_estimation_windows` would be wrong). Build a boolean mask directly:

- `quiet_mask = (t_rebased >= 4) & (t_rebased <= 56)` on the weekly grid, aligned with the user spec (“quiet window: weeks 4–56 rebased after 2-week skip”).

**Why a single contiguous quiet window is OK for `fit_theta0_gompertz`:** With one segment, `gap_end_idx` is empty ([code/KCOR.py](code/KCOR.py) ~1352–1358), so no wave $\delta_i$ are accumulated; the iterator still performs seed fit + pooled refit on `quiet ∪ anchor` with $\Delta=0$. That matches a **wave-free** stress test.

## Pre-run calibration of $k$ (Czech empirical scale)

**Verify before running the epsilon sweep:** A fixed guess such as $h_{\mathrm{cohort}}(0)\approx 5\times10^{-4}$ is only illustrative. If $k$ is too low, the same $\epsilon$ values represent a **larger fraction** of baseline hazard and plausibility judgments (and Grok-style “small $\epsilon$” claims) become **misleading**.

**Procedure (implementation step, not hand-waving):**

1. Load the same Czech aggregate the paper uses. **Default path:** [data/Czech/KCOR_CMR.xlsx](data/Czech/KCOR_CMR.xlsx) (same as [test/alpha/code/estimate_alpha.py](test/alpha/code/estimate_alpha.py) line 527: `repo_root / "data" / "Czech" / "KCOR_CMR.xlsx"`). This file **is present** in the project’s full data checkout; some minimal clones may omit it (then run `code/KCOR_CMR.py` / `make` per [README.md](README.md)). Support `--cmr-xlsx PATH` and **exit with a clear message** if the chosen path is missing.
2. Build weekly hazards with the **same** discrete-week definition as KCOR: `MR = Dead/Alive` (clipped), then `K.hazard_from_mr` (or equivalent) on **post–skip-week** bins in an early-follow-up band comparable to rebased weeks $0$–$3$ on the simulator grid.
3. Summarize an empirical reference hazard $h_{\mathrm{ref}}$ (e.g. median or trimmed mean across those bins and strata — pick one rule, document it in the script and CSV metadata).
4. **Set simulator $k$** so the **uncontaminated** frailty cohort hazard at $t_{\mathrm{rebased}}=0$ matches that scale (for the cohort used as reference, e.g. dose 0 with $\theta_{0,B}=4$ or a pooled rule — document). Because $h(0)=k$ in the frailty-only formula at $H_{0,\mathrm{eff}}=0$, this pins $k \approx h_{\mathrm{ref}}$ up to the small-$t$ discrete-step convention used in [test/theta0/theta0_nwave.py](test/theta0/theta0_nwave.py).
5. In **output tables and figures**, report each tested $\epsilon$ as an **absolute** hazard add-on **and** as **$\epsilon / h_{\mathrm{ref}}$** (or $\epsilon$ as % of $h_{\mathrm{ref}}$) so contamination is directly comparable to Czech weekly hazard at the relevant age strata.
6. Revisit `PLAUSIBLE_DYNAMIC_HVE_MAX` / `PLAUSIBLE_COMORBIDITY_MAX`: express them in the same units as $\epsilon$ **and** optionally as fractions of $h_{\mathrm{ref}}$ so the printed warnings stay meaningful after calibration.

## Data-generating process

- **Per cohort:** $N=50{,}000$, **130** discrete weeks ($t_{\mathrm{week}}=0,\ldots,129$).
- **Shared** Gompertz scale: $k$ **set from the Czech calibration above** (not a hard-coded $5\times10^{-4}$). Document $k$, $h_{\mathrm{ref}}$, enrollment sheet(s), YoB filter, and resulting $h_{\mathrm{frailty}}(0)$ for both $\theta$ values in the script header / CSV metadata.
- **Frailty:** Cohort A (dose 1): $\theta_{0,A}=0.5$; Cohort B (dose 0): $\theta_{0,B}=4.0$. **Same** $k$, **no** wave multipliers → true null cumulative contrast after correct normalization is 1 (same generative baseline hazard).
- **Stochastic deaths:** Binomial draws per week (as in `theta0_nwave`), **not** deterministic fractional deaths, with a **per-rep** RNG seed derived from `(contamination_case, epsilon, rep_id)`.
- **Contamination (only cohort A / dose 1):** Additive hazard **only** when `quiet_mask` is true:

  - **Type 1 (dynamic):** $g(t)=\epsilon_{\mathrm{dyn}}\exp(-\lambda_{\mathrm{decay}} t_{\mathrm{rebased}})$, $\lambda_{\mathrm{decay}}=0.05$, $\epsilon_{\mathrm{dyn}}\in\{0,5\times10^{-4},10^{-3},2\times10^{-3},5\times10^{-3}\}$.
  - **Type 2 (comorbidity):** $g(t)=\epsilon_{\mathrm{comorb}}$, $\epsilon_{\mathrm{comorb}}\in\{0,10^{-4},3\times10^{-4},10^{-3},3\times10^{-3}\}$.
  - **Type 3 (combined — primary rebuttal case):** $g(t)=\epsilon_{\mathrm{dyn}}\exp(-\lambda_{\mathrm{decay}} t_{\mathrm{rebased}})+\epsilon_{\mathrm{comorb}}$ on the same quiet mask. Use the **full outer product of the strictly positive** Type 1 and Type 2 grids: $\epsilon_{\mathrm{dyn}}\in\{5\times10^{-4},10^{-3},2\times10^{-3},5\times10^{-3}\}$ (4 values) $\times$ $\epsilon_{\mathrm{comorb}}\in\{10^{-4},3\times10^{-4},10^{-3},3\times10^{-3}\}$ (4 values) $\Rightarrow$ **16 cells**; at `n_{\mathrm{rep}}=20$, **320 simulation runs** for Type 3 alone (cheap). Record `contamination_type=combined`, `epsilon_dyn`, `epsilon_comorb` on every row.

**Interpretation:** If sign reversal requires implausible $\epsilon$ in Types 1 and 2 separately, the scientifically decisive check is often Type 3: implausible **pairs** $(\epsilon_{\mathrm{dyn}},\epsilon_{\mathrm{comorb}})$ or an implausible **effective total** (optional diagnostic: e.g. $\max_t g(t)$ or $\frac{1}{|W|}\int_W g$ over quiet weeks $W$) relative to $h_{\mathrm{ref}}$.

Implementation note: compute frailty-consistent $h_{\mathrm{frailty}}(t)$ first, then `h = h_frailty + g(t)` (clip $h$ if needed to keep probabilities valid), then `p = 1 - exp(-h)`.

## Pipeline per rep

For each cohort array `Alive`, `Dead`, `t_week`:

1. `MR = Dead / Alive` (clip), `hazard_obs = K.hazard_from_mr(MR)` (or equivalent `−log(1−MR)` with the same clipping as [test/alpha/code/estimate_alpha.py](test/alpha/code/estimate_alpha.py)).
2. `hazard_eff = hazard_obs` for `t_week >= DYNAMIC_HVE_SKIP_WEEKS`, else `0`.
3. `H_obs = cumsum(hazard_eff)`.
4. `(k_hat, theta_hat), diag = K.fit_theta0_gompertz(hazard_eff, t_rebased, quiet_mask, k_anchor_weeks, gamma_per_week, deaths_arr=Dead)`.
5. `H0 = K.invert_gamma_frailty(H_obs, theta_hat if success else 0.0)` — mirror production policy when fit flags degeneracy (document exact rule in code comments; align with alpha test: use `theta_hat` when `diag['success']` and finite).
6. Build **anchored** `KCOR(t) = (H0_A/H0_B) / (H0_A/H0_B)[norm_week]`.
7. Record **KCOR_asymptote** = value at last week with valid denominator (or mean over last ~26 weeks for stability — pick one primary definition and log it in CSV metadata).
8. **KCOR_slope:** OLS slope of `KCOR(t)` vs `t_rebased` for `t_rebased ≥ 8` (same spirit as [test/sim_grid/code/generate_cox_bias_sim.py](test/sim_grid/code/generate_cox_bias_sim.py) `post_start_week`).

## Assertions, rep success, and thresholds

- Let assertions **propagate** as requested for a completed rep: `theta_hat >= 0`, `KCOR_asymptote > 0`, `isfinite`.
- Treat **estimator failure** (`diag['success']` false, `theta_hat` nan, or zero population) as a **failed rep** for the 16/20 rule rather than asserting inside the fitter.
- **Per contamination level:** require **≥16/20** successful reps; otherwise `raise RuntimeError` and skip reporting that row (or abort the whole run — choose **abort that epsilon** and continue the sweep vs **fail-fast**; recommend **fail that epsilon only** so the scan completes).
- **Sign-reversal threshold:** smallest $\epsilon$ in the tested **1D** grids (Types 1 and 2) such that **mean** `KCOR_asymptote > 1` (and record fraction of reps with `KCOR_asymptote > 1`). For **Type 3**, report the **pair** $(\epsilon_{\mathrm{dyn}},\epsilon_{\mathrm{comorb}})$ at which mean `KCOR_asymptote` first crosses 1 under a prespecified grid ordering (e.g. sort by lexicographic $(\epsilon_{\mathrm{dyn}},\epsilon_{\mathrm{comorb}})$ or by a scalar “total contamination” — document the rule).
- **Plausibility warnings:** implement `PLAUSIBLE_DYNAMIC_HVE_MAX` / `PLAUSIBLE_COMORBIDITY_MAX` against the **threshold** $\epsilon$ values **and**, where helpful, against $\epsilon/h_{\mathrm{ref}}$.

## Outputs ([test/quiet_window_contamination/out/](test/quiet_window_contamination/out/))

1. **CSV** (long format): `contamination_type` (`dynamic` / `comorbidity` / `combined`), `epsilon_dyn`, `epsilon_comorb`, `rep`, `theta0_hat_A`, `theta0_hat_B`, `theta0_err_A`, `theta0_err_B`, `k_hat_A`, `k_hat_B`, `KCOR_asymptote`, `KCOR_slope`, `sign_reversal_rep`, `fit_status_A`, `fit_status_B`, `h_ref`, `epsilon_dyn_over_href`, `epsilon_comorb_over_href` (and optional combined-total metric columns).
2. **Aggregated CSV:** group by contamination level → `mean`/`std` of errors and KCOR, `n_success`, `frac_reps_kcor_gt_1`.
3. **Figure A:** panels for Types 1 and 2 as before; **add a Type 3 panel** (e.g. heatmap of mean `KCOR_asymptote` over $(\epsilon_{\mathrm{dyn}},\epsilon_{\mathrm{comorb}})$, or mean $\theta_0$ error heatmap) **or** a separate figure file for combined — choose one, keep SI-friendly resolution.
4. **Figure B:** mean anchored `KCOR_asymptote` vs $\epsilon$ for Types 1–2 with horizontal line at 1; **for Type 3**, either a faceted surface/heatmap or a 1D slice with the clearest narrative (document choice).
5. **Stdout summary block** (multiple lines, copy-paste friendly for the paper):
   - Types 1–2: sign reversal thresholds and plausibility verdict (as before), using $h_{\mathrm{ref}}$-scaled commentary where useful.
   - **Required Type 3 line:** `Combined (dyn + comorb) sign reversal at epsilon_dyn=X, epsilon_comorb=Y` (if no grid point reverses: state `none in grid` explicitly).
   - Closing sentence tying **individual** vs **joint** implausibility to the Grok-style “both mechanisms” argument.

## File layout

- Implement as [`test/quiet_window_contamination/run_contamination_test.py`](test/quiet_window_contamination/run_contamination_test.py) **or** [`test/quiet_window_contamination/code/run_contamination_test.py`](test/quiet_window_contamination/code/run_contamination_test.py) if you want consistency with [`test/sim_grid/code/`](test/sim_grid/code/) — only substantive difference is `sys.path` bootstrapping to import [`code/KCOR.py`](code/KCOR.py).

## Paper integration (after numbers exist)

- Draft **§3.x “Quiet-window contamination stress test”** in [documentation/preprint/paper.md](documentation/preprint/paper.md) / [documentation/preprint/main.tex](documentation/preprint/main.tex): one paragraph + **Figure B** + summary table in main text; full epsilon grid in SI.
- Cross-reference alongside existing S4.3 / positive-control language so readers see the distinction: **effect-window multiplicative injection** vs **quiet-window additive non-frailty curvature**.

## Optional follow-up (out of scope unless you want it in the same PR)

- Bisection search for sharper thresholds between grid points.
- Report **both** raw and anchored KCOR to show anchoring does not hide the pathology.

```mermaid
flowchart LR
  DGP[Gompertz_frailty_DGP] --> inject[Add_quiet_only_g]
  inject --> weekly[Binomial_deaths]
  weekly --> hazards[MR_to_hazard_eff]
  hazards --> theta[fit_theta0_gompertz]
  theta --> inv[invert_gamma_frailty]
  inv --> kcor[Anchored_KCOR_t]
```
