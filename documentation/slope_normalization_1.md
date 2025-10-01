# KCOR slope normalization spec SIN method (obsolete)

Slope-from-Integral Normalization (SIN)
Version: 1.0 — 2025-09-30

## Summary
This specification defines Slope-from-Integral Normalization (SIN) as the primary and only normalization used before computing KCOR. SIN makes each cohort's log-hazard flat over a pre-specified window by estimating a per-cohort growth rate from the ratio of cumulative hazard to the starting hazard. This removes trend (slope) so week-level bumps/dips inside the window do not matter. No secondary normalization is applied by default. 

The slope over the reference period could be positive or negative for each cohort. The slope factor will be determined that normalizes the slope of each cohort to 0 over the slope Window (W) period.

Note: An optional level match step is provided if you later decide to equalize integrals over the window.

## Stratum
A stratum is defined by: (age band, cutoff date, set of dose cohorts). All choices below are applied within each stratum.

## Notation
Weekly index t = 0,1,... (week 0 is the first ISO week after cutoff).
For cohort k:
- d_k(t): deaths during week t
- N_k: cohort size at enrollment (constant)
- Y_k(t): at-risk at start of week t = N_k minus sum over u < t of d_k(u)
- h_k(t): discrete hazard = d_k(t) / Y_k(t); treat 0/0 as 0; drop rows with Y_k(t)=0 if they arise
- H_k(t): cumulative hazard up to t = sum over u <= t of h_k(u)
- R_k(t): cumulative risk up to t = 1 - exp(-H_k(t))

KCOR for pair (a,b): KCOR(t) = R_b(t) / R_a(t).

## Slope window (W)
Choose a Slope Level Window (SLW) W = [t_a, t_b] with length L = t_b - t_a + 1 weeks.
Pre-specify per stratum:
- L >= 20 weeks
- minimal expected differential exposure effect within W
- each cohort has H_k,W > 0

Record W by ISO start and end dates.

Initially, all cohorts will use the same window definition. This may be changed at a later point.

## SIN method
For each cohort k, over W do:

1) Start hazard (denominator stabilization). Let m (default = 4). Define
   hbar_k0 = (1/m) * sum from t=t_a to t=t_a+m-1 of h_k(t).

2) Integrated hazard over W:
   H_kW = sum from t=t_a to t=t_b of h_k(t).

3) Ratio:
   r_k = H_kW / (L * hbar_k0).

   Under a log-linear hazard model within W, h_k(t) approximately equals hbar_k0 * exp(beta_k * (t - t_a)).
   This implies (continuous approximation):
   (exp(x) - 1) / x = r_k, with x = beta_k * L.

4) Solve for beta_k by root-finding on f(x) = (exp(x) - 1)/x - r_k.
   - Initial guess: x0 = 2 * (r_k - 1).
   - Newton updates for 2-4 steps.
     f'(x) = (x * exp(x) - (exp(x) - 1)) / x^2.
   - If Newton fails or abs(x) > x_max (default 2), fall back to bisection on x in [-2, 2].
   Set beta_hat_k = x / L.

5) Flatten slopes for all weeks t (full follow-up). Choose t0 = midpoint of W:
   htilde_k(t) = h_k(t) * exp(-beta_hat_k * (t - t0)).

6) Rebuild cumulative hazard and risk:
   Htilde_k(t) = cumulative sum of htilde_k(t) over weeks
   Rtilde_k(t) = 1 - exp(-Htilde_k(t))

7) Compute KCOR on the flattened series:
   KCOR(t) = Rtilde_b(t) / Rtilde_a(t).

Notes:
- Within W, log(htilde_k(t)) is flat (slope approx 0).
- Outside W, no artificial trend is imposed.

## Optional level match (OFF by default; do not implement)
If you also want cohorts to have the same integral over W after flattening:
- Htilde_kW = sum of htilde_k(t) over t in W
- weights: w_k = sum of Y_k(t) over t in W
- H_ref = exp( sum_k w_k * log(Htilde_kW) / sum_k w_k )
- c_k = H_ref / Htilde_kW
- Final hazards: hhat_k(t) = c_k * htilde_k(t). Rebuild Hhat/Rhat and KCOR on hhat.

This step is once per cohort within the stratum (not per pair).

## Parameter
- m = 4 (allow 4-6)
- L between 20 and 28 weeks
- t0 = midpoint of W
- x_max = 2.0, Newton iters = 8, tolerance = 1e-10
- level_match = False by default; if True, use weighted geometric mean

## Diagnostics (report only; do not change method)
- In W, regress log(htilde_k(t)) on t; abs(slope) < 0.001/week is recommended.
- Placebo split (random halves of a cohort) processed through SIN yields KCOR approx 1 in W.
- Overlay of pre/post SIN KCOR shows trend removal inside W.

## Edge cases
- If H_kW = 0 or hbar_k0 = 0 for any cohort, the stratum is SIN-infeasible with current W. Remedy (pre-specified): expand W by +4 weeks up to max 34; if still infeasible, print error and skip the stratum.
- If abs(r_k - 1) < 1e-6, set beta_hat_k = 0.
- Very small at_risk late in follow-up: keep beta_hat_k; optional display truncation is allowed but reported.

## Inputs
Per stratum weekly table with columns:
- week (int or ISO date), cohort_id, deaths, at_risk
- N_k at enrollment if at_risk is reconstructed

Align weeks across cohorts (fill missing with deaths=0; carry at_risk).

## Outputs
- beta_hat_k per cohort
- htilde_k, Htilde_k, Rtilde_k per cohort
- KCOR(t) for requested pairs
- If level_match used: c_k, H_ref, and final series
- W, m, and solver diagnostics for audit

## Function signature (Python)
```python
def kcor_sin(df, window, m=4, level_match=False, weights='at_risk',
             x_max=2.0, newton_iters=8, tol=1e-10):
    ...
```

## Pseudocode
```python
import numpy as np
import pandas as pd

def estimate_beta_from_integral(h, m, L):
    h0 = np.mean(h[:m])
    H  = np.sum(h)
    r  = H / (L * h0)
    if abs(r - 1.0) < 1e-6:
        return 0.0
    def f(x):  return np.expm1(x)/x - r
    def fp(x):
        ex = np.exp(x)
        return (x*ex - (ex - 1.0)) / (x*x)
    x = 2.0*(r - 1.0)
    for _ in range(8):
        fx = f(x); d = fp(x)
        if d == 0: break
        step = fx/d
        x -= step
        if abs(step) < 1e-10: break
    x = max(min(x, 2.0), -2.0)  # clamp
    return x / L

def apply_sin(df, t_a, t_b, m=4, level_match=False, weights_col='at_risk'):
    L = t_b - t_a + 1
    t0 = 0.5*(t_a + t_b)
    df = df.copy().sort_values(['cohort','week'])
    df['h'] = df['deaths'] / df['at_risk']
    betas = {}
    for k, g in df.groupby('cohort'):
        maskW = (g['week']>=t_a) & (g['week']<=t_b)
        hW = g.loc[maskW, 'h'].to_numpy()
        betas[k] = estimate_beta_from_integral(hW, m, L)
    df['beta'] = df['cohort'].map(betas)
    df['h_tilde'] = df['h'] * np.exp(-df['beta'] * (df['week'] - t0))
    if level_match:
        maskW_all = (df['week']>=t_a) & (df['week']<=t_b)
        Hw = df[maskW_all].groupby('cohort')['h_tilde'].sum()
        wk = df[maskW_all].groupby('cohort')[weights_col].sum()
        logHref = (np.log(Hw)*wk).sum()/wk.sum()
        Href = np.exp(logHref)
        c = (Href / Hw)
        df['c'] = df['cohort'].map(c)
        df['h_hat'] = df['h_tilde'] * df['c']
        hcol = 'h_hat'
    else:
        hcol = 'h_tilde'
    df = df.sort_values(['cohort','week'])
    df['H_out'] = df.groupby('cohort')[hcol].cumsum()
    df['R_out'] = 1 - np.exp(-df['H_out'])
    return df, betas
```

## Reproducibility
- Fix ISO week calendar and time zone in preprocessing.
- Record software versions and seeds if bootstrap is used elsewhere.
- Store W, m, beta_hat_k, and (if used) c_k with outputs.

## Values
- Slope window is [2023-18, 2023-44]

## Claifications

- Window and baseline
  - KCOR display normalization stays at week 4 (unchanged) and is independent of t0 (midpoint of W).
  - Skip-week rules (e.g., DYNAMIC_HVE_SKIP_WEEKS) do not affect the computation and normalization. Skip-weeks only affect the KCOR computation as they normally do. 

- Data alignment and hazards
  - All cohorts share the same weekly grid (fill missing with d=0; carry Y).
  - ε for logs (e.g., log(h + ε)) used only inside the β fit is given by ε(t) = 0.5 / Y(t)
  - Final hazards are not ε-padded.

- Identifiability and feasibility
  - Require H_kW > 0 and h̄_k0 > 0; otherwise print error and stop because something is wrong

- Parameter pins
  - Fix m=5, L (20–28), x_max (2.0), Newton iters (8), tol (1e-10) as defaults.
  - Record W (ISO start/end), t0, β̂_k (with SE/CI if available) in outputs.

- Interaction with other knobs
  - SIN replaces all prior per-pair/per-dose slope normalization. You can also remove the CZECH special normalization code.
  - Note that any dataset-specific slope corrections (e.g., Czech unvax) should be removed.

- Outputs/diagnostics
  - Persist β̂_k, W, diagnostics (placebo RR(t) slope, |slope| in W after SIN).
  - Provide primary (SIN) and “no-normalization” overlays for audit.

- Baseline: KCOR display still normalized at week 4 (independent of t0).  
- Ordering: “apply SIN → compute CH/R → KCOR; no other slope normalization in primary.”  
- ε constant: if you need a constant, use: ε = 0.5 / median_Y_in_W. Otherwise, use the per (t) value above for epsilon
- Interaction: all other dataset-specific mortality rate tweaks (e.g., Czech unvax MR) should be removed
- Feasibility: keep the “expand W then skip stratum” rule and note the exact logging you’ll emit when skipping: "Unable to compute slope for YoB=xxxx"
- Output audit: ensure β̂_k, W (ISO dates), t0, and diagnostics are persisted in outputs/metadata.


