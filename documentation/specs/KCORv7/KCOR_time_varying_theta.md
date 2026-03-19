# KCOR v7
The original KCORv6 derives a estimated FIXED theta for each cohort from a 2023 quiet window and used the fixed theta for the correction.

That is very crude. It's fine for the vaccinated (perfect actually since it is impacted by the shots and is low).

But for the unvaccinated, a fixed correction is not precise. We can do this better.

We can identify quiet windows in the data where there is no COVID or other excess mortality. 

From each point in that quiet window, we can use H(t) to estimate the theta(0) value.

But the important caveat is that the quiet window used for estimation should be as close as possible to enrollment because once depletion occurs, you cannot recover theta(0). It's like starting with a group of 100 year olds and 10 year olds and if you look 2 years later when all the 100 year olds are dead, there's no way to see how fast they were depleting.

### The ideal v7 procedure:
So the v7 procedure is:

In each quiet window, 
- use the same nonlinear regression as v6 to estimate θ(t*) at each timepoint t* within that window
- You now have θ(t*) and H(t*) at each anchor point
- Apply the inversion formula to get θ(0) from each anchor point
- Pool across all anchor points and windows → single MLE θ(0)
- Propagate forward using θ(t) = θ(0) / (1 + θ(0)·H(t))²


### what the code actually does right now
For each quiet window w:
    1. Fit one nonlinear regression across all timepoints in w
       → yields θ_anchor for window w
    2. For each timepoint t in w:
       - Compute θ(t) from closed form using θ_anchor and H(t)
       - Invert to get θ(0) estimate at timepoint t

Pool all θ(0) estimates across windows → single MLE θ(0)

### comment from Cursor
Your lines 8-20 are a good conceptual summary of v7, and the current code does implement the key pillars:

multi-window theta0 estimation,
stable --branch inversion,
discriminant checks with skip counts,
pooled theta0,
forward propagation into normalization.

One important implementation detail differs from that wording:

In code/KCOR.py, v7 currently fits one theta_anchor per quiet window (via fit_k_theta_cumhaz on window data), then derives per-timepoint theta(t*) from the closed form within that window before inverting to theta0.
So it does NOT run a separate nonlinear regression at every anchor timepoint.

Also, the normalization path applies theta0 via invert_gamma_frailty(H_obs, theta_for_h0) (with theta_for_h0 = theta0_hat in v7 mode), rather than explicitly writing a per-timepoint theta(t) adjustment step in that section.

If you want strict one-to-one alignment with 8-20 wording, I can refactor the estimator to do per-anchor regressions explicitly and update comments/spec text to match exactly.

## the approach

In KCOR v7, get rid of the constant theta assumption and instead used the closed form to compute theta at any point in time given you know θ(t₀):

θ(t) = θ(0) / (1 + θ(0)·H(t))²

So The entire trajectory of θ is fully determined by:

θ(0) — estimated from your pre-COVID quiet window
H(t) — the cumulative baseline hazard, which you already observe in your data


But it's even better because if we know theta at ANY point in time (t*), we can compute the theta 0 from that point. Then from there we use the formula above to compute the theta value as a function of time.

θ(0) = [1 - 2θ(t*)·k ± √(1 - 4θ(t*)·k)] / (2θ(t*)·k²)
Where k = H(t*)

Which root to take:
The + root gives the physically meaningful solution since θ(0) > θ(t*) > 0, and you can verify by checking that plugging back in recovers θ(t*).

Validity condition:
Requires 1 - 4θ(t*)·H(t*) ≥ 0, i.e.:
H(t*) ≤ 1/(4θ(t*))
Which will be satisfied for reasonable cumulative hazards and frailty variances in your cohort data.


So let's create KCOR v7 which has some new parameters in the yaml file and that makes the adjustment to the H(t) values based on the current value of theta computed from theta(0)

For each quiet window w:
    For each timepoint t in w:
        Compute H(t) from observed cumulative mortality
        Compute θ(t) from observed data
        Invert closed form → θ(0) estimate at t
        
Pool all θ(0) estimates across all windows and timepoints
→ Single MLE estimate of θ(0)
→ Confidence interval from variance of estimates
→ Goodness of fit from scatter/drift across windows

Once you have theta(0), you can derive theta(t) for any t value and make the proper correction to H(t) using the correct theta at that point in time.

So the steps are:
1. Compute theta(0) from a set of quiet windows. They should all compute the same value.
2. Use theta(0) in the formula along with unadjusted cum hazards (H(t)) to compute the value for theta at each point in time
3. Compute adjusted H(t) at each point in time just like we do now, except using the theta value for that time we just computed instead of a constant theta (which is what we did in v6)

### yaml additions
time_varying_theta: 
   enabled: true
   apply_to: unvaccinated_only           # unvaccinated_only | vaccinated_only | both_cohorts
   diagnostics:
      plot_theta0_estimates_by_window: true
      plot_theta_trajectory: true    # plot the full θ(t) trajectory from t=0 to end of follow-up.
      report_consistency_test: true
   theta_estimation_windows:   # start ISOweek, stopISOweek, window name
      - ['2023-22', '2023-37', 'post COVID']   
      - ['2022-22', '2022-25', 'post booster']
      - ['2021-26', '2021-36', 'post primary']

In the console log, tell me the goodness of fit for the the theta values for each cohort (d0, d1, d2, etc).

## consistency test output
```
θ(0) estimates by window:

Window 2021-26 to 2021-36:  mean=0.42  sd=0.03  n=10
Window 2022-22 to 2022-25:  mean=0.41  sd=0.04  n=4
Window 2023-22 to 2023-37:  mean=0.43  sd=0.02  n=16

Pooled θ(0) MLE:            0.42
95% CI:                      [0.39, 0.45]

Cross-window variance:       0.001
Within-window variance:      0.003
F-statistic:                 1.21
p-value (drift test):        0.31  → no significant drift detected
```

## theta0 estimates by window plot

It would plot the individual θ(0) estimates derived from each timepoint, grouped by window.

What it shows:

X axis: calendar time (the actual date of each timepoint)
Y axis: θ(0) estimate derived from that timepoint via the inversion formula
One dot per timepoint — each dot is an independent estimate of θ(0)
Color coded by window — all dots from window 1 in blue, window 2 in red, window 3 in green etc.
Horizontal line: the pooled MLE θ(0) estimate
Shaded band: 95% CI around the pooled estimate
What you would expect to see if the model is correct:

All dots scattered horizontally around the pooled line
No systematic trend across time
Dots from different windows at the same vertical level
Scatter is random noise, not structured
What pathological patterns would look like:

Pattern	Interpretation
Dots drift upward over time	θ(0) estimates increasing — model misspecification
Dots drift downward over time	Systematic drift — frailty not truly gamma
One window consistently higher	Something non-stationary happened between windows
High scatter within one window	That window is noisy — maybe not truly quiet
Outlier dots at window edges	Edge effects — consider trimming window boundaries
Why this plot is more informative than just the consistency report numbers:

A table tells you the F-statistic
This plot tells you where and how the model breaks down if it does
Visually distinguishes random scatter (acceptable) from systematic drift (problematic)
Immediately shows if one window is an outlier vs. all windows drifting together
Makes the identification assumption transparent — you can see exactly what data is driving θ(0)
The key diagnostic value:

If all three windows show dots clustered tightly around the same horizontal line, that is extremely strong validation that:

Gamma frailty is correctly specified
θ(0) is stably identified
The time-varying correction in v7 is built on a solid foundation
That plot belongs in the paper.

## Two important corrections
1. Use the - root selection, not the + selection. And use the umerically stable form which is:
θ(0) = 2θ(t) / [1 - 2θ(t)H(t) + √(1 - 4θ(t)H(t))]

2. Add validity check in code by verifying that:
1 - 4θ(t)H(t) ≥ 0 
at every anchor point (i.e., a point used to determine theta(0)) and warn if violated

### New info
Done — I added the diagnostic table fields and reran 2021_24.

What the step2_ratio tells you:
For each window, step2_ratio = theta_window_start / step2_theta_limit:
Windowstep2_ratioInterpretationpost primary0.084theta_window_start is 12x below the limit — passes easilypost COVID6.38theta_window_start is 6x above the limit — failspost booster18.3theta_window_start is 18x above the limit — fails badly

The ratio is a clean dimensionless diagnostic. Values below 1.0 pass, above 1.0 fail. The magnitude tells you how far from the boundary you are.

Add step2_ratio as a flagged field in the consistency report:
Window        step2_limit  step2_ratio  anchors_used  status
post primary  189.6        0.084        11            ✓ CONTRIBUTING
post booster  3.13         18.3         0             ✗ TRANSPORT LIMIT
post COVID    1.76         6.38         0             ✗ TRANSPORT LIMIT

And add a cohort-level flag:
single_window_estimate: true    # when only one window contributes
consistency_check: unavailable  # cross-window validation not possible
reliability: reduced            # flag for downstream interpretation



Let me sharpen the diagnostic criteria so the threshold is principled rather than arbitrary.

* * * *

**The identifiability condition for a window:**

The gamma frailty theta is identified from the **curvature** of H(t) within the window. The amount of curvature available is captured by:

```
z_end = theta_anchor * H_rel_end

```

This is the key quantity. When z\_end is small, the gamma frailty curve is nearly indistinguishable from a straight line --- theta is weakly identified and any value near zero fits equally well.

Specifically:

-   z\_end >> 1 → strong curvature, theta well identified
-   z\_end ~ 0.1 → weak curvature, theta poorly identified
-   z\_end << 0.1 → effectively linear, theta unidentified

* * * *

**A principled threshold:**

From the Taylor expansion of the gamma frailty model, the fractional curvature signal is approximately 2-z\_end. A window contributes meaningful identification when:

```
z_end = theta_anchor * H_rel_end ≥ z_min

```

A reasonable default for z\_min is **0.01** --- below this the curvature signal is less than 2% of the linear term and theta is essentially unidentified from that window.

* * * *

**The filter spec for Cursor:**

```
# After fitting theta_anchor and computing H_rel for a window:
z_end = float(theta_anchor) * float(h_rel[-1])

if z_end < Z_MIN_IDENTIFIABLE:  # default 0.01, configurable
    flag window as "weak_id"
    skip all anchors from this window
    log reason: f"z_end={z_end:.3e} < z_min={Z_MIN_IDENTIFIABLE:.3e}"
    continue to next window

```

* * * *

**What to log per window:**

```
Window        theta_anchor  H_rel_end  z_end     status
post COVID    1.2e-12       0.0148     1.8e-14   ✗ WEAK_ID  (z_end < 0.01)
post booster  9.5e-02       0.0033     3.1e-04   ✗ WEAK_ID  (z_end < 0.01)
post primary  16.0          0.0148     0.237     ✓ IDENTIFIED

```

* * * *

**YAML config addition:**

```
time_varying_theta:
  theta_identifiability:
    z_min: 0.01          # minimum theta*H_rel_end to accept window
    flag_weak_id: true   # log but don't crash on weak windows

```

* * * *

**Important interaction with the step2 filter:**

The two filters catch different failure modes and should be applied in order:

1.  **z\_end < z\_min** → weak identifiability → skip entire window before any inversion
2.  **step1 discriminant** → local frame inversion failure → skip individual anchor
3.  **step2 discriminant** → transport failure → skip individual anchor

A window that passes z\_end check but fails step2 for all anchors is a **transport limit** (1940 case). A window that fails z\_end is a **flat likelihood** (1930 case). These are distinct and the log should distinguish them clearly.

* * * *

**One thing to verify empirically:**

After applying the z\_end filter, check whether the cross-window ratios for YoB=1930 improve. If post COVID is filtered out as weak\_id, the remaining windows should give more consistent theta0 estimates. If they still disagree after filtering, that points to a deeper model misspecification for 1930 rather than an identifiability artifact.

### newer
What Cursor needs to implement:

Sort quiet windows by start date
Select the first window where window_start > enrollment_date
Estimate theta0 from that window only
For all subsequent quiet windows: compute theta0 independently and report cross-window ratio as a diagnostic — but do not pool
Remove z_min and theta_min filters entirely — they are no longer needed as gatekeepers
Flag cohorts where no quiet window exists after enrollment
