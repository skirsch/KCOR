# KCOR v7
The original KCORv6 derives a estimated FIXED theta for each cohort from a 2023 quiet window and used the fixed theta for the correction.

That is very crude. It's fine for the vaccinated (perfect actually since it is impacted by the shots and is low).

But for the unvaccinated, a fixed correction is not precise. We can do this better.

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

