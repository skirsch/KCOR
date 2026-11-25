# Coverage of pooled multiwindow KCOR

This is the follow-up to [the initial paired fitter experiment](../paired-seed-fit/assessment.md). It tests the current revised production procedure with its fixed-rate seed and its selectable joint seed. The method, bootstrap resampling unit, safeguards and optional correction are those of the current revised edition. The separate experimental weekly survival inversion in objection 1.2 is pending approval and is outside this comparison.

## Design

| Quantity | Value |
|---|---|
| Population | 1,000,000 people per independent dataset, randomly assigned to four equally probable age-by-dose cells |
| Age strata | 1940 and 1950, with modeled weekly baseline rates of 0.002 and 0.001 |
| Frailty variances | 4 and 0.5 in dose 0 and 1, respectively, within each age |
| Calendar | Enrollment 2021-24, final/reporting week 2022-24, 53 bins |
| Stabilization and anchor | Two skipped weeks and reference at the end of the fourth accumulating week |
| Quiet windows | Zero-based enrollment weeks 2–26 and 35–52 inclusive |
| Scenarios | No wave with a gap between the quiet windows, and a nonlinear wave in weeks 27–34 inclusive, with supplied $\alpha=1.18$ |
| Wave exposure | Common to both doses within an age stratum, with an increment of three times that stratum's baseline rate in each wave week |
| Population target | 1 for each within-age anchored ratio and every pooled ratio |
| Replicates | 100 independently sampled datasets per scenario, 199 person-bootstrap draws per dataset, seed namespace 20261124 |

The generating frailty distribution describes the survivors at the modeled origin. The two skipped weeks use frailty-independent mortality with the same rate in both arms. Modeled deaths are sampled from the exact Gompertz–gamma survival probabilities, or from survival with the nonlinear wave component, independently integrated as in the 3.2 synthetic experiment. Equal underlying exposure within each age makes the target equal to one for any pooling weights.

## Production equivalence

The fast array replay calls the current production fitter, degeneracy guard, ordinary gamma inversion and optional NPH correction. It implements the workbook's age-specific anchored ratios and geometric pooling. The Monte Carlo workbook computes age weights once over its full batch of original and resampled records. The fast replay uses those same batch-wide weights. This differs from ordinary-mode runs performed separately per replicate.

[check_workbook.py](check_workbook.py) ran the actual Monte Carlo workbook entry point for the original sample and five person-bootstrap draws under each scenario and each fitter. The pooled row (YearOfBirth 0), both age-specific reporting-date rows, and reported per-age percentile intervals all matched. The production text summary omits the age-zero pooled rows, so their percentile intervals are calculated from the exported pooled values, using the same percentile rule.

The procedure retains finite values with configured diagnostic fallbacks. The study reports unconditional coverage for the whole pipeline, alongside the numbers of original datasets and bootstrap draws with insufficient-signal flags and failed fits.

Run from the repository root.

    py -3.11 -B "KCOR critique/objections/1.3/multiwindow-coverage/check_workbook.py"
    py -3.11 -B "KCOR critique/objections/1.3/multiwindow-coverage/study.py"

[Study code](study.py). The two scenarios address pooled, multiwindow coverage under a fully specified model. They use a supplied exponent, so their results describe coverage conditional on that exponent.

On noiseless expected death counts with the same design and the same pooling procedure, the fixed and joint fitters give 1.008376 and 1.000000 for the no-wave case, and 1.005736 and 0.997455 for the nonlinear-wave case. These deterministic numbers isolate the fitted method from sampling variation.

## Main coverage results

| Scenario | Fitter | Covered / 100 [exact 95% interval for coverage] | Mean KCOR [95% MC interval] | Mean interval width [95% MC interval] | RMSE |
|---|---|---:|---:|---:|---:|
| no_wave | fixed | 78/100 [68.6%, 85.7%] | 1.0098 [1.0080, 1.0116] | 0.0409 [0.0395, 0.0424] | 0.0134 |
| no_wave | joint | 94/100 [87.4%, 97.8%] | 1.0025 [0.9987, 1.0063] | 0.0744 [0.0733, 0.0756] | 0.0194 |
| nonlinear_wave | fixed | 88/100 [80.0%, 93.6%] | 1.0064 [1.0036, 1.0093] | 0.0522 [0.0512, 0.0533] | 0.0156 |
| nonlinear_wave | joint | 92/100 [84.8%, 96.5%] | 0.9980 [0.9936, 1.0023] | 0.0779 [0.0768, 0.0791] | 0.0220 |

Without a wave, the joint fitter gained coverage on 17 datasets and lost it on one (exact paired p = 0.000145). With the nonlinear wave it gained coverage on eight and lost it on four (p = 0.388). Both joint-fit coverage intervals include the nominal 95%; both fixed-fit intervals lie below 95%. The mean point estimates are closer to one with the joint fitter, while root mean squared point errors are larger. Mean interval width rises by 0.03350 [0.03218, 0.03481] without a wave and 0.02569 [0.02477, 0.02661] with a wave, using intervals for paired differences in mean widths.

Using the first 99 bootstrap draws, coverage for fixed/joint was 76%/93% and 87%/91% for the two scenarios. With all 199 draws it was 78%/94% and 88%/92%.

Neither scenario had insufficient-signal or failed fits in the original samples or bootstrap draws. The no-wave scenario exercised the zero-gap clamp: at least one gap was clamped in 99/100 fixed-fit original datasets and 100/100 joint-fit original datasets. In bootstrap draws those counts were 19,544/19,900 and 19,873/19,900. No nonlinear-wave gap was clamped in the original samples or draws.

[Full numerical results](results.json) · [All 80,000 pooled estimates, age-specific estimates and diagnostics](coverage-draws.npz)

## Additional fallback stress

Because the main study had no insufficient-signal fallbacks, an exploratory comparison uses the nonlinear-wave scenario with 50,000 people total and a four-week first quiet window (enrollment indices 2–5). The later quiet window, wave, generating parameters, target and pooling procedure are retained. A small preliminary check confirmed that this design can trigger the existing fallback. The subsequent check uses a fresh seed namespace (20261126), 50 independent datasets and 99 person-bootstrap draws per dataset. The original low-frailty investigation remains a separate deferred item. [Stress replay](fallback_stress.py).

| Fitter | Covered / 50 [exact 95% interval for coverage] | Mean interval width [95% MC interval] | Median interval width [order-statistic 95% interval] | Original datasets with insufficient-signal flags | Draws with insufficient-signal flags |
|---|---:|---:|---:|---:|---:|
| Fixed | 48/50 [86.3%, 99.5%] | 0.331 [0.311, 0.352] | 0.319 [0.284, 0.353] | 0/50 | 245/4,950 |
| Joint | 50/50 [92.9%, 100%] | 421.335 [281.891, 560.779] | 213.056 [1.480, 532.287] | 8/50 | 1,779/4,950 |

The joint fitter had 177 failed component fits and the fixed fitter had one. The existing fallback was applied and every pooled endpoint remained finite. The joint coverage is misleading if considered without interval width. For example, stress dataset 34 (zero-based) had original joint KCOR 812.675 [0.848, 744.878] with generating KCOR one. Its 1940 dose-1 fitted frailty variance was 91.84, versus its generating variance of 0.5, and the fit did not trigger the insufficient-signal flag. The result links this stress design to the deferred low-frailty recovery question. The original and resampled point estimates are stored in the archive.

The stress replay also matched the ordinary Monte Carlo workbook for a flagged original sample and five resamples under both fitters. Run `py -3.11 -B "KCOR critique/objections/1.3/multiwindow-coverage/check_workbook.py" --stress` to repeat this workbook check.

[Stress summary](fallback-stress.json) · [Stress draws and fitted variances](fallback-stress-draws.npz).

## All-window candidate follow-up

[Complete candidate coverage study](../all-window-coverage/README.md): 97/100 no-wave coverage, 90/100 nonlinear-wave coverage, verified real Monte Carlo workbook entry. The linked study also tests 400 original single-window contamination datasets. It remains experimental.
