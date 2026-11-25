# Weekly survival inversion during the optional nonlinear wave

**Trial pending lead-author approval.** The author asked to implement and test the in-wave use of the survival equation already adopted for post-wave continuation in §2.7.3. The [selectable revised method](<../../../iterative review/revised/code/KCOR.py>) supplies the fitted baseline rate, frailty variance, wave calendar, stabilization skip, and exponent. The experiment replaces the wave correction callable in its isolated replay process. The current revised and checkpoint editions remain available.

## Equation

The original intent is to compare cumulative hazard at frailty one, retaining baseline and wave mortality after correcting for survivor selection. Under the existing individual hazard model, survival at a week boundary is

$$
L(B,W)=E_{Z\sim\Gamma(1/\theta,\theta)}
\left[\exp(-ZB-Z^\alpha W)\right].
$$

Let $B_{j-1},W_{j-1}$ denote the preceding week's estimated exposures and $b_j$ the integrated fitted Gompertz increment for week $j$. Let $H_{\rm obs}$ be observed cumulative hazard. The weekly conditional equation is

$$
\frac{L(B_j,W_j)}{L(B_{j-1},W_{j-1})}
=\exp\left[-(H_{\rm obs,j}-H_{\rm obs,j-1})\right].
$$

For a week with mortality above the fitted baseline conditional on the estimated survivors, set $B_j=B_{j-1}+b_j$ and solve for the nonnegative wave increment $W_j-W_{j-1}$. The conditional survival depends strictly monotonically on that increment. The corrected total is $B_j+W_j$, and the solved exposures are carried forward to the next week. At wave entry use the observed inverted total through the preceding modeled week and $W=0$, as in 2.4. After wave exit use the already adopted post-wave equation with both recovered exposures as input.

### Deficit handling in this trial

Sampling fluctuation can produce mortality below the fitted baseline conditional on the estimated survivors. In that case a positive wave increment cannot satisfy the equation with the fitted baseline increment. The current trial retains $W_j=W_{j-1}$ and solves the conditional equation for a nonnegative baseline increment instead. Each subsequent week again proposes the fitted Gompertz increment. This is a trial attribution choice. It permits baseline accumulation to differ from its fitted path in deficit weeks and preserves the prior wave exposure. Every such week is counted for review. Exact numerical reconciliation is conditional on the chosen attribution and fitted inputs.

## Implementation and checks

- [trial.py](trial.py) evaluates the survival function using the same gamma probability quadrature as the existing post-wave solver. Bracketed univariate root-finding gives the exposure increments.
- [check.py](check.py) integrates the generating survival independently. It covers seven variances, including zero, and four exponents from 0.7 to 2. The four-week wave includes a week with zero wave increment. The cases also cover entry after the stabilization skip, a baseline deficit, a week with zero deaths, and the wave tail. All cases recovered the generating cumulative hazard at every boundary within 3e-14 absolute error for the main grid.
- [replay.py](replay.py) uses the Table 8 enrollment populations and the same 999 person-bootstrap draws per enrollment as the [entry-moment edition](../entry-moment/README.md). Both the joint first-window fitter and the original fixed-rate fitter are included. The fitted exponent is the existing forced Czech sensitivity value 1.18. The bootstrap is conditional on that exponent.

Run from the repository root.

    py -3.11 -B "KCOR critique/objections/1.2/weekly-survival/check.py"
    py -3.11 -B "KCOR critique/objections/1.2/weekly-survival/replay.py"
    py -3.11 -B "KCOR critique/objections/1.2/weekly-survival/check_workbook.py"

Known-input recovery establishes the equation and numerical inversion. Accuracy for estimated baseline and frailty, and repeated-sample coverage, are separate questions.

## Table 8 comparison

Both methods used the same 999 person-bootstrap draws for each enrollment. Intervals are pointwise 95 percent percentile intervals. The reported date is December 26, 2022. The six trial joint-fit point estimates independently matched the ordinary workbook entry point to relative tolerance 1e-10. Fit flags and failed-fit counts matched the preceding joint-fit replay. The maximum absolute discrepancy in cumulative observed hazard at the weekly solves was 3.86e-15.

| Fitter | Enrollment | Pseudo-dose | Current KCOR [95% interval] | Trial KCOR [95% interval] | Width before | Width trial |
|---|---|---|---:|---:|---:|---:|
| joint | 2021_24 | 1 vs 0 | 1.0083 [0.9564, 1.0584] | 1.0081 [0.9562, 1.0582] | 0.1021 | 0.1020 |
| joint | 2021_24 | 2 vs 0 | 1.0313 [0.9739, 1.0945] | 1.0309 [0.9736, 1.0940] | 0.1205 | 0.1204 |
| joint | 2021_24 | 2 vs 1 | 1.0227 [0.9640, 1.0889] | 1.0225 [0.9637, 1.0885] | 0.1249 | 0.1248 |
| joint | 2022_06 | 1 vs 0 | 1.0302 [0.3219, 1.2358] | 1.0305 [0.3213, 1.2359] | 0.9138 | 0.9147 |
| joint | 2022_06 | 2 vs 0 | 1.0724 [0.3413, 1.1927] | 1.0729 [0.3413, 1.1951] | 0.8514 | 0.8537 |
| joint | 2022_06 | 2 vs 1 | 1.0409 [0.8471, 1.2133] | 1.0412 [0.8483, 1.2134] | 0.3662 | 0.3651 |
| fixed | 2021_24 | 1 vs 0 | 1.0114 [0.9581, 1.0623] | 1.0112 [0.9581, 1.0618] | 0.1041 | 0.1038 |
| fixed | 2021_24 | 2 vs 0 | 1.0280 [0.9725, 1.0940] | 1.0276 [0.9721, 1.0935] | 0.1215 | 0.1214 |
| fixed | 2021_24 | 2 vs 1 | 1.0165 [0.9603, 1.0858] | 1.0162 [0.9600, 1.0855] | 0.1256 | 0.1255 |
| fixed | 2022_06 | 1 vs 0 | 1.0073 [0.9647, 1.0485] | 1.0083 [0.9655, 1.0500] | 0.0838 | 0.0844 |
| fixed | 2022_06 | 2 vs 0 | 1.0128 [0.9718, 1.0630] | 1.0175 [0.9767, 1.0707] | 0.0912 | 0.0941 |
| fixed | 2022_06 | 2 vs 1 | 1.0055 [0.9618, 1.0563] | 1.0091 [0.9658, 1.0622] | 0.0945 | 0.0963 |

All 12 trial intervals include one. The largest joint point movement was about 0.00054. The larger fixed-fitter movement was about 0.00468 in the 2022 comparison 2 vs 0. Full paired change intervals are in the [comparison JSON](table8-comparison.json).

| Fitter | Enrollment | Original deficit weeks / evaluated subgroup-weeks | Bootstrap deficit weeks / evaluated subgroup-weeks |
|---|---|---:|---:|
| joint | 2021_24 | 2/216 | 2,734/215,760 (1.3%) |
| joint | 2022_06 | 11/35 | 10,347/33,870 (30.5%) |
| fixed | 2021_24 | 0/216 | 1,474/214,320 (0.7%) |
| fixed | 2022_06 | 15/45 | 18,239/44,955 (40.6%) |

The denominator counts fitted subgroup-weeks with positive applied frailty evaluated by the trial solver. Weeks routed through the existing zero-frailty shortcut are excluded. The below-baseline attribution occurs frequently in the 2022 enrollment. The comparison therefore tests the full trial, including its stated deficit handling, and the lead-author decision covers that rule as well.

[Full-precision estimates, paired change intervals and diagnostics](table8-comparison.json) · [All matched trial draws](table8-draws.npz)

## Survivor frailty by week

The [weekly comparison](weekly-comparison.csv) reports every modeled wave week for the nine fitted subgroups in each Table 8 enrollment, using the joint fitter. Each estimate has its pointwise 95 percent percentile interval from the same 999 person resamples. [weekly_comparison.py](weekly_comparison.py) reproduces the table and checks the trial reporting-date KCOR values against the previous replay.

The old gamma proxy is the scale $\theta/(1+\theta\widetilde H_0)$ passed to the mean-one gamma moment formula in later weeks. At the first wave week, the old correction uses the opening moment divided by the opening mean. The table identifies that actual divisor separately from the computed end-of-week proxy. The trial uses the starting gamma law reweighted by cumulative survival at each week boundary. Its mean and variance are on the original frailty scale, and its relative variance $\operatorname{Var}(Z\mid\text{alive})/E[Z\mid\text{alive}]^2$ makes comparison with a mean-one proxy possible. The trial survivor moment ratio describes the end-of-week survivors. The weekly inversion solves the full survival equation.

Wave exposure is cumulative, measured on the frailty-one hazard scale. The two corrected hazard columns give the complete adjusted cumulative hazard at the end of each modeled wave week.

## Combined all-window fitter follow-up

[Paired exact-wave and all-window test](../all-window-weekly-survival/README.md) completed. Coverage remains 90/100 and 48/50 in the main and stress nonlinear scenarios, respectively. Noiseless true-frailty substitutions isolate the remaining model mismatch. Table 8 endpoint changes remain small. Approval pending.
