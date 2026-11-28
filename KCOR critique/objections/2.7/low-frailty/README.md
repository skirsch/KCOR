# Numerical investigation of weak frailty recovery

This follows the null-bias audit in 2.7 and the [pooled coverage stress result](../../1.3/multiwindow-coverage/README.md#additional-fallback-stress). These are diagnostic experiments on the current selectable KCOR edition. The existing method and manuscript are retained.

## Why the data can contain very little information about frailty

With the Gompertz baseline and mean-one gamma frailty, the instantaneous cohort hazard in a quiet period is

$$
h(t)=\frac{k e^{\gamma t}}{1+\theta k(e^{\gamma t}-1)/\gamma}.
$$

Near the modeled origin, its log slope is approximately $\gamma-\theta k$. When $k$ and $\theta$ are small, changes in frailty have little effect on the observed trajectory over a short interval. The exact discrete weekly version is used in all fits here.

The single-window experiment generates counts under the exact model, at a weekly baseline rate of 0.000456 and Gompertz slope 0.085/52 per week. It uses 200 independent samples of each combination of 50,000, 500,000 and 5,000,000 people per cohort and true variance 0.5 or 4. The four fitted window lengths (4, 13, 52 and 104 weeks) use the same sampled trajectory. Both current fitters are called directly. A third diagnostic holds $k$ at its generating value and minimizes the same weekly hazard residuals over variance only.

For true variance 0.5, the model gives the following information scales for cohorts of 50,000 people. The relative depletion signal compares the frailty multiplier at the last week with the first one, and the expected count CV approximates the relative fluctuation of a weekly death count.

| Quiet weeks | Expected deaths in window | Depletion multiplier change | Weekly count CV | Local SE for variance, joint parameters | Local SE for variance, rate known |
|---:|---:|---:|---:|---:|---:|
| 4 | 91 | -0.07% | 20.9% | 204.62 | 99.90 |
| 13 | 298 | -0.28% | 20.9% | 33.63 | 16.81 |
| 52 | 1,215 | -1.20% | 20.7% | 4.04 | 2.04 |
| 104 | 2,488 | -2.50% | 20.4% | 1.36 | 0.69 |

The local information calculation differentiates the exact weekly mortality probability with respect to log rate and frailty variance and uses the expected risk set. The inverse information gives the local unconstrained likelihood uncertainty. These SE values describe the strength of the information and are especially approximate near the parameter bounds. They differ from the actual bounded least-squares estimates and their bootstrap intervals.

## Actual fits in the single-window experiment

The following are the means of the raw fitted variances before any upper-bound fallback. Brackets are 95% Monte Carlo intervals for the means over 200 independent samples. The true variance is 0.5.

| People per cohort | Quiet weeks | Original fixed-rate fitter | Joint fitter | Generating rate supplied |
|---:|---:|---:|---:|---:|
| 50,000 | 4 | 19.921 [16.080, 23.761] | 41.784 [35.299, 48.269] | 36.058 [30.199, 41.917] |
| 50,000 | 13 | 11.586 [9.236, 13.936] | 14.469 [11.636, 17.302] | 8.934 [7.207, 10.660] |
| 50,000 | 52 | 2.397 [1.911, 2.884] | 1.748 [1.387, 2.110] | 1.171 [0.971, 1.371] |
| 50,000 | 104 | 1.208 [0.977, 1.439] | 0.730 [0.604, 0.857] | 0.576 [0.501, 0.652] |
| 5,000,000 | 4 | 2.087 [1.667, 2.507] | 8.705 [6.948, 10.462] | 4.235 [3.391, 5.080] |
| 5,000,000 | 52 | 0.546 [0.478, 0.613] | 0.531 [0.483, 0.578] | 0.489 [0.464, 0.513] |
| 5,000,000 | 104 | 0.485 [0.446, 0.525] | 0.507 [0.490, 0.525] | 0.499 [0.489, 0.509] |

Even with the generating rate supplied, variance is poorly recovered in a short quiet window. The joint estimates in the 50,000-person, four-week case include 97/200 below 1e-6 and 64/200 at the upper bound. The latter receive an applied variance of zero under the current guard. The per-dataset distribution is strongly asymmetric near the nonnegative constraint. The original fixed-rate method additionally estimates the rate from the first four observed hazards, which contain both noise and depletion.

On exact, noiseless counts and a 52-week window, the original fixed-rate fit recovers 0.472387 for true variance 0.5 and 3.773386 for true variance 4. The joint fit recovers both at their generating values to the precision shown. The four-week noiseless fixed fit obtains 0.137489 for true variance 0.5, because the early-rate average already absorbs part of the depletion effect.

The estimates for all 72 combinations, including central 95% ranges across independent estimates, rates, boundary counts and the applied variance means, are in [fit-summary.csv](fit-summary.csv). [Information calculations](information.csv) · [Noiseless controls](noiseless.csv) · [14,400 individual fit records](fit-estimates.csv).

A constant residual rescaling in an isolated solver replay tests whether small numerical residuals cause premature termination. Dividing the least-squares residual by the generating hazard rate preserves its mathematical minimizer. Across the 1,600 matched fixed and joint fits at 50,000 and 5,000,000 people with 4 and 52 weeks, mean absolute differences in raw variance ranged from less than 1e-6 to 0.0024, with a maximum individual difference of 0.0513. Thus the large observed failures survive substantially tighter effective gradient tolerances. [Tolerance comparison](tolerance-comparison.csv).

## The extreme multiwindow case from 1.3

Stress dataset 34 had 12,402 people at modeled entry in the 1940 dose-1 cell. The generating baseline rate was 0.002 and generating variance was 0.5. Its four first-window death counts were 33, 25, 16 and 24. Conditional on the observed risk sets, the generating model predicted 24.79, 24.74, 24.70 and 24.69. The joint seed interpreted the downward fluctuation as a larger baseline rate (0.002898) and variance 92.064. The best first-window fit with variance fixed at 0.5 had squared hazard residual 9.41e-7, whereas the joint seed attained 4.10e-7. The large variance is preferred by that noisy-window objective.

The later quiet window does not repair the estimate. Reconstruction using the erroneous seed gives a gap offset of 378.968, which the pooled refit accepts as cumulative excess. The true wave exposure throughout the supplied wave was 0.048. In four subsequent theta refits the estimate converges to 91.8409 and the implied offset to 370.376. The fixed-rate fitter starts from variance 19.324 and ends at 0.2854 with gap offset 0.04843. Both fits report status OK in this case.

The initial four-week fit has an unconstrained local variance SE of about 44.9 even under the true model. The dangerously high seed is then amplified by the exponential gamma inversion and the offset iteration. A convergence check only establishes that the iteration reached a stable value.

| Substitution in this single problematic cohort | Pooled KCOR [95% person-bootstrap interval] |
|---|---:|
| Fitted rate and variance | 812.675 [0.848, 744.878] |
| Generating variance, fitted rate | 0.997 [0.354, 1.193] |
| Generating variance and rate | 0.996 [0.353, 1.193] |

The other cohorts, weights, observed death histories and wave correction are identical, including in the matching 99 resamples. The substitution establishes that this cohort's variance error accounts for the extreme pooled result. Its interval is still wide because uncertainty in the other cohorts persists.

[First-window counts and fitted predictions](stress-first-window.csv) · [Profiled first-window objective](stress-seed-profile.csv) · [Fit trace and matched substitutions](stress-trace.json).

## Link to the empirical Table 8 results

The first joint seed is based on 11 modeled quiet weeks for the 2021 enrollment and 4 quiet weeks for the 2022 enrollment. Its local log-rate/variance correlation in the 2022 data is 0.997–0.998 (the analogous 2021 estimates range from 0.836 to 0.867). This reflects the strong dependence of a level estimate on the frailty parameter over a short delayed window. In the two 2022 groups that hit the final variance bound, the joint seeds were 49.489 and 76.859. Their local variance SE scales were 35.872 and 26.960.

These calculations use the fitted seed model and the actual first-window risk sets. They measure the information geometry of the fit. The true Czech frailty and the shape of the mortality preceding the first quiet window are unknown. [All 18 original seed fits and local information](table8-first-window-information.csv).

## Reproduce

Run from the repository root.

    py -3.11 -B "KCOR critique/objections/2.7/low-frailty/investigate.py"
    py -3.11 -B "KCOR critique/objections/2.7/low-frailty/check_tolerance.py"
    py -3.11 -B "KCOR critique/objections/2.7/low-frailty/trace_stress.py"
    py -3.11 -B "KCOR critique/objections/2.7/low-frailty/table8_information.py"

The investigation identifies limited statistical information, the early-rate versus joint-fit tradeoff, and the amplification through the gap reconstruction. It provides evidence for the next method discussion.

## Information from all quiet windows and uncertain gap offsets

The earlier single-window approximation describes the first-window seed. With all quiet windows, the leading-order information for variance with known offsets grows with the weighted spread in the modeled time indices. In the Czech configuration the 41 quiet weeks of 2021 span modeled weeks 0–144, and the 30 quiet weeks of 2022 span modeled weeks 14–110. The time separation is substantial.

The exact local calculation uses the proportional Gompertz–gamma working model of the core fitter. Write $F(t)=(e^{\gamma t}-1)/\gamma$ and $X(t)=kF(t)+\Delta_j$ in quiet window $j$. Each observed quiet week has integrated hazard

$$
q_{ab}=\frac{1}{\theta}\log\frac{1+\theta X(b)}{1+\theta X(a)}.
$$

Across an entire intervening gap the same expression applies at its two boundaries, allowing the cumulative offset to change. The observed count over that gap provides one binomial survival transition, without specifying its weekly wave shape. A preceding gap is included if the first quiet window is delayed.

For death probability $p_i=1-e^{-q_i}$ and starting risk set $Y_i$, the expected local information for parameters $\beta=(\log k,\theta,\Delta_1,\ldots)$ is

$$
I_{uv}=\sum_i Y_i\frac{1-p_i}{p_i}
\frac{\partial q_i}{\partial \beta_u}
\frac{\partial q_i}{\partial \beta_v}.
$$

The uncertainty of $\theta$ with rate and offsets estimated is the inverse square root of its information after projecting out their nuisance derivatives. Equivalently, for nonsingular nuisance information it is

$$
\operatorname{SE}(\hat\theta)\approx
\left[I_{\theta\theta}-I_{\theta\eta}I_{\eta\eta}^{-1}I_{\eta\theta}\right]^{-1/2},
$$

where $\eta$ contains the rate and gap parameters. The reference variance is 0.5. The reference rate is calibrated to the observed quiet hazards at that variance, and each reference gap transition matches the observed cumulative hazard across the gap. These are local information calculations for a hypothetical joint fit under the proportional gamma model. They permit local variation of signed offsets. The empirical frailty truth is unknown, nonlinear-wave selection can violate this gamma model, and an efficient fit is an information benchmark rather than an assumption about the current estimator.

Examples from the actual Czech age-dose data illustrate why retaining the gap counts matters.

| Enrollment | Birth decade | Dose | First-window local SE | All quiet weeks, gaps unknown and gap counts omitted | All quiet and gap counts, gaps known | All quiet and gap counts, gaps unknown |
|---|---|---:|---:|---:|---:|---:|
| 2021_24 | 1940s | 0 | 7.315 | 2.265 | 0.142 | 0.189 |
| 2021_24 | 1940s | 2 | 11.038 | 3.130 | 0.175 | 0.299 |
| 2022_06 | 1940s | 0 | 47.446 | 3.366 | 0.283 | 0.628 |
| 2022_06 | 1940s | 3 | 45.142 | 3.067 | 0.248 | 0.628 |

The complete [multiwindow information table](multiwindow-information.csv) includes every 1940s, 1950s and 1960s age-dose group in the two actual Czech enrollments and all eighteen Table 8 fitted subgroups. It gives the local low-frailty resolution thresholds for 95% intervals of approximately ±50% and ±25%. These use a reference variance of 0.5 and should not be extrapolated as actual attainable precision at very high frailty.

### Propagating the uncertainty through the current sequential fit

The [local propagation replay](propagate_multiwindow.py) perturbs each weekly integrated hazard in the original Table 8 data and runs the existing joint-seed and multiwindow fitter again. For weekly binomial mortality, $\operatorname{Var}(\hat h_i)\approx p_i/[Y_i(1-p_i)]$. With the numerical sensitivity of the final fitted variance to each week, the first-order propagated variance is

$$
\operatorname{Var}(\hat\theta)\approx
\sum_i\left(\frac{\partial\hat\theta}{\partial h_i}\right)^2
\frac{p_i}{Y_i(1-p_i)}.
$$

This includes the sensitivity of the seed rate, offset reconstruction and pooled refit to each week. It does not freeze any reconstructed offset when a weekly hazard is perturbed. The partial variance contributions below use the original death-count variance estimate. Two perturbation sizes agree locally for the displayed interior fits. The linearization cannot account for switches to a different optimizer branch or bounds under large perturbations. The full person-bootstrap intervals supply the empirical sampling comparison.

| Enrollment | Table 8 subgroup | Pseudo-dose | Fitted variance [95% bootstrap interval] | Local propagated SE | Share from first quiet week block | Share from later quiet weeks | Share from other modeled weeks |
|---|---|---:|---:|---:|---:|---:|---:|
| 2021_24 | unvaccinated | 0 | 1.053 [0.788, 1.336] | 0.137 | 86.92% | 12.83% | 0.24% |
| 2021_24 | unvaccinated | 1 | 3.105 [2.484, 3.741] | 0.329 | 88.02% | 11.68% | 0.30% |
| 2021_24 | unvaccinated | 2 | 8.186 [6.435, 9.978] | 0.940 | 88.92% | 10.76% | 0.32% |
| 2022_06 | unvaccinated | 0 | 0.820 [0.368, 3.428] | 0.189 | 80.53% | 19.29% | 0.18% |
| 2022_06 | unvaccinated | 1 | 11.901 [0.000, 15.060] | 22.111 | 99.96% | 0.04% | <0.01% |
| 2022_06 | unvaccinated | 2 | 24.560 [0.000, 42.569] | 41.334 | 99.94% | 0.05% | <0.01% |

For the 2021 examples, the propagated local SE agrees closely with the empirical bootstrap width. The 2022 intervals show the difficulty of linearizing a constrained and sometimes unstable seed. Most of the local propagated uncertainty comes from mortality in the first quiet window, even though all four or three quiet windows are used in the final fit. Sampling variation in the reconstructed gap deaths is much less influential directly in these examples. Errors originating in the first window can still propagate into the gap offsets, as the stress trace shows.

[All 18 subgroup results and branch checks](multiwindow-pipeline-propagation.csv). Reproduce both calculations with

    py -3.11 -B "KCOR critique/objections/2.7/low-frailty/multiwindow_information.py"
    py -3.11 -B "KCOR critique/objections/2.7/low-frailty/propagate_multiwindow.py"

### All-window joint-fitting trial

[Isolated all-window experiment](all-window-joint/README.md): estimates the same common k and theta across all quiet windows, investigates the gap-offset feedback and optimizer starting branch. No adoption or paper change.

### Theoretical information limits

[Full multi-window Fisher information and KCOR endpoint analysis](theoretical-limits/README.md), including exact nonlinear wave survival, unknown gap exposures, low-frailty likelihood profiles, Czech data references, and comparison with the experimental fitter. Separately recorded mathematical contribution, not adopted into the paper.
