# Theoretical information limits for KCOR and gamma frailty

Exploratory mathematical investigation associated with root cause 2.7, also relevant to 1.3 (uncertainty), 3.1 (boundary/diagnostic instability) and 3.2 (quiet-window information). No paper or official method change. The all-window fitting experiment remains a candidate for lead-author approval.

## Model and derivation

For mean-one gamma frailty Z with variance theta, let B be cumulative proportional hazard, and W cumulative wave hazard acting through Z^alpha. With the Gompertz slope gamma fixed,

    F(t) = (exp(gamma*t)-1)/gamma
    B(t) = k F(t) + cumulative proportional gap excess
    S(t) = E[exp(-Z B(t) - Z^alpha W(t))].

For alpha=1 the exact survival is (1+theta(B+W))^(-1/theta). For alpha=1.18, evaluate the exact integral numerically, rather than assuming that the survivor frailty distribution after the wave is gamma. Alpha is treated as fixed; estimating it jointly would require another nuisance dimension and can only weaken regular information on the other parameters.

We include every configured quiet week, with the same common k and theta across windows. Every nonquiet interval has an unknown aggregate excess hazard, not a known offset or a stipulated within-gap shape. Split gaps at the configured wave start/end and the KCOR anchor and reporting endpoint, so that all observations necessary for the target ratio are retained. Before the first quiet window, any gap is included too. For fixed exponent within a free gap, total deaths and survival across the gap provide the relevant likelihood factor when the hazard within it is unrestricted. Later quiet weeks constrain the shared scale and frailty through the survival history. In synthetic runs, nonwave gap excess is truly zero, but not supplied to the fitter as a known quantity for information calculation.

For modeled interval i, q_i = log S(a_i)-log S(b_i), p_i=1-exp(-q_i), and expected risk Y_i=N*S(a_i). Conditional binomial information is

    I(beta) = sum_i Y_i (1-p_i)/p_i * grad(q_i) grad(q_i)^T

for beta=(log k,theta,all unknown gap excesses). This follows from dp_i/dbeta=(1-p_i) dq_i/dbeta. The noise in gap death counts is included, and they are not treated as known exposures. The inverse matrix includes covariance of k, theta and all nuisance parameters. Use a scaled SVD to obtain the inverse. Expected risk is calculated from model survival rather than substituting random observed risk counts.

    SE(theta) >= sqrt((I^-1)[theta,theta])
    G = (B(report)+W(report))/(B(anchor)+W(anchor))
    SE(log G) >= sqrt(grad(log G)^T I^-1 grad(log G))
    log KCOR = log G_numerator-log G_denominator

For disjoint cohorts add the log-growth variances; for independent age strata combine with squared fixed pooling weights. At equal true ratios, the random pooling-weight derivative vanishes to first order. The Table 8 pooled strata overlap, so this calculation does not claim a strict pooled Table 8 CRLB. Individual Table 8 comparisons and actual Czech age/dose comparisons are included as model-reference calculations. The exact independent synthetic pooled comparison is available.

This is a regular, local, model-conditional lower bound on the variance of locally unbiased estimators, not a universally attainable confidence limit. For nuisance gap excess equal to zero, use the local unconstrained derivative extension (a conservative benchmark relative to exploiting the nonnegative constraint). Frailty theta is bounded below by zero. We therefore also compute an expected-data likelihood-ratio profile with nonnegative theta and gap exposures. This is a useful resolution diagnostic, not a guaranteed 95% finite-sample coverage interval. Bootstrap or simulation validation remains needed. The near-zero boundary reference for Table 8 is theta=0.1, not a CRLB at theta=0.

Real-data references: the actual cohort size and configured calendar windows, a baseline rate calibrated from observed quiet mortality at reference theta=0.5, and nonnegative gap exposures reconstructed from observed gap mortality. If a gap mortality falls below its model baseline, its reference excess is zero. Across the hypothetical theta grid, keep that calibrated k fixed as a design reference and reconstruct gap excesses for each theta. Table 8 fitted-reference comparisons separately use fitted k and theta (or 0.1 near the zero boundary). The theoretical reference ratios are not actual empirical KCOR estimates. They depend on the stipulated model, alpha and correct gap/quiet classification, not a claim that the observed mortality pattern follows the model exactly.

## Simple limiting formula

For one contiguous quiet window, approximately constant baseline weekly hazard k, low cumulative mortality and small theta*k*t, q_i approximately k-theta*k^2*(i+1/2). Its log(k) derivative is approximately k and its theta derivative is approximately -k^2*(i+1/2). The expected binomial weight is approximately N/k. Projecting the theta derivative off the unknown baseline-rate derivative gives

    I_eff(theta) approximately N*k^3 * sum_i (t_i - mean(t))^2
    = N*k^3 * L*(L^2-1)/12

and

    SE(theta) approximately sqrt(12/(N*k^3*L*(L^2-1))).

Thus in this limit SE scales as N^(-1/2), k^(-3/2), and L^(-3/2). Halving k requires roughly eight times the population to preserve precision; doubling the contiguous quiet duration can be worth roughly eight times the population. This approximation is not used for the multi-window numerical tables, whose separation, growth, mortality and gap uncertainty all matter. For a fixed model and window schedule information is proportional to N exactly. If sigma is the local SE at N, the cohort size necessary for Wald relative 95% half-width r is

    N_req = N * (1.96*sigma/(r*theta))^2.

To halve the relative half-width requires four times the population. This is a local resolution criterion, not a guarantee.

## Known-truth synthetic stress case

50,000 people in the full population, about 12,500 per age/dose cell, 4+18 quiet weeks, an 8-week nonlinear wave, true alpha=1.18. k=0.002 (1940s) and 0.001 (1950s). Each age has theta=4 and theta=0.5.

| Age | True theta | Fisher SE | Local 95% half-width | Median experimental bootstrap limits |
|---|---:|---:|---:|---|
| 1940s | 4 | 1.040 | 2.039 | [1.930,6.262] |
| 1940s | 0.5 | 0.812 | 1.592 | [0,1.956] |
| 1950s | 4 | 2.469 | 4.838 | [0,8.499] |
| 1950s | 0.5 | 2.218 | 4.348 | [0,4.331] |

For theta=0.5 in the 1940s cell, roughly 507,000 people are required for +/-50% local precision and 2.03 million for +/-25%. In the 1950s cell, 3.78 million and 15.12 million are required respectively under the same schedule. The expected likelihood profile, using the nominal chi-square(1) reference cutoff and nonnegative gap exposures, retains zero and reaches about 1.60 (1940s) or 3.44 (1950s). This confirms that low theta is intrinsically poorly resolved.

The pooled ratio is nevertheless much more precise:

| Pooled KCOR metric | Value |
|---|---:|
| True ratio | 1 |
| Model information log-SE | 0.06701 |
| Local 95% interval at true ratio | [0.8769,1.1404] |
| Local interval width | 0.2634 |
| Experimental log-point SD (50 datasets) | 0.07475 |
| Median experimental bootstrap log-SE | 0.07927 |
| Median experimental 95% width | 0.31459 |
| Empirical coverage | 48/50 (96%) |

The experimental estimator is close to this model-based benchmark (about 11.6% greater log-point SD and 18.3% greater median bootstrap log-SE), even though frailty variance itself is very uncertain. It is not an unbiased efficient estimator: the previous stress run showed a +3.68% mean endpoint bias. Low theta point SD can also be smaller than the regular Fisher SE because of the nonnegative boundary and bias. The benchmark is for the exact survival model and target, whereas the experimental fitter retains KCOR's approximate in-wave fitting/correction. It does not resolve all implementation or model discrepancies.

## Actual Czech design examples at reference theta=0.5

All quiet windows and gap mortality included, alpha fixed to 1.18. Population at modeled origin (after skip). Half-width is a local Wald resolution metric; negative lower endpoints are not meaningful intervals for theta.

| Enrollment | Age | Dose | N | Local 95% half-width | N for +/-50% | N for +/-25% |
|---|---|---:|---:|---:|---:|---:|
| 2021_24 | 1940s | 0 | 151,775 | 0.364 | 321,227 | 1,284,907 |
| 2021_24 | 1940s | 2 | 504,907 | 0.582 | 2,737,575 | 10,950,300 |
| 2021_24 | 1950s | 0 | 280,822 | 1.023 | 4,701,595 | 18,806,380 |
| 2021_24 | 1960s | 0 | 423,254 | 4.259 | 122,852,500 | 491,410,000 |
| 2022_06 | 1940s | 0 | 92,518 | 1.232 | 2,247,538 | 8,990,151 |
| 2022_06 | 1940s | 3 | 526,882 | 1.231 | 12,784,390 | 51,137,560 |
| 2022_06 | 1950s | 0 | 168,920 | 3.515 | 33,400,360 | 133,601,400 |
| 2022_06 | 1960s | 0 | 265,167 | 13.297 | 750,165,600 | 3,000,662,000 |

Full table includes all 21 actual Czech cohorts, 18 Table 8 cells, six theta references 0.1/0.25/0.5/1/2/4, each under alpha=1 and alpha=1.18. These do not imply that true Czech frailty variances equal 0.5.

## Table 8 fitted reference

The earlier one-start all-window experiment's observed bootstrap intervals can be compared to the theoretical local precision at the fitted reference. Example 2022 unvaccinated pseudo-dose rows:

| Pseudo-dose | Fitted theta [bootstrap 95%] | Model-reference 95% half-width |
|---|---|---:|
| 0 | 2.365 [2.042,2.728] | 0.434 |
| 1 | 6.494 [5.581,7.337] | 1.115 |
| 2 | 14.578 [11.931,16.917] | 3.180 |

The agreement in order of magnitude is encouraging, not proof that the real model is correct or the regular bound applies literally to the biased estimator. The two-dose 2021 fits sit at theta=0; the near-zero interior reference information is finite and a collapsed bootstrap interval at the constraint must not be read as exact knowledge of zero.

## Planning examples

Same synthetic short-first-window configuration: four initial quiet weeks; 29 nonquiet weeks including an eight-week alpha=1.18 wave; then extend the second quiet window. First number of total quiet weeks needed for a local 95% half-width of +/-50% or +/-25% when theta=0.5:

| Weekly k | Cohort size | +/-50% | +/-25% |
|---:|---:|---:|---:|
| 0.0005 | 100,000 | >212 | >212 |
| 0.0005 | 500,000 | 195 | >212 |
| 0.0005 | 1,000,000 | 156 | >212 |
| 0.0010 | 100,000 | 178 | >212 |
| 0.0010 | 500,000 | 98 | 165 |
| 0.0010 | 1,000,000 | 72 | 129 |
| 0.0020 | 100,000 | 87 | 155 |
| 0.0020 | 500,000 | 23 | 78 |
| 0.0020 | 1,000,000 | 9 | 50 |

All numbers conditional on the design. Add 29 nonquiet weeks to total quiet weeks for modeled calendar duration. >212 means not attained within the scanned horizon, not necessarily impossible. These results are not a proposed alteration to the actual quiet windows.

## Results and reproduction

- information-limits.csv: per-cohort Fisher limits and sample size targets for all references.
- ratio-limits.csv: model-reference independent-cohort pair limits; synthetic ratios are exactly 1.
- expected-profiles.csv: expected likelihood-ratio profile for low frailty in the synthetic stress design.
- table8-fitted-reference.csv: matched actual fitted parameter and bootstrap intervals, local model benchmarks.
- synthetic-pooled-comparison.csv and synthetic-frailty-comparison.csv: comparison with experimental fitter.
- planning-grid.csv and planning-thresholds.csv: full sample-size, mortality, theta, wave and quiet duration scan.
- limits.py, compare.py, planning.py, verify.py: run in that order with Python 3.11. Verification checks independent adaptive exact survival integration for alpha=1.18 and theta up to 15, exact alpha=1 survival, inverse-root sample size scaling, and the weak-hazard approximation. The synthetic survival also matches the independently generated probabilities.
