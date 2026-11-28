# 1.2 · Frailty at the opening of the first modeled wave

## What changed

The wave correction uses the fitted gamma frailty distribution at enrollment and conditions it on baseline-only survival up to the opening of the first modeled wave week. The opening baseline exposure $B$ comes from the inverted cumulative hazard through the preceding week and is zero at the start of modeled accumulation. For enrollment variance $\theta>0$, survival weighting by $\exp(-BZ)$ leaves the gamma shape $s=1/\theta$ intact and gives the survivors scale $c=\theta/(1+\theta B)$. Their mean is $m=sc$. Their $\alpha$ moment on the observed-hazard scale is $F=c^\alpha\Gamma(s+\alpha)/\Gamma(s)$. These are model-based quantities.

For a small wave increment at entry, observed hazard increases by $F\,dW$. Ordinary gamma inversion has derivative $\exp(\theta H_{\rm obs})=1+\theta B=1/m$ there. Consequently, the divisor for positive excess already on the inverted scale is $F/m$. This same conversion gives 1 at $\alpha=1$, consistent with the ordinary inversion identity. Without frailty the mean and moment are both 1.

The factor is evaluated before any deaths from the first modeled wave week enter the calculation. When the modeled observation begins during a calendar wave, the initial distribution describes the cohort at the modeled origin. Any earlier nonlinear selection is outside this calculation. Later wave weeks still use the empirical mean-one proxy and the existing cumulative excess rescaling. This change addresses the opening survivor moment only.

## Check with known exposure

In the review's example the enrollment frailty is unit exponential. After baseline exposure 1, survivors are exponential with rate 2. For $\alpha=2$ their mean is 0.5, their second moment is 0.5, and the opening divisor on the inverted scale is 1.0. The former mean-one proxy evaluated at the opening scale gives 1.5. In the historical code the proxy was evaluated at the end of the first week, so its actual value also depended on that week's deaths.

The check integrates the generating survival law independently of KCOR. The full first week adds baseline exposure 0.02 and wave exposure 0.05. Shorter durations reduce both proportionally. These are deterministic population endpoints.

| Fraction of a week | True wave exposure | Previous estimate | Revised estimate |
|---:|---:|---:|---:|
| 1 | 0.050000000 | 0.031900856 | 0.047331788 |
| 0.1 | 0.005000000 | 0.003317431 | 0.004970385 |
| 0.01 | 0.000500000 | 0.000333172 | 0.000499700 |
| 0.001 | 0.000050000 | 0.000033332 | 0.000049997 |

Applying an opening factor to a finite week approximates the changing risk set within the week. The single-week discrepancy shrinks as the interval shortens. Other checks cover zero and near-zero frailty, noninteger exponents, the first week after stabilization, a later wave week, the proportional identity and a week with zero excess.

## Effect on Table 8

The replay used the archived Czech workbook, the December 26, 2022 reporting date, a fixed wave exponent of 1.18, and the same 999 person-bootstrap draws per enrollment as the preceding edition. The first-week corrected cumulative hazard changed in four of nine fitted subgroups in the 2021 original sample and five of nine in 2022 (absolute difference above 1e-12). The first-week factor enters directly only in that week. The existing algorithm recalculates cumulative excess at subsequent wave weeks. In these data the entry change did not propagate to the six pooled reporting-date endpoints.

| Enrollment | Pseudo-dose | Joint-fit KCOR before and after [95% interval] | Interval width before and after |
|---|---|---:|---:|
| 2021_24 | 1 vs 0 | 1.008322 [0.956378, 1.058436] | 0.102058 |
| 2021_24 | 2 vs 0 | 1.031261 [0.973904, 1.094452] | 0.120548 |
| 2021_24 | 2 vs 1 | 1.022750 [0.963995, 1.088868] | 0.124873 |
| 2022_06 | 1 vs 0 | 1.030193 [0.321927, 1.235764] | 0.913837 |
| 2022_06 | 2 vs 0 | 1.072358 [0.341336, 1.192704] | 0.851367 |
| 2022_06 | 2 vs 1 | 1.040929 [0.847118, 1.213319] | 0.366200 |

The point and every matched bootstrap endpoint were identical at stored precision for both the joint fit and the fixed-rate sensitivity fit. The paired change and its 95% percentile interval were 0 [0, 0] for each comparison. Interval widths use unrounded percentile limits. A separate full-workbook run reproduced all six joint-fit original-sample estimates within relative tolerance 1e-10.

## Reproduction

From the repository root, run

    py -3.11 -B "KCOR critique/objections/1.2/entry-moment/check.py"
    py -3.11 -B "KCOR critique/objections/2.2/wave-reference/replay.py" --entry-moment
    py -3.11 -B "KCOR critique/objections/2.2/wave-reference/check.py" --workbook --edition entry-moment

[Review example](<../../../iterative review/evaluations/3/checks.md>) · [Independent survival check](check.py) · [Before this change](../../2.2/wave-reference/code/KCOR.py) · [Revised estimates and intervals](table8_bootstrap.json) · [Estimates by draw](table8-draws.npz)
