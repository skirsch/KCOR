# All-window joint fitting experiment (2.7 root-cause investigation)

Status: exploratory, not adopted into the paper or the selectable KCOR pipeline. Lead-author approval is needed for a change to the core fitting algorithm.

## Question

The current joint fitter jointly fits the common Gompertz scale k and gamma frailty variance theta in the initial quiet window only. It freezes k when refitting theta over the full set of quiet windows. An unstable initial fit can therefore propagate through the later windows via the gap offsets. The existing paper's baseline model, a common k with age-related Gompertz growth, is retained in every trial. The existing early anchor mask, positive gap-offset rule, wave correction and normalization are also retained.

Three isolated variants use the same weekly integrated hazard least-squares residual:

- Frozen: jointly update log(k) and theta with the reconstructed gap offsets fixed at each iteration, then rebuild the offsets. Pilot only.
- Profile: reconstruct the gap offsets for each candidate pair (k, theta) inside the residual, rather than treating them as constants during the fit. The original first-window seed remains the initial guess.
- Multistart: same profile objective, additionally try theta=0.5 as an initial guess and select the solution with the lower residual sum of squares. This is not imposing theta=0.5 on the result. The second start checks for optimizer trapping.

For the candidate theta, the observed cumulative hazard H up to the beginning of a quiet window gives the equivalent baseline-scale cumulative exposure expm1(theta*H)/theta (limit H at theta=0). Subtract k times the unit Gompertz cumulative function at that boundary, then apply the original incremental nonnegative gap rule. Use those offsets in the exact weekly integrated gamma-frailty hazard. All methods retain the original convention for the first quiet window, including delayed entry; this is not a resolution of objection 3.2.

## Paired design

Table 8: original archived workbook, all six pooled endpoints (2021_24 and 2022_06), 999 paired person-bootstrap draws per enrollment, using the current wave correction and alpha=1.18. Synthetic stress design: 50 independent populations of 50,000, four subgroup cells, known theta=4 and 0.5, common baseline within age, alpha=1.18, first quiet block four weeks, second quiet block 18 weeks, exact true KCOR=1. Each dataset has 99 paired person-bootstrap draws. Existing data, draws, weighting, and endpoint estimator are unchanged. The replay's current arm is checked against the archived stress draws.

## Initial results

The frozen-offset pilot did not resolve the main stress outlier: dataset 34 pooled KCOR 815.504 versus 812.675 under the current fit. Reconstructing the offsets inside the objective gave 1.01611. Its failing cell's theta moved from 91.8409 to 0.4616 (true 0.5). The one-start profile retained some extreme bootstrap draws, giving the dataset 34 interval [0.8628,328.1881]. Trying the second initialization under the same objective gave [0.8748,1.1314] in the matched pilot, indicating an optimizer-branch issue in addition to the freeze of the initial rate.

The one-start profile full comparison and all fitted Table 8 parameter intervals are saved in the CSV and NPZ files here. Current stress intervals covered 50/50 datasets, with median width 213.056 and RMSE 162.635. Profile intervals covered 48/50, median width 0.4043, RMSE 0.09656; some interval widths remained pathological (maximum 1024.2). Table 8 had no insufficient-signal flags or failed calls across the 18,000 subgroup fits in the one-start profile run. All six intervals contained 1; the three 2022 widths decreased by 90.4%, 89.0%, and 73.4%; the three 2021 widths increased by 10.5%, 12.2%, and 9.2%. Narrower intervals alone are not evidence of better calibration.

The exact expected-count nonlinear wave case yielded KCOR 0.99613 under the current fitter versus 0.99589 under the profile trial, so this change does not remove the small nonlinear-wave approximation error in the original method.

Reproduce using Python 3.11: run experiment.py with profile, multistart, or frozen pilot. Run table_parameters.py for profile subgroup parameter draws, then summarize.py for the corresponding tables.

## Full multistart result

The 50 matched stress datasets were completed, and the existing fitter's 5,000 vectors reproduce the archived current joint-fitter results. The multistart trial retained 48/50 coverage (96%; exact binomial 95% interval 86.3–99.5%), versus 50/50 (100%; 92.9–100%) for the current fit.

| Measure | Current joint-seed fitter | All-window profile with two starts |
|---|---:|---:|
| Mean KCOR (95% interval for mean across datasets) | 43.5064 [-1.5602, 88.5730] | 1.03676 [1.01453, 1.05899] |
| RMSE versus true KCOR=1 | 162.635 | 0.08572 |
| Median absolute endpoint error | 0.06352 | 0.05459 |
| Median within-dataset 95% interval width | 213.056 | 0.31459 |
| Mean interval width (95% CI across datasets) | 421.335 [281.891,560.779] | 0.32496 [0.30443,0.34548] |
| Maximum interval width | 1629.002 | 0.51366 |
| Dataset 34 KCOR (95% bootstrap interval) | 812.675 [0.8483,744.8778] | 1.01611 [0.87484,1.13139] |
| Dataset 34 failing cell theta (true 0.5) | 91.8409 [0,91.6394] | 0.46161 [0,2.22698] |
| Failed subgroup fits (originals and draws) | 177 | 0 |
| Upper-bound/insufficient-signal flags (originals and draws) | 2043 | 243 |

All Table 8 numbers match the one-start profile to three decimal places, with no failures or insufficient-signal flags. Maximum difference across all matched Table 8 endpoint draws is 0.000292. For the full Table 8 endpoint comparison, see table-comparison.csv. The subgroup parameter CSV uses the one-start profile, not the multistart trial.

The recovered low variance remains individually imprecise in the tiny synthetic groups. Median fitted theta over the 50 originals for the two true-0.5 cells was 0.367 and 0.136; for the current fitter, 0.797 and 0.276. The corrected endpoint is therefore much more stable without implying that low theta can always be estimated accurately. The residual mean endpoint bias of +3.68% remains, and the empirical coverage is based on only 50 datasets. This does not resolve the no-added-comorbidity experiment or establish general calibration.

Decision for the lead author: investigate/adopt only with approval. This is strong evidence of a fitting/optimization contributor to the low-frailty root cause, not proof that all weak-identification or wave-window issues are solved. No production or paper modification was made. The revised code SHA256 remained 429FF5DA8D385B54F4906EE678EFD21BC73624D03E6D0F11A546A3A851CA9FAE.
