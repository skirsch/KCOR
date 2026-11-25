# KCOR revision — lead-author briefing

## Problems that motivated the investigation

- **Table 8:** two original 2022 negative-control results excluded 1: 1.0756 [1.055, 1.097] and 1.0911 [1.070, 1.112]. Recalculated intervals included 1 but some were extremely wide. Low frailty proved difficult to characterize.
- **Contamination experiment:** without extra mortality, mean KCOR was 1.050 [1.034, 1.066], although the true value was 1.

Two common causes emerged: numerical instability of the fitter and limited statistical information about frailty.

## Main propositions

### 1. All-window joint fitter

- **Before:** estimate baseline rate k from early weeks, then fit frailty variance theta with k fixed. The intermediate joint fitter estimated both in the first window but still fixed k thereafter.
- **Proposed:** fit one common k and theta across all quiet windows, recalculating gap exposure at each candidate pair.
- **Fixes:** propagation of a poor first-window estimate. The 2022 Table 8 interval widths fell by 73–90%. In a stress test with true KCOR = 1, the extreme estimate changed from 812.675 [0.848, 744.878] to 1.016 [0.875, 1.131].

### 2. Theoretical precision limits

- **Before:** empirical uncertainty estimates without a theoretical benchmark.
- **Proposed:** use Fisher information to relate attainable precision in theta and KCOR to population size, mortality, frailty and quiet-window timing, including survival through gaps.
- **Addresses:** insufficient information versus fitter failure. Low theta produces weak curvature and can be biased upward by the nonnegative constraint. Importantly, KCOR can remain precise even when theta is uncertain. The stress-test theoretical KCOR interval width was 0.263 versus the observed median bootstrap width of 0.315.

### 3. Exact nonlinear wave inversion

- **Before:** use an approximate gamma survivor distribution to adjust cumulative excess.
- **Proposed:** given the fitted baseline rate, solve the survival equation for added wave exposure each week, carrying both cumulative exposures forward.
- **Fixes:** nonlinear selection changes the survivor distribution away from gamma. Exact inversion recovers the correct result with correct inputs. Below-baseline weeks require approval of the trial's attribution rule (zero additional wave exposure, smaller baseline increment).

### 4. Consistent frailty fitting after waves

- **Before:** continue using the gamma fitting model after nonlinear selection.
- **Proposed:** fit quiet mortality with the same survivor law as the wave inversion.
- **Fixes:** inconsistent fitted frailty. Known-truth recovery becomes exact; sampled mean KCOR moves from 0.98982 [0.98493, 0.99471] to 1.00093 [0.99596, 1.00591]. Coverage is unchanged.

### 5. Pre-entry correction

- **Before:** selection after cohort definition but before the first quiet window could distort the fitted baseline.
- **Proposed:** reconstruct preceding proportional exposure from observed survival to that window, as for later gaps.
- **Fixes:** baseline-rate distortion after an early shock. Sampled mean KCOR moves from 1.20386 [1.19425, 1.21347] to 0.99774 [0.99321, 1.00228]. Unmarked nonlinear exposure remains unidentified.

### 6. Higher-information contamination experiment

- **Before:** the small-cohort experiment had a biased null.
- **Proposed:** retain it and add larger cohorts.
- **Addresses:** contamination effects versus finite-sample bias. With the new fitter, mean null KCOR was 1.0494 [1.0305, 1.0684] at 50,000 people per cohort and 1.0013 [0.9993, 1.0033] at five million.

### 7. Alpha and reporting

The original pooled alpha near 1.18–1.19 remains unidentified with the new machinery. Keep 1.18 as sensitivity-only. Finalize Table 8 with the approved fitter, bootstrap intervals and fit diagnostics.

## Other mistakes already corrected

| Before | Correction |
|---|---|
| Instantaneous rather than weekly integrated hazards | Fit and invert exact weekly hazards. |
| Incorrect gap and wave boundaries | Account for the last gap week and first wave week. |
| Nonpositive gap reset fitted frailty | Retain the fit; flag material deficits. |
| Extra adjustment at alpha = 1; incorrect survivor state | Restore proportional identity and correct wave-entry/exit calculations. |
| Weekly-block resampling | Resample people; retain death timing. |
| Scan excluded failures | Use the corrected fitter and count all attempts. |
| Validation descriptions and processing mismatched | Verify fitted controls, correct provenance, pooling and MC processing. |
| Contamination threshold ignored biased null | Compare with measured zero-addition result. |
| Overbroad paper claims | Clarify assumptions, Cox comparison, scaling, example selection and uncertainty. |

**Decision:** approve the proposed fitter, scientific additions and wave/entry corrections separately. KCOR's original objective remains unchanged.

[Full presentation](PR-PRESENTATION.md) · [Detailed evidence](PR-SCIENTIFIC-EVIDENCE.md)

*Simulation-average intervals represent Monte Carlo uncertainty; stress-case intervals are person-bootstrap intervals. Table 8 is an empirical control, not a known-truth null.*
