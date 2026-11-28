**I would still recommend major revision, but two of my earlier objections are resolved in `m94`.** The first quiet window now initializes a joint fit of baseline rate \(k\) and frailty variance \(\theta\), including exposure accumulated before that window. The rate is free to move in the final fit. For a specified nonlinear wave, KCOR now uses the survivor distribution both during the wave and when fitting later quiet weeks. Those are substantive mathematical repairs. In the labeled proportional-shock simulation, the entry correction moved mean KCOR from 1.204 to 0.998 against a target of one. [Current estimator, PDF p. 11][fit] · [Weekly survivor equation][wave] · [Validation, PDF p. 65][s8]

What remains is a question of **what the mortality data determine, how the estimator behaves when its assumptions are imperfect, and which conclusions its validation supports**.

### 1. Exact reconstruction of survival still leaves wave attribution uncertain

During a wave, the observed weekly mortality supplies one equation for two unknown increments: ordinary baseline exposure \(b\) and wave exposure \(w\). KCOR resolves this by assigning the Gompertz increment to baseline whenever deaths exceed its prediction. When deaths fall short, it assigns zero to the wave and reduces baseline. That convention reproduces observed survival, but the reported endpoint \(B+W\) depends on the allocation. Under the paper’s own survival law, moving along a curve of identical survival gives
\[
\frac{dB}{dW}=-\frac{E[Z^\alpha\mid\mathrm{alive}]}{E[Z\mid\mathrm{alive}]}.
\]
For \(\alpha\ne1\), this generally differs from \(-1\). Consequently, different allocations can reproduce *exactly the same deaths* and give different KCOR values. Weekly sampling noise also has an asymmetric effect: an upward fluctuation can add wave exposure, while a downward fluctuation reduces baseline and leaves accumulated wave exposure in place. The manuscript appropriately calls this an attribution convention. Its importance to the endpoint deserves a quantitative result. [Wave allocation and deficit rule][wave]

The proportional fit has a related restriction: incremental gap offsets are constrained to be nonnegative. A temporary mortality deficit before quiet re-entry is scientifically possible. Under the proportional model there is a direct alternative. At each quiet-week start, use the observed survival state to predict the increment as
\[
q_s=\theta^{-1}\log\!\left(1+\theta\,\Delta G_s\,e^{-\theta H_{\rm obs}(s)}\right).
\]
This follows from KCOR’s gamma identity and allows either sign of prior departure from the Gompertz path. **I would compare this fit with the constrained fit, resampling the whole history in each draw.** For nonlinear waves, I would report the range of KCOR under plausible baseline paths, the number of deficit weeks, and the amount of mortality reassigned to baseline. A small survival-equation residual measures numerical accuracy; it cannot settle the allocation. [Gamma identity][gamma] · [Gap constraint and objective][fit]

### 2. The new simulations show a meaningful tradeoff that the paper should report

The current edition **does** contain coverage evidence for the revised nonlinear estimator. Its main simulation gave mean KCOR 1.00093 against truth 1, with 90 of 100 nominal 95% intervals covering. The reported Monte Carlo interval for coverage is 82.4%–95.1%. In the stress design, 48 of 50 intervals covered, but mean KCOR was 1.04019 against truth 1. The supplied results also record clamped gaps in 47 of those 50 original fits and 241 resamples with insufficient signal. Those results leave substantial uncertainty about calibration and show why reporting coverage alone can conceal point-estimate error. [Supplement §S8][s8] · [Stress-run counts][stress]

The most informative comparison is in the supplied diff. In the 200-dataset scenario with all nonlinear pulses inside the marked window, the revised result moved the mean closer to one, from 1.0222 to 1.0102. Yet its absolute error **increased in 161 of 200 paired datasets**, and its central 95% range widened from 0.966–1.083 to 0.804–1.336. In the short-first-fit scenario the result went the other way: absolute error improved in 197 of 200 datasets. A reader deciding whether to use KCOR on one registry needs both findings. [All-pulses comparison][mixed] · [Short-first-fit comparison][short]

**Fix.** Put these paired results in the main validation table, with the generating target, bias, absolute error, interval width, coverage, and frequency of clamping or fallback. Make the fallback rule auditable. In the supplied code, fewer than three eligible nonlinear fitting rows return the initial proportional fit and its diagnostics without a distinct nonlinear-failure status. A flagged nonlinear fit can also return \(\theta=0\). Give each outcome its own status and specify how it enters reported intervals and pass counts. This is a code-path concern; the supplied evidence does not show that it caused a particular Table 8 result. [Early return][fallback] · [Diagnostic flag and returned variance][flag]

### 3. The Czech application needs the current module-off result

The revision correctly reports that the Czech data did not identify \(\alpha\). The historical analysis lacked an external intervention-effect scale. The updated objective reached the lower boundary on both tested grids, and the positive-excess filter retained only 62 cohort-week rows across eight cohorts. Its input was a hazard difference from dose zero, which can include baseline differences as well as wave mortality. There is also limited independent reassurance in agreement between the two objectives: for the same residuals and equal weights, the sum of squared pairwise differences is proportional to their variance. [Czech replay and retained observations, PDF p. 29][alpha] · [Objective definitions][objectives]

Table 8 now labels \(\alpha=1.18\) as a sensitivity analysis, reports substantially narrower 2022 intervals, and explicitly acknowledges that the age-shift comparison has no known exact target. Those are improvements. But its two columns are historical v4.3 and the **current sensitivity run**. A reader still cannot see what the current estimator reports under the paper’s stated default when \(\alpha\) is unidentified. Likewise, the updated window scan’s 78/78 passes and its much smaller dose-1 variance ranges apply to \(\alpha=1.18\). [Table 8, PDF p. 44][t8] · [Default rule][workflow] · [Updated window scan][scan]

**Fix.** Put the current module-off and 1.18 results side by side using the same data, windows, anchor, weights and bootstrap draws. Show the paired change in the endpoint. Report observed mortality using common age weights as well: the raw dose-0/dose-2 hazard ratio is 0.316 in the pooled cohort and above 1 in every displayed age band. That reversal makes the choice of population consequential before frailty modeling even begins. [Raw pooled and age-specific results][age]

### 4. Anchoring can determine the direction of the result

The revised contamination experiment gives a particularly clear demonstration. With the **same fitted curves** at the largest dynamic contamination, mean anchored KCOR was **0.782** when the reference included the first contaminated week and **2.116** when it ended one week earlier. The revision appropriately withdraws the earlier interpretation of a general contamination threshold. This result also matters outside the stress test. Anchoring divides by an estimated early cohort contrast. It can erase a persistent proportional difference or turn an early mortality disturbance into apparent later divergence. [Contamination design and anchor comparison][anchorcheck] · [Definition of anchored KCOR][anchor]

**Fix.** Report the raw cumulative risks, unanchored KCOR, the numerical anchor ratio, and anchored KCOR at the same horizon. Identify the population surviving the stabilization skip. In Table 9, retain the revised fitted benefit and harm results, 0.834 and 1.162, but add the matched raw results, 0.822 and 1.113. Both raw comparisons already detect the injected direction. To test recovery of *magnitude*, calculate the target week-80 cumulative ratio from the generator: the injected multiplier acts on weekly marginal hazard only in weeks 20–79, and the fit uses a quiet window beginning in week 81. [Table 9 and effect window][t9] · [Retrospective timing][timing] · [Matched raw benefit][rawbenefit] · [Matched raw harm][rawharm]

### 5. Several scientific claims still describe a different or more strongly identified method

- **Scope the identification proposition to the survivor history.** During quiet Gompertz follow-up after a nonlinear wave, differentiation gives \(h'=\gamma h-\mathrm{CV}^2(Z\mid\mathrm{alive})h^2\). Under proportional gamma depletion the survivor squared coefficient of variation equals the starting \(\theta\). A prior nonlinear wave generally changes it. The revised fitter accounts for this through the full survival law; the local derivative formula alone cannot recover enrollment \(\theta\) in that setting. Also publish the actual slope and endpoint sensitivity. The defaults table still gives only “Fixed implementation value.” [Proposition][prop] · [Nonlinear fitting rule][nonlinear] · [Defaults][defaults]
- **Correct the algorithm and conclusion.** The workflow caption still tells a replicator to hold \(\hat k\) fixed, in direct conflict with the revised objective. The conclusion still describes a preferred pooled Czech \(\alpha\) region, although the revised results say the historical preference disappeared. Both discrepancies are present in the combined PDF. [Workflow, PDF p. 17][caption] · [Conclusion, PDF p. 36][conclusion] · [Current identification result][alphasummary]
- **Evaluate comparators against their own targets.** The Cox demonstration gives KCOR the generating frailty variances. It is a useful inversion check. In that simulation, different frailty distributions also generate genuinely different marginal survival curves even though individual hazards agree at fixed frailty. An RMST difference can therefore be the correct population answer. Calling that “depletion bias,” or treating a Cox estimate’s distance from one as evidence of KCOR’s superior accuracy, conflates different estimands. Report each method’s error against its own generating target. [Simulation and known variances][cox] · [Comparator table][comparators]

My recommendation is to retain the revised estimator and present it as a **model-dependent description of cumulative mortality at reference frailty**. The strongest evidence now shows when its mathematical corrections work and when they exchange bias for variability. The outstanding scientific claim is whether the adjustment reliably measures depletion in a particular registry. That requires showing how much the endpoint changes under alternative baseline paths, wave allocations, anchors, and diagnostic rules consistent with the observed deaths.

I reviewed the `m94` combined PDF, its sources under `documentation/preprint`, and the explicitly supplied diff, all read-only. The simulation numbers above are reported results checked against those inputs. I did not independently rerun the registry or bootstrap analyses.

[fit]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:365>
[wave]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:517>
[s8]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/supplement.md:459>
[gamma]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:297>
[stress]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/turns/evaluated-to-current.diff:1297>
[mixed]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/turns/evaluated-to-current.diff:803>
[short]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/turns/evaluated-to-current.diff:941>
[fallback]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/turns/evaluated-to-current.diff:1775>
[flag]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/turns/evaluated-to-current.diff:1805>
[alpha]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:830>
[objectives]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/supplement.md:102>
[t8]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:1160>
[workflow]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:537>
[scan]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/supplement.md:306>
[age]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/supplement.md:423>
[anchorcheck]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:780>
[anchor]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:164>
[t9]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:1171>
[timing]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:758>
[rawbenefit]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/turns/evaluated-to-current.diff:316>
[rawharm]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/turns/evaluated-to-current.diff:378>
[prop]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:354>
[nonlinear]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:383>
[defaults]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/supplement.md:174>
[caption]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:606>
[conclusion]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:1012>
[alphasummary]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:850>
[cox]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:645>
[comparators]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:1189>