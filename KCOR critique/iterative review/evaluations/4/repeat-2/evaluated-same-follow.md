## Separate addendum to the saved `m94` review

**Two numerical objections are resolved.** I reran the archived no-wave example and obtained KCOR **0.999998** against a true value of 1. I also reran the nonlinear-wave example and recovered **0.200000** of wave exposure against a true value of 0.2. The old Czech fit-failure counts are superseded as well. The objections that still matter most concern the reporting rule for genuinely weak fits, the rationale for choosing quiet windows, and the use of an unidentified wave exponent in the default Czech configuration. [Archived objections][archive] · [Current proportional fitter][gap] · [Current wave solver][wave]

### Fitting and reporting

| Earlier objection | Verdict for `m94` |
|---|---|
| **A zero-excess gap can turn a true ratio of 1 into 0.7402.** | **Resolved for the reported counterexample.** I used the archived 90 weeks, quiet weeks 0–29 and 40–89, \(k=0.002\), \(\gamma=0.085/52\), and frailty variances 4 and 0. The same roundoff-sized offset appeared, **−2.498×10⁻¹⁶**. The fitter recorded the clamp, retained \(\hat\theta=4.000000\), and returned status `ok`. The zero-frailty cohort fitted at \(1.83\times10^{-5}\), giving the small residual deviation from unity. Genuine negative gap departures are still clamped. In particular, successive raw offsets of \(-a\) and zero can add \(+a\) at the second re-entry because the code differences successive *raw* offsets and clips each increment. That remaining issue deserves a separate signed-gap test. [Clamp and joint objective][gap] |
| **The ordinary workbook fixes \(k\), whereas Table 8 requests a joint fit.** | **Resolved for the current estimation path.** The ordinary caller supplies calendar weeks to the public fitter. The proportional branch explicitly requests the entry-aware joint fit, and the nonlinear branch refits both parameters. Table 8’s `joint_fit` wraps the same public function. The old comparison of 1.0677 versus 1.0129 therefore describes an earlier estimator. The workflow caption and Czech configuration comments still tell readers that \(k\) is held fixed and should be corrected. [Ordinary caller][caller] · [Public fitter][dispatch] · [Table 8 adapter][adapter] · [Workflow caption][caption] |
| **Early anchor weeks contribute outside the designated quiet windows.** | **Still true, now partly documented.** The proportional fit uses `quiet OR anchor`. The nonlinear fit uses the same union with marked wave weeks removed. Section 2.5 discloses this, while §2.4.4 still says that only calendar-quiet bins identify frailty. The paper should distinguish the weeks supplying residuals, the weeks supplying survivor history, and the week used to anchor the reported ratio. [Proportional mask][mask] · [Nonlinear mask][nmask] · [§2.5 disclosure][anchorrows] · [§2.4.4 claim][quietclaim] |
| **Four of nine original fits and all 999 draws were flagged in the 2022 Czech analysis.** | **Superseded.** The current saved run records zero original insufficient-signal flags, zero draws with that flag, and zero unsuccessful stratum fits in **both** enrollments. I recalculated all six percentile intervals from the stored draws. For 2022 pseudo-dose 2 versus 0, the current sensitivity interval is **0.9963–1.1211**. Inclusion of one still does not establish equivalence, and the paper now says so. [Saved counts][counts] · [Stored draws][draws] · [Current interpretation][t8text] |
| **A failed identification check can still yield a published normalized ratio.** | **Still applies as a general code-path objection.** For example, too few quiet-window deaths returns `success=True`, applied \(\theta=0\), `insufficient_deaths=True`, and `insufficient_signal=False`. The Table 8 routine counts failures and insufficient-signal flags, then computes and pools the ratios. Its insufficient-signal counter would miss that low-death case. A nonlinear fit with fewer than three eligible rows can also return its proportional initialization and diagnostics. I found no such failure among the original Czech sensitivity fits I recomputed. The remedy is to give valid zero, low information, fallback, and numerical failure distinct reporting outcomes, then validate the interval procedure with those rules in force. [Low-death return][low] · [Nonlinear fallback][fallback] · [Pooling code][pool] · [Manuscript reporting rule][gate] |

### Czech wave adjustment and window selection

**The old in-wave approximation is resolved.** With \(\theta=1\), \(\alpha=2\), and ten weekly wave increments of 0.02, I generated survival independently from the gamma integral and passed it to the current correction routine. It recovered exit exposure **0.1999999999999997**, with zero deficit weeks. The maximum absolute error in the reference-frailty cumulative hazard over the tested path was \(1.4\times10^{-15}\). I withdraw the old 0.1496 objection for this implementation. The remaining uncertainty is scientific: registry deaths alone do not determine the baseline path or which weeks contain wave exposure. [Archived test specification][oldwave] · [Current equation and deficit convention][wavepaper] · [Implemented solver][wave]

**The forced Czech exponent remains operational, although the manuscript now labels it correctly.** The YAML sets `mode: forced_alpha` and sets **both** `forced_alpha` and `identified_alpha` to 1.18. Table 8 explicitly requires correction at 1.18. Switching the mode to `identified_only` without clearing `identified_alpha` would still apply it. The revised empirical analysis itself reports that the historical interior optimum disappeared and that \(\alpha\) is unidentified. I would clear the identified value, make module-off the primary Czech calculation, and retain 1.18 as a separately selected sensitivity run. [Czech configuration][yaml] · [Mode selection][mode] · [Table 8 requirement][require] · [Revised identification result][alpha]

I recomputed the previously missing comparison from the supplied Table 8 workbook. These are **anchored, pooled pseudo-dose 2 versus 0 point estimates at December 26, 2022**, using the same records, skip, reference week and pooling weights. I refitted the cohorts for each KCOR specification.

| Enrollment | Observed, anchored | Current core, module off | Current sensitivity, \(\alpha=1.18\) |
|---|---:|---:|---:|
| 2021 week 24 | 1.02427 | 1.04560 | 1.03309 |
| 2022 week 6 | 1.02873 | 1.01662 | 1.05566 |

The sensitivity column reproduces the published points. For 2022, activating the module raises the endpoint by **0.0390** and moves it from below the observed anchored ratio to above it. That is a concrete reason to show the module-off result in Table 8. The saved comparison labeled `gamma_working_fit` also uses wave inversion and answers a different question. [Input workbook][input] · [Risk sets and pooling][pool] · [Published table][table8] · [Saved fitter comparison][comparison]

**The prespecification objection still applies.** The current Czech configuration says calendar 2023 was chosen because it showed the “least vaccine harm.” The active configuration lists four windows. That comment does not establish how all four were chosen, but it leaves the outcome-independent rationale for the 2023 choice unresolved. The scan’s 78/78 pass count is also a count of **individual fits**. Its pass calculation contains no comparison across windows. The old dose-1 ranges of 7.70 and 14.51 are superseded: the current ranges are approximately **0–1.56** and **0–0** in the two birth decades. I would publish the selection chronology and the range of the *endpoint* across candidate windows and slopes. [Configuration rationale and dates][windows] · [Pass calculation][pass] · [Current scan][scan]

### Identification and validation

- **Slope dependence and baseline differences — still apply.** The Czech slope is checkable in code and configuration: **0.085 per year**, divided by 52 for weekly fitting. The supplement still lists only “Fixed implementation value.” In the proportional quiet model, the paper’s formula implies a local variance change of \(\delta\gamma/h\) for a slope change \(\delta\gamma\). A shared slope error can therefore affect cohorts differently. Mean-one frailty within each cohort leaves persistent baseline-level differences in \(k_d\), and anchoring cancels a constant ratio regardless of its origin. Show observed, unanchored, and anchored endpoints over the same slope grid. [Configured slope][slope] · [Cancellation claim and proposition][prop] · [Frailty convention][meanone] · [Anchoring definition][anchor]
- **Known-frailty Cox comparison and fixed-rate multiwave example — still limited in the way alleged.** The Cox experiment passes the generating variances directly to inversion. The separate multiwave example uses expected counts and the generating baseline rate. They are legitimate component checks. A marginal survival or RMST difference can be real under their conditional treatment null. The new sampled survivor-model experiments supply additional evidence for the fitted estimator and should be reported separately from these checks. [Cox simulation code][cox] · [Multiwave design][multi] · [Current validation][s8]
- **Positive controls — fitted endpoints repaired, magnitude objection remains.** Table 9 now uses the current fitter. The generator still multiplies the marginal hazard among survivors in weeks 20–79 and resumes the original hazard schedule afterward. S7 does update survivors within frailty groups, but its analysis still calls the earlier cumulative-hazard fitter. The missing validation is a known **horizon-specific cumulative target** under individual-level injection, scored with the current procedure. [Table 9][table9] · [Injection code][inject] · [S7 survivor update][s7gen] · [S7 fit][s7fit]
- **The claimed 66% contamination threshold — withdrawn in `m94`.** The historical zero-addition cell did have mean 1.034 with 15/20 endpoints above one. The revised paper now compares additions with zero: at 0.0003 per week the estimated mean increment was **0.010**, with Monte Carlo interval **−0.012 to 0.032**. It also reports that moving the anchor by one week changed the largest dynamic-contamination result from **0.782 to 2.116** on the same fitted curves. The old criticism of a claimed universal threshold is resolved. The dependence on anchor and population size remains consequential. [Historical grid][grid] · [Revised comparisons][contam]
- **Coverage — partly resolved.** The earlier 502/600 result and the paired 160/200 versus 194/200 result belong to earlier procedures. In the paired study, mean interval widths were 0.0478 and 0.1488. `m94` now has direct sampled evidence for the revised nonlinear estimator: **90/100** covered in the main design and **48/50** in the stress design. In the latter, mean KCOR was **1.0402** against truth 1, and 241 resamples had insufficient signal. It would be inaccurate to say there is no current coverage study. Nominal calibration across offset constraints, fallbacks, anchor choices, and the empirical pooling procedure remains unestablished. [Historical paired results][paired] · [Historical coverage qualification][historicalcov] · [Current main and stress studies][s8] · [Stress diagnostics][stress]

The saved open answer stands separately. This addendum’s new calculations were read-only. For the numerical checks I disabled the configuration loader’s ancestor-directory search in memory and used the specified edition’s Czech YAML. I recomputed the two noiseless examples and the Czech point estimates above, and checked the stored bootstrap quantiles without rerunning the 999-draw simulations. [Configuration loader][loader]

[archive]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/turns/2-historical-blind.md:1>
[oldwave]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/turns/2-historical-blind.md:34>
[gap]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/code/KCOR.py:1482>
[wave]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/code/kcor_survivor.py:79>
[caller]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/code/KCOR.py:6989>
[dispatch]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/code/KCOR.py:10188>
[adapter]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/analysis/table8/compare.py:9>
[caption]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:606>
[mask]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/code/KCOR.py:1242>
[nmask]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/code/kcor_survivor.py:189>
[anchorrows]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:381>
[quietclaim]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:338>
[counts]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/analysis/table8/table8_bootstrap.json:20>
[draws]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/analysis/survivor_validation/table8-draws.npz>
[t8text]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:747>
[low]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/code/KCOR.py:1274>
[fallback]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/code/kcor_survivor.py:247>
[pool]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/analysis/table8/table8_bootstrap.py:64>
[gate]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:175>
[wavepaper]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:519>
[yaml]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/data/Czech/Czech.yaml:14>
[mode]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/code/KCOR.py:2150>
[require]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/analysis/table8/table8_bootstrap.py:51>
[alpha]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:830>
[input]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/analysis/table8/input.xlsx>
[table8]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:1160>
[comparison]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/analysis/survivor_validation/table8-comparison.csv:1>
[windows]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/data/Czech/Czech.yaml:22>
[pass]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/analysis/quiet_windows/scan.py:70>
[scan]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/supplement.md:304>
[slope]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/data/Czech/Czech.yaml:34>
[prop]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:334>
[meanone]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:275>
[anchor]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:164>
[cox]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/test/sim_grid/code/generate_cox_bias_sim.py:234>
[multi]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:725>
[s8]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/supplement.md:459>
[table9]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:1171>
[inject]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/test/positive_control/code/generate_positive_control.py:114>
[s7gen]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/test/sim_grid/code/generate_s7_sim.py:159>
[s7fit]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/test/sim_grid/code/generate_s7_sim.py:253>
[grid]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/analysis/contamination/historical-grid.csv:7>
[contam]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:784>
[paired]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/validation/person_bootstrap/paired.json:6>
[historicalcov]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/documentation/preprint/paper.md:589>
[stress]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/analysis/survivor_validation/stress-results.json:34>
[loader]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m94/code/KCOR.py:1804>