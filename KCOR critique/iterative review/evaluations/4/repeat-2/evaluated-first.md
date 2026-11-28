**I would recommend major revision.** KCOR’s gamma-frailty inversion is mathematically correct under its stated model, including the exact update across week boundaries. The paper also correctly shows that a known Gompertz slope can identify frailty variance from an exact quiet-period hazard curve. The main problems arise when estimating that variance from registry data, carrying it across epidemic waves, and interpreting the validation results as evidence for the complete procedure. I would retain KCOR as a model-dependent descriptive analysis. The evidence in this edition does not establish that it reliably removes selection-induced mortality differences. [Gamma inversion and weekly update, §§2.4–2.5][inverse] · [Identification proposition, PDF p. 11][proposition]

### 1. A quiet window can fit perfectly and give the wrong baseline rate

This is the clearest defect in the core estimator. Step 1 fixes the cumulative offset to zero in the first quiet window. Step 4 then holds the resulting baseline rate \(\hat k\) fixed. Suppose a wave occurred **before** that first quiet window and left cumulative excess \(C>0\), exactly the kind of persistent offset the paper allows. Since the Gompertz cumulative hazard \(G(s;k)\) is linear in \(k\), the manuscript’s own weekly prediction satisfies

\[
q(t;k,\theta,C)=q\!\left(t;\frac{k}{1+\theta C},\theta,0\right).
\]

The seed can therefore fit every quiet week exactly while reporting \(k/(1+\theta C)\) as the baseline rate. Good residuals cannot reveal the error. Freezing that rate then affects the pooled frailty refit and the no-wave reference used to calculate excess mortality. This matters for the Czech window scan, where the first candidate window begins in April 2022 for a June 2021 enrollment. [Weekly prediction and zero-offset fit][seed] · [Offset reconstruction and fixed-rate refit][refit] · [Czech window dates][scan]

**Fix.** Under the proportional-frailty core, use survival accumulated up to the **start** of each quiet week. The model gives \(1+\theta H_0(s)=\exp\{\theta H_{\rm obs}(s)\}\), so a weekly prediction can use that starting state and the quiet-week Gompertz increment without assuming a zero pre-window offset. Jointly estimate \(k,\theta\), propagate uncertainty in the observed starting state, and test recovery with a wave deliberately placed before the first quiet window. For waves that act nonlinearly on frailty, the survivor model also needs the change described in finding 3. [Start-of-week inversion][weekly]

### 2. Diagnostic acceptance gives little assurance about the amount of depletion removed

For any observed cumulative hazard \(H\) and any chosen \(\theta>0\), the baseline \(B_\theta=(e^{\theta H}-1)/\theta\) reproduces exactly the same survival under gamma frailty. The restriction that distinguishes these explanations is the assumed baseline shape. The paper’s proposition makes the sensitivity explicit: at a given instantaneous hazard \(h\), changing the assumed slope by \(\delta\gamma\) changes the locally implied variance by \(\delta\gamma/h\). Low weekly mortality can make a small slope error consequential. Checking the fitted shape on the weeks that determined the fit supplies little independent evidence about its cause. [Inversion][inverse] · [Slope and identification claims][proposition]

The manuscript supplies a particularly informative result here. **All 78 Czech window fits passed**, yet for dose 1 the estimated variance ranged from effectively zero to 7.70 in the 1940s birth cohort and to 14.51 in the 1950s cohort. The windows also overlap. That finding establishes sensitivity of the parameter. Whether it materially changes KCOR at the reporting horizon is the next quantity readers need. The supplement gives the Gompertz slope default only as “Fixed implementation value.” [Window scan and pass counts][scan] · [Operational defaults][defaults]

**Fix.** Publish the numerical slope and units, window dates, acceptance thresholds, and joint uncertainty in \(k,\theta\). For each plausible slope and window, report the resulting **unanchored and anchored KCOR at the same horizon**, alongside the raw contrast. Score predictions in held-out quiet windows. Where materially different explanations fit equally well, report the range of contrasts they support.

### 3. The optional wave correction is approximate where it matters most

The revised post-wave conditional-survival equation is sound **given correct exposures at wave exit**. The in-wave procedure supplies those exposures by applying ordinary gamma inversion, subtracting a fitted reference, and dividing accumulated positive excess by the current week’s frailty factor. The paper acknowledges this approximation. An exact limiting example shows its consequence. Take a wave independent of frailty, \(\alpha=0\), with cumulative wave hazard \(W\) and baseline cumulative hazard \(B\). Then
\[
S=e^{-W}(1+\theta B)^{-1/\theta}.
\]
The divisor is \(E[Z^0]=1\), so the proposed in-wave correction leaves the ordinary inversion at \(e^{\theta W}B+(e^{\theta W}-1)/\theta\). The desired cumulative hazard at reference frailty \(Z=1\) is \(B+W\). They differ even with noiseless data and known parameters. A nonlinear wave can also change the distribution of survivors entering **later quiet windows** used to fit \(\theta\). [In-wave rule and post-wave equation, PDF pp. 16–17][wave] · [Order of estimation][workflow]

There is a separate reporting issue. The paper declares the Czech exponent unidentified and says the module is inactive in that circumstance. The revised Table 8 nevertheless fixes the Czech exponent at **1.18**, near the weak empirical optimum. Its intervals condition on that choice. [Inactivity rule][inactive] · [Identification result][alpha] · [Table 8 configuration][table8method]

**Fix.** Make the module-off calculation the primary Czech result and label 1.18 as a sensitivity scenario. For a model-based wave adjustment, use the paper’s own \(L(B,W)=E[e^{-ZB-Z^\alpha W}]\) **during** the wave as well as after it. Given a specified baseline path and exponent, fit nonnegative weekly wave increments to observed survival and carry the resulting survivor distribution forward. Show sensitivity to the assumed baseline and external effect scale.

### 4. The validation results belong to several different estimators

| Reported evidence | What it establishes |
|---|---|
| Cox comparison across the frailty grid | KCOR receives the **generating frailty variances**, including for its interval calculation. This checks the inversion. The cohorts genuinely have different marginal survival under the simulated conditional treatment null, so a marginal hazard or RMST difference is not itself estimation error. [Simulation design][cox] · [Interval construction and approximate comparator curve][coxfig] |
| Multiwave recovery and positive controls | The recovery example uses expected deaths and fixes the baseline rate at its generating value. The positive control multiplies the **marginal hazard among survivors** for part of follow-up. A cumulative ratio at week 80 has its own target. Values of 0.825 and 1.107 establish direction, but magnitude recovery requires calculating that target from the generator. [Recovery example][recovery] · [Injection rule][injection] · [Positive-control results][positive] |
| Nominal 95% intervals | One person-bootstrap study covered the truth in 189/200 datasets (94.5%). Another covered it in 502/600 (83.7%), although all 600 original fits passed diagnostics. The designs differ, and **both predate the current week-boundary fitter**. Coverage for the published estimator remains unverified. [Coverage account, PDF p. 19][coverage] |
| Czech age-shift control | The revised fitter was used here. All six intervals include one, but the 2022 pseudo-dose 1 versus 0 interval spans **0.322–1.268**. That interval allows substantial departures from one. The construction changes age, and the published ratios are anchored, so there is no known unanchored population ratio of one to recover. [Control construction and anchoring][age] · [Table 8, PDF p. 50][table8] |

**Fix.** Freeze one complete estimator and run it on independently sampled cohorts with known horizon-specific targets. Estimate every parameter that would be estimated in practice. Include unequal baseline levels, waves before the seed window, nonlinear waves, quiet-window contamination, and sparse events. Report bias, interval width, coverage, diagnostic rejection rates, and coverage among accepted fits. Keep the known-parameter runs as checks of the algebra.

### 5. The contrast needs a narrower scientific interpretation

Each cohort’s frailty is assigned mean one. If one cohort has twice the underlying mean risk of the other, that difference can be absorbed into its \(k_d\). Even with perfect variance estimation and no intervention effect, its unanchored KCOR can therefore equal two. Anchoring divides out any constant ratio, including a genuine persistent proportional effect. Skipping the first weeks further changes the population to people who survived to the new origin and omits any acute effect during those weeks. These are consequential features of the estimand when describing a “healthy vaccinee” adjustment. [Mean-one frailty model][frailty] · [Primary and anchored estimands][anchor] · [Skip rule][inactive]

The Czech results illustrate why the distinction matters. The raw dose-0 versus dose-2 cumulative-hazard ratio is below one in the all-ages population (0.316) and above one in **every displayed age band** (1.28–2.13). A pooled frailty fit does not put the cohorts on a common age distribution. Likewise, the quiet-window contamination experiment varies particular positive perturbations in one cohort over 20 replicates per cell. Its reported 66%-of-baseline crossing is a result for that configuration. It cannot bound residual confounding in the Czech population. [Age-specific and pooled hazards, PDF p. 74][raw] · [Contamination design and interpretation][contam]

**Fix.** Lead the application with observed survival or mortality risk at a specified horizon, using common age weights where appropriate. Show unanchored KCOR alongside it, then use anchored KCOR specifically to describe change relative to the reference week. Label all three with their time origin and population. That presentation would let readers see which conclusions come from observed outcomes, which depend on the frailty model, and which arise from anchoring.

My review was read-only and limited to the combined PDF and sources under `documentation/preprint`. The counterexamples above follow from the published equations. The simulation and registry figures are the manuscript’s reported results; I did not independently rerun them.

[inverse]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:297>
[proposition]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:354>
[seed]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:371>
[refit]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:409>
[weekly]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:396>
[scan]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/supplement.md:304>
[defaults]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/supplement.md:169>
[wave]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:576>
[workflow]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:613>
[inactive]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:485>
[alpha]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:925>
[table8method]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:820>
[cox]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:702>
[coxfig]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:743>
[recovery]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:798>
[injection]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/supplement.md:132>
[positive]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:1248>
[coverage]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:669>
[age]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:804>
[table8]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:1237>
[frailty]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:275>
[anchor]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:153>
[raw]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/supplement.md:423>
[contam]: </C:/Users/jasmi/AppData/Local/Temp/kcor-s4-approved/m83/documentation/preprint/paper.md:853>