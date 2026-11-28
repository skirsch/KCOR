# KCOR: Methodological Review

**Manuscript:** Steven T. Kirsch, *KCOR: A depletion-neutralized framework for retrospective cohort comparison under latent frailty*.

**Version assessed:** The supplied 78-page `KCOR.pdf`. All manuscript page references below refer to that file, rather than to a subsequently revised online version.

**Review date:** November 8, 2026.

**Scope:** Scientific interpretation, mathematical coherence, and statistical validation. This review assesses the manuscript and checks its relevant mathematical relationships. It does not include an audit of the implementation or a rerun of the simulation suite. It concentrates on issues that can change the meaning or reliability of the results.

**Overall assessment: major revisions are warranted.** KCOR addresses a real problem: people with greater underlying susceptibility tend to experience an event earlier, changing the composition of the surviving population. Its gamma-frailty inversion is mathematically correct under the stated model. However, a correct inversion does not establish that the fitted curvature represents the mechanism the procedure intends to remove. The manuscript has not yet demonstrated that its accepted analyses reliably distinguish useful normalization from removal or creation of scientifically meaningful contrasts.

The most consequential concerns are the interpretation of fitted frailty, the mathematical basis of the optional epidemic-wave correction, and the reliability of uncertainty estimates and diagnostic acceptance. There are also circular elements in the validation arguments. These concerns do not imply that every KCOR result is wrong, or that the method cannot be useful as an explicitly conditional sensitivity analysis.

**1. The central limitation is the interpretation of the correction, not the existence of a fitted parameter.** The manuscript assumes an individual hazard of the form

$$
h_i(t)=z_i h_0(t),\qquad E[z_i]=1,\qquad \operatorname{Var}(z_i)=\theta,
$$

where frailty $z_i$ represents persistent differences in individual susceptibility. With gamma-distributed frailty, the observed population cumulative hazard satisfies

$$
H_{\mathrm{obs}}(t)=\frac{\log\{1+\theta H_0(t)\}}{\theta},
$$

and the inverse is

$$
H_0(t)=\frac{\exp\{\theta H_{\mathrm{obs}}(t)\}-1}{\theta}.
$$

These identities are correct (pp. 9–10). The difficulty is determining what the estimated $\theta$ represents in the application.

Under the specified Gompertz baseline, $h_0(t)=ke^{\gamma t}$, the observed hazard is

$$
h_{\mathrm{obs}}(t)=
\frac{ke^{\gamma t}}
{1+\theta(k/\gamma)(e^{\gamma t}-1)}.
$$

Exactly this observed curve can also arise in a homogeneous population, with no frailty variation, if every individual directly experiences that time-varying hazard. The two mechanisms produce the same survival data. Consequently, residuals, parameter stability, and agreement across estimation windows cannot always distinguish them. The general identification difficulty is established in the frailty literature [1].

This is not an argument that $\theta$ can never be identified within a restricted parametric model. It distinguishes identification *conditional on that model* from evidence that the model describes the source of the observed change in risk.

The manuscript acknowledges this distinction on pp. 12 and 34, but its assurances that violations will be detected or produce conservative behavior are stronger than the evidence supports (pp. 14 and 38). A time-varying intervention effect can resemble depletion closely enough to be removed without generating a conspicuous diagnostic failure.

KCOR remains a well-defined mathematical transformation in that situation. What becomes uncertain is the interpretation that the transformation removed selection rather than a genuine change in risk. Calling the output descriptive does not eliminate this problem if it is then interpreted as evidence that differences have survived an adequate selection adjustment.

The appropriate response is to make the remaining assumptions explicit, test plausible alternative mechanisms, and limit claims of automatic failure detection. No diagnostic based only on these survival curves can resolve every observationally equivalent explanation.

**2. The optional epidemic-wave correction requires a coherent derivation of the complete procedure.** The manuscript introduces a non-proportional-hazard extension: external waves can affect people differently according to their frailty. The exponent $\alpha$ describes this dependence. Several relationships in this extension do not follow from one another as written (pp. 15–17).

Consider the natural individual-level mechanism

$$
h_i(t)=z_i b(t)+z_i^\alpha A(t),
$$

where $b(t)$ is baseline hazard and $A(t)$ is wave intensity. Among survivors, the observed hazard is

$$
h_{\mathrm{obs}}(t)
=b(t)E[z\mid T>t]+A(t)E[z^\alpha\mid T>t].
$$

If one could undo linear survivor selection by dividing by the actual surviving population's mean frailty, the remaining wave multiplier would be

$$
\frac{E[z^\alpha\mid T>t]}{E[z\mid T>t]}.
$$

The manuscript instead defines $F(t;\alpha)=E[z^\alpha\mid T>t]$ and applies that factor to excess cumulative hazard *after* gamma inversion. The hazard-scale motivation therefore does not establish the proposed correction on the transformed scale.

The limiting case $\alpha=1$ exposes the inconsistency. Ordinary gamma inversion already handles a hazard proportional to $z$ throughout follow-up. Yet the written factor is then $F(t;1)=E[z\mid T>t]$, generally different from one. Applying it again changes the already corrected excess, contrary to the statement that the module is inactive at $\alpha=1$. A special code branch would not, by itself, derive the formula or establish coherent behavior near that value.

There is a second problem: the individual-level wave mechanism changes the distribution of frailty among survivors. Writing $B(t)=\int_0^t b(s)\,ds$ and $W(t)=\int_0^t A(s)\,ds$, that distribution is proportional to

$$
f_\Gamma(z)\exp\{-zB(t)-z^\alpha W(t)\}.
$$

For $\alpha\ne1$, this is generally not gamma. Subsequent quiet periods therefore need not satisfy the same gamma inversion, even after the external wave ends. A persistent additive offset cannot generally repair a changed shape of the survivor distribution. The post-wave rule on p. 17 preserves the original inverted path's subsequent increments, so it does not establish a correction for that problem.

This matters beyond the optional module: disabling the module does not restore the gamma assumption for cohorts whose survivors have already been selected by such a wave. The core method's later quiet-window estimation can also be affected.

Finally, dividing cumulative excess by a time-varying factor is not equivalent to adjusting its increments:

$$
\frac{C_{\mathrm{exc}}(t)}{F(t)}
\ne
\int_0^t\frac{dC_{\mathrm{exc}}(s)}{F(s)}
$$

in general. A cumulative operator can be proposed, but its interpretation needs its own derivation and validation.

The necessary evidence is an end-to-end experiment that generates actual event histories under the individual-level wave mechanism, allows the survivor distribution to evolve, estimates the parameters, applies the full correction, and compares the result with a clearly specified target. Recovery of an exponent alone is insufficient.

The manuscript appropriately reports that $\alpha$ is not reliably identified in the Czech application and leaves the correction inactive (pp. 31–33 and 40). The criticism is that a valid general correction has not been established, not that the paper secretly obtained an identified correction in that application.

**3. The uncertainty procedure has an unresolved statistical problem, in addition to incomplete coverage results.** Section 2.9.1, p. 18, describes resampling contiguous blocks of raw death counts and risk sets across event time. It retains the original quiet-window definitions and states that enforcing the survivor-count recursion preserves the necessary covariance structure.

The difficulty is that these time blocks are not interchangeable. Aging, depletion, and waves deliberately make their distributions change over time. Resampling blocks can move high-mortality periods into quiet windows or rearrange the baseline trend. Maintaining

$$
N(t)=N(0)-\sum_{s<t}d(s)
$$

preserves risk-set bookkeeping, but does not establish that the resampling scheme reproduces the sampling distribution of the estimator.

A useful limiting argument is to imagine increasingly large cohorts with the same underlying mortality trajectory. Random fluctuations in observed death rates become negligible. Resampling raw blocks can nevertheless continue to produce substantial differences by relocating high- and low-risk periods. Such variation reflects rearrangement of the time-dependent signal rather than finite-cohort event noise.

As described, the procedure therefore needs a statistical justification or convincing calibration of the exact implementation. A scheme that preserves the time-dependent event probabilities, with appropriate treatment of dependence, would address a different question from moving raw time blocks. This review does not establish that the reported intervals are always too narrow: invalid resampling can also widen or displace intervals.

There is an additional uncertainty omission when the wave extension is used. The workflow permits $\alpha$ to be estimated from the data, but the bootstrap holds it and the external intervention-effect scale fixed (pp. 17–18). Those intervals are conditional on the chosen inputs. They do not propagate estimation uncertainty in $\alpha$, uncertainty in the external effect estimate, or their relationship to the other fitted quantities. This concern does not directly apply to the Czech results for which the wave module is inactive.

Table 12 on p. 55 leaves every empirical coverage entry blank. The manuscript is transparent about this, but transparency is not calibration. A nominal 95% interval has not been shown here to contain its target in approximately 95% of the relevant repeated analyses.

Because KCOR reports results only when diagnostics permit interpretation, validation must report both the probability of acceptance and error rates among accepted analyses. Overall simulation performance is not enough to substantiate the claim that passing analyses are reliable.

**4. Some validation arguments cannot establish what the manuscript asks them to establish.** The clearest example concerns the origin of frailty differences. Page 37 acknowledges that treatment-induced changes in frailty could be absorbed into the normalization, then treats flat normalized curves as evidence against substantial treatment-induced frailty changes.

That reasoning is circular. If the normalization can remove such a change, a flat corrected curve is compatible with the change having occurred. Flatness after adjustment does not establish that the adjustment removed only pre-existing selection.

This distinction is scientifically central. A procedure that removes pre-existing susceptibility differences and one that removes an intervention's effect on susceptibility can produce the same-looking normalized output. Evidence about the origin of frailty must therefore come from information or experiments that can distinguish these possibilities.

The positive controls are useful but do not settle this issue. The described injected effects are separated from the windows used to estimate frailty (pp. 66–67). They establish preservation of certain effects under that separation. They do not establish preservation of effects that overlap estimation, gradually imitate depletion, or change frailty itself.

The empirical controls also require narrower interpretation. Supplement p. 62 describes age-shift controls for which depletion is expected to be reduced and fitted frailty variances small or weakly identified. Success in those settings supplies limited evidence that large corrections in strongly selected cohorts are accurate. For anchored controls, near-flatness also concerns subsequent evolution after initial level differences have been normalized away.

Two Table 8 results are materially non-null:

| Enrollment | Comparison | KCOR | Reported 95% interval |
|---|---|---:|---|
| 2022_06 | 2 versus 0 | 1.0756 | [1.055, 1.097] |
| 2022_06 | 2 versus 1 | 1.0911 | [1.070, 1.112] |

These do not automatically refute KCOR. The estimator could be biased, the proposed control's assumed null could be wrong, or both. They do require explanation before this exercise is cited as reassuring evidence for interpreting contrasts of a similar magnitude.

**5. Validation needs a consistent definition of the quantity being estimated.** The manuscript explicitly defines KCOR as a descriptive, model-dependent contrast rather than a causal effect. That scope is legitimate. The concern is that some comparisons and stress tests use an absence of treatment effect as though it necessarily implied a null descriptive contrast.

For example, pp. 28–29 introduce an additional cohort-specific, non-frailty mortality component while retaining a null target and describing departures above one as sign reversal. If that component represents genuine additional mortality, absence of an intervention effect does not by itself imply equal descriptive hazards after depletion normalization. If the intended target excludes that component, the paper must define that target and explain why the proposed procedure should recover it. Otherwise, preservation of a real difference can be labeled bias, while removal of that difference can be labeled success.

The same distinction matters for the Cox comparison. With a common baseline hazard but different frailty distributions, observed population hazards can genuinely differ even without a treatment effect. A Cox summary and KCOR need not estimate the same quantity. Producing a non-null Cox result in that situation does not demonstrate that KCOR estimates a common target more accurately. The manuscript acknowledges this mismatch in places, but some of its broader comparative claims remain too strong. Figure 5's shared-frailty comparison is also explicitly an approximation rather than a series of actual fitted models (p. 24).

Observed survival curves and restricted mean survival time, the expected event-free time within a specified follow-up period, can correctly describe differences between populations. Those differences do not become estimation errors merely because a frailty transformation removes them. Adjusted survival summaries are also available within existing survival-analysis frameworks [2].

A useful comparison would fit the same Gompertz–gamma model directly, using the same information and an explicitly shared target. This would help establish what KCOR's estimation procedure adds beyond its chosen parametric assumptions.

Anchoring requires equally careful interpretation. If an unanchored KCOR is constantly 1.3, dividing by its reference value makes the anchored curve constantly one. An anchored contrast therefore describes change relative to its reference, not the total contrast already present there. Neither version is directly a ratio of observed death probabilities or a measure of deaths caused by an intervention.

**6. Several mathematical statements require correction, although these are secondary to the preceding issues.** They should be repaired without treating those repairs as a substitute for substantive validation.

| Statement or operation | Required correction | Consequence |
|---|---|---|
| Multiplying observed hazards by a constant leaves the fitted frailty parameter unchanged (p. 11). | Within the stated family, multiplication by $c$ corresponds to $k^*=ck$ and $\theta^*=\theta/c$. | The parameter changes. A common multiplier can still cancel in the final cohort ratio. |
| The weekly recurrence is the stated exact inversion (p. 12). | For a weekly observed cumulative-hazard increment $h_j$, exact inversion gives $H_{j+1}-H_j=(1+\theta H_j)(e^{\theta h_j}-1)/\theta$. | The written update using $h_j(1+\theta H_j)$ is first-order. Its practical error must be quantified. |
| Nonzero observed cumulative-hazard curvature is equivalent to positive frailty variance (p. 11). | With $\gamma>0$, $\theta=0$ still gives a curved Gompertz cumulative hazard. Conversely, $\theta=\gamma/k>0$ gives an exactly constant observed hazard and linear observed cumulative hazard. | Curvature alone is not a valid indicator of frailty or weak identification, even within the model. |
| Correct normalization should yield a linear cumulative hazard in ordinary time (p. 14). | A Gompertz cumulative hazard is linear in $(e^{\gamma t}-1)/\gamma$, not generally in $t$. | The diagnostic must specify the expected model shape or justify a short-window linear approximation. |

The curvature counterexample does not refute parametric identifiability with a fixed baseline specification. It refutes the stated equivalence and cautions against using observed flatness alone as an identification rule. Likewise, a near-zero frailty estimate is not automatically evidence of weak identification: a boundary value can sometimes be precisely estimated. Parameter profiles and uncertainty are needed to distinguish these cases.

**The revisions that would most affect my assessment are substantive and testable.** I would prioritize the following:

1. State precisely what each reported contrast estimates, including anchoring, and define simulation truth when the working model is violated.
2. Replace general assurances of automatic conservative behavior with evidence on plausible competing mechanisms, including effects that overlap estimation windows or alter susceptibility.
3. Derive the wave correction from an explicit event-generating mechanism and validate the entire procedure, including its effect on later survivors and quiet periods.
4. Justify or revise the bootstrap, propagate relevant fitted-input uncertainty, and complete interval calibration.
5. Report diagnostic acceptance rates together with bias, large-error frequencies, and coverage among accepted analyses; explain the non-null empirical controls.
6. Correct the algebraic statements and carry out comparisons with models fitted to the same data and evaluated against comparable targets.

These requirements leave room for KCOR to be useful as a conditional normalization or sensitivity analysis. They are necessary before treating the procedure as a validated way to remove selection-induced distortion, particularly when the scientific conclusion depends heavily on the size of the correction.

**References**

[1] Balan, T. A., and Putter, H. (2020). “A tutorial on frailty models.” *Statistical Methods in Medical Research*, 29(11), 3424–3454. See especially Section 2.5 on identifiability. [Journal article](https://doi.org/10.1177/0962280220921889).

[2] Hernán, M. A. (2010). “The Hazards of Hazard Ratios.” *Epidemiology*, 21(1), 13–15. [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC3653612/).

**Primary source:** Kirsch, Steven T. *KCOR: A depletion-neutralized framework for retrospective cohort comparison under latent frailty*. Supplied 78-page `KCOR.pdf`; all page, section, figure, and table references in this review refer to that supplied version.
