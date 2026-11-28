"""Wording changes made after reviewing the frozen November 21 edition."""
import re

from review_links import reference


# Bound each change by two literal phrases in one source paragraph.
# The same spans drive the Markdown, the clean PDF and the cumulative redline.
PAPER = [
    ("N7 N8", "A distinct and important concern is whether", "experiment to bound this empirically.",
     "This stress test asks whether adding mortality unrelated to frailty within a quiet fitting window distorts the frailty estimate and can raise anchored KCOR above 1 when the cohorts share the same underlying baseline hazard."),
    ("N7", "The true cumulative contrast is null by construction (identical baseline hazard).", "",
     "The two cohorts have identical underlying baseline hazards, so their baseline cumulative hazard ratio is 1."),
    ("N8", "Each cell used 20 stochastic replicates", "anchored KCOR).",
     "The figures and table use 20 replicates per cell and the original fitter. The first four accumulating weeks supply the baseline rate. Following the simulation design, KCOR is normalized at the end of the fifth accumulating week, so the reference ratio includes the first week of added deaths."),
    ("N8", "Contamination drove", "toward apparent benefit, not harm.",
     "Dynamic contamination reduced the fitted frailty variance in cohort A. Mean KCOR fell from 1.063 [0.998, 1.128] at zero added hazard to 0.782 [0.735, 0.829] at the largest dynamic amplitude (0.005). All 20 endpoints at that amplitude were below 1. Brackets give pointwise 95 percent Monte Carlo confidence intervals."),
    ("N7 N8", "KCOR increased monotonically with", "from measured comorbidity imbalance alone.",
     "With 200 datasets per level, mean KCOR was 1.050 [1.034, 1.066] at zero added hazard and 1.060 [1.045, 1.076] at an added weekly hazard of 0.0003. The difference was 0.010 [-0.012, 0.032]. The smallest tested addition whose mean increase over zero had an interval wholly above zero was 0.001 per week, with a difference of 0.179 [0.154, 0.204]. Brackets give pointwise 95 percent Monte Carlo confidence intervals. Mean KCOR already exceeded 1 at zero added hazard."),
    ("N8", "combined sign-reversal frequency.", "",
     "the fraction of combined simulations with KCOR above 1."),
    ("N8", "The joint reversal-frequency heatmap", None,
     "Mean KCOR exceeded 1 in 8 of the 16 combined cells. The heatmaps show the mean and the fraction of replicates above 1 for each combination."),
    ("N7 N8", "Substantial apparent harm on the anchored scale", "independently.",
     "Added deaths enter the frailty fit and cumulative endpoint, and the first contaminated week also enters the normalizing ratio. A sensitivity analysis used the same fitted curves with the workbook reference at the end of week four, immediately before the added deaths. At the largest dynamic amplitude this gave mean KCOR 2.116 [1.992, 2.240]. The direction of the anchored contrast therefore depended on whether its reference included the first contaminated week."),
    ("N7", "These results bound the contamination hypothesis", "for the reported Czech cohorts.",
     "The size of any corresponding quiet-window hazard increment in the Czech cohorts remains unknown."),
    ("N7 N8", "Left: Type 1 (dynamic HVE style)", "across 20 replicates.",
     "The left and center panels show the dynamic and comorbidity sweeps with pointwise 95 percent Monte Carlo confidence intervals for the means. The right panel shows mean KCOR on the combined grid. Each cell uses 20 simulations."),
    ("N8", "grid (joint contamination).", "",
     "grid (joint contamination), using the fifth accumulating week as the reference. The frailty error bars show one standard deviation across simulations."),
    ("N7 N8", "For positive", "Full grid in",
     "Results use the fifth accumulating week as the reference. Brackets give pointwise 95 percent Monte Carlo confidence intervals for the means from 20 simulations per cell. Full grid in"),
    ("3.3", "Positive controls (injected harm/benefit)", "and sampling noise).",
     "Supplementary Section S3 describes the two positive controls. Table 9 reports their fitted week-80 endpoints."),
    ("3.3", "In positive-control simulations with injected multiplicative hazard shifts,", "when an effect is present.",
     "The joint-fit week-80 ratio was below 1 for benefit and above 1 for harm. Both 95 percent intervals excluded 1. The quiet window used for the fit began in week 81, so these comparisons are retrospective."),
    ("3.3", "Positive control results comparing injected hazard multipliers", "estimator detecting injected effects.",
     "Week-80 KCOR after a joint baseline-rate and frailty fit. Brackets give 95 percent percentile intervals from 999 resamples per scenario, with both cohorts refitted in each draw. Resampling used the rounded synthetic death and risk-set counts, treating residual risk-set losses as nondeath exits."),
    ("Recovery", "Under correct specification, the revised estimator should recover", "under a true null effect.",
     "In the multiwave recovery example, weekly deaths equal their model expectations, so the first observed hazard gives the generating baseline rate exactly. With that rate fixed, the example checks recovery of the enrollment-time frailty variance and alignment of the quiet windows. The synthetic negative control also checks whether KCOR remains near 1 under a true null effect."),
    ("Fig1", "After estimating cohort-specific depletion geometry", "the working model.",
     "The earlier monthly normalization adjusted each cohort's hazards using the slope fitted in its quiet period. The resulting cumulative hazards are more nearly linear."),
    ("Grid", "All methods were applied to identical simulation outputs.", "",
     "All methods were applied to identical simulation outputs. KCOR estimates in this grid used the earlier procedure, which fitted cumulative hazard over a single quiet window."),
    ("Pilot", "An end-to-end simulation pipeline now evaluates", "conditionally on diagnostic pass status.",
     "The historical pilot used weekly Poisson resampling and compared its intervals with a KCOR estimate fitted to one reference dataset. Four scenarios reused the same expected death counts across simulated datasets, so their reported coverage cannot establish how often intervals contain the population value across independently sampled cohorts."),
    ("N1b", "Uncertainty bands (", "are shown but are narrow due to large sample sizes.",
     "Error bars show pointwise 95% person-bootstrap intervals for week-52 KCOR, with the generating frailty variances held fixed. All seven intervals include the null value of 1."),
    ("N10b", "delta-iteration fitting across prespecified quiet windows", "",
     "delta-iteration fitting across configured quiet windows"),
    ("N10", "The normalization logic is illustrated empirically in Figure", "from the Czech analysis pipeline.",
     "Figure @fig:kcor_empirical_intuition illustrates normalization in a Czech cohort comparison. The selection rule restricted final KCOR to 0.85–1.15 and favored examples in which normalization reduced the between-cohort difference in curvature."),
    ("N10", "a representative analysis example", "", "an analysis example"),
    ("N9", "Throughout this manuscript, cohorts define the primary comparison groups", "frailty parameters are not estimated separately by stratum.",
     "Each bootstrap replicate refits the baseline rate and frailty variance for the same comparison groups used in the primary analysis. Strata used solely for resampling or standardization retain their membership during resampling and share the fit for their comparison group. For Table 8, each subgroup and pseudo-dose combination defines a separate fitted group. The anchored dose-pair ratios are then pooled across subgroups."),
    ("N8", "When an anchored presentation is used,", "after the anchor time.",
     "When an anchored presentation is used, KCOR at each time $t$ is divided by its value at the prespecified reference week $t_0$, as defined in §2.1.1."),
    ("N5", "empirical robustness of fitted frailty parameters to", "demonstrated in",
     "sensitivity of fitted frailty variance to quiet-window placement is shown in"),
    ("N5", ", supporting stability when this condition holds approximately", "", ""),
    ("N4", "by anchoring", "",
     "using the cumulative hazard just before the first modeled wave week (zero at the start of modeled follow-up) and the fitted baseline increment for that week"),
    ("N2w", "inside the prespecified NPH date window.", "",
     r"inside the prespecified NPH date window. The factor $(e^\gamma-1)/\gamma$ converts the starting rate into a full-week increment and equals 1 in the limit of zero slope."),
    ("N3", "For each prespecified wave end time", "",
     "At the start of the first quiet week after each gap, denoted by"),
    ("N1", " under identical information constraints", "", ""),
    ("N1", "Observed cumulative hazards were estimated from cohort-level event counts,", "exploiting simulator-only knowledge.",
     "Observed cumulative hazards were estimated from the simulated event data. For the KCOR inversion, we used the frailty variance specified by the simulation for each cohort. This experiment therefore evaluates the inversion with known frailty variances."),
    ("2b 2d", "In discrete weekly time (§2.3),", "",
     r"For $\alpha\ne1$, let $B$ be cumulative baseline exposure immediately before the first modeled wave week. KCOR takes $B$ from the preceding end-of-week inverted cumulative hazard, or uses zero at the modeled origin. Under baseline-only selection and positive enrollment frailty variance $\theta$, survivors retain gamma shape $s=1/\theta$, with scale $c=\theta/(1+\theta B)$ and mean $m=sc$. Their observed-hazard moment at entry is $F=c^\alpha\Gamma(s+\alpha)/\Gamma(s)$. Ordinary gamma inversion multiplies a small observed hazard increase at entry by $1/m$, so the corresponding divisor for inverted excess is $\widetilde F=F/m$. With zero frailty variance both factors equal one. Applying this opening divisor to a whole first week approximates selection within the week. For later wave weeks the implementation uses the existing mean-one gamma proxy at the propagated scale as $\widetilde F$. Dividing accumulated excess by the current divisor gives the approximate wave exposure supplied to the post-wave equation. In discrete weekly time (§2.3),"),
    ("2d", "and correct only strictly positive wave excess using", "as in §2.7.3.",
     r"and divide strictly positive inverted excess by $\widetilde F_d(t;\alpha)$ as in §2.7.3."),
    ("6.1", "Identification of depletion geometry is", "approximately common across cohorts,",
     "The fitted frailty variance depends on the scale of the observed cohort hazard,"),
    ("1", "diagnostics that indicate structured depletion", "residual time-varying effects.",
     "diagnostics of fit and stability."),
    ("6.1", "Under minimal aggregated data,", "primary threats to curvature interpretation.",
     "A common multiplier on individual baseline hazards preserves the frailty variance. Rescaling observed cohort hazards changes the fitted variance (§2.4.4)."),
    ("6.1 6.3", "is nonetheless identifiable from curvature structure", "heterogeneous multiplicative scaling and diagnostics).",
     "is estimated from the quiet-window hazard trajectory under the specified Gompertz model (§2.4.4)."),
    ("6.3", "This induces **downward", "true null treatment effect.",
     "With a constant individual baseline hazard this selection produces a concave cohort cumulative hazard, even under a null treatment effect. With an increasing baseline, the sign of the curvature also depends on baseline growth."),
    ("6.3", "relies on curvature in cumulative-hazard space rather than absolute hazard levels.", "",
     "uses the level and time course of the cohort hazard under the specified Gompertz baseline."),
    ("6.1", "The estimator identifies curvature-driven structure", "for heterogeneous scaling and diagnostics).",
     "The distinction between scaling individual baseline hazards and scaling observed cohort hazards is given in §2.4.4."),
    ("6.4", "post-normalization linearity within quiet windows", "",
     "agreement with the fitted cumulative Gompertz shape within quiet windows"),
    ("6.1", "Refinement: effect-scale invariance of curvature-based identification.", "",
     "Effect of hazard scaling on the fitted variance."),
    ("6.1", "Heterogeneous multiplicative effects across cohorts may introduce", "are assessed via diagnostics.",
     "The effect of cohort-specific multipliers depends on the scale on which they act."),
    ("6.1", "Curvature-based identification of", "are evaluated with diagnostics, while",
     "Rescaling observed cohort hazards changes the fitted frailty variances (§2.4.4), while"),
    ("6.3", "Proposition (Identification under curvature).", "", "Proposition (Identification under a fixed Gompertz slope)."),
    ("6.3", "Within the **parametric", "under the stated model.",
     r"In a quiet interval under the Gompertz–gamma model with known slope $\gamma>0$, the positive instantaneous cohort hazard $h$ satisfies $h'=\gamma h-\theta_{0,d}h^2$. Its level and derivative therefore determine $\theta_{0,d}=(\gamma h-h')/h^2$ in the exact model."),
    ("6.3", "This follows from the", "(derivation omitted).",
     r"When $\theta_{0,d}=0$, Gompertz growth alone curves the observed cumulative hazard. When $k_d\theta_{0,d}=\gamma$, the observed hazard is constant and the cumulative hazard is linear despite positive frailty."),
    ("6.3", "sufficient curvature to distinguish", "quiet windows that remain coherent",
     r"sufficient information to distinguish $k_d$ from $\theta_{0,d}$, quiet windows that remain coherent"),
    ("6.3", "Practical identifiability and weak curvature.", "", "Practical identifiability with weekly counts."),
    ("6.3", "Practical identifiability depends on sufficient curvature", "diagnostics are used as intended.",
     "With finite weekly counts, different combinations of baseline rate and frailty variance can give similar fitted hazards. Fit stability and uncertainty then determine how precisely the data support the contrast."),
    ("6.4", "Approximate linearity after normalization remains a diagnostic,", "sole identification anchor.",
     "Agreement with the fitted cumulative Gompertz curve after normalization is one diagnostic."),
    ("6.4", "approximate linearity of", "within the quiet window,",
     "agreement of the normalized cumulative hazard with its fitted Gompertz shape within the quiet window,"),
    ("6.4", "Post-normalization linearity in quiet periods.", "", "Post-normalization shape in quiet periods."),
    ("6.4", "Within prespecified quiet windows (see §2.4.4),", "failure of the offset structure.",
     r"Within a quiet window, the normalized cumulative hazard should follow the fitted cumulative Gompertz curve, allowing for the offset carried into that window. In the continuous model this curve is linear in $(e^{\gamma t}-1)/\gamma$, where $t$ is rebased event time. Systematic departures from the fitted curve warrant investigation."),
    ("6.4", "while linearity after normalization", "under the working model.",
     "while agreement with the fitted Gompertz shape after normalization is consistent with the working model."),
    ("1", "Violations are expected to manifest", "", "Violations can manifest"),
    ("1", "are expected to manifest as diagnostic failures", "", "can manifest as diagnostic failures"),
    ("1", ", in which case diagnostic failure rather than estimator output is the primary signal", "", ""),
    ("1", "Under frailty misspecification, KCOR can degrade", "large unsupported contrasts.",
     "Under frailty misspecification the contrast may move in either direction, even when diagnostic criteria are met."),
    ("1", "or weak identification signal that", "", "or weak identification can signal that"),
    ("1", "would be conservative with respect to the true effect", "", "could attenuate or amplify the contrast"),
    ("1", "Such crossover attenuates", "sustained net benefit or harm.",
     "The resulting change in the contrast depends on who crosses over and when."),
    ("1 6.4", "including poor cumulative-hazard fit and reduced post-normalization linearity,", "explicitly signal",
     "including poor cumulative-hazard fit and departures from the expected Gompertz shape, can signal"),
    ("1 6.4", "Rather than presuming identifiability,", "absorbing those failures into model-dependent estimates.",
     "Degraded fit, offset instability, departures from the expected cumulative shape, or unstable NPH objectives can flag weak identification. Passing these checks leaves identification conditional on the working assumptions."),
]

SUPPLEMENT = [
    ("3.3", "After gamma-frailty normalization (inversion), KCOR should deviate", "illustrate this behavior.",
     "The figure shows the historical positive-control comparison. Main-text Table 9 gives the fitted week-80 ratios and intervals."),
    ("Recovery", "Under the revised estimator, this synthetic setting", "aligns quiet windows coherently.",
     "The separate multiwave recovery example uses expected weekly deaths and fixes the baseline rate at the first observed hazard, which equals the generating rate. It checks recovery of enrollment-time frailty variance and alignment of the quiet windows under those conditions."),
    ("N6", "Positive controls are constructed by starting", "over a prespecified interval:",
     r"During the effect window, the generator multiplies the treated cohort's weekly hazard among survivors by $r$, relative to what that same cohort's hazard would have been without the injection."),
    ("N6", "The injection multiplies the treatment cohort's baseline hazard", "unchanged.",
     r"The treatment cohort's weekly hazard among survivors follows its unperturbed curve outside the effect window and is multiplied by $r$ within it. The control cohort follows its own unperturbed curve throughout."),
    ("N5", "For each window placement, parameters were estimated", "used elsewhere in this SI.",
     "For each placement we fitted baseline rate and frailty jointly to that quiet window using the revised weekly equation (§2.5). The final refit used the window plus the four early post-skip anchor weeks. We retained the original enrollment time scale throughout. The pass rate counts fits accepted by the pipeline across all 13 attempted windows per group, including failures in the denominator."),
    ("N5", "The purpose of this scan is technical rather than substantive:", "not interpreted).",
     "All six groups passed the pipeline diagnostics in all 13 windows (78 of 78 fits). The estimated frailty variance still depended on window placement. For dose 1 it ranged from effectively zero to 7.70 in the 1940s birth cohort and to 14.51 in the 1950s cohort. The windows overlap, so these counts describe sensitivity to placement within the same data."),
    ("N5", "Marker shape denotes birth decade", "any one cohort narrative.",
     "Rows show birth decade and columns show dose. Each dot gives the fitted frailty variance for one window. All 78 fits passed."),
    ("N4 N2w", "is anchored at wave entry", "",
     "starts from the cumulative hazard before the first modeled wave week and integrates the fitted baseline rate over each full week"),
    ("2b 2d", "Applying the NPH correction after gamma-frailty inversion", "amplification (main text §2.7.3).",
     r"For $\alpha\ne1$ the first modeled wave week uses the opening survivor gamma distribution under baseline-only selection. Its $\alpha$ moment divided by its mean gives the opening divisor on the inverted scale. Using it throughout the first week is approximate. Later weeks use the mean-one gamma proxy at the propagated scale (main text §2.7.3)."),
    ("2d", "only strictly positive excess above that path is divided", "and the corrected cumulative hazard",
     r"only strictly positive inverted excess is divided by $\widetilde F_d(t;\alpha)$, and the corrected cumulative hazard"),
    ("1", "KCOR is designed to **fail", "non-identifiability.",
     "Residual structure, unstable fits, or disagreement across quiet windows can reveal a violation. A disturbance that produces a trajectory consistent with the fitted model can pass all of these checks."),
    ("6.1", "Curvature-based identification is invariant to multiplicative scaling", "(main text §2.4.4).",
     "Multiplying individual baseline hazards by a constant preserves the frailty variance. Multiplying observed cohort hazards by a constant changes the fitted variance (main text §2.4.4)."),
    ("6.3", "I2 Curvature sufficiency", "", "I2 Information in quiet windows"),
    ("6.3", "Mortality geometry contains enough curvature", "in the quiet-window regime).",
     "The observed hazard trajectory distinguishes the baseline rate from the frailty variance with sufficient precision under the specified Gompertz model."),
    ("1", "Any failure", "diagnostics indicate non-identifiability.", "A required diagnostic fails."),
    ("6.4", "Assess curvature removal in cumulative-hazard space.", "", "Compare the normalized cumulative hazard with its fitted Gompertz curve."),
    ("6.4", "Approximate linearity after normalization within quiet windows as a diagnostic check under the working model", "",
     "Agreement with the fitted shape after allowing for the window offset"),
    ("6.4", "After frailty normalization, the depletion-neutralized baseline cumulative hazards", "or quiet-window contamination.",
     "In these Czech quiet windows the normalized curves appear approximately straight in event time. The model reference is the fitted cumulative Gompertz curve with the window offset (main text §2.6.2). Deviations can reflect baseline misspecification or time-varying risk."),
]


def apply_edits(text, edits, *, tex=False, marked=False):
    def boundary(phrase):
        if tex:
            phrase = phrase.replace("**", r"\textbf{")
        return re.escape(phrase).replace(r"\ ", r"\s+")

    for issue, first, last, new in edits:
        pattern = boundary(first)
        if last is None:
            pattern += r"(?:(?!\n\s*\n).)*?(?=\n\s*\n|\Z)"
        elif last:
            pattern += r"(?:(?!\n\s*\n).)*?" + boundary(last)
        matches = list(re.finditer(pattern, text, re.S))
        if len(matches) != 1:
            raise ValueError(f"Expected one span for {first!r}, found {len(matches)}")
        m, = matches
        if tex and issue == "N1b":
            new = new.replace("%", r"\%")
        if tex and "@fig:" in new:
            new = re.sub(r"@fig:([a-zA-Z0-9_-]+)", lambda label: r"\ref{fig:" + label[1] + "}", new)
        if tex and marked:
            old = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", m[0])
            words = re.findall(r"\\\(.*?\\\)|\S+", old, re.S)
            deleted = " ".join(r"\reviewdel{" + w + "}" for w in words)
            original_words = " ".join(m[0].split())
            if new.startswith(original_words + " "):
                new = m[0] + " " + reference(issue) + r"{\color{red}" + new[len(original_words) + 1:] + "}"
            else:
                new = (" " if first.startswith(" ") else "") + reference(issue) + deleted + (r" {\color{red}" + new + "}" if new else "")
        text = text[:m.start()] + new + text[m.end():]
    return text
