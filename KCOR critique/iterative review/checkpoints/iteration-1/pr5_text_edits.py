"""Wording changes made after reviewing the frozen November 21 edition."""
import re

from review_links import reference


# Bound each change by two literal phrases in one source paragraph.
# The same spans drive the Markdown, the clean PDF and the cumulative redline.
PAPER = [
    ("N1", " under identical information constraints", "", ""),
    ("N1", "Observed cumulative hazards were estimated from cohort-level event counts,", "exploiting simulator-only knowledge.",
     "Observed cumulative hazards were estimated from the simulated event data. For the KCOR inversion, we used the frailty variance specified by the simulation for each cohort. This experiment therefore evaluates the inversion with known frailty variances."),
    ("2b", "In discrete weekly time (§2.3),", "",
     r"For $\alpha\ne1$, the implementation approximates the in-wave factor with a mean-one gamma moment evaluated at the propagated scale. The distribution of frailty among survivors generally has a different mean, and a nonlinear wave also changes its shape. Dividing accumulated excess by the current week's factor is an empirical approximation. The resulting wave exposure is the input to the post-wave equation. In discrete weekly time (§2.3),"),
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
    ("2b", "Applying the NPH correction after gamma-frailty inversion", "amplification (main text §2.7.3).",
     r"For $\alpha\ne1$ the in-wave calculation uses a mean-one gamma moment as a proxy for the survivor moment. Rescaling cumulative excess by this time-varying proxy gives an approximate wave exposure (main text §2.7.3)."),
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
        if last:
            pattern += r"(?:(?!\n\s*\n).)*?" + boundary(last)
        matches = list(re.finditer(pattern, text, re.S))
        if len(matches) != 1:
            raise ValueError(f"Expected one span for {first!r}, found {len(matches)}")
        m, = matches
        if tex and marked:
            old = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", m[0])
            words = re.findall(r"\\\(.*?\\\)|\S+", old, re.S)
            deleted = " ".join(r"\reviewdel{" + w + "}" for w in words)
            new = (" " if first.startswith(" ") else "") + reference(issue) + deleted + (r" {\color{red}" + new + "}" if new else "")
        text = text[:m.start()] + new + text[m.end():]
    return text
