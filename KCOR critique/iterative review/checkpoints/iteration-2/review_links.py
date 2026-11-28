"""Source notes and intra-document links for the cumulative redline."""
from urllib.parse import quote


UPSTREAM = "https://github.com/skirsch/KCOR/blob/82f546cd5cc1af2ea99e4501a1aa92d0c9335f78/"


def source(item, filename="README.md"):
    return f"KCOR critique/{item:02} - Objection {item}/{filename}"


# Short labels identify individual concerns within the original numbered objections.
NOTES = {
    "1": ("PR 5, objection 1. Diagnostic failures and bias direction",
          "Quiet-window contamination can resemble depletion and pass the checks. When assumptions fail, the adjusted contrast can move in either direction.",
          source(1, "KCOR_Objection_1_Corrected_Framing.md")),
    "2a": ("PR 5, objection 2. Proportional wave",
           r"At $\alpha=1$ ordinary gamma inversion already recovers the combined hazard. The original equation applied another adjustment to positive excess.",
           source(2, "INITIAL_WAVE_AUDIT.md")),
    "2b": ("PR 5, objection 2. Exposure during a nonlinear wave",
           "Survivors have a changing frailty distribution. Dividing cumulative excess by the current week's approximate frailty factor does not establish cumulative exposure. The revision qualifies this calculation.",
           source(2, "INITIAL_WAVE_AUDIT.md")),
    "2c": ("PR 5, objection 2. Increments after wave exit",
           "A nonlinear wave changes which people survive. Propagating ordinary inverted increments beyond the wave omits this selection. The revised equation conditions on survival through wave exit and the estimated exit exposures.",
           source(2, "post_wave_error/README.md")),
    "3a": ("PR 5, objection 3. Resampling unit",
           "Relocating blocks of weeks can assign deaths to weeks with different underlying risks. The revised procedure samples people at enrollment and retains each event history. The method used for historical bands requires its own evidence.",
           source(3)),
    "3b": ("PR 5, objection 3. Interval coverage",
           "Coverage depends on both the bootstrap draws and the estimator refitted in each draw. The cited simulation results apply to the designs and fitters that produced them.",
           source(3)),
    "4a": ("PR 5, objection 4. Origin of frailty",
           "A flat adjusted mortality contrast cannot distinguish a treatment-induced change in frailty from pre-existing selection. That attribution requires independent evidence.",
           source(4, "CONTROL_CLAIM_AUDIT.md")),
    "4b": ("PR 5, objection 4. Table 8 negative control",
           "The historical near-null caption obscured two 2022 departures. Table 8 now reports the historical and revised estimates separately. The revised column uses the person bootstrap (3a) and exact weekly reconstruction (6.2).",
           source(4, "CONTROL_CLAIM_AUDIT.md")),
    "5": ("PR 5, objection 5. Target in the Cox comparison",
          "The treatment hazard ratio equals one at fixed frailty. With different frailty distributions, that conditional null does not imply that the two cohorts have equal marginal hazards.",
          source(5, "INITIAL_TARGET_AUDIT.md")),
    "6.1": ("PR 5, objection 6.1. Hazard scaling",
            "A constant multiplier on the observed cohort hazard changes the fitted frailty variance. A constant multiplier on individual baseline hazards preserves the generating variance.",
            source(6, "PROPOSED_MANUSCRIPT_WORDING.md#L5-L10")),
    "6.2": ("PR 5, objection 6.2. Weekly reconstruction",
            "The instantaneous hazard identity was used as a finite weekly update. Exact weekly increments come from differencing the cumulative inversion at successive week boundaries.",
            source(6, "PROPOSED_MANUSCRIPT_WORDING.md#L11-L18")),
    "6.3": ("PR 5, objection 6.3. Identification from curvature",
            "Gompertz growth can produce curvature at zero frailty. Positive frailty can coexist with a straight cumulative curve. Identification in the specified model uses the hazard level and derivative at a fixed Gompertz slope.",
            source(6, "PROPOSED_MANUSCRIPT_WORDING.md#L19-L26")),
    "6.4": ("PR 5, objection 6.4. Shape after normalization",
            "The quiet-window reference is a cumulative Gompertz curve with the window offset. That curve is linear in Gompertz-transformed time and may be curved in event time.",
            source(6, "PROPOSED_MANUSCRIPT_WORDING.md#L27-L32")),
    "N1": ("Evaluation N1. Known frailty in the Cox experiment",
           "The evaluation of the November 21 edition (repeat 3, review A, Cox comparison) observed that the paper claimed KCOR estimated frailty from the simulated data. The generator supplies the generating variances directly to the inversion. Section 2.11.1 now identifies what the experiment tested.",
           "test/sim_grid/code/generate_cox_bias_sim.py#L234-L235"),
    "N2": ("Evaluation N2. Matching the weekly observation in the fit",
           "The first-window fit, the pooled refit and the residuals compared deaths accumulated over a week with a model hazard at week entry. The revision predicts the observed hazard between the two week boundaries. Table 8 uses that prediction in the original sample and in every bootstrap draw.",
           "code/KCOR.py#L1541-L1549"),
    "N3": ("Evaluation N3. Final week of a gap",
           "The fitter read cumulative hazards before the last gap week and applied the resulting offset at the next quiet week. A shock in that last gap week was omitted. Step 3 now reads both totals at the start of the next quiet week, including the entire gap.",
           "code/KCOR.py#L1454-L1467"),
    "N4": ("Evaluation N4. First week of a wave",
           "The reference originally started at the observed end-of-week total for the first wave week. This assigned zero excess to that week by construction. The revised reference starts before the first modeled wave week and adds its fitted baseline increment.",
           "code/KCOR.py#L2605-L2614"),
    "N5": ("Evaluation N5. Quiet-window robustness scan",
           "The original scan used a constant baseline and discarded failed windows before computing each pass rate. The revised scan runs the joint weekly fitter on the same 13 placements per group and counts every attempt. The fitted variances show how much estimates change as the window moves.",
           "test/quiet_window/code/quiet_window_scan_theta_czech_2021_24.py#L258-L273"),
    "N6": ("Evaluation 2.6 (N6). Positive-control injection",
           "The generator applies the multiplier to the treated cohort's own weekly hazard among survivors. The supplement previously applied it to the individual baseline hazard and used the control cohort as the reference in the equation.",
           "test/positive_control/code/generate_positive_control.py#L114-L130"),
    "N8": ("Evaluation 2.8 (N8). Reference week for anchored KCOR",
           "Section 2.1.1 defines the anchor as the cumulative ratio at one reference week. Section 2.8 previously described a mean over several weeks. The revised sentence matches the implemented single-week denominator.",
           "code/KCOR.py#L4353-L4364"),
    "N9": ("Evaluation 2.9 (N9). Which groups receive a separate fit",
           "The original bootstrap description assigned all strata a shared cohort fit. Table 8 fits each subgroup and pseudo-dose combination, then pools its anchored contrasts. The revision distinguishes fitted comparison groups from strata used solely for resampling or standardization.",
           "documentation/preprint/supplement.md#L398-L413"),
    "N10": ("Evaluation 2.10 (N10). Selection of the Figure 1 comparison",
            "The script screens on final KCOR and gives priority to comparisons with a reduction in the curvature gap. The lead-in discloses these criteria and the caption describes the selected comparison as an example.",
            "code/generate_kcor_empirical_intuition_figure.py#L101-L146"),
    "N10b": ("Evaluation 2.10 (N10). Quiet windows in the worked example",
             "Section 2.12 describes the settings used in the worked example. The configuration records the intervals used. It does not establish when their boundaries were chosen relative to inspecting the results.",
             "data/Czech/Czech.yaml#L22-L32"),
    "N1b": ("Evaluation follow-up. Uncertainty in the KCOR points in the Cox experiment",
            "The original caption described bootstrap bands, but the plot displayed only point estimates. Vertical bars now show pointwise person-bootstrap intervals with each cohort's generating frailty held fixed.",
            "test/sim_grid/code/plot_cox_bias_sim.py#L59-L84"),
    "Recovery": ("Evaluation follow-up. Multiwave frailty recovery",
                 "Weekly deaths equal their expectations. The noiseless first hazard gives the generating baseline rate, which stays fixed while frailty is estimated. This experiment checks recovery under those conditions.",
                 "test/theta0/theta0_nwave.py#L175-L197"),
    "Fig1": ("Evaluation follow-up. Procedure illustrated in Figure 1",
             "The Czech figure uses monthly hazards adjusted with fitted quiet-period slopes. The caption identifies the procedure that produced the curves.",
             "code/kcor_mortality.py#L353-L375"),
    "Grid": ("Evaluation follow-up. Fitter used in the historical simulation grid",
             "The simulation comparison used one quiet window and fitted cumulative hazard. The caption identifies the estimator used for these results.",
             "test/sim_grid/code/generate_sim_grid.py#L856-L867"),
    "Pilot": ("Evaluation follow-up. Interpretation of the historical coverage pilot",
              "Four scenarios repeated deterministic expected counts. The pilot measured inclusion of an estimate fitted to one reference dataset. Section 2.9.1 explains why those results cannot establish coverage across independently sampled cohorts.",
              "test/sim_grid/code/compute_bootstrap_coverage.py#L201-L229"),
}


def reference(issues):
    keys = issues.split()
    for key in keys:
        if key not in NOTES:
            raise ValueError(f"Unknown objection {key}")
    return "".join(r"\reviewref{" + key + "}" for key in keys)


def mark_document(tex):
    for key in NOTES:
        if reference(key) not in tex:
            raise ValueError(f"No manuscript passage cites objection {key}")
    definition = r"""
\DeclareRobustCommand{\reviewref}[1]{\ifmmode\text{\hyperlink{kcor-obj-#1}{\textcolor{blue}{\normalfont\sffamily\scriptsize[#1]}}}\else\hyperlink{kcor-obj-#1}{\textcolor{blue}{\normalfont\sffamily\scriptsize[#1]}}\fi}
\pdfstringdefDisableCommands{\def\reviewref#1{}}
"""
    tex = tex.replace(r"\begin{document}", definition + r"\begin{document}", 1)
    legend = (r"\noindent\small Red text marks additions. Struck-through text marks removals. "
              r"Select a blue bracketed label to read the objection behind a change.\par")
    tex = tex.replace(r"\subsection{Manuscript metadata}", "{" + legend + "}\n" + r"\subsection{Manuscript metadata}", 1)
    notes = [r"\clearpage\section*{Objections cited in the redline}"]
    for key, (title, description, path) in NOTES.items():
        if key == "4b":
            notes.append(r"\clearpage\section*{Objections cited in the redline (continued)}")
        url = UPSTREAM + quote(path, safe="/#-")
        link = {"N1": "Generator, lines 234--235", "N2": "Earlier fitting objective",
                "N3": "Earlier gap calculation", "N4": "Earlier wave-entry calculation",
                "N5": "Earlier scan summary", "N8": "Reference-week calculation",
                "N9": "Age-specific and pooled-population fits in the supplement",
                "N10": "Figure 1 selection rule", "N10b": "Czech window settings",
                "N1b": "Original KCOR plot", "Recovery": "Multiwave recovery calculation",
                "Fig1": "Monthly slope adjustment", "Grid": "Single-window fit",
                "Pilot": "Pilot reference calculation"}.get(key, "PR 5 source")
        notes.append(r"\noindent\hypertarget{kcor-obj-" + key + r"}{\textbf{[" + key + "] " + title
                     + r"}}\par\noindent " + description + r" \href{" + url
                     + r"}{\textcolor{blue}{" + link + r"}}\par\medskip")
    return tex.replace(r"\end{document}", "\n".join(notes) + "\n" + r"\end{document}")
