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
        link = "Generator, lines 234--235" if key == "N1" else "PR 5 source"
        notes.append(r"\noindent\hypertarget{kcor-obj-" + key + r"}{\textbf{[" + key + "] " + title
                     + r"}}\par\noindent " + description + r" \href{" + url
                     + r"}{\textcolor{blue}{" + link + r"}}\par\medskip")
    return tex.replace(r"\end{document}", "\n".join(notes) + "\n" + r"\end{document}")
