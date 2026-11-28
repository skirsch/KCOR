"""Build clean and cumulative redline PDFs of the reviewed combined manuscript."""
import argparse
import csv
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import pr5_text_edits
from review_links import mark_document, reference

HERE = Path(__file__).resolve().parent
PREPRINT = HERE.parents[1] / "documentation" / "preprint"
PAPER = Path("documentation/preprint/paper.md")
SUPPLEMENT = PAPER.with_name("supplement.md")
# Each tuple identifies the exact boundaries in the reviewed Markdown and TeX.
# The heading is a separate span so both versions keep its typography.
EDITS = [
    ("1", "The goal is not to assert", "a misleading normalized estimate.",
     "Poor fit or instability can alert the analyst to model misspecification or contamination of a quiet window. Either can go undetected when the observed mortality pattern is consistent with the fitted model. Passing the checks therefore leaves interpretation conditional on the model and quiet-window assumptions."),
    ("1", "Violations of identifiability assumptions lead", "contrasts.",
     "When the identifiability assumptions fail, the adjusted contrast can be attenuated or amplified. A fitted frailty variance near zero does not establish the direction of the distortion."),
    ("1", "Conservativeness under overlap.", "Conservativeness under overlap.",
     "Treatment effects during a quiet window."),
    ("1", "When treatment effects overlap temporally", "assumptions and stress tests.",
     "If a treatment effect changes mortality during a quiet window, the frailty fit can absorb part of that change. Poor fit or instability when the window boundaries are varied can reveal a problem. If the change resembles the pattern expected from depletion, however, the fit can remain stable. In that case, the diagnostics alone cannot establish how much of the change the adjustment absorbed or whether the adjusted contrast is conservative."),
    ("4a", "The 2022_06 enrollment cohort (Table", "changes detectable at the population level.",
     "Distinguishing a treatment-induced change in frailty from pre-existing selection requires independent evidence about the source of the mortality pattern."),
    ("5", "Thus, the hazard ratio", "between cohorts",
     "Thus, the treatment hazard ratio at fixed frailty"),
    ("5", "the true hazard ratio being 1 by construction", "the true hazard ratio being 1 by construction",
     "the treatment hazard ratio at fixed frailty being 1 by construction"),
    ("5", "Bias in Cox-based estimators increases", "across the same parameter range.",
     "The standard Cox estimate departs progressively farther from the null treatment-effect benchmark as frailty variance increases on this grid. KCOR remains close to its cumulative-hazard target of 1 across the same parameter range."),
    ("6.1", "A constant multiplicative hazard effect (i.e.,", "within the identification regime.",
     r"With the Gompertz slope fixed, multiplying the observed cohort hazard by a constant $c>0$ gives model parameters $c k_d$ and $\theta_{0,d}/c$, whereas multiplying each individual's baseline hazard by $c$ preserves the frailty variance $\theta_{0,d}$."),
    ("6.2", "so the Step 2 recursion", "for arbitrary cohort hazards.",
     "where the hazards are instantaneous. Under the gamma-frailty working model, differencing the cumulative inversion at consecutive week boundaries gives the exact weekly update in Step 2."),
]


def edit(text, marked=False, *, tex=False):
    for issue, first, last, replacement in EDITS:
        if tex:
            first = first.replace("_", r"\_")
        pattern = re.escape(first).replace(r"\ ", r"\s+")
        if first != last:
            pattern += r".*?" + re.escape(last)
        matches = list(re.finditer(pattern, text, re.DOTALL))
        if len(matches) != 1:
            raise ValueError(f"Expected one passage starting with {first!r}")
        match, = matches
        if marked:
            # Box each word or complete math expression so paragraphs still wrap.
            passage = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", match.group())
            words = re.findall(r"\\\(.*?\\\)|\S+", passage, re.DOTALL)
            old = " ".join(r"\reviewdel{" + w + "}" for w in words)
            replacement = reference(issue) + old + r" {\color{red}" + replacement + "}"
        text = text[:match.start()] + replacement + text[match.end():]
    return weekly_inversion_edit(text, tex=tex, marked=marked)


def weekly_inversion_edit(text, *, tex, marked):
    old = r"h_{0,d}^{\mathrm{eff}}(t)=h_d^{\mathrm{adj}}(t)\left(1+\hat{\theta}_{0,d}^{(m)}H_{0,d}^{\mathrm{eff}}(t)\right),"
    new = r"h_{0,d}^{\mathrm{eff}}(t)=\left(1+\theta H_{0,d}^{\mathrm{eff}}(t)\right)\frac{\operatorname{expm1}(\theta h_d^{\mathrm{adj}}(t))}{\theta},"
    assert text.count(old) == 1
    if marked:
        # Give the deleted equation its own display so both versions fit.
        new = reference("6.2") + r"\reviewdel{\(\displaystyle " + old + r"\)}" + "\n\\]\n\\[\n" + r"{\color{red}" + new + "}"
    text = text.replace(old, new)
    note = (r" Here $\theta=\hat{\theta}_{0,d}^{(m)}$ and $h_d^{\mathrm{adj}}(t)$ is the observed integrated "
            r"hazard for that week after the stabilization skip. We evaluate $\operatorname{expm1}(x)=e^x-1$ "
            r"for numerical stability. At $\theta=0$ the limiting increment equals $h_d^{\mathrm{adj}}(t)$.")
    if tex:
        note = re.sub(r"\$(.+?)\$", lambda m: r"\(" + m[1] + r"\)", note)
    start = text.index("starting from ", text.index(new))
    pos = text.index("\n\n", start)
    return text[:pos] + (reference("6.2") + r"{\color{red}" + note + "}" if marked else note) + text[pos:]


def wave_identity(text, *, tex=False, marked=False, supplement=False):
    """Insert the proportional-wave identity and qualify the existing division."""
    def insert_before(anchor, addition):
        nonlocal text
        if text.count(anchor) != 1:
            raise ValueError(f"Expected one wave-correction anchor: {anchor!r}")
        if tex:
            addition = re.sub(r"\$(.+?)\$", lambda m: r"\(" + m[1] + r"\)", addition)
        addition = (reference("2a") + r"{\color{red}" + addition + "}") if marked else addition
        text = text.replace(anchor, addition + anchor)

    identity = r"$H_{\mathrm{corr},d}(t)=\tilde H_{0,d}(t)$"
    if supplement:
        insert_before("In discrete weekly bins, a Gompertz reference path",
                      rf"For $\alpha=1$, {identity} throughout follow-up. "
                      r"The following rescaling applies only for $\alpha\ne1$. ")
    else:
        insert_before("The wave-induced excess in cumulative-hazard space is",
                      r"When $\alpha=1$, both baseline and wave hazards are proportional "
                      r"to individual frailty. Ordinary gamma inversion then recovers "
                      r"their combined cumulative hazard under the working model. "
                      rf"The identity case is {identity} for every week $t$. ")
        old = r"H_{\mathrm{ref},d}(t_{0,d})=\tilde H_{0,d}(t_{0,d})"
        week_factor = r"\,\frac{e^\gamma-1}{\gamma}"
        new = (r"H_{\mathrm{ref},d}(t_{0,d})=\tilde H_{0,d}(t_{0,d}-1)"
               r"+\hat{k}_d\,\exp\!\left(\gamma\,t_{\mathrm{rebased}}(t_{0,d})\right)" + week_factor)
        if text.count(old) != 1:
            raise ValueError("Expected the original wave-entry equation")
        if marked:
            new = (reference("N4 N2w") + r"\reviewdel{\(\displaystyle " + old + r"\)}"
                   + "\n\\]\n\\[\n" + r"{\color{red}" + new + "}")
        text = text.replace(old, new)
        recurrence = r"H_{\mathrm{ref},d}(t)=H_{\mathrm{ref},d}(t-1)+\hat{k}_d\,\exp\!\left(\gamma\, t_{\mathrm{rebased}}(t)\right)"
        if text.count(recurrence) != 1:
            raise ValueError("Expected one subsequent-week wave reference equation")
        text = text.replace(recurrence, recurrence + (
            reference("N2w") + r"{\color{red}" + week_factor + "}" if marked else week_factor))
        anchor = r"{F_d(t;\alpha)},"
        if text.count(anchor) != 1:
            raise ValueError("Expected one positive wave-excess equation")
        condition = r"\qquad(\alpha\ne1)"
        divisor = r"\widetilde F_d(t;\alpha)"
        if marked:
            divisor = (reference("2d") + r"\reviewdel{\(F_d(t;\alpha)\)}" + r"{\color{red}" + divisor + "}")
        text = text.replace(anchor, "{" + divisor + "}" + (
            reference("2a") + r"{\color{red}" + condition + "}" if marked else condition) + ",")
    return text



POST_WAVE = r"""At the last wave boundary $e$, retain the baseline and wave contributions from the in-window calculation. When the cumulative excess is positive, set $B_e=H_{\mathrm{ref},d}(e)$ and $W_e=\Delta H_{\mathrm{wave},d}(e)/\widetilde F_d(e;\alpha)$ before the cumulative maximum repair. Otherwise set $B_e=\tilde H_{0,d}(e)$ and $W_e=0$. Here $W_e$ is the estimated wave hazard at reference frailty $Z=1$. The exponent and exposure are held fixed after wave exit.

Let $Z$ have the starting mean-one gamma distribution for cohort $d$. Write $L(B,W)=E[\exp(-ZB-Z^\alpha W)]$. For each later week $t$, solve for the baseline increase $u(t)$ since wave exit using the observed survival relative to that boundary.

$$
\frac{L(B_e+u(t),W_e)}{L(B_e,W_e)}
=\exp\!\left[-\bigl(H_{\mathrm{obs},d}(t)-H_{\mathrm{obs},d}(e)\bigr)\right].
$$

At wave exit $u(e)=0$. The left side decreases strictly as $u$ increases. Set $H_{\mathrm{corr},d}(t)=H_{\mathrm{corr},d}(e)+u(t)$ and difference this curve to obtain weekly increments. This retains the wave contribution in the all-cause comparison and accounts for selection of survivors through wave exit. The post-wave equation is exact conditional on the exit exposures and the starting frailty model. Accuracy with fitted inputs depends on the in-wave attribution."""

SUPP_TAIL = (" After wave exit, hold the estimated cumulative wave exposure fixed and "
             "recover each baseline increase from survival conditional on reaching wave exit "
             "(main text §2.7.3).")
WORKFLOW_TAIL = " After wave exit, apply the conditional survival equation to subsequent weeks."


def post_wave_edit(text, *, tex=False, marked=False, supplement=False):
    def new(content):
        if tex:
            content = re.sub(r"\$\$\n(.*?)\n\$\$", lambda m: r"\[" + m[1] + r"\]", content, flags=re.S)
            content = re.sub(r"\$([^$]+)\$", lambda m: r"\(" + m[1] + r"\)", content)
        return (reference("2c") + r"{\color{red}" + content + "}") if marked else content

    if supplement:
        anchor = "be non-decreasing." if tex else "enforced to be non-decreasing."
        start = text.index("Post-inversion correction.")
        idx = text.index(anchor, start) + len(anchor)
        return text[:idx] + new(SUPP_TAIL) + text[idx:]

    start = text.index(r"For weeks \textbf{after}" if tex else "For weeks **after**")
    end = text.index("Weekly corrected hazard increments", start)
    old = text[start:end].strip()
    if marked:
        # Preserve the original displayed equation and allow prose to wrap.
        old = old.replace(r"\textbf{after}", "after").replace(r"\textbf{continuous}", "continuous")
        tokens = re.findall(r"\\\[.*?\\\]|\\\(.*?\\\)|\S+", old, re.S)
        parts = []
        for word in tokens:
            if word.startswith(r"\["):
                parts.append("\n\n" + r"\[\reviewdel{\(\displaystyle " + word[2:-2] + r"\)}\]" + "\n\n")
            else:
                parts.append(r"\reviewdel{" + word + "}")
        old = reference("2c") + " ".join(parts) + "\n\n"
    else:
        old = ""
    text = text[:start] + old + new(POST_WAVE) + "\n\n" + text[end:]
    # Remove the former summary, which described only the in-window transform.
    summary = re.compile(r"Thus the\s+optional module adjusts only positive wave excess above the fitted\s+frailty-neutral baseline path; it does not replace the core KCOR\s+estimator\.")
    match = summary.search(text, start)
    if match is None:
        raise ValueError("Expected the original wave summary")
    deleted = reference("2c") + " ".join(r"\reviewdel{" + word + "}" for word in match[0].split()) if marked else ""
    text = text[:match.start()] + deleted + text[match.end():]
    idx = text.index("as in §2.7.3.", text.index("During the NPH window, construct")) + len("as in §2.7.3.")
    return text[:idx] + new(WORKFLOW_TAIL) + text[idx:]


def person_bootstrap_edit(text, *, tex=False, marked=False, supplement=False):
    """Update the resampling unit in the manuscript and its combined TeX edition."""
    def replace(pattern, replacement, *, issue="3a", count=1):
        nonlocal text
        matches = list(re.finditer(pattern, text, re.S))
        if len(matches) != count:
            raise ValueError(f"Expected {count} bootstrap passages for {pattern!r}, found {len(matches)}")
        for m in reversed(matches):
            new = replacement
            if marked:
                old = re.sub(r"\\textbf\{([^{}]+)\}", r"\1", m[0])
                words = re.findall(r"\\\(.*?\\\)|\S+", old, re.S)
                deleted = " ".join(r"\reviewdel{" + word + "}" for word in words)
                new = reference(issue) + deleted + (r" {\color{red}" + new + "}" if new else "")
            text = text[:m.start()] + new + text[m.end():]

    if supplement:
        replace(r"When uncertainty bands or bootstrap intervals are reported.*?resampling individuals\.",
                "For real-data Monte Carlo intervals, the resampling unit is one eligible "
                "person at enrollment. Each draw retains the complete record and contributes "
                "to the counts for its original cohort and follow-up weeks (main text §2.9.1).")
        # The two historical captions explicitly attributed their bands to aggregate resampling.
        # Remove that unsupported attribution. The plotted lines are described in each caption.
        if tex:
            pattern = r"Uncertainty bands \(95\\%\s+bootstrap intervals; aggregated cohort--time\s+resampling\) are\s+shown\."
            # Also correct the unprinted alternate text without adding redline commands to it.
            text = re.sub(r"^\\pandocbounded\{.*$", lambda m: re.sub(pattern, "", m[0]), text, flags=re.M)
        else:
            pattern = r"Uncertainty bands \(95% bootstrap intervals; aggregated cohort--time resampling\) are shown\."
        replace(pattern, "", count=2)
        return text

    replace(r"stratified bootstrap resampling", "person-level bootstrap resampling")
    replace(r"Stratified bootstrap\s+procedure", "Person-level bootstrap procedure")
    replace(r"The stratified bootstrap procedure", "The person-level bootstrap procedure")
    replace(r"Resample cohorts using a block-preserving aggregated\s+bootstrap\.",
            "Resample people at enrollment.")
    replace(r"For each cohort and stratum, bootstrap replicates are generated by.*?sequentially so\s+that",
            "Use all eligible people alive at the start of the selected enrollment week "
            "for the original estimate. For each bootstrap replicate, draw the same number "
            "of complete records with replacement from this pooled population. Retain each "
            "drawn person's cohort characteristics and event history, including the original "
            "death week if a death occurred. Rebuild weekly deaths and numbers at risk from "
            "the sampled records. In each cohort,")
    replace(r"holds identically within each replicate\..*?while operating on aggregated cohort.{1,2}time data\.",
            "holds within each replicate because each sampled copy contributes to the risk "
            "set at the start of each week it is still alive.")
    replace(r"Importantly, event-time bins are.*?resampling individual-level records\.",
            "Cohort and stratum sizes can vary across draws from the enrollment population. "
            "The original estimate is excluded when taking percentiles of the bootstrap replicates.")
    replace(r"Bootstrap intervals are calibrated under the working gamma frailty\s+model\.",
            "Coverage depends on the resampling procedure and the estimator refitted in each draw.", issue="3b")
    replace(r"Sub-nominal coverage arises in regimes.*?results are not interpreted\.",
            "Two person-bootstrap studies used one million people, unequal cohort frailties, "
            "and no wave adjustment. With the joint initial fit specified in section 2.4, "
            "the nominal 95 percent interval contained the true week-52 ratio in 189 of 200 "
            "datasets. In a separate study the production fitter fixed the baseline rate "
            "from early weeks. Its percentile interval contained the true ratio in 502 of "
            "600 datasets, although all 600 original fits passed the study diagnostics. "
            "The studies used different designs and do not isolate the effect of the fitter. "
            "Both coverage studies predate the week-boundary fitting equations in section 2.5.", issue="3b N2")
    bold_start, bold_end = (r"\\textbf\{", r"\}") if tex else (r"\*\*", r"\*\*")
    replace(bold_start + r"Whether empirical coverage is close to nominal in.*?definitive\s+calibration\."
            + bold_end + " ", "", issue="Pilot")
    pilot_ref = r"\\ref\{tbl:bootstrap_coverage\}" if tex else "@tbl:bootstrap_coverage"
    replace(r"; Table " + pilot_ref, "", issue="Pilot")
    replace(bold_start + "Interpretation of sub-nominal bootstrap coverage" + bold_end,
            "", issue="Pilot")
    replace("Table " + pilot_ref + r" reports empirical coverage.*?supporting diagnostic gating for inference\.",
            "", issue="Pilot")
    text = retire_coverage_table(text, tex=tex, marked=marked)
    return table8_edit(text, tex=tex, marked=marked)


def retire_coverage_table(text, *, tex, marked):
    """Remove the historical placeholder, retaining its struck cells in the redline."""
    if not tex:
        start = text.index("Table: Bootstrap coverage for KCOR uncertainty intervals")
        end = text.index(r"\newpage", start)
        return text[:start] + text[end:]

    caption = text.index(r"\caption{Bootstrap coverage for KCOR uncertainty intervals")
    start = text.rindex(r"\newpage", 0, caption)
    end = text.index(r"\end{longtable}", caption) + len(r"\end{longtable}")
    if not marked:
        return text[:start] + text[end:]

    def strike(words):
        words = re.sub(r"\\(?:textbf|emph)\{([^{}]+)\}", r"\1", words)
        tokens = re.findall(r"\\\(.*?\\\)|\S+", words, re.S)
        return " ".join(r"\reviewdel{" + word + "}" for word in tokens)

    table = text[start:end]
    table = re.sub(r"\\caption\{(.*?)\}\\label", lambda m: r"\caption{" + reference("Pilot")
                   + strike(m[1]) + r"}\label", table, count=1, flags=re.S)
    table = re.sub(r"(\\raggedright\n)([^\\]+)(\n\\end\{minipage\})",
                   lambda m: m[1] + strike(m[2]) + m[3], table)
    first_row = table.index("\n", table.index(r"\endlastfoot")) + 1
    last_row = table.index(r"\end{longtable}", first_row)
    rows = re.sub(r"([^&]+?)( & | \\\\)", lambda m: strike(m[1]) + m[2],
                  table[first_row:last_row]).strip()
    return text[:start] + table[:first_row] + rows + "\n" + table[last_row:] + text[end:]


def table8_edit(text, *, tex, marked):
    def added(words, issues="4b"):
        return reference(issues) + r"{\color{red}" + words + "}" if marked else words

    results = json.loads((HERE.parent / "objections/1.2/entry-moment/table8_bootstrap.json").read_text())
    assert results["draws_per_enrollment"] == 999
    summary = results["summary"]
    start = text.index("empirical negative control (pooled/ASMR summaries)")
    end = text.index(r"\end{longtable}" if tex else "\n\n", text.index("1.0911", start))
    part = text[start:end]
    contrast = re.search(r", showing near-null\s+behavior under large composition differences\.", part)
    old = contrast.group()
    deleted = " ".join(r"\reviewdel{" + word + "}" for word in old.split()) if marked else ""
    note = (". The historical estimates and analytic intervals were reproduced with KCOR v4.3. "
            "The joint-fit estimates and person-bootstrap intervals use the same input and reporting date "
            "(December 26, 2022).")
    part = part[:contrast.start()] + deleted + added(note, "4b 3.1 N2w 2d") + part[contrast.end():]
    col = "Joint fit [95% CI]"
    if tex:
        header_end = r"95\% CI \\"
        assert part.count(header_end) == 2
        part = part.replace(header_end, r"95\% CI & " + added(r"Joint fit [95\% CI]", "3a 6.2 N2 N3 N4") + r" \\")
    else:
        part = part.replace("| 95% CI |", "| 95% CI | " + col + " |")
        part = part.replace("|---|---|---:|---|", "|---|---|---:|---|---|")
    for sh, values in summary.items():
        label = sh.replace("_", r"\_") if tex else sh
        for j, (n, d) in enumerate(results["pairs"]):
            key = f"{label} & {n} vs {d} &" if tex else f"| {label} | {n} vs {d} |"
            row = re.search("^" + re.escape(key) + ".*$", part, re.M)
            cell = f"{values['joint'][j]:.4f} [{values['lower'][j]:.3f}, {values['upper'][j]:.3f}]"
            row_end = r" \\" if tex else " |"
            assert row.group().endswith(row_end)
            part = part[:row.end()-len(row_end)] + (" & " if tex else " | ") + (r"{\color{red}" + cell + "}" if marked else cell) + part[row.end()-len(row_end):]
    prefix = text[:start]
    if tex:
        pos = prefix.rindex(r"\begin{longtable}[]{@{}llrl@{}}")
        prefix = prefix[:pos] + prefix[pos:].replace("llrl@", "llrll@", 1)
    text = prefix + part + text[end:]
    anchor = "provides numeric summaries."
    assert text.count(anchor) == 1
    fixed = summary["2022_06"]["fixed"]
    row = summary["2022_06"]
    fixed_results = [f"{v:.4f} [{lo:.3f}, {hi:.3f}]" for v, lo, hi in
                     zip(fixed, row["fixed_lower"], row["fixed_upper"])]
    included = sum(lo <= 1 <= hi for v in summary.values() for lo, hi in zip(v["lower"], v["upper"]))
    inclusion = ("All six joint-fit intervals included 1." if included == 6 else
                 f"The joint-fit intervals included 1 in {included} of the six comparisons.")
    addition = (" For the revised estimates, the fit used week-boundary hazard increments (§2.5). "
                "Baseline rate and frailty were fitted jointly in the first quiet "
                "window, with the Czech wave exponent fixed at 1.18. Each of 999 person-bootstrap draws per "
                "enrollment preserved subgroup membership and week of death, refitted the cohorts, and recomputed "
                "the pooling weights. The intervals give the 2.5th and 97.5th percentiles. Compared with v4.3, "
                "the two 2022 comparisons involving pseudo-dose 2 moved closer to 1, while the pseudo-dose "
                f"1 versus 0 estimate moved above 1. {inclusion} Estimating the baseline rate from the early weeks "
                f"gave 2022 ratios (with 95 percent intervals) of {fixed_results[0]}, {fixed_results[1]} and {fixed_results[2]} "
                "for comparisons 1 versus 0, 2 versus 0 and 2 versus 1, respectively.")
    if all(abs(x-1) < abs(y-1) for x, y in zip(fixed, row["joint"])):
        addition += " All three were closer to 1 than their joint-fit counterparts."
    return text.replace(anchor, anchor + added(addition, "4b 3a N2 N3 N4 3.1"))


def weekly_fit_edit(text, *, tex, marked, paper=True, residuals=1):
    """Use a week-boundary prediction in the seed, pooled fit and diagnostics."""
    seed = r"\frac{k_d e^{\gamma t_{\mathrm{rebased}}}}{1+\theta_{0,d} H_{\mathrm{gom},d}(t_{\mathrm{rebased}})}"
    pooled = r"\frac{\hat{k}_d e^{\gamma t_{\mathrm{rebased}}}}{1+\theta_{0,d}\left(H_{\mathrm{gom},d}(t_{\mathrm{rebased}})+\Delta_d(t)\right)}"
    def predictor(old, new, count):
        nonlocal text
        if text.count(old) != count:
            raise ValueError(f"Expected {count} weekly fitting expressions for {old}")
        if marked:
            new = (r"\begin{gathered}" + reference("N2") + r"\reviewdel{\(\displaystyle "
                   + old + r"\)}\\{\color{red}" + new + r"}\end{gathered}")
        text = text.replace(old, new)

    if paper:
        step = "This offset calculation summarizes"
        assert text.count(step) == 1
        note = ("A nonpositive incremental estimate contributes zero to the cumulative offset. "
                "The signed estimate is recorded, and frailty is refitted across the aligned quiet windows.")
        text = text.replace(step, (reference("3.1") + r"{\color{red}" + note + "}" if marked else note)
                            + " " + step)
        intro = "Estimate " + (r"\(" if tex else "$") + r"(k_d,\theta_{0,d}^{(0)})"
        assert text.count(intro) == 1
        note = (r"Write $s=t_{\mathrm{rebased}}$ for the start of week $t$, and write $G(s;k)$ for the "
                r"cumulative Gompertz baseline in §2.4.3 at scale $k$. For a cumulative offset $a$ "
                r"held fixed within the week, the predicted observed weekly hazard is" + "\n\n"
                + r"$$" + "\n" + r"q_d(t;k,\theta,a)=\frac{1}{\theta}\log\!\left(1+"
                r"\frac{\theta [G(s+1;k)-G(s;k)]}{1+\theta [G(s;k)+a]}\right)." + "\n$$\n\n"
                + r"At $\theta=0$ this equals $G(s+1;k)-G(s;k)$. Both fitting steps compare "
                r"this week-long increment with the observed weekly increment.")
        if tex:
            note = re.sub(r"\$\$\n(.*?)\n\$\$", lambda m: r"\[" + m[1] + r"\]", note, flags=re.S)
            note = re.sub(r"\$([^$]+)\$", lambda m: r"\(" + m[1] + r"\)", note)
        if marked:
            note = reference("N2") + r"{\color{red}" + note + "}"
        text = text.replace(intro, note + "\n\n" + intro)
        predictor(seed, r"q_d(t;k_d,\theta_{0,d},0)", 1)
        predictor(pooled, r"q_d(t;\hat{k}_d,\theta_{0,d},\Delta_d(t))", 1)
    predictor(pooled.replace(r"\theta_{0,d}", r"\hat{\theta}_{0,d}"),
              r"q_d(t;\hat{k}_d,\hat{\theta}_{0,d},\Delta_d(t))", residuals)
    supp_anchor = "Fit residuals should be examined in hazard space."
    if supp_anchor in text:
        desc = " The weekly prediction is defined in Step 1 of main-text §2.5."
        text = text.replace(supp_anchor, supp_anchor + (reference("N2") + r"{\color{red}" + desc + "}" if marked else desc))
    return text


def positive_control_replay(text, *, tex=False, marked=False):
    caption = "Positive control validation: KCOR tracks injected effects in the expected direction."
    bottom = r"Bottom row shows $\mathrm{KCOR}(t)$ deviating from 1.0 in the expected direction during the effect window."
    edits = [
        ("3.3", caption, "", "Historical positive-control comparison."),
        ("3.3", "Bottom row shows", "during the effect window.",
         "The lower panels show anchored cumulative hazard ratios reproducible directly from the raw counts. Table 9 gives the fitted endpoints."),
    ]
    if tex:
        text, count = re.subn(r"\\caption\{Positive control validation:.*?\\label\{fig:pos_control_injected\}",
            lambda m: pr5_text_edits.apply_edits(m[0], edits, tex=True, marked=marked), text, flags=re.S)
        assert count == 1
        text = re.sub(r"alt=\{Positive control validation:.*?\}\]\{figures/fig_pos_control_injected.png",
            lambda _: "alt={Raw positive-control hazard curves and anchored cumulative hazard ratios.}]{figures/fig_pos_control_injected.png", text)
    elif caption in text:
        assert bottom in text
        text = pr5_text_edits.apply_edits(text, edits)
    if "tbl:pos_control_summary" not in text or ("| Benefit |" not in text and not tex):
        return text

    summary = json.loads((HERE.parent / "objections/3.3/results.json").read_text())["scenarios"]
    for scenario, old in (("benefit", "0.825"), ("harm", "1.107")):
        kcor, lo, hi = summary[scenario]["fitted"]
        value = f"{kcor:.3f} [{lo:.3f}, {hi:.3f}]"
        def replace_row(m):
            row = m[0]
            old_weeks = "20--80" if tex else "20–80"
            new_weeks = "20--79" if tex else "20–79"
            assert old_weeks in row and row.count(old) == 1
            if marked:
                new_weeks = r"\reviewdel{" + old_weeks + r"} {\color{red}" + new_weeks + "}"
                cell = r"\reviewdel{" + old + r"} {\color{red}" + value + "}"
            else:
                cell = value
            return row.replace(old_weeks,new_weeks).replace(old,cell)
        prefix = scenario.title() + " & week " if tex else "| " + scenario.title() + " | week "
        text, count = re.subn("^" + re.escape(prefix) + r".*$", replace_row, text, flags=re.M)
        assert count == 1
    return text


def positive_control_equation(text, *, marked=False):
    old = r"h_{0,\mathrm{treated}}(t) = r \cdot h_{0,\mathrm{control}}(t) \quad \text{for } t \in [t_1, t_2],"
    new = r"h_{\mathrm{treated,injected}}(t) = r \, h_{\mathrm{treated,unperturbed}}(t) \quad \text{for } t_1 \le t < t_2,"
    if text.count(old) != 1:
        raise ValueError("Expected one positive-control injection equation")
    if marked:
        new = (r"\begin{gathered}" + reference("N6") + r"\reviewdel{\(\displaystyle "
               + old + r"\)}\\{\color{red}" + new + r"}\end{gathered}")
    return text.replace(old, new)


def contamination_table(text, *, tex=False, marked=False):
    grid = HERE.parent / "objections/2.8/planned-anchor-grid.csv"
    with grid.open(newline="") as stream:
        cells = {(row["contamination_type"], float(row["epsilon_dyn"]), float(row["epsilon_comorb"])): row
                 for row in csv.DictReader(stream)}
    rows = (
        ("dynamic_agg", .005, 0, "0.782", "0.00"),
        ("comorbidity_agg", 0, .0001, "0.989", "0.40"),
        ("comorbidity_agg", 0, .0003, "1.062", "0.65"),
        ("comorbidity_agg", 0, .001, "1.194", "1.00"),
        ("combined_agg", .0005, .0003, "1.016", "0.60"),
        ("combined_agg", .001, .0003, "0.992", "0.50"),
        ("combined_agg", .0005, .001, "1.188", "1.00"),
    )
    for mode, dynamic, comorbidity, old_mean, old_fraction in rows:
        row = cells[mode, dynamic, comorbidity]
        mean, lo, hi, fraction = (float(row[key]) for key in
                                 ("KCOR_asymptote_mean", "lower95", "upper95", "frac_reps_kcor_gt_1"))
        separator = "&" if tex else "|"
        pattern = re.escape(old_mean) + r"\s+" + re.escape(separator) + r"\s+" + re.escape(old_fraction)
        interval = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
        incidence = f"{fraction:.2f}"
        if marked:
            interval = rf"\reviewdel{{{old_mean}}} {{\color{{red}}{interval}}}"
            if incidence != old_fraction:
                incidence = rf"\reviewdel{{{old_fraction}}} {{\color{{red}}{incidence}}}"
        text, count = re.subn(pattern, lambda _: f"{interval} {separator} {incidence}", text)
        if count != 1:
            raise ValueError(f"Expected one historical contamination row for {mode}, {dynamic}, {comorbidity}")
    old_path = "test/quiet_window_contamination/out/quiet_contamination_agg.csv"
    new_path = "KCOR critique/objections/2.8/planned-anchor-grid.csv"
    if tex:
        old_path = r"\protect\texttt{" + old_path.replace("_", r"\_") + "}"
        new_path = r"\texttt{KCOR critique/objections/2.8/planned-anchor-grid.csv}"
    else:
        old_path = chr(96) + old_path + chr(96)
        new_path = chr(96) + new_path + chr(96)
    if text.count(old_path) != 1:
        raise ValueError("Expected one historical contamination data citation")
    if marked:
        new_path = rf"\reviewdel{{{old_path}}} {{\color{{red}}{new_path}}}"
    return text.replace(old_path, new_path)


def build(marked, output):
    source = (PREPRINT / "paper.tex").read_text(encoding="utf-8")
    # The Pandoc TeX is shared with the earlier editions. Select the revised
    # plot here and edit its printed caption separately from the alt text.
    plot = "figures/fig_si_quiet_window_theta_scan_czech_2021_24.png"
    figure_path = HERE / "revised" / SUPPLEMENT.parent / plot
    source, count = re.subn(r"alt=\{Quiet-window robustness scan:.*?\}\]\{" + re.escape(plot) + r"\}",
                            lambda _: "alt={Frailty variance by quiet-window midpoint for six Czech groups}]{" +
                            figure_path.as_posix() + "}", source)
    if count != 1 or not figure_path.is_file():
        raise ValueError("Expected one revised quiet-window scan figure")
    plot = "figures/fig_cox_bias_kcor_vs_theta.png"
    figure_path = HERE / "revised" / PAPER.parent / plot
    source, count = re.subn(r"alt=\{[^\n]*?\}\]\{" + re.escape(plot) + r"\}",
                            lambda _: "alt={Week-52 KCOR at seven frailty variances. All seven "
                            "vertical 95 percent person-bootstrap intervals include one.}]{"
                            + figure_path.as_posix() + "}", source)
    if count != 1 or not figure_path.is_file():
        raise ValueError("Expected the revised KCOR uncertainty figure")
    # Pandoc duplicates Figure 1's caption in its alt text. Keep accessibility
    # text clean while marking the edit in the printed caption below.
    source, count = re.subn(r"(alt=\{Empirical illustration of the KCOR workflow[^}]*?)a representative analysis example",
                            r"\1an analysis example", source)
    if count != 1:
        raise ValueError("Expected one Figure 1 alt description")
    source = re.sub(r"^\\pandocbounded\{.*?alt=\{Empirical illustration of the KCOR workflow.*$",
                    lambda m: pr5_text_edits.apply_edits(m[0], [entry for entry in pr5_text_edits.PAPER
                                                               if entry[0] == "Fig1"]), source, flags=re.M)
    for filename, alt in (
        ("fig_quiet_contam_kcor_asymptote.png", "Mean KCOR at the fifth-week reference for all three contamination sweeps."),
        ("fig_quiet_contam_theta_error.png", "Frailty fit errors and the fraction of combined runs above one at the fifth-week reference."),
    ):
        plot = f"figures/{filename}"
        figure_path = HERE / "revised" / PAPER.parent / plot
        source, count = re.subn(r"alt=\{[^\n]*?\}\]\{" + re.escape(plot) + r"\}",
                                lambda _: f"alt={{{alt}}}]{{{figure_path.as_posix()}}}", source)
        if count != 1 or not figure_path.is_file():
            raise ValueError(f"Expected one revised contamination figure for {filename}")
    for changes in (pr5_text_edits.PAPER, pr5_text_edits.SUPPLEMENT):
        source = pr5_text_edits.apply_edits(source, changes, tex=True, marked=marked)
    source = contamination_table(source, tex=True, marked=marked)
    source = positive_control_equation(source, marked=marked)
    source = edit(source, marked, tex=True)
    source = weekly_fit_edit(source, tex=True, marked=marked, residuals=2)
    source = wave_identity(source, tex=True, marked=marked)
    source = wave_identity(source, tex=True, marked=marked, supplement=True)
    source = post_wave_edit(source, tex=True, marked=marked)
    source = post_wave_edit(source, tex=True, marked=marked, supplement=True)
    source = person_bootstrap_edit(source, tex=True, marked=marked)
    source = person_bootstrap_edit(source, tex=True, marked=marked, supplement=True)
    source = positive_control_replay(source, tex=True, marked=marked)
    if marked:
        source = source.replace(r"\begin{document}", r"""
\DeclareRobustCommand{\reviewdel}[1]{\leavevmode{\setbox0=\hbox{#1}\rlap{\raisebox{.5ex}{\rule{\wd0}{.45pt}}}\box0\relax}}
\begin{document}""", 1)
        source = mark_document(source)
    with tempfile.TemporaryDirectory(prefix="kcor-typeset-") as tmp:
        work = Path(tmp)
        tex = work / "manuscript.tex"
        tex.write_text(source, encoding="utf-8")
        for _ in range(2):
            result = subprocess.run(
                ["xelatex", "-disable-installer", "-interaction=nonstopmode",
                 "-halt-on-error", f"-output-directory={work}", str(tex)],
                cwd=PREPRINT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(result.stdout[-4000:].decode(errors="replace"))
        shutil.copyfile(work / "manuscript.pdf", output)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--initialize-approved-text", action="store_true",
                      help="Replace the candidate Markdown with the approved baseline edits.")
    args.add_argument("--version", choices=["clean", "redline", "both"], default="both")
    options = args.parse_args()
    for path in (PAPER, SUPPLEMENT):
        baseline = (HERE / "baseline" / path).read_text(encoding="utf-8")
        changes = pr5_text_edits.PAPER if path == PAPER else pr5_text_edits.SUPPLEMENT
        baseline = pr5_text_edits.apply_edits(baseline, changes)
        if path == PAPER:
            baseline = contamination_table(baseline)
        if path == SUPPLEMENT:
            baseline = positive_control_equation(baseline)
        approved = (wave_identity(edit(baseline)) if path == PAPER else
                    wave_identity(baseline, supplement=True))
        approved = post_wave_edit(approved, supplement=(path == SUPPLEMENT))
        approved = person_bootstrap_edit(approved, supplement=(path == SUPPLEMENT))
        approved = weekly_fit_edit(approved, tex=False, marked=False, paper=(path == PAPER))
        approved = positive_control_replay(approved)
        candidate = HERE / "revised" / path
        if options.initialize_approved_text:
            candidate.write_text(approved, encoding="utf-8")
        if candidate.read_text(encoding="utf-8") != approved:
            raise ValueError(f"Record the candidate edits in {path} before building.")
    for name in ("clean", "redline"):
        if options.version in ("both", name):
            build(name == "redline", HERE / f"KCOR-{name}.pdf")

