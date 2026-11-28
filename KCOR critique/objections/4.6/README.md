# 4.6 — Ordinary output versus refitted uncertainty

**High for inference — continuing objection with new verified example**

Table 9 fitted benefit point 0.833593: ordinary 95% interval [0.814370,0.853270], refitted bootstrap [0.774117,0.897055]. Harm point 1.161701: [1.137278,1.186649] versus [1.086518,1.248454]. The ordinary workbook also has a fallback/reporting discrepancy when fits are insufficient. No claim all ordinary intervals are invalid for their narrower conditional quantity.

**Root cause / related objections:** Related 1.3, 1.4 and 3.3. Clarify or align production intervals and flagged-output policy.

**Source review:** [repeat-1/evaluated-handoff.md](<../../iterative review/evaluations/4/repeat-1/evaluated-handoff.md>) · [repeat-3/original-same.md](<../../iterative review/evaluations/4/repeat-3/original-same.md>)

[Series 4 comparison](<../../iterative review/evaluations/4/convergence.md>) · [Independent verification](<../../iterative review/evaluations/4/verification.md>) · [Counterexamples](<../../iterative review/evaluations/4/counterexamples.json>)

**Corrected in candidate m99.** Ordinary workbook reporting now runs the person-bootstrap procedure by default with 999 draws, using the ordinary fitting and pooling pipeline for each draw. The preceding analytical intervals are retained separately as conditional count-based intervals. Fit flags are carried through the resampling and reporting, rather than discarded. Unknown overlap or incomplete valid draws yield unavailable bootstrap intervals, not mislabeled analytical intervals. Standard mutually exclusive fixed enrollment cohorts use the disjoint layout; the known age-shift/Table 8 construction uses the overlapping layout explicitly (wired into the negative-control Makefile). Aggregations that do not determine the person histories need the appropriate original histories or a supported reconstruction.

**End-to-end verification.** Both original positive-control scenarios were run with 99 draws through the ordinary workbook, and the full set of point/interval calculations at the original reporting date was independently reproduced from the same actual resampled histories to within 4.5e-16.

| Scenario | KCOR | Analytical 95% interval | Ordinary person bootstrap (99 draws) |
|---|---:|---|---|
| Benefit | 0.833593 | [0.814370,0.853270] | [0.783882,0.891125] |
| Harm | 1.161701 | [1.137278,1.186649] | [1.089553,1.238733] |

Both had zero flagged draws. The original prior 999-draw results are kept separately, not asserted to be numerically identical to the new 99-draw run. The known Table 8 overlapping-control ordinary-workbook path also completed a three-draw smoke check: 3093 post-anchor rows received intervals, no flagged draws, 120 early/unavailable rows. The three draws are not an inferential interval validation. Disjoint and overlapping history reconstructions passed risk-set and cohort-count tests. Deliberately using disjoint layout for the overlapping control leaves the bootstrap interval unavailable while preserving the separately labeled analytical interval. Full 999-draw end-to-end revalidation remains pending; the default is 999.

[Reporting implementation](bootstrap_reporting.py) · [Integration patch](KCOR.py.patch) · [Independent comparison](check.py) · [Results](verification.json) · [Benefit workbook](benefit-99.xlsx) · [Harm workbook](harm-99.xlsx) · [Paper](KCOR-clean.pdf). Use environment KCOR_BOOTSTRAP_DRAWS to specify draw count, KCOR_BOOTSTRAP_WORKERS for parallelism, KCOR_BOOTSTRAP_LAYOUT=table8 for the known overlapping control, and KCOR_BOOTSTRAP_SEED to reproduce draws. These are ordinary-mode settings; MC and sensitivity-analysis paths are not recursively bootstrapped. This record concerns the hypothetical integrated edition, not a claim that the actual draft PR has been merged.
