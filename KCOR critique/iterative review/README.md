# KCOR revision workspace

**Current evaluated edition: [m185](candidates/m185/README.md).** The previously packaged `revised/` folder and earlier checkpoints remain historical editions. For m185 use the candidate materializer, current PDF pair and [round 21 status](../CURRENT-STATE.md); the older build instructions below apply to the earlier editions.

Use the [x.y objection index](../README.md) to find the evidence, checks and decisions for any individual objection.

| Material | Location |
|---|---|
| Preserved pre-revision method and paper | [Baseline files](baseline/) and [source manifest](baseline/manifest.json) |
| Iteration 1 checkpoint, including the N1 manuscript correction | [Clean paper](checkpoints/iteration-1/KCOR-clean.pdf), [redline](checkpoints/iteration-1/KCOR-redline.pdf), [method and manuscript sources](checkpoints/iteration-1/revised/), [Table 8](checkpoints/iteration-1/rounds/round-06-weekly-inversion/table8_bootstrap.json) |
| Current method and manuscript sources | [Revised files](revised/) |
| Iteration 2 checkpoint, immediately before the 3.1 correction | [Method and manuscript sources](checkpoints/iteration-2/revised/) · [clean PDF](checkpoints/iteration-2/KCOR-clean.pdf) · [redline](checkpoints/iteration-2/KCOR-redline.pdf) · [Table 8](checkpoints/iteration-2/table8_bootstrap.json) |
| Objections still open against the corrected edition | [Remaining PR #5 items, evaluation findings and deferred decisions](open-objections.md) |
| Approved changes, deferred questions and approval status | [Decision log](decisions.md) |
| Whether earlier revisions preserve the documented method and experiment designs | [Original intent audit](original-intent-audit.md) |
| Modification records, numerical checks and earlier targeted reviews | [Revision records by objection](../revision-records.md) |
| Original question, three evaluation paths and convergence comparison | [LLM evaluation procedure](evaluations/README.md) |
| Author's proposed analytic companion to the Cox experiment | [Author proposal](author-proposals/objection-5-cox-gamma-derivation.md) |

The [clean manuscript](KCOR-clean.pdf) and [cumulative redline](KCOR-redline.pdf) include the supplement. The redline compares the current edition with the combined manuscript reviewed in PR #5. Select a blue bracketed label beside an edit to jump to its objection in the same PDF. The notes identify the PR #5 source or the later evaluation finding. Table 8 cites the objections for its revised values in the caption and column heading. Run `python build_pdfs.py` in this directory to rebuild both PDFs.

To regenerate the revised KCOR figure from the Cox experiment, run `make -C test/sim_grid revised-cox-kcor-figure` from the repository root. The command recomputes the person-bootstrap intervals, checks the seven estimates and both endpoints against the [archived values](../objections/2.1/figure_intervals.csv), and writes the figure used by both PDFs. On Windows with the Python 3.11 launcher, append `"PYTHON=py -3.11"`.

The [deferred discussion list](decisions.md#deferred-discussion-after-the-objection-review) includes Supplementary Figure S5 and its window-level data.

The [critique index](../README.md) links to the original paper, independent reviews and PR #5 evidence.

Iteration 3 starts with [the quiet-gap correction and Table 8 replay](../objections/3.1/README.md). The [series 3 evaluation](evaluations/3/README.md) reviewed iteration 2. The saved editions can be selected directly. To regenerate either checkpoint's PDFs, run its `rebuild.py OUTPUT_DIRECTORY` from the corresponding checkpoint folder. The [iteration 1 manifest](checkpoints/iteration-1/checkpoint.json) and [iteration 2 manifest](checkpoints/iteration-2/checkpoint.json) record the saved files and hashes.
