---
name: fix-reviewer-findings
overview: Fix the two reviewer-facing issues by synchronizing Figure 1 with the current v7.5 Methods description and correcting the manuscript’s reproducibility command to use an actual safe Make target.
todos:
  - id: fix-figure1-sync
    content: Align Figure 1’s caption, generator, and rendered asset so the workflow topology, step count, notation, and optional-NPH placement all match `paper.md` §2.5–§2.7.
    status: completed
  - id: fix-build-command
    content: Replace the broken `make paper-full` instruction in `paper.md` with a real Make target that readers can run safely without private deployment side effects.
    status: completed
  - id: run-targeted-qa
    content: Verify that Figure 1 now matches the Methods step-by-step, that the caption wording is consistent with the generated graphic, and that the reproducibility section names an existing Make target.
    status: completed
isProject: false
---

# Fix Two Reviewer-Facing Problems

## Scope

Address only the two concrete issues surfaced in review:

- Figure 1 is not fully synchronized with the current v7.5 Methods narrative.
- The reproducibility section points readers to a non-existent build command.

Primary files:

- [documentation/preprint/paper.md](documentation/preprint/paper.md)
- [code/generate_kcor_workflow_figure.py](code/generate_kcor_workflow_figure.py)
- [documentation/preprint/figures/fig_kcor_workflow.png](documentation/preprint/figures/fig_kcor_workflow.png)
- [Makefile](Makefile)

## Problem 1: Figure 1 Synchronization

Current mismatch:

- In [documentation/preprint/paper.md](documentation/preprint/paper.md), the Figure 1 caption still presents the workflow as a “structured four-step estimator,” while the generator in [code/generate_kcor_workflow_figure.py](code/generate_kcor_workflow_figure.py) renders six numbered boxes.
- The generator also uses script-style labels such as `H0_eff`, `delta_i`, `Delta_d(t)`, and `tilde H0_d(t)` rather than fully manuscript-aligned notation/phrasing.
- The optional NPH box is visually attached to the core estimator column, but the Methods in `§2.7.1` define it as preprocessing before accumulation/inversion rather than as an internal step of the universal core.

Essential current snippets:

```text
paper.md:
(B) ... reconstructed over the full trajectory using all weeks ...
(C) ... optional NPH extension may adjust wave-period hazards before accumulation and inversion ...
```

```text
generate_kcor_workflow_figure.py:
1. Seed fit in nearest quiet window
2. Reconstruct effective cumulative hazard
3. Compute persistent offsets
4. Pooled quiet-window refit
5. Gamma-frailty inversion
6. KCOR(t) comparison
```

Planned fix:

- Update the Figure 1 caption in [documentation/preprint/paper.md](documentation/preprint/paper.md) so the title and panel description no longer conflict with the rendered step structure.
- Decide on one consistent framing:
  - either reserve “four-step” strictly for `§2.5` and explicitly describe steps 5–6 as downstream normalization/comparison,
  - or remove “four-step” from the figure title and describe the whole pipeline more neutrally.
- Revise [code/generate_kcor_workflow_figure.py](code/generate_kcor_workflow_figure.py) so the rendered figure:
  - matches `§2.5` step-by-step,
  - uses manuscript-consistent notation and wording,
  - makes the iteration loop readable,
  - visually separates the optional NPH module from the core KCOR pipeline,
  - shows the optional module affecting preprocessing in a way consistent with `§2.7.1`.
- Regenerate [documentation/preprint/figures/fig_kcor_workflow.png](documentation/preprint/figures/fig_kcor_workflow.png) from the script in the same pass.

## Problem 2: Broken Reproducibility Command

Current mismatch:

- [documentation/preprint/paper.md](documentation/preprint/paper.md) tells readers to run `make paper-full`.
- [Makefile](Makefile) does not define `paper-full`.
- The available paper targets include `paper`, `paper-pdf`, `paper-tex`, and `paper-all`.
- `paper` is not the safest public-facing recommendation because it includes an `scp` deployment side effect, so the manuscript should point to a build-only target instead.

Relevant current snippet:

```text
paper.md:
... using the root Makefile paper target `make paper-full`.
```

Planned fix:

- Update [documentation/preprint/paper.md](documentation/preprint/paper.md) to reference a real target.
- Prefer `make paper-pdf` as the default reader-safe build command because it builds the manuscript PDF without the deployment side effect in `make paper`.
- Optionally mention `make paper-all` only if the text benefits from clarifying how to build combined plus split outputs; otherwise keep the wording minimal.
- Leave [Makefile](Makefile) unchanged unless a follow-up request specifically asks to add or rename targets.

## Execution Order

1. Update the Figure 1 caption wording in [documentation/preprint/paper.md](documentation/preprint/paper.md) so the intended workflow framing is locked.
2. Update [code/generate_kcor_workflow_figure.py](code/generate_kcor_workflow_figure.py) to match that framing and the current Methods topology.
3. Regenerate [documentation/preprint/figures/fig_kcor_workflow.png](documentation/preprint/figures/fig_kcor_workflow.png).
4. Fix the reproducibility command in [documentation/preprint/paper.md](documentation/preprint/paper.md) from `make paper-full` to the chosen real target.
5. Run a targeted QA pass:

- confirm Figure 1 matches `§2.5`/`§2.7.1` step-by-step
- confirm the caption and image use the same framing
- confirm the documented Make command exists and is side-effect-safe

## Success Criteria

- Figure 1, its caption, and its generator all describe the same v7.5 workflow without step-count ambiguity.
- The optional NPH extension is visibly outside the universal core and attached to the workflow in a way that matches `§2.7.1`.
- The reproducibility section in [documentation/preprint/paper.md](documentation/preprint/paper.md) names an actual Make target that a reader can run successfully.
- No new wording drift is introduced between the updated figure, caption, and Methods text.

