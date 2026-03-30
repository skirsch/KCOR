---
name: fig5-empirical-plan
overview: Polish Fig. 5 in the alpha sandbox/manuscript pipeline and add a separate empirical KCOR intuition figure in Methods, using one coherent cohort comparison from existing analysis outputs where possible.
todos:
  - id: polish-fig5
    content: Update the existing synthetic manuscript plotter for Fig. 5 in `test/alpha/code/estimate_alpha.py` without changing its underlying synthetic data source.
    status: pending
  - id: sync-manuscript-text
    content: Apply the Fig. 5 caption/text updates in the preprint sources so the prose matches the polished synthetic figure.
    status: pending
  - id: design-empirical-figure
    content: Add a new Methods figure spec and manuscript insertion for `fig_kcor_empirical_intuition.png` using one coherent cohort comparison across all three panels.
    status: pending
  - id: wire-empirical-data
    content: Reuse or minimally extend existing KCOR outputs so the empirical intuition figure is reproducible from real pipeline data.
    status: pending
  - id: rebuild-and-qa
    content: Regenerate figures, rebuild the preprint PDF, and verify placement, numbering, caption accuracy, and figure-reference integrity.
    status: pending
isProject: false
---

# Fig 5 And Empirical Intuition Plan

## Scope

Keep the synthetic `alpha` recovery figure in `§3.4.1`; do **not** replace it with an empirical plot. The recommended empirical alternative in `documentation/preprint/fig5_and_intuition.md` is a **new Methods figure** that illustrates the KCOR workflow on a single real comparison: raw trajectories, post-normalization trajectories, and the resulting `KCOR(t)`.

## Numbering Note

- If the empirical intuition figure remains in Methods before `§3.4.1`, manuscript numbering will shift and the synthetic `alpha` recovery figure will no longer be Figure 5 in the compiled PDF.
- Treat the synthetic recovery figure as the fixed `§3.4.1` validation figure regardless of final figure number, unless the empirical figure is later moved to preserve the old numbering.

## Confirmed Targets

- Fig. 5 manuscript reference already exists in [paper.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md) and points to `figures/fig_alpha_synthetic_recovery.png`.
- The current manuscript figure generator already writes all three NPH figures from [estimate_alpha.py](C:\Users\stk\Documents\GitHub\KCOR\test\alpha\code\estimate_alpha.py):

```965:1018:C:\Users\stk\Documents\GitHub\KCOR\test\alpha\code\estimate_alpha.py
# existing manuscript Fig. 5 plotter
noise_models = list(df["noise_model"].dropna().unique())
fig, axes = plt.subplots(1, len(noise_models), figsize=(6 * len(noise_models), 4.5), sharey=True)
...
ax.set_title(str(noise_model))
ax.set_xlabel("True alpha")
ax.set_ylabel("Estimated alpha")
```

- The manuscript build path is already centralized in [estimate_alpha.py](C:\Users\stk\Documents\GitHub\KCOR\test\alpha\code\estimate_alpha.py):

```1224:1241:C:\Users\stk\Documents\GitHub\KCOR\test\alpha\code\estimate_alpha.py
synthetic_df = pd.read_csv(outdir / "alpha_synthetic_recovery.csv")
...
plot_manuscript_synthetic_recovery(synthetic_df, manuscript_dir / "fig_alpha_synthetic_recovery.png")
plot_manuscript_czech_objective(curves_df, best_df, manuscript_dir / "fig_alpha_czech_objective.png")
plot_manuscript_czech_diagnostics(loo_df, bootstrap_df, best_df, manuscript_dir / "fig_alpha_czech_diagnostics.png")
```

## Implementation Plan

1. **Polish Fig. 5 in the existing alpha manuscript plotter**

- Update [estimate_alpha.py](C:\Users\stk\Documents\GitHub\KCOR\test\alpha\code\estimate_alpha.py) `plot_manuscript_synthetic_recovery()` to match the brief in [fig5_and_intuition.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\fig5_and_intuition.md): fixed two-panel layout, explicit panel titles, shared axis ranges, smaller legend, and panel-level MAE annotations computed from plotted data.
- Keep the data source unchanged: `alpha_synthetic_recovery.csv` from the existing synthetic branch defined in [params_alpha.yaml](C:\Users\stk\Documents\GitHub\KCOR\test\alpha\params_alpha.yaml).
- Leave error bars as replicate SDs to stay aligned with the current manuscript text.

1. **Update the Fig. 5 manuscript wording only where needed**

- In [paper.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md) and likely [paper.tex](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.tex), replace the Fig. 5 caption with the stronger wording from [fig5_and_intuition.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\fig5_and_intuition.md).
- Add the optional one-sentence clarification in `§3.4.1` after the existing “illustrated in Figure 5” sentence if it still improves flow after the caption update.

1. **Add the empirical intuition figure as a new Methods figure**

- Insert a new figure reference in `§2.1` of [paper.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md), after the three strategy bullets and before `§2.4`, using the empirical label/filename recommended in [fig5_and_intuition.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\fig5_and_intuition.md): `#fig:kcor_empirical_intuition` and `figures/fig_kcor_empirical_intuition.png`.
- Default cohort-selection rule: choose a **clean, representative, null-style or negative-control comparison** if available; otherwise use a Czech comparison that visibly shows curvature reduction after normalization without looking cherry-picked or policy-driven.
- Log the selected cohort pair, strata, and time window used for the empirical figure to a small JSON or text artifact during figure generation so the choice is auditable.
- Use the **same cohort pair and preprocessing path in all three panels**:
  - Panel A: use cumulative hazard unless raw hazard is already smooth; do not apply heavy smoothing to hazard curves.
  - Panel B: depletion-neutralized cumulative hazard.
  - Panel C: `KCOR(t)` or anchored `KCOR(t; t0)` only if anchoring is needed to remove pre-existing level differences.
- Assert before plotting that all three panels use identical cohort identifiers and the same event-time range; fail loudly on mismatch.
- All three panels must share the same x-axis range.
- Prefer reusing existing KCOR outputs rather than hand-built arrays. The repo already exposes depletion-normalized quantities in [KCOR.py](C:\Users\stk\Documents\GitHub\KCOR\code\KCOR.py) and has an existing schematic plotting helper in [generate_kcor_workflow_figure.py](C:\Users\stk\Documents\GitHub\KCOR\code\generate_kcor_workflow_figure.py), so the new work should be a small reproducible plotting path rather than a one-off graphic.

1. **Choose the lightest reproducible code path for the empirical figure**

- First check whether current KCOR outputs already provide the needed raw and normalized cumulative-hazard series for one cohort pair.
- If yes, add a focused plotting utility near the existing manuscript/figure code and generate `fig_kcor_empirical_intuition.png` directly from those outputs.
- If not, add a thin extraction step that exports the needed observed cumulative hazard, depletion-neutralized cumulative hazard, and KCOR trajectory for one selected comparison, without rewriting the main pipeline.
- If anchored KCOR is used, include the anchor definition `t0` explicitly in the caption and make the reference level visually clear with a horizontal line at `1`.
- Use a schematic fallback only if every reasonable real comparison is too noisy or visually ambiguous.

1. **Rebuild and verify manuscript integration**

- Regenerate alpha figures through the existing sandbox entrypoint documented in [README.md](C:\Users\stk\Documents\GitHub\KCOR\test\alpha\README.md).
- Rebuild the preprint PDF and verify:
  - The synthetic recovery figure still appears in `§3.4.1` as the NPH validation figure, even if its compiled figure number changes.
  - The new empirical intuition figure appears in Methods, not Results.
  - Caption numbering and `@fig:` references resolve cleanly.
  - The empirical figure caption accurately states whether Panel C is raw `KCOR(t)` or anchored `KCOR(t; t0)`.

## Risks To Watch

- `generate_manuscript_figures()` currently requires the alpha CSV set, so partial alpha runs can break manuscript figure regeneration.
- An empirical intuition figure can easily look cherry-picked; the cohort-pair selection rule needs to be conservative and documented.
- If real hazards are too noisy, cumulative hazards may be the safer Panel A choice.
- There is an older numbering/storyline conflict in alternate docs, so implementation should follow [paper.md](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.md) / [paper.tex](C:\Users\stk\Documents\GitHub\KCOR\documentation\preprint\paper.tex), not stale drafts.

