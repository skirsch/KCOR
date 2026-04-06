---
name: Populate §3.3.2 results
overview: Flesh out [documentation/preprint/paper.md](documentation/preprint/paper.md) §3.3.2 with experiment results prose, two figure includes with new IDs, a summary table, and copy generated PNGs into [documentation/preprint/figures/](documentation/preprint/figures/) for Pandoc builds—without editing other manuscript sections unless you explicitly extend scope to `main.tex`.
todos:
  - id: run-quiet-sim
    content: Regenerate out/ PNGs and agg CSV if stale (make quiet_sim).
    status: completed
  - id: copy-figures
    content: Copy both fig_quiet_contam_*.png to documentation/preprint/figures/.
    status: completed
  - id: replace-332
    content: "Swap paper.md §3.3.2 placeholder for full prose + 2 figures + table. Use Option A: leave the generic SI line unchanged (current line ~836: \"Additional validation results—including full simulation grids...\")."
    status: completed
  - id: verify-numbers
    content: Cross-check table/narrative vs quiet_contamination_agg.csv for selected cells; do not skip before commit (seeds/CMR rerun can shift cells).
    status: completed
isProject: false
---

# Populate §3.3.2 with quiet-window contamination results

**Execution order:** Run the todos in sequence: `run-quiet-sim` → `copy-figures` → `replace-332` → `verify-numbers`. Do not skip `verify-numbers` before committing.

## Scope (per your instructions)

- **In scope:** [documentation/preprint/paper.md](documentation/preprint/paper.md) only for narrative/table/figures; copy two PNGs into the preprint `figures/` directory.
- **Out of scope unless you add it:** [documentation/preprint/main.tex](documentation/preprint/main.tex) (still has a shorter §3.3.2 paragraph with `\texttt{...}` paths—no sync in your spec), supplement files, [test/quiet_window_contamination/run_contamination_test.py](test/quiet_window_contamination/run_contamination_test.py).

## 1. Regenerate figures if needed

Run `make quiet_sim` (or `python test/quiet_window_contamination/run_contamination_test.py`) so [test/quiet_window_contamination/out/](test/quiet_window_contamination/out/) contains current `fig_quiet_contam_kcor_asymptote.png` and `fig_quiet_contam_theta_error.png` matching the prose (combined panel = `frac_reps_kcor_gt_1` heatmap).

## 2. Copy figures into preprint tree

Copy (binary-safe copy, preserve overwrite):

- `test/quiet_window_contamination/out/fig_quiet_contam_kcor_asymptote.png` → `documentation/preprint/figures/fig_quiet_contam_kcor_asymptote.png`
- `test/quiet_window_contamination/out/fig_quiet_contam_theta_error.png` → `documentation/preprint/figures/fig_quiet_contam_theta_error.png`

These paths match the `![](figures/...)` convention already used elsewhere in `paper.md` (e.g. §3.4.1).

## 3. Replace §3.3.2 body in `paper.md`

**Anchor:** Heading at line 832:

`#### 3.3.2 Quiet-window contamination stress test`

**Replace** the current single placeholder paragraph (line 834) with your supplied markdown: **Design** / **Results** (Types 1–3, Šmíd citation) / **Interpretation**, plus:

- Two `![...](figures/...){#fig:...}` blocks with IDs `fig:quiet_contam_kcor` and `fig:quiet_contam_theta`.
- Pandoc-style table with `{#tbl:quiet_contam}` on the caption line (as in your draft).

**SI transition (mandatory):** Keep the paragraph beginning `Additional validation results—including full simulation grids...` (currently ~line 836) **unchanged**. It is the generic SI pointer for all of §3.3, not only §3.3.2. Replace **only** the `#### 3.3.2` heading plus the placeholder paragraph that follows it (current ~line 834), then the SI line stays immediately after the new subsection + figures + table.

**Cross-refs:** Ensure `@fig:quiet_contam_kcor`, `@fig:quiet_contam_theta`, and `@tbl:quiet_contam` are used consistently in the new prose (as in your draft).

## 4. Numeric consistency (before committing)

Spot-check the table and narrative against [test/quiet_window_contamination/out/quiet_contamination_agg.csv](test/quiet_window_contamination/out/quiet_contamination_agg.csv) (`dynamic_agg`, `comorbidity_agg`, `combined_agg` rows): mean `KCOR_asymptote_mean`, `frac_reps_kcor_gt_1`, and implied $\epsilon/h_{\mathrm{ref}}$ with $h_{\mathrm{ref}}=0.000456$. If a future run changes seeds or CMR calibration, update the prose/table numbers or add “representative run” wording.

## 5. Optional follow-up (not required by your spec)

- Mirror the same content in `main.tex` §3.3.2 for PDF parity.
- Add `fig:quiet_contam_*` / `tbl:quiet_contam` to any Pandoc crossref metadata if the build complains (depends on your existing `paper.md` header/filter setup).
