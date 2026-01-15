Here’s a **Cursor punchlist** to address both reviewer comments with minimal disruption: (1) add a single “KCOR bias vs θ” synthesis plot, and (2) explicitly document the bootstrap procedure for aggregated cohort counts.

---

## Cursor punchlist: Validation clarity + bootstrap clarification

### Part A — Add “KCOR bias vs θ” summary plot (SI-first, low risk)

#### A1) Create a new SI figure from existing simulation outputs

1. **Find the existing simulation grid outputs** you already use for Table 6 / Table 11 (whatever file holds per-θ bias/coverage metrics).

   * Likely under something like:
     `test/sim_grid/out/` or `test/sensitivity/out/` or wherever Table 6/11 are generated from.
2. **Add a small script** (or extend the existing table-generation script) to output one figure:

   * **File name:** `figures/fig_si_kcor_bias_vs_theta.png`
   * **Plot content:**

     * x-axis: `theta` (frailty variance)
     * y-axis: `bias` (deviation from true cumulative hazard ratio)
     * recommended: plot **mean bias** with **error bars** (±SD across simulations or ±MC SE)
     * optional (nice): overlay Cox bias (same y-axis) if you already compute it; otherwise skip.

**Definition (use the same as Table 6/11):**

* If you already define bias as deviation from the *true cumulative hazard ratio* at a fixed horizon, reuse it exactly.
* If not explicitly defined, standardize to something like:

  * `bias(theta) = mean( KCOR_hat(t*) ) - 1` under null (true ratio = 1)
  * pick `t*` equal to whatever horizon you report in Table 6/11 (end of follow-up or end of eval window).

3. **Update the build rules** (Makefile or whatever drives figure generation) so the new figure is produced whenever the underlying results file changes.

#### A2) Add figure callout in the SI near Table 6 / Table 11 discussion

4. **Edit:** `supplement.md`
5. **Insert** a short paragraph in the **simulation validation results area** (where you discuss the null simulations / Table 6 and/or Table 11), immediately after the paragraph that says bias/coverage are summarized.

Add:

```markdown
Figure @fig:si_kcor_bias_vs_theta provides a compact summary of KCOR bias as a function of frailty variance $\theta$ under the simulated null, complementing the tabulated bias and coverage results.
```

6. **Add the figure block** right after that paragraph:

```markdown
![Simulated-null summary: KCOR bias as a function of frailty variance $\theta$. Bias is computed relative to the known null cumulative hazard ratio (=1) using the same definition as in Table @tbl:<YOUR_TABLE_LABEL>. Error bars indicate Monte Carlo variability across simulation replicates.](figures/fig_si_kcor_bias_vs_theta.png){#fig:si_kcor_bias_vs_theta}
```

* Replace `@tbl:<YOUR_TABLE_LABEL>` with the label you use for the table that defines bias/coverage (e.g., `@tbl:sim_null_summary`).

#### A3) Add a one-sentence pointer in the main paper (optional but helpful)

7. **Edit:** `paper.md`
8. In the **Validation** section where you mention simulated nulls / Table 6 or overall operating characteristics, add one sentence:

```markdown
A compact summary of simulated-null bias as a function of frailty variance $\theta$ is provided in the Supplementary Information (Figure @fig:si_kcor_bias_vs_theta).
```

(Only do this if your paper already references SI figures; otherwise keep it SI-only.)

---

### Part B — Clarify bootstrap procedure (aggregated counts vs individuals)

#### B1) Add explicit bootstrap language in the Methods (main paper)

9. **Edit:** `paper.md`
10. Find the paragraph where you mention uncertainty bands / bootstrap CIs (often near your hazard estimation or KCOR computation steps).

Add this text (drop-in):

```markdown
Because KCOR is computed from cohort-aggregated event counts rather than individual-level records, bootstrap uncertainty is obtained at the aggregated level (resampling cohort–time cells / event counts), not by resampling individuals. This preserves the cohort aggregation structure and exposure denominators while providing uncertainty intervals for cumulative-hazard–based quantities.
```

If you use a specific mechanism (e.g., Poisson):

* Replace the parenthetical with the exact method:

Option (Poisson count bootstrap):

```markdown
...bootstrap uncertainty is obtained by resampling event counts under a Poisson approximation at the cohort–time level, not by resampling individuals...
```

Option (multinomial / block bootstrap):

```markdown
...bootstrap uncertainty is obtained by resampling cohort–time cells with replacement, not by resampling individuals...
```

Pick the one that matches what your code actually does.

#### B2) Mirror the same clarification in SI (so it’s impossible to miss)

11. **Edit:** `supplement.md`
12. Under **S5.1 Fit diagnostics** (or wherever you first mention “95% bootstrap intervals”), add 1–2 sentences:

```markdown
Uncertainty bands (95% bootstrap intervals) are computed using an aggregated-data bootstrap at the cohort–time level, since individual-level resampling is not available/used in KCOR_CMR inputs. The resampling unit is the cohort–time count process rather than individuals.
```

#### B3) Make the resampling unit explicit in figure captions (quick fix)

13. **Edit:** `supplement.md` (and `paper.md` if needed)
14. For any caption that currently says:

* “Uncertainty bands (95% bootstrap intervals) are shown.”

Optionally change to:

```markdown
Uncertainty bands (95% bootstrap intervals; aggregated cohort–time resampling) are shown.
```

Do this only for 1–2 representative figure captions (no need to edit every caption if you add B1/B2).

---

## Verification checklist (do after edits)

1. Search for `bootstrap` in both `paper.md` and `supplement.md`:

   * confirm the wording explicitly distinguishes **aggregated resampling vs individual resampling**.
2. Build the SI and confirm:

   * `Figure S?` for `@fig:si_kcor_bias_vs_theta` appears
   * it’s referenced in the SI text at least once
3. Confirm the “bias vs θ” figure uses **the same bias definition** as Table 6/11 (no new metric).
4. Confirm the new figure file is tracked properly (LFS if needed) and included in your repo build.

---


