# Independent review of KCOR, candidate m169

I found no algebraic error in the core proportional gamma inversion. Given the post-skip mean-one gamma law, survival is $(1+\theta H_0)^{-1/\theta}$. With the Gompertz slope fixed, the manuscript's instantaneous quiet-period identity `h' = gamma*h - theta*h^2` follows by differentiation. These results identify a model-conditional descriptive contrast. The data alone cannot identify a treatment effect from it.

## 1. The limitations section understates an exact nonidentifiability (moderate, interpretation)

Section 5.2, paragraph beginning “In degenerate scenarios” (`documentation/preprint/paper.md:977`), describes a constant proportional intervention effect that cannot be distinguished from baseline differences as pathological, and suggests diagnostics can help identify such cases. The confounding of an individual hazard multiplier with the cohort baseline rate persists for *every* frailty variance and throughout the ordinary parameter space. It does not require absence of a depletion signature.

Consider a cohort with rate `k = 0.001` and an intervention multiplying each person's hazard by 1.5. A second explanation uses rate `k = 0.0015` with no intervention. For the same fixed slope and any fixed positive frailty variance, the conditional hazard in both explanations is exactly `z * 0.0015 * exp(gamma*t)`. All observed survival laws, weekly counts in distribution, quiet-window residuals, gap checks and normalized contrasts are identical. Compared with a cohort with rate 0.001, the unanchored reference-frailty contrast can capture the level difference of 1.5. The anchored contrast removes that constant ratio entirely. Neither comparison attributes the difference to its cause. The main methods already partly state this at `paper.md:346–350`.

**Fix.** Replace the pathological-case paragraph with the general rate–multiplier alias and distinguish the two estimands. Clarify that cohort-internal fit diagnostics cannot discriminate between these observationally identical explanations. This edit preserves the method and its stated noncausal scope.

## 2. A supplied inclusive total can silently replace the intended point population (moderate, ordinary workbook)

For the standard converter layout, `code/KCOR.py:4296` selects supplied all-ages rows whenever present. Ordinary processing validates their internal death-only history, but does not reconcile those counts with the disjoint components before using them (`code/KCOR.py:6344–6376`). The ordinary bootstrap *does* check that equality (`code/bootstrap_reporting.py:62–75`). A coherent history belonging to the wrong population therefore produces an apparently successful full-population point estimate.

I ran the ordinary workbook on 20 weeks of the candidate's two-age, two-dose death-only fixture from `test/test_round7_workbook.py`. I tripled dose-1 deaths from week index 8 onward in the 1950 group and reconstructed its opening risk sets exactly. The direct all-ages endpoint was **1.78036113**. Appending a supplied “all ages” history copied from just the 1940 group yielded **0.99994426**, with normalization and both fits marked OK. Passing that same input to the bootstrap history constructor raised “aggregate -2 does not match disjoint source counts.” This is a defect in identifying a malformed standard converter extract. It does not establish any error in the Czech source counts.

**Fix.** Share the aggregate-consistency check between the point and resampling paths for an input that declares a complete converter layout. Flag the affected supplied total if its weekly counts disagree, while retaining valid unrelated strata. An expressly partial display extract can legitimately have an independently supplied total whose population extends beyond the displayed ages, as in the SI S6 replay. Distinguish that layout instead of imposing component equality on it.

## 3. Sensitivity-mode anchoring diverges from the ordinary estimator (low, verified numerical discrepancy)

The sensitivity sweep at `code/KCOR.py:6214–6216` still requires the reference ratio to exceed `EPS = 1e-12`, divides it into the endpoint directly, and assigns status OK without checking representability of that quotient. The ordinary calculation at `code/KCOR.py:4355–4389` accepts any strictly positive finite reference and uses a checked ratio. Section 2.8 expressly specifies the latter convention.

I exercised the two normal entry points on the same closed-cohort workbook, Czech module-off configuration, 2021-24 enrollment, 2023 quiet window, four reference weeks and report date. To reach the edge case I used 160 synthetic weeks, 10^15 people per age/dose cell, one numerator death per reference week and 1% denominator mortality in those weeks. The ordinary reference was **7.432377e-14**, the unanchored endpoint **0.229272**, and the anchored endpoint **3.084780e12**. The sweep returned a missing value and “unavailable.” These extreme counts demonstrate the inconsistent rule; I have no evidence of an effect on the published Czech grid.

**Fix.** Use the same checked `_finite_hazard_ratio` to re-anchor the sweep, and assign OK only to a finite result. Add one sensitivity-versus-ordinary check with a positive reference below the old floor.

## 4. A redundant exposure column disables an otherwise identified bootstrap (low, ordinary workbook)

`code/bootstrap_reporting.py:25` rejects any input containing a column named `PT` before examining the supplied death and opening-risk histories. Ordinary fitting derives its exposure from `Alive` at `code/KCOR.py:6449–6452`. I passed the same standard two-age fixed-cohort fixture to the bootstrap constructor twice. It accepted the original; adding the verbatim duplicate `PT = Alive` caused it to raise “person-time histories require individual histories.” The extra column removes no person-history information.

**Fix.** Accept an extra PT column at least when it matches the declared opening counts and the death-only histories validate. Continue to distinguish genuinely person-time-only summaries, which cannot reconstruct complete person histories.

These checks used the candidate paper, supplement, analysis material and implementation. I used small synthetic workbooks in temporary directories and did not rerun the published bootstrap studies.
