# Round 15 convergence - m173

Two independent blind GPT-6.2 Astra High reviews and one independent targeted Astra High resolution review. Revised question (verbatim): "Examine KCOR closely. What, if anything, is wrong that matters to the validity of the method or its results, and how would you fix it? Distinguish substantive flaws from minor corrections, and do not propose unnecessary redesigns."

## Scientific convergence

Both blind reviewers confirmed the central proportional gamma inversion and identification equations under their stated working conditions. No new fundamental algebraic flaw was identified. The published Table 8 and S6 results were independently reproduced; the targeted review found all 1,192 S6 point rows and all 37 common fields identical between m169 and m173, and verified byte-identical main archives. Full 999-draw reruns were not performed.

Nine round-14 objections independently resolved: 14.3 and 14.5-14.12. Partial: 14.1/14.2 (corrected code, obsolete distributed manufacturer workbooks) and 14.4 (original single-total reproduction resolved, broader aggregate coverage/nested consistency not fully resolved).

## Consolidated findings

- **[15.1](../../../objections/15.1/README.md) - Historical manufacturer workbooks still distribute superseded outputs** (High for ancillary distributed sheets; no indication of primary paper impact; R). Corrected producer and converter pass fresh tests, but released KCOR.xlsx manufacturer sheets still carry pre-enrollment unadjusted ratios under KCOR labels, and historical input manufacturer partition fails reconciliation.
- **[15.2](../../../objections/15.2/README.md) - Partial extract silently constructs an all-ages result from displayed strata** (High for affected ordinary partial workbook output; A, B). Both blind reviewers reproduced wrong full-population point and three-draw interval when a partial extract contains an older displayed stratum plus a supplied wider 1960+ total, but no all-ages total. A separate exhaustive case loses a valid pair when only one dose has an explicit total.
- **[15.3](../../../objections/15.3/README.md) - Nested inclusive totals can describe impossible populations** (Moderate to high for affected malformed inputs; A, B, R). All reviewers verified that all-ages and born-1960+ totals can both report OK even when the latter contains more people than the former. Also accepted with aggregate-only default inputs.
- **[15.4](../../../objections/15.4/README.md) - Worked S6 example omits the declared primary unanchored contrast** (Moderate; material interpretation/reporting, not a wrong anchored estimate; B). S6 labels its anchored values correctly but does not present the defined primary unanchored contrast. Six displayed age-specific unanchored point estimates are below unity while six anchored points are above; all-age point is above in both. These are different estimands, not contradictory estimates. No unanchored intervals were computed in this review.
- **[15.5](../../../objections/15.5/README.md) - Illustrative S6/S5.6 figure replays do not assess signed gap assumptions** (Minor to moderate diagnostic reporting; B). Figure replay reports numeric fit and residual flags but does not return the signed gap screen. The curves use unmarked gaps; no evidence of numerical change.
- **[15.6](../../../objections/15.6/README.md) - Negative-control model-consistency wording overstates what is demonstrated** (Minor scientific wording; A). Section 3.1.2 uses wording suggesting the empirical control evaluates behavior under model-consistent conditions. The fixed-cohort audit and diagnostic flags do not establish model consistency; other passages already label it a descriptive audit.
- **[15.7](../../../objections/15.7/README.md) - NPH sequential shorthand could imply inconsistent survivor fitting** (Minor wording; A). Box 2 and section 2.7.1 shorthand can suggest proportional gamma fitting followed by correction even when nonlinear exponent is supplied. Operative sections specify consistent survivor law.
- **[15.8](../../../objections/15.8/README.md) - Figure reference timing described only as four weeks** (Minor caption clarification; B). S5.6/S6.1.3 captions say at four weeks without specifying accumulating weeks following skip.
- **[15.9](../../../objections/15.9/README.md) - Within-bin hazard wording unnecessarily requires a constant hazard** (Minor mathematical wording; B). Negative log of conditional interval survival is integrated hazard even with a varying within-bin hazard; constant hazard is needed only for constant instantaneous interpretation. Computation correct.
- **[15.10](../../../objections/15.10/README.md) - Seed diagnostic bound wording is too broad** (Minor diagnostic wording; B). Description says numerical bound; code rejects frailty ceiling and both rate bounds but permits zero frailty, a structural boundary.


## Scope and recommended stopping criterion

Three specific software/delivery issues (15.1-15.3) and one interpretive reporting question (15.4) merit discussion. Six remaining changes are diagnostic/caption/mathematical wording clarifications, not reasons to redesign KCOR. The Gompertz-slope and retrospective eligibility sensitivity results are retained separately as [scientific context](scientific-context/README.md), not counted as new defects; the reviewers did not establish a mathematical error or contradiction in the scoped claims. The recurring population issues share a root cause: explicitly identifying which people are represented by each supplied or reconstructed aggregate.

The evidence supports convergence of the scientific method, conditional on its stated assumptions, rather than zero open code/reporting issues. Resolve the specific material issues and minor true errors with minimal edits. Do not automatically launch another evaluation or implement any finding without the user's direction.

[Blind A](blind-a.md) | [Blind B](blind-b.md) | [Targeted resolution](resolution.md) | [Frozen inputs](inputs.json).

## Author-side triage after review

The user dismissed 15.6-15.10 as editorial churn; no edits adopted. Keep 15.1-15.5 active for individual treatment. Their original review reports and numbered records remain intact for lineage, but the dismissed issues must not be treated as unresolved requirements in the next targeted review.

15.1 addressed in m174: all six obsolete active manufacturer tabs replaced with unavailable notices; original workbooks preserved and unrelated parts byte-identical. [Evidence](manufacturer-archive/README.md). Remaining active: 15.2-15.5.

15.2 addressed in m175: available-age label across output paths, complete dose-specific reconstruction, and matched bootstrap population. Actual S6 1,192-row parity verified. [Evidence](available-ages/README.md). Remaining active: 15.3-15.5.

15.3 addressed in m176: nested totals and complement consistency validated; impossible aggregates withheld; valid age intervals unchanged. Actual Czech 2021_24 source passes all checks. [Evidence](nested-totals/README.md). Remaining active: 15.4 and 15.5.

## Final user-approved round-15 disposition (m177)

15.1-15.3 implemented in m174-m176. The original intent of 15.4 was confirmed by the user: S6 illustrates anchored KCOR, so the request to present an additional unanchored estimate is dismissed; only the explicit purpose sentence added. 15.5 addressed by one identical signed-gap unassessed sentence in each of the four relevant illustrative captions. 15.6-15.10 dismissed, no edits. No active round-15 objections remain; independent verification of the implemented fixes remains pending. No further evaluation launched.
