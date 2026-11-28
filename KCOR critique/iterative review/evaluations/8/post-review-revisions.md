# Round 8 post-review corrections

m135 (evaluated frozen edition) preserved. m136 includes the straightforward batch only.

- **8.2**: Corrected per-date pooled export and removed nearest-date substitution. Undefined date rows remain explicitly unavailable. Actual exported Table 8 rows for both enrollments, both modes and all 3 pairs reproduce the archived points within 2.22e-16. 174 and 140 dates per pair are now exported, respectively. Archived intervals unchanged; no full resampling rerun needed because fit objective and dedicated Table 8 calculation are unchanged. The missing-source-component policy is tracked separately as 8.1. [Verification](workbook-verification.json).
- **8.9**: Corrected remaining diagnostic policy wording in sections 2.1.2 and 4.4, and the reversed SI S2 I1/I7 condition/consequence. No new diagnostic threshold or estimator change.
- **8.10**: Qualified section 2.13 reproduction scope. Default Table 8 refit now starts fresh in a separate directory and preserves archive. Explicit resume verifies matching input/config/source/settings provenance. A one-draw smoke test scheduled four fits on fresh run, zero on matching explicit resume, and rejected incompatible draw count. Existing 999-draw archives unmodified. [Runner patch](table8_replay.patch).
- **8.11**: Corrected section 2.4.4 to name early post-skip anchors and marked-wave exclusion, consistent with section 2.5.
- **8.12**: Qualified section 1.5 bias benchmark and removed blanket prerequisite language while preserving depletion-adjusted target.
- **8.13**: Corrected SI S6.1.2: zero frailty follows Gompertz baseline, not necessarily linear cumulative hazard; weak identification assessed separately.

Remaining for individual discussion: 8.1, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8. No other modifications or review round.

- **8.1 / m137:** validation before aggregation, matching bootstrap history checks, legitimate exhaustion preserved. Four Czech inputs pass, all 12 Table 8 actual points match archived estimates and intervals unchanged. Existing whole-sheet bootstrap unavailability on malformed input retained. Remaining: 8.3-8.8.

- **8.3 + 8.8 / m138:** corrected sensitivity analysis to reuse ordinary paths and regenerated all 1,323 current grid cells; common color scale and actual sensitivity reported. Both module-off/forced-alpha matched ordinary tests passed. Historical grid and prior candidate preserved. Remaining: 8.4-8.7.

- **8.4 / investigation only (m138 unchanged):** verified March 25, 2026 safeguard intent, exact-model and matched S6 policy-only comparison; uncertainty/decision notes archived. Lead-author consideration required before changing unresolved-fit handling.

- **8.4 / expanded ceilings trial (m138 unchanged):** 200 and 500 matched refits of all 999 S6 draws and 600 original low-variance synthetic datasets. Youngest bound hits 287/999 -> 7/999 -> 0/999, KCOR intervals recorded. No production cap change.

- **8.4 / m139-review:** approved ceiling 500, explicit near-ceiling status and retained constrained estimates in both fitting paths. Full S6 and primary Table 8 replay complete; remaining long empirical replays running in m139. m138 preserved. Next discussion 8.5.

- **8.5 / m140:** qualified external-scale claim in paper and SI, verified exact local identification counterexample, no algorithm/alpha result change. Integrated completed cap500 empirical refreshes from m139; only forced-alpha Table 8 remains pending. Remaining discussion: 8.6, 8.7.

- **8.6 / m141:** clarify post-skip within-cohort mean reference, equal-reference null for unanchored and proportional-reference null for anchored KCOR. Exact example verified, no method change. Remaining objection: 8.7.

- **8.7 / m142:** complete five-structure replicated known-null study, publish results in SI S9, correct historical evidence description. All 500 datasets completed, coverage 92--99%, no numerical failures. No estimator change. Full cap500 forced-alpha Table 8 replay integrated; all six displayed values and intervals unchanged. All 13 round-8 items addressed in candidate, independent closure not yet evaluated.
