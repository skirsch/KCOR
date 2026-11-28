# 5.4 - Round 5 finding

**Severity:** High for ordinary registry inference

Standard converter emits sex O, DCCI and overlapping aggregates that default bootstrap rejects. A single unsupported sheet can make intervals unavailable.

**Related issues:** 4.6, production integration

**Review evidence:** B - [Series 5 convergence](../../iterative%20review/evaluations/5/convergence.md) ([A](../../iterative%20review/evaluations/5/blind-a.md), [B](../../iterative%20review/evaluations/5/blind-b.md), [R](../../iterative%20review/evaluations/5/resolution.md)).

**Status:** corrected in m106; m105 preserved.

The ordinary disjoint bootstrap now preserves Sex F/M/O and DCCI, including birth year -1 (unknown). It reconstructs death/exit/survival cells within each age/dose comparison group, retaining the existing fixed age/dose totals and joint sampling of subcells. Where Dead_COVID is present it separates COVID and other death categories so the sampled all-cause totals remain coherent. Converter aggregates -2 (all ages, including unknown age) and -60 (born 1960 or later) are validated against the original disjoint counts and rebuilt from the same draws, never independently resampled. Ambiguous or inconsistent aggregate layouts remain unavailable. Unsupported sheets no longer disable supported enrollments; seed indices remain tied to the original sheet order.

Verification:
- [History regression](history-verification.json): prior code rejected standard layout; 216,000 synthetic people sampled once, exact aggregate identity in ten draws, old simple F/M and special Table 8 draw bit-for-bit unchanged. Mismatched aggregate correctly rejected.
- [Synthetic ordinary workbook regression](workbook-verification.json): two enrollments, one inconsistent aggregate; valid enrollment received 377 intervals (including both aggregate categories), invalid enrollment none, five draws.
- [Real registry history verification](real-history-verification.json): Czech 2021_24 (9,904,933 people), 2022_06 (9,835,139), FOIA 2022_47 (9,199,116), one draw each; verified aggregate identities where present.
- [Czech ordinary workbook smoke test](real-workbook-verification.json): actual 2021_24 registry, primary module-off, three draws; 4,539/5,745 reported rows received intervals, including all-ages and born-1960-or-later. Of these, 1,851 were labeled with fit flags. No scientific inference from three draws. Unknown age is included in the all-ages count; existing reporting does not emit it as a separate age row. [Workbook](Czech-2021_24-3-draw-smoke.xlsx).

No changes to the fitter, estimator, conditional age/dose sampling design or Table 8 layout. The sampling-design wording vs implementation is the separate objection 5.5. Full 999-draw production registry replay not performed. Paper section 2.9.1 updated; [PDF](KCOR-clean.pdf), [paper patch](paper.patch), [code patch](bootstrap_reporting.py.patch), [synthetic test](verify_histories.py), [real layout test](verify_real_layout.py). Existing draft PR unchanged.
