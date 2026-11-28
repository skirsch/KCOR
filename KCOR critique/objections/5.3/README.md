# 5.3 - Round 5 finding

**Severity:** High for reporting

Undefined/zero reference ratio silently becomes 1 and a reported anchored ratio becomes an unanchored one; primary and optional estimands not separately exported.

**Related issues:** 2.8, reporting/estimand alignment

**Review evidence:** A B - [Series 5 convergence](../../iterative%20review/evaluations/5/convergence.md) ([A](../../iterative%20review/evaluations/5/blind-a.md), [B](../../iterative%20review/evaluations/5/blind-b.md), [R](../../iterative%20review/evaluations/5/resolution.md)).

**Status:** corrected in m105; m104 preserved.

The ordinary age-specific and directly aggregated workbook paths now select the exact prespecified time from both cohort trajectories, never replace an invalid reference by 1 or another week, and export KCOR_unanchored, KCOR_anchored, anchor_status, anchor_date, and anchor_ratio. The legacy KCOR column remains anchored. Existing pooled anchored weighted geometric mean across available strata is unchanged; if no positively weighted stratum has a valid reference, explicitly report an unavailable pooled endpoint. No new pooled unanchored estimand is introduced. The existing auxiliary sensitivity/direct age-shift paths no longer use a unit fallback either.

Bootstrap reports separate unanchored percentile intervals for age-specific and directly aggregated ratios, counts undefined references and missing rows separately, and withholds anchored intervals if any required draw is undefined. The summary workbook exposes the same fields. Existing cohort fits, anchor week, aggregation weights and uncertainty sampling design unchanged.

Verification: [results](verification.json), [test](verify.py), [synthetic input/output workbooks](.). Valid reference produced both ratios and intervals; zero numerator/denominator and missing week tests passed. For a zero reference in all 3 draws, anchored interval unavailable but unanchored interval available; for a valid observed anchor with 4/12 undefined bootstrap anchors, anchored interval withheld, unanchored interval uses all 12. Direct all-ages case verified with bootstrap disabled (layout issue 5.4), pooled missing-reference cases verified. This is a smoke test, not precision evidence. The auxiliary SA/direct age-shift branches were source-checked, not separately exercised end-to-end.

Table 8 ordinary-workbook six-point check reproduced prior points in both modes (maximum absolute differences 2.22e-16 and 4.44e-16). Existing 999-draw intervals from m100 not rerun. [Primary check](identified_only-table8-check.json), [sensitivity check](forced_alpha-table8-check.json). The initial checks completed the assertions but Windows temporary cleanup raised a file-handle error after saving the results; the reproduction script now runs the writer in a separate process.

Paper section 2.8 is updated. [Clean review PDF](KCOR-clean.pdf), [paper patch](paper.patch), [code patch](KCOR.py.patch), [bootstrap patch](bootstrap_reporting.py.patch). Existing draft PR unchanged.
