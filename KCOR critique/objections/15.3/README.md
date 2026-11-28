# 15.3 - Nested inclusive totals can describe impossible populations

**Severity:** Moderate to high for affected malformed inputs

**Origin:** Related scope gap of 14.4

**Related issues / root causes:** 14.4, 15.2.

**Finding:** All reviewers verified that all-ages and born-1960+ totals can both report OK even when the latter contains more people than the former. Also accepted with aggregate-only default inputs.

**Proposed scope (for discussion):** Cross-check nested counts and feasible complementary histories by dose/week, accounting for known older and unknown-age contributors; flag incompatible totals without dropping valid unrelated strata.

**Review evidence:** A, B, R. [Blind A](../../iterative%20review/evaluations/15/blind-a.md) | [Blind B](../../iterative%20review/evaluations/15/blind-b.md) | [Resolution](../../iterative%20review/evaluations/15/resolution.md)

**Status:** Implemented in m176; targeted workbook and actual Czech source validation completed.


**Approved correction:** Shared reconciliation now verifies nested inclusive totals and complementary risk/death histories per dose/week, including older and unknown-birth-year components. Applied to aggregate-only and mixed inputs. Incompatible totals withheld; unrelated age groups and independently verified exhaustive totals preserved. The actual Czech 2021_24 source passes. No estimator change. [Verification](../../iterative%20review/evaluations/15/nested-totals/README.md).


**Round 16 independent disposition:** Original integer examples resolved; fractional-count regression and partial cause-count validation gap remain (16.3 and 16.4). [Resolution](../../iterative%20review/evaluations/16/resolution.md).
