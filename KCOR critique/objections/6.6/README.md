# 6.6 ? Time-window Cox comparator omits at-risk survivors

**Severity:** High for historical Cox comparison

The historical comparator selects terminal records within a window and misses those surviving beyond its end. It also independently truncates fractional deaths. This comparator's risk set is wrong. Check original intended comparison and use coherent risk sets, or remove unsupported comparative results. Separate individual-record Cox demonstration unaffected.

**Related issues:** 1.5, 2.1.

**Origin:** New code defect.

**Review evidence:** [B](../../iterative%20review/evaluations/6/blind-b.md).

**Status:** addressed in m123, preserving m122.

Independent verification against m122: intercepted the actual `PHReg` inputs from the historical comparator while supplying two coherent integer-count cohorts of 100 each, with 10 deaths per cohort in week 0, 10 in week 25, and 80 end-of-follow-up survivors. The first interval receives only 20 records, all events, instead of 200 at risk (20 deaths, 180 censored at the interval boundary). The second receives only 20 records, all events, instead of 180 at risk (20 deaths, 160 censored). The original code filters terminal time to `t_start <= time < t_end` before passing records to PHReg. This directly establishes the risk-set omission independently of any fractional-count or estimator-target issue. No change to the historical comparator was made during verification.

The corrected comparator retains everyone at risk at interval entry, administratively censors the survivors at its end, and uses differences of rounded cumulative expected deaths to conserve the integer risk population. The original intervals, uniform within-week spacing and PHReg estimator are retained. An equivalent NumPy trapezoidal integration compatibility change enabled the normal grid runner. [Code patch](cox.patch), [risk-set regression](verify.py).

The targeted six-scenario replay verified the regenerated counts against the saved input workbook, and the full normal grid run reproduced the corrected Cox summaries and all original non-Cox outputs within numerical tolerance. Only the Cox summary fields and rows of the saved diagnostic/comparison CSVs are updated, preserving the other original values exactly. The sparse-event Cox comparison now returns a finite fit. The six scenario inputs are deterministic expected counts; the between-window SD measures variation across time intervals, not a sampling CI. [Before-after](before-after.csv), [replay results](cox-replay.json), [replay script](replay.py), [updated diagnostics](sim_grid_diagnostics.csv), [updated comparison](comparison_table.csv). Table 11 now reports the gamma-null mean HR 1.069 and between-window SD 0.044 and describes the corrected calculation. Separate comparator-interpretation concerns are reserved for 6.7. [Paper patch](paper.patch).
