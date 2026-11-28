# 7.8 - Paper build and archived-study reproducibility claims

**Severity:** Moderate

**Origin:** New documentation/evidence defect

**Related issues / root causes:** 6.4, 6.6.

**Finding:** SI S7 says make paper regenerates all artifacts but it renders the manuscript from existing figures. Stored S8 nonlinear-wave study intervals and summaries exist but no generator in the frozen scientific tree was located.

**Proposed scope:** Correct build instructions; locate and include the actual archived study runner and seeds/settings if available, or clearly label the stored evidence as archived rather than claim rerunnability.

**Review evidence:** [Blind B](../../iterative%20review/evaluations/7/blind-b.md)

**Status:** addressed in m129, preserving m128 (approved batch of straightforward corrections).


The original generation scripts were located in the historical archive, and copied with their available dependencies and hashes into analysis/survivor_validation/historical-source. They depend on historical estimator editions and are labeled as such, not claimed to be a standalone current simulation runner. Exact matched draws for both arms of both S8 studies are now included. The standalone archive verifier independently reproduces all 300 individual-dataset intervals, 90/100 and 48/50 coverage in each arm, and the stated means. Main design: seed 20261124, 100 datasets and 199 draws; stress seed 20261126, 50 datasets and 99 draws. The paper and analysis README distinguish rendering from regenerating studies. No expensive new simulations.

[Combined batch code patch](../../iterative%20review/evaluations/7/post-review-batch-1/workbook.patch) | [Paper patch](../../iterative%20review/evaluations/7/post-review-batch-1/paper.patch) | [Tests](../../iterative%20review/evaluations/7/post-review-batch-1/tests.patch)

[Archived validation data and source](archived-validation/)

**Round 8 independent disposition:** partial; remaining concrete issue tracked in [8.10](../8.10/README.md).
