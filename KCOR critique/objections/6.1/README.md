# 6.1 ? All-ages fit population differs from normalized population

**Severity:** High

The direct aggregate includes unknown birth years while fitting uses only positive birth years. In Czech 2021-24, dose 0 has 1,587,166 unknown-year members out of 5,861,036; known-only theta 31.384 versus full-aggregate theta 31.065. Across doses unknown years account for 16.0% and 16.1% in the two enrollments. Align the fitted and accumulated cohort; endpoint impact remains unmeasured.

**Related issues:** 5.7.

**Origin:** Previously noted in 5.7, independently reproduced and quantified.

**Review evidence:** [B](../../iterative%20review/evaluations/6/blind-b.md) ? [R](../../iterative%20review/evaluations/6/resolution.md).

**Status:** implemented in m118.

Implemented in m118. The existing -2 row, when supplied, is the actual cohort for both fitting and normalization. If absent, the reconstructed aggregate includes retained known and unknown birth years. Existing age-filter semantics are preserved. Ordinary workbook fixture with distinct unknown-year mortality reproduces the same fitted aggregate parameters for four layouts (supplied, reconstructed, aggregate-only, restricted-age with supplied total). [Patch](KCOR.py.patch), [verification script](verify.py). No full latest-version bootstrap replay.
