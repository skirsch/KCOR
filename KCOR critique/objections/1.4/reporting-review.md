# 1.4: Table 8 reporting and interpretation

Discussion note for lead-author review; no manuscript edit at this stage.

## Original issue and present wording

The empirical age-shift control relabels different age groups within the same underlying vaccination category. The construction has no vaccination contrast, but the underlying age groups may have different mortality trends. Thus near-one behavior is a diagnostic consistency check, not proof of an exact population null or equivalence. The original issue about attributing fitted frailty was addressed separately.

The revised paper section 3.1.2 (paper.md line 820) already reports the previous joint fit with fixed wave exponent 1.18 and 999 person-bootstrap draws, and says all six intervals contain one. Table 8 caption (line 1237) identifies the historical v4.3 and joint-fit procedures, but does not explicitly label the joint-fit column as sensitivity-only. The operational summary says: "When alpha is not identified, step (iv) is inactive and does not affect reported results." Section 5.4 likewise says an unidentified exponent should be inactive rather than forced for stable inference.

Two distinct issues:
- The previous reported 2022 joint fit has 2/9 original insufficient-signal subgroup fits and 932/999 bootstrap draws with at least one flagged subgroup. The intervals include the configured zero-frailty fallback. A flagged component does not prove every pooled comparison fails, but the limitation must be visible when interpreting the table.
- The Czech exponent was not identified. The configuration expressly treats 1.18 as a forced sensitivity parameter, not an identified estimate. This applies even if the new fit passes every frailty diagnostic.

The latest candidate combined nonlinear survivor fit and exact weekly inversion has zero failed or insufficient-signal fits across 18,000 Table 8 subgroup fits. Its 2022 ratios with 95% intervals are 1.01255 [0.95757,1.07539], 1.05566 [0.99627,1.12107] and 1.04257 [0.97537,1.10948]. All six intervals contain 1. This addresses the specific numerical flag problem under that experimental fitter, but neither identifies alpha nor proves equivalence.

## Minimal reporting proposal

Label forced-alpha results explicitly as sensitivity estimates conditional on the supplied exponent and configured wave dates. State that the Czech data did not identify alpha, so these are not presented as identified NPH-adjusted contrasts. Retain the established methodological purpose of the age-shift control. If retaining the older joint-fit column, disclose its original and bootstrap flag counts and that intervals include the fallback. If the lead approves replacement with the new fit, report its actual flag counts instead.

Proposed addition to the Table 8 caption or section 3.1.2:

> The NPH-adjusted estimates use a prespecified sensitivity value of alpha=1.18 because alpha was not identified in these data. Their intervals are conditional on the supplied exponent and wave windows.

The primary core KCOR results with the optional NPH correction inactive can be reported alongside the sensitivity column after selecting the approved fitter. This does not assume alpha is physically equal to 1; it follows the existing optional-module reporting rule. Do not silently switch procedures, exclude flagged draws, or adopt the experimental method before lead-author approval.

[Revised manuscript section 3.1.2](<../../iterative review/revised/documentation/preprint/paper.md#312-empirical-negative-control-using-national-registry-data-czech-republic>) · [Historical flags and replay](../1.2/entry-moment/table8_bootstrap.json) · [Latest candidate Table 8](../3.2/nonlinear-survivor-fit/README.md).

## New survivor-law alpha information diagnostic

[Conditional Fisher benchmark](alpha-information/README.md) finds weak temporal information about alpha even with the exact nonlinear survivor model. The 2021 Czech fitted-reference pooled local SE is about 1; the 2022 design is effectively unidentified from shape alone. This is not a rerun of the external-scale alpha estimator. Forced alpha remains sensitivity-only.

The [historical pooled empirical replay](alpha-pooled-replay/README.md) now reproduces the original pairwise/collapse minima 1.190/1.185 exactly. Both improved-fitting arms reach the lower boundary 1.00; the exact arm reaches 0.70 on a wider diagnostic grid. No identification is established, including after excluding insufficient-signal groups. The original illustrative run supplies no external effect scale. The historical selected populations differ from the current-config shape-only diagnostic. Retain 1.18 as sensitivity-only pending lead-author reporting decision; no experimental change adopted.
