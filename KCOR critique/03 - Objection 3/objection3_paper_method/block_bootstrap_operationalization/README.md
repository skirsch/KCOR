# Aggregate-block interval implementation of the paper's outline

Start with the [paper-first assessment](ASSESSMENT.md). The main four-week version covered **38/200** model-consistent datasets, and the paired eight-week check covered **10/60**. These are conditional results for this module's explicit rules.

This module makes the manuscript's §2.9.1 block-bootstrap outline executable for the **one-window, one-stratum, no-wave** model-consistent scenario used in the paper Step 1 check. It keeps the paper's joint baseline-rate/frailty objective, exact cumulative gamma inversion, week-5 normalization, and percentile endpoints. The [frozen design](design.json) records every choice that the paper does not specify.

The primary implementation samples four-week runs of weekly event-time bins within three fixed calendar segments: before the quiet window, within it, and after it. A sampled source bin contributes its observed death fraction. Deaths are applied to the current reconstructed risk set and rounded to a whole count; the risk set then decreases by that count. This preserves source-block order and exact within-replicate risk-set accounting. An eight-week block version checks sensitivity to a missing paper choice. The study uses independent simulated datasets under the paper's working gamma-frailty and Gompertz model. It does not resample individual people.

**Attribution limit:** The manuscript supplies no block length, source/target placement rule, seam rule, or numeric diagnostic pass thresholds. This module is one explicit implementation of its outline. Its coverage is the module's result, not an observed coverage rate for a uniquely specified or historically run paper algorithm. The [paper-method parent](../README.md) remains the main interpretation; production-code and alternate-estimator studies are supporting comparisons.

The main run reports both point-estimate behavior and interval coverage, together with optimizer and boundary-fit frequencies. It cannot report the paper's claimed *conditional on full diagnostic acceptance* coverage without that gate's numeric definition. See the [source gap](../BLOCK_BOOTSTRAP_GAP.md).
