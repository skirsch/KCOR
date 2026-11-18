# Why a risk-set recurrence is not enough

The paper requires the resampled deaths and risk sets to obey the same survivor accounting as the original data. That is necessary, but it does not determine which weekly death pattern a bootstrap replicate represents.

Write the observed death fraction in cohort `d` at week `t` as `p_d(t) = deaths_d(t) / risk_d(t)`. In this module, a source block is copied to a target position and its death fractions are applied to the reconstructed target risk set. Ignoring only integer rounding, the expected bootstrap death fraction at a target week is a weighted average of eligible **source weeks**:

`E_boot[p*_d(t) | observed data] = sum_s w(t,s) p_d(s)`.

That equals the observed `p_d(t)` only under restrictive conditions, such as a flat hazard within the source pool or weights concentrated on nearby weeks. KCOR's working Gompertz hazard changes with time, and the two cohorts' frailty values give their death fractions different shapes. Moving blocks can therefore change the fitted curvature and the cumulative ratio even when the original weekly counts equal their exact model expectation. Rebuilding the survivor counts correctly does not undo that change.

The [exact-count block check](exact_counts_block_check.json) isolates this effect: the paper's joint fit is essentially 1 before resampling, while this module's broad-segment four-week block bootstrap is centered near 1.10. The [placement check](exact_counts_placement_check.json) shows that restricting source blocks to roughly 12-week calendar segments reduces the center shift to about 0.011. This is a post-pilot sensitivity check, not a claim that the local-segment rule is the paper's intended rule or is calibrated.

Consequently, a full specification must state not only the block length and risk-set join but also the source-time pool and how the secular Gompertz trend is respected. A different bootstrap could be better calibrated; that claim requires its own frozen definition and coverage study.
