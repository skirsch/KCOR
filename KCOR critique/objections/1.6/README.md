# Objection 6: mathematical statements and approximations

The original item 6 in [the independent review](../KCOR_Independent_Review_EN.md) lists four issues to verify: constant hazard scaling and fitted frailty; the exact versus first-order weekly inversion; curvature as a frailty indicator; and the expected shape of the corrected Gompertz cumulative hazard.

Status: [constant observed-hazard scaling](objection6_constant_scaling/README.md), [weekly inversion](objection6_weekly_inversion/README.md), and the two [shape claims](objection6_shape_claims/README.md) now have derivations and reproducible checks. The separate [multi-window check](objection6_weekly_inversion_multiwindow/README.md) shows that changing the internal weekly approximation can switch a hard diagnostic result near a zero-offset threshold in model-generated data. [Proposed manuscript wording](PROPOSED_MANUSCRIPT_WORDING.md) addresses all four items.

The [follow-up](objection6_weekly_inversion_multiwindow/README.md#follow-up-on-simulated-and-saved-czech-counts) found 28 decision changes in 720 simulated datasets, and zero in 108 cohort fits from three saved Czech enrollments. Among the 33 Czech comparisons that passed both ways, the largest absolute KCOR change was 0.000562. These measurements apply to the tested fitter and replay scope.

The proposed resolution is ready for adoption as two local patches. Apply the [software correction](<../07 - Bugs/weekly_reconstruction/README.md>) and the [manuscript and supplement correction](manuscript_corrections.patch) together so their descriptions agree. The four mathematical claims and the observed sensitivity have documented corrections. The tested fitter still uses a fixed baseline rate in its seed fit, which is tracked as a separate discrepancy.

