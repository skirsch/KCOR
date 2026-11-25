# Objection 4: what the validation controls show

The paper already describes its controls as checks on the method under specific conditions. The remaining concern is that one discussion passage and the empirical control table caption draw stronger conclusions than the controls can support. [Manuscript §3.1 and limitations](../../documentation/preprint/paper.md)

## Flat adjusted curves and the origin of frailty

The limitations paragraph on frailty exogeneity acknowledges that the adjustment can absorb differences caused by treatment. It then cites flat adjusted Czech curves as partial evidence against large treatment-induced changes in frailty. If a treatment-induced change produces the same mortality pattern as pre-existing frailty, the fit can absorb either source and leave a flat adjusted curve. The adjusted curve alone cannot tell those two sources apart. This is a claim about what the example establishes. The general assumption and its limit are already disclosed. [Manuscript, frailty exogeneity paragraph](../../documentation/preprint/paper.md)

Proposed replacement for the last sentence of that paragraph:

> An approximately flat anchored KCOR trajectory shows little change in the adjusted cumulative hazard ratio relative to its reference value. Determining whether fitted frailty reflects pre-existing selection or a treatment-induced change requires independent information about the source of the mortality pattern.

## Czech age-shift control

The construction reuses mortality counts from different age groups within the same underlying vaccination category, or from age groups pooled across categories. The labels called “dose” identify age groups in this control. Relabeling creates no vaccination contrast within each construction, while real age groups can still have different mortality trends. Anchoring makes the plotted comparison equal to 1 at the chosen reference week. Later departures describe how the age groups' adjusted cumulative hazards evolved relative to that reference. The paper and supplement already say this is a pipeline check under composition differences and that a different simulation stresses large frailty adjustments. [Control construction](../../test/negative_control/README.md) · [Supplement §S4.2.2](../../documentation/preprint/supplement.md)

Four of six reported endpoints are between 0.986 and 1.021, rounded to three decimals. The two 2022_06 comparisons involving pseudo-dose 2 end at 1.0756 and 1.0911. The caption should acknowledge those exceptions. Their endpoints alone cannot distinguish age-related changes from limitations of the adjustment. The source of the reported intervals remains an open item in the earlier provenance audit. [Empirical control table](../../documentation/preprint/paper.md) · [Interval provenance audit](<../03 - Objection 3/objection3_step2_interval_provenance_2026-11-10/assessment.txt>)

Proposed replacement caption:

> End-of-window KCOR values for Czech age-shift pseudo-cohorts. Four of six estimates are close to 1. The two 2022_06 comparisons involving pseudo-dose 2 are 1.076 and 1.091. These comparisons check how the pipeline behaves when age groups have different baseline mortality and may have different mortality trends.

## Positive controls

The reported positive-control examples show that KCOR responds in the expected direction to the injected hazard multipliers. The joint frailty and treatment-effect simulation in §S4.6 explicitly places the effect outside the quiet windows. That design supports the result under the stated separation. A claim about effects that overlap the fit window or alter frailty would need evidence from a design that includes those mechanisms. The comparison here concerns the reported positive controls. The critique of broader claims about detecting overlap belongs with the existing Objection 1 assessment. [Manuscript §3.2](../../documentation/preprint/paper.md) · [Supplement §§S4.3 and S4.6](../../documentation/preprint/supplement.md) · [Objection 1 framing](<../01 - Objection 1/KCOR_Objection_1_Corrected_Framing.md>)
