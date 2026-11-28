# Round 11 post-review revisions

Preserved m156 (frozen evaluated checkpoint); m157 is the approved straightforward correction batch. No change to frailty model or weighting intent.

- [11.1](../../../objections/11.1/README.md): Aggregate membership determined before display-year filter. Both input representations give exactly the same point and three matched ordinary bootstrap intervals: 0.999967 [0.956208, 1.013158] (smoke only).
- [11.2](../../../objections/11.2/README.md): Dedicated replay no longer clips 100% mortality into a finite increment; exhausted 2/0 and 2/1 unavailable, unaffected comparison still 1.016193.
- [11.3](../../../objections/11.3/README.md): Unclipped positive finite geometric mean; sole-stratum extreme test direct and pooled both 21082653773.60724; numerical-range failure unavailable.
- [11.4](../../../objections/11.4/README.md): Separate residual flag and unavailable indicator for point/direct and actual pooled contributors, separate draw counts. Tested propagation and written workbook columns; no estimator change.
- [11.6](../../../objections/11.6/README.md): Shared existing expected-deaths weighting fallback. Sparse test yields 0.938236, 0.912340, 0.972400, matching ordinary reference.
- [11.7](../../../objections/11.7/README.md): Shared reader validates raw integer counts before conversion; fractional fixture rejected.
- [11.8](../../../objections/11.8/README.md): Point and bootstrap use same enrollment parser; hyphen and underscore produce identical person-history cells.
- [11.9](../../../objections/11.9/README.md): Explicit About field/value pairs and active settings, written workbook verified.
- [11.12](../../../objections/11.12/README.md): Main and SI sources updated; primary 999-draw refresh matches all old points and draws to 2.22e-16; prior-seed forced-alpha full intervals explicitly archived, fresh 3-draw smoke separate.
- [11.13](../../../objections/11.13/README.md): Cumulative opening-risk person-weeks computed within dose/age after weekly aggregation; written workbook matches cumulative risk sets.
- [11.14](../../../objections/11.14/README.md): Wave diagnostics export mode, actual wave weeks, strict deficit count and zero-excess count; missing strict-count mode explicitly unavailable. Forced-alpha actual point run verified diagnostic records.
- [11.15](../../../objections/11.15/README.md): Week-52 endpoint and observed deviation replacing misleading asymptote and bias labels in Cox text/figures; archive column name retained for reproduction. Existing intervals used for plotting.
- [11.16](../../../objections/11.16/README.md): Geometric pooled rows have blank fitted theta cells; separately fitted all-ages theta retained. Written workbook checked.
- [11.18](../../../objections/11.18/README.md): Known gap reentry retained with NaN unavailable signed offset; detailed row marked unassessed with seed status, and previous seed ceiling regression passes.

Primary Table 8 999 matched draws and all diagnostic category arrays agree with m156 to floating-point precision (maximum ratio difference 2.22e-16). All six displayed 95% intervals unchanged. The fresh forced-alpha three-draw smoke is not a full-interval replay; prior-seed forced-alpha 999-draw intervals remain archived. S6 prior 999-draw results unaffected: direct stratum fitting unchanged. Previous round-10 regressions passed.

Remaining for individual discussion: 11.5, 11.10, 11.11, 11.17.

[Batch patches, figures, outputs and PDFs](post-review-batch-1/).

- m158: 11.5 wording correction in main and supplement; code unchanged. Remaining: 11.10, 11.11, 11.17.

- m159: 11.10 Cox comparator wording and nominal interval labels; numerical results unchanged. Remaining 11.11 and 11.17.

- m160: 11.11 competing-event scope qualification in Sections 1.1 and 5; code unchanged. Remaining 11.17.

- m161: 11.17 historical S7 parameters and archive references in S4.6. All 18 round-11 findings now have implemented responses; independent closure pending.
