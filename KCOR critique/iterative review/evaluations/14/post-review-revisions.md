# Round 14 post-review revisions

Frozen m169 preserved. Persistent candidate: `work/kcor-review-candidates/m170`. No redesign of core fitter or inversion.

- [14.5](../../../objections/14.5/README.md): Optional negative-control workbook uses the exact requested reporting date, retains unsupported dates, exports ReportDate and endpoint_status, and uses the shared finite quotient. Integrated exhaustion test reports the requested date, unavailable, and blank KCOR rather than substituting an earlier value.
- [14.6](../../../objections/14.6/README.md): Sensitivity sweep now uses the ordinary finite-ratio helper and unanchored support status. An integrated extreme-count sensitivity fixture with a reference below the old EPS floor returns a finite endpoint marked OK. Zero and nonrepresentable quotients follow ordinary rules.
- [14.7](../../../objections/14.7/README.md): Bootstrap accepts redundant PT exactly equal to opening Alive, drops the redundant field before resampling, and rejects genuine exposure summaries. Both acceptance and rejection tested.
- [14.8](../../../objections/14.8/README.md): Independent anchored and raw point/replicate flags now exported. At reference indices 59 and 60, the same earlier unanchored value 1.068548 [0.904564, 1.115133] has 3/3 valid draws, zero flagged draws, unflagged point and identical percentile method. The exhausted anchored reference remains unavailable and flagged. Three draws are a regression smoke test, not inferential precision.
- [14.9](../../../objections/14.9/README.md): Both point and bootstrap paths validate the declared layout before processing. Invalid Disjoint spelling raises ValueError; existing round-13 valid-layout and future-birth tests pass.
- [14.10](../../../objections/14.10/README.md): Default contamination sweep calculates pointwise Student-t 95% Monte Carlo mean intervals, not SD bars, and calls the statistic endpoint. The paper states the last eligible week of 130. An explicit historical-grid plotting CLI reproduces both published PNGs byte-for-byte; legacy file and CSV names retained.
- [14.11](../../../objections/14.11/README.md): Canonical paper.tex regenerated from the current combined Markdown sources using the manuscript toolchain, including the benefit audit caveat and finite-endpoint text. Clean and redline PDFs regenerated.
- [14.12](../../../objections/14.12/README.md): Missing S6 offset values serialized as strict JSON null. All 13,986 gap arrays parse strictly; 349 changed cells, other CSV columns identical. Recomputed all 42 gap diagnostic rows including counts, limits and status; 30 no deficit detected and 12 unassessed, identical to archived summaries.

Table 8 primary archive (four files), Table 9 gap archive (three files), and S6 numeric results are unchanged. A new 999-draw primary run was not performed because the fixes affect ancillary reporting, input validation and artifact delivery, not these numerical results.

Remaining for individual discussion: 14.1, 14.2, 14.3, 14.4. Independent review pending; no next round launched.

[Patch, strict JSON archive and paper artifacts](post-review-batch-1/).

## Manufacturer correction (m171)

Both [14.1 and 14.2](manufacturer-batch/README.md) implemented following original-intent verification. Integrated tests and actual sample converter passed; archived incompatible full Czech manufacturer input is withheld until regenerated. The paper and core method are unchanged. Remaining: 14.3 and 14.4.

## Wording and redline (m172)

[14.3](../../../objections/14.3/README.md) approved and implemented in Section 5.2. Redline references made visible as blue bracketed clickable objection numbers, with group references in the changed table captions. Earlier rounds remain in their checkpoint redlines. No estimator or reported numerical change. Remaining: 14.4.

## Aggregate reconciliation (m173)

[14.4](../../../objections/14.4/README.md) implemented. Synthetic complete/malformed/partial written workbook tests pass, including matched age-specific intervals; actual Czech S6 partial extract has exact parity across 1,192 rows and 39 existing fields. Existing main archives unchanged. Paper and linked redline updated. All twelve round-14 objections now have implemented responses; independent review pending. No additional evaluation launched.
