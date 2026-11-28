# 16.2 - Independent Monte Carlo pooling

Persistent m180, parent m179. The same expected-death weighting formula and first four distinct enrollment weeks are used separately in each mc_id. The two pooling paths, calendar support, and normalization-flag lookup use the matching replicate. Ordinary sheet-level weights retain the original convention. No change to the estimator or manuscript.

## Verification

In the test fixture (two independent 20-week cohorts, deliberately different age composition), the original m179 pooled endpoint of replicate 1 was 1.428234 alone and 1.073988 when another replicate was appended. In m180 the first replicate is 1.428234 whether processed alone, batched, reordered or randomly shuffled; its weights remain 0.409910/0.590090. Replicate 2 likewise retains its own weights and endpoint 1.002413 in every mode. The reported pooled rows are exactly identical, including status fields. The numerical endpoints are individual MC realization points; there is no person-bootstrap interval for a single deterministic simulation. The actual written MC workbook and controlled positive-weight normalization-flag lookup pass.

The ordinary partial-coverage bootstrap regression also passes, and the real Czech S6 point replay is identical over every one of 1,192 rows and 46 previous fields. All 51 checked numerical archive and figure files, manuscript, PDF and updated ancillary intervals are unchanged. No full simulation or bootstrap rerun.

All five round-16 active findings are implemented; independent review pending.
