# Separate KCOR Monte Carlo software fix

This bundle fixes two defects in the inspected `code/KCOR.py`: bootstrap samples shared a time counter, and diagnostic reporting did not consistently recognize sample-specific fit records. The original source and the completed objection 3 experiment remain unchanged. Nothing has been applied, staged, or submitted by this bundle.

## What the defects mean

Each bootstrap sample is another version of the same follow-up period. Its clock must restart independently. Previously, the time counter grouped by birth year and dose but omitted the sample identifier. Adding samples therefore changed the times assigned to the original observations, which could change the quiet-window fit and the original estimate itself.

Monte Carlo fit records also have an extra sample identifier in their keys. Several diagnostic summaries expected the ordinary three-part keys. This could omit failed fits from the compact summary or fail to retrieve the corresponding fit details. The patch retains sample identifiers in Monte Carlo diagnostic tables and includes failed fits even when their fitted parameters are missing.

These are implementation findings, not a mathematical rejection of KCOR. The completed coverage experiment describes the original implementation. Its results cannot establish the coverage of the corrected implementation, or establish that every published result is affected. Fresh coverage validation is still required.

## Submission contents

`KCOR_MC_sample_isolation.patch` changes only:

- `code/KCOR.py`: independent sample clocks and sample-aware diagnostic reporting.
- `test/monte_carlo/test_sample_isolation.py`: three regression tests.

`submission/` holds the patched source and tests for isolated execution. Its unchanged `data/Czech/Czech.yaml` copy supports execution and is not included in the patch. The fitting algorithm, interval formulas, resampling algorithm, manuscript, and completed experiment are not changed.

The patch is based on source SHA-256 `3d9e0320739baa3d11b1058e8b134335f7edcd9a0d7df6ec22234c2abf1f49ef`. Exact hashes are in `manifest.json`.

## Verification

`verification/verification.json` records the completed checks:

- Original source: three tests ran, with two expected failures; process exit code 1.
- Patched source: all three tests passed; process exit code 0.
- Tests compare original-alone, combined, shuffled, and relabeled samples; check failure counts and sample-specific diagnostics; and check the ordinary non-Monte-Carlo path on the fixture.
- Saved ten-sample input from the completed experiment was replayed against the patched copy. All 100 original-sample curve points exactly matched the retained original-alone result. At week 80, the old combined result was 1.053172263346009; both the original-alone and corrected combined results were 1.004049494273245. The corrected original fit reported `OK`.
- The retained input and original source hashes were unchanged. `git apply --check` passed.

The fixture tests use age-specific outputs with the optional wave correction disabled. They do not constitute a new coverage study or an audit of every production configuration. Logs and workbook outputs are retained under `verification/`.

## Using the separate patch

In the intended submission checkout, review and check the patch before applying it:

```powershell
git apply --check "path/to/KCOR_MC_sample_isolation.patch"
git apply "path/to/KCOR_MC_sample_isolation.patch"
python test/monte_carlo/test_sample_isolation.py
```

The apply command above is documentation only; it has not been executed here. Tests require the existing KCOR runtime dependencies and Czech configuration.

`build_patch.py` documents how the isolated copy and patch were generated. `verify_bundle.py` documents the original-versus-fixed tests and historical replay. Both preserve existing output by refusing to overwrite their bundle output directories; use a fresh copy of this bundle without those generated directories to repeat the full preparation workflow.
