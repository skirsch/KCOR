# 5.7 - Round 5 finding

**Severity:** High for active-wave all-ages results

All-ages output recomputes ordinary inversion after conditional survivor correction; also a potential missing draw identifier.

**Related issues:** 1.2, 4.6, reporting bypass

**Review evidence:** B - [Series 5 convergence](../../iterative%20review/evaluations/5/convergence.md) ([A](../../iterative%20review/evaluations/5/blind-a.md), [B](../../iterative%20review/evaluations/5/blind-b.md), [R](../../iterative%20review/evaluations/5/resolution.md)).

**Status:** corrected in m109; m108 preserved.

The ordinary all-ages reporting path now reuses the already normalized hazard increments, adjusted cumulative hazard and observed cumulative hazard from the shared normalization stage, including the conditional survivor-law correction. It no longer inverts gamma a second time or looks up a theta from the wrong Monte Carlo draw. All-ages abnormal-fit metadata now uses the matching draw identifier. The legacy slope-based reporting branch and all fitting/inversion equations remain unchanged.

Verification:
- [Synthetic workbook](wave-reporting-input.xlsx) has one birth-year group and an identical all-ages aggregate, with distinct generating frailty variances and a known nonlinear wave. [Workbook regression](workbook-verification.json) compares both versions and both modes with five matched person-bootstrap draws. Under forced alpha=1.18, old all-ages endpoint 1.008094 [0.999642,1.023726] differed from its identical per-age group 1.000004 [0.991564,1.015432]; both new paths agree exactly at 1.000004 [0.991564,1.015432]. Module-off point and interval outputs match the previous version exactly (within floating point tolerance). These five-draw intervals are a software smoke test, not inference.
- [Trajectory regression](verify_trajectory.py) captures the actual normalization output and checks all-ages reporting directly. Maximum adjusted cumulative hazard error went from 0.0121693 to zero in ordinary reporting, and from 0.3975595 to zero in a two-draw reporting fixture. Both MC draw IDs and flags are preserved. This is a direct MC reporting test, not a full simulation run.

Table 8 uses the per-age pooled reporting path, not the all-ages row; this patch does not change its fitter or bootstrap. No full new Table 8 replay. Paper already specifies the corrected cumulative-hazard ratio, so no new wording is required.

Separate observation for later author discussion (not changed here): the existing all-ages fit constructs its input from positive birth years even when the supplied -2 total includes unknown birth year -1; see KCOR.py all-ages fitting block and 5.4 real-layout verification. This is distinct from the reporting bypass and can change the module-off fit too. Do not conflate or automatically change the population definition.

[Code patch](KCOR.py.patch), [synthetic generation](generate.py), [old trajectory check](m108-trajectory-check.json), [new trajectory check](m109-trajectory-check.json). Existing draft PR unchanged.
