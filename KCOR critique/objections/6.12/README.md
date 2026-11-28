# 6.12 ? Nonlinear survival quadrature is not monotonic at a branch switch

**Severity:** High for extreme nonlinear fits

At theta=99, alpha=1.18, B=0.01, a tiny increase in W around 2.1270330577 reduces computed cumulative hazard from 0.05372711 to 0.05211058. Independent integration at alpha=1.30, W=1 also disagrees with the fast fitter. Verify and repair numerical evaluation; no evidence yet of effect on ordinary empirical results.

**Related issues:** 1.2, 5.11.

**Origin:** New numerical defect; nonlinear supplied-alpha path.

**Review evidence:** [B](../../iterative%20review/evaluations/6/blind-b.md).

**Status:** addressed in m126, preserving m125.

## Numerical fix

The fitter's moment-corrected approximation switched abruptly to a separate quadrature, and at the stated high-frailty test its error was 0.001616535 in cumulative hazard. The existing weekly correction agreed with independent integration at those specific inputs. The exact starting-gamma survival law is unchanged. Fitting, in-wave inversion and post-wave inversion now share one positive-weight gamma probability integral, comparing consecutive orders (32, 64, 128, 256, 512) with absolute tolerance 2e-13 and relative tolerance 2e-12 in log wave survival. If the refinement fails to converge, an explicit arithmetic error is raised instead of accepting an inaccurate value. Zero frailty, proportional alpha=1, and zero wave exposure use analytical expressions. No change to wave attribution or the ordinary proportional path.

| Case (theta=99, B=0.01) | Before fitter | Revised | Independent integration |
|---|---:|---:|---:|
| alpha 1.18, W=2.1270330556 | 0.05372711120165 | 0.05211057609163 | 0.05211057609163 |
| alpha 1.18, W=2.1270330598 | 0.05211057610856 | 0.05211057610858 | 0.05211057610858 |
| alpha 1.30, W=1 | 0.04435511987201 | 0.04528938592823 | 0.04528938592823 |

Independent integration: 960 cases over theta 0.01 to 100, alpha 0.5 to 4, B 0 to 2 and W 0 to 10; maximum absolute error 3.62e-12. Separately checked both exposure axes over 184,800 consecutive grid comparisons (tolerance 5e-12), exact proportional identity, root recovery and the existing post-wave continuity regression. Three independent noiseless full-wave examples at theta 2, 20 and 99 recover the generating k, theta and cumulative B+W, with max curve error 2.54e-13. Reference integration emitted a roundoff warning at its tight requested tolerance but agreed within the test's 1e-9 tolerance in all cases.

## Empirical impact and runtime

The two archived Czech Table 8 enrollment point vectors and two matched draws each were replayed through the same fitter and correction routines before and after; max absolute KCOR change 7.33e-7, no failed or insufficient-signal fits. Median time per nine-stratum enrollment went from 0.931 s to 4.371 s on this machine. A further targeted replay of the 24 bootstrap replicates bracketing the twelve archived percentile endpoints changed KCOR by at most 4.88e-6, with no failed or insufficient-signal fits. Replacing only those draws in the archived array changed the calculated percentiles by at most 1.57e-6; all twelve rounded endpoints stayed the same. This is a partial check, not a complete rebootstrap. The primary module-off column is unaffected; retain the archived forced-alpha 999-draw sensitivity intervals with explicit provenance in Table 8. No claim that all 999 draws were rerun.

[Code patch](survival.patch), [tests](tests.patch), [benchmark](benchmark.py), [percentile check](percentile_check.py), [summary](summary.json), [paper patch](paper.patch).
