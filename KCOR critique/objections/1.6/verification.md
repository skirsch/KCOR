# PR #5 objection 6.2 — weekly reconstruction

The revised fitter applies the exact gamma-frailty increment in the offset iteration and in its final diagnostic reconstruction. The limit at zero frailty is the observed weekly increment. The paper distinguishes the instantaneous hazard identity in §2.4.2 from the integrated weekly increment in Step 2 of §2.5. The baseline edition and the first-window fitting decision are preserved.

The revised fitter reproduced the exact-update points, frailty estimates and diagnostic flags in all four [archived PR cases](<objection6_weekly_inversion_multiwindow/results.json>). The single-window control gave the same point with either update. In the two-window checks, the no-wave and one-arm-wave cases passed the diagnostics with the exact increment. Both had failed with the earlier approximation. The three existing sample-isolation regression tests passed.

## Table 8

Using the same Czech input, reporting date, joint first-window fit and 999 person-bootstrap draws per enrollment as in the [previous run](../1.3/table8_bootstrap.json), all six pooled point estimates matched the independently pooled stratum rows from the full workbook to relative tolerance 1e-10. The largest point estimate change was 0.000026. The largest endpoint change was the upper bound for 2022_06, 2 vs 1, from 1.228 to 1.209. All six intervals included 1.

The original data produced zero flagged fits among the nine contributing fits for 2021_24 and four for 2022_06, as before. The number of bootstrap draws with at least one insufficient-signal flag changed from 271 to 512 of 999 for 2021_24 and stayed at 998 of 999 for 2022_06. All 17,982 stratum fits returned successfully, including those using the configured fallback. Interpretation of the flags remains deferred for discussion with the project lead.

Run `py -3.11 replay.py` from this directory to reproduce the [full-precision estimates and intervals](table8_bootstrap.json). The script reconstructs the archived input from Git, runs the revised joint fit through the workbook entry point and checks its six independently pooled point estimates against the bootstrap runner. The original run is preserved in the earlier review directory.
