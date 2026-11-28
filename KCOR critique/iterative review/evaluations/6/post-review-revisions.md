# Round 6 post-review revisions

- m118 implements 6.1, 6.2, 6.9, 6.10 and 6.11 as approved together. The frozen m117 review input is preserved. Per-objection patches apply in that order (6.11's paper patch follows 6.9). Tests check actual all-ages workbook population consistency, age-specific and aggregate flags, and both proportional and nonlinear baseline-rate bound cases. No general algorithm redesign or new bootstrap study.

[6.1](../../../objections/6.1/README.md) | [6.2](../../../objections/6.2/README.md) | [6.9](../../../objections/6.9/README.md) | [6.10](../../../objections/6.10/README.md) | [6.11](../../../objections/6.11/README.md)

- m119 addresses 6.3 for the quiet-window scan. Reproduces all 78 fitted values and 63 exploratory residual flags, exports flagged weeks and distinguishes the two criteria in the figure and supplement. The separate historical S6 table will be addressed with 6.4. [Evidence](../../../objections/6.3/README.md).

- m120 addresses 6.4 and the remaining S6 diagnostic claim in 6.3. Historical counts nearly recover the published raw numbers under a known-age-only total, shifted birth bands and two-week skip. Current replay uses the actual full aggregate and documented age bands, records all 14 fit diagnostics, and regenerates the three S6.1.2 tables. Six residual flags are reported. [Provenance and replay](../../../objections/6.4/README.md).

- m121 adds the 999-draw raw frailty intervals and anchored KCOR intervals to the S6.1.2 table. It marks the full-range poorly resolved 40-49 dose-2 variance and qualifies the two low-information boundary estimates. [Matched results](../../../objections/6.4/all-age-precision.md).

- m122 addresses 6.5, replacing the unsupported historical fixed cohort-size precision claim with the existing model-conditional information limits and observed event-count/quiet-window requirements. [Paper patch](../../../objections/6.5/paper.patch).

- m123 addresses 6.6, correcting interval risk sets and cumulative death rounding in the historical Cox comparator. All six scenarios replay successfully, matching the stored inputs and other historical outputs within numerical tolerance. Table 11 and the Cox columns of its supporting results are revised. [Evidence](../../../objections/6.6/README.md).

- m124 addresses 6.7. Clarifies Cox/RMST estimands, removes the unverified shared-frailty table and its calibrated proxy curve, preserves the reproducible numerical comparator results and existing KCOR intervals. [Evidence](../../../objections/6.7/README.md).

- m125 addresses 6.8: both age-shift negative control curves regenerated with the all-window fitter and 999 matched person-bootstrap draws; flatness conditional on proportional normalized trajectories; all endpoint intervals include unity, no unsuccessful fits, residual screens disclosed. [Evidence](../../../objections/6.8/README.md).

- m126 addresses 6.12: shared accuracy-controlled gamma survival integral for fitting and inversion, with independent extreme-regime checks, monotonicity tests, exact wave recovery, and matched Czech original/percentile-draw replays. No mathematical model change. Table 8 forced-alpha intervals remain explicitly archived. [Evidence](../../../objections/6.12/README.md).

- m127 addresses the 6.13 manuscript claim: instantaneous actual-survivor wave component vs weekly excess proxies, and dependence of pairwise/collapse objectives. Full weekly alpha localization remains deferred as agreed. Supplied-alpha inversion, ordinary estimator and results unchanged. [Evidence](../../../objections/6.13/README.md).

- m128 addresses 6.14 by explicit Czech study population disclosure, verified product-withdrawal context, exact aggregate eligibility audit and clarification of all-ages labels. User confirmed retrospective population definition remains intentional; no code or results changed. Application sensitivity to alternative eligibility remains unquantified. [Evidence](../../../objections/6.14/README.md).

All 14 round-6 findings have now been dispositioned under the approved scope. The broader weekly alpha-localization investigation and Czech alternative-eligibility sensitivity remain open, as documented; no further evaluation or PR submission was initiated. Final local candidate for this series: m128.
