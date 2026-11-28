# Blind review A — candidate m185

## Verdict

The proportional gamma inversion and weekly Gompertz hazard equation are mathematically correct under their stated model. The cohort-specific mean-one reference scale, noncausal interpretation, externally fixed optional nonlinear exponent, conditional uncertainty and the distinction between model diagnostics and numerical convergence are explicit. I found no new mathematical defect in those core identities. I found one consequential gap in an ancillary reporting path and a smaller calendar-reporting inconsistency. Neither changes the published primary Table 8 results by itself.

## Substantive issue for ancillary negative controls

The optional `NEGATIVE_CONTROL_MODE` in `code/KCOR.py:8364–8425` computes same-dose age-pair normalized cumulative-hazard contrasts using the ordinary per-cohort cumulative hazards, but discards the corresponding fit, residual-screen and gap-assumption qualification. It assigns `endpoint_status="ok"` solely if the contrast is finite and `anchor_status="ok"` if the reference is numerically positive. This is particularly dangerous for a negative-control summary because a numerically near-one contrast may appear to validate the correction even if the frailty fits were not identified and fell back to zero.

Verified using the existing `test/test_round7_workbook.py` `input_frame('2023_01')`, all four cohort histories complete, 20 weeks from 2023-01-02 through 2023-05-15. Ordinary `return_rows=True` rows for 1940 and 1950 report both numerator and denominator `theta0_status=NOT_IDENTIFIED`. Running the ordinary written workflow with `K.NEGATIVE_CONTROL_MODE=1`, no bootstrap, and the report date fixed to the last observed week, produces this independent Excel summary:

| YoB1 | YoB2 | Dose | KCOR | Unanchored | Endpoint status | Anchor status |
|---|---|---|---|---|---|---|
| 1940 | 1950 | 0 | 0.992621 | 0.673626 | ok | ok |
| 1940 | 1950 | 1 | 0.993567 | 0.710272 | ok | ok |

The special summary lacks any fit qualification. The numerical values are legitimate fallback audit values, but cannot be interpreted as successfully normalized controls. Section 2.1.2 and Box 2 prescribe withholding substantive interpretation in this situation. The fix is to carry both contributing cohort fit statuses and relevant residual/gap-screen information through the same-date age-pair summary, explicitly qualifying finite fallback values as audit-only. Retain the present numerical outputs and anchor conventions. Add a regression with weak/failed fits and a separately well-identified comparison. If this hard-coded optional mode is obsolete, retire it rather than leaving a misleading independent reporting path. This differs from the main Table 8 validation, which documents its diagnostic flags extensively.

## Minor correction — ordinary ISO labels

`KCOR.py:6752` correctly derives quiet-window membership from `DateDied`. The MC source validator at 4358–4366 rejects contradictory labels and canonicalizes dates. Ordinary processing at 6656 still selects the first supplied `ISOweekDied` without checking it. Verified on the same ordinary fixture with the dose-1, 1940 labels shifted eight weeks while all dates and count histories remain correct. Its last row has `Date=2023-05-15` and both emitted ISO fields `2023-28` (should be `2023-20`), yet KCOR is unchanged at 1.003326 and `normalization_status=ok`. This is an erroneous displayed calendar, unable to reproduce quiet-window selection by reading the output labels. Share the existing canonicalization/contradiction check with ordinary inputs and bootstrap reconstruction, or label the emitted ISO weeks directly from the validated dates. Correct labels should leave all point estimates unchanged. This is a reporting defect, not a wrong fit for the tested ordinary case.

## Important scientific limitations already disclosed

The use of the Gompertz slope, multiplicative working frailty and quiet-window assumptions still determines the interpretation. The inversion cannot distinguish model-conforming depletion from sufficiently similar nonfrailty curvature. The manuscript makes these conditional, defines cohort-relative reference frailty, states nonparametric non-identifiability and presents the joint fitting objective explicitly. This is an assumption-dependent descriptive estimand; an alternative causal design is not needed to establish the stated mathematical target. The observed Czech age-shift example has no known exact null and the manuscript makes its post-fitting residual failures and historical quiet-window choice explicit. A substantive empirical conclusion from these examples would still be unjustified; no extra redesign follows from that limitation. The optional wave exponent procedure's approximation and unresolved localization from weekly data are also disclosed.

The full person-bootstrap was not run. The targeted ordinary workbook probes ran in temporary directories with bytecode creation disabled. I examined the frozen m185 source and manuscript only.
