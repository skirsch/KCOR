---
name: execute-kcor-insufficient-deaths
overview: Execute and validate the insufficient-deaths/safety alignment in KCOR by running targeted checks, confirming expected control-flow/status behavior, and fixing any residual gaps before final sign-off.
todos:
  - id: verify-control-flow-order
    content: Confirm fit_theta0_gompertz ordering and status transitions match spec.
    status: pending
  - id: run-validation-checks
    content: Run syntax/lint and Czech execution path checks.
    status: pending
  - id: inspect-logs-diagnostics
    content: Verify BAD_FIT and insufficient-deaths/signal behavior in logs and diagnostic outputs.
    status: pending
  - id: confirm-summary-consistency
    content: Validate theta0_status_summary and diagnostics schema/count consistency.
    status: pending
  - id: close-gaps-and-rerun
    content: Address any discrepancies and rerun until acceptance criteria pass.
    status: pending
isProject: false
---

# Execute KCOR Insufficient-Deaths Alignment

## Objective

Validate and complete the `KCOR.py` insufficient-deaths/safety implementation so behavior, logs, and diagnostics match the v7 spec expectations.

## Scope

- Code under [C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py](C:/Users/stk/Documents/GitHub/KCOR/code/KCOR.py)
- Reference specs:
  - [C:/Users/stk/Documents/GitHub/KCOR/documentation/specs/KCORv7/insufficient_deaths.md](C:/Users/stk/Documents/GitHub/KCOR/documentation/specs/KCORv7/insufficient_deaths.md)
  - [C:/Users/stk/Documents/GitHub/KCOR/documentation/specs/KCORv7/insufficient_signal.md](C:/Users/stk/Documents/GitHub/KCOR/documentation/specs/KCORv7/insufficient_signal.md)
- Dataset config for run validation:
  - [C:/Users/stk/Documents/GitHub/KCOR/data/Czech/Czech.yaml](C:/Users/stk/Documents/GitHub/KCOR/data/Czech/Czech.yaml)

## Execution Steps

1. Re-verify control-flow order in `fit_theta0_gompertz`:
  - pre-fit deaths gate -> fit -> delta/bound insufficient-signal -> bad-fit relRMSE -> OK.
2. Run static checks and targeted runtime execution on Czech dataset:
  - ensure no syntax/lint regressions,
  - ensure pipeline completes and outputs diagnostics/summaries.
3. Validate expected outcomes in logs and diagnostics:
  - `INSUFFICIENT_DEATHS` path exits pre-fit with theta zero,
  - `INSUFFICIENT_SIGNAL` paths include delta/bound and bad-fit cases,
  - `[KCOR7_BAD_FIT]` appears when threshold is exceeded.
4. Verify status aggregation consistency:
  - `theta0_diagnostics` and `theta0_status_summary` include stable schema and correct counts for `OK`, `INSUFFICIENT_DEATHS`, `INSUFFICIENT_SIGNAL`, `NOT_IDENTIFIED`.
5. Resolve any mismatches found during execution, then re-run validation to green.

## Acceptance Criteria

- Control-flow semantics match spec order exactly.
- SA/per-age/all-ages paths produce consistent status and logging behavior.
- No new lint/syntax errors.
- Output diagnostics and summary remain schema-stable and internally consistent.

## Deliverables

- Verified working implementation in `KCOR.py`.
- Brief validation report with:
  - commands run,
  - pass/fail checks,
  - any edge cases observed,
  - final go/no-go statement for execution readiness.

