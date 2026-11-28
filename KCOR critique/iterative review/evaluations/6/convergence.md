# Round 6 ? reduced independent evaluation of m117

## Protocol

Two fresh GPT-6.2 Astra High blind reviews and a separate Astra Medium targeted resolution review. Each received the same frozen scientific tree, with the original question verbatim; only the resolution reviewer received round-5 records. No full bootstraps, estimator changes or PR updates were performed.

- [Blind A](blind-a.md)
- [Blind B](blind-b.md)
- [Resolution](resolution.md)
- [Input manifest](inputs.json)
- [Input verification](input-verification.json)

## Convergence

Both blind reviewers affirm the proportional gamma inversion and the fixed-slope identification proposition under their model assumptions. The resolution reviewer independently verifies ten of the twelve round-5 objections within their original scope. The two partial items are 5.6 (actual application eligibility) and 5.11 (remaining fallback and analytical interval limitations). The original 5.10 claims were narrowed successfully; alpha localization remains deferred, and both blind reviews sharpened its instantaneous-versus-weekly distinction.

The newest findings primarily concern cohort construction, reporting and validation, with a specific optional nonlinear numerical defect. No general failure of the core inversion was found. All twelve archived Table 8 intervals reproduced from the stored draws and contain one; they predate m117, and a new full latest-version replay is pending.

## Numbered findings

| ID | Severity | Finding | Review | Origin |
|---|---|---|---|---|
| [6.1](../../../objections/6.1/README.md) | High | All-ages fit population differs from normalized population | B, R | Previously noted in 5.7, independently reproduced and quantified |
| [6.2](../../../objections/6.2/README.md) | Moderate | Incorrect propagation of aggregate diagnostic flags | R | New reporting defect; independently reproduced by resolution reviewer |
| [6.3](../../../objections/6.3/README.md) | High for diagnostic claims | Numerical fit acceptance reported as passing all diagnostics | A, B | New evidence about an existing diagnostic distinction |
| [6.4](../../../objections/6.4/README.md) | High for empirical example | Historical SI S6.1.2 numbers lack a reproducible matching configuration | B | New reproducibility discrepancy; source configuration remains unverified |
| [6.5](../../../objections/6.5/README.md) | High for precision claim | The 5,000-per-arm precision claim relies on deterministic simulations | B | New validation defect |
| [6.6](../../../objections/6.6/README.md) | High for historical Cox comparison | Time-window Cox comparator omits at-risk survivors | B | New code defect |
| [6.7](../../../objections/6.7/README.md) | Moderate to high for comparator claims | RMST/Cox targets and shared-frailty comparative evidence | A, B | Earlier interpretation issue resurfaces in additional passages |
| [6.8](../../../objections/6.8/README.md) | Moderate | Age-shift control captions claim missing interval bands and guaranteed flatness | A | Sharpened existing control limitation plus new caption issue |
| [6.9](../../../objections/6.9/README.md) | Moderate | Anchoring interpretation overstates post-reference isolation | A | New interpretation issue |
| [6.10](../../../objections/6.10/README.md) | Minor | Five-year age bands labeled single birth years | A | New documentation issue |
| [6.11](../../../objections/6.11/README.md) | Moderate for high event rates | Active baseline-rate bounds not flagged | B | New edge case |
| [6.12](../../../objections/6.12/README.md) | High for extreme nonlinear fits | Nonlinear survival quadrature is not monotonic at a branch switch | B | New numerical defect; nonlinear supplied-alpha path |
| [6.13](../../../objections/6.13/README.md) | Moderate for optional alpha localization | Instantaneous wave moment identity versus weekly observed excess | A, B | Sharpened deferred alpha objection |
| [6.14](../../../objections/6.14/README.md) | High for Czech population selection | Actual post-enrollment manufacturer eligibility exclusion | A, B, R | Existing 5.6 only partially resolved; new quantitative evidence |


## Round-5 resolution

Resolved within original scope: 5.1, 5.2, 5.3, 5.4, 5.5, 5.7, 5.8, 5.9, 5.10 (claim only), 5.12. Partial: 5.6 (see 6.14), 5.11 (other flagged fallback and secondary analytical interval limits remain). Earlier open issues include baseline-model conditionality (1.1), general coverage (1.3, 4.6), age-shift control target (1.4), low-information frailty resolution (2.7), unmarked nonlinear history (3.2, 4.3), and alpha localization (4.7). The full resolution contains evidence and qualifications by ID.

## Discussion order

Start with 6.1, the mismatch between the all-ages fitted population and its reported trajectory. Then discuss 6.2?6.12 individually, checking the intended scientific comparison before making any revisions. 6.13 remains tied to deferred alpha work; 6.14 requires an explicit application population decision.
