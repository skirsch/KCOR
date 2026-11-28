# Post-review revisions

- 4.1: m94 remains frozen; m95 contains only the nonlinear eligibility correction. [Patch and verification](../../../objections/4.1/README.md). No paper or actual PR change.

- 4.3: m96 adds the approved wave-window misspecification disclosure to the paper; no algorithm change. [Evidence and patch](../../../objections/4.3/README.md). 4.2 was subsequently implemented in m97.

- 4.2 and 4.4: m97 implements consistent wave eligibility at alpha=1 and the applied cumulative-offset recurrence. Paper updated. Counterexamples and the 4.1 regression pass. Table 8 points and first paired draw checked, but full interval rerun remains pending; frozen m94–m96 preserved. [4.2](../../../objections/4.2/README.md) · [4.4](../../../objections/4.4/README.md).

- 4.5: m98 removes the first-window seed restriction and attempts the unchanged pooled optimization from an alternative initialization. Paper updated; regression and Table 8 spot checks pass. [Evidence](../../../objections/4.5/README.md).

- 4.6: m99 integrates ordinary person-bootstrap reporting, explicit conditional analytical columns and fit flags. Default 999 draws; end-to-end 99-draw positive controls verified against independent array calculations, and overlapping age-shift smoke check passed. Full 999-draw final pipeline validation pending. [Evidence](../../../objections/4.6/README.md).

- 4.7: m100 clears identified alpha, defaults to module-off Czech primary, retains explicit forced 1.18 sensitivity, discloses historical quiet-window rationale and updates Table 8 with both completed 999-draw replays. No quiet-window dates changed. [Evidence](../../../objections/4.7/README.md).

- 4.8: m101 clarifies the positive-control validation scope under baseline misspecification; no experiment or pipeline change. [Evidence](../../../objections/4.8/README.md).

- 4.9: m102 aligns the workflow figure and manuscript with the implementation, corrects the historical Table 8 source, qualifies the proportional-history identity and discloses the numerical cap. No estimator change. [Evidence](../../../objections/4.9/README.md).
