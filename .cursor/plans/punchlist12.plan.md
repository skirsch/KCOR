# Punchlist 12 — Notation/Estimand Invariants Enforcement Plan with fixed math

## Scope and constraints

- **In-scope file**: [`documentation/preprint/paper.md`](documentation/preprint/paper.md)
- **Rules source of truth**: [`documentation/preprint/punchlist12_instructions.md`](documentation/preprint/punchlist12_instructions.md) and [`documentation/preprint/punchlist12_jlw_feedback.md`](documentation/preprint/punchlist12_jlw_feedback.md)
- **Hard constraints** (mechanical enforcement only):
- Do **not** change scientific meaning.
- Do **not** introduce new symbols or simplify notation.
- Do **not** inline composite expressions; use **display math** for tilde/hats/multi-subscripts.
- Do **not** convert display math into inline math.
- Include in-text figure references (e.g., “Figure 3 shows …”) when checking for notational consistency.
- Do not rename symbols to “clarify” meaning (e.g., do not replace $H_{0,d}$ with $H^{(0)}_d$ or similar).

## Phase 1 (required first): build plan / violation report (no edits)

Produce a structured list of violations with file locations in `paper.md`, grouped by invariant category. Each entry should include:

- **Location**: section heading + nearby unique text (and line numbers if available)
- **Current form**: the exact math/prose snippet
- **Violation type**: which rule(s) it breaks
- **Canonical target form**: exactly matching the canonical forms in `punchlist12_jlw_feedback.md`
- **Minimal mechanical fix**: what to replace, without altering surrounding prose unless required

### Violation categories to scan for (mechanically)

- **Cumulative hazard notation stability**
- Ensure the three distinct objects are consistently notated with explicit cohort index:

$$H_{\mathrm{obs},d}(t),\qquad H_{0,d}(t),\qquad \tilde{H}_{0,d}(t)$$

- Remove disallowed observational superscripts (e.g.,

$$H_d^{\mathrm{obs}}(t)$$).

- **Index placement invariance**
- Cohort index must stay attached to the same symbol in the same position everywhere.
- No “context-carries-index” usages.
- **Parameters vs. estimates discipline**
- Conceptual parameters:

$$\theta_d,\qquad k_d$$(no hats)

- Quiet-window fitted estimates:

$$\hat{\theta}_d,\qquad \hat{k}_d$$

- Never alternate hats/unhatted for the same fitted quantity; never drop cohort subscript.
- **Gamma-frailty identity canonicalization**
- Mixture identity must be exactly:

$$H_{\mathrm{obs},d}(t)=\frac{1}{\theta_d}\log\!\left(1+\theta_d\,H_{0,d}(t)\right)$$

- Inversion must be exactly:

$$H_{0,d}(t)=\frac{\exp\!\left(\theta_d\,H_{\mathrm{obs},d}(t)\right)-1}{\theta_d}$$

- Flag any algebraically-equivalent but visually different variants.
- **Quiet-window constant-baseline form**
- If assuming constant baseline over the fit window, enforce:

$$H_{0,d}(t)=k_d\,t$$

- Remove/flag auxiliary baseline-shape symbols introduced but unused.
- **Estimand vs framework naming**
- “KCOR” = framework; **estimand must be** $KCOR(t)$.
- Estimand definition must match:

$$KCOR(t)=\frac{\tilde{H}*{0,A}(t)}{\tilde{H}*{0,B}(t)}$$

- Flag any places where “KCOR” is used to mean the curve/value at time $t$.
- **Display math requirement for composite expressions**
- Any expression with hats/tilde/multi-subscripts (esp.

$$\tilde{H}_{0,d}(t)$$,$$\hat{\theta}_d$$, etc.) must be in **display math**.

- Flag inline occurrences that must be converted to display math (without changing the math).
- **Notational isomorphism within `paper.md`**
- Methods, figure captions, appendices sections inside `paper.md` must use the same canonical symbols (no shorthand like $H_0$ unless explicitly defined identically and used consistently).

## Phase 2 (after approval): apply fixes (minimal mechanical edits)

Once you approve the Phase 1 report, apply edits to [`documentation/preprint/paper.md`](documentation/preprint/paper.md) only:

- Replace non-canonical symbols/placements with the canonical ones.
- Ensure all required composite expressions are in display math and none are demoted from display to inline.
- Preserve all cohort subscripts and observational/baseline/depletion-neutralized distinctions.
- Keep prose changes to the minimum necessary to avoid notational ambiguity.