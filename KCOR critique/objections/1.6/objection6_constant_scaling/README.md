# Objection 6: what a constant hazard multiplier does to fitted frailty

This checks the manuscript's §2.4.4 statement that multiplying the **observed population hazard** by a constant does not alter curvature-based identification of the frailty parameter. The manuscript explicitly writes `h_observed(t) = c × h_true(t)`. The conclusion depends on whether the multiplier is applied **after** aggregation over frailty or to **each individual's hazard before** aggregation. Those are different operations.

## 1. Multiply the observed population hazard

With fixed Gompertz slope `γ`, baseline scale `k`, and gamma-frailty variance `θ`, the paper's observed hazard is

`h(t;k,θ) = k exp(γt) / [1 + θ(k/γ)(exp(γt)-1)]`.

If the observed hazard itself is multiplied by a positive constant `c`, direct substitution gives the exact identity

`c h(t;k,θ) = h(t; c k, θ/c)`.

This is not merely an approximation. Another way to see it is

`1/h(t;k,θ) = θ/γ + (1/k - θ/γ) exp(-γt)`.

Multiplying `h` by `c` divides both coefficients of this line by `c`. Therefore the model parameter recovered from the transformed observed curve is **`θ/c`**, while its fitted baseline scale is **`c k`**, assuming the paper's joint fit has enough information to identify both parameters. For example, with `θ=4`, doubling the entire observed hazard gives `θ*=2`, not 4. Multiplying by one half gives `θ*=8`.

This transformation does **not** destroy parametric identifiability: the scaled curve still belongs to the same two-parameter family, and its new parameter pair can still be identified from an informative interval with fixed `γ`. The distinction is between **being able to identify a parameter** and **recovering the same numerical parameter value after scaling**. The independent review interpreted the manuscript as claiming the latter; the manuscript's phrase “does not alter curvature-based identification” could instead mean only the former. Its explicit observed-hazard formula and “unchanged curvature” rationale make the intended claim unclear, so precise wording is warranted even under that narrower reading.

The observed cumulative hazard also becomes `H_obs*(t)=c H_obs(t)`. Using the correspondingly refitted `θ*=θ/c` in the paper's *exact* gamma inversion yields `H_neutral*(t)=c H_neutral(t)`. Thus a stable final contrast can coexist with a changed fitted frailty parameter. If the multiplier is common to both cohorts, it cancels from the unanchored ratio. If each cohort has its own **time-constant** multiplier, it also cancels from a ratio normalized to an earlier reference time. These ratio statements assume exact refitting and the same multipliers at the reference and target times; they do not establish invariance to time-varying effects or model violations.

## 2. Multiply every individual's hazard before aggregation

Suppose instead the individual model starts with `h_i(t)=z_i k exp(γt)` and the same constant `c` multiplies every person's hazard. The new individual model has baseline scale `c k` and the **same frailty distribution**, hence the same `θ`. Its marginal population hazard is `h(t;c k,θ)`, which is generally **not** `c h(t;k,θ)` after survival selection has begun. The larger hazard depletes high-frailty people faster, changing the shape of the observed curve. In this different mechanism, the paper's joint fit can recover unchanged `θ` under the model.

## Verdict and scope

For the paper's explicit parenthetical operation `h_observed*=c h_observed`, the **numerical frailty parameter changes to `θ/c`**. If “identification” means only that the parameter remains mathematically identifiable, that part can survive; it must not be read as invariance of the fitted value. An unchanged `θ` is instead appropriate for a constant multiplier of the *individual-level baseline hazard* before frailty marginalization, under the gamma-frailty assumptions. The sentence that a scaled observed cumulative hazard has unchanged “curvature” is also imprecise: its second derivative scales by `c`, although zero/nonzero status and some dimensionless shape measures are preserved. Those shape properties alone do not hold the numerical value of `θ` fixed.

This result concerns the mathematical model and the manuscript's **joint** `k,θ` fit. The current production fitter reads an early baseline rate and holds it fixed; its finite-sample behavior under a multiplier is a separate implementation question. The small [arithmetic check](check.py) and [saved result](results.json) verify the exact identities to floating-point precision. No KCOR source, earlier result, or manuscript file was changed.

**Suggested manuscript correction:** distinguish a multiplier of individual hazards from a multiplier of the already observed marginal hazard. For the latter, state `k*=ck, θ*=θ/c` under the Gompertz–gamma model. If the intended claim is merely preserved *identifiability*, say so explicitly. Discuss invariance of the final anchored contrast separately from parameter invariance.
