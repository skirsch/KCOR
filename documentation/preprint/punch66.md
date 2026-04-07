# update NPH to match implementation

Yes. I’d split the current NPH material into two distinct subsections:

1. **how (\alpha) is identified**
2. **how the NPH correction is actually applied once (\alpha) is available**

Right now the paper mixes those ideas. In the current draft, §2.7.1 frames the optional NPH module and says (\alpha) is estimated from cross-cohort structure, while §2.7.2 gives the pairwise and collapse estimators, but the manuscript does not cleanly separate identification from the actual post-inversion correction now implemented in the code.  The current text also emphasizes that (\alpha) is not separately identifiable from cohort-specific multiplicative effects in minimal aggregated data, which is exactly why you want to say that VE must come from outside the KCOR core. 

I’d recommend this structure:

### Proposed section structure

* **2.7.1 Optional NPH exponent model**
  Keep this as the conceptual introduction.

* **2.7.2 Identification of (\alpha) given an externally estimated VE**
  New subsection focused only on identification logic.

* **2.7.3 Application of the NPH correction once (\alpha) is specified**
  New subsection focused only on the correction formula and ordering.

* **2.7.4 Practical estimation workflow**
  Short summary list if you still want a procedure checklist.

That solves the main clarity problem.

Here is manuscript-ready text you can drop in.

---

### Replacement for current §2.7.2

#### 2.7.2 Identification of $\alpha$ given an externally estimated VE

Under minimal aggregated data, the NPH exponent $\alpha$ is not separately identifiable from cohort-specific multiplicative intervention effects if both are left free. Accordingly, when the optional NPH module is used to identify $\alpha$, the intervention effect scale must be supplied from outside the KCOR core, for example from external clinical evidence, randomized data, or other prespecified analyses designed to estimate VE under separate assumptions. KCOR does not itself identify VE from the same aggregated wave-period contrasts used to identify $\alpha$. This separation is important because otherwise frailty-dependent amplification and intervention-associated hazard reduction can generate similar cross-cohort excess-hazard ratios, producing flat objectives and boundary-seeking estimates. 

Let $x$ denote the externally estimated vaccine efficacy against the relevant wave-period hazard, so that the frailty-neutral excess hazard in the protected cohort is assumed to satisfy

$$
\Delta h_A^{\mathrm{true}}(t) = (1-x),\Delta h_B^{\mathrm{true}}(t),
$$

where $A$ and $B$ denote two cohorts being compared. Under the gamma-frailty working model, the observed excess hazard satisfies

$$
\Delta h_d^{\mathrm{obs}}(t)=\Delta h_d^{\mathrm{true}}(t),F_d(t;\alpha),
\qquad
F_d(t;\alpha)=E_d[z^\alpha\mid t].
$$

Therefore,

$$
\frac{\Delta h_A^{\mathrm{obs}}(t)}{\Delta h_B^{\mathrm{obs}}(t)}
=================================================================

(1-x),
\frac{F_A(t;\alpha)}{F_B(t;\alpha)}.
$$

This relation identifies $\alpha$, if at all, from the mismatch between the observed cross-cohort excess-hazard ratio and the ratio predicted from the externally supplied VE together with the cohort-specific depletion geometry. Equivalently, $\alpha$ is chosen so that the model-implied amplification ratio matches the observed excess ratio after adjustment for the prespecified VE scale.

Operationally, this can be implemented with pairwise or pooled objective functions analogous to those used below, but with the VE term treated as fixed external input rather than estimated jointly from the same data. If diagnostics indicate weak localization, estimator disagreement, or boundary-seeking behavior, $\alpha$ is treated as not identified.

---

### New subsection after that

#### 2.7.3 Application of the NPH correction once $\alpha$ is specified

Once $\alpha$ has been specified or identified, the NPH correction is applied after gamma-frailty inversion in cumulative-hazard space. This ordering is important: the correction is applied to the frailty-neutral cumulative hazard rather than to the raw observed hazard, so that the wave-period excess is defined relative to a frailty-neutral baseline path.

Let $\tilde H_{0,d}(t)$ denote the frailty-neutral cumulative hazard after inversion. During prespecified NPH periods, define a no-wave frailty-neutral reference path using the fitted Gompertz baseline:

$$
H_{\mathrm{ref},d}(t)
=====================

\tilde H_{0,d}(t_{\mathrm{wave\ start}})
+
\int_{t_{\mathrm{wave\ start}}}^{t} h_{\mathrm{base},d}(s),ds,
$$

where $h_{\mathrm{base},d}(s)=\hat k_d e^{\gamma s}$ is the fitted cohort-specific Gompertz baseline hazard on rebased time.

The wave-induced excess in cumulative-hazard space is then

$$
\Delta H_{\mathrm{wave},d}(t)=\tilde H_{0,d}(t)-H_{\mathrm{ref},d}(t).
$$

Only positive excess above this baseline path is corrected. The NPH-corrected cumulative hazard is

$$
H_{\mathrm{corr},d}(t)
======================

H_{\mathrm{ref},d}(t)
+
\frac{\Delta H_{\mathrm{wave},d}(t)}{F_d(t;\alpha)}
\qquad \text{for } \Delta H_{\mathrm{wave},d}(t)>0,
$$

and is left unchanged when $\Delta H_{\mathrm{wave},d}(t)\le 0$. For weeks after the last in-window date $t_{\mathrm{end},d}$, wave-excess rescaling is not applied; the implementation continues $\tilde H_{0,d}(t)$ with a constant level shift so that $H_{\mathrm{corr},d}$ is continuous at $t_{\mathrm{end},d}$ and post-window weekly increments match $\tilde H_{0,d}$ (same as §2.7.3 in `paper.md`).

Weekly corrected hazard increments are then obtained by differencing the corrected cumulative hazard, and the downstream KCOR pipeline proceeds unchanged on that corrected cumulative-hazard scale. Thus, the optional NPH module modifies only the wave-period excess above the fitted frailty-neutral baseline path; it does not replace the core KCOR estimator.

---

### Then shorten current §2.7.3 into a workflow subsection

You can turn the current checklist into:

#### 2.7.4 Practical workflow for the optional NPH module

1. Estimate cohort-specific depletion geometry from prespecified quiet windows to obtain $\hat\theta_{0,d}$ and $\hat k_d$.
2. Obtain or prespecify an external estimate of VE for the relevant wave-period hazard.
3. Use the cross-cohort excess-hazard relation together with the external VE scale to identify $\alpha$, if diagnostics support identification.
4. Apply gamma-frailty inversion to obtain frailty-neutral cumulative hazards.
5. During the NPH window, construct the Gompertz no-wave reference path and correct only the positive wave excess using $F_d(t;\alpha)$.
6. Continue with the standard KCOR pipeline on the corrected cumulative-hazard scale.
7. If identification diagnostics fail, report $\alpha$ as not identified and treat the NPH module as inactive.

---

### One sentence to add earlier in §2.7.1

After the sentence saying the module is optional, add:

> Because $\alpha$ is not separately identifiable from cohort-specific multiplicative intervention effects in minimal aggregated data, application of this module requires an externally estimated VE or other prespecified intervention-effect input not learned from the same KCOR wave-period contrasts. 

---

### One sentence to fix elsewhere

In §2.5, the paper currently says the optional NPH preprocessing “is applied prior to inversion to preserve the validity of the gamma-frailty mapping.” That is now outdated relative to the corrected implementation logic.  You should revise that sentence so it says the optional NPH correction is applied **after** inversion in cumulative-hazard space, relative to the fitted Gompertz baseline path.

So yes: I would make this change now, because the current text still reflects the older framing more than the corrected method.
