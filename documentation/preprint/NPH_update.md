## updating the paper for the symmetric NPH correction

Yes — **if the symmetric formulation survives the sandbox and becomes your intended NPH model, then the paper should absolutely be updated to reflect it.**

Right now the manuscript’s NPH language is written at the level of a **shared wave amplitude times a cohort-specific frailty moment**, which is compatible with a symmetric correction idea. In Methods §2.7.1–§2.7.2, the paper already frames excess hazard as
(h_d(t)=h_{0,d}(t)+A(t)E_d[z^\alpha\mid t]), with identification coming from cross-cohort relative structure rather than absolute levels.  That is the right conceptual foundation for a symmetric neutralization.

What would need to change is the **operational description**.

## What to change in the paper if symmetric NPH wins

You should revise the text so it says, in substance:

* NPH adjustment is applied to **all cohorts during the shock window**.
* The correction removes the cohort-specific amplification term (F_d(t;\alpha)=E[z^\alpha\mid t]) from each cohort’s excess hazard.
* Dose 0 may still be used to help define a reference or excess component, but it is **not held fixed as an uncorrected anchor**.
* The NPH module is a **symmetric neutralization step**, not a one-sided adjustment to non-reference cohorts.

## Where to update

### 1. Methods §2.7.1

This is the most important place.
Right now this is where you define the NPH model and what (\alpha) means. If symmetric correction is the final design, this section should explicitly say that neutralization divides out or otherwise removes (F_d(t;\alpha)) **for each cohort** during the NPH period, rather than treating one cohort as fixed. 

### 2. Methods §2.7.2

This section should then describe estimation and application in sequence:

1. estimate (\alpha) from cross-cohort structure in the shock window,
2. compute cohort-specific (F_d(t;\alpha)),
3. neutralize excess hazard for **each** cohort,
4. pass the neutralized hazard stream into the ordinary KCOR pipeline.

That stepwise description is especially important because right now the paper gives the ingredients, but not quite the full operational sequence.

### 3. Section 3.4

If you keep NPH in the paper, the results section should say clearly whether the reported sandbox results are for:

* estimator-only alpha fitting, or
* full symmetric neutralization downstream.

If you later show downstream coherence gains from the symmetric correction, that belongs here too.

### 4. Limitations

You should add one sentence saying that even under symmetric correction, interpretation remains conditional on:

* correct shock-window specification,
* adequate excess-hazard definition,
* and identifiability of (\alpha).

## What not to change unless justified

Do **not** rewrite the paper yet if this is still just a conceptual fix and not sandbox-validated.

The right order is:

1. sandbox symmetric correction,
2. compare against anchored/reference version,
3. verify it improves coherence and behaves sensibly,
4. then update manuscript language.

## Best way to phrase it in the paper

Something like:

> During prespecified NPH periods, excess hazard is modeled as a common wave component multiplied by a cohort-specific frailty moment (F_d(t;\alpha)). When the optional NPH module is applied, neutralization is performed symmetrically across cohorts by removing this cohort-specific amplification term from each cohort’s excess hazard prior to cumulative-hazard accumulation and inversion.

That would be the key sentence.

## Bottom line

Yes. If symmetric NPH is the formulation you keep, the paper should be updated to match it. But I would wait until the sandbox confirms that this is the version you actually want.

If you want, I can draft the exact replacement text for §2.7.1–§2.7.2.
