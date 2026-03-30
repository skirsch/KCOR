

### **3.4 NPH exponent estimation**

We evaluated estimation of the NPH exponent (\alpha) using a staged validation framework consisting of synthetic recovery, pooled empirical estimation, and diagnostic sensitivity analyses. Throughout, (\alpha) is treated as a parameter of the working model describing how excess hazard scales with latent frailty, rather than as a causal or biological quantity.

---

#### **3.4.1 Synthetic validation**

We first assessed whether the estimation procedure can recover known values of (\alpha) under controlled conditions. Synthetic cohorts were generated under the gamma-frailty working model with specified (\alpha), varying frailty variance and noise structure.

Figure X shows estimated versus true (\alpha) under two regimes. In the baseline synthetic setting, both the pairwise and collapse estimators recover (\alpha) with low bias across the tested range. Under heteroskedastic noise, recovery remains directionally correct but exhibits increased dispersion, reflecting reduced information rather than structural failure.

These results confirm that the estimation procedure is capable of recovering (\alpha) when the working model holds and sufficient cross-cohort variation is present.

---

#### **3.4.2 Pooled Czech analysis**

We next applied the same estimation procedure to the Czech primary analysis using pooled cohorts over the main wave period.

Figure X shows the corresponding objective functions for the pairwise and collapse estimators. Both exhibit interior minima in the vicinity of (\alpha \approx 1.18\text{–}1.19), and the two estimators are numerically close (pairwise (1.190), collapse (1.185)), indicating a consistent preferred region under the working model.

However, the objective functions are shallow, with normalized curvature metrics well below the prespecified identifiability threshold. In particular, the curvature values (approximately (7.8\times10^{-4}) and (8.4\times10^{-4})) indicate that the data provide limited information to localize (\alpha) precisely.

Accordingly, while the pooled analysis suggests a region of plausible (\alpha) values, it does not satisfy the criteria required to report (\alpha) as identified.

---

#### **3.4.3 Diagnostic assessment**

We evaluated identifiability using bootstrap, leave-one-out, and sensitivity analyses.

Bootstrap resampling shows substantial dispersion and strong boundary-seeking behavior, with a large fraction of resampled estimates accumulating at the limits of the search grid. Leave-one-out analysis reveals instability, with several cohort omissions producing large shifts in the estimated optimum.

Sensitivity analyses across anchor choice, excess-hazard definition, and time segmentation further demonstrate inconsistency. In particular, certain configurations produce boundary-seeking optima or materially different estimates, and segmented analyses show reduced agreement relative to the pooled specification.

Taken together, these diagnostics indicate that the apparent pooled optimum does not reflect a stable, well-identified parameter.

---

#### **3.4.4 Summary of identification status**

Across all prespecified criteria—objective curvature, estimator agreement, stability under resampling, and robustness to specification—the Czech pooled analysis fails to meet the threshold required for identification.

We therefore report (\alpha) as **not identified** for this dataset. The observed region near (\alpha \approx 1.18\text{–}1.19) should be interpreted as a model-dependent preference rather than a reliably estimated parameter.

This outcome is consistent with the role of the identifiability diagnostics: the procedure may locate a shallow optimum even when the data do not contain sufficient information to support a stable estimate, and in such cases the appropriate conclusion is non-identification rather than point estimation.

