# KCOR Depletion-Mode Slope Normalization (for b < 0 cohorts)
## Specification (v5 – all time in weeks, full deployment-window fit)

This specification defines the slope-normalization procedure for cohorts whose log-hazard initially slopes downward due to depletion/selection. The method fits a depletion-shape curve over the *entire post-enrollment window* and uses it to flatten the cohort.

Goal:

> Under pure selection bias (no intervention effect), the normalized hazard must be flat, so KCOR = 1.

---

# 1. Deployment Window = Fit Window  
(all time expressed in weeks)

For any cohort requiring depletion-mode normalization (identified by **b_lin < 0** in the initial linear fit):

### Define the deployment window

- Enrollment week = cohort’s enrollment ISO week  
- slope7_end_ISO = final week used for KCOR analysis  
- Deployment (fit) window:

\[
\mathcal{W}^{\text{dep}} = \{\, w_{\text{enroll}},\ w_{\text{enroll}}+1,\ \dots,\ w_{\text{slope7\_end}} \,\}
\]

### Convert ISO weeks to continuous time axis

Define time in **weeks since enrollment**:

\[
s(w) = w - w_{\text{enroll}}
\]

Thus:

- \(s = 0\) at enrollment  
- \(s = S_{\max}\) at slope7_end_ISO  
- **No centering**  
- **No quiet-window search**  
- **The fit always uses the full deployment window**

This ensures a single consistent definition for both fitting and normalization.

---

# 2. Parameter Estimation Using Levenberg–Marquardt  
(non-linear least squares, all time in weeks)

We model the slope of log-hazard as an exponential relaxation:

### Instantaneous slope model

\[
b(s) = \frac{d}{ds}\log h(s)
     = k_{\infty} + (k_0 - k_{\infty})\,e^{-s/\tau}
\]

where:

- \(k_0\) = slope at enrollment (may be negative)  
- \(k_{\infty} > 0\) = long-run background slope  
- \(\tau > 0\) = depletion timescale in **weeks**

### Integrated log-hazard model

Integrating the slope curve gives the depletion-shape model:

\[
\log h(s)
= C
+ k_{\infty}s
+ (k_0 - k_{\infty})\,\tau\left(1 - e^{-s/\tau}\right)
\]

### Parameter fitting (required method)

Estimate parameters:

\[
\theta = (C,\; k_{\infty},\; k_0,\; \tau)
\]

using **nonlinear least squares (Levenberg–Marquardt)** applied to **all** \(s \in \mathcal{W}^{\text{dep}}\):

\[
\theta = 
\arg\min_\theta 
\sum_{s \in \mathcal{W}^{\text{dep}}}
\left[
\log h_{\text{obs}}(s) -
\bigl(C + k_{\infty}s + (k_{0}-k_{\infty})\tau(1-e^{-s/\tau})\bigr)
\right]^2
\]

This yields the cohort-specific depletion-shape parameters:

\[
(C_c,\; k_{\infty,c},\; k_{0,c},\; \tau_c)
\]

All parameters are defined strictly in **weeks**.

---

# 3. Slope Normalization Formula  
(applied to the entire post-enrollment window)

### Construct the fitted depletion curve

For all \(s \ge 0\):

\[
\widehat{\log h}(s)
= C_c
+ k_{\infty,c}s
+ (k_{0,c}-k_{\infty,c})\,\tau_c\left(1 - e^{-s/\tau_c}\right)
\]

### Normalized log-hazard

\[
\log h_{\text{norm}}(s)
= \log h(s) - \widehat{\log h}(s)
\]

### Equivalent hazard-scale expression

\[
h_{\text{norm}}(s)
= h(s)\,
\exp\!\Big[
- C_c
- k_{\infty,c}s
- (k_{0,c}-k_{\infty,c})\tau_c\bigl(1-e^{-s/\tau_c}\bigr)
\Big]
\]

### Expected behavior under pure selection bias

\[
h_{\text{norm}}(s) \approx \text{constant}
\]

i.e., the cohort becomes flat when no real intervention effect exists.

---

# 4. Centering Convention (no centering used)

For depletion-mode normalization:

- **Time origin is fixed at enrollment**:  
  \[
  s = w - w_{\text{enroll}}
  \]

- **No centering relative to the middle of the window** (unlike linear-mode KCOR)

This ensures:

- \(k_0\) captures the **true initial depletion slope**  
- The depletion curve is biologically interpretable  
- The same procedure applies uniformly across all depleted cohorts
