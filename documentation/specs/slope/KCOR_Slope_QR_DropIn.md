# KCOR Slope Normalization – Quantile Regression Drop-in for OLS

This note describes how to replace the **OLS-based slope fit** used in KCOR slope normalization with a **quantile-regression-based** slope fit, while keeping the rest of the pipeline unchanged.

We only address the **slope determination** step (how to get \(\beta_c\)). Everything else (using \(\beta_c\) to neutralize slope, KCOR ratios, window selection, etc.) stays the same.

---

## 1. OLS-based slope (current approach)

On the baseline window \(T_{\text{base}}\), for cohort \(c\):

- Time indices: \(t \in T_{\text{base}}\)  
- Hazards: \(h_c(t)\)  
- Log-hazards: \(y_t = \log h_c(t)\)  
- (Optional) Centered time:
  \[
  x_t = t - \bar{t}, \quad
  \bar{t} = \frac{1}{|T_{\text{base}}|} \sum_{t \in T_{\text{base}}} t
  \]

**OLS model:**

\[
y_t \approx \alpha_c + \beta_c x_t.
\]

**OLS estimate:**

\[
(\hat{\alpha}_c, \hat{\beta}_c)
=
\arg\min_{\alpha,\beta}
\sum_{t \in T_{\text{base}}}
\bigl[ y_t - (\alpha + \beta x_t) \bigr]^2.
\]

Use \(\hat{\beta}_c\) as the slope for slope neutralization:

\[
\tilde{h}_c(t)
=
\exp\bigl(-\hat{\beta}_c x_t\bigr) \, h_c(t).
\]

### Python (OLS) example

```python
import numpy as np
import statsmodels.api as sm

# Inputs:
# t_base: 1D numpy array of time indices on the baseline window (e.g., weeks)
# h_base: 1D numpy array of hazards for cohort c on the same window

x = t_base - np.mean(t_base)      # center time
y = np.log(h_base)                # log-hazard

X = sm.add_constant(x)            # [1, x_t]
ols_model = sm.OLS(y, X)
ols_res   = ols_model.fit()

alpha_hat_ols, beta_hat_ols = ols_res.params  # intercept, slope

# Example slope-neutralized hazards for a full time grid t_full, h_full:
x_full   = t_full - np.mean(t_base)          # center using same baseline mean
h_tilde  = np.exp(-beta_hat_ols * x_full) * h_full
```

---

## 2. Quantile-regression-based slope (replacement)

The idea is to estimate a **baseline slope** using a **quantile regression** fit instead of OLS, so that the slope reflects the *lower envelope* (baseline) rather than being pulled by peaks/waves.

We use the same setup:

- \(y_t = \log h_c(t)\)  
- \(x_t = t - \bar{t}\)

**Quantile regression model (level \(\tau\)):**

\[
y_t \approx \alpha_c^{(\tau)} + \beta_c^{(\tau)} x_t.
\]

**Quantile regression estimate:**

\[
(\alpha_c^{(\tau)}, \beta_c^{(\tau)})
=
\arg\min_{\alpha,\beta}
\sum_{t\in T_{\text{base}}}
\rho_\tau\bigl( y_t - (\alpha + \beta x_t) \bigr),
\]

where the check loss is:

\[
\rho_\tau(u) =
\begin{cases}
\tau u & \text{if } u \ge 0, \\
(\tau - 1)u & \text{if } u < 0.
\end{cases}
\]

Typical choices:

- \(\tau = 0.25\) (25th percentile baseline)  
- or \(\tau\) in the 0.2–0.3 range

Use \(\hat{\beta}_c^{(\tau)}\) as the slope.

**Slope-neutralization with quantile slope:**

\[
\tilde{h}_c(t)
=
\exp\bigl(-\hat{\beta}_c^{(\tau)} x_t\bigr) \, h_c(t).
\]

Note that this is identical in form to the OLS version; you are only swapping how \(\beta_c\) is estimated.

---

## 3. Python: Drop-in replacement using `statsmodels` QuantReg

Below is a drop-in replacement for the OLS block above, using `statsmodels.api.QuantReg` to estimate \(\beta_c\) by quantile regression.

```python
import numpy as np
import statsmodels.api as sm

# Inputs:
# t_base: 1D numpy array of time indices on the baseline window (e.g., weeks)
# h_base: 1D numpy array of hazards for cohort c on the same window

x = t_base - np.mean(t_base)      # center time
y = np.log(h_base)                # log-hazard

X = sm.add_constant(x)            # [1, x_t]

tau = 0.25  # quantile level (e.g., 0.25 for 25th percentile baseline)

qr_model = sm.QuantReg(y, X)
qr_res   = qr_model.fit(q=tau)

alpha_hat_qr, beta_hat_qr = qr_res.params  # intercept, slope

# Example slope-neutralized hazards for a full time grid t_full, h_full:
x_full  = t_full - np.mean(t_base)              # center using same baseline mean
h_tilde = np.exp(-beta_hat_qr * x_full) * h_full
```

In the KCOR code:

- Wherever you previously computed `beta_hat_ols` using OLS,
- You can substitute `beta_hat_qr` from the quantile regression block above.
- All downstream logic (slope neutralization, cumulative hazards, KCOR ratios) remains unchanged.
