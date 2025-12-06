# KCOR Slope Normalization — Slope Determination Only (Horizontal Zero-Drift Method)

This document defines **only the slope-normalization step** used in KCOR — the process that identifies and removes the time-dependent drift in each cohort’s hazard curve.

No assumptions are made about:
- window selection  
- level adjustments  
- KCOR ratios  
- enrollment definitions  

This section covers **slope determination only**.

---

# 1. Objective

Each cohort \(c\) has a hazard time series:

\[
h_c(t).
\]

We want to isolate and remove the **time-dependent drift** in the log-hazard for that cohort over a chosen baseline window:

\[
T_{	ext{base}}.
\]

We do this by estimating the cohort’s **log-hazard slope** independently of all other cohorts.

The resulting slope parameter \(eta_c\) is then used to construct a **slope-neutralized** hazard series whose fitted log-hazard is horizontal (zero slope) over the baseline window.

---

# 2. Log-Hazard Slope Model

For each cohort \(c\), fit the model on the baseline window:

\[
\log h_c(t) pprox lpha_c + eta_c t,
\]

where:

- \(lpha_c\): intercept  
- \(eta_c\): slope (time-dependent drift)  
- \(arepsilon_t = \log h_c(t) - (lpha_c + eta_c t)\): residual

### Why include the residual term?

- Real hazard data are noisy and not exactly exponential.
- \(arepsilon_t\) ensures the regression is well-defined.
- It is **not** a parameter — it is the leftover error after fitting.

---

# 3. Slope Estimation

Estimate \(lpha_c\) and \(eta_c\) using ordinary least squares:

\[
(lpha_c, eta_c)
=
rg\min_{lpha,eta}
\sum_{t \in T_{	ext{base}}}
\left[
  \log h_c(t) - (lpha + eta t)
ight]^2.
\]

The fitted slope:

\[
eta_c,
\]

is the cohort's **time-drift parameter** over the baseline window.

---

# 4. Zero-Slope (Horizontal) Normalization

Choose a pivot time \(t_0\), typically the mean of \(t\) in the baseline window.

Remove the drift term from the log-hazard:

\[
\log 	ilde{h}_c(t)
=
\log h_c(t) - eta_c (t - t_0).
\]

Convert back to hazard space:

\[
	ilde{h}_c(t)
=
\exp[-eta_c (t - t_0)] \, h_c(t).
\]

### Properties

- Over the baseline window \(T_{	ext{base}}\),  
  \[
  \log 	ilde{h}_c(t) pprox 	ext{constant}
  \]
  → zero slope.
- The transformation:
  - removes **only** the time-dependent drift,
  - does **not** force matching to another cohort,
  - preserves cohort-specific levels.

---

# 5. Summary (Slope Normalization Only)

For each cohort \(c\):

1. Fit  
   \[
   \log h_c(t) pprox lpha_c + eta_c t
   \quad (t \in T_{	ext{base}}).
   \]
2. Extract drift slope \(eta_c\).
3. Apply slope-neutralization:  
   \[
   	ilde{h}_c(t) = e^{-eta_c (t - t_0)} \, h_c(t).
   \]
4. \(	ilde{h}_c(t)\) has zero log-slope on the baseline window.

This document defines **slope normalization only**.  
All other KCOR components are intentionally omitted here.
