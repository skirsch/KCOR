# KCOR 6.0 – Sample Python Code (Cursor + Pandoc Safe)

This file provides **reference Python code** that implements KCOR 6.0 exactly as specified.
All math formatting follows the **Cursor + Pandoc–compatible standard**:

- Inline math: `$ … $`
- Display math: `$$ … $$`

No `\( … \)` or `\[ … \)` is used.

---

## Conceptual overview

We observe deaths $D_{d,t}$ and person-time $Y_{d,t}$ for each cohort (dose) $d$ in discrete time bins $t$ (typically weeks since enrollment).

Observed hazards are:

$$
h^{obs}_{d}(t) = \frac{D_{d,t}}{Y_{d,t}}.
$$

Observed cumulative hazards are:

$$
H^{obs}_{d}(t) = \sum_{s \le t} h^{obs}_{d}(s)\,\Delta t.
$$

Under gamma frailty with $g(t)=1$, the observable cumulative hazard satisfies:

$$
H^{obs}_{d}(t) = \frac{1}{\theta_d}\log\!\left(1 + \theta_d k_d t\right).
$$

The depletion-neutralized cumulative hazard is obtained by inversion:

$$
H_{0,d}(t) = \frac{e^{\theta_d H^{obs}_{d}(t)} - 1}{\theta_d}.
$$

KCOR is then computed as:

$$
\text{KCOR}(t) = \frac{H_{0,d_1}(t)}{H_{0,d_0}(t)}.
$$

---

## Inputs expected

A pandas DataFrame `df` with one row per cohort–time bin:

- `dose` : int (0, 1, 2, …)
- `iso_year` : int
- `iso_week` : int
- `t` : float (weeks since cohort enrollment)
- `deaths` : int ($D_{d,t}$)
- `person_time` : float ($Y_{d,t}$)

Cohorts must be **fixed at enrollment**.

---

## Quiet window utilities (ISO week)

```python
def iso_to_int(iso_year: int, iso_week: int) -> int:
    return iso_year * 100 + iso_week

QUIET_START = iso_to_int(2022, 24)
QUIET_END   = iso_to_int(2024, 16)

def in_quiet_window(row) -> bool:
    x = iso_to_int(int(row["iso_year"]), int(row["iso_week"]))
    return QUIET_START <= x <= QUIET_END
```

---

## Step 1: Compute observed hazard and cumulative hazard

```python
import numpy as np
import pandas as pd

def add_h_and_Hobs(df: pd.DataFrame, dt: float = 1.0) -> pd.DataFrame:
    df = df.copy()

    df["h_obs"] = df["deaths"] / df["person_time"].replace(0, np.nan)
    df["h_obs"] = df["h_obs"].fillna(0.0)

    df = df.sort_values(["dose", "t"])
    df["H_obs"] = df.groupby("dose")["h_obs"].cumsum() * dt

    return df
```

---

## Step 2: Fit $(k_d, \theta_d)$ on the quiet window

### Option A: Poisson MLE in hazard space

Assume:

$$
D_{d,t} \sim \text{Poisson}\!\left(Y_{d,t} h_d(t)\right),
\qquad
h_d(t) = \frac{k_d}{1 + \theta_d k_d t}.
$$

```python
from scipy.optimize import minimize

def nll_poisson(params, t, deaths, person_time):
    k, theta = params
    if k <= 0 or theta < 0:
        return np.inf
    h = k / (1.0 + theta * k * t)
    mu = person_time * h
    mu = np.clip(mu, 1e-30, None)
    return -np.sum(deaths * np.log(mu) - mu)

def fit_k_theta_poisson(df_d: pd.DataFrame):
    t = df_d["t"].to_numpy(float)
    deaths = df_d["deaths"].to_numpy(float)
    pt = df_d["person_time"].to_numpy(float)

    k0 = max(df_d["h_obs"].mean(), 1e-8)
    theta0 = 0.1

    res = minimize(
        nll_poisson,
        x0=[k0, theta0],
        args=(t, deaths, pt),
        method="L-BFGS-B",
        bounds=[(1e-12, None), (0.0, None)],
    )
    return res.x, res
```

---

### Option B: Direct fit in cumulative-hazard space (recommended)

Fit:

$$
H^{obs}(t) \approx \frac{1}{\theta}\log\!\left(1 + \theta k t\right).
$$

```python
from scipy.optimize import least_squares

def H_model(t, k, theta):
    if theta < 1e-10:
        return k * t
    return (1.0 / theta) * np.log1p(theta * k * t)

def residuals_cumhaz(params, t, H_obs):
    k, theta = params
    if k <= 0 or theta < 0:
        return 1e6 * np.ones_like(H_obs)
    return H_model(t, k, theta) - H_obs

def fit_k_theta_cumhaz(df_d: pd.DataFrame):
    t = df_d["t"].to_numpy(float)
    H_obs = df_d["H_obs"].to_numpy(float)

    k0 = max(np.polyfit(t, H_obs, 1)[0], 1e-8)
    theta0 = 0.1

    res = least_squares(
        residuals_cumhaz,
        x0=[k0, theta0],
        bounds=([1e-12, 0.0], [np.inf, np.inf]),
    )
    return res.x, res
```

---

## Step 3: Apply gamma-frailty inversion

```python
def invert_gamma_frailty(H_obs: np.ndarray, theta: float) -> np.ndarray:
    if theta < 1e-10:
        return H_obs.copy()
    return np.expm1(theta * H_obs) / theta

def add_H0(df: pd.DataFrame, params_by_dose: dict) -> pd.DataFrame:
    df = df.copy()
    out = []
    for dose, g in df.groupby("dose", sort=False):
        k, theta = params_by_dose[int(dose)]
        H0 = invert_gamma_frailty(g["H_obs"].to_numpy(float), theta)
        out.append(pd.Series(H0, index=g.index))
    df["H0"] = pd.concat(out).sort_index()
    return df
```

---

## Step 4: Compute KCOR

```python
def compute_kcor(df: pd.DataFrame, ref_dose: int, tgt_dose: int) -> pd.DataFrame:
    ref = df[df["dose"] == ref_dose][["t", "H0"]].rename(columns={"H0": "H0_ref"})
    tgt = df[df["dose"] == tgt_dose][["t", "H0"]].rename(columns={"H0": "H0_tgt"})

    merged = pd.merge(tgt, ref, on="t", how="inner").sort_values("t")
    merged["KCOR"] = merged["H0_tgt"] / merged["H0_ref"].replace(0, np.nan)
    return merged
```

---

## End-to-end example

```python
df2 = add_h_and_Hobs(df)

df_quiet = df2[df2.apply(in_quiet_window, axis=1)]

params_by_dose = {}
for dose, g in df_quiet.groupby("dose"):
    (k, theta), _ = fit_k_theta_cumhaz(g)
    params_by_dose[int(dose)] = (float(k), float(theta))

df3 = add_H0(df2, params_by_dose)

kcor_01 = compute_kcor(df3, ref_dose=0, tgt_dose=1)
kcor_02 = compute_kcor(df3, ref_dose=0, tgt_dose=2)
```

---

## Practical notes

- Fit parameters only on **quiet windows**
- Always inspect $(k_d, \theta_d)$ stability across window choices
- Use age stratification + standardization as a robustness check
- KCOR compares **cumulative outcomes**, not instantaneous hazards

This file is **Cursor-preview safe** and **Pandoc → Word safe**.
