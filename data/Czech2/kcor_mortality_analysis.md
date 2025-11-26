
# KCOR Mortality Analysis Pipeline  
## Instructions for Steve's AI Coding Engine

Goal:  
Estimate whether **vaccination increased or decreased the probability of death** for recipients, using the Czech event-level dataset and Steve's KCOR methodology.

This pipeline:

1. Builds **fixed cohorts** based on vaccination status at a chosen **enrollment date**.
2. Constructs **person-month survival data** from that date forward.
3. Computes **cohort-specific hazard curves**.
4. Applies **KCOR slope-normalization** to remove age/health structure bias (Gompertz slope).
5. Computes **adjusted cumulative hazards** and **KCOR ratios** of vaccinated vs unvaccinated.

All code is written in Python with pandas and (optionally) statsmodels.

---

## 0. Assumptions and Column Names

Input file: `foo.csv` (same as before)

Columns (lowercased):

- `id_zeny` : person ID
- `rok_narozeni` : birth year
- `udalost` : event type  
  - values include `umrti`, `covid ockovani`, `covid onemocneni`, `porod`, etc.
- `rok_udalosti` : year of event
- `mesic_udalosti` : month of event (1–12)
- death ICD column: contains ICD-10 code for `umrti` events (e.g., contains "diag" in name)
- `covid_ocko_poradi_davky` : dose number for vaccination events
- `covid_ocko_typ` : vaccine type (CO01, CO02, etc.)
- `covid_onemoc_poradi_infekce` : infection number for infection events

You can adjust column names if they differ.

---

## 1. Setup and Loading

```python
import pandas as pd
import numpy as np

df = pd.read_csv("foo.csv", dtype=str)
df.columns = [c.strip().lower() for c in df.columns]

# Convert numeric fields
for c in ["rok_udalosti", "mesic_udalosti", "rok_narozeni", 
          "covid_ocko_poradi_davky", "covid_onemoc_poradi_infekce"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
```

---

## 2. Choose Enrollment Date and Build Baseline Cohorts

Pick an **enrollment date** (year, month).  
Typical choices: `2021-01`, `2021-07`, `2022-01`.

```python
ENROLL_YEAR  = 2021
ENROLL_MONTH = 7  # July 2021 as an example

def ym_to_index(year, month, base_year, base_month):
    return (year - base_year) * 12 + (month - base_month)

# Index of enrollment month = 0
ENROLL_INDEX = 0
```

### 2.1 Per-person vaccination history

```python
vax_events = df[df["udalost"].str.lower() == "covid ockovani"].copy()
vax_events["t_index"] = vax_events.apply(
    lambda r: ym_to_index(r["rok_udalosti"], r["mesic_udalosti"], ENROLL_YEAR, ENROLL_MONTH),
    axis=1
)
```

Keep only vaccinations **on or before enrollment** to define baseline status:

```python
vax_before = vax_events[vax_events["t_index"] <= 0].copy()

baseline_dose = (
    vax_before.groupby("id_zeny")["covid_ocko_poradi_davky"]
    .max()
    .rename("baseline_dose")
)
```

Create baseline cohort label:

```python
# Default unvaccinated (0 doses at baseline)
persons = df["id_zeny"].dropna().unique()
baseline = pd.DataFrame({"id_zeny": persons})

baseline = baseline.merge(baseline_dose, on="id_zeny", how="left")
baseline["baseline_dose"] = baseline["baseline_dose"].fillna(0).astype(int)

def label_cohort(dose):
    if dose == 0:
        return "dose0_unvaccinated"
    elif dose == 1:
        return "dose1"
    elif dose == 2:
        return "dose2"
    elif dose >= 3:
        return "dose3plus"
    # you can adjust if you want 3,4,5,... separately

baseline["cohort"] = baseline["baseline_dose"].apply(label_cohort)
```

---

## 3. Determine Death Time and Censor Time

```python
death_events = df[df["udalost"].str.lower() == "umrti"].copy()
death_events["t_index"] = death_events.apply(
    lambda r: ym_to_index(r["rok_udalosti"], r["mesic_udalosti"], ENROLL_YEAR, ENROLL_MONTH),
    axis=1
)

death_time = (
    death_events.groupby("id_zeny")["t_index"]
    .min()  # first (and only) death for each person
    .rename("death_t")
)

# Merge into baseline
baseline = baseline.merge(death_time, on="id_zeny", how="left")
```

### 3.1 Define follow-up window

Choose a follow-up horizon (in months).  
Example: 24 months after enrollment.

```python
MAX_FU_MONTHS = 24
```

For people who die before enrollment (death_t < 0), they are **not in the risk set**; exclude them:

```python
baseline = baseline[(baseline["death_t"].isna()) | (baseline["death_t"] >= 0)].copy()
```

---

## 4. Build Person-Month Table from Enrollment Onward

We want one row per person per month from `t=0` to `t=min(death_t, MAX_FU_MONTHS)`.

```python
rows = []

for _, row in baseline.iterrows():
    pid    = row["id_zeny"]
    cohort = row["cohort"]
    death_t = row["death_t"]  # may be NaN

    # last month observed for this person
    if np.isnan(death_t):
        last_t = MAX_FU_MONTHS
    else:
        last_t = min(int(death_t), MAX_FU_MONTHS)

    for t in range(0, last_t + 1):
        event = 0
        if not np.isnan(death_t) and int(death_t) == t:
            event = 1
        rows.append((pid, cohort, t, event))

pm = pd.DataFrame(rows, columns=["id_zeny", "cohort", "t", "event"])
```

At this point:

- `pm` is your **person-month** survival frame.
- `t` is months since enrollment.
- `event = 1` only in the death month, else 0.

---

## 5. Compute Cohort Hazard Curves

Hazard in month `t` for cohort `c`:

- **risk set**: number of persons alive at start of month `t`
- **events**: number of deaths in that month

```python
# Number of events per cohort and month
events = (
    pm.groupby(["cohort", "t"])["event"]
    .sum()
    .rename("deaths")
    .reset_index()
)

# Number at risk at start of each month:
# A person is at risk in month t if they have any row with t >= current t and event not yet occurred before t
# Easier: count unique persons who have any row at month t (by construction they are alive at start of t)
risk = (
    pm.groupby(["cohort", "t"])["id_zeny"]
    .nunique()
    .rename("at_risk")
    .reset_index()
)

haz = events.merge(risk, on=["cohort", "t"], how="right").fillna(0)
haz["hazard"] = haz["deaths"] / haz["at_risk"]
```

Now `haz` has:

- `cohort`
- `t`
- `deaths`
- `at_risk`
- `hazard` (per month)

You can save:

```python
haz.to_csv("kcor_hazard_raw.csv", index=False)
```

---

## 6. KCOR Slope Normalization (Gompertz Adjustment)

KCOR idea:  
Each cohort has a different underlying Gompertz slope (age/health structure).  
We remove this by:

1. Selecting a **quiet period** (no big COVID waves) in terms of `t` (e.g., months `[Q1, Q2]`).
2. Fitting a line to `log(hazard)` vs `t` in that window.
3. Removing cohort-specific slope so curves become comparable.

### 6.1 Choose Quiet Period

Example: months 3–10 after enrollment.

```python
QUIET_T_MIN = 3
QUIET_T_MAX = 10
```

### 6.2 Fit slope per cohort

You can use numpy or statsmodels. Here is a simple numpy version.

```python
import numpy as np

haz["log_hazard"] = np.log(haz["hazard"].replace(0, np.nan))

slopes = []

for cohort, g in haz[(haz["t"] >= QUIET_T_MIN) & (haz["t"] <= QUIET_T_MAX)].groupby("cohort"):
    g = g.dropna(subset=["log_hazard"])
    if len(g) < 2:
        continue
    x = g["t"].values
    y = g["log_hazard"].values
    # simple linear regression: y = a + b x
    b, a = np.polyfit(x, y, 1)
    slopes.append((cohort, a, b))

slopes_df = pd.DataFrame(slopes, columns=["cohort", "intercept", "slope"])
slopes_df.to_csv("kcor_slopes.csv", index=False)
```

### 6.3 Apply slope normalization

For each cohort `c`, with slope `b_c`, define:

- adjusted hazard: `haz_adj_c(t) = hazard_c(t) * exp(-b_c * t)`

This flattens the log-hazard trend (slope 0 in quiet period).

```python
haz = haz.merge(slopes_df[["cohort", "slope"]], on="cohort", how="left")

haz["hazard_adj"] = haz["hazard"] * np.exp(-haz["slope"] * haz["t"])
```

---

## 7. Compute Adjusted Cumulative Hazards (per Cohort)

```python
haz = haz.sort_values(["cohort", "t"])

haz["cum_hazard_adj"] = (
    haz.groupby("cohort")["hazard_adj"]
    .cumsum()
)
```

Now, for each cohort, `cum_hazard_adj(t)` is the KCOR-style **slope-normalized cumulative hazard**.

Save:

```python
haz.to_csv("kcor_hazard_adjusted.csv", index=False)
```

---

## 8. Compute KCOR Ratios (Vaccinated vs Unvaccinated)

Define which cohorts are “vaccinated” and which is the unvaccinated reference.

```python
# Unvaccinated cohort name:
REF_COHORT = "dose0_unvaccinated"

# Example vaccinated cohorts to compare:
VAX_COHORTS = ["dose1", "dose2", "dose3plus"]
```

Build a table of cumulative hazards at each t:

```python
ref = haz[haz["cohort"] == REF_COHORT][["t", "cum_hazard_adj"]].rename(
    columns={"cum_hazard_adj": "cum_ref"}
)

kcor_rows = []

for vc in VAX_COHORTS:
    sub = haz[haz["cohort"] == vc][["t", "cum_hazard_adj"]].rename(
        columns={"cum_hazard_adj": "cum_vax"}
    )
    merged = sub.merge(ref, on="t", how="inner")
    merged["cohort"] = vc
    merged["kcor_ratio"] = merged["cum_vax"] / merged["cum_ref"]
    kcor_rows.append(merged)

kcor = pd.concat(kcor_rows, ignore_index=True)
kcor.to_csv("kcor_ratios.csv", index=False)
```

Interpretation:

- `kcor_ratio > 1` → higher adjusted cumulative hazard (harm) in vaccinated cohort.
- `kcor_ratio < 1` → lower adjusted cumulative hazard (benefit).
- `kcor_ratio ≈ 1` → neutral effect.

Plot:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
for vc in VAX_COHORTS:
    sub = kcor[kcor["cohort"] == vc]
    plt.plot(sub["t"], sub["kcor_ratio"], label=vc)

plt.axhline(1.0, color="black", linestyle="--")
plt.xlabel("Months since enrollment")
plt.ylabel("KCOR adjusted cumulative hazard ratio")
plt.title("KCOR: Vaccinated vs Unvaccinated (slope-normalized)")
plt.legend()
plt.tight_layout()
plt.savefig("kcor_ratio_plot.png")
plt.show()
```

---

## 9. Sensitivity Checks

You should run the above KCOR pipeline under multiple choices to test robustness:

1. Different enrollment dates:  
   - `2021-01`, `2021-07`, `2022-01`.
2. Different quiet periods for slope estimation:  
   - 3–10 months, 6–15 months, etc.
3. Different follow-up horizons:  
   - 12 months vs 24 months.
4. Different cohort definitions:  
   - group 3+ doses together vs separate 3,4,5,6.

For each configuration, re-generate:

- `kcor_hazard_raw.csv`
- `kcor_hazard_adjusted.csv`
- `kcor_ratios.csv`
- `kcor_ratio_plot.png`

The key question across all runs:

- Do vaccinated cohorts consistently show KCOR ratios **above 1**, **below 1**, or close to 1?

This answers:

> Did vaccination increase or decrease the death risk for recipients, after controlling for Gompertz slope (age/health structure) using KCOR?

---

## 10. Optional: Stratify by Age

You can run the KCOR pipeline within age bands to reduce residual confounding:

```python
# Add age at enrollment
# Approximate: age at enrollment year
people_age = (
    df.groupby("id_zeny")[["rok_narozeni"]].first().reset_index()
)
people_age["age_enroll"] = ENROLL_YEAR - people_age["rok_narozeni"]

baseline = baseline.merge(people_age[["id_zeny", "age_enroll"]], on="id_zeny", how="left")

# Then subset baseline to 65–89, etc., and re-run steps 3–9 on that subset only.
```

---

## 11. Outputs Summary

The KCOR pipeline will produce:

- `kcor_hazard_raw.csv` – hazards per cohort and month.
- `kcor_slopes.csv` – Gompertz slopes per cohort (quiet period).
- `kcor_hazard_adjusted.csv` – slope-normalized hazards and cumulative hazards.
- `kcor_ratios.csv` – KCOR adjusted cumulative hazard ratios over time.
- `kcor_ratio_plot.png` – plots of KCOR over time.

These are the core artifacts you need to determine if vaccination increased or decreased mortality risk for the vaccinated, according to KCOR.

---

# End of File
