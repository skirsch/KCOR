
# ICD-10 Cause of Death Analysis Pipeline  
## Instructions for Steve's AI Coding Engine

This document contains **all steps, logic, and code specifications** needed for an automated pipeline that analyzes cause‑of‑death differences between **vaccinated** and **unvaccinated** individuals using the Czech event‑level dataset.  
Everything here is deterministic and ready to implement.

---

# 1. Load and Normalize the Dataset

```python
import pandas as pd

df = pd.read_csv("data.csv", dtype=str)
df.columns = [c.strip().lower() for c in df.columns]

num_cols = [
    "id_zeny", "rok_udalosti", "mesic_udalosti",
    "covid_ocko_poradi_davky", "covid_onemoc_poradi_infekce"
]

for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
```

---

# 2. Extract Death Records + ICD Codes

```python
death_df = df[df["udalost"].str.lower() == "umrti"].copy()
icd_col = [c for c in death_df.columns if "diag" in c][0]
death_df["icd"] = death_df[icd_col].str.strip()
```

---

# 3. Classify Vaccination Status

```python
vaccinated_ids = set(
    df[df["udalost"].str.lower() == "covid ockovani"]["id_zeny"].unique()
)

death_df["vaccinated"] = death_df["id_zeny"].isin(vaccinated_ids)
```

---

# 4. Build ICD Histograms

```python
vaccinated_counts = (
    death_df[death_df["vaccinated"]]
    .groupby("icd").size().sort_values(ascending=False)
)

unvaccinated_counts = (
    death_df[~death_df["vaccinated"]]
    .groupby("icd").size().sort_values(ascending=False)
)
```

Normalize:

```python
vaccinated_dist = vaccinated_counts / vaccinated_counts.sum()
unvaccinated_dist = unvaccinated_counts / unvaccinated_counts.sum()
```

---

# 5. Compare Distributions + Output Table

```python
comparison = pd.DataFrame({
    "vaccinated_pct": vaccinated_dist,
    "unvaccinated_pct": unvaccinated_dist
}).fillna(0)

comparison["difference"] = (
    comparison["vaccinated_pct"] - comparison["unvaccinated_pct"]
)

comparison.sort_values("difference", ascending=False, inplace=True)
comparison.to_csv("icd_comparison.csv")
```

---

# 6. Plot Difference Chart

```python
import matplotlib.pyplot as plt

top_n = 25
top_icds = comparison.abs().sort_values("difference", ascending=False).head(top_n)

plt.figure(figsize=(14,10))
plt.barh(top_icds.index, top_icds["difference"],
         color=["red" if x>0 else "blue" for x in top_icds["difference"]])
plt.xlabel("Vaccinated % minus Unvaccinated %")
plt.title("Difference in Cause-of-Death Distribution (ICD-10)")
plt.tight_layout()
plt.savefig("icd_difference_plot.png")
plt.show()
```

---

# 7. Age-Stratified ICD Comparison

## 7.1 Compute Age at Death

```python
death_df["age"] = death_df["rok_udalosti"] - death_df["rok_narozeni"]
```

## 7.2 Define Age Bins

```python
bins = [0,40,60,70,80,90,200]
labels = ["0-39","40-59","60-69","70-79","80-89","90+"]

death_df["age_group"] = pd.cut(death_df["age"], bins=bins, labels=labels, right=False)
```

## 7.3 Compute ICD Distribution Per Age Group

```python
results = {}

for ag in labels:
    sub = death_df[death_df["age_group"] == ag]

    vacc = (sub[sub["vaccinated"]].groupby("icd").size() /
            sub[sub["vaccinated"]].shape[0])
    unvacc = (sub[~sub["vaccinated"]].groupby("icd").size() /
              sub[~sub["vaccinated"]].shape[0])

    combined = pd.DataFrame({
        "vacc_pct": vacc,
        "unvacc_pct": unvacc
    }).fillna(0)

    combined["diff"] = combined["vacc_pct"] - combined["unvacc_pct"]
    combined.sort_values("diff", ascending=False, inplace=True)

    results[ag] = combined
    combined.to_csv(f"icd_agegroup_{ag}.csv")
```

---

# 8. Time-Since-Last-Dose → ICD Distribution

## 8.1 Extract Vaccine Events Per Person

```python
vax_events = df[df["udalost"].str.lower() == "covid ockovani"][
    ["id_zeny","rok_udalosti","mesic_udalosti","covid_ocko_poradi_davky"]
].sort_values(["id_zeny","rok_udalosti","mesic_udalosti","covid_ocko_poradi_davky"])
```

## 8.2 Get Last Dose for Each Person

```python
last_vax = vax_events.groupby("id_zeny").last().reset_index()
```

## 8.3 Compute Time Since Last Dose to Death

```python
death_df = death_df.merge(last_vax, on="id_zeny", how="left", suffixes=("","_vax"))

death_df["months_since_dose"] = (
    (death_df["rok_udalosti"] - death_df["rok_udalosti_vax"]) * 12 +
    (death_df["mesic_udalosti"] - death_df["mesic_udalosti_vax"])
)
```

## 8.4 Bin Time Since Dose

```python
bins = [-999,0,2,4,7,13,1000]
labels = ["0-0","1-2","3-4","5-7","8-12","12+"]

death_df["post_vax_bin"] = pd.cut(
    death_df["months_since_dose"], bins=bins, labels=labels
)
```

## 8.5 ICD Distribution Per Bin

```python
for b in labels:
    sub = death_df[
        (death_df["post_vax_bin"] == b) & 
        (death_df["vaccinated"] == True)
    ]

    icd_counts = sub.groupby("icd").size().sort_values(ascending=False)
    icd_pct = icd_counts / icd_counts.sum()

    icd_pct.to_csv(f"icd_postvax_bin_{b}.csv")
```

---

# 9. Dose‑Specific ICD Comparison

```python
dose_groups = df[df["udalost"].str.lower()=="covid ockovani"].groupby("id_zeny")["covid_ocko_poradi_davky"].max()

death_df = death_df.merge(dose_groups, on="id_zeny", how="left")
death_df["dose_count"] = death_df["covid_ocko_poradi_davky"]
```

For each dose count:

```python
for d in sorted(death_df["dose_count"].dropna().unique()):
    sub = death_df[death_df["dose_count"] == d]
    icd_counts = sub.groupby("icd").size().sort_values(ascending=False)
    icd_pct = icd_counts / icd_counts.sum()
    icd_pct.to_csv(f"icd_by_dose_{int(d)}.csv")
```

---

# 10. Organ System Classification

Include a lookup file:

```
icd_prefix,system
I,cardio
C,oncology
E,endocrine
G,neurologic
J,respiratory
N,renal
U,covid
S,injury
R,other
F,mental
A,infection
```

Apply it:

```python
lookup = pd.read_csv("icd_system_lookup.csv")

death_df["icd_prefix"] = death_df["icd"].str[:1]

death_df = death_df.merge(lookup, on="icd_prefix", how="left")

organ_dist = (
    death_df.groupby(["system","vaccinated"]).size()
    .unstack(fill_value=0)
)

organ_dist.to_csv("icd_by_system.csv")
```

---

# 11. Output Files Generated

- icd_comparison.csv  
- icd_difference_plot.png  
- icd_agegroup_*.csv  
- icd_postvax_bin_*.csv  
- icd_by_dose_*.csv  
- icd_by_system.csv  

---

# End of File
