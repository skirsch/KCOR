
# ICD-10 Cause of Death Comparison: Vaccinated vs Unvaccinated

This file provides code you can run in your coding engine (Python, Jupyter, Cursor, etc.) to compare **cause-of-death distributions** between **vaccinated** and **unvaccinated** individuals using the Czech event-level dataset.

---

## 1. Load and clean data

```python
import pandas as pd

df = pd.read_csv("data.csv", dtype=str)

# Normalize column names
df.columns = [c.strip().lower() for c in df.columns]

# Convert numeric fields where appropriate
for c in ["id_zeny", "rok_udalosti", "mesic_udalosti",
          "covid_ocko_poradi_davky", "covid_onemoc_poradi_infekce"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
```

---

## 2. Identify death records

```python
death_df = df[df["udalost"].str.lower() == "umrti"].copy()

# Identify ICD column
icd_col = [c for c in death_df.columns if "diag" in c][0]

death_df["icd"] = death_df[icd_col].str.strip()
```

---

## 3. Determine vaccination status per person

```python
vaccinated_ids = set(
    df[df["udalost"].str.lower() == "covid ockovani"]["id_zeny"].unique()
)

death_df["vaccinated"] = death_df["id_zeny"].isin(vaccinated_ids)
```

---

## 4. Build ICD histograms

### Vaccinated deaths:
```python
vaccinated_counts = (
    death_df[death_df["vaccinated"] == True]
    .groupby("icd")
    .size()
    .sort_values(ascending=False)
)
```

### Unvaccinated deaths:
```python
unvaccinated_counts = (
    death_df[death_df["vaccinated"] == False]
    .groupby("icd")
    .size()
    .sort_values(ascending=False)
)
```

---

## 5. Normalize distributions

```python
vaccinated_dist = vaccinated_counts / vaccinated_counts.sum()
unvaccinated_dist = unvaccinated_counts / unvaccinated_counts.sum()
```

---

## 6. Create comparison table

```python
comparison = pd.DataFrame({
    "vaccinated_pct": vaccinated_dist,
    "unvaccinated_pct": unvaccinated_dist
}).fillna(0)

comparison["difference"] = comparison["vaccinated_pct"] - comparison["unvaccinated_pct"]
comparison.sort_values("difference", ascending=False, inplace=True)

comparison.to_csv("icd_comparison.csv")
```

---

## 7. Plot differences

```python
import matplotlib.pyplot as plt

top_n = 20
top_icds = comparison.abs().sort_values("difference", ascending=False).head(top_n)

plt.figure(figsize=(12,8))
plt.barh(top_icds.index, top_icds["difference"])
plt.xlabel("Vaccinated % minus Unvaccinated %")
plt.title("Difference in Cause-of-Death Distribution (by ICD-10 Category)")
plt.tight_layout()
plt.savefig("icd_difference_plot.png")
plt.show()
```

---

## Interpretation Guide

### ICDs more common among vaccinated deaths:
- Cardiovascular: **I21–I24**, **I46**, **I63**
- Myocarditis/pericarditis: **I40–I42**
- Respiratory: **J12–J18**, **J80**
- Cancers: **C00–C97**
- Neurologic: **G93**
- Renal: **N17–N19**

### ICDs more common among unvaccinated deaths:
May represent:
- older/unhealthier baseline population
- true biological differences
- misclassification patterns
- random distributional noise for small categories

---

## Optional Extensions

- ICD grouping by organ system  
- Time-from-last-dose → ICD distribution  
- Dose-specific ICD differences  
- Age-stratified ICD comparisons  
- Logistic regression models  
- Cause-specific KCOR hazard curves  

Ask and I can generate any of these.

