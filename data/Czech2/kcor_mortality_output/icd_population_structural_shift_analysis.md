
# Population-Level ICD-10 Structural Shift Analysis  
## Additional Instructions for Steve's AI Coding Engine

This file adds the **critical next analysis** needed to determine whether the shifts in causes of death between vaccinated vs unvaccinated individuals are due to **selection bias** or reflect a **true population-level biological shift** after vaccine rollout.

**Goal:**  
Analyze *all deaths*, regardless of vaccination status, by **year**, **age**, and **ICD cause of death**, to detect any **structural break** around 2021.

---

# 1. Load and Prepare the Death Dataset

```python
import pandas as pd

df = pd.read_csv("foo.csv", dtype=str)
df.columns = [c.strip().lower() for c in df.columns]

death_df = df[df["udalost"].str.lower() == "umrti"].copy()

# Identify ICD column
icd_col = [c for c in death_df.columns if "diag" in c][0]
death_df["icd"] = death_df[icd_col].str.strip()

# Convert key fields to numeric
death_df["year"] = pd.to_numeric(death_df["rok_udalosti"], errors="coerce")
death_df["age"] = pd.to_numeric(death_df["rok_udalosti"], errors="coerce") -                   pd.to_numeric(death_df["rok_narozeni"], errors="coerce")
```

---

# 2. Restrict to a Clean Analysis Window

Use years where reporting is complete and comparable:

```python
death_df = death_df[death_df["year"].between(2010, 2023)]
```

(You can widen to 2000–2023 if stable.)

---

# 3. Compute ICD Distributions Per Year

```python
year_icd_counts = (
    death_df.groupby(["year","icd"])
    .size()
    .rename("count")
    .reset_index()
)

# Convert raw counts to % of deaths for each ICD within each year
year_totals = year_icd_counts.groupby("year")["count"].transform("sum")
year_icd_counts["pct"] = year_icd_counts["count"] / year_totals
```

This gives a long table of:

```
year | icd | count | pct
```

---

# 4. Compare Pre-Vaccine vs Post-Vaccine ICD Patterns

Define eras:

```python
pre = year_icd_counts[year_icd_counts["year"].between(2010, 2019)]
covid_pre_vax = year_icd_counts[year_icd_counts["year"] == 2020]
post = year_icd_counts[year_icd_counts["year"].between(2021, 2023)]
```

Compute average ICD shares before vs after rollout:

```python
pre_mean  = pre.groupby("icd")["pct"].mean()
post_mean = post.groupby("icd")["pct"].mean()

shift = pd.DataFrame({
    "pre_pct": pre_mean,
    "post_pct": post_mean
}).fillna(0)

shift["diff"] = shift["post_pct"] - shift["pre_pct"]
shift.sort_values("diff", ascending=False, inplace=True)
shift.to_csv("icd_population_shift.csv")
```

This file answers:

- Did diabetes deaths rise nationally after 2021?
- Did heart failure rise?
- Did stroke or MI fall or rise?
- Did respiratory/infectious causes shift?
- Did dementia rise more than expected?
- How much of the vax/unvax difference reflects true population change?

---

# 5. Add Organ-System Analysis (Optional, Recommended)

Load your existing lookup:

```python
lookup = pd.read_csv("icd_system_lookup.csv")

# Categorize each death by organ system
death_df["icd_prefix"] = death_df["icd"].str[:1]
death_df = death_df.merge(lookup, on="icd_prefix", how="left")
```

Compute system shares per year:

```python
sys_year = (
    death_df.groupby(["year","system"])
    .size()
    .rename("count")
    .reset_index()
)

# Convert to %
sys_totals = sys_year.groupby("year")["count"].transform("sum")
sys_year["pct"] = sys_year["count"] / sys_totals
sys_year.to_csv("icd_system_yearly.csv")
```

Now compute pre vs post shifts:

```python
sys_pre  = sys_year[sys_year["year"].between(2010,2019)]
sys_post = sys_year[sys_year["year"].between(2021,2023)]

sys_shift = (
    sys_post.groupby("system")["pct"].mean()
    - sys_pre.groupby("system")["pct"].mean()
)

sys_shift.sort_values(ascending=False).to_csv("icd_system_shift.csv")
```

This will show:

- Did **cardiac**, **respiratory**, **neurologic**, **endocrine**, **mental**, **infectious** causes change in share nationally?
- Did chronic diseases increase nationally after 2021?

These patterns tell you whether vaccinated/unvaccinated differences are mere selection bias or reflect wider changes.

---

# 6. Age-Restricted Population Analysis (Critical)

To remove demographic drift across 14 years:

```python
senior = death_df[death_df["age"].between(65, 89)]
```

Then re-run **ALL** previous steps on this subset:

- `year_icd_counts`
- `pre_mean` vs `post_mean`
- `shift`
- organ system shifts

Save:

```python
shift.to_csv("icd_population_shift_age65_89.csv")
sys_shift.to_csv("icd_system_shift_age65_89.csv")
```

If this age-restricted population shows **no major ICD shifts**, but vaccinated/unvaccinated differences are large, that strongly favors **selection bias**.

If this population **does** show structural shifts after 2021, and those shifts **match the vaccinated signature**, that argues against selection bias alone.

---

# 7. Deliverables Produced

Your coder will generate the following files:

- `icd_population_shift.csv`  
- `icd_system_yearly.csv`  
- `icd_system_shift.csv`  
- `icd_population_shift_age65_89.csv`  
- `icd_system_shift_age65_89.csv`  

These files allow you to directly check:

- Did the Czech population’s cause-of-death structure change after 2021?
- By how much?
- Which ICD categories shifted?
- Did the direction match the vaccinated pattern?

This is the **gold-standard test** for distinguishing  
**vaccine-caused biological shifts** vs **selection bias**.

---

# End of File
