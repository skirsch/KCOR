"""Manufacturer subsets of the ordinary fixed enrollment population."""
import numpy as np
import pandas as pd

STRATA = ["YearOfBirth", "Sex", "DCCI"]
TARGETS = {"2021_24": 2, "2022_06": 3, "2022_47": 4}


def build_history(alive_at_enroll, enrollment, weeks, dose):
    """Partition the already assigned enrollment dose, never dose at death."""
    start = pd.Timestamp(enrollment)
    dates = pd.to_datetime(pd.Series(weeks) + "-1", format="%G-%V-%u")
    dates = dates[dates >= start]
    if dates.empty:
        raise ValueError("No post-enrollment weeks")
    sub = alive_at_enroll.loc[alive_at_enroll.dose_group.eq(dose)].copy()
    col = f"Dose{dose}_MFG"
    sub["MFG"] = sub[col].where(sub[col].isin(["P", "M"]), "O")
    keys = STRATA + ["MFG"]
    pop = sub.groupby(keys, dropna=False).size().rename("opening").reset_index()
    if pop.empty:
        return pd.DataFrame(columns=["ISOweekDied", "DateDied", *keys, "Dose", "Alive", "Dead"])
    grid = pop.merge(pd.DataFrame({"DateDied": dates.to_numpy()}), how="cross")
    dead = sub.loc[sub.DateOfDeath.notna() & sub.WeekOfDeath.notna(), keys + ["WeekOfDeath"]].copy()
    dead["DateDied"] = pd.to_datetime(dead.WeekOfDeath + "-1", format="%G-%V-%u")
    counts = dead.groupby(keys + ["DateDied"], dropna=False).size().rename("Dead").reset_index()
    grid = grid.merge(counts, on=keys + ["DateDied"], how="left").sort_values(keys + ["DateDied"])
    grid["Dead"] = grid.Dead.fillna(0).astype(int)
    grid["Alive"] = grid.opening - grid.groupby(keys, dropna=False).Dead.transform(lambda x: x.cumsum().shift(fill_value=0))
    grid["Dose"] = dose
    grid["ISOweekDied"] = grid.DateDied.dt.strftime("%G-%V")
    grid["DateDied"] = grid.DateDied.dt.strftime("%Y-%m-%d")
    return grid[["ISOweekDied", "DateDied", *keys, "Dose", "Alive", "Dead"]]


def reconcile(history, ordinary, enrollment, dose):
    """Reject incompatible archived or malformed manufacturer populations."""
    start = pd.Timestamp(enrollment)
    keys = STRATA + ["DateDied"]
    cols = ["Alive", "Dead"]
    def counts(frame):
        f = frame.copy()
        f["DateDied"] = pd.to_datetime(f.DateDied)
        f = f.loc[(f.DateDied >= start) & f.Dose.eq(dose) & (f.YearOfBirth >= -1)]
        if not np.isfinite(f[cols].to_numpy(float)).all():
            raise ValueError("Nonfinite manufacturer reconciliation counts")
        return f.groupby(keys, dropna=False)[cols].sum()
    if history.empty or not history.MFG.isin(["P", "M", "O"]).all() or not history.Dose.eq(dose).all():
        raise ValueError("Missing or invalid manufacturer categories")
    a, b = counts(history).align(counts(ordinary), fill_value=0)
    if not a.equals(b):
        # Excel may represent the same integer counts with different dtypes.
        if not np.array_equal(a.to_numpy(), b.to_numpy()):
            raise ValueError("Manufacturer histories do not reconcile to ordinary enrollment cohort")
