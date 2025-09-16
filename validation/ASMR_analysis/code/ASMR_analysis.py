# ASMR analysis from KCOR_CMR.xlsx for a single sheet

# This code is used to analyze the ASMR (Age-Specific Mortality Rate) from the KCOR_CMR.xlsx file for a single sheet.

# this analysis assumes that the vaccine didn't raise the baseline mortality rate of the vaccinated before the enrollment date
# which obviously isn't true. So this analysis is a datapoint but it doesn't mean that the vaccine was safe or not.
# This analysis is also not very accurate since it doesn't take into account the baseline correction feature of KCOR.

# this is a quick and dirty analysis to get a sense of the ASMR values for the vaccinated and unvaccinated cohorts.
# this is a STANDALONE analysis and not part of the KCOR pipeline.
# you can call this from the command line or use the Makefile: 
# python ASMR_analysis.py --input ../data/Czech/KCOR_CMR.xlsx --sheet 2021_24 --baseline 2021-06-14 --last_date 2024-04-01 --out_dir ../data/Czech/ASMR_analysis --bands 50-54,55-59,60-64,65-69,70-74,75-79,80-84,85-89,90-94,95-99
# or make ASMR
# the output will be in the ../data/Czech/ASMR_analysis directory


import argparse
import pandas as pd
import numpy as np
import os


def approx_age_at(date_ts: pd.Timestamp, yob: int) -> int:
    dob = pd.Timestamp(year=int(yob), month=7, day=1)
    return max(0, int((date_ts - dob).days // 365.2425))


def main():
    ap = argparse.ArgumentParser(description="ASMR analysis from KCOR_CMR.xlsx for a single sheet")
    ap.add_argument("--input", default=os.path.join("data","Czech","KCOR_CMR.xlsx"), help="Path to KCOR_CMR.xlsx")
    ap.add_argument("--sheet", default="2021_24", help="Sheet name (e.g., 2021_14 or 2021-14)")
    ap.add_argument("--baseline", default="2021-06-14", help="Baseline date (YYYY-MM-DD)")
    ap.add_argument("--last_date", default="2024-04-01", help="Last date to include (YYYY-MM-DD)")
    ap.add_argument("--out_dir", default=os.path.join("validation","ASMR_analysis","out"), help="Output directory")
    ap.add_argument("--bands", default="50-54,55-59,60-64,65-69,70-74,75-79,80-84,85-89,90-94,95-99", help="Comma-separated age bands lo-hi")
    args = ap.parse_args()

    sheet = args.sheet.replace('-', '_')
    baseline_date = pd.to_datetime(args.baseline)
    last_date = pd.to_datetime(args.last_date)
    os.makedirs(args.out_dir, exist_ok=True)

    # Parse bands and default equal standard population weights
    age_bands = []
    standard_pop = {}
    for part in args.bands.split(','):
        lo, hi = part.split('-')
        lo = int(lo.strip()); hi = int(hi.strip())
        age_bands.append((lo, hi))
        standard_pop[(lo, hi)] = 1.0

    # Read CMR sheet and aggregate across sexes
    xls = pd.ExcelFile(args.input)
    df = pd.read_excel(xls, sheet_name=sheet)
    df.columns = [c.strip() for c in df.columns]
    df["DateDied"] = pd.to_datetime(df["DateDied"])
    # Limit to last date
    df = df[df["DateDied"] <= last_date].copy()

    agg = (df.groupby(["DateDied","YearOfBirth","Dose"], as_index=False)
             .agg({"ISOweekDied":"first","Alive":"sum","Dead":"sum"}))

    # Assign fixed age at baseline per YearOfBirth
    agg["age_at_baseline"] = agg["YearOfBirth"].apply(lambda y: approx_age_at(baseline_date, int(y)) if str(y).isdigit() else np.nan)

    # Map to age bands (fixed per person at baseline)
    def to_band(a):
        if pd.isna(a):
            return None
        a = int(a)
        for lo, hi in age_bands:
            if lo <= a <= hi:
                return (lo, hi)
        return None

    agg["age_band"] = agg["age_at_baseline"].apply(to_band)
    agg = agg[agg["age_band"].notna()].copy()

    # Compute fixed denominators (Alive) at baseline date per (age_band, dose)
    base = agg[agg["DateDied"].dt.normalize() == baseline_date.normalize()].copy()
    if base.empty:
        base_week = agg[agg["DateDied"] >= baseline_date]["DateDied"].min()
        base = agg[agg["DateDied"] == base_week].copy()

    denom = (base.groupby(["age_band","Dose"], as_index=False)["Alive"].sum()
                  .rename(columns={"Alive":"cohort_n_at_baseline"}))

    # Build weekly tidy table per (week, age_band, dose)
    weekly = (agg.groupby(["DateDied","age_band","Dose"], as_index=False)[["Dead","Alive"]].sum())
    weekly = weekly.merge(denom, on=["age_band","Dose"], how="left")
    weekly = weekly.rename(columns={"DateDied":"week","Dose":"dose","Dead":"deaths","Alive":"alive"})

    # Weekly MR per 100k using FIXED cohort denominators at baseline
    weekly["mr"] = np.where(weekly["cohort_n_at_baseline"]>0, weekly["deaths"]/(weekly["cohort_n_at_baseline"]+1e-12)*1e5, np.nan)

    # Piecewise slopes over specified periods (example windows)
    def piecewise_slope(g, start, mid, end):
        g = g[(g["week"]>=start)&(g["week"]<end)].sort_values("week")
        tt = (g.set_index("week")["mr"].resample("W-MON").sum().reset_index())
        pre = tt[(tt["week"]>=start)&(tt["week"]<mid)]
        post= tt[(tt["week"]>=mid)&(tt["week"]<end)]
        def slope(h):
            if len(h)<3: return np.nan
            x = (h["week"]-h["week"].min()).dt.days.values
            y = h["mr"].values
            return np.polyfit(x, y, 1)[0]
        return slope(pre)*7, slope(post)*7

    start = pd.Timestamp("2021-06-21"); mid = pd.Timestamp("2021-09-20"); end = pd.Timestamp("2021-12-20")
    rows = []
    for band in age_bands:
        for dose in [0,2]:
            g = weekly[(weekly["dose"]==dose)&(weekly["age_band"]==band)]
            pre_s, post_s = piecewise_slope(g, start, mid, end)
            rows.append({"age_band":band,"dose":dose,"slope_summer":pre_s,"slope_delta":post_s})
    slopes = pd.DataFrame(rows)

    # ASMR (direct standardization) and 2/0 ratio per week
    out_rows = []
    for w, g in weekly.groupby("week"):
        asm = {}
        for dose in [0,2]:
            s = 0.0; wtot = 0.0
            for band in age_bands:
                gi = g[(g["dose"]==dose)&(g["age_band"]==band)]
                mr = gi["mr"].sum()
                wgt = standard_pop[band]
                s += mr*wgt; wtot += wgt
            asm[dose] = s/wtot if wtot>0 else np.nan
        if 0 in asm and 2 in asm and asm[0]>0:
            out_rows.append({
                "week": w,
                "asmr0": asm[0],
                "asmr2": asm[2],
                "ratio_2_over_0": asm[2]/asm[0]
            })
    asmr = pd.DataFrame(out_rows).sort_values("week")

    # Write outputs
    weekly_path = os.path.join(args.out_dir, f"ASMR_weekly_{sheet}.csv")
    slopes_path = os.path.join(args.out_dir, f"ASMR_slopes_{sheet}.csv")
    asmr_path = os.path.join(args.out_dir, f"ASMR_ratio_{sheet}.csv")
    weekly.to_csv(weekly_path, index=False)
    slopes.to_csv(slopes_path, index=False)
    asmr.to_csv(asmr_path, index=False)

    print("Slopes:\n", slopes)
    print("\nASMR tail:\n", asmr.tail(10))


if __name__ == "__main__":
    main()
