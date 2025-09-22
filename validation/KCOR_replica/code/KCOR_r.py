# this is KCOR created by ChatGPT.
# it is a re-creation of the KCOR algorithm from the description in the README.md file.

"""
Here’s what I set you up with:

A ready-to-run Python script that takes your “digested” table (fixed cohorts by week) and outputs KCOR(t) plus an optional PNG.

It expects columns like:

cohort (e.g., D0, D1, D2), week (int index from your enrollment), deaths, and (ideally) population.

Optional: age_band. If present, you can filter to one band; if omitted, it will aggregate across all ages.

If mr isn’t present, it computes mr = deaths / population.

What it does (aligned to your KCOR steps):

Fits a slope β of log(mr) vs week per (cohort, age_band) from your enrollment week forward.

Neutralizes each cohort’s MR with mr_adj = mr * exp(-β*(week - enroll_week)).

Discrete hazard: prefers h = deaths / population (if pop exists); otherwise uses 1 - exp(-mr_adj).

Cumulative hazards per cohort, align weeks, form R(t)=H_A/H_B.

Scale the whole curve by R(scale_week) (default week 4).

Output a tidy CSV of week, H_A, H_B, R, KCOR, and (optionally) a plot.

To use:

python KCOR_r.py --input my_fixed_enrollment.csv \
  --doseA D2 --doseB D0 --age_band "80-89" \
  --enroll_week 0 --scale_week 4 \
  --out kcor_out.csv --plot kcor_plot.png

"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cohort", "week", "deaths"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    # Normalize column names
    if "age_band" not in df.columns:
        df["age_band"] = "ALL"
    if "population" not in df.columns:
        df["population"] = np.nan  # will trigger alt hazard calc
    # Aggregate if duplicates exist
    grp_cols = ["cohort", "age_band", "week"]
    df = df.groupby(grp_cols, as_index=False).agg({"deaths":"sum", "population":"sum"})
    # Mortality rate per week
    if "mr" in df.columns:
        # if someone included an mr column in input, recompute to ensure consistency
        df["mr"] = df["deaths"] / df["population"]
    else:
        df["mr"] = df["deaths"] / df["population"]
    return df

def fit_beta(df_group: pd.DataFrame, enroll_week: int) -> float:
    """Fit slope beta of log(mr) ~ week on weeks >= enroll_week with mr>0 and finite."""
    d = df_group[df_group["week"] >= enroll_week].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["mr"])
    d = d[d["mr"] > 0]
    if len(d) < 2:
        return 0.0
    x = d["week"].to_numpy()
    y = np.log(d["mr"].to_numpy())
    # simple least squares
    A = np.vstack([x, np.ones_like(x)]).T
    beta, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(beta)

def neutralize_slope(df: pd.DataFrame, enroll_week: int) -> pd.DataFrame:
    out = []
    for (cohort, age), d in df.groupby(["cohort", "age_band"], as_index=False):
        beta = fit_beta(d, enroll_week)
        d = d.copy()
        # neutralize to flat slope at enroll_week
        d["mr_adj"] = d["mr"] * np.exp(-beta * (d["week"] - enroll_week))
        d["beta"] = beta
        out.append(d)
    return pd.concat(out, ignore_index=True)

def discrete_hazard(row) -> float:
    # Prefer discrete hazard via deaths/pop if population known and >0
    pop = row["population"]
    if not (isinstance(pop, float) or isinstance(pop, int)):
        pop = np.nan
    if not np.isnan(pop) and pop > 0:
        return float(row["deaths"]) / float(pop)
    # Else use continuous approx on adjusted rate
    mr_adj = row.get("mr_adj", np.nan)
    if np.isnan(mr_adj):
        return np.nan
    return 1.0 - math.exp(-float(mr_adj))

def compute_hazards(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "mr_adj" not in df.columns:
        df["mr_adj"] = df["mr"]
    df["h"] = df.apply(discrete_hazard, axis=1)
    return df

def cumulative_hazard(df: pd.DataFrame, cohort: str, age_band: str | None) -> pd.DataFrame:
    d = df[df["cohort"] == cohort].copy()
    if age_band:
        d = d[d["age_band"] == age_band].copy()
    # if multiple age bands are present and no filter, aggregate across age bands first (deaths/pop then recompute mr_adj)
    if age_band is None and d["age_band"].nunique() > 1:
        agg = d.groupby(["cohort","week"], as_index=False).agg({"deaths":"sum","population":"sum"})
        agg["mr"] = agg["deaths"] / agg["population"]
        # For mr_adj, we don't have beta across combined ages; approximate by re-fitting on aggregate
        beta = fit_beta(agg.rename(columns={"week":"week","mr":"mr"}), enroll_week=agg["week"].min())
        agg["mr_adj"] = agg["mr"] * np.exp(-beta*(agg["week"]-agg["week"].min()))
        agg["h"] = agg.apply(lambda r: (r["deaths"]/r["population"]) if r["population"]>0 else (1.0 - math.exp(-r["mr_adj"])), axis=1)
        d = agg
    d = d.sort_values("week")
    d["H"] = d["h"].cumsum()
    return d[["week","H"]].reset_index(drop=True)

def compute_kcor(df: pd.DataFrame,
                 doseA: str,
                 doseB: str,
                 age_band: str | None,
                 enroll_week: int,
                 scale_week: int):
    # 1-2: slope neutralize
    df2 = neutralize_slope(df, enroll_week=enroll_week)
    # 3: hazards
    df3 = compute_hazards(df2)
    # 4: cumulative hazards per cohort
    HA = cumulative_hazard(df3, doseA, age_band)
    HB = cumulative_hazard(df3, doseB, age_band)
    # align weeks
    merged = pd.merge(HA, HB, on="week", suffixes=(f"_{doseA}", f"_{doseB}"), how="inner")
    merged["R"] = merged[f"H_{doseA}"] / merged[f"H_{doseB}"]
    # 5: scale by value at scale_week
    if scale_week in set(merged["week"]):
        denom = float(merged.loc[merged["week"]==scale_week, "R"].iloc[0])
        if denom != 0 and np.isfinite(denom):
            merged["KCOR"] = merged["R"] / denom
        else:
            merged["KCOR"] = np.nan
    else:
        # if scale week not present, scale by first available week >= scale_week or the first week
        candidates = merged[merged["week"]>=scale_week]
        if len(candidates)==0:
            denom = float(merged["R"].iloc[0])
        else:
            denom = float(candidates["R"].iloc[0])
        merged["KCOR"] = merged["R"] / denom if denom!=0 and np.isfinite(denom) else np.nan
    return merged[["week", f"H_{doseA}", f"H_{doseB}", "R", "KCOR"]].copy()

def main():
    ap = argparse.ArgumentParser(description="Compute KCOR(t) from a digested fixed-cohort dataset.")
    ap.add_argument("--input", required=True, help="Path to CSV with columns: cohort, age_band (opt), week, deaths, population (opt)")
    ap.add_argument("--doseA", required=True, help="Exposed/vaccinated cohort label (e.g., D2)")
    ap.add_argument("--doseB", required=True, help="Control cohort label (e.g., D0)")
    ap.add_argument("--age_band", default=None, help="Age band to filter (exact match). Omit to aggregate across all ages.")
    ap.add_argument("--enroll_week", type=int, default=0, help="Week index considered as enrollment/reference for slope neutralization")
    ap.add_argument("--scale_week", type=int, default=4, help="Week at which to scale the ratio to 1.0")
    ap.add_argument("--out", default="kcor_out.csv", help="Output CSV path for KCOR timeseries")
    ap.add_argument("--plot", default=None, help="Optional PNG path for a plot of KCOR(t)")
    args = ap.parse_args()

    df = read_data(args.input)
    k = compute_kcor(df, args.doseA, args.doseB, args.age_band, args.enroll_week, args.scale_week)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    k.to_csv(args.out, index=False)

    if args.plot:
        plt.figure()
        plt.plot(k["week"], k["KCOR"])
        plt.xlabel("Week")
        plt.ylabel("KCOR(t) (scaled)")
        plt.title(f"KCOR: {args.doseA}/{args.doseB}" + (f", age {args.age_band}" if args.age_band else ""))
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.savefig(args.plot, dpi=160, bbox_inches="tight")
        plt.close()

    print(f"Wrote KCOR timeseries to {args.out}")
    if args.plot:
        print(f"Wrote plot to {args.plot}")

if __name__ == "__main__":
    main()
