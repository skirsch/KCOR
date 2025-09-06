#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-Cohort Direct Standardized Cumulative-Hazard Ratio (DS-CMRR)
Enhanced: CLI flags, balance report, and 'dose_pairs' export.

Direct Standardized Cumulative-Hazard Ratio (DS-CMRR)

Model-free, fixed cohorts, and no slope normalization.
We build vaccinated (dose ≥1) vs unvaccinated (dose 0) cohorts as of the enrollment sheet, 
compute age-specific cumulative hazards over time, then direct-standardize both cohorts to a 
baseline reference distribution (you can pick vax_pt or expected_deaths). Finally we form a 
cumulative ratio and a delta-method CI (Poisson counts, age-wise independence) with baseline 
anchoring to 1 at week ANCHOR_WEEKS.

So: it isn't doing weekly ASMR (i.e., instant ASMR) it is producing a cumulative, age-standardized ratio on 
calendar time using the weighting scheme you picked (default = expected_deaths, which is why 
you're calling it the “ASMR case”).

"""

import sys, os, math, argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

EPS = 1e-12

def iso_enrollment_start(year, week):
    # approximate ISO-week start (Monday)
    # For consistency across sheets, we just need a monotone cutoff date.
    d = datetime.fromisocalendar(int(year), int(week), 1)
    return d

def safe_log(x):  return np.log(np.clip(x, EPS, None))
def safe_exp(x):  return np.exp(np.clip(x, -50, 50))

def dual_print_factory(out_dir, log_filename="DS_CMRR.log"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, log_filename)
    fh = open(path, "w", encoding="utf-8")
    def _p(*args):
        s = " ".join(str(x) for x in args)
        print(s)
        print(s, file=fh)
        fh.flush()
    return _p, fh, path

def prep_sheet(df, sheet_name, year_min, year_max, printer):
    df = df.copy()
    # normalize columns (case-insensitive match where possible)
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in df.columns: return n
            if n.lower() in cols: return cols[n.lower()]
        raise KeyError(f"Missing column among {names}")
    col_date = pick("DateDied","date")
    col_iso  = pick("ISOweekDied","iso_week","ISOweek")
    col_yob  = pick("YearOfBirth","yob","born")
    col_dose = pick("Dose","dose")
    col_alive= pick("Alive","alive","pop","person_weeks")
    col_dead = pick("Dead","dead","deaths")

    df[col_date] = pd.to_datetime(df[col_date])
    df = df[(df[col_yob] >= year_min) & (df[col_yob] <= year_max)].copy()

    # restrict by enrollment start from sheet name "YYYY_WW"
    if "_" in sheet_name:
        yy, ww = sheet_name.split("_")
        start = iso_enrollment_start(int(yy), int(ww))
        df = df[df[col_date] >= start]
        printer(f"[{sheet_name}] start {start.date()}, rows {len(df)}")

    # aggregate sexes by (age, dose, date)
    gb = df.groupby([col_yob, col_dose, col_date], as_index=False)
    a = (gb.agg(**{
            "ISOweekDied": (col_iso, "first"),
            "Alive": (col_alive, "sum"),
            "Dead": (col_dead, "sum")
        }))

    a["PT"] = a["Alive"].astype(float).clip(lower=0.0)
    a["Dead"] = a["Dead"].astype(float).clip(lower=0.0)
    a["MR"] = np.where(a["PT"]>0, a["Dead"]/(a["PT"]+EPS), 0.0)
    a = a.sort_values([col_yob, col_dose, col_date]).reset_index(drop=True)
    a["hazard"] = -np.log(1.0 - np.clip(a["MR"], 0.0, 0.999))
    a["CMR"] = a.groupby([col_yob, col_dose])["hazard"].cumsum()
    a["cumPT"] = a.groupby([col_yob, col_dose])["PT"].cumsum()
    a["cumD"]  = a.groupby([col_yob, col_dose])["Dead"].cumsum()
    a.rename(columns={col_yob:"YearOfBirth", col_dose:"Dose", col_date:"Date"}, inplace=True)
    # time index per (age, dose)
    a["t"] = a.groupby(["YearOfBirth","Dose"]).cumcount()
    return a

def pick_baseline_dates(df, baseline_weeks):
    dates = df["Date"].drop_duplicates().sort_values().head(baseline_weeks).tolist()
    return set(dates)

def compute_weights(df, baseline_dates, scheme, printer):
    """
    Returns weights dict: {YearOfBirth -> weight}, sum to 1 over ages present.
      - 'vax_pt':             w_a ∝ sum_{baseline, dose>=1} PT_a
      - 'expected_deaths':    w_a ∝ (mean baseline MR across all doses)_a × (baseline PT across all doses)_a
    """
    b = df[df["Date"].isin(baseline_dates)].copy()
    ages = sorted(b["YearOfBirth"].unique())
    weights = {}
    for a in ages:
        ga = b[b["YearOfBirth"]==a]
        if scheme == "vax_pt":
            w = float(ga[ga["Dose"]>=1]["PT"].sum())
        elif scheme == "expected_deaths":
            pt_all = float(ga["PT"].sum())
            mra = float(ga["MR"].mean()) if len(ga)>0 else 0.0
            w = mra * pt_all
        else:
            raise ValueError("weights must be 'vax_pt' or 'expected_deaths'")
        weights[a] = max(float(w), 0.0)

    tot = sum(weights.values())
    if tot <= 0:
        weights = {a: 1.0/len(ages) for a in ages}
    else:
        weights = {a: w/tot for a,w in weights.items()}

    # log summary (first 10 ages only)
    s = ", ".join([f"{a}:{weights[a]:.4f}" for a in ages[:10]])
    printer(f"[weights:{scheme}] ages={len(ages)} sample: {s}")
    return weights

def build_per_age_series(df):
    # Pre-compute per-age vax and unvax series aligned by date
    per_age = {}
    for a, g in df.groupby("YearOfBirth"):
        g = g.sort_values(["Dose","Date"])
        u = g[g["Dose"]==0].sort_values("Date")[["Date","CMR","cumD","cumPT"]].drop_duplicates("Date")
        gv = g[g["Dose"]>=1][["Date","hazard","Dead","PT"]].groupby("Date",as_index=False).sum().sort_values("Date")
        gv["CMR"]   = gv["hazard"].cumsum()
        gv["cumD"]  = gv["Dead"].cumsum()
        gv["cumPT"] = gv["PT"].cumsum()
        if not u.empty or not gv.empty:
            per_age[a] = {"u": u, "v": gv}
    return per_age

def standardized_ratio(df, weights, sheet_name, anchor_weeks):
    out_rows = []
    all_dates = sorted(df["Date"].unique())
    iso_map = (df.drop_duplicates(subset=["Date"])[["Date","ISOweekDied"]]
                 .set_index("Date")["ISOweekDied"].to_dict())
    per_age = build_per_age_series(df)
    ages = sorted(df["YearOfBirth"].unique())

    for dt in all_dates:
        num_cmrs, den_cmrs = [], []
        num_d, den_d = [], []
        for a in ages:
            if a not in weights or a not in per_age: 
                continue
            w = weights[a]
            vrow = per_age[a]["v"][per_age[a]["v"]["Date"].eq(dt)]
            urow = per_age[a]["u"][per_age[a]["u"]["Date"].eq(dt)]
            if vrow.empty or urow.empty:
                continue
            vcmr = float(vrow["CMR"].values[0])
            ucmr = float(urow["CMR"].values[0])
            vd   = float(vrow["cumD"].values[0])
            ud   = float(urow["cumD"].values[0])

            num_cmrs.append(w * vcmr)
            den_cmrs.append(w * ucmr)
            num_d.append((w**2) * vd)
            den_d.append((w**2) * ud)

        if not num_cmrs or not den_cmrs:
            continue
        num = sum(num_cmrs)
        den = sum(den_cmrs)
        if den <= 0: 
            continue
        K = num / den
        out_rows.append([sheet_name, pd.Timestamp(dt).date(), iso_map.get(dt), K, sum(num_d), sum(den_d)])

    out = pd.DataFrame(out_rows, columns=["EnrollmentDate","Date","ISOweekDied","K_std","Dnum_w2","Dden_w2"])
    if out.empty:
        return out

    idx0 = anchor_weeks if len(out)>anchor_weeks else 0
    anchor = float(out["K_std"].iloc[idx0]) if len(out)>0 else 1.0
    if not np.isfinite(anchor) or anchor<=0: anchor = 1.0
    out["K_std"] = out["K_std"] / anchor

    base_num = float(out["Dnum_w2"].iloc[idx0]) if len(out)>idx0 else 0.0
    base_den = float(out["Dden_w2"].iloc[idx0]) if len(out)>idx0 else 0.0

    se_logs = []
    for i in range(len(out)):
        vn = out["Dnum_w2"].iloc[i]
        vd = out["Dden_w2"].iloc[i]
        se2 = (vn/(vn+EPS)) + (vd/(vd+EPS)) + (base_num/(base_num+EPS)) + (base_den/(base_den+EPS))
        se = math.sqrt(max(se2, 0.0))
        se_logs.append(se)
    out["SE_logK"] = se_logs
    out["CI_lower"] = out["K_std"] * np.exp(-1.96 * out["SE_logK"])
    out["CI_upper"] = out["K_std"] * np.exp( 1.96 * out["SE_logK"])

    out["YearOfBirth"] = 0
    out["Dose_num"] = 1
    out["Dose_den"] = 0
    out.rename(columns={"K_std":"KCOR"}, inplace=True)
    cols = ["EnrollmentDate","ISOweekDied","Date","YearOfBirth","Dose_num","Dose_den",
            "KCOR","CI_lower","CI_upper"]
    return out[cols]

def smd_continuous(x, g, weights=None):
    """
    Standardized mean difference (continuous) between groups g=0/1.
    If weights is provided (same length), uses weighted mean/var.
    """
    x = np.asarray(x, float)
    g = np.asarray(g, int)
    if weights is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(weights, float)
    mask0 = g==0
    mask1 = g==1
    if mask0.sum()==0 or mask1.sum()==0:
        return np.nan
    def wmean(x, w): return (x*w).sum()/(w.sum()+EPS)
    def wvar (x, w):
        mu = wmean(x,w)
        return ((w*(x-mu)**2).sum())/(w.sum()+EPS)
    m0, v0 = wmean(x[mask0], w[mask0]), wvar(x[mask0], w[mask0])
    m1, v1 = wmean(x[mask1], w[mask1]), wvar(x[mask1], w[mask1])
    sp = math.sqrt( (v0+v1)/2.0 )
    if sp==0: return 0.0
    return (m1-m0)/sp

def balance_report(df, baseline_dates, weights_dict):
    """
    Build a balance table:
      - age distribution (proportions) in baseline window for vax vs unvax
      - target weights w_a
      - L1 distance to target for each group
      - SMD for YearOfBirth (pre) between vax and unvax
    """
    base = df[df["Date"].isin(baseline_dates)]
    if base.empty:
        return pd.DataFrame({"msg":["no baseline rows"]})

    # age distributions by group
    tbl = []
    for grp_name, dose_filter in [("unvax (dose=0)", lambda d: d["Dose"].eq(0)),
                                  ("vax (dose>=1)", lambda d: d["Dose"].ge(1))]:
        bg = base[dose_filter(base)]
        total_pt = float(bg["PT"].sum()) + EPS
        dist = (bg.groupby("YearOfBirth")["PT"].sum()/total_pt).to_dict()
        for a, prop in dist.items():
            tbl.append([grp_name, a, prop])
    dist_df = pd.DataFrame(tbl, columns=["group","YearOfBirth","prop"])

    # target weights
    wtbl = pd.DataFrame([["target", a, w] for a,w in weights_dict.items()],
                        columns=["group","YearOfBirth","prop"])

    # combine
    comb = pd.concat([dist_df, wtbl], ignore_index=True)

    # L1 distances to target within overlapping ages
    def L1_to_target(group_name):
        g = comb[comb["group"].eq(group_name)]
        merged = pd.merge(g, wtbl, on="YearOfBirth", how="inner", suffixes=("", "_target"))
        return float(np.abs(merged["prop"] - merged["prop_target"]).sum())
    L1_unvax = L1_to_target("unvax (dose=0)")
    L1_vax   = L1_to_target("vax (dose>=1)")

    # SMD for YearOfBirth pre-standardization
    # Build person-level using PT as weights (approx) within baseline window
    base_small = base[["YearOfBirth","Dose","PT"]].copy()
    base_small["grp"] = np.where(base_small["Dose"]>=1, 1, 0)
    smd_age = smd_continuous(base_small["YearOfBirth"].values,
                             base_small["grp"].values,
                             weights=base_small["PT"].values)

    meta = pd.DataFrame({
        "metric":["L1_dist_to_target_unvax","L1_dist_to_target_vax","SMD_YearOfBirth_pre"],
        "value":[L1_unvax, L1_vax, smd_age]
    })
    return comb, meta

def export_dose_pairs(excel_writer, df_main, sheet_name="dose_pairs"):
    """
    Export a 'dose_pairs' sheet compatible with many of Steve's downstream flows:
      Columns: EnrollmentDate, YearOfBirth, ISOweekDied, Date, dose_num, dose_den, y, lo, hi
      where y=KCOR, lo=CI_lower, hi=CI_upper.
    """
    out = df_main.rename(columns={
        "KCOR":"y",
        "CI_lower":"lo",
        "CI_upper":"hi",
        "Dose_num":"dose_num",
        "Dose_den":"dose_den",
        "YearOfBirth":"YearOfBirth",
        "ISOweekDied":"ISOweekDied",
        "EnrollmentDate":"EnrollmentDate",
        "Date":"Date"
    })[["EnrollmentDate","YearOfBirth","ISOweekDied","Date","dose_num","dose_den","y","lo","hi"]]
    out.to_excel(excel_writer, index=False, sheet_name=sheet_name)

def process_book(args):
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    printer, fh, log_path = dual_print_factory(out_dir, args.log)

    printer("="*80)
    printer("Fixed-Cohort Direct Standardized Cumulative-Hazard Ratio (DS-CMRR)")
    printer("="*80)
    printer(f"Input : {args.input}")
    printer(f"Output: {args.output}")
    printer(f"Log   : {log_path}")
    printer(f"Config: anchor_weeks={args.anchor_weeks}, baseline_weeks={args.baseline_weeks}, "
            f"weights={args.weights}, year_range=({args.year_min},{args.year_max}), "
            f"sheet_filter={args.sheet if args.sheet else 'all'}")
    printer("="*80)

    xls = pd.ExcelFile(args.input)
    sheets = xls.sheet_names
    if args.sheet:
        wanted = set(args.sheet)
        sheets = [sh for sh in sheets if sh in wanted]
        if not sheets:
            printer("No output rows produced.")
            fh.close()
            return False

    all_out = []
    balances = []
    meta_rows = []

    for sh in sheets:
        printer(f"\n[Sheet] {sh}")
        df_raw = pd.read_excel(args.input, sheet_name=sh)
        df = prep_sheet(df_raw, sh, args.year_min, args.year_max, printer)
        if df.empty:
            printer("  (empty after filters)")
            continue

        baseline_dates = pick_baseline_dates(df, args.baseline_weeks)
        weights = compute_weights(df, baseline_dates, args.weights, printer)
        out_sh = standardized_ratio(df, weights, sh, args.anchor_weeks)
        if out_sh.empty:
            printer("  (no aligned dates across cohorts)")
            continue
        all_out.append(out_sh)

        if args.balance_report:
            comb, meta = balance_report(df, baseline_dates, weights)
            comb.insert(0, "EnrollmentDate", sh)
            meta.insert(0, "EnrollmentDate", sh)
            balances.append(comb)
            meta_rows.append(meta)

    if not all_out:
        printer("No output rows produced.")
        fh.close()
        return False

    combined = (pd.concat(all_out, ignore_index=True)
                  .sort_values(["EnrollmentDate","Date"])
                  .reset_index(drop=True))

    with pd.ExcelWriter(args.output, engine="openpyxl") as w:
        # About
        about = pd.DataFrame({
            "Field":[
                "Method","Anchor week index","Baseline weeks","Weight scheme",
                "Generated","Input file","Output file","Year min","Year max"
            ],
            "Value":[
                "Fixed-Cohort Direct Standardized Cumulative-Hazard Ratio (DS-CMRR)",
                args.anchor_weeks, args.baseline_weeks, args.weights,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.input, args.output,
                args.year_min, args.year_max
            ]
        })
        about.to_excel(w, index=False, sheet_name="About")
        # Main
        combined.to_excel(w, index=False, sheet_name="DS_CMRR")
        # Optional dose_pairs export
        if args.export_dose_pairs:
            export_dose_pairs(w, combined, sheet_name="dose_pairs")
        # Optional balance sheet(s)
        if args.balance_report and balances:
            pd.concat(balances, ignore_index=True).to_excel(w, index=False, sheet_name="BalanceShares")
            pd.concat(meta_rows, ignore_index=True).to_excel(w, index=False, sheet_name="BalanceMeta")

    printer(f"\n[Done] Wrote {len(combined)} rows to {args.output}")
    fh.close()
    return True

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Fixed-cohort standardized cumulative hazard ratio (DS-CMRR)"
    )
    p.add_argument("input", help="Input Excel workbook with per-sheet enrollment data")
    p.add_argument("output", help="Output Excel workbook")
    p.add_argument("--log", default="DS_CMRR.log", help="Log filename (default: DS_CMRR.log)")
    p.add_argument("--weights", choices=["vax_pt","expected_deaths"], default="expected_deaths",
                   help="Standardization weight scheme")
    p.add_argument("--anchor-weeks", type=int, default=4,
                   help="Index of row to anchor ratio to 1 (default: 4)")
    p.add_argument("--baseline-weeks", type=int, default=4,
                   help="Number of initial weeks to define baseline window (default: 4)")
    p.add_argument("--year-min", type=int, default=1920, help="Minimum YearOfBirth to keep")
    p.add_argument("--year-max", type=int, default=2000, help="Maximum YearOfBirth to keep")
    p.add_argument("--keep-details", action="store_true",
                   help="(reserved) keep per-age detail outputs (not used presently)")
    p.add_argument("--export-dose-pairs", action="store_true",
                   help="Also write a 'dose_pairs' sheet for downstream plotting")
    p.add_argument("--balance-report", action="store_true",
                   help="Write BalanceShares and BalanceMeta sheets with age-share and SMD diagnostics")
    p.add_argument("--sheet", action="append",
                   help="Only process specific sheet(s); can be repeated (e.g., --sheet 2021_24)")
    return p

def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    ok = process_book(args)
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
