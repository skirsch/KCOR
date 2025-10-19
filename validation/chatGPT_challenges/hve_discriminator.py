#!/usr/bin/env python3
"""
hve_discriminator.py

Distinguishes dynamic HVE (Healthy Vaccinee Effect) from immediate post-vaccine harm.
Performs calendar-symmetric masking and pseudo-exposure null control.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
CMR_PATH  = HERE / "../../data/Czech/KCOR_CMR.xlsx"
KCOR_PATH = HERE / "../../data/Czech/KCOR.xlsx"
OUTDIR = HERE / "out"
OUTDIR.mkdir(exist_ok=True)

def load_weekly_cmr(cmr_df, yob_lo, yob_hi, dose):
    df = cmr_df[(cmr_df['Dose']==dose) & (cmr_df['YearOfBirth'].between(yob_lo, yob_hi))].copy()
    if 'DateDied' in df.columns and df['DateDied'].notna().any():
        df = df.sort_values('DateDied')
    else:
        df = df.sort_values('ISOweekDied')
    df['Alive'] = df['Alive'].fillna(0).astype(int)
    df['Dead']  = df['Dead'].fillna(0).astype(int)
    df['N'] = df['Alive'] + df['Dead']
    df = df[df['N']>0].reset_index(drop=True)
    df['MR'] = (df['Dead'] / df['N']).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

def kcor_like(num_series, den_series):
    mask = ~np.isnan(num_series) & ~np.isnan(den_series)
    n = num_series[mask]
    d = den_series[mask]
    cnum = np.cumsum(n)
    cden = np.cumsum(d)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.where(cden>0, cnum / cden, np.nan)
    return r

def apply_mask(series, mask):
    out = series.copy()
    out[mask] = np.nan
    return out

def endpoints_from_series(series, t_points=(26,52,78)):
    out = []
    T_max = len(series)-1
    for T in t_points:
        idx = min(T, T_max)
        out.append((idx, float(series[idx]) if idx>=0 and not np.isnan(series[idx]) else np.nan))
    return out

def main():
    ap = argparse.ArgumentParser(description="Discriminate dynamic-HVE vs immediate post-vax harm via symmetric and pseudo masking.")
    ap.add_argument("--enroll", required=True)
    ap.add_argument("--yob", nargs=2, type=int, required=True, metavar=("YOB_LO","YOB_HI"))
    ap.add_argument("--num", type=int, default=3)
    ap.add_argument("--den", type=int, default=0)
    ap.add_argument("--k", type=int, default=3, help="Early weeks to mask")
    ap.add_argument("--weeks", type=int, default=26)
    ap.add_argument("--R", type=int, default=500, help="Null draws for pseudo-masking")
    args = ap.parse_args()

    cxl = pd.ExcelFile(CMR_PATH)
    cmr = pd.read_excel(cxl, args.enroll)

    num_df = load_weekly_cmr(cmr, args.yob[0], args.yob[1], args.num)
    den_df = load_weekly_cmr(cmr, args.yob[0], args.yob[1], args.den)

    T = min(len(num_df), len(den_df))
    num_mr = num_df['MR'].values[:T]
    den_mr = den_df['MR'].values[:T]

    # Mask for first K calendar weeks
    K = min(args.k, T)
    M = np.zeros(T, dtype=bool)
    if K > 0:
        M[:K] = True

    # A) Asymmetric skip: mask numerator only
    num_A = apply_mask(num_mr, M)
    den_A = den_mr.copy()
    series_A = kcor_like(num_A, den_A)

    # B) Calendar-symmetric skip: mask both
    num_B = apply_mask(num_mr, M)
    den_B = apply_mask(den_mr, M)
    series_B = kcor_like(num_B, den_B)

    # C) Random pseudo masks
    rng = np.random.default_rng(1)
    null_endpoints = []
    one_series_C = None
    for r in range(args.R):
        idxs = rng.choice(T, size=K, replace=False) if K>0 else np.array([], dtype=int)
        M_rand = np.zeros(T, dtype=bool)
        M_rand[idxs] = True
        sC = kcor_like(apply_mask(num_mr, M_rand), apply_mask(den_mr, M_rand))
        for t_idx, val in endpoints_from_series(sC):
            null_endpoints.append({"variant":"C_random","draw":r,"t_weeks":t_idx,"K_like":val})
        if r==0:
            one_series_C = sC

    ser_df = pd.DataFrame({
        "t": np.arange(len(series_A)),
        "A_asym_num_only": series_A,
        "B_sym_both": series_B,
        "C_rand_example": one_series_C if one_series_C is not None else np.full(len(series_A), np.nan)
    })
    ser_path = OUTDIR / f"hve_disc_series_{args.enroll}_{args.yob[0]}_{args.yob[1]}_{args.num}v{args.den}_K{args.k}.csv"
    ser_df.to_csv(ser_path, index=False)

    # Collect endpoints
    end_rows = []
    for label, s in [("A_asym", series_A), ("B_sym", series_B)]:
        for t_idx, val in endpoints_from_series(s):
            end_rows.append({"variant":label,"t_weeks":t_idx,"K_like":val})

    end_df = pd.DataFrame(end_rows)
    null_df = pd.DataFrame(null_endpoints)
    end_path = OUTDIR / f"hve_disc_endpoints_{args.enroll}_{args.yob[0]}_{args.yob[1]}_{args.num}v{args.den}_K{args.k}.csv"
    null_path = OUTDIR / f"hve_disc_endpoints_null_{args.enroll}_{args.yob[0]}_{args.yob[1]}_{args.num}v{args.den}_K{args.k}.csv"
    end_df.to_csv(end_path, index=False)
    null_df.to_csv(null_path, index=False)

    print("Wrote:", ser_path, end_path, null_path)

if __name__ == "__main__":
    main()
