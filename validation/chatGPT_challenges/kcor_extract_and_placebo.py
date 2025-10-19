#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, os
from pathlib import Path

HERE = Path(__file__).parent
CMR_PATH  = HERE / "../../data/Czech/KCOR_CMR.xlsx"
KCOR_PATH = HERE / "../../data/Czech/KCOR.xlsx"
OUTDIR = HERE / "out"
OUTDIR.mkdir(exist_ok=True)

def kcor_at_t(dose_pairs: pd.DataFrame, enrollment: str, yob_lo: int, yob_hi: int, dose_num: int, dose_den: int, t_points=(26,52,78)):
    df = dose_pairs.copy()
    df = df[(df['EnrollmentDate']==enrollment) & (df['YearOfBirth'].between(yob_lo, yob_hi)) &
            (df['Dose_num']==dose_num) & (df['Dose_den']==dose_den)]
    if df.empty:
        return pd.DataFrame(columns=["EnrollmentDate","AgeBand_YOB","Dose_num","Dose_den","t_weeks","KCOR"])
    tcol = 't_num' if 't_num' in df.columns else ('t' if 't' in df.columns else None)
    kcol = 'KCOR_o' if 'KCOR_o' in df.columns else [c for c in df.columns if isinstance(c,str) and c.upper().startswith('KCOR')][0]
    out = []
    for t in t_points:
        i = (df[tcol]-t).abs().idxmin()
        row = df.loc[i]
        out.append({
            "EnrollmentDate": enrollment,
            "AgeBand_YOB": f"{yob_lo}-{yob_hi}",
            "Dose_num": dose_num,
            "Dose_den": dose_den,
            "t_weeks": int(row[tcol]),
            "KCOR": float(row[kcol]) if pd.notna(row[kcol]) else np.nan
        })
    return pd.DataFrame(out)

def placebo_split_cmr(cmr_df: pd.DataFrame, yob_lo: int, yob_hi: int, n_trials=200, seed=0):
    rng = np.random.default_rng(seed)
    df = cmr_df[(cmr_df['Dose']==0) & (cmr_df['YearOfBirth'].between(yob_lo, yob_hi))].copy()
    # order by week proxy
    if 'DateDied' in df.columns and df['DateDied'].notna().any():
        df = df.sort_values('DateDied')
    else:
        df = df.sort_values('ISOweekDied')
    wk = df[['Alive','Dead']].fillna(0).astype(int).reset_index(drop=True)
    if wk.empty:
        return pd.DataFrame(columns=["CMRR_A_over_B"])
    res = []
    for _ in range(n_trials):
        dead_A = rng.binomial(wk['Dead'], 0.5)
        alive_A = rng.binomial(wk['Alive'], 0.5)
        dead_B = wk['Dead'] - dead_A
        alive_B = wk['Alive'] - alive_A
        cum_dead_A = int(dead_A.sum())
        cum_pt_A   = int((alive_A + dead_A).sum())
        cum_dead_B = int(dead_B.sum())
        cum_pt_B   = int((alive_B + dead_B).sum())
        if cum_pt_A==0 or cum_pt_B==0: 
            cmrr = np.nan
        else:
            cmr_A = cum_dead_A / cum_pt_A
            cmr_B = cum_dead_B / cum_pt_B if cum_pt_B>0 else np.nan
            cmrr  = cmr_A / cmr_B if (cmr_A>0 and cmr_B>0) else np.nan
        res.append(cmrr)
    return pd.DataFrame({"CMRR_A_over_B": res})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enroll", required=True, help="Enrollment sheet, e.g., 2021_24")
    ap.add_argument("--yob", nargs=2, type=int, required=True, metavar=("YOB_LO","YOB_HI"))
    ap.add_argument("--num", type=int, default=3, help="Numerator dose, default 3")
    ap.add_argument("--den", type=int, default=0, help="Denominator dose, default 0")
    ap.add_argument("--n_trials", type=int, default=200, help="Placebo trials")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    kxl = pd.ExcelFile(KCOR_PATH)
    dps = pd.read_excel(kxl, "dose_pairs")

    # KCOR at 26/52/78
    k_tbl = kcor_at_t(dps, args.enroll, args.yob[0], args.yob[1], args.num, args.den)
    k_out = OUTDIR / f"kcor_{args.enroll}_{args.yob[0]}_{args.yob[1]}_{args.num}v{args.den}.csv"
    k_tbl.to_csv(k_out, index=False)

    # Placebo split on the matching CMR sheet
    cxl = pd.ExcelFile(CMR_PATH)
    cmr = pd.read_excel(cxl, args.enroll)
    p_tbl = placebo_split_cmr(cmr, args.yob[0], args.yob[1], n_trials=args.n_trials, seed=args.seed)
    p_out = OUTDIR / f"placebo_{args.enroll}_{args.yob[0]}_{args.yob[1]}_v0split.csv"
    p_tbl.to_csv(p_out, index=False)

    # Summary stats
    if not p_tbl.empty:
        s = p_tbl['CMRR_A_over_B'].dropna()
        summ = pd.DataFrame([{
            "Label": f"{args.enroll} YOB {args.yob[0]}-{args.yob[1]} v0 split",
            "N": int(s.shape[0]),
            "Mean": float(s.mean()) if s.shape[0]>0 else np.nan,
            "Std": float(s.std(ddof=1)) if s.shape[0]>1 else np.nan,
            "p05": float(np.percentile(s,5)) if s.shape[0]>0 else np.nan,
            "p95": float(np.percentile(s,95)) if s.shape[0]>0 else np.nan,
        }])
        summ.to_csv(OUTDIR / f"placebo_summary_{args.enroll}_{args.yob[0]}_{args.yob[1]}.csv", index=False)

    print("Wrote:", k_out, p_out)

if __name__ == "__main__":
    main()
