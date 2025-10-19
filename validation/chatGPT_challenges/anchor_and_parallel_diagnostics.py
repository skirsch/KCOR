#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np, statsmodels.api as sm
from pathlib import Path

HERE = Path(__file__).parent
CMR_PATH  = HERE / "../../data/Czech/KCOR_CMR.xlsx"
OUTDIR = HERE / "out"
OUTDIR.mkdir(exist_ok=True)

def slope_and_curvature(cmr_df: pd.DataFrame, yob_lo: int, yob_hi: int, dose: int, weeks: int):
    sub = cmr_df[(cmr_df['Dose']==dose) & (cmr_df['YearOfBirth'].between(yob_lo, yob_hi))].copy()
    # Order by week proxy
    if 'DateDied' in sub.columns and sub['DateDied'].notna().any():
        sub = sub.sort_values('DateDied')
    else:
        sub = sub.sort_values('ISOweekDied')
    sub['N'] = sub['Alive'].fillna(0) + sub['Dead'].fillna(0)
    sub = sub[sub['N']>0].reset_index(drop=True)
    sub['MR'] = sub['Dead'] / sub['N']
    sub = sub.head(weeks).reset_index(drop=True)
    if len(sub) < 10 or (sub['MR']<=0).all():
        return None
    sub['t'] = np.arange(len(sub))
    sub['lnMR'] = np.log(sub['MR'].clip(lower=1e-12))
    X1 = sm.add_constant(sub['t'])
    X2 = sm.add_constant(np.column_stack([sub['t'], sub['t']**2]))
    m1 = sm.OLS(sub['lnMR'], X1).fit()
    m2 = sm.OLS(sub['lnMR'], X2).fit()
    return {
        "n_weeks": len(sub),
        "beta_linear": float(m1.params[1]),
        "beta_se": float(m1.bse[1]),
        "curv_q": float(m2.params[2]),
        "curv_q_se": float(m2.bse[2]),
        "r2_lin": float(m1.rsquared),
        "r2_quad": float(m2.rsquared),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enroll", required=True)
    ap.add_argument("--yob", nargs=2, type=int, required=True, metavar=("YOB_LO","YOB_HI"))
    ap.add_argument("--dose", nargs="+", type=int, required=True, help="e.g. 0 3")
    ap.add_argument("--weeks", type=int, default=26)
    ap.add_argument("--shifts", nargs="+", type=int, default=[-8,0,8,12], help="anchor shifts to probe")
    args = ap.parse_args()

    cxl = pd.ExcelFile(CMR_PATH)
    cmr = pd.read_excel(cxl, args.enroll)

    # baseline (first 'weeks' weeks) slope & curvature per dose
    rows = []
    for d in args.dose:
        est = slope_and_curvature(cmr, args.yob[0], args.yob[1], d, args.weeks)
        rows.append({"dose": d, **(est or {})})
    base = pd.DataFrame(rows)
    base.to_csv(OUTDIR / f"parallel_{args.enroll}_{args.yob[0]}_{args.yob[1]}_weeks{args.weeks}.csv", index=False)

    # crude anchor shift sensitivity: recompute with windows starting at offset k and length 'weeks'
    # We approximate by skipping the first k rows (k may be negative => clamp at 0)
    sens = []
    for d in args.dose:
        for k in args.shifts:
            sub = cmr[(cmr['Dose']==d) & (cmr['YearOfBirth'].between(args.yob[0], args.yob[1]))].copy()
            if 'DateDied' in sub.columns and sub['DateDied'].notna().any():
                sub = sub.sort_values('DateDied')
            else:
                sub = sub.sort_values('ISOweekDied')
            sub['N'] = sub['Alive'].fillna(0) + sub['Dead'].fillna(0)
            sub = sub[sub['N']>0].reset_index(drop=True)
            start = max(0, k if isinstance(k,int) else 0)
            window = sub.iloc[start:start+args.weeks].copy()
            if len(window) < 10:
                continue
            window['t'] = np.arange(len(window))
            window['MR'] = (window['Dead'] / window['N']).clip(lower=1e-12)
            window['lnMR'] = np.log(window['MR'])
            X = sm.add_constant(window['t'])
            m = sm.OLS(window['lnMR'], X).fit()
            sens.append({"dose": d, "shift": k, "beta": float(m.params[1]), "beta_se": float(m.bse[1])})
    pd.DataFrame(sens).to_csv(OUTDIR / f"anchor_sensitivity_{args.enroll}_{args.yob[0]}_{args.yob[1]}_weeks{args.weeks}.csv", index=False)

    print("Wrote diagnostics to", OUTDIR)

if __name__ == "__main__":
    main()
