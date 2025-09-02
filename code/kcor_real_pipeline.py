
#!/usr/bin/env python3
"""
KCOR real-data pipeline with quantile regression slope removal (tau = 0.10).

Input workbook schema per sheet (e.g., '2021_13', '2021_24', ...):
    ISOweekDied, DateDied, YearOfBirth, Sex, Dose, Alive, Dead

Outputs (one sheet per input sheet + "ALL"):
    Columns:
      Sheet, ISOweekDied, Date, YearOfBirth (0 = ASMR pooled), Dose_num, Dose_den,
      KCOR, MoE,
      MR_num, MR_adj_num, CMR_num, MR_den, MR_adj_den, CMR_den

Usage:
    python kcor_real_pipeline.py KCOR_output.xlsx KCOR_processed_REAL.xlsx
"""
import sys
import math
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

# Dependencies: statsmodels
try:
    import statsmodels.api as sm
except Exception as e:
    print("ERROR: statsmodels is required (pip install statsmodels).", e)
    sys.exit(1)

# ---------------- Config (adjust as needed) ----------------
PAIRS = [(1,0), (2,0), (2,1)]   # (numerator dose, denominator dose)
ANCHOR_WEEKS = 4                # anchor KCOR to 1 at this index if available else first row
SLOPE_WINDOW = 52               # weeks used for quantile regression slope fitting
TAU = 0.10                      # quantile level for QR
EPS = 1e-12                     # numerical floor to avoid log(0) - increased from 1e-18
MAX_SE_LOGK = 10.0             # maximum allowed SE(log K) to prevent overflow
# ----------------------------------------------------------

def safe_log(x, eps=EPS):
    """Safe logarithm with clipping to avoid log(0) or log(negative)."""
    return np.log(np.clip(x, eps, None))

def safe_exp(x, max_val=1e6):
    """Safe exponential with clipping to prevent overflow."""
    return np.clip(np.exp(x), 0, max_val)

def quantile_slope_log_y(t, y, tau=TAU):
    """Return slope b of log(y) ~ a + b*t via Quantile Regression at tau."""
    y = np.asarray(y, float).clip(min=EPS)
    t = np.asarray(t, float)
    
    # Check for sufficient data and variation
    if len(y) < 3:
        return 0.0
    
    if np.var(y) < EPS or np.var(t) < EPS:
        return 0.0
    
    # Check for all identical values
    if np.allclose(y, y[0], rtol=1e-10) or np.allclose(t, t[0], rtol=1e-10):
        return 0.0
    
    X = sm.add_constant(t)
    
    # Robust fit with multiple fallback strategies
    try:
        # Try quantile regression first
        res = sm.QuantReg(np.log(y), X).fit(q=tau, max_iter=1000)
        b = float(res.params[1])
        
        # Check if result is reasonable
        if not np.isfinite(b) or abs(b) > 100:
            raise ValueError("QR result unreasonable")
            
    except Exception as e:
        try:
            # Fallback to OLS with regularization
            X_reg = X.copy()
            X_reg[0, 0] += 1e-8  # Add small regularization to intercept
            a, b = np.linalg.lstsq(X_reg, np.log(y), rcond=1e-10)[0]
            b = float(b)
            
            # Check if result is reasonable
            if not np.isfinite(b) or abs(b) > 100:
                raise ValueError("OLS result unreasonable")
                
        except Exception:
            # Final fallback: simple linear regression
            try:
                # Use numpy's polyfit as last resort
                coeffs = np.polyfit(t, np.log(y), 1)
                b = float(coeffs[0])
                if not np.isfinite(b) or abs(b) > 100:
                    b = 0.0
            except Exception:
                b = 0.0
    
    return b

def compute_group_slopes_qr(df):
    """Slope per (YearOfBirth,Dose) on log(MR) over the last SLOPE_WINDOW observations."""
    slopes = {}
    for (yob, dose), g in df.groupby(["YearOfBirth","Dose"], sort=False):
        g2 = g.dropna(subset=["MR"])
        if g2.empty or len(g2) < 3:
            slopes[(yob,dose)] = 0.0
            continue
            
        gwin = g2.iloc[-SLOPE_WINDOW:] if len(g2) > SLOPE_WINDOW else g2
        
        # Additional data quality checks
        if gwin["MR"].var() < EPS or gwin["t"].var() < EPS:
            slopes[(yob,dose)] = 0.0
            continue
            
        slopes[(yob,dose)] = quantile_slope_log_y(gwin["t"].values, gwin["MR"].values, tau=TAU)
    return slopes

def adjust_mr(df, slopes, t0=ANCHOR_WEEKS):
    """Multiplicative slope removal on MR with anchoring at week index t0."""
    def f(row):
        b = slopes.get((row["YearOfBirth"], row["Dose"]), 0.0)
        # Clip the slope to prevent extreme adjustments
        b = np.clip(b, -10, 10)
        return row["MR"] * safe_exp(-b * (row["t"] - float(t0)))
    return df.assign(MR_adj=df.apply(f, axis=1))

def safe_sqrt(x, eps=EPS):
    """Safe square root with clipping."""
    return np.sqrt(np.clip(x, eps, None))

def build_kcor_rows(df, sheet_name):
    """
    Build per-age KCOR rows for all PAIRS and ASMR pooled rows (YearOfBirth=0).
    Assumptions:
      - Person-time PT = Alive
      - MR = Dead / PT
      - MR_adj slope-removed via QR
      - CMR = cumD_adj / cumPT
      - KCOR = (CMR_num / CMR_den), anchored to 1 at week ANCHOR_WEEKS if available
      - MoE uses SE(log K) ≈ sqrt(1/cumD_adj_num + 1/cumD_adj_den)
      - ASMR pooling uses fixed baseline weights = sum of PT in the first 4 weeks per age (time-invariant).
    """
    out_rows = []
    # Fast access by (age,dose)
    by_age_dose = {(y,d): g.sort_values("DateDied")
                   for (y,d), g in df.groupby(["YearOfBirth","Dose"], sort=False)}

    # -------- per-age KCOR rows --------
    for yob in df["YearOfBirth"].unique():
        for num, den in PAIRS:
            gv = by_age_dose.get((yob, num))
            gu = by_age_dose.get((yob, den))
            if gv is None or gu is None:
                continue
            merged = pd.merge(
                gv[["DateDied","ISOweekDied","MR","MR_adj","CMR","cumD_adj"]],
                gu[["DateDied","ISOweekDied","MR","MR_adj","CMR","cumD_adj"]],
                on="DateDied", suffixes=("_num","_den"), how="inner"
            ).sort_values("DateDied")
            if merged.empty:
                continue

            merged["K_raw"] = merged["CMR_num"] / (merged["CMR_den"] + EPS)
            t0_idx = ANCHOR_WEEKS if len(merged) > ANCHOR_WEEKS else 0
            anchor = merged["K_raw"].iloc[t0_idx]
            if not (np.isfinite(anchor) and anchor > EPS):
                anchor = 1.0
            merged["KCOR"] = merged["K_raw"] / anchor

            # Safe calculation of SE_logK with clipping
            merged["SE_logK"] = safe_sqrt(
                1.0/(merged["cumD_adj_num"] + EPS) + 1.0/(merged["cumD_adj_den"] + EPS)
            )
            
            # Clip SE_logK to prevent overflow in MoE calculation
            merged["SE_logK"] = np.clip(merged["SE_logK"], 0, MAX_SE_LOGK)
            
            # Safe MoE calculation
            merged["MoE"] = merged["KCOR"] * (safe_exp(1.96*merged["SE_logK"]) - 1.0)

            out = merged[["DateDied","ISOweekDied_num","KCOR","MoE",
                          "MR_num","MR_adj_num","CMR_num",
                          "MR_den","MR_adj_den","CMR_den"]].copy()
            out["Sheet"] = sheet_name
            out["YearOfBirth"] = yob
            out["Dose_num"] = num
            out["Dose_den"] = den
            out.rename(columns={"ISOweekDied_num":"ISOweekDied",
                                "DateDied":"Date"}, inplace=True)
            out_rows.append(out)

    # -------- ASMR pooled rows (YearOfBirth = 0) --------
    # Fixed baseline weights per age = sum of PT over the first 4 *distinct weeks* across all doses
    weights = {}
    df_sorted = df.sort_values("DateDied")
    for yob, g_age in df_sorted.groupby("YearOfBirth", sort=False):
        first_weeks = g_age.drop_duplicates(subset=["DateDied"]).head(4)
        weights[yob] = float(first_weeks["PT"].sum())

    pooled_rows = []
    all_dates = sorted(df_sorted["DateDied"].unique())

    for num, den in PAIRS:
        # Per-age anchors at t0 for this (num,den)
        anchors = {}
        for yob, g_age in df_sorted.groupby("YearOfBirth", sort=False):
            gvn = g_age[g_age["Dose"] == num].sort_values("DateDied")
            gdn = g_age[g_age["Dose"] == den].sort_values("DateDied")
            if gvn.empty or gdn.empty:
                continue
            t0_idx = ANCHOR_WEEKS if len(gvn) > ANCHOR_WEEKS and len(gdn) > ANCHOR_WEEKS else 0
            c1 = gvn["CMR"].iloc[t0_idx]
            c0 = gdn["CMR"].iloc[t0_idx]
            if np.isfinite(c1) and np.isfinite(c0) and c1 > EPS and c0 > EPS:
                anchors[yob] = c1 / c0

        for dt in all_dates:
            logs, wts, var_terms = [], [], []
            for yob, g_age in df_sorted.groupby("YearOfBirth", sort=False):
                if yob not in anchors:
                    continue
                gv = g_age[(g_age["Dose"]==num) & (g_age["DateDied"]==dt)]
                gu = g_age[(g_age["Dose"]==den) & (g_age["DateDied"]==dt)]
                if gv.empty or gu.empty:
                    continue
                k = (gv["CMR"].values[0]) / (gu["CMR"].values[0] + EPS)
                k0 = anchors[yob]
                if not (np.isfinite(k) and np.isfinite(k0) and k0 > EPS and k > EPS):
                    continue
                kstar = k / k0
                logs.append(safe_log(kstar))
                wts.append(weights.get(yob, 0.0))
                Dv = float(gv["cumD_adj"].values[0])
                Du = float(gu["cumD_adj"].values[0])
                if Dv > EPS and Du > EPS:
                    var_terms.append((weights.get(yob,0.0)**2) * (1.0/Dv + 1.0/Du))

            if logs and sum(wts) > 0:
                logK = np.average(logs, weights=wts)
                Kpool = float(safe_exp(logK))
                SE = safe_sqrt(sum(var_terms)) / sum(wts)
                # Clip SE to prevent overflow
                SE = min(SE, MAX_SE_LOGK)
                MoE = Kpool * (safe_exp(1.96*SE) - 1.0)
                pooled_rows.append({
                    "Sheet": sheet_name,
                    "ISOweekDied": df_sorted.loc[df_sorted["DateDied"]==dt, "ISOweekDied"].iloc[0],
                    "Date": dt,
                    "YearOfBirth": 0,      # ASMR pooled row
                    "Dose_num": num,
                    "Dose_den": den,
                    "KCOR": Kpool,
                    "MoE": MoE,
                    "MR_num": np.nan, "MR_adj_num": np.nan, "CMR_num": np.nan,
                    "MR_den": np.nan, "MR_adj_den": np.nan, "CMR_den": np.nan,
                })

    if out_rows or pooled_rows:
        return pd.concat(out_rows + [pd.DataFrame(pooled_rows)], ignore_index=True)
    return pd.DataFrame(columns=[
        "Sheet","ISOweekDied","Date","YearOfBirth","Dose_num","Dose_den",
        "KCOR","MoE","MR_num","MR_adj_num","CMR_num","MR_den","MR_adj_den","CMR_den"
    ])

def process_workbook(src_path: str, out_path: str):
    # Suppress specific warnings that we're handling
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='statsmodels')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandas')
    
    xls = pd.ExcelFile(src_path)
    all_out = []
    for sh in xls.sheet_names:
        print(f"[Info] Processing sheet: {sh}")
        df = pd.read_excel(src_path, sheet_name=sh)
        # prep
        df["DateDied"] = pd.to_datetime(df["DateDied"])
        df = df.sort_values(["YearOfBirth","Dose","DateDied"]).reset_index(drop=True)
        # person-time proxy and MR
        df["PT"]   = df["Alive"].astype(float).clip(lower=0.0)
        df["Dead"] = df["Dead"].astype(float).clip(lower=0.0)
        df["MR"]   = np.where(df["PT"] > 0, df["Dead"]/(df["PT"] + EPS), np.nan)
        df["t"]    = df.groupby(["YearOfBirth","Dose"]).cumcount().astype(float)

        # QR slope removal
        slopes = compute_group_slopes_qr(df)
        df = adjust_mr(df, slopes, t0=ANCHOR_WEEKS)

        # adjusted deaths, cumulative, CMR
        df["D_adj"]    = df["MR_adj"] * df["PT"]
        df["cumD_adj"] = df.groupby(["YearOfBirth","Dose"])["D_adj"].cumsum()
        df["cumPT"]    = df.groupby(["YearOfBirth","Dose"])["PT"].cumsum()
        df["CMR"]      = np.where(df["cumPT"] > 0, df["cumD_adj"]/df["cumPT"], np.nan)

        out_sh = build_kcor_rows(df, sh)
        all_out.append(out_sh)

    combined = pd.concat(all_out, ignore_index=True).sort_values(["Sheet","YearOfBirth","Dose_num","Dose_den","Date"])

    # write
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for sh in combined["Sheet"].unique():
            combined.loc[combined["Sheet"]==sh].to_excel(writer, index=False, sheet_name=sh)
        combined.to_excel(writer, index=False, sheet_name="ALL")

    print(f"[Done] Wrote {len(combined)} rows to {out_path}")
    return combined

def main():
    if len(sys.argv) < 3:
        print("Usage: python kcor_real_pipeline.py <input.xlsx> <output.xlsx>")
        sys.exit(2)
    src = sys.argv[1]
    dst = sys.argv[2]
    process_workbook(src, dst)

if __name__ == "__main__":
    main()
