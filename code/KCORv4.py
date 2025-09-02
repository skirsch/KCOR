
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
    python KCORv4.py KCOR_output.xlsx KCOR_processed_REAL.xlsx
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
PAIRS = [(2,0)]   # (numerator dose, denominator dose) - only 2 vs 0 for debugging
ANCHOR_WEEKS = 4                # anchor KCOR to 1 at this index if available else first row
TAU = 0.10                      # quantile level for QR
EPS = 1e-12                     # numerical floor to avoid log(0) - increased from 1e-18
MAX_SE_LOGK = 10.0             # maximum allowed SE(log K) to prevent overflow

# Slope calculation using lookup table method
SLOPE_LOOKUP_TABLE = {
    '2021_24': (53, 114)  # (offset1, offset2) from enrollment for slope calculation
}
MA_TOTAL_LENGTH = 8  # Total length of centered moving average (8 weeks = 4 weeks on either side)
CENTERED = True  # Use centered moving average (4 weeks before + 4 weeks after each point)

# Debug parameters - limit scope for debugging
DEBUG_AGE_ONLY = 1940           # Only process this age group (set to None to process all)
DEBUG_SHEET_ONLY = "2021_24"   # Only process this sheet (set to None to process all)
DEBUG_DOSE_PAIR_ONLY = (2, 0)  # Only process this dose pair (set to None to process all)
DEBUG_VERBOSE = True            # Print detailed debugging info for each date
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

def compute_group_slopes_lookup(df, sheet_name):
    """Slope per (YearOfBirth,Dose) using lookup table method."""
    slopes = {}
    
    # Get lookup table values for this sheet
    if sheet_name not in SLOPE_LOOKUP_TABLE:
        print(f"[WARNING] No lookup table entry for sheet {sheet_name}, using default slope 0.0")
        for (yob, dose), g in df.groupby(["YearOfBirth","Dose"], sort=False):
            slopes[(yob,dose)] = 0.0
        return slopes
    
    offset1, offset2 = SLOPE_LOOKUP_TABLE[sheet_name]
    T = offset2 - offset1  # Time difference between the two points
    
    for (yob, dose), g in df.groupby(["YearOfBirth","Dose"], sort=False):
        g_sorted = g.sort_values("t")
        
        # Find MR values at the two lookup points
        point1 = g_sorted[g_sorted["t"] == offset1]
        point2 = g_sorted[g_sorted["t"] == offset2]
        
        if point1.empty or point2.empty:
            slopes[(yob,dose)] = 0.0
            # if DEBUG_VERBOSE and (yob, dose) in [(1940, 0), (1940, 2)]:
            #     print(f"  [DEBUG] Missing lookup points for Age {yob}, Dose {dose}: t={offset1} or t={offset2}")
            continue
        
        # Get MR values at the two points
        A = point1["MR_smooth"].iloc[0]  # MR at offset1
        B = point2["MR_smooth"].iloc[0]  # MR at offset2
        
        # Calculate exponential slope: slope = ln(B/A) / T
        if A > EPS and B > EPS:
            slope = (np.log(B) - np.log(A)) / T
        else:
            slope = 0.0
        
        slopes[(yob,dose)] = slope
        
        # Debug: Show lookup table calculation
        # if DEBUG_VERBOSE and (yob, dose) in [(1940, 0), (1940, 2)]:
        #     print(f"  [DEBUG] Lookup slope calculation for Age {yob}, Dose {dose}:")
        #     print(f"    Point 1: t={offset1}, MR={A:.6f}")
        #     print(f"    Point 2: t={offset2}, MR={B:.6f}")
        #     print(f"    T={T}, slope = ln({B:.6f}/{A:.6f})/{T} = {slope:.6f}")
    
    return slopes

def adjust_mr(df, slopes, t0=ANCHOR_WEEKS):
    """Multiplicative slope removal on MR with anchoring at week index t0."""
    def f(row):
        b = slopes.get((row["YearOfBirth"], row["Dose"]), 0.0)
        return row["MR"] * safe_exp(-b * (row["t"] - float(t0)))
    return df.assign(MR_adj=df.apply(f, axis=1))

def safe_sqrt(x, eps=EPS):
    """Safe square root with clipping."""
    return np.sqrt(np.clip(x, eps, None))

def apply_moving_average(df, window=MA_TOTAL_LENGTH, centered=CENTERED):
    """Apply centered moving average smoothing to MR values before quantile regression.
    
    Args:
        window: Total length of moving average (e.g., 8 weeks = 4 weeks on either side)
        centered: If True, use centered MA (4 weeks before + 4 weeks after each point)
    """
    df_smooth = df.copy()
    
    for (yob, dose), g in df.groupby(["YearOfBirth","Dose"], sort=False):
        # Sort by time to ensure proper order
        g_sorted = g.sort_values("t")
        
        # Apply centered moving average to MR values
        if centered:
            g_sorted["MR_smooth"] = g_sorted["MR"].rolling(window=window, center=True, min_periods=1).mean()
        else:
            g_sorted["MR_smooth"] = g_sorted["MR"].rolling(window=window, center=False, min_periods=1).mean()
        
        # Update the original dataframe
        df_smooth.loc[g_sorted.index, "MR_smooth"] = g_sorted["MR_smooth"]
    
    return df_smooth

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
            # Apply debug dose pair filter
            if DEBUG_DOSE_PAIR_ONLY and (num, den) != DEBUG_DOSE_PAIR_ONLY:
                continue
            gv = by_age_dose.get((yob, num))
            gu = by_age_dose.get((yob, den))
            if gv is None or gu is None:
                continue
            # Ensure we have exactly one row per date by taking the first occurrence
            gv_unique = gv[["DateDied","ISOweekDied","MR","MR_adj","CMR","cumD_adj"]].drop_duplicates(subset=["DateDied"], keep="first")
            gu_unique = gu[["DateDied","ISOweekDied","MR","MR_adj","CMR","cumD_adj"]].drop_duplicates(subset=["DateDied"], keep="first")
            
            merged = pd.merge(
                gv_unique,
                gu_unique,
                on="DateDied", suffixes=("_num","_den"), how="inner"
            ).sort_values("DateDied")
            if merged.empty:
                continue
                
            # Debug: Print detailed info for each date
            # if DEBUG_VERBOSE:
            #     print(f"\n[DEBUG] Age {yob}, Doses {num} vs {den}")
            #     print(f"  Number of merged rows: {len(merged)}")
            #     print(f"  Date range: {merged['DateDied'].min().strftime('%m/%d/%Y')} to {merged['DateDied'].max().strftime('%m/%d/%Y')}")
            #     
            #     # Check for duplicate dates
            #     date_counts = merged['DateDied'].value_counts()
            #     if date_counts.max() > 1:
            #         print(f"  WARNING: Found duplicate dates!")
            #         print(f"    Max rows per date: {date_counts.max()}")
            #         print(f"    Dates with multiple rows: {len(date_counts[date_counts > 1])}")
            #         print(f"    Sample duplicate dates:")
            #         for date, count in date_counts[date_counts > 1].head(3).items():
            #             print(f"      {date.strftime('%m/%d/%Y')}: {count} rows")
            #     
            #     print(f"  Sample data points:")
            #     for i, row in merged.head(5).iterrows():
            #         print(f"    Date: {row['DateDied'].strftime('%m/%d/%Y')}")
            #         print(f"      Dose {num}: MR={row['MR_num']:.6f}, MR_adj={row['MR_adj_num']:.6f}, CMR={row['CMR_num']:.6f}, cumD_adj={row['cumD_adj_num']:.6f}")
            #         print(f"      Dose {den}: MR={row['MR_den']:.6f}, MR_adj={row['MR_adj_den']:.6f}, CMR={row['CMR_den']:.6f}, cumD_adj={row['cumD_adj_den']:.6f}")
            #     print()

            # Handle division by zero or very small denominators
            valid_denom = merged["CMR_den"] > EPS
            merged["K_raw"] = np.where(valid_denom, 
                                      merged["CMR_num"] / merged["CMR_den"], 
                                      np.nan)
            t0_idx = ANCHOR_WEEKS if len(merged) > ANCHOR_WEEKS else 0
            anchor = merged["K_raw"].iloc[t0_idx]
            if not (np.isfinite(anchor) and anchor > EPS):
                anchor = 1.0
            merged["KCOR"] = np.where(np.isfinite(merged["K_raw"]), 
                                     merged["K_raw"] / anchor, 
                                     np.nan)
            

            
            # Debug: Check for suspiciously large KCOR values
            # if merged["KCOR"].max() > 10:
            #     print(f"\n[DEBUG] Large KCOR detected in {sheet_name}, Age {yob}, Doses {num} vs {den}")
            #     print(f"  CMR_num range: {merged['CMR_num'].min():.6f} to {merged['CMR_num'].max():.6f}")
            #     print(f"  CMR_den range: {merged['CMR_den'].min():.6f} to {merged['CMR_den'].max():.6f}")
            #     print(f"  K_raw range: {merged['K_raw'].min():.6f} to {merged['K_raw'].max():.6f}")
            #     print(f"  Anchor value: {anchor:.6f}")
            #     print(f"  KCOR range: {merged['KCOR'].min():.6f} to {merged['KCOR'].max():.6f}")
            #     print(f"  Sample data points:")
            #     for i, row in merged.head(3).iterrows():
            #         print(f"    Date: {row['DateDied']}, CMR_num: {row['CMR_num']:.6f}, CMR_den: {row['CMR_den']:.6f}, K_raw: {row['K_raw']:.6f}, KCOR: {row['KCOR']:.6f}")
            #     print()

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
            # Convert Date to standard pandas date format (same as debug sheet)
            out["Date"] = pd.to_datetime(out["Date"]).apply(lambda x: x.date())
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
                # Take first occurrence if multiple rows per date
                gv = gv.iloc[[0]]
                gu = gu.iloc[[0]]
                denom_cmr = gu["CMR"].values[0]
                if denom_cmr > EPS:
                    k = (gv["CMR"].values[0]) / denom_cmr
                else:
                    continue  # Skip this comparison if denominator is too small
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
                

                
                # Debug: Check for suspiciously large pooled KCOR values
                if Kpool > 10:
                    print(f"\n[DEBUG] Large pooled KCOR detected: {Kpool:.6f}")
                    print(f"  Dose combination: {num} vs {den}, Date: {dt}")
                    print(f"  Number of age groups: {len(logs)}")
                    print(f"  Log K values: {[f'{x:.6f}' for x in logs[:5]]}...")
                    print(f"  Weights: {[f'{w:.6f}' for w in wts[:5]]}...")
                    print(f"  Final logK: {logK:.6f}")
                    print()
                pooled_rows.append({
                    "Sheet": sheet_name,
                    "ISOweekDied": df_sorted.loc[df_sorted["DateDied"]==dt, "ISOweekDied"].iloc[0],
                    "Date": dt.date(),  # Convert to standard pandas date format (same as debug sheet)
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
    
    # Apply debug filters
    sheets_to_process = [DEBUG_SHEET_ONLY] if DEBUG_SHEET_ONLY else xls.sheet_names
    if DEBUG_AGE_ONLY:
        print(f"[DEBUG] Limiting to age group: {DEBUG_AGE_ONLY}")
    if DEBUG_SHEET_ONLY:
        print(f"[DEBUG] Limiting to sheet: {DEBUG_SHEET_ONLY}")
    
    for sh in sheets_to_process:
        print(f"[Info] Processing sheet: {sh}")
        df = pd.read_excel(src_path, sheet_name=sh)
        # prep
        df["DateDied"] = pd.to_datetime(df["DateDied"])
        
        # Filter out unreasonably large birth years (keep -1 for "not available")
        df = df[df["YearOfBirth"] <= 2020]
        
        # Filter to start from enrollment date (sheet name format: YYYY_WW)
        if "_" in sh:
            year_str, week_str = sh.split("_")
            enrollment_year = int(year_str)
            enrollment_week = int(week_str)
            # Convert ISO week to date more accurately
            from datetime import datetime, timedelta
            # ISO week 24 of 2021 should be around June 14, 2021
            # Let's use a more precise calculation
            jan1 = datetime(enrollment_year, 1, 1)
            # Find the first Monday of the year (ISO week 1)
            days_to_monday = (7 - jan1.weekday()) % 7
            if days_to_monday == 0 and jan1.weekday() != 0:
                days_to_monday = 7
            first_monday = jan1 + timedelta(days=days_to_monday)
            # Calculate the start of the specified ISO week
            enrollment_date = first_monday + timedelta(weeks=enrollment_week-1)
            df = df[df["DateDied"] >= enrollment_date]
            print(f"[DEBUG] Filtered to start from enrollment date {enrollment_date.strftime('%m/%d/%Y')}: {len(df)} rows")
        
        # Apply debug age filter
        if DEBUG_AGE_ONLY:
            df = df[df["YearOfBirth"] == DEBUG_AGE_ONLY]
            print(f"[DEBUG] Filtered to {len(df)} rows for age {DEBUG_AGE_ONLY}")
        
        # Apply debug dose filter - only process doses 0 and 2
        df = df[df["Dose"].isin([0, 2])]
        print(f"[DEBUG] Filtered to doses 0 and 2: {len(df)} rows")
        
        # Aggregate across sexes for each dose/date/age combination
        df = df.groupby(["YearOfBirth", "Dose", "DateDied"]).agg({
            "ISOweekDied": "first",
            "Alive": "sum",
            "Dead": "sum"
        }).reset_index()
        print(f"[DEBUG] Aggregated across sexes: {len(df)} rows")
        
        df = df.sort_values(["YearOfBirth","Dose","DateDied"]).reset_index(drop=True)
        # person-time proxy and MR
        df["PT"]   = df["Alive"].astype(float).clip(lower=0.0)
        df["Dead"] = df["Dead"].astype(float).clip(lower=0.0)
        df["MR"]   = np.where(df["PT"] > 0, df["Dead"]/(df["PT"] + EPS), np.nan)
        df["t"]    = df.groupby(["YearOfBirth","Dose"]).cumcount().astype(float)

        # Apply centered moving average smoothing before quantile regression
        df = apply_moving_average(df, window=MA_TOTAL_LENGTH, centered=CENTERED)
        
        # Debug: Show effect of moving average
        # if DEBUG_VERBOSE:
        #     print(f"\n[DEBUG] Centered moving average smoothing (total length={MA_TOTAL_LENGTH} weeks, centered={CENTERED}):")
        #     for (yob, dose), g in df.groupby(["YearOfBirth","Dose"], sort=False):
        #         if len(g) > 0:
        #             mr_orig = g["MR"].values
        #             mr_smooth = g["MR_smooth"].values
        #             print(f"  Age {yob}, Dose {dose}:")
        #             print(f"    Original MR range: {mr_orig.min():.6f} to {mr_orig.max():.6f}")
        #             print(f"    Smoothed MR range: {mr_smooth.min():.6f} to {mr_smooth.max():.6f}")
        
        # Lookup table slope calculation (using smoothed MR values)
        slopes = compute_group_slopes_lookup(df, sh)
        
        # Debug: Show computed slopes
        if DEBUG_VERBOSE:
            print(f"\n[DEBUG] Computed slopes for sheet {sh}:")
            for (yob, dose), slope in slopes.items():
                print(f"  Age {yob}, Dose {dose}: slope = {slope:.6f}")
            print()
        
        df = adjust_mr(df, slopes, t0=ANCHOR_WEEKS)
        
        # Debug: Show MR values week by week, especially weeks with no deaths
        # if DEBUG_VERBOSE:
        #     print(f"\n[DEBUG] MR values by week for sheet {sh}:")
        #     for dose in df["Dose"].unique():
        #         dose_data = df[df["Dose"] == dose].sort_values("DateDied")
        #         print(f"  Dose {dose}:")
        #         print(f"    Original MR range: {dose_data['MR'].min():.6f} to {dose_data['MR'].max():.6f}")
        #         print(f"    Adjusted MR range: {dose_data['MR_adj'].min():.6f} to {dose_data['MR_adj'].max():.6f}")
        #     
        #     # Show weeks with extreme MR_adj values
        #     extreme_mr = dose_data[dose_data["MR_adj"] > 100]
        #     if not extreme_mr.empty:
        #         print(f"    WARNING: {len(extreme_mr)} weeks with MR_adj > 100:")
        #         for _, row in extreme_mr.head(5).iterrows():
        #             print(f"      Date: {row['DateDied']}, Original MR: {row['MR']:.6f}, MR_adj: {row['MR_adj']:.6f}, PT: {row['PT']:.6f}, Dead: {row['Dead']:.6f}")
        #     
        #     # Show weeks with no deaths but high MR_adj
        #     no_deaths_high_mr = dose_data[(dose_data["Dead"] == 0) & (dose_data["MR_adj"] > 10)]
        #     if not no_deaths_high_mr.empty:
        #         print(f"    WARNING: {len(no_deaths_high_mr)} weeks with no deaths but MR_adj > 10:")
        #         for _, row in no_deaths_high_mr.head(5).iterrows():
        #             print(f"      Date: {row['DateDied']}, Original MR: {row['MR']:.6f}, MR_adj: {row['MR_adj']:.6f}, PT: {row['PT']:.6f}, Dead: {row['Dead']:.6f}")
        #     
        #     print()

        # adjusted deaths, cumulative, CMR
        df["D_adj"]    = df["MR_adj"] * df["PT"]
        
        # For the first entry (week 0), use original values without adjustments
        first_week_mask = df["t"] == 0
        df.loc[first_week_mask, "MR_adj"] = df.loc[first_week_mask, "MR"]
        df.loc[first_week_mask, "D_adj"] = df.loc[first_week_mask, "Dead"]
        
        # Reset cumulative counters to start from enrollment date (week 0)
        df["cumD_adj"] = df.groupby(["YearOfBirth","Dose"])["D_adj"].cumsum()
        df["cumPT"]    = df.groupby(["YearOfBirth","Dose"])["PT"].cumsum()
        
        df["CMR"] = np.where(df["cumPT"] > 0, df["cumD_adj"]/df["cumPT"], np.nan)
        
        # Debug: Print detailed CMR calculation info
        # if DEBUG_VERBOSE:
        #     print(f"\n[DEBUG] CMR calculation details for sheet {sh}:")
        #     for dose in df["Dose"].unique():
        #         dose_data = df[df["Dose"] == dose]
        #         if not dose_data.empty:
        #             print(f"  Dose {dose}:")
        #             print(f"    PT range: {dose_data['PT'].min():.6f} to {dose_data['PT'].max():.6f}")
        #             print(f"    MR_adj range: {dose_data['MR_adj'].min():.6f} to {dose_data['MR_adj'].max():.6f}")
        #             print(f"    D_adj range: {dose_data['D_adj'].min():.6f} to {dose_data['D_adj'].max():.6f}")
        #             print(f"    cumD_adj range: {dose_data['cumD_adj'].min():.6f} to {dose_data['cumD_adj'].max():.6f}")
        #             print(f"    cumPT range: {dose_data['cumPT'].min():.6f} to {dose_data['cumPT'].max():.6f}")
        #             print(f"    CMR range: {dose_data['CMR'].min():.6f} to {dose_data['CMR'].max():.6f}")
        #             print()
        
        # Debug: Check for extreme CMR values
        # extreme_cmr = df[df["CMR"] > 1000]
        # if not extreme_cmr.empty:
        #     print(f"\n[DEBUG] Extreme CMR values detected in sheet {sh}:")
        #     for _, row in extreme_cmr.head(3).iterrows():
        #         print(f"  Age: {row['YearOfBirth']}, Dose: {row['Dose']}, CMR: {row['CMR']:.6f}")
        #         print(f"    cumD_adj: {row['cumD_adj']:.6f}, cumPT: {row['cumPT']:.6f}")
        #         print(f"    MR_adj: {row['MR_adj']:.6f}, PT: {row['PT']:.6f}")
        #     print()

        out_sh = build_kcor_rows(df, sh)
        all_out.append(out_sh)

    combined = pd.concat(all_out, ignore_index=True).sort_values(["Sheet","YearOfBirth","Dose_num","Dose_den","Date"])

    # Debug: Show Date column info for main sheets
    # print(f"\n[DEBUG] Main sheets Date column info:")
    # print(f"  Date column dtype: {combined['Date'].dtype}")
    # print(f"  Sample Date values: {combined['Date'].head(3).tolist()}")
    # print(f"  Date range: {combined['Date'].min()} to {combined['Date'].max()}")
    


    # Report KCOR values at end of 2022 for each dose combo and age
    print("\n" + "="*80)
    print("KCOR VALUES AT END OF 2022 - 2021_24 SHEET ONLY")
    print("="*80)
    
    # Filter for end of 2022 (last date in 2022) and ONLY 2021_24 sheet
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined_2022 = combined[
        (combined["Date"].dt.year == 2022) & 
        (combined["Sheet"] == "2021_24")
    ]
    
    if not combined_2022.empty:
        # Get the last date in 2022
        last_date_2022 = combined_2022["Date"].max()
        print(f"Last date in 2022: {last_date_2022.strftime('%Y-%m-%d')}")
        print(f"Sheet: 2021_24")
        
        # Filter for that specific date
        end_2022_data = combined_2022[combined_2022["Date"] == last_date_2022]
        
        # Group by dose combinations and show results
        for (dose_num, dose_den) in PAIRS:
            print(f"\nDose combination: {dose_num} vs {dose_den}")
            print("-" * 50)
            
            # Get data for this dose combination
            dose_data = end_2022_data[
                (end_2022_data["Dose_num"] == dose_num) & 
                (end_2022_data["Dose_den"] == dose_den)
            ]
            
            if dose_data.empty:
                print("  No data available for this dose combination")
                continue
            
            # Show results by age (including ASMR = 0)
            for age in sorted(dose_data["YearOfBirth"].unique()):
                age_data = dose_data[dose_data["YearOfBirth"] == age]
                if not age_data.empty:
                    kcor_val = age_data["KCOR"].iloc[0]
                    moe_val = age_data["MoE"].iloc[0]
                    sheet_name = age_data["Sheet"].iloc[0]
                    
                    if age == 0:
                        age_label = "ASMR (pooled)"
                    elif age == -1:
                        age_label = "Age (unknown)"
                    else:
                        age_label = f"Age {age}"
                    
                    print(f"  {age_label:15} | KCOR: {kcor_val:8.4f} | MoE: {moe_val:8.4f} | Sheet: {sheet_name}")
    else:
        print("No data available for 2022 in 2021_24 sheet")
    
    print("="*80)

    # Create debug sheet with individual dose curves
    if DEBUG_VERBOSE:
        debug_data = []
        for dose in [0, 2]:
            # Aggregate across sexes for each dose/date combination
            dose_data = df[df["Dose"] == dose].groupby("DateDied").agg({
                "ISOweekDied": "first",
                "Dead": "sum",
                "Alive": "sum", 
                "PT": "sum",
                "MR": "mean",  # Average MR across sexes
                "MR_adj": "mean",  # Average MR_adj across sexes
                "CMR": "mean",  # Average CMR across sexes
                "cumD_adj": "sum",  # Sum cumulative deaths
                "cumPT": "sum",  # Sum cumulative person-time
                "MR_smooth": "mean",  # Average smoothed MR across sexes
                "t": "first"  # Time index (should be same for all sexes)
            }).reset_index().sort_values("DateDied")
            
            for _, row in dose_data.iterrows():
                # Calculate smoothed adjusted MR using the slope
                # Since we're only processing age 1940, use that for slope lookup
                slope = slopes.get((1940, dose), 0.0)
                smoothed_adj_mr = row["MR_smooth"] * safe_exp(-slope * (row["t"] - float(ANCHOR_WEEKS)))
                
                debug_data.append({
                    "Date": row["DateDied"].date(),  # Use standard pandas date format (was working perfectly - DON'T TOUCH)
                    "Dose": dose,  # Add dose column
                    "ISOweek": row["ISOweekDied"],
                    "Dead": row["Dead"],
                    "Alive": row["Alive"],
                    "MR": row["MR"],
                    "MR_adj": row["MR_adj"],
                    "Cum_MR": row["CMR"],
                    "Cumu_Adj_Deaths": row["cumD_adj"],
                    "Cumu_Person_Time": row["cumPT"],
                    "Smoothed_Raw_MR": row["MR_smooth"],
                    "Smoothed_Adjusted_MR": smoothed_adj_mr
                })
        
        debug_df = pd.DataFrame(debug_data)
        # print(f"\n[DEBUG] Created debug sheet with {len(debug_df)} rows")
        # print(f"  Date range: {debug_df['Date'].min()} to {debug_df['Date'].max()}")
        # print(f"  Date column dtype: {debug_df['Date'].dtype}")
        # print(f"  Sample Date values: {debug_df['Date'].head(3).tolist()}")
        # print(f"  Doses included: {debug_df['ISOweek'].nunique()} unique ISO weeks")
        

    
    # write
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Add debug sheet first (working format)
        if DEBUG_VERBOSE:
            debug_df.to_excel(writer, index=False, sheet_name="DEBUG_1940_curves")
        
        # Write main sheets
        for sh in combined["Sheet"].unique():
            sheet_data = combined.loc[combined["Sheet"]==sh].copy()
            # Ensure Date column stays as date objects (not timestamps)
            sheet_data["Date"] = sheet_data["Date"].apply(lambda x: x.date() if hasattr(x, 'date') else x)
            sheet_data.to_excel(writer, index=False, sheet_name=sh)
        
        # Ensure ALL sheet Date column stays as date objects
        all_data = combined.copy()
        all_data["Date"] = all_data["Date"].apply(lambda x: x.date() if hasattr(x, 'date') else x)
        all_data.to_excel(writer, index=False, sheet_name="ALL")

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
