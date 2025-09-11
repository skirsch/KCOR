# generate_negative_control.py

# Author: Steve Kirsch
# Date: 2025-09-07
# Version: 1.0


# Description: This program generates the negative control test file for the KCOR analysis.
# It is used to compare the KCOR values of the unvaccinated with the unvaccinated, but different ages.
# It is also used to compare the KCOR values of the vaccinated with the vaccinated, but different ages.
# Run from the main Makefile as make test,

# The unvaxxed appears as age 1950, so it should have the most reliable 1.0 signal.
# The vaxxed will have differences with age, so the signal there will deviate from 1.0.

# The thing to keep in mind is that the source data is not random and COVID was a non-proportional hazard
# so we need to be careful about how we interpret these negative control tests because the signal
# they show here is likely a real signal, not a negative control failure.

# on ideal data, KCOR will always return 1.0, but on real data it will deviate from 1.0.


import os
import sys
import pandas as pd


def build_negative_control_sheet(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Build a synthetic sheet per README groupings.

    mode:
      - 'unvax': use Dose==0 source rows; constant YoB=1950
      - 'vax2' : use Dose==2 source rows; constant YoB=1940
    Mapping (target dose -> source YoB set):
      0 -> {1930, 1935}
      1 -> {1940, 1945}
      2 -> {1950, 1955}
    """
    required_cols = {"ISOweekDied", "DateDied", "YearOfBirth", "Sex", "Dose", "Alive", "Dead"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input sheet missing required columns: {sorted(missing)}")

    if mode not in {"unvax", "vax2", "total", "all"}:
        raise ValueError("mode must be 'unvax', 'vax2', or 'total'")

    is_unvax = mode == "unvax"
    is_vax2  = mode == "vax2"
    is_total = mode in {"total", "all"}

    source_dose = 0 if is_unvax else (2 if is_vax2 else None)
    constant_yob = 1950 if is_unvax else (1940 if is_vax2 else 1960)

    target_to_source_yobs = {0: {1930, 1935}, 1: {1940, 1945}, 2: {1950, 1955}}

    parts = []
    for target_dose, yob_set in target_to_source_yobs.items():
        if is_total:
            src = df[df["YearOfBirth"].isin(list(yob_set))].copy()
            # Sum across original doses to get TOTAL for these cohorts
            for col in ["Alive", "Dead"]:
                src[col] = pd.to_numeric(src[col], errors="coerce").fillna(0)
            src = src.groupby(["ISOweekDied","DateDied","YearOfBirth","Sex"], as_index=False)[["Alive","Dead"]].sum()
            # After summing across doses, set the synthetic cohort YoB to 1960
            src["YearOfBirth"] = 1960
        else:
            src = df[(df["Dose"] == source_dose) & (df["YearOfBirth"].isin(list(yob_set)))].copy()
        if src.empty:
            continue
        src["Dose"] = target_dose
        src["YearOfBirth"] = constant_yob
        parts.append(src)

    if not parts:
        return df.iloc[0:0][["ISOweekDied", "DateDied", "YearOfBirth", "Sex", "Dose", "Alive", "Dead"]]

    out = pd.concat(parts, ignore_index=True)
    out["Dose"] = out["Dose"].astype(int)
    out["YearOfBirth"] = out["YearOfBirth"].astype(int)
    return out[["ISOweekDied", "DateDied", "YearOfBirth", "Sex", "Dose", "Alive", "Dead"]]


def generate(input_path: str, output_path: str, sheets=None) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    xls = pd.ExcelFile(input_path)

    default_target_sheets = ["2021_24", "2022_06"]
    sheet_names = sheets if sheets else [s for s in xls.sheet_names if s in default_target_sheets]
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        for sh in sheet_names:
            df = pd.read_excel(xls, sheet_name=sh)
            df.columns = [c.strip() for c in df.columns]
            if "Dose" in df.columns:
                df["Dose"] = pd.to_numeric(df["Dose"], errors="coerce").fillna(-1).astype(int)
            if "YearOfBirth" in df.columns:
                df["YearOfBirth"] = pd.to_numeric(df["YearOfBirth"], errors="coerce").fillna(-1).astype(int)
            # Always build all three cohorts: 1950 from unvax, 1940 from vax dose 2, 1960 from TOTAL
            out_unvax = build_negative_control_sheet(df, "unvax")
            out_vax2  = build_negative_control_sheet(df, "vax2")
            out_total = build_negative_control_sheet(df, "total")
            out = pd.concat([out_unvax, out_vax2, out_total], ignore_index=True)
            if not out.empty:
                out.to_excel(writer, index=False, sheet_name=sh)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_in = os.path.join(root, "data", "Czech", "KCOR_CMR.xlsx")
    default_out = os.path.join(root, "test", "negative_control", "data", "KCOR_synthetic_neg_control.xlsx")

    src = sys.argv[1] if len(sys.argv) >= 2 else default_in
    dst = sys.argv[2] if len(sys.argv) >= 3 else default_out
    # Optional third argument may be sheets (comma-separated). Backward compat: if it looks like a mode word,
    # ignore it and take sheets from the fourth argument.
    arg3 = sys.argv[3] if len(sys.argv) >= 4 else None
    mode_words = {"unvax","vax2","total","both","all"}
    if arg3 and arg3 in mode_words:
        sheets_arg = sys.argv[4] if len(sys.argv) >= 5 else None
    else:
        sheets_arg = arg3
    sheets = [s.strip() for s in sheets_arg.split(',')] if sheets_arg else None
    generate(src, dst, sheets)
    print(f"Wrote synthetic (all cohorts) test workbook to {dst}")


