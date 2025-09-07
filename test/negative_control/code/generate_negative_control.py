import os
import sys
import pandas as pd


def build_unvax_negative_control_sheet(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"ISOweekDied", "DateDied", "YearOfBirth", "Sex", "Dose", "Alive", "Dead"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input sheet missing required columns: {sorted(missing)}")

    dose_to_source_yob = {0: 1930, 1: 1940, 2: 1950}
    parts = []
    for target_dose, source_yob in dose_to_source_yob.items():
        src = df[(df["Dose"] == 0) & (df["YearOfBirth"] == source_yob)].copy()
        if src.empty:
            continue
        src["Dose"] = target_dose
        src["YearOfBirth"] = 1950
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
            out = build_unvax_negative_control_sheet(df)
            if not out.empty:
                out.to_excel(writer, index=False, sheet_name=sh)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_in = os.path.join(root, "data", "Czech", "KCOR_CMR.xlsx")
    default_out = os.path.join(root, "test", "negative_control", "data", "KCOR_synthetic_test.xlsx")

    src = sys.argv[1] if len(sys.argv) >= 2 else default_in
    dst = sys.argv[2] if len(sys.argv) >= 3 else default_out
    generate(src, dst)
    print(f"Wrote synthetic test workbook to {dst}")


