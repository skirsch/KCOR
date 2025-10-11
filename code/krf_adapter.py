"""
KRF → Czech Adapter

Call using make command: 

    make CMR_from_krf DATASET=japan   # or any other dataset name

Purpose: Convert a KRF CSV (ID, YearOfBirth, DeathDate, VnDate/VnBrand, etc.) into a
synthetic "Czech-like" CSV with the exact 53-column header KCOR_CMR.py expects.

Why: Lets us use the existing KCOR_CMR pipeline unchanged. The adapter maps:
- YearOfBirth → YearOfBirth (as 4-digit year string)
- Sex (M/F/O) → numeric codes expected by KCOR_CMR input (1=Male, 2=Female, else empty)
- DeathDate (YYYY-MM-DD) → ISO week string YYYY-WW in DateOfDeath
- VnDate → first 4 doses mapped to Date_FirstDose..Date_FourthDose as ISO week strings
- Fills the rest of the 53 columns with blanks

Usage:
  python3 krf_adapter.py <krf_input.csv> <czech_like_output.csv>
"""

import sys
import pandas as pd
from datetime import date


CZECH_HEADER = [
    'ID', 'Infection', 'Sex', 'YearOfBirth', 'DateOfPositiveTest', 'DateOfResult', 'Recovered', 'Date_COVID_death',
    'Symptom', 'TestType', 'Date_FirstDose', 'Date_SecondDose', 'Date_ThirdDose', 'Date_FourthDose',
    'Date_FifthDose', 'Date_SixthDose', 'Date_SeventhDose', 'VaccineCode_FirstDose', 'VaccineCode_SecondDose',
    'VaccineCode_ThirdDose', 'VaccineCode_FourthDose', 'VaccineCode_FifthDose', 'VaccineCode_SixthDose',
    'VaccineCode_SeventhDose', 'PrimaryCauseHospCOVID', 'bin_Hospitalization', 'min_Hospitalization',
    'days_Hospitalization', 'max_Hospitalization', 'bin_ICU', 'min_ICU', 'days_ICU', 'max_ICU', 'bin_StandardWard',
    'min_StandardWard', 'days_StandardWard', 'max_StandardWard', 'bin_Oxygen', 'min_Oxygen', 'days_Oxygen',
    'max_Oxygen', 'bin_HFNO', 'min_HFNO', 'days_HFNO', 'max_HFNO', 'bin_MechanicalVentilation_ECMO',
    'min_MechanicalVentilation_ECMO', 'days_MechanicalVentilation_ECMO', 'max_MechanicalVentilation_ECMO',
    'Mutation', 'DateOfDeath', 'Long_COVID', 'DCCI'
]


def to_iso_week_str(dt_series: pd.Series) -> pd.Series:
    """Convert a date series to ISO week strings YYYY-WW; blank for NaT."""
    dt = pd.to_datetime(dt_series, errors='coerce')
    out = pd.Series("", index=dt.index, dtype=object)
    mask = dt.notna()
    if mask.any():
        iso = dt.dt.isocalendar()
        year = iso['year']
        week = iso['week']
        out.loc[mask] = year[mask].astype(int).astype(str) + '-' + week[mask].astype(int).astype(str).str.zfill(2)
    return out


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python3 krf_adapter.py <krf_input.csv> <czech_like_output.csv>")
        sys.exit(1)
    krf_path, out_path = sys.argv[1], sys.argv[2]

    krf = pd.read_csv(krf_path, dtype=str, low_memory=False)
    req = {"ID", "YearOfBirth", "DeathDate"}
    if not req.issubset(set(krf.columns)):
        print(f"ERROR: Input does not look like KRF (missing required columns: {sorted(req - set(krf.columns))})")
        sys.exit(2)

    # Build base output with all empty strings
    out = pd.DataFrame({col: "" for col in CZECH_HEADER}, index=krf.index).copy()

    # Map straightforward fields
    out['ID'] = krf['ID'].astype(str)
    # Convert YearOfBirth numeric to 5-year bin labels (e.g., 1950-1954)
    try:
        yob_num = pd.to_numeric(krf['YearOfBirth'], errors='coerce')
        cohort_start = (yob_num // 5) * 5
        cohort_end = cohort_start + 4
        out['YearOfBirth'] = pd.Series(
            [f"{int(s)}-{int(e)}" if pd.notna(s) else "" for s, e in zip(cohort_start, cohort_end)],
            index=krf.index,
            dtype=object,
        )
    except Exception:
        out['YearOfBirth'] = krf['YearOfBirth'].astype(str)
    # Sex: KRF M/F/O → Czech numeric codes: 1 male, 2 female, else blank
    sex_map = {'M': '1', 'F': '2'}
    if 'Sex' in krf.columns:
        out['Sex'] = krf['Sex'].map(sex_map).fillna("")

    # DCCI: Japan has no DCCI; set explicit UNKNOWN = -1 for all rows
    out['DCCI'] = '-1'

    # DeathDate to ISO week string (YYYY-WW)
    out['DateOfDeath'] = to_iso_week_str(krf['DeathDate'])

    # Gather vaccination dates; pick first 4 by chronological order
    vdate_cols = [c for c in krf.columns if c.startswith('V') and c.endswith('Date')]
    if vdate_cols:
        # Melt to long with a stable row id
        vd = krf.reset_index()[['index'] + vdate_cols].rename(columns={'index': 'row_id'})
        long = vd.melt(id_vars='row_id', var_name='vcol', value_name='date')
        long['date'] = pd.to_datetime(long['date'], errors='coerce')
        long = long.dropna(subset=['date']).sort_values(['row_id', 'date'])
        long['rank'] = long.groupby('row_id').cumcount() + 1
        first4 = long[long['rank'] <= 4].copy()
        if not first4.empty:
            first4['week'] = to_iso_week_str(first4['date'])
            week_wide = first4.pivot(index='row_id', columns='rank', values='week')
            week_wide.columns = [
                f'Date_{name}' for name in ['FirstDose', 'SecondDose', 'ThirdDose', 'FourthDose'][: len(week_wide.columns)]
            ]
            # Map back by row_id to original index
            for c in week_wide.columns:
                out.loc[week_wide.index, c] = week_wide[c].astype(str)

    # Write CSV with exact header
    out = out[CZECH_HEADER]
    out.to_csv(out_path, index=False, encoding='utf-8')
    print(f"KRF adapter wrote: {out_path}")


if __name__ == '__main__':
    main()


