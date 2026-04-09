"""Load Czech population CSV; map columns to English (same as KCOR_ts / KCOR_variable)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd

EXPECTED_COL_COUNT = 53

# After English rename (before enrollment engineering)
REQUIRED_COLUMNS_AFTER_LOAD = [
    "ID",
    "Infection",
    "YearOfBirth",
    "DateOfPositiveTest",
    "Date_COVID_death",
    "Date_FirstDose",
    "Date_SecondDose",
    "Date_ThirdDose",
    "DateOfDeath",
    "VaccineCode_FirstDose",
    "VaccineCode_SecondDose",
    "VaccineCode_ThirdDose",
    "VaccineCode_FourthDose",
]

# After build_enrollment_table
REQUIRED_COLUMNS_AFTER_ENROLLMENT = [
    "age_bin",
    "birth_band_start",
    "enrollment_monday",
    "cohort_dose0",
    "cohort_dose1",
    "cohort_dose2",
    "cohort_dose3",
    "infection_monday",
    "death_monday",
    "covid_death_monday",
    "death_monday_allcause",
    "first_dose_monday",
    "second_dose_monday",
    "third_dose_monday",
    "weeks_second_dose_to_enrollment",
]

CZECH_TO_ENGLISH_COLUMNS = [
    "ID",
    "Infection",
    "Sex",
    "YearOfBirth",
    "DateOfPositiveTest",
    "DateOfResult",
    "Recovered",
    "Date_COVID_death",
    "Symptom",
    "TestType",
    "Date_FirstDose",
    "Date_SecondDose",
    "Date_ThirdDose",
    "Date_FourthDose",
    "Date_FifthDose",
    "Date_SixthDose",
    "Date_SeventhDose",
    "VaccineCode_FirstDose",
    "VaccineCode_SecondDose",
    "VaccineCode_ThirdDose",
    "VaccineCode_FourthDose",
    "VaccineCode_FifthDose",
    "VaccineCode_SixthDose",
    "VaccineCode_SeventhDose",
    "PrimaryCauseHospCOVID",
    "bin_Hospitalization",
    "min_Hospitalization",
    "days_Hospitalization",
    "max_Hospitalization",
    "bin_ICU",
    "min_ICU",
    "days_ICU",
    "max_ICU",
    "bin_StandardWard",
    "min_StandardWard",
    "days_StandardWard",
    "max_StandardWard",
    "bin_Oxygen",
    "min_Oxygen",
    "days_Oxygen",
    "max_Oxygen",
    "bin_HFNO",
    "min_HFNO",
    "days_HFNO",
    "max_HFNO",
    "bin_MechanicalVentilation_ECMO",
    "min_MechanicalVentilation_ECMO",
    "days_MechanicalVentilation_ECMO",
    "max_MechanicalVentilation_ECMO",
    "Mutation",
    "DateOfDeath",
    "Long_COVID",
    "DCCI",
]


def read_csv_flex(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    try:
        df = pd.read_csv(path, dtype=str, low_memory=False, encoding="utf-8")
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    for enc in ("utf-8-sig", None, "latin1"):
        attempts = (
            {"sep": ","},
            {"sep": ";", "engine": "python"},
            {"sep": "\t", "engine": "python"},
            {"sep": None, "engine": "python"},
        )
        for opts in attempts:
            try:
                common_kwargs: dict[str, Any] = {"dtype": "str", "encoding": enc}
                if opts.get("engine") != "python":
                    common_kwargs["low_memory"] = False
                df = pd.read_csv(path, **opts, **common_kwargs)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    return pd.read_csv(path, dtype=str, engine="python", sep=None)


def _load_mfg_codes(repo_code_dir: Path):
    path = repo_code_dir / "mfg_codes.py"
    spec = importlib.util.spec_from_file_location("mfg_codes_cfr", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"mfg_codes not found at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_mfg, mod.PFIZER, mod.MODERNA


def load_czech_records(
    csv_path: str | Path,
    *,
    restrict_to_pfizer_moderna: bool = True,
    single_infection_only: bool = False,
    repo_code_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Read and standardize Czech NR CSV.

    If restrict_to_pfizer_moderna is True, drop any row that has a non-empty
    manufacturer code on doses 1–4 that is not Pfizer or Moderna (matches KCOR_ts).
    Rows with only empty dose codes (unvaccinated) are kept.
    repo_code_dir must be the repo ``code/`` directory when that filter is on.

    If single_infection_only is True, rows with ``Infection`` > 1 are dropped (legacy first-infection-only).
    Default is False so all infection episodes (1, 2, 3, …) are kept unless the filter is enabled in YAML.
    """
    csv_path = Path(csv_path)
    df = read_csv_flex(csv_path)
    if df.shape[1] != EXPECTED_COL_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COL_COUNT} columns, got {df.shape[1]} in {csv_path}"
        )
    df = df.copy()
    df.columns = CZECH_TO_ENGLISH_COLUMNS

    if single_infection_only:
        df = df[df["Infection"].fillna("0").astype(str).str.strip().replace("", "0").astype(int) <= 1].copy()

    if restrict_to_pfizer_moderna:
        if repo_code_dir is None:
            raise ValueError("repo_code_dir required when restrict_to_pfizer_moderna=True")
        parse_mfg, PFIZER, MODERNA = _load_mfg_codes(repo_code_dir)

        vaccine_code_cols = [
            "VaccineCode_FirstDose",
            "VaccineCode_SecondDose",
            "VaccineCode_ThirdDose",
            "VaccineCode_FourthDose",
        ]
        has_non_mrna = pd.Series(False, index=df.index)
        for col in vaccine_code_cols:
            if col not in df.columns:
                continue
            mfg_values = df[col].apply(
                lambda x: parse_mfg(x) if pd.notna(x) and str(x).strip() != "" else None
            )
            non_mrna_mask = (mfg_values.notna()) & (mfg_values != PFIZER) & (mfg_values != MODERNA)
            has_non_mrna = has_non_mrna | non_mrna_mask
        df = df[~has_non_mrna].copy()

    return df


def validate_loaded_schema(df: pd.DataFrame, *, stage: str) -> None:
    """Fail fast if canonical columns are missing (bad CSV mapping or wrong file)."""
    if stage == "load":
        need = REQUIRED_COLUMNS_AFTER_LOAD
    elif stage == "enrollment":
        need = REQUIRED_COLUMNS_AFTER_ENROLLMENT
    else:
        raise ValueError(f"unknown stage {stage}")
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(
            f"CFR pipeline schema check failed after {stage}: missing columns {missing}. "
            "Expected 53-column Czech NR export with standard English rename."
        )
