"""
Weekly incident booster identifiability emulation (Czech 2021).

Implements the design in documentation/preprint/identifiability.md.

Key semantics:
- ISO-week strings are interpreted as the Monday of that ISO week.
- Cohort membership and risk sets are defined at week-start.
- Transition censoring is rule B: censor starting the week AFTER the transition week.
- Hazards use discrete-time transform: h(t) = -ln(1 - dead/alive).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Force a headless Matplotlib backend (WSL/servers often have no display).
# This must run BEFORE importing matplotlib.pyplot anywhere.
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except Exception:
    # If matplotlib isn't installed, plotting will fail later with an import error.
    pass


def _find_repo_root(start: Path) -> Optional[Path]:
    """
    Find the repo root by walking upwards until we find code/mfg_codes.py.

    This keeps imports working regardless of where this script lives (or current cwd).
    """
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "code" / "mfg_codes.py").exists():
            return parent
    return None


# Ensure we can import repo-local modules (notably code/mfg_codes.py) regardless of cwd.
_REPO_ROOT = _find_repo_root(Path(__file__))
if _REPO_ROOT is not None:
    sys.path.insert(0, str(_REPO_ROOT / "code"))


DOSE_DATE_COLS = [
    "Date_FirstDose",
    "Date_SecondDose",
    "Date_ThirdDose",
    "Date_FourthDose",
    "Date_FifthDose",
    "Date_SixthDose",
]

VCODE_COLS = [
    "VaccineCode_FirstDose",
    "VaccineCode_SecondDose",
    "VaccineCode_ThirdDose",
    "VaccineCode_FourthDose",
]


def _read_csv_flex(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Read CSV with robust delimiter/encoding handling. Returns dtype=str."""
    # Fast path
    try:
        return pd.read_csv(path, dtype=str, low_memory=False, encoding="utf-8", nrows=max_rows)
    except Exception:
        pass
    for enc in ("utf-8-sig", None, "latin1"):
        attempts = (
            {"sep": ","},
            {"sep": ";", "engine": "python"},
            {"sep": "\t", "engine": "python"},
            {"sep": None, "engine": "python"},  # sniff
        )
        for opts in attempts:
            try:
                common_kwargs = {"dtype": str, "encoding": enc, "nrows": max_rows}
                if opts.get("engine") != "python":
                    common_kwargs["low_memory"] = False
                df = pd.read_csv(path, **opts, **common_kwargs)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    return pd.read_csv(path, dtype=str, engine="python", sep=None, nrows=max_rows)


def iso_week_str_to_monday_ts(iso_week_str: pd.Series) -> pd.Series:
    """Convert series of ISO week strings YYYY-WW to pandas Timestamp for Monday of that week."""
    # Expect strings like "2021-24"; tolerate junk by errors='coerce'
    return pd.to_datetime(iso_week_str.astype(str) + "-1", format="%G-%V-%u", errors="coerce")


def parse_dates_inplace(df: pd.DataFrame) -> None:
    """Parse relevant ISO-week columns in-place into pandas Timestamps (Monday of week)."""
    for col in DOSE_DATE_COLS:
        if col in df.columns:
            df[col] = iso_week_str_to_monday_ts(df[col])
        else:
            df[col] = pd.NaT

    # DateOfDeath is ISO-week; strip any non [0-9-] first (mirrors KCOR_CMR.py)
    if "DateOfDeath" in df.columns:
        s = df["DateOfDeath"].astype(str).str.replace(r"[^0-9-]", "", regex=True)
        df["DateOfDeath"] = pd.to_datetime(s + "-1", format="%G-%V-%u", errors="coerce")
    else:
        df["DateOfDeath"] = pd.NaT


def parse_birth_year(df: pd.DataFrame) -> None:
    """Parse YearOfBirth field into integer birth_year (or -1 if missing)."""
    if "YearOfBirth" not in df.columns:
        df["birth_year"] = -1
        return
    by = df["YearOfBirth"].astype(str).str.extract(r"(\d{4})")[0]
    by_num = pd.to_numeric(by, errors="coerce")
    df["birth_year"] = by_num.fillna(-1).astype(int)


def filter_infection_single(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Infection <= 1; treat missing as 0."""
    if "Infection" not in df.columns:
        return df
    inf = pd.to_numeric(df["Infection"].fillna("0"), errors="coerce").fillna(0).astype(int)
    return df[inf <= 1].copy()


def filter_non_mrna(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out any record with non-mRNA vaccines in doses 1-4 (mirrors KCOR_CMR/KCOR_ts behavior)."""
    try:
        from mfg_codes import parse_mfg, PFIZER, MODERNA
    except Exception as e:
        raise RuntimeError(f"Could not import mfg_codes.parse_mfg: {e}") from e

    if not any(c in df.columns for c in VCODE_COLS):
        return df

    has_non_mrna = pd.Series(False, index=df.index)
    for col in VCODE_COLS:
        if col not in df.columns:
            continue
        codes = df[col]
        mfg_values = codes.apply(lambda x: parse_mfg(x) if pd.notna(x) and x != "" else None)
        non_mrna = (mfg_values.notna()) & (mfg_values != PFIZER) & (mfg_values != MODERNA)
        has_non_mrna = has_non_mrna | non_mrna
    return df[~has_non_mrna].copy()


def safe_div(num: float, den: float) -> float:
    if den == 0 or not math.isfinite(den):
        return float("nan")
    return num / den


def discrete_time_hazard(dead: int, alive: int) -> float:
    """h = -ln(1 - dead/alive); returns NaN if alive==0 or dead>alive or invalid."""
    if alive <= 0:
        return float("nan")
    if dead < 0 or dead > alive:
        return float("nan")
    mr = dead / alive
    # Clamp to avoid -ln(0) for mr=1 due to data issues.
    mr = min(max(mr, 0.0), 1.0 - 1e-12)
    return -math.log(1.0 - mr)


@dataclass(frozen=True)
class CohortSpec:
    name: str
    baseline_dose: int
    transition_col: str  # next dose date column name


COHORTS = [
    CohortSpec(name="dose0", baseline_dose=0, transition_col="Date_FirstDose"),
    CohortSpec(name="dose2", baseline_dose=2, transition_col="Date_ThirdDose"),
    CohortSpec(name="dose3", baseline_dose=3, transition_col="Date_FourthDose"),
]


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def enrollment_label(E: pd.Timestamp) -> str:
    return E.strftime("%Y%m%d")


def build_baseline_masks(df: pd.DataFrame, S: pd.Timestamp, E: pd.Timestamp) -> dict[str, pd.Series]:
    """Return boolean masks for baseline cohorts using as-of S and alive at E."""
    # Alive at E (week-start): death missing or death >= E
    alive_at_E = df["DateOfDeath"].isna() | (df["DateOfDeath"] >= E)

    d1 = df["Date_FirstDose"]
    d2 = df["Date_SecondDose"]
    d3 = df["Date_ThirdDose"]

    dose3_incident = (d3.notna()) & (d3 >= S) & (d3 < E)  # default; overridden by dose3_window_start in caller
    dose2_prevalent = (d2.notna()) & (d2 <= S) & (d3.isna() | (d3 > S))
    dose0_prevalent = d1.isna() | (d1 > S)

    # Apply alive-at-E gate to all cohorts
    return {
        "dose3": alive_at_E & dose3_incident,
        "dose2": alive_at_E & dose2_prevalent,
        "dose0": alive_at_E & dose0_prevalent,
    }


def build_baseline_masks_with_dose3_window(
    df: pd.DataFrame,
    *,
    S: pd.Timestamp,
    E: pd.Timestamp,
    dose3_window_start: pd.Timestamp,
) -> dict[str, pd.Series]:
    """Like build_baseline_masks, but dose3_incident uses [dose3_window_start, E)."""
    masks = build_baseline_masks(df, S=S, E=E)
    d3 = df["Date_ThirdDose"]
    d4 = df["Date_FourthDose"] if "Date_FourthDose" in df.columns else pd.Series(pd.NaT, index=df.index)
    alive_at_E = df["DateOfDeath"].isna() | (df["DateOfDeath"] >= E)
    # Dose 3 incident cohort: got dose 3 in the last N weeks before enrollment,
    # and did NOT already have dose 4 before enrollment.
    dose3_incident = (d3.notna()) & (d3 >= dose3_window_start) & (d3 < E) & (d4.isna() | (d4 >= E))
    masks["dose3"] = alive_at_E & dose3_incident
    return masks


def compute_counts_for_cohort(
    df: pd.DataFrame,
    mask: pd.Series,
    cohort: CohortSpec,
    E: pd.Timestamp,
    followup_weeks: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Return (alive_series, dead_series, edge_sameweek_series) for t=0..followup_weeks-1.

    Edge case counted: DateOfDeath == transition_date (same ISO-week) AND within follow-up.
    """
    sub = df.loc[mask, ["DateOfDeath", cohort.transition_col]].copy()

    death = sub["DateOfDeath"]
    trans = sub[cohort.transition_col]

    # Rule B: censor starting the week AFTER death/transition week.
    death_censor_start = death + pd.Timedelta(days=7)
    trans_censor_start = trans + pd.Timedelta(days=7)

    # NaT -> +inf for min()
    death_censor_start = death_censor_start.fillna(pd.Timestamp.max)
    trans_censor_start = trans_censor_start.fillna(pd.Timestamp.max)
    censor_start = pd.concat([death_censor_start, trans_censor_start], axis=1).min(axis=1)

    alive_series: list[int] = []
    dead_series: list[int] = []
    edge_series: list[int] = []

    for t in range(followup_weeks):
        week_start = E + pd.Timedelta(days=7 * t)
        at_risk = censor_start > week_start
        alive = int(at_risk.sum())
        dead_this_week = int(((death == week_start) & at_risk).sum())

        # Edge-case count: death and transition same week (within follow-up)
        edge = int(((death == week_start) & (trans == week_start) & at_risk).sum())

        alive_series.append(alive)
        dead_series.append(dead_this_week)
        edge_series.append(edge)

    return alive_series, dead_series, edge_series


def plot_per_enrollment(
    outdir: Path,
    enroll_label: str,
    t: np.ndarray,
    h0: np.ndarray,
    h2: np.ndarray,
    h3: np.ndarray,
    hr20: np.ndarray,
    hr30: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    # Hazards
    plt.figure(figsize=(9, 5))
    plt.plot(t, h0, label="h0 (dose0)")
    plt.plot(t, h2, label="h2 (dose2)")
    plt.plot(t, h3, label="h3 (dose3 incident)")
    plt.xlabel("t (weeks since enrollment)")
    plt.ylabel("h(t) = -ln(1 - dead/alive)")
    plt.title(f"Hazards by cohort (E={enroll_label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"h_curves_E{enroll_label}.png", dpi=160)
    plt.close()

    # HR
    plt.figure(figsize=(9, 5))
    plt.plot(t, hr20, label="HR20 = h2/h0")
    plt.plot(t, hr30, label="HR30 = h3/h0")
    plt.xlabel("t (weeks since enrollment)")
    plt.ylabel("Hazard ratio")
    plt.title(f"HR curves (E={enroll_label})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"HR_curves_E{enroll_label}.png", dpi=160)
    plt.close()


def plot_spaghetti(outdir: Path, series_df: pd.DataFrame, which: str) -> None:
    """which in {'HR20','HR30'}."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    for enroll_date, g in series_df.groupby("enrollment_date"):
        g2 = g.sort_values("t")
        plt.plot(g2["t"].values, g2[which].values, alpha=0.7, linewidth=1.5, label=str(enroll_date))
    plt.xlabel("t (weeks since enrollment)")
    plt.ylabel(which)
    plt.title(f"{which} spaghetti across enrollments")
    # Too many legend entries; omit by default.
    plt.tight_layout()
    plt.savefig(outdir / f"{which}_spaghetti.png", dpi=160)
    plt.close()


def summarize_enrollment(g: pd.DataFrame) -> dict[str, object]:
    """Compute summary metrics for one enrollment group g (rows for t=0..)."""
    g2 = g.sort_values("t")
    hr30 = g2["HR30"].to_numpy(dtype=float)
    hr20 = g2["HR20"].to_numpy(dtype=float)

    def peak_info(arr: np.ndarray) -> tuple[Optional[int], float]:
        if arr.size == 0:
            return None, float("nan")
        if np.all(~np.isfinite(arr)):
            return None, float("nan")
        idx = int(np.nanargmax(arr))
        return idx, float(arr[idx])

    peak_t_hr30, peak_val_hr30 = peak_info(hr30)
    peak_t_hr20, peak_val_hr20 = peak_info(hr20)

    def at_t(arr: np.ndarray, t: int) -> float:
        if t < 0 or t >= arr.size:
            return float("nan")
        return float(arr[t])

    return {
        "enrollment_date": g2["enrollment_date"].iloc[0],
        "peak_week_HR30": peak_t_hr30,
        "peak_value_HR30": peak_val_hr30,
        "HR30_at_t0": at_t(hr30, 0),
        "HR30_at_t2": at_t(hr30, 2),
        "peak_week_HR20": peak_t_hr20,
        "peak_value_HR20": peak_val_hr20,
        "HR20_at_t0": at_t(hr20, 0),
        "HR20_at_t2": at_t(hr20, 2),
        "edge_death_transition_sameweek_dose0": int(g2["edge_sameweek0"].sum()),
        "edge_death_transition_sameweek_dose2": int(g2["edge_sameweek2"].sum()),
        "edge_death_transition_sameweek_dose3": int(g2["edge_sameweek3"].sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/Czech/records.csv")
    ap.add_argument("--outdir", default="identifiability/Czech/booster")
    ap.add_argument("--enrollment-start", default="2021-10-18", help="YYYY-MM-DD (Monday)")
    ap.add_argument("--n-enrollments", type=int, default=10)
    ap.add_argument("--followup-weeks", type=int, default=26)
    # Default: no birth-year restriction (all ages). Provide both to enable filtering.
    ap.add_argument("--birth-year-min", type=int, default=None)
    ap.add_argument("--birth-year-max", type=int, default=None)
    ap.add_argument("--lookback-days", type=int, default=7)
    ap.add_argument(
        "--dose3-incident-lookback-weeks",
        type=int,
        default=4,
        help="Dose 3 incident cohort window length (weeks before enrollment).",
    )
    ap.add_argument("--filter-non-mrna", dest="filter_non_mrna", action="store_true", default=True)
    ap.add_argument("--no-filter-non-mrna", dest="filter_non_mrna", action="store_false")
    ap.add_argument("--max-rows", type=int, default=None, help="Debug: read only first N rows")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    input_path = args.input
    print(f"Reading {input_path} ...", flush=True)
    df_raw = _read_csv_flex(input_path, max_rows=args.max_rows)

    expected_col_count = 53
    if df_raw.shape[1] != expected_col_count:
        raise SystemExit(
            f"ERROR: Input parsed into {df_raw.shape[1]} columns, expected {expected_col_count}. "
            "Delimiter/encoding mismatch likely."
        )

    # Rename to standard schema (same as KCOR_CMR.py / KCOR_ts.py)
    df_raw.columns = [
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

    # Keep only columns needed for this analysis to reduce memory pressure.
    keep_cols = ["Infection", "YearOfBirth", "DateOfDeath"] + DOSE_DATE_COLS + VCODE_COLS
    keep_cols = [c for c in keep_cols if c in df_raw.columns]
    df = df_raw.loc[:, keep_cols].copy()
    del df_raw

    parse_birth_year(df)
    if (args.birth_year_min is None) ^ (args.birth_year_max is None):
        raise SystemExit("ERROR: Provide both --birth-year-min and --birth-year-max, or neither (for all ages).")
    if args.birth_year_min is not None and args.birth_year_max is not None:
        df = df[(df["birth_year"] >= args.birth_year_min) & (df["birth_year"] <= args.birth_year_max)].copy()
        print(f"After birth-year filter [{args.birth_year_min},{args.birth_year_max}]: {len(df):,} rows", flush=True)
    else:
        print(f"Birth-year filter: disabled (all ages; includes unknown birth years)", flush=True)

    df = filter_infection_single(df)
    print(f"After Infection<=1 filter: {len(df):,} rows", flush=True)

    if args.filter_non_mrna:
        before = len(df)
        df = filter_non_mrna(df)
        print(f"After non-mRNA filter: kept {len(df):,}/{before:,}", flush=True)

    parse_dates_inplace(df)

    enrollment_start = pd.to_datetime(args.enrollment_start)
    if enrollment_start.weekday() != 0:
        print("WARNING: enrollment_start is not a Monday; week-start semantics assume Monday.", flush=True)

    followup_weeks = int(args.followup_weeks)
    rows: list[dict[str, object]] = []

    for i in range(int(args.n_enrollments)):
        E = enrollment_start + pd.Timedelta(days=7 * i)
        S = E - pd.Timedelta(days=int(args.lookback_days))

        dose3_window_start = E - pd.Timedelta(days=7 * int(args.dose3_incident_lookback_weeks))
        masks = build_baseline_masks_with_dose3_window(df, S=S, E=E, dose3_window_start=dose3_window_start)
        enroll_lbl = enrollment_label(E)
        print(
            f"\nEnrollment {i+1}/{args.n_enrollments}: E={E.date().isoformat()} (label={enroll_lbl}), S={S.date().isoformat()}",
            flush=True,
        )
        print(
            f"  Baseline sizes: dose0={int(masks['dose0'].sum()):,}, dose2={int(masks['dose2'].sum()):,}, dose3_incident={int(masks['dose3'].sum()):,}",
            flush=True,
        )

        alive0, dead0, edge0 = compute_counts_for_cohort(df, masks["dose0"], COHORTS[0], E=E, followup_weeks=followup_weeks)
        alive2, dead2, edge2 = compute_counts_for_cohort(df, masks["dose2"], COHORTS[1], E=E, followup_weeks=followup_weeks)
        alive3, dead3, edge3 = compute_counts_for_cohort(df, masks["dose3"], COHORTS[2], E=E, followup_weeks=followup_weeks)

        for t in range(followup_weeks):
            week_start = (E + pd.Timedelta(days=7 * t)).date().isoformat()
            h0 = discrete_time_hazard(dead0[t], alive0[t])
            h2 = discrete_time_hazard(dead2[t], alive2[t])
            h3 = discrete_time_hazard(dead3[t], alive3[t])
            rows.append(
                {
                    "enrollment_date": E.date().isoformat(),
                    "t": t,
                    "calendar_week": week_start,
                    "dead0": dead0[t],
                    "alive0": alive0[t],
                    "h0": h0,
                    "dead2": dead2[t],
                    "alive2": alive2[t],
                    "h2": h2,
                    "dead3": dead3[t],
                    "alive3": alive3[t],
                    "h3": h3,
                    "HR20": safe_div(h2, h0),
                    "HR30": safe_div(h3, h0),
                    "edge_sameweek0": edge0[t],
                    "edge_sameweek2": edge2[t],
                    "edge_sameweek3": edge3[t],
                }
            )

        # Plots per enrollment
        g = pd.DataFrame([r for r in rows if r["enrollment_date"] == E.date().isoformat()])
        plot_per_enrollment(
            outdir,
            enroll_lbl,
            t=g["t"].to_numpy(dtype=int),
            h0=g["h0"].to_numpy(dtype=float),
            h2=g["h2"].to_numpy(dtype=float),
            h3=g["h3"].to_numpy(dtype=float),
            hr20=g["HR20"].to_numpy(dtype=float),
            hr30=g["HR30"].to_numpy(dtype=float),
        )

        if (g["dead0"] > g["alive0"]).any() or (g["dead2"] > g["alive2"]).any() or (g["dead3"] > g["alive3"]).any():
            print("WARNING: Found dead > alive in some week; check parsing/semantics.", flush=True)

    series_df = pd.DataFrame(rows)
    series_path = outdir / "series.csv"
    series_df.to_csv(series_path, index=False)
    print(f"\nWrote {series_path} ({len(series_df):,} rows)", flush=True)

    summaries = [summarize_enrollment(g) for _, g in series_df.groupby("enrollment_date")]
    summary_df = pd.DataFrame(summaries).sort_values("enrollment_date")
    summary_path = outdir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(summary_df):,} rows)", flush=True)

    plot_spaghetti(outdir, series_df, "HR30")
    plot_spaghetti(outdir, series_df, "HR20")

    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

