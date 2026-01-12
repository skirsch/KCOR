"""
Post-run analysis for identifiability outputs.

Reads <outdir>/series.csv and prints:
- Peak timing in t and in calendar time for HR20, HR30 (mixture), and HR30_w1..wK bins
- Simple locking heuristics:
  - t_lock_score: max frequency of peak_t across enrollments / n_enrollments
  - cal_lock_score: max frequency of peak calendar_week across enrollments / n_enrollments
  - corr(enrollment_date, peak_t): near 0 suggests t-locked; strongly negative suggests calendar-locked
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PeakResult:
    enrollment_date: str
    curve: str
    t_peak: int
    calendar_peak: str
    peak_value: float


def _corr_enroll_vs_t(peaks: list[PeakResult]) -> float:
    # Convert enrollment_date (YYYY-MM-DD) to ordinal
    enroll_ord = np.array([date.fromisoformat(p.enrollment_date).toordinal() for p in peaks], dtype=float)
    t = np.array([p.t_peak for p in peaks], dtype=float)
    if len(peaks) < 3:
        return float("nan")
    if np.all(t == t[0]):
        return 0.0
    return float(np.corrcoef(enroll_ord, t)[0, 1])


def summarize_curve(peaks: list[PeakResult]) -> dict[str, object]:
    t_vals = [p.t_peak for p in peaks]
    cal_vals = [p.calendar_peak for p in peaks]
    n = len(peaks)
    t_counts = Counter(t_vals)
    cal_counts = Counter(cal_vals)
    t_lock_score = (t_counts.most_common(1)[0][1] / n) if n else float("nan")
    cal_lock_score = (cal_counts.most_common(1)[0][1] / n) if n else float("nan")
    return {
        "n_enrollments": n,
        "t_range": f"{min(t_vals)}..{max(t_vals)}" if n else "",
        "t_std": float(np.std(np.array(t_vals, dtype=float), ddof=0)) if n else float("nan"),
        "t_lock_score": t_lock_score,
        "calendar_unique": len(cal_counts),
        "cal_lock_score": cal_lock_score,
        "cal_mode": cal_counts.most_common(1)[0][0] if n else "",
        "corr_enroll_vs_t": _corr_enroll_vs_t(peaks),
        "calendar_modes": cal_counts.most_common(5),
    }


def alignment_scores(series: pd.DataFrame, curve: str) -> dict[str, float]:
    """
    Compare how well a curve aligns across enrollments in:
    - t-space (weeks since enrollment)
    - calendar-space (absolute calendar_week)

    Returns SSE-style scores (lower is better):
      - sse_t: sum of squared deviations from mean curve by t
      - sse_cal: sum of squared deviations from mean curve by calendar week
      - sse_ratio_cal_over_t: sse_cal / sse_t

    Intuition:
    - Calendar-locked curve => sse_cal << sse_t => ratio < 1
    - t-locked curve => sse_t << sse_cal => ratio > 1
    """
    if curve not in series.columns:
        return {"sse_t": float("nan"), "sse_cal": float("nan"), "sse_ratio_cal_over_t": float("nan")}

    df = series.loc[:, ["enrollment_date", "t", "calendar_week", curve]].copy()
    df[curve] = pd.to_numeric(df[curve], errors="coerce")
    df = df[df[curve].notna()].copy()
    if df.empty:
        return {"sse_t": float("nan"), "sse_cal": float("nan"), "sse_ratio_cal_over_t": float("nan")}

    mean_by_t = df.groupby("t", as_index=False)[curve].mean().rename(columns={curve: "mean"})
    df_t = df.merge(mean_by_t, on="t", how="left")
    sse_t = float(((df_t[curve] - df_t["mean"]) ** 2).sum())

    mean_by_cal = df.groupby("calendar_week", as_index=False)[curve].mean().rename(columns={curve: "mean"})
    df_c = df.merge(mean_by_cal, on="calendar_week", how="left")
    sse_cal = float(((df_c[curve] - df_c["mean"]) ** 2).sum())

    ratio = float("nan") if sse_t == 0 else (sse_cal / sse_t)
    return {"sse_t": sse_t, "sse_cal": sse_cal, "sse_ratio_cal_over_t": ratio}


def auc_summary(series: pd.DataFrame, curve: str, auc_weeks: int) -> dict[str, float]:
    """
    Compute AUC over weeks t=0..auc_weeks-1 for each enrollment,
    then summarize its dispersion across enrollments.
    """
    if curve not in series.columns:
        return {"auc_mean": float("nan"), "auc_std": float("nan"), "auc_cv": float("nan")}
    df = series.loc[:, ["enrollment_date", "t", curve]].copy()
    df[curve] = pd.to_numeric(df[curve], errors="coerce")
    df = df[(df["t"] >= 0) & (df["t"] < int(auc_weeks)) & df[curve].notna()].copy()
    if df.empty:
        return {"auc_mean": float("nan"), "auc_std": float("nan"), "auc_cv": float("nan")}
    auc = df.groupby("enrollment_date")[curve].sum()
    m = float(auc.mean())
    sd = float(auc.std(ddof=0))
    cv = float("nan") if m == 0 else (sd / m)
    return {"auc_mean": m, "auc_std": sd, "auc_cv": cv}


def _sorted_lead_cols(cols: list[str]) -> list[str]:
    lead = [c for c in cols if c.startswith("HR_lead")]

    def _k(c: str) -> int:
        try:
            return int(c.replace("HR_lead", ""))
        except Exception:
            return 10**9

    return sorted(lead, key=_k)


def _perm_outside_fractions(
    series: pd.DataFrame,
    *,
    curve: str,
    perm: Optional[pd.DataFrame],
) -> dict[str, float]:
    """
    Compare observed curve to permutation band (5–95%) if available.

    Returns:
      - perm_outside_frac_all: fraction of (enrollment,t) points outside band
      - perm_outside_frac_pretreat: for HR_leadK, restrict to t < K; else NaN
    """
    if perm is None or curve not in series.columns:
        return {"perm_outside_frac_all": float("nan"), "perm_outside_frac_pretreat": float("nan")}

    need_cols = {"enrollment_date", "t", "perm_q05", "perm_q95"}
    if not need_cols.issubset(set(perm.columns)):
        return {"perm_outside_frac_all": float("nan"), "perm_outside_frac_pretreat": float("nan")}

    df = series.loc[:, ["enrollment_date", "t", curve]].copy()
    df[curve] = pd.to_numeric(df[curve], errors="coerce")
    p = perm.loc[:, ["enrollment_date", "t", "perm_q05", "perm_q95"]].copy()
    p["perm_q05"] = pd.to_numeric(p["perm_q05"], errors="coerce")
    p["perm_q95"] = pd.to_numeric(p["perm_q95"], errors="coerce")

    m = df.merge(p, on=["enrollment_date", "t"], how="inner")
    ok = m[curve].notna() & m["perm_q05"].notna() & m["perm_q95"].notna()
    if not ok.any():
        return {"perm_outside_frac_all": float("nan"), "perm_outside_frac_pretreat": float("nan")}

    m = m.loc[ok].copy()
    outside = (m[curve] < m["perm_q05"]) | (m[curve] > m["perm_q95"])
    frac_all = float(outside.mean()) if len(m) else float("nan")

    frac_pre = float("nan")
    if curve.startswith("HR_lead"):
        try:
            k = int(curve.replace("HR_lead", ""))
            m_pre = m[m["t"] < k]
            if len(m_pre):
                outside_pre = (m_pre[curve] < m_pre["perm_q05"]) | (m_pre[curve] > m_pre["perm_q95"])
                frac_pre = float(outside_pre.mean())
        except Exception:
            pass

    return {"perm_outside_frac_all": frac_all, "perm_outside_frac_pretreat": frac_pre}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True, help="Directory containing series.csv")
    ap.add_argument("--auc-weeks", type=int, default=8, help="Weeks for AUC stability metric (t=0..K-1).")
    ap.add_argument(
        "--label",
        default=None,
        help="Optional label to include in metrics CSV (e.g., variant name).",
    )
    ap.add_argument(
        "--metrics-csv",
        default=None,
        help="Optional path to write a machine-readable metrics CSV (one row per curve).",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    series_path = outdir / "series.csv"
    if not series_path.exists():
        raise SystemExit(f"ERROR: {series_path} not found")

    s = pd.read_csv(series_path)
    if "enrollment_date" not in s.columns or "t" not in s.columns or "calendar_week" not in s.columns:
        raise SystemExit("ERROR: series.csv missing required columns")

    # Identify bins
    bin_cols_30 = sorted([c for c in s.columns if c.startswith("HR30_w")], key=lambda x: int(x.split("_w")[1]))
    bin_cols_32 = sorted([c for c in s.columns if c.startswith("HR32_w")], key=lambda x: int(x.split("_w")[1]))

    # Optional falsification curves: "future booster" placebo HRs
    future_cols = [c for c in ["HR_future30", "HR_future32"] if c in s.columns]

    # Selection/eligibility suite curves (optional)
    selection_cols: list[str] = []
    if "HR_tt32" in s.columns:
        selection_cols.append("HR_tt32")
    lead_cols = _sorted_lead_cols(list(s.columns))

    curves = ["HR20", "HR30", "HR32"] + future_cols + selection_cols + lead_cols + bin_cols_30 + bin_cols_32

    perm: Optional[pd.DataFrame] = None
    perm_path = outdir / "perm_summary.csv"
    if perm_path.exists():
        try:
            perm = pd.read_csv(perm_path)
        except Exception:
            perm = None

    peaks_all: list[PeakResult] = []
    for ed, g in s.groupby("enrollment_date"):
        g = g.sort_values("t").reset_index(drop=True)
        for c in curves:
            if c not in g.columns:
                continue
            vals = pd.to_numeric(g[c], errors="coerce")
            if not vals.notna().any():
                # All-NaN curve for this enrollment (e.g., cohort size 0) -> skip.
                continue
            i = int(vals.idxmax())
            peaks_all.append(
                PeakResult(
                    enrollment_date=str(ed),
                    curve=c,
                    t_peak=int(g.loc[i, "t"]),
                    calendar_peak=str(g.loc[i, "calendar_week"]),
                    peak_value=float(g.loc[i, c]),
                )
            )

    # Print compact peak table
    print(f"Series: {series_path}")
    print(f"Enrollments: {s.enrollment_date.nunique()}  Followup weeks: {int(s.t.max())+1}")
    print("")
    print("=== Peak table (per enrollment) ===")
    if not peaks_all:
        print("(no finite values available for peak detection in any curve)")
    else:
        dfp = pd.DataFrame([p.__dict__ for p in peaks_all]).sort_values(["curve", "enrollment_date"])
        print(dfp.to_string(index=False))

    print("\n=== Locking summary ===")
    metrics_rows: list[dict[str, object]] = []
    for c in curves:
        peaks = [p for p in peaks_all if p.curve == c]
        summ = summarize_curve(peaks)
        align = alignment_scores(s, c)
        auc = auc_summary(s, c, auc_weeks=int(args.auc_weeks))
        perm_cmp = _perm_outside_fractions(s, curve=c, perm=perm)
        print(
            f"{c}: "
            f"t_range={summ['t_range']} t_std={summ['t_std']:.2f} "
            f"t_lock_score={summ['t_lock_score']:.2f} "
            f"cal_unique={summ['calendar_unique']} cal_lock_score={summ['cal_lock_score']:.2f} "
            f"corr(enroll,t)={summ['corr_enroll_vs_t']:.2f} "
            f"sse_ratio(cal/t)={align['sse_ratio_cal_over_t']:.3f} "
            f"auc{int(args.auc_weeks)}_cv={auc['auc_cv']:.3f} "
            f"cal_mode={summ['cal_mode']} "
            f"perm_outside={perm_cmp['perm_outside_frac_all']:.3f}"
        )
        print(f"  calendar_modes={summ['calendar_modes']}")
        print(
            f"  align: sse_t={align['sse_t']:.3g} sse_cal={align['sse_cal']:.3g} "
            f"(ratio<1 => calendar-locked; ratio>1 => t-locked)"
        )
        print(f"  auc: mean={auc['auc_mean']:.3g} std={auc['auc_std']:.3g} cv={auc['auc_cv']:.3g}")
        if c.startswith("HR_lead"):
            print(f"  perm_outside_pretreat(t<k)={perm_cmp['perm_outside_frac_pretreat']:.3f}")

        metrics_rows.append(
            {
                "label": args.label if args.label is not None else "",
                "outdir": str(outdir),
                "curve": c,
                "n_enrollments": summ["n_enrollments"],
                "t_range": summ["t_range"],
                "t_std": summ["t_std"],
                "t_lock_score": summ["t_lock_score"],
                "calendar_unique": summ["calendar_unique"],
                "cal_lock_score": summ["cal_lock_score"],
                "cal_mode": summ["cal_mode"],
                "corr_enroll_vs_t": summ["corr_enroll_vs_t"],
                "sse_t": align["sse_t"],
                "sse_cal": align["sse_cal"],
                "sse_ratio_cal_over_t": align["sse_ratio_cal_over_t"],
                f"auc{int(args.auc_weeks)}_mean": auc["auc_mean"],
                f"auc{int(args.auc_weeks)}_std": auc["auc_std"],
                f"auc{int(args.auc_weeks)}_cv": auc["auc_cv"],
                "perm_outside_frac_all": perm_cmp["perm_outside_frac_all"],
                "perm_outside_frac_pretreat": perm_cmp["perm_outside_frac_pretreat"],
            }
        )

    print("\nInterpretation hint:")
    print("- If a curve is calendar-locked, you typically see high cal_lock_score and corr(enroll,t) strongly negative.")
    print("- If a curve is t-locked (dose-time), you typically see high t_lock_score and corr(enroll,t) near 0.")
    print("- sse_ratio(cal/t) < 1 suggests better alignment by calendar week; > 1 suggests better alignment by t.")
    print("- aucK_cv closer to 0 suggests the early-window burden is stable across enrollments.")

    if args.metrics_csv:
        metrics_path = Path(args.metrics_csv)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
        print(f"\nWrote metrics CSV: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

