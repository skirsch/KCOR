"""Figures for CFR pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no Tk/X11 (SSH, WSL, invalid DISPLAY)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _wave_shade(ax, iso_weeks_sorted: list[str], wave_start: str, wave_end: str, **kwargs):
    wk = list(iso_weeks_sorted)
    try:
        i0 = wk.index(wave_start) if wave_start in wk else None
        i1 = wk.index(wave_end) if wave_end in wk else None
        if i0 is not None and i1 is not None:
            ax.axvspan(i0 - 0.5, i1 + 0.5, **kwargs)
    except ValueError:
        pass


def plot_case_rate(
    weekly: pd.DataFrame,
    out_path: Path,
    *,
    age_bin: str = "all",
    wave_start: str,
    wave_end: str,
    dpi: int = 120,
) -> None:
    sub = weekly[weekly["age_bin"] == age_bin].copy()
    wk = sorted(sub["iso_week"].unique())
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(wk))
    for cohort in sorted(sub["cohort"].unique()):
        s = sub[sub["cohort"] == cohort].set_index("iso_week").reindex(wk)
        ax.plot(x, s["case_rate"].values, marker="o", ms=2, label=cohort)
    ax.set_xticks(x[:: max(1, len(x) // 12)])
    ax.set_xticklabels([wk[i] for i in x[:: max(1, len(x) // 12)]], rotation=45, ha="right")
    ax.set_ylabel("cases / population at risk")
    ax.set_xlabel("ISO week")
    ax.set_title(f"Weekly case rate (calendar time, age_bin={age_bin})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _wave_shade(ax, sorted(sub["iso_week"].unique()), wave_start, wave_end, alpha=0.15, color="orange")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_cfr(
    weekly: pd.DataFrame,
    out_path: Path,
    *,
    age_bin: str = "all",
    cfr_col: str = "cfr_covid",
    wave_start: str,
    wave_end: str,
    dpi: int = 120,
) -> None:
    sub = weekly[weekly["age_bin"] == age_bin].copy()
    wk = sorted(sub["iso_week"].unique())
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(wk))
    for cohort in sorted(sub["cohort"].unique()):
        s = sub[sub["cohort"] == cohort].set_index("iso_week").reindex(wk)
        ax.plot(x, s[cfr_col].values, marker="o", ms=2, label=cohort)
    ax.set_xticks(x[:: max(1, len(x) // 12)])
    ax.set_xticklabels([wk[i] for i in x[:: max(1, len(x) // 12)]], rotation=45, ha="right")
    ax.set_ylabel(cfr_col)
    ax.set_xlabel("ISO week")
    ax.set_title(f"Weekly CFR ({cfr_col}, calendar time, age_bin={age_bin})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _wave_shade(ax, sorted(sub["iso_week"].unique()), wave_start, wave_end, alpha=0.15, color="orange")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_mortality_rate(
    weekly: pd.DataFrame,
    out_path: Path,
    *,
    age_bin: str = "all",
    wave_start: str,
    wave_end: str,
    dpi: int = 120,
) -> None:
    sub = weekly[weekly["age_bin"] == age_bin].copy()
    wk = sorted(sub["iso_week"].unique())
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(wk))
    for cohort in sorted(sub["cohort"].unique()):
        s = sub[sub["cohort"] == cohort].set_index("iso_week").reindex(wk)
        ax.plot(x, s["mortality_rate_all_cause"].values, marker="o", ms=2, label=cohort)
    ax.set_xticks(x[:: max(1, len(x) // 12)])
    ax.set_xticklabels([wk[i] for i in x[:: max(1, len(x) // 12)]], rotation=45, ha="right")
    ax.set_ylabel("all-cause deaths / pop at risk")
    ax.set_xlabel("ISO week")
    ax.set_title(f"Weekly all-cause mortality rate (calendar time, age_bin={age_bin})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _wave_shade(ax, sorted(sub["iso_week"].unique()), wave_start, wave_end, alpha=0.15, color="orange")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_covid_mortality_per_million(
    weekly: pd.DataFrame,
    out_path: Path,
    *,
    age_bin: str = "all",
    wave_start: str,
    wave_end: str,
    cohorts: list[str],
    dpi: int = 120,
) -> None:
    """Weekly COVID deaths per million at risk vs ISO week (calendar time); one line per cohort."""
    rate_col = "mortality_rate_covid"
    if rate_col not in weekly.columns:
        return
    sub = weekly[weekly["age_bin"] == age_bin].copy()
    if sub.empty:
        return
    wk = sorted(sub["iso_week"].unique())
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(wk))
    any_line = False
    for cohort in cohorts:
        if cohort not in set(sub["cohort"].unique()):
            continue
        s = sub[sub["cohort"] == cohort].set_index("iso_week").reindex(wk)
        y = pd.to_numeric(s[rate_col], errors="coerce").to_numpy(dtype=float) * 1.0e6
        ax.plot(x, y, marker="o", ms=2, label=cohort)
        any_line = True
    if not any_line:
        plt.close(fig)
        return
    ax.set_xticks(x[:: max(1, len(x) // 12)])
    ax.set_xticklabels([wk[i] for i in x[:: max(1, len(x) // 12)]], rotation=45, ha="right")
    ax.set_ylabel("COVID deaths per million at risk (weekly)")
    ax.set_xlabel("ISO week")
    ax.set_title(
        f"Weekly COVID mortality per million at risk (calendar time, age_bin={age_bin})\n"
        "per enrollment cohort in analytic file — not national vital statistics"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    _wave_shade(ax, sorted(sub["iso_week"].unique()), wave_start, wave_end, alpha=0.15, color="orange")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_debug_weekly_covid_deaths(
    weekly: pd.DataFrame,
    out_path: Path,
    *,
    age_bin: str,
    wave_start: str,
    wave_end: str,
    cohorts: list[str],
    dpi: int = 120,
) -> None:
    """Raw weekly COVID death counts (no per-million scaling) vs ISO week — sanity check for sparse strata."""
    col = "deaths_covid"
    if col not in weekly.columns:
        return
    sub = weekly[weekly["age_bin"] == age_bin].copy()
    if sub.empty:
        return
    wk = sorted(sub["iso_week"].unique())
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(wk))
    any_line = False
    for cohort in cohorts:
        if cohort not in set(sub["cohort"].unique()):
            continue
        s = sub[sub["cohort"] == cohort].set_index("iso_week").reindex(wk)
        y = pd.to_numeric(s[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", ms=4, linestyle="-", label=cohort)
        any_line = True
    if not any_line:
        plt.close(fig)
        return
    ax.set_xticks(x[:: max(1, len(x) // 12)])
    ax.set_xticklabels([wk[i] for i in x[:: max(1, len(x) // 12)]], rotation=45, ha="right")
    ax.set_ylabel("COVID deaths (weekly count)")
    ax.set_xlabel("ISO week")
    ax.set_title(
        f"DEBUG: weekly COVID deaths by enrollment cohort (age_bin={age_bin})\n"
        "analytic file only — sum dose0+dose1+dose2 ≠ national 70+ vital stats"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    _wave_shade(ax, sorted(sub["iso_week"].unique()), wave_start, wave_end, alpha=0.15, color="orange")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_debug_weekly_population_at_risk(
    weekly: pd.DataFrame,
    out_path: Path,
    *,
    age_bin: str,
    wave_start: str,
    wave_end: str,
    cohorts: list[str],
    dpi: int = 120,
) -> None:
    """Weekly population at risk (denominator for rates; person-weeks in weekly bins) vs ISO week."""
    col = "population_at_risk"
    if col not in weekly.columns:
        return
    sub = weekly[weekly["age_bin"] == age_bin].copy()
    if sub.empty:
        return
    wk = sorted(sub["iso_week"].unique())
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(wk))
    any_line = False
    for cohort in cohorts:
        if cohort not in set(sub["cohort"].unique()):
            continue
        s = sub[sub["cohort"] == cohort].set_index("iso_week").reindex(wk)
        y = pd.to_numeric(s[col], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, y, marker="o", ms=2, label=cohort)
        any_line = True
    if not any_line:
        plt.close(fig)
        return
    ax.set_xticks(x[:: max(1, len(x) // 12)])
    ax.set_xticklabels([wk[i] for i in x[:: max(1, len(x) // 12)]], rotation=45, ha="right")
    ax.set_ylabel("Population at risk (weekly, ≈ person-weeks)")
    ax.set_xlabel("ISO week")
    ax.set_title(
        f"DEBUG: weekly population at risk by cohort (age_bin={age_bin})\n"
        "denominator for rates; same enrolled analytic cohort as above"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    _wave_shade(ax, sorted(sub["iso_week"].unique()), wave_start, wave_end, alpha=0.15, color="orange")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_debug_weekly_covid_deaths_enrolled_total(
    weekly: pd.DataFrame,
    out_path: Path,
    *,
    age_bin: str,
    wave_start: str,
    wave_end: str,
    cohorts: list[str],
    dpi: int = 120,
) -> None:
    """Single series: sum of ``deaths_covid`` over enrollment cohorts (same week × age_bin)."""
    col = "deaths_covid"
    if col not in weekly.columns:
        return
    sub = weekly[(weekly["age_bin"] == age_bin) & (weekly["cohort"].isin(cohorts))].copy()
    if sub.empty:
        return
    tot = sub.groupby("iso_week", sort=False)[col].sum()
    wk = sorted(tot.index.tolist())
    if not wk:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(wk))
    y = tot.reindex(wk).fillna(0).to_numpy(dtype=float)
    ax.plot(x, y, marker="o", ms=4, linestyle="-", color="black", label="sum(dose cohorts)")
    ax.set_xticks(x[:: max(1, len(x) // 12)])
    ax.set_xticklabels([wk[i] for i in x[:: max(1, len(x) // 12)]], rotation=45, ha="right")
    ax.set_ylabel("COVID deaths (weekly count, summed)")
    ax.set_xlabel("ISO week")
    ax.set_title(
        f"DEBUG: enrolled cohort total weekly COVID deaths (age_bin={age_bin})\n"
        "dose0 + dose1 + dose2 in analytic file — compare to ministry totals only with care"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    _wave_shade(ax, wk, wave_start, wave_end, alpha=0.15, color="orange")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_decomposition(
    weekly: pd.DataFrame,
    out_path: Path,
    *,
    cohort: str,
    age_bin: str = "all",
    wave_start: str,
    wave_end: str,
    dpi: int = 120,
) -> None:
    plot_decomposition_compare(
        weekly,
        out_path,
        cohorts=[cohort],
        age_bin=age_bin,
        wave_start=wave_start,
        wave_end=wave_end,
        dpi=dpi,
    )


def plot_decomposition_compare(
    weekly: pd.DataFrame,
    out_path: Path,
    *,
    cohorts: list[str],
    age_bin: str = "all",
    wave_start: str,
    wave_end: str,
    dpi: int = 120,
) -> None:
    """Observed mortality vs case_rate×cfr_allcause (episode) for each cohort."""
    wk = sorted(weekly.loc[weekly["age_bin"] == age_bin, "iso_week"].unique())
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(wk))
    col = "decomp_mort_implied"
    for cohort in cohorts:
        sub = weekly[(weekly["age_bin"] == age_bin) & (weekly["cohort"] == cohort)]
        s = sub.set_index("iso_week").reindex(wk)
        ax.plot(
            x,
            s["mortality_rate_all_cause"].values,
            lw=2,
            label=f"{cohort} observed mortality",
        )
        ax.plot(
            x,
            s[col].values,
            "--",
            lw=1.5,
            label=f"{cohort} case_rate×CFR_allcause",
        )
    ax.set_xticks(x[:: max(1, len(x) // 12)])
    ax.set_xticklabels([wk[i] for i in x[:: max(1, len(x) // 12)]], rotation=45, ha="right")
    ax.set_title(f"Decomposition (cohorts={','.join(cohorts)}, age_bin={age_bin})")
    leg_ncol = 2 if len(cohorts) >= 3 else 1
    ax.legend(fontsize=8, loc="upper left", ncol=leg_ncol)
    ax.grid(True, alpha=0.3)
    _wave_shade(ax, wk, wave_start, wave_end, alpha=0.15, color="orange")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_expected_vs_observed(
    evo: pd.DataFrame,
    out_path: Path,
    *,
    ve: float,
    wave_start: str,
    wave_end: str,
    dpi: int = 120,
) -> None:
    if evo.empty:
        return
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    wk = evo["iso_week"].tolist()
    x = np.arange(len(wk))
    ax0.bar(x - 0.2, evo["observed_deaths_covid"], width=0.4, label="observed COVID deaths")
    ax0.bar(x + 0.2, evo["expected_deaths_covid_ve"], width=0.4, label=f"expected (VE={ve:.0%})")
    ax0.set_ylabel("deaths per week")
    ax0.set_title("Observed vs expected COVID deaths (vaccinated cohort)")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    _wave_shade(ax0, sorted(evo["iso_week"].unique()), wave_start, wave_end, alpha=0.12, color="orange")

    ax1.plot(x, evo["cumulative_observed_deaths"], lw=2, label="cumulative observed")
    ax1.plot(x, evo["cumulative_expected_deaths"], "--", lw=2, label="cumulative expected")
    ax1.set_ylabel("cumulative deaths")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x[:: max(1, len(x) // 12)])
    ax1.set_xticklabels([wk[i] for i in x[:: max(1, len(x) // 12)]], rotation=45, ha="right")
    tot_o = evo["cumulative_observed_deaths"].iloc[-1]
    tot_e = evo["cumulative_expected_deaths"].iloc[-1]
    ax1.text(
        0.02,
        0.98,
        f"Cumulative observed: {tot_o:.0f}\nCumulative expected: {tot_e:.0f}\nDiff (O−E): {tot_o - tot_e:.0f}",
        transform=ax1.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_expected_vs_observed_cumulative_only(
    evo: pd.DataFrame,
    out_path: Path,
    *,
    ve: float,
    wave_start: str,
    wave_end: str,
    dpi: int = 120,
) -> None:
    """Dedicated cumulative observed vs expected COVID deaths (debate figure)."""
    if evo.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    wk = evo["iso_week"].tolist()
    x = np.arange(len(wk))
    ax.plot(x, evo["cumulative_observed_deaths"], lw=2.5, label="cumulative observed COVID deaths")
    ax.plot(
        x,
        evo["cumulative_expected_deaths"],
        "--",
        lw=2.5,
        label=f"cumulative expected (VE_death={ve:.0%} scenario)",
    )
    ax.set_ylabel("cumulative deaths (vaccinated cohort)")
    ax.set_title("Cumulative observed vs expected COVID deaths under fixed VE (vaccinated cohort)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x[:: max(1, len(x) // 12)])
    ax.set_xticklabels([wk[i] for i in x[:: max(1, len(x) // 12)]], rotation=45, ha="right")
    _wave_shade(ax, sorted(evo["iso_week"].unique()), wave_start, wave_end, alpha=0.12, color="orange")
    tot_o = evo["cumulative_observed_deaths"].iloc[-1]
    tot_e = evo["cumulative_expected_deaths"].iloc[-1]
    ax.text(
        0.02,
        0.98,
        f"Final cumulative observed: {tot_o:.0f}\nFinal cumulative expected: {tot_e:.0f}\nO − E: {tot_o - tot_e:.0f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.55),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_cumulative_cases_deaths(
    weekly: pd.DataFrame,
    out_path: Path,
    *,
    age_bin: str = "all",
    wave_start: str,
    wave_end: str,
    dpi: int = 120,
) -> None:
    sub = weekly[weekly["age_bin"] == age_bin].copy()
    wk = sorted(sub["iso_week"].unique())
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    x = np.arange(len(wk))
    for cohort in sorted(sub["cohort"].unique()):
        s = sub[sub["cohort"] == cohort].set_index("iso_week").reindex(wk)
        c = s["cases"].fillna(0).cumsum()
        d = s["deaths_all"].fillna(0).cumsum()
        ax0.plot(x, c.values, label=f"{cohort} cases")
        ax1.plot(x, d.values, label=f"{cohort} deaths")
    ax0.set_ylabel("cumulative cases")
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax1.set_ylabel("cumulative all-cause deaths")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(x[:: max(1, len(x) // 12)])
    ax1.set_xticklabels([wk[i] for i in x[:: max(1, len(x) // 12)]], rotation=45, ha="right")
    wks = sorted(sub["iso_week"].unique())
    _wave_shade(ax0, wks, wave_start, wave_end, alpha=0.12, color="orange")
    _wave_shade(ax1, wks, wave_start, wave_end, alpha=0.12, color="orange")
    fig.suptitle(f"Cumulative cases and deaths (age_bin={age_bin})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_wave_ve_summary(
    wave_ve_summary: pd.DataFrame,
    out_path: Path,
    *,
    compare_cohort: str = "dose2",
    reference_cohort: str = "dose0",
    dpi: int = 120,
) -> None:
    """Grouped bars: four implied VEs vs reference by age_bin (decomposition)."""
    if wave_ve_summary.empty:
        return
    sub = wave_ve_summary[wave_ve_summary["cohort"] == compare_cohort].copy()
    if sub.empty:
        return
    metrics = [
        "ve_case_rate",
        "ve_cfr_covid",
        "ve_covid_death_rate",
        "ve_allcause_death_rate",
    ]
    if not all(m in sub.columns for m in metrics):
        return
    age_bins = sorted(sub["age_bin"].unique(), key=lambda x: (x != "all", str(x)))

    def _ve(ab: str, col: str) -> float:
        r = sub[sub["age_bin"] == ab]
        if r.empty:
            return float("nan")
        v = r.iloc[0][col]
        return float(v) if pd.notna(v) else float("nan")

    x = np.arange(len(age_bins))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, m in enumerate(metrics):
        vals = np.array([_ve(ab, m) for ab in age_bins], dtype=float)
        ax.bar(x + (i - 1.5) * width, vals, width, label=m.replace("ve_", "VE ").replace("_", " "))
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(age_bins, rotation=25, ha="right")
    ax.set_ylabel("implied VE (1 − RR)")
    ax.set_title(f"Wave implied VE: {compare_cohort} vs {reference_cohort}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_vaccine_coverage_by_age(
    coverage_df: pd.DataFrame,
    out_path: Path,
    *,
    age_bin: str | None = None,
    dpi: int = 120,
) -> None:
    """Time series: ``coverage_ge1`` vs ISO week, one line per fine ``age_bin`` (excludes ``all``)."""
    if coverage_df.empty or "coverage_ge1" not in coverage_df.columns:
        return
    sub = coverage_df[coverage_df["age_bin"] != "all"].copy()
    if sub.empty:
        return
    wk = sorted(sub["iso_week"].unique())
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(wk))
    age_bins = sorted(sub["age_bin"].unique(), key=lambda z: str(z))
    for ab in age_bins:
        s = sub[sub["age_bin"] == ab].set_index("iso_week").reindex(wk)
        y = s["coverage_ge1"].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", ms=2, label=ab)
    step = max(1, len(x) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([wk[i] for i in x[::step]], rotation=45, ha="right")
    ax.set_ylabel("coverage_ge1 (≥1 dose)")
    ax.set_xlabel("ISO week")
    ax.set_title("Vaccine coverage by enrollment age bin (descriptive)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_vaccine_coverage_all(
    coverage_df: pd.DataFrame,
    out_path: Path,
    *,
    dpi: int = 120,
) -> None:
    """``coverage_ge1`` for ``age_bin == 'all'`` only."""
    if coverage_df.empty:
        return
    sub = coverage_df[coverage_df["age_bin"] == "all"].sort_values("iso_week", kind="mergesort")
    if sub.empty:
        return
    wk = sub["iso_week"].tolist()
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(wk))
    ax.plot(x, sub["coverage_ge1"].to_numpy(dtype=float), marker="o", ms=2, color="C0", label="all")
    step = max(1, len(x) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([wk[i] for i in x[::step]], rotation=45, ha="right")
    ax.set_ylabel("coverage_ge1 (≥1 dose)")
    ax.set_xlabel("ISO week")
    ax.set_title("Vaccine coverage — full population (age_bin=all)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_km_survival_curves(
    km_summary: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    xlabel: str,
    dpi: int = 120,
    cohort_order: tuple[str, ...] | list[str] | None = None,
) -> None:
    """Shared KM step plot with dynamic survival y-limits."""
    if km_summary.empty:
        return
    sub = km_summary.copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    seen = set(sub["cohort"].unique())
    if cohort_order:
        ordered = [c for c in cohort_order if c in seen]
        ordered.extend(sorted(c for c in seen if c not in ordered))
    else:
        ordered = list(sub["cohort"].unique())
    for cohort in ordered:
        s = sub[sub["cohort"] == cohort].sort_values("timeline")
        ax.step(s["timeline"], s["KM_estimate"], where="post", label=cohort)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("survival (all-cause)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    yvals = pd.to_numeric(sub["KM_estimate"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(yvals) == 0:
        ax.set_ylim(0.0, 1.02)
    else:
        y_min = float(yvals.min())
        y_max = float(yvals.max())
        span = max(y_max - y_min, 0.04)
        pad_lo = max(0.01, span * 0.06)
        pad_hi = max(0.004, span * 0.04)
        y_lo = max(0.0, y_min - pad_lo)
        y_hi = min(1.01, y_max + pad_hi)
        if y_hi - y_lo < 0.06:
            y_hi = min(1.02, y_lo + 0.10)
        ax.set_ylim(y_lo, y_hi)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_km_post_infection(
    km_summary: pd.DataFrame,
    out_path: Path,
    *,
    age_bin: str | None = None,
    dpi: int = 120,
) -> None:
    if km_summary.empty:
        return
    sub = km_summary.copy()
    if age_bin is not None and "age_bin" in sub.columns:
        sub = sub[sub["age_bin"] == age_bin].copy()
    if sub.empty:
        return
    # Title label when table has no age_bin column (aggregate KM): use explicit age_bin kwarg only for display
    age_bin_title = age_bin
    if age_bin_title is None and "age_bin" not in sub.columns:
        age_bin_title = "all"

    if age_bin_title is not None:
        age_title = str(age_bin_title)
    elif "age_bin" in sub.columns:
        uniq = sub["age_bin"].dropna().unique()
        age_title = str(uniq[0]) if len(uniq) == 1 else ",".join(sorted(map(str, uniq)))
    else:
        age_title = "all"

    plot_km_survival_curves(
        sub,
        out_path,
        title=(
            "Kaplan–Meier survival post first infection (all-cause death)\n"
            f"enrollment age_bin={age_title}"
        ),
        xlabel="weeks since infection",
        dpi=dpi,
    )


def plot_old_young_difference(
    diff_weekly: pd.DataFrame,
    out_path: Path,
    *,
    break_iso_week: str,
    dpi: int = 120,
) -> None:
    """Old-minus-young weekly rate differences for COVID and non-COVID mortality."""
    if diff_weekly.empty:
        return
    wk = diff_weekly["iso_week"].tolist()
    x = np.arange(len(wk))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    series = [
        ("mortality_rate_covid_diff_old_minus_young", "COVID mortality rate diff"),
        ("mortality_rate_non_covid_diff_old_minus_young", "Non-COVID mortality rate diff"),
    ]
    for ax, (col, title) in zip(axes, series):
        if col not in diff_weekly.columns:
            continue
        y = diff_weekly[col].to_numpy(dtype=float)
        ax.plot(x, y, lw=2)
        ax.axhline(0.0, color="gray", lw=0.8)
        _wave_shade(ax, wk, break_iso_week, break_iso_week, alpha=0.20, color="orange")
        ax.set_ylabel("old - young")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    step = max(1, len(x) // 12)
    axes[-1].set_xticks(x[::step])
    axes[-1].set_xticklabels([wk[i] for i in x[::step]], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_placebo_break_scan(
    placebo_scan: pd.DataFrame,
    out_path: Path,
    *,
    actual_break_iso_week: str,
    dpi: int = 120,
) -> None:
    """Absolute slope change by placebo break week for old-minus-young differences."""
    if placebo_scan.empty:
        return
    outcomes = [
        "mortality_rate_covid_diff_old_minus_young",
        "mortality_rate_non_covid_diff_old_minus_young",
    ]
    sub = placebo_scan[placebo_scan["outcome"].isin(outcomes)].copy()
    if sub.empty:
        return
    wk = sorted(sub["break_iso_week"].unique())
    x = np.arange(len(wk))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for outcome in outcomes:
        s = sub[sub["outcome"] == outcome].set_index("break_iso_week").reindex(wk)
        ax.plot(
            x,
            s["slope_change"].abs().to_numpy(dtype=float),
            marker="o",
            ms=3,
            lw=1.5,
            label=outcome.replace("_diff_old_minus_young", "").replace("_", " "),
        )
    if actual_break_iso_week in wk:
        ax.axvline(wk.index(actual_break_iso_week), color="orange", lw=1.2, linestyle="--", label="actual break")
    step = max(1, len(x) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([wk[i] for i in x[::step]], rotation=45, ha="right")
    ax.set_ylabel("|slope change|")
    ax.set_title("Placebo breakpoint scan")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
