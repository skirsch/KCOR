"""
Dot plots: wave-segment slopes vs calendar midpoint from
debate/data/owid_slope/owid_all_locations_wave_slopes.csv (vaccination inputs from owid_source/).

1) All OWID locations with R² ≥ threshold: one scatter + single log10(slope) vs time trend
   (legend includes mean OWID doses/million across locations in the chart, snapshot date).
2) Top / bottom N by vaccination (defaults: N = 25 and 70) — two scatters + separate regressions each.
   Override with --vax-n-each N [N ...]. If fewer than 2N locations have vax + wave data, skip.

Vaccination metric: total_vaccinations_per_hundred × 10_000 (doses per million people),
fallback to people_vaccinated_per_hundred × 10_000 if total is missing. ISO rows only
(excludes OWID_* aggregates).

Reads debate/data/owid_slope/owid_all_locations_wave_slopes.csv (and owid_source vax).
Writes PNGs only under debate/figures/owid_slopes/ (no CSV output). Default log y-limits 1–400.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OWID_SOURCE_DIR = REPO_ROOT / "debate" / "data" / "owid_source"
OWID_SLOPE_DATA_DIR = REPO_ROOT / "debate" / "data" / "owid_slope"
DEFAULT_INPUT = OWID_SLOPE_DATA_DIR / "owid_all_locations_wave_slopes.csv"
DEFAULT_VAX = OWID_SOURCE_DIR / "OWID_vaccinations.csv"
DEFAULT_FIG_DIR = REPO_ROOT / "debate" / "figures" / "owid_slopes"
DEFAULT_OUTPUT = DEFAULT_FIG_DIR / "owid_wave_slopes_midpoint_r2ge99_dotplot.png"
DEFAULT_VAX_FIG_DIR = DEFAULT_FIG_DIR
DEFAULT_VAX_N_EACH = [25, 70]


def ols_r2_line(
    x_num: np.ndarray, y: np.ndarray, log_y: bool
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Return (poly_coeffs_high_to_low, r_squared, y_pred_on_points).
    If log_y: fit log10(y) = p[0]*x + p[1]; r² is for log10(y); y_pred = 10**polyval.
    Else: fit y = p[0]*x + p[1]; r² for y.
    """
    if log_y:
        ly = np.log10(y)
        p = np.polyfit(x_num, ly, 1)
        pred_l = np.polyval(p, x_num)
        ss_res = float(np.sum((ly - pred_l) ** 2))
        ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return p, r2, 10.0**pred_l
    p = np.polyfit(x_num, y, 1)
    pred = np.polyval(p, x_num)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return p, r2, pred


def load_vaccinations_per_million(
    vax_path: Path,
    cutoff: pd.Timestamp,
) -> pd.Series:
    """location -> total doses per million (or people vaccinated / M fallback). ISO locations only."""
    v = pd.read_csv(vax_path, parse_dates=["date"])
    v = v.loc[v["date"] <= cutoff].sort_values(["location", "date"]).groupby("location", as_index=False).last()
    v = v.loc[~v["iso_code"].astype(str).str.startswith("OWID")].copy()
    tvp = pd.to_numeric(v["total_vaccinations_per_hundred"], errors="coerce") * 10_000.0
    pvp = pd.to_numeric(v["people_vaccinated_per_hundred"], errors="coerce") * 10_000.0
    dpm = tvp.fillna(pvp)
    return pd.Series(dpm.values, index=v["location"].values, dtype=float)


def mean_dpm_across_countries(vax: pd.Series, countries: set[str]) -> float:
    """Unweighted mean doses/million among country names that appear in vax."""
    vals: list[float] = []
    for c in countries:
        if c not in vax.index:
            continue
        v = float(vax.loc[c])
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def ranked_vax_among_wave_countries(
    vax: pd.Series,
    countries_in_waves: set[str],
) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for loc in countries_in_waves:
        if loc not in vax.index:
            continue
        val = float(vax.loc[loc])
        if not np.isfinite(val):
            continue
        rows.append((loc, val))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def add_segment_midpoint(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t0 = out["week_start"].values.astype("datetime64[ns]")
    t1 = out["week_end"].values.astype("datetime64[ns]")
    out["segment_midpoint"] = pd.to_datetime((t0.astype(np.int64) + t1.astype(np.int64)) / 2.0)
    return out


def trend_legend_label(
    r2: float,
    y_fit_start: float,
    y_fit_end: float,
    prefix: str,
    *,
    cohort_mean_dpm: float | None = None,
) -> str:
    ratio = y_fit_end / y_fit_start if y_fit_start > 0 else float("nan")
    s = f"{prefix} R² = {r2:.3f}, fitted end/start = {ratio:.2f}x"
    if cohort_mean_dpm is not None and np.isfinite(cohort_mean_dpm):
        s += f", cohort mean doses/M = {cohort_mean_dpm:,.0f}"
    return s


def plot_all_segments(
    df: pd.DataFrame,
    out_path: Path,
    *,
    r2_min: float,
    use_log: bool,
    ylim_log: tuple[float, float],
    mean_dpm_locations: float | None = None,
    vax_cutoff: pd.Timestamp | None = None,
) -> None:
    y = df["slope_ols_deaths_per_million_per_week"].astype(float).values
    mid = df["segment_midpoint"]
    x_num = mdates.date2num(mid)

    coef, r2_line, _pred = ols_r2_line(x_num, y, log_y=use_log)
    x_line_num = np.linspace(float(x_num.min()), float(x_num.max()), 200)
    if use_log:
        y_line = 10.0 ** np.polyval(coef, x_line_num)
    else:
        y_line = np.polyval(coef, x_line_num)
    y_fit_start = float(y_line[0])
    y_fit_end = float(y_line[-1])
    fit_ratio = y_fit_end / y_fit_start if y_fit_start > 0 else float("nan")
    snap = f", ≤{vax_cutoff.date()}" if vax_cutoff is not None else ""
    if use_log:
        trend_legend = trend_legend_label(
            r2_line,
            y_fit_start,
            y_fit_end,
            "log10(slope) trend",
            cohort_mean_dpm=mean_dpm_locations,
        )
    else:
        trend_legend = trend_legend_label(
            r2_line,
            y_fit_start,
            y_fit_end,
            "slope trend",
            cohort_mean_dpm=mean_dpm_locations,
        )

    if mean_dpm_locations is not None and np.isfinite(mean_dpm_locations):
        seg_lbl = (
            f"Auto-identified wave segments (mean {mean_dpm_locations:,.0f} doses/million{snap}, "
            f"by location)"
        )
    else:
        seg_lbl = "Auto-identified wave segments"

    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    ax.scatter(
        mid,
        y,
        s=14,
        alpha=0.35,
        c="#1f77b4",
        edgecolors="none",
        zorder=3,
        label=seg_lbl,
    )
    ax.plot(
        mdates.num2date(x_line_num),
        y_line,
        color="#c0392b",
        linewidth=2.2,
        zorder=4,
        label=trend_legend,
    )
    ax.set_xlabel("Segment calendar midpoint (mean of week_start and week_end)")
    ax.set_ylabel("OLS slope (cumulative deaths per million per week)")
    ax.set_title(
        f"All OWID countries · All auto-identified wave segments · R² ≥ {r2_min:g}\n"
        f"Slope vs segment calendar midpoint (n = {len(df)})",
        fontsize=11,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.grid(True, alpha=0.28, zorder=0)
    if use_log:
        ax.set_yscale("log")
        lo, hi = ylim_log
        if lo <= 0 or hi <= 0:
            raise SystemExit("--ylim-log values must be positive for log scale")
        ax.set_ylim(lo, hi)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    extra = ""
    if mean_dpm_locations is not None and np.isfinite(mean_dpm_locations):
        extra = f"  |  mean doses/M (by location)={mean_dpm_locations:,.0f}"
    print(
        f"Wrote {out_path} ({len(df)} points)  |  trend R²={r2_line:.4f}, "
        f"fitted {y_fit_start:.2f} → {y_fit_end:.2f} d/m/wk (end/start={fit_ratio:.2f}x){extra}"
    )


def plot_vaccination_split(
    df: pd.DataFrame,
    ranked: list[tuple[str, float]],
    n_each: int,
    out_path: Path,
    *,
    vax_cutoff: pd.Timestamp,
    r2_min: float,
    use_log: bool,
    ylim_log: tuple[float, float],
    verbose: bool = False,
) -> None:
    """Top n_each vs bottom n_each by vaccination; two regressions on log10(slope) vs time."""
    if len(ranked) < 2 * n_each:
        raise SystemExit(
            f"Need ≥{2 * n_each} locations with vaccination + wave data; have {len(ranked)}"
        )

    top_names = [loc for loc, _ in ranked[:n_each]]
    bottom_names = [loc for loc, _ in ranked[-n_each:]]
    mean_dpm_hi = float(np.mean([v for _, v in ranked[:n_each]]))
    mean_dpm_lo = float(np.mean([v for _, v in ranked[-n_each:]]))

    d_hi = df.loc[df["country"].isin(top_names)].copy()
    d_lo = df.loc[df["country"].isin(bottom_names)].copy()

    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")

    color_hi, color_lo = "#1a5276", "#8b4513"
    line_hi, line_lo = "#2980b9", "#e74c3c"

    x_all_num: list[float] = []
    legends_lines: list[str] = []
    r2_hi = r2_lo = float("nan")
    ratio_hi = ratio_lo = float("nan")

    if d_hi.empty:
        print("  warning: no wave segments for top-vaccination group")
    else:
        xm = mdates.date2num(d_hi["segment_midpoint"])
        ys = d_hi["slope_ols_deaths_per_million_per_week"].astype(float).values
        x_all_num.extend(xm.tolist())
        ax.scatter(
            d_hi["segment_midpoint"],
            ys,
            s=22,
            alpha=0.45,
            c=color_hi,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
            label=(
                f"Top {n_each} by vax: mean {mean_dpm_hi:,.0f} doses/million "
                f"(n={len(d_hi)} segs, {d_hi['country'].nunique()} locs)"
            ),
        )
    if d_lo.empty:
        print("  warning: no wave segments for bottom-vaccination group")
    else:
        xm = mdates.date2num(d_lo["segment_midpoint"])
        ys = d_lo["slope_ols_deaths_per_million_per_week"].astype(float).values
        x_all_num.extend(xm.tolist())
        ax.scatter(
            d_lo["segment_midpoint"],
            ys,
            s=22,
            alpha=0.45,
            c=color_lo,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
            label=(
                f"Bottom {n_each} by vax: mean {mean_dpm_lo:,.0f} doses/million "
                f"(n={len(d_lo)} segs, {d_lo['country'].nunique()} locs)"
            ),
        )

    if len(x_all_num) < 2:
        plt.close(fig)
        raise SystemExit("Not enough points to plot vaccination split")

    x_lo, x_hi = min(x_all_num), max(x_all_num)
    x_line_num = np.linspace(x_lo, x_hi, 200)

    if not d_hi.empty and len(d_hi) >= 2:
        xm = mdates.date2num(d_hi["segment_midpoint"])
        ys = d_hi["slope_ols_deaths_per_million_per_week"].astype(float).values
        coef, r2_g, _ = ols_r2_line(xm, ys, log_y=use_log)
        if use_log:
            y_line = 10.0 ** np.polyval(coef, x_line_num)
        else:
            y_line = np.polyval(coef, x_line_num)
        y0, y1 = float(y_line[0]), float(y_line[-1])
        prefix = "High vax log10(slope)" if use_log else "High vax slope"
        leg = trend_legend_label(r2_g, y0, y1, prefix, cohort_mean_dpm=mean_dpm_hi)
        legends_lines.append(leg)
        r2_hi = r2_g
        ratio_hi = y1 / y0 if y0 > 0 else float("nan")
        ax.plot(mdates.num2date(x_line_num), y_line, color=line_hi, linewidth=2.4, zorder=4, label=leg)
    elif not d_hi.empty:
        print("  warning: top-vax group has <2 segments — no regression line")

    if not d_lo.empty and len(d_lo) >= 2:
        xm = mdates.date2num(d_lo["segment_midpoint"])
        ys = d_lo["slope_ols_deaths_per_million_per_week"].astype(float).values
        coef, r2_g, _ = ols_r2_line(xm, ys, log_y=use_log)
        if use_log:
            y_line = 10.0 ** np.polyval(coef, x_line_num)
        else:
            y_line = np.polyval(coef, x_line_num)
        y0, y1 = float(y_line[0]), float(y_line[-1])
        prefix = "Low vax log10(slope)" if use_log else "Low vax slope"
        leg = trend_legend_label(r2_g, y0, y1, prefix, cohort_mean_dpm=mean_dpm_lo)
        legends_lines.append(leg)
        r2_lo = r2_g
        ratio_lo = y1 / y0 if y0 > 0 else float("nan")
        ax.plot(mdates.num2date(x_line_num), y_line, color=line_lo, linewidth=2.4, zorder=4, label=leg)
    elif not d_lo.empty:
        print("  warning: bottom-vax group has <2 segments — no regression line")

    ax.set_xlabel("Segment calendar midpoint (mean of week_start and week_end)")
    ax.set_ylabel("OLS slope (cumulative deaths per million per week)")
    reg_desc = "log10(slope) vs time" if use_log else "slope vs time"
    ax.set_title(
        f"Top {n_each} vs bottom {n_each} by OWID total doses per million (≤ {vax_cutoff.date()})\n"
        f"Among locations with ≥1 auto-identified segment · segment R² ≥ {r2_min:g} · "
        f"two separate {reg_desc} regressions",
        fontsize=9.5,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.grid(True, alpha=0.28, zorder=0)
    if use_log:
        ax.set_yscale("log")
        lo, hi = ylim_log
        ax.set_ylim(lo, hi)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    if verbose:
        print(f"Wrote {out_path}")
        print(f"  Top {n_each} cohort mean doses/million: {mean_dpm_hi:,.0f}")
        print(f"  Bottom {n_each} cohort mean doses/million: {mean_dpm_lo:,.0f}")
        print(f"  Top {n_each}: {', '.join(top_names)}")
        print(f"  Bottom {n_each}: {', '.join(bottom_names)}")
        for line in legends_lines:
            print(f"  {line}")
    else:
        print(
            f"Wrote {out_path}  |  high vax: mean doses/M={mean_dpm_hi:,.0f}, R²={r2_hi:.3f}, "
            f"end/start={ratio_hi:.2f}x  |  low vax: mean doses/M={mean_dpm_lo:,.0f}, R²={r2_lo:.3f}, "
            f"end/start={ratio_lo:.2f}x"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Wave slopes CSV (default: debate/data/owid_slope/owid_all_locations_wave_slopes.csv)",
    )
    ap.add_argument(
        "--vax-csv",
        type=Path,
        default=DEFAULT_VAX,
        help="OWID vaccinations CSV (default: debate/data/owid_source/OWID_vaccinations.csv)",
    )
    ap.add_argument(
        "--vax-cutoff",
        type=str,
        default="2022-01-01",
        help="Use latest row per location on or before this date (default: 2022-01-01)",
    )
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG (all locations)")
    ap.add_argument(
        "--output-vax-dir",
        type=Path,
        default=DEFAULT_VAX_FIG_DIR,
        help="Directory for vaccination split PNGs (default: debate/figures/owid_slopes)",
    )
    ap.add_argument(
        "--vax-n-each",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Top/bottom sizes (default: 25 70)",
    )
    ap.add_argument(
        "--vax-verbose",
        action="store_true",
        help="Print full country lists and regression legend lines for each vax split plot",
    )
    ap.add_argument(
        "--no-vax-plots",
        action="store_true",
        help="Skip vaccination comparison figures (only plot all locations)",
    )
    ap.add_argument(
        "--r2-min",
        type=float,
        default=0.99,
        help="Keep segments with r2_fit >= this (default: 0.99)",
    )
    ap.add_argument("--linear-y", action="store_true", help="Use linear y-axis (default: log)")
    ap.add_argument(
        "--ylim-log",
        type=float,
        nargs=2,
        default=(1.0, 400.0),
        metavar=("YMIN", "YMAX"),
        help="Y limits when using log scale (default: 1 400)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["week_start", "week_end"])
    df = df.loc[np.isfinite(df["r2_fit"]) & np.isfinite(df["slope_ols_deaths_per_million_per_week"])].copy()
    df = df.loc[df["r2_fit"] >= args.r2_min].copy()
    if df.empty:
        raise SystemExit(f"No rows with r2_fit >= {args.r2_min} in {args.input}")

    df = add_segment_midpoint(df)
    y = df["slope_ols_deaths_per_million_per_week"].astype(float).values
    if (y <= 0).any():
        raise SystemExit("Non-positive slopes cannot be shown on log y; fix data or use --linear-y")

    use_log = not args.linear_y
    ylim_log = (float(args.ylim_log[0]), float(args.ylim_log[1]))

    vax_cutoff = pd.Timestamp(args.vax_cutoff)
    mean_dpm_main: float | None = None
    vax: pd.Series | None = None
    ranked: list[tuple[str, float]] = []
    if args.vax_csv.is_file():
        vax = load_vaccinations_per_million(args.vax_csv, vax_cutoff)
        countries_waves = set(df["country"].unique())
        mean_dpm_main = mean_dpm_across_countries(vax, countries_waves)
        ranked = ranked_vax_among_wave_countries(vax, countries_waves)
    elif not args.no_vax_plots:
        raise SystemExit(f"Vaccination file not found: {args.vax_csv} (use --no-vax-plots for main chart only)")

    plot_all_segments(
        df,
        args.output,
        r2_min=args.r2_min,
        use_log=use_log,
        ylim_log=ylim_log,
        mean_dpm_locations=mean_dpm_main,
        vax_cutoff=vax_cutoff if mean_dpm_main is not None and np.isfinite(mean_dpm_main) else None,
    )

    if args.no_vax_plots:
        return

    assert vax is not None

    n_list = args.vax_n_each if args.vax_n_each is not None else DEFAULT_VAX_N_EACH
    for n in n_list:
        if n < 1:
            raise SystemExit(f"Invalid --vax-n-each value {n!r} (must be ≥1)")
        if 2 * n > len(ranked):
            print(
                f"  skip top{n:02d}/bottom{n:02d}: need {2 * n} distinct locations "
                f"(top + bottom), only {len(ranked)} have vaccination + wave data"
            )
            continue
        out_vax = args.output_vax_dir / f"owid_wave_slopes_vaccination_top{n:02d}_bottom{n:02d}.png"
        plot_vaccination_split(
            df,
            ranked,
            n,
            out_vax,
            vax_cutoff=vax_cutoff,
            r2_min=args.r2_min,
            use_log=use_log,
            ylim_log=ylim_log,
            verbose=args.vax_verbose,
        )


if __name__ == "__main__":
    main()
