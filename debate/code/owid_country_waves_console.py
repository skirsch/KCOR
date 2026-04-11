"""
One country: waves = contiguous weekly windows where cumulative deaths vs week
index are nearly linear (R² ≥ R2_MIN), on OWID cumulative deaths per million.

Activity islands: smoothed weekly new deaths (3w MA) g > G_ACTIVE. Before the first
such week, optional prefix islands use g > G_PREFIX so the small spring-2020 stair
(israel-style) is not missed; prefix pieces that end the week before main activity
starts are dropped so the July–Aug ramp stays part of the first major wave.

Each island is recursively split at deep interior troughs. Within each piece we take
the longest high-R² subinterval, then left/right gaps — multiple waves per piece.

Daily rows through STOP_DATE (inclusive); weekly series built from that range.

Default country: Israel. Override with --country \"Czechia\" etc.

For a single-country run, writes a PNG of cumulative deaths with each wave’s OLS fit
overlaid (default path next to the CSV). Use --no-plot to skip.

Use --all-countries to process every OWID column (except date) into one CSV; no PNGs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "debate" / "data" / "OWID_total_deaths_per_million.csv"
DEFAULT_COUNTRY = "Israel"

# Include all daily observations on or before this date (waves detected through this cutoff).
STOP_DATE = pd.Timestamp("2022-04-01")

SMOOTH_WEEKS = 3
G_ACTIVE = 8.0
# Weeks with index < first week where g > G_ACTIVE: also form islands with this lower
# bar (captures low early surges). Must stay below G_ACTIVE to avoid duplicating main islands.
G_PREFIX = 1.2
SPLIT_TROUGH_FRAC = 0.42
# Do not try to split a piece smaller than this (weeks).
MIN_PIECE_WEEKS_FOR_SPLIT = 14
R2_MIN = 0.992
R2_FALLBACK = 0.99
# Primary minimum window length (weeks). Shorter surges (e.g. sharp Omicron) may not
# have 6 consecutive weeks with R²≥0.99 on cumulative; allow 5-week windows as fallback.
MIN_WAVE_WEEKS = 6
MIN_WAVE_WEEKS_SHORT = 5
# Prefix-only (pre–g>G_ACTIVE): very small early stairs may only support 4w at R²≥0.99.
MIN_WAVE_WEEKS_PREFIX = 4


def slug_country(name: str) -> str:
    s = name.strip().lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s or "country"


def default_output_csv(country: str) -> Path:
    return REPO_ROOT / "debate" / "data" / f"owid_{slug_country(country)}_wave_slopes.csv"


def default_plot_png(country: str) -> Path:
    return REPO_ROOT / "debate" / "data" / f"owid_{slug_country(country)}_wave_fits.png"


DEFAULT_ALL_COUNTRIES_CSV = REPO_ROOT / "debate" / "data" / "owid_all_locations_wave_slopes.csv"


def country_wave_analysis(
    df: pd.DataFrame,
    country: str,
) -> tuple[list[dict], pd.DatetimeIndex, np.ndarray, list[dict]]:
    """
    Build weekly cumulative, detect waves, return CSV rows and data for optional PNG.
    df must include 'date' and `country`, already filtered to the analysis end date.
    """
    rows: list[dict] = []
    plot_segments: list[dict] = []

    if country not in df.columns:
        return rows, pd.DatetimeIndex([]), np.array([]), plot_segments

    daily = pd.to_numeric(df[country], errors="coerce")
    if daily.isna().all():
        return rows, pd.DatetimeIndex([]), np.array([]), plot_segments

    daily = daily.ffill().bfill()
    daily.index = pd.DatetimeIndex(df["date"])
    w_cum = daily.resample("W-SUN").last()
    if w_cum.size == 0:
        return rows, w_cum.index, np.array([]), plot_segments

    w_new = w_cum.diff().fillna(0.0).clip(lower=0.0)
    g_series = w_new.rolling(SMOOTH_WEEKS, min_periods=1).mean()
    g = g_series.values.astype(float)
    cum = w_cum.values.astype(float)
    idx = w_cum.index

    prefix = prefix_activity_pieces(g)
    prefix_set = set(prefix)
    islands = activity_islands(g)
    pieces: list[tuple[int, int]] = list(prefix)
    for a, b in islands:
        pieces.extend(fully_split_island(g, a, b))
    pieces.sort(key=lambda t: t[0])

    waves: list[tuple[int, int, float, str]] = []
    for lo, hi in pieces:
        waves.extend(all_waves_in_piece(cum, lo, hi, prefix_piece=(lo, hi) in prefix_set))
    waves.sort(key=lambda t: t[0])

    for k, (lo, hi, _r2_win, tag) in enumerate(waves, start=1):
        y = cum[lo : hi + 1]
        slope, intercept, r2 = ols_slope_intercept_r2(y)
        n = hi - lo + 1
        delta = float(y[-1] - y[0])
        secant = delta / (n - 1) if n > 1 else float("nan")
        span_days = int((pd.Timestamp(idx[hi]) - pd.Timestamp(idx[lo])).days)
        rows.append(
            {
                "country": country,
                "wave": k,
                "r2_fit": r2,
                "fit_tier": tag,
                "week_start": str(idx[lo].date()),
                "week_end": str(idx[hi].date()),
                "n_weekly_observations": n,
                "n_week_intervals": n - 1,
                "span_calendar_days": span_days,
                "slope_ols_deaths_per_million_per_week": slope,
                "slope_secant_delta_over_n_minus_1": secant,
                "delta_cumulative": delta,
            }
        )
        plot_segments.append(
            {
                "lo": lo,
                "hi": hi,
                "wave": k,
                "slope": slope,
                "intercept": intercept,
                "r2": r2,
            }
        )

    return rows, idx, cum, plot_segments


def write_wave_fit_png(
    out_path: Path,
    country: str,
    stop_date: pd.Timestamp,
    week_ends: pd.DatetimeIndex,
    cum: np.ndarray,
    segments: list[dict],
) -> None:
    """Full weekly cumulative plus OLS fit line and points for each wave window."""
    import matplotlib.dates as mdates
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    ax.plot(
        week_ends,
        cum,
        color="#2d2d2d",
        linewidth=1.25,
        alpha=0.88,
        label="Weekly cumulative (W-SUN)",
        zorder=1,
    )

    for seg in segments:
        lo = int(seg["lo"])
        hi = int(seg["hi"])
        k = int(seg["wave"])
        slope = float(seg["slope"])
        intercept = float(seg["intercept"])
        r2 = float(seg["r2"])
        n = hi - lo + 1
        x_rel = np.arange(n, dtype=float)
        y_fit = slope * x_rel + intercept
        widx = week_ends[lo : hi + 1]
        y_act = cum[lo : hi + 1]
        color = f"C{(k - 1) % 10}"
        ax.plot(
            widx,
            y_fit,
            color=color,
            linewidth=2.5,
            linestyle="-",
            label=f"Wave {k} fit  (R²={r2:.4f})",
            zorder=2 + k,
        )
        ax.scatter(
            widx,
            y_act,
            color=color,
            s=26,
            zorder=5 + k,
            edgecolors="white",
            linewidths=0.6,
        )
        if np.isfinite(slope):
            t = ax.annotate(
                f"{slope:.4f}",
                xy=(widx[0], float(y_fit[0])),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=8,
                color=color,
                fontweight="semibold",
                zorder=10 + k,
            )
            t.set_path_effects(
                [pe.withStroke(linewidth=2.8, foreground="white", capstyle="round")]
            )

    ax.set_title(
        f"{country} — cumulative COVID-19 deaths per million (OWID), through {stop_date.date()}"
    )
    ax.set_xlabel("Week ending (Sunday)")
    ax.set_ylabel("Cumulative deaths / million")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def ols_slope_intercept_r2(y: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    x = np.arange(n, dtype=float)
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan")
    slope = np.sum((x - xm) * (y - ym)) / sxx
    intercept = ym - slope * xm
    y_hat = slope * x + intercept
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - ym) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def r2_only(y: np.ndarray) -> float:
    _s, _i, r2 = ols_slope_intercept_r2(y)
    return float(r2)


def activity_islands_above(g: np.ndarray, thr: float, lo: int, hi: int) -> list[tuple[int, int]]:
    """Contiguous runs with g > thr on inclusive index range [lo, hi]."""
    out: list[tuple[int, int]] = []
    i = lo
    while i <= hi:
        if g[i] <= thr:
            i += 1
            continue
        j = i
        while j + 1 <= hi and g[j + 1] > thr:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def activity_islands(g: np.ndarray) -> list[tuple[int, int]]:
    n = len(g)
    if n == 0:
        return []
    return activity_islands_above(g, G_ACTIVE, 0, n - 1)


def prefix_activity_pieces(g: np.ndarray) -> list[tuple[int, int]]:
    """Small pre-main surges (e.g. spring 2020): g > G_PREFIX only before first g > G_ACTIVE week."""
    n = len(g)
    if n == 0 or G_PREFIX >= G_ACTIVE:
        return []
    first_main = next((i for i in range(n) if g[i] > G_ACTIVE), n)
    if first_main == 0:
        return []
    raw = activity_islands_above(g, G_PREFIX, 0, first_main - 1)
    # Drop islands that end the last week before main activity — same stair as the first major wave.
    keep: list[tuple[int, int]] = []
    last_pre = first_main - 1
    for a, b in raw:
        if b < last_pre:
            keep.append((a, b))
    return keep


def fully_split_island(g: np.ndarray, a: int, b: int) -> list[tuple[int, int]]:
    """Recursively split [a,b] at the deepest interior trough until pieces are indivisible."""

    leaves: list[tuple[int, int]] = []

    def recurse(a0: int, b0: int) -> None:
        n = b0 - a0 + 1
        if n < MIN_PIECE_WEEKS_FOR_SPLIT:
            leaves.append((a0, b0))
            return
        seg = g[a0 : b0 + 1]
        edge = max(3, n // 10)
        if n <= 2 * edge + 4:
            leaves.append((a0, b0))
            return
        inner = seg[edge:-edge]
        if inner.size == 0:
            leaves.append((a0, b0))
            return
        rel = int(np.argmin(inner)) + edge
        trough = float(seg[rel])
        peak = float(np.max(seg))
        if peak <= 0 or trough > SPLIT_TROUGH_FRAC * peak:
            leaves.append((a0, b0))
            return
        mid = a0 + rel
        recurse(a0, mid)
        recurse(mid + 1, b0)

    recurse(a, b)
    return leaves


def longest_r2_subinterval(
    cum: np.ndarray,
    lo: int,
    hi: int,
    r2_min: float,
    min_len: int,
) -> tuple[int, int, float] | None:
    best: tuple[int, int, float] | None = None
    best_len = -1
    for i in range(lo, hi + 1):
        for j in range(i + min_len - 1, hi + 1):
            rr = r2_only(cum[i : j + 1])
            if rr >= r2_min and j - i + 1 > best_len:
                best_len = j - i + 1
                best = (i, j, rr)
    return best


def all_waves_in_piece(
    cum: np.ndarray, lo: int, hi: int, *, prefix_piece: bool = False
) -> list[tuple[int, int, float, str]]:
    """Longest high-R² window in [lo,hi], then recurse in left/right gaps (multiple waves per piece)."""
    found: list[tuple[int, int, float, str]] = []
    min_gap = MIN_WAVE_WEEKS_PREFIX if prefix_piece else MIN_WAVE_WEEKS_SHORT

    def pick_longest(a: int, b: int) -> tuple[int, int, float] | None:
        w = longest_r2_subinterval(cum, a, b, R2_MIN, MIN_WAVE_WEEKS)
        if w is None:
            w = longest_r2_subinterval(cum, a, b, R2_MIN, MIN_WAVE_WEEKS_SHORT)
        if w is None:
            w = longest_r2_subinterval(cum, a, b, R2_FALLBACK, MIN_WAVE_WEEKS)
        if w is None:
            w = longest_r2_subinterval(cum, a, b, R2_FALLBACK, MIN_WAVE_WEEKS_SHORT)
        if w is None and prefix_piece:
            w = longest_r2_subinterval(cum, a, b, R2_MIN, MIN_WAVE_WEEKS_PREFIX)
        if w is None and prefix_piece:
            w = longest_r2_subinterval(cum, a, b, R2_FALLBACK, MIN_WAVE_WEEKS_PREFIX)
        return w

    def rec(a: int, b: int) -> None:
        if b - a + 1 < min_gap:
            return
        w = pick_longest(a, b)
        if w is None:
            return
        i, j, r2 = w
        tag = "primary" if r2 >= R2_MIN else f"fallback≥{R2_FALLBACK}"
        found.append((i, j, r2, tag))
        if a <= i - 1 and i - 1 - a + 1 >= min_gap:
            rec(a, i - 1)
        if j + 1 <= b and b - (j + 1) + 1 >= min_gap:
            rec(j + 1, b)

    rec(lo, hi)
    return sorted(found, key=lambda x: x[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-countries",
        action="store_true",
        help="Process every location column into a single CSV (no per-country PNGs)",
    )
    parser.add_argument(
        "--country",
        type=str,
        default=DEFAULT_COUNTRY,
        help=f"OWID column name (default: {DEFAULT_COUNTRY}; ignored with --all-countries)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "CSV path (default: owid_<country>_wave_slopes.csv, or "
            "owid_all_locations_wave_slopes.csv with --all-countries)"
        ),
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write CSV; only print to stdout",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not write the wave-fit PNG",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="PNG path (default: debate/data/owid_<country>_wave_fits.png)",
    )
    args = parser.parse_args()
    path = DEFAULT_CSV
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = df.loc[df["date"] <= STOP_DATE].copy()

    if args.all_countries:
        out_csv = args.output if args.output is not None else DEFAULT_ALL_COUNTRIES_CSV
        cols = [c for c in df.columns if c != "date"]
        all_rows: list[dict] = []
        for col in cols:
            rows, _idx, _cum, _segs = country_wave_analysis(df, col)
            all_rows.extend(rows)
        out_df = pd.DataFrame(all_rows)
        if not out_df.empty:
            out_df = out_df.sort_values(["country", "wave"], kind="mergesort").reset_index(drop=True)
        if not args.no_csv:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            out_df.to_csv(out_csv, index=False)
            n_loc = int(out_df["country"].nunique()) if not out_df.empty else 0
            print(
                f"Through {STOP_DATE.date()} — {len(cols)} columns, {n_loc} locations with ≥1 wave, "
                f"{len(out_df)} wave rows → {out_csv}"
            )
        else:
            print("(CSV skipped: --no-csv)")
        return

    country = args.country.strip()
    out_csv = args.output if args.output is not None else default_output_csv(country)
    out_png = args.plot_output if args.plot_output is not None else default_plot_png(country)

    if country not in df.columns:
        raise SystemExit(f"No column {country!r} in {path}. Check spelling (OWID names, e.g. Czechia, Israel).")

    rows, idx, cum, plot_segments = country_wave_analysis(df, country)

    print(f"{country} — OWID cumulative COVID deaths per million")
    print(
        f"Through {STOP_DATE.date()} inclusive — daily ends {df['date'].max().date()}, "
        f"weekly ends ~{idx[-1].date()}"
    )
    print(
        f"g = {SMOOTH_WEEKS}w MA of weekly new deaths. Activity: g>{G_ACTIVE}; "
        f"before first such week, prefix islands g>{G_PREFIX} (ends abutting main dropped). "
        f"Recursive split if interior trough <{SPLIT_TROUGH_FRAC:.0%} of piece peak "
        f"(min piece {MIN_PIECE_WEEKS_FOR_SPLIT}w)."
    )
    print(
        f"Per piece: longest subinterval with R²≥{R2_MIN} (else ≥{R2_FALLBACK}), "
        f"length≥{MIN_WAVE_WEEKS}w (or ≥{MIN_WAVE_WEEKS_SHORT}w if none; "
        f"prefix pieces ≥{MIN_WAVE_WEEKS_PREFIX}w at R²≥{R2_FALLBACK}); "
        f"then same in left/right gaps (multiple waves).\n"
    )

    for r in rows:
        k = int(r["wave"])
        print(f"Wave {k}  ({r['fit_tier']})")
        print(
            f"  week ending: {r['week_start']} .. {r['week_end']}  "
            f"({r['n_weekly_observations']} weekly observations = {r['n_week_intervals']} week-intervals, "
            f"{r['span_calendar_days']} calendar days)"
        )
        print(f"  R² (cumulative vs week): {float(r['r2_fit']):.5f}")
        print(f"  OLS slope (deaths/million/week): {float(r['slope_ols_deaths_per_million_per_week']):.4f}")
        print(f"  Secant Δcum/(n-1): {float(r['slope_secant_delta_over_n_minus_1']):.4f}")
        print(f"  Δ cumulative: {float(r['delta_cumulative']):.2f}")
        print()

    out_df = pd.DataFrame(rows)
    print("— Summary (slopes) —")
    if out_df.empty:
        print("  (no waves)")
    else:
        for _, r in out_df.iterrows():
            print(
                f"  Wave {int(r['wave'])}: OLS slope = {r['slope_ols_deaths_per_million_per_week']:.4f} "
                f"deaths/million/week  |  R² = {r['r2_fit']:.5f}  |  "
                f"{r['week_start']} .. {r['week_end']}"
            )
        print()

    if not args.no_csv and not out_df.empty:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")
    elif args.no_csv:
        print("(CSV skipped: --no-csv)")

    if not args.no_plot and plot_segments:
        write_wave_fit_png(out_png, country, STOP_DATE, idx, cum, plot_segments)
        print(f"Wrote {out_png}")
    elif args.no_plot:
        print("(PNG skipped: --no-plot)")


if __name__ == "__main__":
    main()
