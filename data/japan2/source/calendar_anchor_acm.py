#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


DOSE_COLS = [
    "Date_FirstDose",
    "Date_SecondDose",
    "Date_ThirdDose",
    "Date_FourthDose",
    "Date_FifthDose",
    "Date_SixthDose",
]


def parse_yw(value: object) -> pd.Timestamp:
    try:
        year, week = str(value).strip().split("-")
        return pd.Timestamp.fromisocalendar(int(year), int(week), 1)
    except Exception:
        return pd.NaT


def parse_birth_year(value: object) -> float:
    try:
        return float(str(value).split("-")[0])
    except Exception:
        return np.nan


def weeks_until_event_or_censor(group: pd.DataFrame, enroll_date: pd.Timestamp, max_weeks: int) -> pd.Series:
    death_date = group["DateOfDeath"]
    end = death_date.where(death_date.notna(), enroll_date + pd.Timedelta(weeks=max_weeks))
    raw = ((end - enroll_date).dt.days // 7).clip(lower=0, upper=max_weeks)
    return raw.astype(int)


def km_cumulative_hazard(group: pd.DataFrame, enroll_date: pd.Timestamp, max_weeks: int) -> pd.Series:
    n_start = len(group)
    if n_start == 0:
        return pd.Series(dtype=float)
    died_after = group["DateOfDeath"].notna()
    wk = ((group.loc[died_after, "DateOfDeath"] - enroll_date).dt.days // 7)
    wk = wk[(wk >= 1) & (wk <= max_weeks)].astype(int)
    death_counts = np.bincount(wk, minlength=max_weeks + 1)
    n_at_risk = n_start
    cum_h = 0.0
    values: list[float] = []
    for week in range(1, max_weeks + 1):
        d_w = int(death_counts[week])
        if n_at_risk > 0:
            mr = min(d_w / n_at_risk, 1.0 - 1e-10)
            cum_h += -np.log1p(-mr)
        values.append(cum_h)
        n_at_risk -= d_w
    return pd.Series(values, index=range(1, max_weeks + 1))


def classify_dose_count(df: pd.DataFrame, enroll_date: pd.Timestamp) -> pd.Series:
    dose_count = pd.Series(0, index=df.index, dtype=int)
    for col in DOSE_COLS:
        dose_count = dose_count + ((df[col].notna()) & (df[col] <= enroll_date)).astype(int)
    return dose_count


def load_data(csv_path: Path, *, birth_year_min: int, birth_year_max: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, na_values=["", "nan", "-1"])
    for col in DOSE_COLS + ["DateOfDeath"]:
        df[col] = df[col].apply(parse_yw)
    df["BirthYearMin"] = df["YearOfBirth"].apply(parse_birth_year)
    df["birth_band"] = df["YearOfBirth"].astype(str)
    df = df[
        df["BirthYearMin"].notna()
        & (df["BirthYearMin"] >= birth_year_min)
        & (df["BirthYearMin"] <= birth_year_max)
    ].copy()
    return df


def build_anchor_age_dose_tables(
    df: pd.DataFrame,
    *,
    enroll_dates: list[pd.Timestamp],
    max_weeks: int,
    min_group_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    age_bands = sorted(df["birth_band"].dropna().unique(), key=lambda s: parse_birth_year(s))
    for enroll_date in enroll_dates:
        alive = df[df["DateOfDeath"].isna() | (df["DateOfDeath"] > enroll_date)].copy()
        alive["dose_count_at_enroll"] = classify_dose_count(alive, enroll_date)
        for age_band in age_bands:
            age_sub = alive[alive["birth_band"] == age_band].copy()
            if age_sub.empty:
                continue
            for dose_num in sorted(age_sub["dose_count_at_enroll"].unique()):
                group = age_sub[age_sub["dose_count_at_enroll"] == dose_num].copy()
                n_start = len(group)
                if n_start < min_group_size:
                    continue
                weeks_at_risk = weeks_until_event_or_censor(group, enroll_date, max_weeks)
                deaths_52w = int(
                    (
                        group["DateOfDeath"].notna()
                        & (((group["DateOfDeath"] - enroll_date).dt.days // 7) >= 1)
                        & (((group["DateOfDeath"] - enroll_date).dt.days // 7) <= max_weeks)
                    ).sum()
                )
                h = km_cumulative_hazard(group, enroll_date, max_weeks)
                risk_52w = float(1.0 - np.exp(-h.iloc[-1])) if len(h) else np.nan
                summary_rows.append(
                    {
                        "enroll_date": enroll_date.date().isoformat(),
                        "birth_band": age_band,
                        "dose_count_at_enroll": int(dose_num),
                        "n_start": int(n_start),
                        "deaths_52w": deaths_52w,
                        "person_weeks_52w": int(weeks_at_risk.sum()),
                        "death_rate_per_person_week_52w": (
                            float(deaths_52w) / float(weeks_at_risk.sum()) if weeks_at_risk.sum() > 0 else np.nan
                        ),
                        "risk_52w": risk_52w,
                        "cum_hazard_52w": float(h.iloc[-1]) if len(h) else np.nan,
                    }
                )
                for week, value in h.items():
                    curve_rows.append(
                        {
                            "enroll_date": enroll_date.date().isoformat(),
                            "birth_band": age_band,
                            "dose_count_at_enroll": int(dose_num),
                            "week": int(week),
                            "cum_hazard": float(value),
                        }
                    )
    return pd.DataFrame(summary_rows), pd.DataFrame(curve_rows)


def plot_anchor_curves(curves: pd.DataFrame, out_dir: Path) -> None:
    if curves.empty:
        return
    for enroll_date in sorted(curves["enroll_date"].unique()):
        sub = curves[curves["enroll_date"] == enroll_date].copy()
        age_bands = sorted(sub["birth_band"].unique(), key=lambda s: parse_birth_year(s))
        n_panels = len(age_bands)
        ncols = 2
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharex=True, sharey=False)
        axes_arr = np.array(axes).reshape(-1)
        colors = plt.cm.tab10(np.linspace(0, 1, 7))
        for ax, age_band in zip(axes_arr, age_bands):
            age_sub = sub[sub["birth_band"] == age_band].copy()
            for dose_num in sorted(age_sub["dose_count_at_enroll"].unique()):
                s = age_sub[age_sub["dose_count_at_enroll"] == dose_num].sort_values("week")
                color = colors[int(dose_num) % len(colors)]
                ax.plot(s["week"], s["cum_hazard"], label=f"dose {int(dose_num)}", lw=2, color=color)
            ax.set_title(f"Birth band {age_band}")
            ax.set_xlabel("Weeks since enrollment")
            ax.set_ylabel("Cumulative hazard H(t)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        for ax in axes_arr[n_panels:]:
            ax.axis("off")
        fig.suptitle(f"52-week ACM cumulative hazard by dose at enrollment - {enroll_date}")
        fig.tight_layout()
        fig.savefig(out_dir / f"calendar_anchor_hazard_by_dose_age_{enroll_date}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_anchor_risk_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        return
    for enroll_date in sorted(summary["enroll_date"].unique()):
        sub = summary[summary["enroll_date"] == enroll_date].copy()
        age_bands = sorted(sub["birth_band"].unique(), key=lambda s: parse_birth_year(s))
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(age_bands))
        dose_levels = sorted(sub["dose_count_at_enroll"].unique())
        width = 0.8 / max(1, len(dose_levels))
        for i, dose_num in enumerate(dose_levels):
            vals = []
            for age_band in age_bands:
                row = sub[(sub["birth_band"] == age_band) & (sub["dose_count_at_enroll"] == dose_num)]
                vals.append(float(row.iloc[0]["risk_52w"]) if len(row) else np.nan)
            ax.bar(x + (i - (len(dose_levels) - 1) / 2) * width, vals, width=width, label=f"dose {int(dose_num)}")
        ax.set_xticks(x)
        ax.set_xticklabels(age_bands, rotation=30, ha="right")
        ax.set_ylabel("52-week ACM risk")
        ax.set_title(f"52-week ACM risk by dose at enrollment and birth band - {enroll_date}")
        ax.legend(ncol=2, fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"calendar_anchor_52w_risk_by_dose_age_{enroll_date}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Japan2 calendar-anchor 52-week ACM by dose and age band")
    ap.add_argument("--config", default="data/japan2/japan2.yaml")
    ap.add_argument("--input", default="data/japan2/records.csv")
    ap.add_argument("--output-dir", default="data/japan2")
    ap.add_argument("--birth-year-min", type=int, default=1930)
    ap.add_argument("--birth-year-max", type=int, default=1960)
    ap.add_argument("--max-weeks", type=int, default=52)
    ap.add_argument("--min-group-size", type=int, default=200)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    enroll_dates = [pd.Timestamp.fromisocalendar(int(x.split("-")[0]), int(x.split("-")[1]), 1) for x in cfg["enrollmentDates"]]
    csv_path = Path(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Japan2 records...")
    df = load_data(csv_path, birth_year_min=args.birth_year_min, birth_year_max=args.birth_year_max)
    print(f"Rows in cohort birth window: {len(df):,}")

    summary, curves = build_anchor_age_dose_tables(
        df,
        enroll_dates=enroll_dates,
        max_weeks=args.max_weeks,
        min_group_size=args.min_group_size,
    )
    summary.to_csv(out_dir / "calendar_anchor_52w_acm_by_dose_age.csv", index=False)
    curves.to_csv(out_dir / "calendar_anchor_hazard_curves_by_dose_age.csv", index=False)
    plot_anchor_curves(curves, out_dir)
    plot_anchor_risk_summary(summary, out_dir)
    print("Wrote calendar-anchor ACM summaries and plots.")


if __name__ == "__main__":
    main()
