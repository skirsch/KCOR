#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MONTHLY_FILE = DATA_DIR / "UCOD_5yr_bands.xls"
YEARLY_FILE = DATA_DIR / "UCOD_5yr_by_year_with_rates.xls"


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str)


def _normalize_month_code(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Month Code"] = out["Month Code"].astype(str).str.strip()
    out["date"] = pd.to_datetime(out["Month Code"] + "/01", format="%Y/%m/%d", errors="coerce")
    out["Deaths"] = pd.to_numeric(out["Deaths"], errors="coerce")
    return out


def _normalize_yearly(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["Deaths"] = pd.to_numeric(out["Deaths"], errors="coerce")
    out["Population"] = pd.to_numeric(out["Population"], errors="coerce")
    out["Crude Rate"] = pd.to_numeric(out["Crude Rate"], errors="coerce")
    return out


def _broad_age_group(label: str) -> str:
    s = str(label).strip()
    if s in {"< 1 year", "1-4 years", "5-9 years", "10-14 years", "15-19 years", "20-24 years", "25-29 years", "30-34 years", "35-39 years"}:
        return "0-39"
    if s in {"40-44 years", "45-49 years", "50-54 years", "55-59 years"}:
        return "40-59"
    if s in {"60-64 years", "65-69 years", "70-74 years", "75-79 years"}:
        return "60-79"
    if s in {"80-84 years", "85-89 years", "90-94 years", "95-99 years", "100 years and over"}:
        return "80+"
    return "other"


def build_monthly_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    df = monthly.copy()
    df = df[df["Sex"].isin(["Female", "Male"])].copy()
    df["broad_age_group"] = df["Five-Year Age Groups"].map(_broad_age_group)

    all_age = (
        df.groupby(["date", "Sex"], as_index=False)["Deaths"]
        .sum()
        .assign(broad_age_group="all")
    )
    broad = (
        df[df["broad_age_group"] != "other"]
        .groupby(["date", "Sex", "broad_age_group"], as_index=False)["Deaths"]
        .sum()
    )
    out = pd.concat([all_age, broad], ignore_index=True)
    combined = (
        out.groupby(["date", "broad_age_group"], as_index=False)["Deaths"]
        .sum()
        .assign(Sex="Combined")
    )
    out = pd.concat([out, combined], ignore_index=True)
    out = out.sort_values(["broad_age_group", "Sex", "date"], kind="mergesort")
    return out


def build_yearly_rate_summary(yearly: pd.DataFrame) -> pd.DataFrame:
    df = yearly.copy()
    df = df[df["Sex"].isin(["Female", "Male"])].copy()
    df["broad_age_group"] = df["Five-Year Age Groups"].map(_broad_age_group)
    df = df[df["broad_age_group"] != "other"].copy()

    grouped = (
        df.groupby(["Year", "Sex", "broad_age_group"], as_index=False)[["Deaths", "Population"]]
        .sum()
    )
    grouped["Crude_Rate_per_100k"] = (grouped["Deaths"] / grouped["Population"]) * 100000.0

    combined = (
        grouped.groupby(["Year", "broad_age_group"], as_index=False)[["Deaths", "Population"]]
        .sum()
        .assign(Sex="Combined")
    )
    combined["Crude_Rate_per_100k"] = (combined["Deaths"] / combined["Population"]) * 100000.0

    all_age = (
        df.groupby(["Year", "Sex"], as_index=False)[["Deaths", "Population"]]
        .sum()
        .assign(broad_age_group="all")
    )
    all_age["Crude_Rate_per_100k"] = (all_age["Deaths"] / all_age["Population"]) * 100000.0

    all_age_combined = (
        all_age.groupby(["Year", "broad_age_group"], as_index=False)[["Deaths", "Population"]]
        .sum()
        .assign(Sex="Combined")
    )
    all_age_combined["Crude_Rate_per_100k"] = (all_age_combined["Deaths"] / all_age_combined["Population"]) * 100000.0

    out = pd.concat([grouped, combined, all_age, all_age_combined], ignore_index=True)
    out = out.sort_values(["broad_age_group", "Sex", "Year"], kind="mergesort")
    return out


def build_female_male_ratio_summary(yearly_rates: pd.DataFrame) -> pd.DataFrame:
    df = yearly_rates[yearly_rates["Sex"].isin(["Female", "Male"])].copy()
    pivot = df.pivot_table(
        index=["Year", "broad_age_group"],
        columns="Sex",
        values="Crude_Rate_per_100k",
        aggfunc="first",
    ).reset_index()
    pivot["female_to_male_rate_ratio"] = pivot["Female"] / pivot["Male"]
    pivot["female_minus_male_rate_per_100k"] = pivot["Female"] - pivot["Male"]
    return pivot


def build_year_over_year_divergence_summary(yearly_rates: pd.DataFrame) -> pd.DataFrame:
    df = yearly_rates[yearly_rates["Sex"] == "Combined"].copy()
    keep_years = {2019, 2020, 2021, 2022, 2023, 2024}
    df = df[df["Year"].isin(keep_years)].copy()
    rows: list[dict[str, object]] = []
    for age_group in ["all", "0-39", "40-59", "60-79", "80+"]:
        sub = df[df["broad_age_group"] == age_group].set_index("Year").sort_index()
        if sub.empty:
            continue
        for target_year, base_year in [
            (2020, 2019),
            (2021, 2019),
            (2022, 2019),
            (2023, 2019),
            (2024, 2019),
            (2021, 2020),
            (2022, 2020),
            (2023, 2020),
            (2024, 2020),
        ]:
            if target_year not in sub.index or base_year not in sub.index:
                continue
            target = float(sub.loc[target_year, "Crude_Rate_per_100k"])
            base = float(sub.loc[base_year, "Crude_Rate_per_100k"])
            rows.append(
                {
                    "broad_age_group": age_group,
                    "target_year": target_year,
                    "base_year": base_year,
                    "target_rate_per_100k": target,
                    "base_rate_per_100k": base,
                    "absolute_change_per_100k": target - base,
                    "pct_change": ((target / base) - 1.0) if base > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_monthly_deaths(monthly_summary: pd.DataFrame) -> None:
    for age_group in ["all", "0-39", "40-59", "60-79", "80+"]:
        sub = monthly_summary[monthly_summary["broad_age_group"] == age_group].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(12, 5))
        for sex, color in [("Female", "tab:red"), ("Male", "tab:blue"), ("Combined", "black")]:
            s = sub[sub["Sex"] == sex].sort_values("date")
            ax.plot(s["date"], s["Deaths"], label=sex, lw=2, color=color)
        ax.axvline(pd.Timestamp("2020-03-01"), color="gray", linestyle="--", lw=1)
        ax.axvline(pd.Timestamp("2021-01-01"), color="purple", linestyle="--", lw=1)
        ax.set_title(f"US UCOD monthly all-cause deaths by sex - age {age_group}")
        ax.set_ylabel("Deaths")
        ax.set_xlabel("Calendar month")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ucod_monthly_deaths_by_sex_age_{age_group}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_yearly_rates(yearly_rates: pd.DataFrame) -> None:
    for age_group in ["all", "0-39", "40-59", "60-79", "80+"]:
        sub = yearly_rates[yearly_rates["broad_age_group"] == age_group].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for sex, color in [("Female", "tab:red"), ("Male", "tab:blue"), ("Combined", "black")]:
            s = sub[sub["Sex"] == sex].sort_values("Year")
            ax.plot(s["Year"], s["Crude_Rate_per_100k"], marker="o", label=sex, lw=2, color=color)
        ax.axvline(2020, color="gray", linestyle="--", lw=1)
        ax.axvline(2021, color="purple", linestyle="--", lw=1)
        ax.set_title(f"US UCOD yearly crude death rate by sex - age {age_group}")
        ax.set_ylabel("Deaths per 100,000")
        ax.set_xlabel("Year")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ucod_yearly_crude_rate_by_sex_age_{age_group}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_female_male_ratio(ratio_summary: pd.DataFrame) -> None:
    for age_group in ["all", "0-39", "40-59", "60-79", "80+"]:
        sub = ratio_summary[ratio_summary["broad_age_group"] == age_group].copy()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4.5))
        s = sub.sort_values("Year")
        ax.plot(s["Year"], s["female_to_male_rate_ratio"], marker="o", lw=2, color="darkgreen")
        ax.axhline(1.0, color="gray", lw=1)
        ax.axvline(2020, color="gray", linestyle="--", lw=1)
        ax.axvline(2021, color="purple", linestyle="--", lw=1)
        ax.set_title(f"US female/male crude death-rate ratio - age {age_group}")
        ax.set_ylabel("Female rate / Male rate")
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ucod_female_male_rate_ratio_age_{age_group}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_combined_yearly_rates(yearly_rates: pd.DataFrame) -> None:
    sub = yearly_rates[yearly_rates["Sex"] == "Combined"].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {
        "0-39": "tab:green",
        "40-59": "tab:orange",
        "60-79": "tab:blue",
        "80+": "tab:red",
        "all": "black",
    }
    for age_group in ["0-39", "40-59", "60-79", "80+", "all"]:
        s = sub[sub["broad_age_group"] == age_group].sort_values("Year")
        if s.empty:
            continue
        ax.plot(
            s["Year"],
            s["Crude_Rate_per_100k"],
            marker="o",
            lw=2,
            color=colors[age_group],
            label=age_group,
        )
    ax.axvline(2020, color="gray", linestyle="--", lw=1)
    ax.axvline(2021, color="purple", linestyle="--", lw=1)
    ax.set_title("US UCOD combined crude death rate by broad age group")
    ax.set_ylabel("Deaths per 100,000")
    ax.set_xlabel("Year")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ucod_yearly_crude_rate_combined_broad_age.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    monthly = _normalize_month_code(_read_tsv(MONTHLY_FILE))
    yearly = _normalize_yearly(_read_tsv(YEARLY_FILE))

    monthly_summary = build_monthly_summary(monthly)
    yearly_rates = build_yearly_rate_summary(yearly)
    ratio_summary = build_female_male_ratio_summary(yearly_rates)
    divergence_summary = build_year_over_year_divergence_summary(yearly_rates)

    monthly_summary.to_csv(OUT_DIR / "ucod_monthly_deaths_by_sex_broad_age.csv", index=False)
    yearly_rates.to_csv(OUT_DIR / "ucod_yearly_crude_rates_by_sex_broad_age.csv", index=False)
    ratio_summary.to_csv(OUT_DIR / "ucod_yearly_female_male_rate_ratio.csv", index=False)
    divergence_summary.to_csv(OUT_DIR / "ucod_year_over_year_divergence_summary.csv", index=False)

    plot_monthly_deaths(monthly_summary)
    plot_yearly_rates(yearly_rates)
    plot_female_male_ratio(ratio_summary)
    plot_combined_yearly_rates(yearly_rates)

    print("Wrote UCOD summaries and plots to", OUT_DIR)


if __name__ == "__main__":
    main()
