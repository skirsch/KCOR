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

ICD_FILE = DATA_DIR / "UCOD_by_year_ICD10.xls"


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    out["Deaths"] = pd.to_numeric(out["Deaths"], errors="coerce")
    out["Population"] = pd.to_numeric(out["Population"], errors="coerce")
    out["Crude Rate"] = pd.to_numeric(out["Crude Rate"], errors="coerce")
    return out


def _broad_age_group(label: str) -> str:
    s = str(label).strip()
    if s in {"< 1 year", "1-4 years", "5-14 years", "15-24 years", "25-34 years"}:
        return "0-34"
    if s in {"35-44 years", "45-54 years"}:
        return "35-54"
    if s in {"55-64 years", "65-74 years"}:
        return "55-74"
    if s in {"75-84 years", "85 years and over"}:
        return "75+"
    return "other"


def classify_cause(label: str) -> str:
    s = str(label).strip()
    if "Drug-induced causes" in s:
        return "Drug-induced"
    if "Alcohol-induced causes" in s:
        return "Alcohol-induced"
    if "COVID-19" in s:
        return "COVID-19"
    if "#Diseases of heart" in s or "Ischemic heart diseases" in s or "Other heart diseases" in s:
        return "Heart disease"
    if "Cerebrovascular diseases" in s:
        return "Cerebrovascular"
    if "Major cardiovascular diseases" in s:
        return "Cardiovascular (other/aggregate)"
    if "#Diabetes mellitus" in s:
        return "Diabetes"
    if "#Chronic lower respiratory diseases" in s or "#Influenza and pneumonia" in s or "#Other diseases of respiratory system" in s:
        return "Respiratory"
    if "#Malignant neoplasms" in s:
        return "Cancer"
    if "#Intentional self-harm" in s:
        return "Suicide"
    if "#Assault (homicide)" in s:
        return "Homicide"
    if "Accidents" in s or "transport accidents" in s or "Falls" in s or "Accidental poisoning" in s:
        return "Accidents / injuries"
    if "#Alzheimer" in s or "#Parkinson" in s:
        return "Neurodegenerative"
    if "#Nephritis" in s:
        return "Renal"
    if "#Septicemia" in s:
        return "Septicemia"
    if "#Symptoms, signs and abnormal clinical and laboratory findings" in s:
        return "Ill-defined / symptoms"
    return "Other"


def build_cause_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["broad_age_group"] = out["Ten-Year Age Groups"].map(_broad_age_group)
    out = out[out["broad_age_group"] != "other"].copy()
    out["cause_group"] = out["ICD-10 113 Cause List"].map(classify_cause)
    grouped = (
        out.groupby(["Year", "broad_age_group", "cause_group"], as_index=False)[["Deaths", "Population"]]
        .sum()
    )
    grouped["Crude_Rate_per_100k"] = (grouped["Deaths"] / grouped["Population"]) * 100000.0
    return grouped


def build_nonelderly_divergence(cause_summary: pd.DataFrame) -> pd.DataFrame:
    df = cause_summary[cause_summary["broad_age_group"].isin(["0-34", "35-54", "55-74"])].copy()
    rows: list[dict[str, object]] = []
    for age_group in ["0-34", "35-54", "55-74"]:
        sub = df[df["broad_age_group"] == age_group].copy()
        for cause_group in sorted(sub["cause_group"].unique()):
            s = sub[sub["cause_group"] == cause_group].set_index("Year").sort_index()
            if 2019 not in s.index or 2021 not in s.index:
                continue
            rate_2019 = float(s.loc[2019, "Crude_Rate_per_100k"])
            rate_2020 = float(s.loc[2020, "Crude_Rate_per_100k"]) if 2020 in s.index else np.nan
            rate_2021 = float(s.loc[2021, "Crude_Rate_per_100k"])
            rows.append(
                {
                    "broad_age_group": age_group,
                    "cause_group": cause_group,
                    "rate_2019_per_100k": rate_2019,
                    "rate_2020_per_100k": rate_2020,
                    "rate_2021_per_100k": rate_2021,
                    "abs_change_2021_vs_2019": rate_2021 - rate_2019,
                    "pct_change_2021_vs_2019": ((rate_2021 / rate_2019) - 1.0) if rate_2019 > 0 else np.nan,
                    "abs_change_2021_vs_2020": rate_2021 - rate_2020 if np.isfinite(rate_2020) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out = out.sort_values(["broad_age_group", "abs_change_2021_vs_2019"], ascending=[True, False], kind="mergesort")
    return out


def build_excluding_selected_causes(cause_summary: pd.DataFrame) -> pd.DataFrame:
    excluded = {"Drug-induced", "Alcohol-induced", "Suicide", "Homicide", "Accidents / injuries"}
    all_grouped = (
        cause_summary.groupby(["Year", "broad_age_group"], as_index=False)["Crude_Rate_per_100k"]
        .sum()
        .rename(columns={"Crude_Rate_per_100k": "all_cause_component_sum_per_100k"})
    )
    kept = (
        cause_summary[~cause_summary["cause_group"].isin(excluded)]
        .groupby(["Year", "broad_age_group"], as_index=False)["Crude_Rate_per_100k"]
        .sum()
        .rename(columns={"Crude_Rate_per_100k": "excluding_external_drug_alcohol_per_100k"})
    )
    out = all_grouped.merge(kept, on=["Year", "broad_age_group"], how="left")
    return out


def build_kidney_category_summary(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["broad_age_group"] = df["Ten-Year Age Groups"].map(_broad_age_group)
    df = df[df["broad_age_group"] != "other"].copy()
    kidney_mask = df["ICD-10 113 Cause List"].astype(str).str.contains(
        "kidney|renal|neph", case=False, na=False
    )
    sub = df[kidney_mask].copy()
    grouped = (
        sub.groupby(["Year", "broad_age_group", "ICD-10 113 Cause List"], as_index=False)[["Deaths", "Population"]]
        .sum()
    )
    grouped["Crude_Rate_per_100k"] = (grouped["Deaths"] / grouped["Population"]) * 100000.0
    return grouped.sort_values(
        ["ICD-10 113 Cause List", "broad_age_group", "Year"], kind="mergesort"
    )


def build_focus_category_summary(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["broad_age_group"] = df["Ten-Year Age Groups"].map(_broad_age_group)
    df = df[df["broad_age_group"] != "other"].copy()
    focus_mask = df["ICD-10 113 Cause List"].astype(str).str.contains(
        "heart|cardiovascular|cerebrovascular|diabetes", case=False, na=False
    )
    sub = df[focus_mask].copy()
    grouped = (
        sub.groupby(["Year", "broad_age_group", "ICD-10 113 Cause List"], as_index=False)[["Deaths", "Population"]]
        .sum()
    )
    grouped["Crude_Rate_per_100k"] = (grouped["Deaths"] / grouped["Population"]) * 100000.0
    return grouped.sort_values(
        ["ICD-10 113 Cause List", "broad_age_group", "Year"], kind="mergesort"
    )


def plot_top_cause_changes(divergence: pd.DataFrame) -> None:
    for age_group in ["0-34", "35-54", "55-74"]:
        sub = divergence[divergence["broad_age_group"] == age_group].copy().head(10)
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(sub["cause_group"], sub["abs_change_2021_vs_2019"], color="tab:purple")
        ax.set_title(f"Top cause-group contributors to 2021 vs 2019 increase - age {age_group}")
        ax.set_xlabel("Absolute crude-rate change per 100,000")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ucod_top_cause_changes_2021_vs_2019_age_{age_group}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_excluding_selected_causes(excluded_summary: pd.DataFrame) -> None:
    for age_group in ["0-34", "35-54", "55-74", "75+"]:
        sub = excluded_summary[excluded_summary["broad_age_group"] == age_group].copy().sort_values("Year")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(sub["Year"], sub["all_cause_component_sum_per_100k"], marker="o", lw=2, label="all included cause groups")
        ax.plot(
            sub["Year"],
            sub["excluding_external_drug_alcohol_per_100k"],
            marker="o",
            lw=2,
            label="excluding drug/alcohol/external causes",
        )
        ax.axvline(2020, color="gray", linestyle="--", lw=1)
        ax.axvline(2021, color="purple", linestyle="--", lw=1)
        ax.set_title(f"UCOD crude-rate trend with selected causes removed - age {age_group}")
        ax.set_ylabel("Deaths per 100,000")
        ax.set_xlabel("Year")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ucod_excluding_selected_causes_age_{age_group}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_kidney_categories(kidney_summary: pd.DataFrame) -> None:
    if kidney_summary.empty:
        return
    for cause_label in sorted(kidney_summary["ICD-10 113 Cause List"].unique()):
        sub = kidney_summary[kidney_summary["ICD-10 113 Cause List"] == cause_label].copy()
        fig, ax = plt.subplots(figsize=(11, 5.5))
        colors = {
            "0-34": "tab:green",
            "35-54": "tab:orange",
            "55-74": "tab:blue",
            "75+": "tab:red",
        }
        for age_group in ["0-34", "35-54", "55-74", "75+"]:
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
        ax.set_title(cause_label)
        ax.set_ylabel("Deaths per 100,000")
        ax.set_xlabel("Year")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = (
            str(cause_label)
            .replace("#", "")
            .replace("/", "_")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
            .replace("-", "_")
        )
        fig.savefig(OUT_DIR / f"ucod_kidney_{safe[:80]}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_focus_categories(focus_summary: pd.DataFrame) -> None:
    if focus_summary.empty:
        return
    for cause_label in sorted(focus_summary["ICD-10 113 Cause List"].unique()):
        sub = focus_summary[focus_summary["ICD-10 113 Cause List"] == cause_label].copy()
        fig, ax = plt.subplots(figsize=(11, 5.5))
        colors = {
            "0-34": "tab:green",
            "35-54": "tab:orange",
            "55-74": "tab:blue",
            "75+": "tab:red",
        }
        for age_group in ["0-34", "35-54", "55-74", "75+"]:
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
        ax.set_title(cause_label)
        ax.set_ylabel("Deaths per 100,000")
        ax.set_xlabel("Year")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = (
            str(cause_label)
            .replace("#", "")
            .replace("/", "_")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace(" ", "_")
            .replace("-", "_")
        )
        fig.savefig(OUT_DIR / f"ucod_focus_{safe[:80]}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    raw = _normalize(_read_tsv(ICD_FILE))
    cause_summary = build_cause_summary(raw)
    divergence = build_nonelderly_divergence(cause_summary)
    excluded_summary = build_excluding_selected_causes(cause_summary)
    kidney_summary = build_kidney_category_summary(raw)
    focus_summary = build_focus_category_summary(raw)

    cause_summary.to_csv(OUT_DIR / "ucod_icd_cause_summary.csv", index=False)
    divergence.to_csv(OUT_DIR / "ucod_icd_nonelderly_divergence_2021_vs_2019.csv", index=False)
    excluded_summary.to_csv(OUT_DIR / "ucod_icd_excluding_selected_causes_summary.csv", index=False)
    kidney_summary.to_csv(OUT_DIR / "ucod_kidney_categories_by_age_year.csv", index=False)
    focus_summary.to_csv(OUT_DIR / "ucod_focus_categories_by_age_year.csv", index=False)

    plot_top_cause_changes(divergence)
    plot_excluding_selected_causes(excluded_summary)
    plot_kidney_categories(kidney_summary)
    plot_focus_categories(focus_summary)

    print("Wrote UCOD ICD summaries and plots to", OUT_DIR)


if __name__ == "__main__":
    main()
