"""Expected vaccinated deaths under a fixed VE_death on CFR, using observed case rates."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_expected_vs_observed(
    weekly: pd.DataFrame,
    *,
    ve_death: float,
    cfr_reference: str,
    cfr_column: str = "cfr_covid",
    vaccinated_cohort: str = "dose2",
    reference_cohort: str = "dose0",
    age_bin: str = "all",
    baseline_iso_weeks: set[str] | None = None,
) -> pd.DataFrame:
    """
    expected_weekly_deaths = pop_vax * case_rate_vax * (1 - VE) * CFR_unvax_ref

    cfr_reference:
      - same_week: use reference cohort CFR that week (``cfr_column``, e.g. cfr_covid)
      - baseline_mean: use mean reference CFR over baseline_iso_weeks (or all ref rows)
    """
    wk = weekly[
        (weekly["age_bin"] == age_bin)
        & (weekly["cohort"].isin([reference_cohort, vaccinated_cohort]))
    ].copy()
    if wk.empty:
        return pd.DataFrame()

    ref = wk[wk["cohort"] == reference_cohort].set_index("iso_week")
    vax = wk[wk["cohort"] == vaccinated_cohort].copy()

    if cfr_reference == "baseline_mean":
        if baseline_iso_weeks:
            vals = ref.loc[ref.index.isin(baseline_iso_weeks), cfr_column].values
        else:
            vals = ref[cfr_column].values
        ref_cfr_mean = float(np.nanmean(vals))
        ref_cfr_series = {iso: ref_cfr_mean for iso in ref.index}
    else:
        ref_cfr_series = ref[cfr_column].to_dict()

    rows = []
    cum_obs = 0.0
    cum_exp = 0.0
    for _, row in vax.sort_values("iso_week").iterrows():
        iso = row["iso_week"]
        pop = row["population_at_risk"]
        cr = row["case_rate"]
        cfr_u = ref_cfr_series.get(iso, np.nan)
        if cfr_reference == "same_week" and (iso not in ref.index):
            cfr_u = np.nan
        if pd.isna(cfr_u) or pd.isna(cr) or pd.isna(pop):
            exp = np.nan
        else:
            exp = float(pop) * float(cr) * (1.0 - ve_death) * float(cfr_u)
        obs = float(row["deaths_covid"]) if pd.notna(row.get("deaths_covid")) else np.nan
        if pd.isna(obs):
            obs = 0.0
        cum_obs += obs
        if not pd.isna(exp):
            cum_exp += exp
        rows.append(
            {
                "iso_week": iso,
                "age_bin": age_bin,
                "population_at_risk_vax": pop,
                "case_rate_vax": cr,
                "cfr_ref_unvax": cfr_u,
                "expected_deaths_covid_ve": exp,
                "observed_deaths_covid": obs,
                "cumulative_expected_deaths": cum_exp,
                "cumulative_observed_deaths": cum_obs,
                "residual_obs_minus_expected": (obs - exp) if not pd.isna(exp) else np.nan,
            }
        )
    return pd.DataFrame(rows)
