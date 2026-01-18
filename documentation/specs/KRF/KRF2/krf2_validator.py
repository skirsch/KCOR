# validate_kcor_population.py
# Pandas validator for the 2-table KCOR population schema

from __future__ import annotations

import sys
import pandas as pd


REQUIRED_STATE_COLS = [
    "cal_week",
    "age_band_5yr",
    "sex",
    "dose_state",
    "n_start",
    "d_death",
]
OPTIONAL_STATE_COLS = ["c_other"]

REQUIRED_TRANS_COLS = [
    "cal_week",
    "age_band_5yr",
    "sex",
    "dose_from",
    "dose_to",
    "x_count",
]

ALLOWED_SEX = {"M", "F", "U"}


def _fail(msg: str) -> None:
    raise ValueError(msg)


def validate(
    state: pd.DataFrame,
    trans: pd.DataFrame,
    *,
    allow_dose_decrease: bool = True,
    warn_on_dose_decrease: bool = True,
    strict_keys: bool = True,
) -> dict:
    """
    Validate schema + invariants.

    Returns summary dict with counts and warnings.
    Raises ValueError on hard failures.
    """

    warnings: list[str] = []

    # ---- Column presence ----
    missing_state = [c for c in REQUIRED_STATE_COLS if c not in state.columns]
    if missing_state:
        _fail(f"State table missing required columns: {missing_state}")

    missing_trans = [c for c in REQUIRED_TRANS_COLS if c not in trans.columns]
    if missing_trans:
        _fail(f"Transition table missing required columns: {missing_trans}")

    if "c_other" not in state.columns:
        state = state.copy()
        state["c_other"] = 0

    # ---- Basic type coercions (best effort) ----
    for c in ["n_start", "d_death", "c_other", "dose_state"]:
        state[c] = pd.to_numeric(state[c], errors="coerce")
    for c in ["x_count", "dose_from", "dose_to"]:
        trans[c] = pd.to_numeric(trans[c], errors="coerce")

    # ---- Null checks ----
    if state[["cal_week", "age_band_5yr", "sex"]].isna().any().any():
        _fail("State table has NA in one or more key columns (cal_week, age_band_5yr, sex).")
    if trans[["cal_week", "age_band_5yr", "sex"]].isna().any().any():
        _fail("Transition table has NA in one or more key columns (cal_week, age_band_5yr, sex).")

    if state[["n_start", "d_death", "c_other", "dose_state"]].isna().any().any():
        _fail("State table has NA in numeric columns (n_start, d_death, c_other, dose_state).")
    if trans[["x_count", "dose_from", "dose_to"]].isna().any().any():
        _fail("Transition table has NA in numeric columns (x_count, dose_from, dose_to).")

    # ---- Sex domain ----
    bad_sex_state = sorted(set(state["sex"].unique()) - ALLOWED_SEX)
    bad_sex_trans = sorted(set(trans["sex"].unique()) - ALLOWED_SEX)
    if bad_sex_state:
        _fail(f"State table has invalid sex values: {bad_sex_state}")
    if bad_sex_trans:
        _fail(f"Transition table has invalid sex values: {bad_sex_trans}")

    # ---- Nonnegative constraints ----
    for c in ["n_start", "d_death", "c_other"]:
        if (state[c] < 0).any():
            _fail(f"State table has negative values in {c}.")
    for c in ["x_count"]:
        if (trans[c] < 0).any():
            _fail("Transition table has negative values in x_count.")

    if (state["dose_state"] < 0).any():
        _fail("State table has negative dose_state.")
    if (trans["dose_from"] < 0).any() or (trans["dose_to"] < 0).any():
        _fail("Transition table has negative dose_from/dose_to.")

    # ---- Key uniqueness (optional strict) ----
    if strict_keys:
        dup_state = state.duplicated(["cal_week", "age_band_5yr", "sex", "dose_state"]).any()
        if dup_state:
            _fail("State table primary key is not unique: (cal_week, age_band_5yr, sex, dose_state).")

        dup_trans = trans.duplicated(["cal_week", "age_band_5yr", "sex", "dose_from", "dose_to"]).any()
        if dup_trans:
            _fail("Transition table primary key is not unique: (cal_week, age_band_5yr, sex, dose_from, dose_to).")

    # ---- No self-transition ----
    if (trans["dose_from"] == trans["dose_to"]).any():
        _fail("Transition table contains dose_from == dose_to (self-transition).")

    # ---- Dose decrease policy ----
    dose_decrease_mask = trans["dose_to"] < trans["dose_from"]
    n_decrease = int(dose_decrease_mask.sum())
    if n_decrease > 0:
        msg = f"Found {n_decrease} dose-decrease transitions (dose_to < dose_from)."
        if not allow_dose_decrease:
            _fail(msg)
        if warn_on_dose_decrease:
            warnings.append(msg)

    # ---- Compute transitions_out_total by (week, age, sex, dose_from) ----
    outflow = (
        trans.groupby(["cal_week", "age_band_5yr", "sex", "dose_from"], as_index=False)["x_count"]
        .sum()
        .rename(columns={"dose_from": "dose_state", "x_count": "transitions_out_total"})
    )

    # ---- Merge outflow into state ----
    merged = state.merge(
        outflow,
        on=["cal_week", "age_band_5yr", "sex", "dose_state"],
        how="left",
        validate="one_to_one" if strict_keys else None,
    )
    merged["transitions_out_total"] = merged["transitions_out_total"].fillna(0)

    # ---- Flow <= risk set constraint ----
    lhs = merged["d_death"] + merged["c_other"] + merged["transitions_out_total"]
    viol = lhs > merged["n_start"]
    n_viol = int(viol.sum())
    if n_viol > 0:
        examples = merged.loc[viol, ["cal_week", "age_band_5yr", "sex", "dose_state", "n_start", "d_death", "c_other", "transitions_out_total"]].head(10)
        _fail(
            f"Flow constraint violated in {n_viol} strata: d_death + c_other + transitions_out_total > n_start.\n"
            f"First rows:\n{examples.to_string(index=False)}"
        )

    # ---- Transition outflow must be backed by a state stratum ----
    # If you have transitions from a stratum not present in state, that's a hard schema error.
    state_keys = set(map(tuple, state[["cal_week", "age_band_5yr", "sex", "dose_state"]].itertuples(index=False, name=None)))
    outflow_keys = set(map(tuple, outflow[["cal_week", "age_band_5yr", "sex", "dose_state"]].itertuples(index=False, name=None)))
    missing_state_for_outflow = outflow_keys - state_keys
    if missing_state_for_outflow:
        sample = list(missing_state_for_outflow)[:10]
        _fail(
            f"Transitions originate from strata not found in state table. Sample missing keys:\n{sample}"
        )

    # ---- Summaries ----
    summary = {
        "state_rows": int(len(state)),
        "transition_rows": int(len(trans)),
        "total_n_start": float(state["n_start"].sum()),
        "total_deaths": float(state["d_death"].sum()),
        "total_c_other": float(state["c_other"].sum()),
        "total_transitions": float(trans["x_count"].sum()),
        "dose_decrease_transitions": n_decrease,
        "warnings": warnings,
    }
    return summary


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("Usage: python validate_kcor_population.py kcor_weekly_state.csv kcor_weekly_transitions.csv", file=sys.stderr)
        return 2

    state_path = argv[1]
    trans_path = argv[2]

    state = pd.read_csv(state_path)
    trans = pd.read_csv(trans_path)

    summary = validate(state, trans)
    print("VALIDATION OK")
    for k, v in summary.items():
        if k != "warnings":
            print(f"{k}: {v}")
    if summary["warnings"]:
        print("\nWARNINGS:")
        for w in summary["warnings"]:
            print(f"- {w}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
