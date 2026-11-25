"""Paired sensitivity checks on simulated and saved Czech cohort counts."""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json

import numpy as np
import pandas as pd

from check import HERE, K, estimate, exact_weekly_variant, population
from production_adapter import FLAGS

ROOT = HERE.parents[2]
WORKBOOK = ROOT / "data/Czech/KCOR_CMR.Czech.xlsx"
SHEETS = ("2021_24", "2022_06", "2022_47")


def scenarios(original, exact):
    rng = np.random.default_rng(602641)
    for n in (400_000, 4_000_000):
        for gap_multiplier in (1.0, 1.03, 1.30):
            for k, theta in ((.0003, [2., .5]), (.001, [4., .5]), (.003, [4., 1.])):
                windows = ((6, 25), (41, 55))
                scenario = dict(n=n, k=k, theta=theta, windows=windows,
                                effect=gap_multiplier, effect_window=(26, 40))
                # Use the same gap multiplier in both arms and preserve their different frailty.
                probabilities = np.asarray([
                    population(dict(scenario, theta=[theta[d], theta[d]]))[0][1]
                    for d in (0, 1)
                ])
                for draw in range(40):
                    counts = rng.multinomial(n, (probabilities * .5).ravel()).reshape(probabilities.shape)
                    try:
                        K.fit_theta0_gompertz = original
                        first = estimate(counts, windows=windows)
                        K.fit_theta0_gompertz = exact
                        second = estimate(counts, windows=windows)
                    finally:
                        K.fit_theta0_gompertz = original
                    yield dict(n=n, k=k, gap_multiplier=gap_multiplier, draw=draw,
                               first=first["accepted"], exact=second["accepted"],
                               first_kcor=first["point"], exact_kcor=second["point"])


def fit_cohort(group, fitter, config, guards):
    weeks = np.arange(len(group), dtype=float)
    date = group.index
    iso = date.isocalendar()
    quiet, _, _ = K._build_theta_quiet_mask(iso.year.to_numpy() * 100 + iso.week.to_numpy(), weeks)
    skip = K.DYNAMIC_HVE_SKIP_WEEKS
    if np.count_nonzero(np.diff(np.r_[False, quiet & (weeks >= skip)].astype(int)) == 1) < 2:
        return None
    alive = group.Alive.to_numpy(dtype=float).clip(min=0)
    dead = group.Dead.to_numpy(dtype=float).clip(min=0)
    mr = np.divide(dead, alive + K.EPS, out=np.full(len(group), np.nan), where=alive > 0)
    hazard = K.hazard_from_mr(np.clip(mr, 0, .999))
    hazard = np.where(weeks >= skip, hazard, 0.)
    (_, theta), diag = fitter(hazard, weeks - skip, quiet, config["k_anchor_weeks"],
                              config["gamma_per_week"], deaths_arr=dead)
    theta, degenerate, _ = K._apply_theta_degenerate_guard(theta, guards)
    flags = [flag for flag in FLAGS if diag.get(flag, False)]
    accepted = bool(diag.get("success") and np.isfinite(theta) and not degenerate and not flags)
    cumulative = K.invert_gamma_frailty(np.cumsum(np.nan_to_num(hazard)), theta) if accepted else None
    return dict(accepted=accepted, status=diag.get("status"), flags=flags,
                theta=float(theta) if np.isfinite(theta) else None,
                curve=pd.Series(cumulative, index=date) if accepted else None)


def replay(original, exact):
    K._DATASET_THETA_ESTIMATION_WINDOWS_CACHE = None
    config = K.get_gompertz_theta_fit_config()
    guards = K.get_theta_degenerate_fit_config()
    cohort_rows, pair_rows = [], []
    with pd.ExcelFile(WORKBOOK, engine="openpyxl") as book:
        for sheet in SHEETS:
            data = book.parse(sheet, usecols=["YearOfBirth", "Dose", "DateDied", "Alive", "Dead"])
            data.DateDied = pd.to_datetime(data.DateDied)
            start = pd.Timestamp.fromisocalendar(*map(int, sheet.split("_")), 1)
            pairs = K.get_dose_pairs(sheet)
            doses = {d for pair in pairs for d in pair}
            data = data.loc[(data.DateDied >= start) & (data.YearOfBirth >= 0) & data.Dose.isin(doses)].copy()
            data.YearOfBirth = data.YearOfBirth.astype(int) // K.AGE_RANGE * K.AGE_RANGE
            data = data.groupby(["YearOfBirth", "Dose", "DateDied"])[["Alive", "Dead"]].sum()
            fits = {}
            for (yob, dose), group in data.groupby(level=[0, 1]):
                group = group.droplevel([0, 1])
                first = fit_cohort(group, original, config, guards)
                second = fit_cohort(group, exact, config, guards)
                if first is None:
                    continue
                fits[int(yob), int(dose)] = (first, second)
                row = dict(sheet=sheet, yob=int(yob), dose=int(dose),
                           first=first["accepted"], exact=second["accepted"])
                for label, fit in (("first", first), ("exact", second)):
                    row[label + "_fit"] = {k: v for k, v in fit.items() if k != "curve"}
                cohort_rows.append(row)
            for yob in sorted({key[0] for key in fits}):
                for num, den in pairs:
                    if (yob, num) not in fits or (yob, den) not in fits:
                        continue
                    row = dict(sheet=sheet, yob=yob, num=num, den=den)
                    reporting_date = pd.Timestamp(K.get_reporting_date(sheet) or "2022-12-31")
                    for idx, label in enumerate(("first", "exact")):
                        a, b = fits[yob, num][idx], fits[yob, den][idx]
                        row[label] = a["accepted"] and b["accepted"]
                        row[label + "_kcor"] = None
                        if row[label]:
                            ratio = (a["curve"] / b["curve"]).dropna()
                            shared_dates = a["curve"].index.intersection(b["curve"].index)
                            anchor = shared_dates[K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE]
                            available = ratio.loc[:reporting_date]
                            if len(available) and anchor in ratio and ratio[anchor] > 0:
                                row[label + "_kcor"] = float(available.iloc[-1] / ratio[anchor])
                    pair_rows.append(row)
            print(f"Replayed {sheet}: {len(fits)} cohort fits", flush=True)
    return dict(cohorts=cohort_rows, pairs=pair_rows,
                workbook_sha256=hashlib.file_digest(WORKBOOK.open("rb"), "sha256").hexdigest())


def summarize(rows):
    counts = {f"{a}->{b}": sum(r["first"] == a and r["exact"] == b for r in rows)
              for a in (False, True) for b in (False, True)}
    changes = [abs(r["exact_kcor"] - r["first_kcor"]) for r in rows
               if r["first"] and r["exact"] and r.get("first_kcor") is not None
               and r.get("exact_kcor") is not None]
    return dict(n=len(rows), decisions=counts, paired_points=len(changes),
                median_abs_kcor_change=float(np.median(changes)) if changes else None,
                max_abs_kcor_change=max(changes, default=None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("simulation", "replay"))
    mode = parser.parse_args().mode
    original, exact, source_hash = exact_weekly_variant()
    tree = ast.parse((ROOT / "code/KCOR.py").read_text(encoding="utf-8"))
    local = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == original.__name__)
    assert ast.dump(local) == ast.dump(ast.parse(inspect.getsource(original)).body[0])
    if mode == "simulation":
        rows = list(scenarios(original, exact))
        report = dict(rows=rows, summary=summarize(rows))
        report["by_gap_multiplier"] = {str(m): summarize([r for r in rows if r["gap_multiplier"] == m])
                                       for m in (1., 1.03, 1.30)}
    else:
        report = replay(original, exact)
        report["summary"] = {name: summarize(report[name]) for name in ("cohorts", "pairs")}
    report["fit_source_sha256"] = source_hash
    (HERE / f"{mode}_sensitivity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
