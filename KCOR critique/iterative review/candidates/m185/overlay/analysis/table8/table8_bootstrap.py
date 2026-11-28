"""Person-bootstrap percentile intervals for the archived Czech Table 8 joint-fit trial."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "paired-seed-fit"))
import compare
import numpy as np
import pandas as pd

INPUT_BLOB = "f2dbb3d7a39d47a06821048de042285df46785ba"
SHEETS = ("2021_24", "2022_06")
PAIRS = ((1, 0), (2, 0), (2, 1))
REPORT = pd.Timestamp("2022-12-26")
SEED = 20261121
DIAGNOSTIC_CATEGORIES = ("insufficient_signal", "residual_screen", "residual_screen_unavailable",
                         "theta_upper_bound", "baseline_rate_bound", "weak_identification",
                         "insufficient_deaths", "failed_fit", "legacy_relative_error", "gap_clamp")


def fit_diagnostic_flags(diag):
    """Numerical, residual and raw-clamp indicators remain distinct."""
    return np.asarray([bool(diag.get("insufficient_signal")), bool(diag.get("sampling_flag")),
        not bool(diag.get("sampling_n", 0)), bool(diag.get("theta_hit_bound")), bool(diag.get("k_hit_bound")),
        bool(diag.get("identifiability_flag")) or diag.get("status") == "weak_identifiability",
        bool(diag.get("insufficient_deaths")), not bool(diag.get("success")),
        bool(diag.get("legacy_relative_error_flag")), bool(diag.get("delta_negative_clamped"))],dtype=bool)


def read_population(workbook, sheet):
    sys.path.insert(0, str(HERE.parents[1] / "code"))
    from bootstrap_reporting import histories
    _,dates,disjoint=histories(pd.read_excel(workbook,sheet_name=sheet),sheet,"table8")
    return dates,disjoint


def initialize(populations, weekly_update="revised"):
    global K, CONFIG, WAVE, GUARD, DATA
    os.environ["MONTE_CARLO"] = "0"
    compare.initialize(weekly_update)
    K = compare.adapter.K
    CONFIG = K.get_gompertz_theta_fit_config()
    WAVE = K.get_covid_correction_config()
    GUARD = K.get_theta_degenerate_fit_config()
    assert not WAVE["apply_correction"] or WAVE["apply_alpha"] == 1.18
    DATA = []
    for dates, counts in populations:
        time = np.arange(len(dates), dtype=float)
        quiet, _, _ = K._build_theta_quiet_mask(dates.strftime("%G%V").astype(int).to_numpy(), time)
        target = dates.get_loc(REPORT)
        # All in-wave dates and the anchor are retained. After wave exit the
        # production inversion depends only on survival at exit and at target.
        keep = (dates <= WAVE["end_dt"]) | (time == K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE) | (time == target)
        DATA.append((dates.to_numpy(), time, quiet, keep, counts))


def estimate(index, draw, joint=True, diagnostics=False, return_composition=False):
    dates, time, quiet, keep, disjoint = DATA[index]
    counts = disjoint.copy()
    if draw:
        rng = np.random.default_rng([SEED, index, draw])
        total = int(disjoint.sum())
        counts = rng.multinomial(total, disjoint.ravel() / total).reshape(disjoint.shape)
    counts[2] += counts[0] + counts[1]
    alive = np.cumsum(counts[:, :, ::-1], axis=2)[:, :, ::-1][:, :, :-1]
    dead = counts[:, :, :-1]
    mr = np.divide(dead,alive,out=np.full(dead.shape,np.nan),where=alive>0)
    early = mr[:, :, :4]
    from pooling import expected_deaths_weights
    weights = expected_deaths_weights(early,alive[:, :, :4].sum(axis=(1,2)),K.EPS)
    end_value = np.full((3,3),np.nan)
    anchor_value = np.full((3,3),np.nan)
    support_ok = np.zeros((3,3),dtype=bool)
    flags = 0
    failed = 0
    fit_flags = np.zeros((3,3,len(DIAGNOSTIC_CATEGORIES)),dtype=bool)
    skip = K.DYNAMIC_HVE_SKIP_WEEKS
    fit = compare.joint_fit if joint else compare.production_fit
    for a, d in np.ndindex(end_value.shape):
        hazard = K.hazard_from_mr(mr[a,d])
        effective = np.where(time >= skip, hazard, 0.)
        diagnostic_kwargs = {"risk_arr": alive[a,d]} if diagnostics else {}
        (k, theta), diag = fit(effective, time-skip, quiet, CONFIG["k_anchor_weeks"],
                               CONFIG["gamma_per_week"], deaths_arr=dead[a, d],
                               iso_week=pd.DatetimeIndex(dates).strftime("%G%V").astype(int), **diagnostic_kwargs)
        if diagnostics:fit_flags[a,d]=fit_diagnostic_flags(diag)
        failed += int(not diag["success"])
        flags += int(bool(diag.get("insufficient_signal")))
        theta = theta if diag["success"] and np.isfinite(theta) and theta >= 0 else 0.
        theta, _, _ = K._apply_theta_degenerate_guard(theta, GUARD)
        observed = np.cumsum(effective)[keep]
        inverted = K.invert_gamma_frailty(observed, theta)
        corrected, _ = K.apply_nph_correction_post_inversion(
            inverted, time[keep], dates[keep], theta, (WAVE["apply_alpha"] if WAVE["apply_correction"] else 1.),
            WAVE["start_dt"], WAVE["end_dt"], k, CONFIG["gamma_per_week"], observed_cumhaz=observed)
        anchor = np.flatnonzero(time[keep] == K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE).item()
        support_ok[a,d] = np.isfinite(corrected[-1]) and np.isfinite(corrected[anchor])
        end_value[a,d] = corrected[-1]
        anchor_value[a,d] = corrected[anchor]
    pooled=[]; members=[]
    expected=np.isfinite(weights)&(weights>0)
    for n,d in PAIRS:
        # A zero (undefined) anchor allows available-strata pooling, as in the
        # ordinary workbook. An exhausted or failed normalization does not.
        reference=K._finite_hazard_ratio(anchor_value[:,n],anchor_value[:,d])
        raw=K._finite_hazard_ratio(end_value[:,n],end_value[:,d])
        ratio=K._finite_hazard_ratio(raw,reference)
        use=expected & np.isfinite(ratio) & (ratio>=0)
        members.append(use)
        if np.any(expected & ~(support_ok[:,n]&support_ok[:,d])) or not use.any():
            pooled.append(np.nan)
        else:
            with np.errstate(over="ignore",under="ignore"):
                value=0. if (ratio[use]==0).any() else float(np.exp(np.average(np.log(ratio[use]),weights=weights[use])))
            pooled.append(value if np.isfinite(value) and (value>0 or (ratio[use]==0).any()) else np.nan)
    members=np.asarray(members,dtype=bool)
    result=(pooled,flags,failed)
    if diagnostics:
        pair_flags=np.stack([fit_flags[use][:,[n,d],:].any(axis=(0,1)) for use,(n,d) in zip(members,PAIRS)])
        result+=(fit_flags,pair_flags)
    return result+(members,) if return_composition else result


def composition_intervals(values, members):
    """Use all draws or report unavailable; do not condition on a changed population."""
    raw=np.asarray(values,float)
    members=np.asarray(members,bool)
    matched=raw.copy()
    changed=np.any(members[1:]!=members[0],axis=-1) & np.isfinite(raw[0])
    matched[1:][changed]=np.nan
    valid=np.isfinite(matched[1:]).sum(axis=0)
    lo=np.full(raw.shape[1],np.nan);hi=lo.copy()
    good=(valid==len(raw)-1)&np.isfinite(raw[0])&(len(raw)>1)
    if good.any():
        lo[good],hi[good]=np.quantile(matched[1:,good],[.025,.975],axis=0)
    return matched,lo,hi,changed.sum(axis=0),valid


def one_draw(task):
    index, draw = task
    return index, draw, estimate(index, draw, return_composition=True)


def workbook_points(path, populations):
    frame = pd.read_excel(path, sheet_name="dose_pairs")
    rows = frame[(pd.to_datetime(frame.Date) == REPORT) & (frame.YearOfBirth == 0)]
    return [[float(rows[(rows.EnrollmentDate==sheet)&(rows.Dose_num==n)&(rows.Dose_den==d)].KCOR.iloc[0]) for n,d in PAIRS] for sheet in SHEETS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=999)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-workbook", type=Path, help="Full-workbook reference for a later revision of the fitter.")
    edition = parser.add_mutually_exclusive_group()
    for mode in ("iteration-2", "gap-boundary", "wave-entry"):
        edition.add_argument("--" + mode, dest="edition", action="store_const", const=mode)
    args = parser.parse_args()
    if args.edition and not args.check_workbook:
        parser.error("Iteration 2 replays require --check-workbook")
    book = io.BytesIO((HERE/"input.xlsx").read_bytes())
    populations = [read_population(book, sh) for sh in SHEETS]
    if args.edition or not args.check_workbook:
        parser.error("Use replay.py to generate the matching workbook")
    weekly_update = "revised"
    initialize(populations, weekly_update)
    point = [estimate(i, 0, return_composition=True) for i in range(2)]
    # Permanent regression check against the six independently reconstructed
    # reporting-date values from the full ordinary-mode joint-fit workbook.
    reference = [[1.00906808859867, 1.0317663863960909, 1.0224943173348615],
                 [1.0503159510304514, 1.0574772450702619, 1.0068182283938316]]
    print("Joint-fit points:", [x[0] for x in point], flush=True)
    tasks = [(i, j) for j in range(1, args.draws+1) for i in range(2)]
    results = [[] for _ in SHEETS]
    with ProcessPoolExecutor(args.workers, initializer=initialize, initargs=(populations, weekly_update)) as pool:
        for n, (i, draw, value) in enumerate(pool.map(one_draw, tasks, chunksize=4), 1):
            results[i].append(value)
            if n % 200 == 0:
                print(f"{n}/{len(tasks)} enrollment draws finished", flush=True)
    if args.check_workbook:
        reference = workbook_points(args.check_workbook, populations)
    np.testing.assert_allclose([x[0] for x in point], reference, rtol=1e-10, atol=1e-12)
    print("All six original estimates match the workbook.", flush=True)
    summary = {}
    for i, sheet in enumerate(SHEETS):
        values = np.array([point[i][0]]+[r[0] for r in results[i]])
        members=np.array([point[i][-1]]+[r[-1] for r in results[i]])
        _,lo,hi,changed,valid=composition_intervals(values,members)
        summary[sheet] = dict(joint=point[i][0], original_insufficient_signal=point[i][1],
                              people=int(populations[i][1].sum()), lower=lo.tolist(),upper=hi.tolist(),
                              point_contributors=members[0].tolist(),composition_changed_draws=changed.tolist(),valid_draws=valid.tolist(),
                              draws_with_insufficient_signal=sum(r[1]>0 for r in results[i]),
                              unsuccessful_stratum_fits=sum(r[2] for r in results[i]))
    report = dict(input_blob=INPUT_BLOB, seed=SEED, draws_per_enrollment=args.draws,
                  date=str(REPORT.date()), pairs=PAIRS, summary=summary)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
