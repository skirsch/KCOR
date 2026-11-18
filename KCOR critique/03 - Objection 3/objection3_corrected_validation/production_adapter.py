"""Isolated adapter to the corrected KCOR production functions from PR #4.

The fast array assembly is verified against process_workbook; it is not a
replacement fitter. Configuration overrides are in-memory, never on disk.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
os.environ['DATASET'] = 'Czech'
os.environ['MONTE_CARLO'] = '1'
os.environ['MC_INCLUDE_AGGREGATES'] = '0'
os.environ['MC_ENROLLMENT_DATE'] = '2021_24'
os.environ['MC_YOB_START'] = '1950'
os.environ['MC_YOB_END'] = '1950'
SOURCE_ROOT = Path(__file__).resolve().parent / 'source_snapshot'
sys.path.insert(0, str(SOURCE_ROOT / 'code'))
import KCOR as K

START = pd.Timestamp.fromisocalendar(2021, 24, 1)
FLAGS = ('identifiability_flag', 'insufficient_signal', 'insufficient_deaths',
         'bad_fit', 'theta_hit_bound', 'delta_negative_clamped')


def configure(windows=((6, 55),)):
    K._load_dataset_dose_pairs_config()
    K._DATASET_COVID_CORRECTION_CACHE = {}  # Optional wave adjustment disabled.
    K._DATASET_DOSE_PAIRS_CACHE = {'2021_24': [(1, 0)], 'default': [(1, 0)]}
    entries = []
    for a, b in windows:
        labels = [(START + pd.Timedelta(weeks=int(t))).strftime('%G-%V') for t in (a, b)]
        entries.append(dict(start_iso=labels[0], end_iso=labels[1],
                            start_int=K.iso_label_to_int(labels[0]),
                            end_int=K.iso_label_to_int(labels[1])))
    K._DATASET_THETA_ESTIMATION_WINDOWS_CACHE = entries
    return dict(gompertz=K.get_gompertz_theta_fit_config(),
                guards=K.get_theta_degenerate_fit_config(),
                skip=K.DYNAMIC_HVE_SKIP_WEEKS,
                anchor=K.KCOR_NORMALIZATION_WEEKS_EFFECTIVE,
                windows=entries, optional_wave_adjustment=False)


def arrays(counts):
    """Rows are dose 0,1; columns are death weeks then end survivor count."""
    counts = np.asarray(counts)
    dead = counts[:, :-1].astype(float)
    alive = counts.sum(axis=1)[:, None] - np.c_[np.zeros(2), np.cumsum(dead, axis=1)[:, :-1]]
    return dead, alive


def estimate(counts, windows=((6, 55),), target=80, return_curves=False,
             iteration=0, total_iterations=1):
    cfg = configure(windows)
    dead, alive = arrays(counts)
    calendar_t = np.arange(dead.shape[1], dtype=float)
    # PR #4 groups time by mc_id, age, and dose. Each complete-grid
    # sample therefore uses calendar weeks independent of batch size or ID.
    t = calendar_t
    quiet = np.zeros(len(t), bool)
    for a, b in windows:
        quiet |= (calendar_t >= a) & (calendar_t <= b)
    curves, diagnostics = [], []
    for d in (0, 1):
        mr = np.divide(dead[d], alive[d] + K.EPS,
                       out=np.full(len(t), np.nan), where=alive[d] > 0)
        h = K.hazard_from_mr(np.clip(mr, 0, .999))
        he = np.where(t >= cfg['skip'], h, 0.)
        (kh, th), diag = K.fit_theta0_gompertz(
            he, t - cfg['skip'], quiet, cfg['gompertz']['k_anchor_weeks'],
            cfg['gompertz']['gamma_per_week'], deaths_arr=dead[d])
        theta = float(th) if diag.get('success', False) and np.isfinite(th) and th >= 0 else 0.
        theta, degenerate, reason = K._apply_theta_degenerate_guard(theta, cfg['guards'])
        # Production normalization replaces nonfinite raw hazards by zero.
        H = np.cumsum(np.where(t >= cfg['skip'], np.nan_to_num(h), 0.))
        curves.append(K.invert_gamma_frailty(H, theta))
        diagnostics.append(dict(k=float(kh), theta=float(theta),
                                success=bool(diag.get('success', False)),
                                status=str(diag.get('status', '')),
                                message=str(diag.get('message', '')),
                                degenerate=bool(degenerate),
                                flags={f: bool(diag.get(f, False)) for f in FLAGS}))
    ch = np.asarray(curves)
    ratio = np.divide(ch[1], ch[0], out=np.full(len(t), np.nan), where=ch[0] > K.EPS)
    anchor = cfg['anchor'] if len(t) > cfg['anchor'] else 0
    baseline = ratio[anchor]
    if not (np.isfinite(baseline) and baseline > K.EPS):
        baseline = 1.
    curve = ratio / baseline
    accepted = all(d['success'] and not d['degenerate'] and not any(d['flags'].values())
                   for d in diagnostics) and np.isfinite(curve[target])
    result = dict(point=float(curve[target]), accepted=bool(accepted), diagnostics=diagnostics)
    if return_curves:
        result.update(curve=curve, ch=ch)
    return result


def intervals(point, draws):
    # Matches create_monte_carlo_summary: iteration zero is never a draw.
    s = pd.Series(np.asarray(draws, dtype=float)).dropna()
    lo, hi = float(s.quantile(.025)), float(s.quantile(.975))
    return dict(percentile=[lo, hi], basic=[2 * point - hi, 2 * point - lo],
                valid_draws=len(s))


def workbook_frame(counts, iteration):
    dead, alive = arrays(counts)
    dates = pd.date_range(START, periods=dead.shape[1], freq='7D')
    return pd.concat([pd.DataFrame(dict(
        ISOweekDied=dates.strftime('%G-%V'), DateDied=dates, YearOfBirth=1950,
        Sex='O', Dose=d, Alive=alive[d], Dead=dead[d], mc_id=iteration))
        for d in (0, 1)], ignore_index=True)
