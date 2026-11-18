"""Candidate parameter fit with KCOR's unchanged inversion and normalization."""
from __future__ import annotations
import sys
from dataclasses import asdict
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'objection3_corrected_validation'))
import numpy as np
from production_adapter import K, arrays, configure
from conditional_fit import fit_conditional


def estimate_candidate(counts, windows=((6, 40),), target=52, starts=(.1, 1., 5.)):
    if len(windows) != 1:
        raise ValueError('Candidate supports one contiguous quiet window only')
    first, last = map(int, windows[0])
    cfg = configure(windows)
    dead, alive = arrays(counts)
    if not (0 <= first < last < dead.shape[1] and target < dead.shape[1]):
        raise ValueError('Invalid window or target')
    fitted, curves = [], []
    t = np.arange(dead.shape[1])
    for arm in (0, 1):
        fit = fit_conditional(dead[arm, first:last+1], alive[arm, first:last+1],
                              gamma_per_week=cfg['gompertz']['gamma_per_week'], starts=starts)
        fitted.append(asdict(fit))
        mr = dead[arm] / (alive[arm] + K.EPS)
        h = K.hazard_from_mr(np.clip(mr, 0, .999))
        H = np.cumsum(np.where(t >= cfg['skip'], np.nan_to_num(h), 0.))
        curves.append(K.invert_gamma_frailty(H, fit.theta))
    ch = np.asarray(curves)
    ratio = np.divide(ch[1], ch[0], out=np.full(len(t), np.nan), where=ch[0] > K.EPS)
    baseline = ratio[cfg['anchor']]
    if not (np.isfinite(baseline) and baseline > K.EPS):
        baseline = 1.
    curve = ratio / baseline
    point = float(curve[target])
    accepted = bool(all(f['success'] for f in fitted) and np.isfinite(point))
    return dict(point=point, accepted=accepted, diagnostics=fitted)
