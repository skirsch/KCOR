"""Validate original fixed-cohort components before any grouping."""
import numpy as np
import pandas as pd

def invalid_components(frame, start, horizon=None, layout="disjoint"):
    """Return (age, dose) groups containing an invalid source history.

    A positive residual risk set cannot silently disappear before the sheet
    horizon. A contiguous suffix missing after complete exhaustion is allowed.
    Non-death exits between observed weeks are allowed; unexplained increases
    and missing weeks are not. Fractional expected counts are permitted here
    (the person bootstrap independently requires integer history cells).
    """
    if frame.empty:
        return {}
    start=pd.Timestamp(start)
    horizon=pd.Timestamp(horizon if horizon is not None else frame.DateDied.max())
    keys=["YearOfBirth","Dose"]+[c for c in ("Sex","DCCI") if c in frame]
    bad={}
    if layout=="table8":
        # Known age-shift input has intentionally relabelled overlapping rows,
        # without original constituent IDs. Validate constant multiplicity
        # within the declared pseudo-cohort and its combined risk history.
        for key,g in frame.groupby(keys,sort=False,dropna=False):
            counts=g.groupby("DateDied").size()
            if counts.nunique()>1:
                bad.setdefault(tuple(key[:2]),[]).append((tuple(key),"inconsistent age-shift source-row multiplicity"))
        frame=frame.groupby(["YearOfBirth","Dose","DateDied"],as_index=False)[["Alive","Dead"]].sum()
        keys=["YearOfBirth","Dose"]

    for key,g in frame.groupby(keys,sort=False,dropna=False):
        x=g.sort_values("DateDied")
        dates=pd.DatetimeIndex(x.DateDied)
        a=pd.to_numeric(x.Alive,errors="coerce").to_numpy(float)
        d=pd.to_numeric(x.Dead,errors="coerce").to_numpy(float)
        tol=1e-9*np.maximum(1.,np.abs(a))
        reason=None
        if dates.has_duplicates or dates[0]!=start or not np.all(np.diff(dates.values)==np.timedelta64(7,"D")):
            reason="missing or duplicate component week"
        elif not (np.isfinite(a).all() and np.isfinite(d).all()) or np.any(a<0) or np.any(d<0) or np.any(d-a>tol):
            reason="invalid component risk/death counts"
        elif np.any(a[1:]>a[:-1]-d[:-1]+tol[:-1]):
            reason="unexplained component risk-set increase"
        elif dates[-1]<horizon and a[-1]-d[-1]>tol[-1]:
            reason="truncated positive-risk component"
        if reason:
            bad.setdefault(tuple(key[:2]),[]).append((tuple(key),reason))
    return bad
