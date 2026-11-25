"""Independent arithmetic check of exact and first-order weekly inversion."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def main():
    gamma=.085/52.;k=.001;skip=2;weeks=100
    baseline=k*np.exp(gamma*(np.arange(weeks)-skip))
    baseline_cum=np.cumsum(baseline)
    arms=[]
    for theta in (4.,.5):
        Hobs=np.log1p(theta*baseline_cum)/theta
        Hobs=np.r_[0.,Hobs]
        observed=np.diff(Hobs)
        observed[:skip]=0.
        cumulative=np.cumsum(observed)
        exact=np.expm1(theta*cumulative)/theta
        approximate=np.empty(weeks)
        current=0.
        for w,h in enumerate(observed):
            current+=h*(1+theta*current)
            approximate[w]=current
        x=theta*observed[skip:]
        factor=np.expm1(x)/x
        exact_weekly_error=float(np.max(factor-1))
        arms.append(dict(theta=theta,
            max_relative_weekly_increment_error=float(exact_weekly_error),
            relative_cumulative_error_week_52=float((approximate[52]-exact[52])/exact[52]),
            exact_cumulative_week_52=float(exact[52]),
            approximate_cumulative_week_52=float(approximate[52])))
    assert all(r['max_relative_weekly_increment_error']>0 and r['relative_cumulative_error_week_52']<0 for r in arms)
    report=dict(model='Gamma-frailty survival with k=.001, gamma=.085/52, skip=2',
        identity='Delta H0 = (1+theta*H0_before)*expm1(theta*h_obs)/theta',
        paper_update='Delta H0 = (1+theta*H0_before)*h_obs',
        arms=arms,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'documentation/preprint/paper.md',ROOT/'code/KCOR.py')})
    out=HERE/'results.json'
    with out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))

if __name__=='__main__':main()
