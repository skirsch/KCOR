"""Decompose 40-49 endpoint variability using the same person draws as the precision check."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate",type=Path)
    args=parser.parse_args()
    raw=pd.read_excel(args.candidate/"data/Czech/KCOR_CMR.xlsx",sheet_name="2021_24",
                      usecols=["YearOfBirth","Dose","ISOweekDied","Alive","Dead"])
    raw["week"]=raw.ISOweekDied.astype(str).str.replace("-","",regex=False).str[:6].astype(int)
    raw=raw[raw.YearOfBirth.isin([1975,1980])&raw.Dose.isin([0,2])&raw.week.between(202124,202416)]
    groups=raw.groupby(["YearOfBirth","Dose","week"],as_index=False)[["Alive","Dead"]].sum()
    rng=np.random.default_rng(20261126)
    counts=np.zeros((999,2,2,149),dtype=np.int64);observed=np.zeros((2,2,149),dtype=np.int64)
    for j,dose in enumerate((0,2)):
        for yob in (1975,1980):
            x=groups[(groups.YearOfBirth==yob)&(groups.Dose==dose)].sort_values("week")
            risk,dead=x.Alive.to_numpy(int),x.Dead.to_numpy(int)
            cells=np.r_[dead,risk[-1]-dead[-1]]
            assert cells.sum()==risk[0]
            boot=rng.multinomial(risk[0],cells/risk[0],size=999)
            counts[:,j,0]+=np.cumsum(boot[:,::-1],axis=1)[:,::-1][:,:-1]
            counts[:,j,1]+=boot[:,:-1]
            observed[j]+=[risk,dead]
    H=-np.log1p(-counts[:,:,1]/counts[:,:,0]);H=H[:,:,2:].cumsum(axis=2)
    Hobs=-np.log1p(-observed[:,1]/observed[:,0]);Hobs=Hobs[:,2:].cumsum(axis=1)
    stored=json.loads((HERE/"frailty-resolution.json").read_text(encoding="utf-8"))
    theta=np.array([p["theta"] for p in stored["point"]])
    rows=pd.read_csv(HERE/"frailty-resolution-draws.csv")
    drawtheta=rows[["theta0","theta2"]].to_numpy()
    def compare(h,th):
        th=np.broadcast_to(th,h.shape);out=h.copy()
        np.divide(np.expm1(th*h),th,out=out,where=th>1e-10)
        raw=out[:,1]/out[:,0]
        return raw[:,-1]/raw[:,3],raw[:,3],raw[:,-1]
    full,ref,endpoint=compare(H,drawtheta[:,:,None])
    assert np.allclose(full,rows.kcor,rtol=1e-10)
    fixed,_,_=compare(H,theta[None,:,None])
    pt,ref_pt,end_pt=compare(Hobs[None,:,:],theta[None,:,None])
    fixed_ref=endpoint/ref_pt[0]
    point_hyp,_,_=compare(Hobs[None,:,:],np.array([theta[0],0.])[None,:,None])
    result=dict(actual_anchor_deaths=observed[:,1,2:6].sum(axis=1).tolist(),
        actual_anchor_ratio=float(ref_pt[0]),actual_endpoint_unanchored_ratio=float(end_pt[0]),
        full_bootstrap=np.percentile(full,[2.5,50,97.5]).tolist(),
        fixed_frailty_bootstrap=np.percentile(fixed,[2.5,50,97.5]).tolist(),
        fixed_reference_ratio_bootstrap=np.percentile(fixed_ref,[2.5,50,97.5]).tolist(),
        bootstrap_anchor_ratio=np.percentile(ref,[2.5,50,97.5]).tolist(),
        point_if_theta2_zero=float(point_hyp[0]))
    (HERE/"anchor-precision.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
