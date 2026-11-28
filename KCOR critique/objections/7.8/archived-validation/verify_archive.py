"""Recompute historical S8 reported intervals from the archived matched draws."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
def main():
    result={}
    for mode,stem in [('main','main-full'),('stress','stress')]:
        a=np.load(HERE/f"{mode}-gamma-draws.npz")["points"]
        b=np.load(HERE/f"{mode}-nonlinear-draws.npz")["points"]
        assert a.shape==b.shape
        stored=pd.read_csv(HERE/f"{stem}-intervals.csv")
        saved=json.loads((HERE/f"{stem}-results.json").read_text(encoding="utf8"))
        result[mode]={}
        for name,arr in [('gamma_working_fit',a),('nonlinear_survivor_fit',b)]:
            rows=stored[stored.fitter==name].sort_values("rep")
            lo,hi=np.quantile(arr[:,1:,0],[.025,.975],axis=1)
            np.testing.assert_allclose(rows.point,arr[:,0,0],rtol=1e-12)
            np.testing.assert_allclose(rows.lower95,lo,rtol=1e-12)
            np.testing.assert_allclose(rows.upper95,hi,rtol=1e-12)
            cover=int(((lo<=1)&(hi>=1)).sum())
            assert cover==saved[name]["covered"]
            assert abs(np.mean(arr[:,0,0])-saved[name]["mean_point"][0])<1e-12
            result[mode][name]=dict(datasets=len(rows),draws=arr.shape[1]-1,coverage=cover,mean=float(np.mean(arr[:,0,0])))
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
