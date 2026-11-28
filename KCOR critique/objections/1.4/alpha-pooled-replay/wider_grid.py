"""Wider-grid sensitivity only after the observed lower-bound minimum."""
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location("pooled_alpha_replay",Path(__file__).resolve().parent/"replay.py")
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
def init():r.setup()
def task(alpha):return r.exact_task(alpha)
def main():
    r.setup()
    alpha=np.round(np.arange(.7,1.,.005),10);alpha=alpha[alpha<1]
    records=[];fits=[]
    with ProcessPoolExecutor(7,initializer=init) as pool:
        for n,(a,x,d) in enumerate(pool.map(task,alpha),1):
            records.extend(x);fits.extend(d)
            if n%10==0:print("lower-grid sensitivity",n,"/",len(alpha),flush=True)
    pd.DataFrame(records).to_csv(r.HERE/"lower-grid-moments.csv",index=False)
    pd.DataFrame(fits).to_csv(r.HERE/"lower-grid-fits.csv",index=False)
if __name__=="__main__":main()

