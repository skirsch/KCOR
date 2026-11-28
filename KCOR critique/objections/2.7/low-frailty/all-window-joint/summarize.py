"""Write reproducible summaries from completed paired draws."""
from pathlib import Path
import numpy as np
import json, csv
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
old=np.load(ROOT/"1.2/entry-moment/table8-draws.npz")
print("archive",[(k,old[k].shape) for k in old.files])
new=np.load(HERE/"profile-table.npz")
arch=json.loads((ROOT/"1.2/entry-moment/table8_bootstrap.json").read_text())
rows=[]
for i,s in enumerate(("2021_24","2022_06")):
    ref=arch["summary"][s]
    lo,hi=np.quantile(new["points"][i,1:],[.025,.975],axis=0)
    for j,pair in enumerate(("1/0","2/0","2/1")):
        rows.append([s,pair,ref["joint"][j],ref["lower"][j],ref["upper"][j],new["points"][i,0,j],lo[j],hi[j],(hi[j]-lo[j])/(ref["upper"][j]-ref["lower"][j])])
with (HERE/"table-comparison.csv").open("w",newline="") as f:
    w=csv.writer(f);w.writerow(["enrollment","pair","current","current_lo","current_hi","trial","trial_lo","trial_hi","width_ratio"]);w.writerows(rows)
print("table flags, failures", new["flags"].sum(axis=1))
if (HERE/"profile-stress.npz").exists():
    v=np.load(HERE/"profile-stress.npz")["points"]
    ref=np.load(ROOT/"1.3/multiwindow-coverage/fallback-stress-draws.npz")["points"]
    np.testing.assert_allclose(v[:,:,0],ref[:,:,1],rtol=1e-9,atol=1e-10)
    rows=[]
    for j,label in enumerate(("current","trial")):
        x=v[:,:,j,0];lo,hi=np.quantile(x[:,1:],[.025,.975],axis=1)
        print(label,"coverage",np.mean((lo<=1)&(hi>=1)),"median width",np.median(hi-lo),"max width",np.max(hi-lo),"median abs error",np.median(abs(x[:,0]-1)),"rep34",x[34,0],lo[34],hi[34],"failures",v[:,:,j,4].sum(),"flags",v[:,:,j,3].sum())
        for a in (7,9):
            y=v[:,:,j,a];print(label,"theta .5",a,"median",np.median(y[:,0]),"rep34",y[34,0],np.quantile(y[34,1:],[.025,.975]))
        rows.extend([[i,label,x[i,0],lo[i],hi[i]] for i in range(50)])
    with (HERE/"stress-comparison.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["dataset","fit","point","lo","hi"]);w.writerows(rows)
if (HERE/"profile-table-parameters.npz").exists():
    p=np.load(HERE/"profile-table-parameters.npz")["parameters"]
    rows=[]
    for i,s in enumerate(("2021_24","2022_06")):
        lo,hi=np.quantile(p[i,1:],[.025,.975],axis=0)
        for j in range(9):
            rows.append([s,j//3,j%3,p[i,0,j,0],lo[j,0],hi[j,0],p[i,0,j,1],lo[j,1],hi[j,1]])
    with (HERE/"table-parameters.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["enrollment","group","pseudo_dose","k","k_lo","k_hi","theta","theta_lo","theta_hi"]);w.writerows(rows)
