from pathlib import Path
import numpy as np, csv
h=Path(__file__).resolve().parent
p=np.load(h/"multistart-stress.npz")["points"]
ref=np.load(h.parents[2]/"1.3/multiwindow-coverage/fallback-stress-draws.npz")["points"]
np.testing.assert_allclose(p[:,:,0],ref[:,:,1],rtol=1e-9,atol=1e-10)
for j,name in enumerate(("current","multistart")):
    x=p[:,:,j,0];lo,hi=np.quantile(x[:,1:],[.025,.975],axis=1)
    print(name, "width",np.median(hi-lo),np.max(hi-lo),"coverage",np.sum((lo<=1)&(hi>=1)),"median abs error",np.median(abs(x[:,0]-1)), "rep34",x[34,0],lo[34],hi[34])
    for a in (7,9):
        y=p[:,:,j,a]; print("theta .5",a,"median",np.median(y[:,0]),"rep34",y[34,0],np.quantile(y[34,1:],[.025,.975]))
    print("flags",p[:,:,j,3].sum(),"failures",p[:,:,j,4].sum())
    with (h/(name+"-stress-summary.csv")).open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["dataset","point","lower95","upper95"]);w.writerows(zip(range(len(x)),x[:,0],lo,hi))
v=np.load(h/"multistart-table.npz")
print("table flags",v["flags"].sum(axis=1))
print("table max point/draw difference vs profile",np.max(abs(v["points"]-np.load(h/"profile-table.npz")["points"])))
