"""Table 8 sensitivity to exclusion of nonquiet early fitting anchors."""
from anchor_diagnostic import diagnostic,r
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

def init(pop):
    r.table.initialize(pop,"entry-moment")
    global fit
    fit=diagnostic(r.table.K)
def task(draw):
    params=[]
    def record(*a,**kw):
        p,d=fit(*a,**kw);params.append([p[0],p[1],int(d["success"])]);return p,d
    r.table.compare.joint_fit=record
    result=r.table.estimate(1,draw)
    return draw,result,np.array(params)

def main():
    t=r.table
    book=r.subprocess.check_output(["git","cat-file","blob",t.INPUT_BLOB],cwd=r.ROOT)
    pop=[t.read_population(r.io.BytesIO(book),s) for s in t.SHEETS]
    v=np.empty((1000,3));par=np.empty((1000,9,3));flags=np.empty((1000,2))
    with ProcessPoolExecutor(6,initializer=init,initargs=(pop,)) as pool:
        for n,(d,(point,f,e),p) in enumerate(pool.map(task,range(1000),chunksize=5),1):
            v[d]=point;par[d]=p;flags[d]=f,e
            if n%100==0:print("quiet-only Table8",n,flush=True)
    np.savez_compressed(r.HERE/"quiet-only-table8.npz",points=v,parameters=par,flags=flags)
    lo,hi=np.quantile(v[1:],[.025,.975],axis=0)
    df=pd.DataFrame(dict(pair=("1/0","2/0","2/1"),point=v[0],lower95=lo,upper95=hi,width=hi-lo))
    df.to_csv(r.HERE/"quiet-only-table8.csv",index=False)
    print(df.to_string(index=False),flags.sum(axis=0),flush=True)
    rows=[]
    for i in range(9):
        for j,name in enumerate(("k","theta")):
            x=par[:,i,j];l,h=np.quantile(x[1:],[.025,.975])
            rows.append(dict(group=("two-dose","unvaccinated","all-persons")[i//3],pseudo_dose=i%3,parameter=name,point=x[0],lower95=l,upper95=h))
    pd.DataFrame(rows).to_csv(r.HERE/"quiet-only-parameters.csv",index=False)
if __name__=="__main__":main()
