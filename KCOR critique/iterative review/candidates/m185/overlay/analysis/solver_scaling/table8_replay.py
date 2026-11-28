"""Full matched Table 8 replay after residual scaling; preserve original seed."""
from pathlib import Path
import os,sys,io,json,argparse,time,hashlib,platform,importlib.metadata
from concurrent.futures import ProcessPoolExecutor
for k in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):os.environ[k]="1"
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"analysis/table8"))
import table8_bootstrap as T
def init(pop,mode):
    os.environ["KCOR_NPH_MODE"]=mode
    T.initialize(pop,"revised")
    assert T.WAVE["apply_correction"]==(mode=="forced_alpha")
def one(job):
    i,j=job
    recorded=[]
    native=T.compare.joint_fit
    def logged(*a,**kw):
        p,d=native(*a,**kw)
        recorded.append([float(p[0]),float(p[1]),float(d.get("theta0_raw",p[1]))])
        return p,d
    T.compare.joint_fit=logged
    try:
        point,flags,failed,group_flags,pair_flags,members=T.estimate(i,j,diagnostics=True,return_composition=True)
    finally:T.compare.joint_fit=native
    return i,j,point,flags,failed,np.asarray(recorded),group_flags,pair_flags,members
def provenance(mode, draws):
    files = [ROOT/"analysis/table8/input.xlsx", Path(__file__).resolve()]
    for folder in ("code", "analysis/table8", "config"):
        files.extend(p for p in (ROOT/folder).rglob("*") if p.is_file() and p.suffix in (".py",".yaml",".yml",".json"))
    files.append(ROOT/"data/Czech/Czech.yaml")
    hashes = {}
    for path in sorted(set(files)):
        with path.open("rb") as f:
            hashes[path.relative_to(ROOT).as_posix()] = hashlib.file_digest(f, "sha256").hexdigest()
    return dict(mode=mode, draws=draws, files=hashes, python=platform.python_version(),
                packages={k:importlib.metadata.version(k) for k in ("numpy","scipy","pandas","openpyxl")},
                environment={k:v for k,v in sorted(os.environ.items()) if k.startswith(("KCOR_", "SA_", "MC_")) or k in ("DATASET","MONTE_CARLO","SENSITIVITY_ANALYSIS")})

def main():
    p=argparse.ArgumentParser();p.add_argument("--mode",choices=["identified_only","forced_alpha","both"],default="both");p.add_argument("--draws",type=int,default=999);p.add_argument("--workers",type=int,default=7);p.add_argument("--output-dir",type=Path,default=HERE/"fresh-replay");p.add_argument("--resume",action="store_true");args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.output_dir.resolve()==HERE.resolve(): p.error("Use a separate output directory to preserve archived results.")
    populations=[T.read_population(io.BytesIO((ROOT/"analysis/table8/input.xlsx").read_bytes()),s) for s in T.SHEETS]
    for mode in (["identified_only","forced_alpha"] if args.mode=="both" else [args.mode]):
        begin=time.monotonic()
        n=args.draws+1
        vals=np.full((2,n,3),np.nan);flags=np.full((2,n,2),np.nan);params=np.full((2,n,9,3),np.nan)
        done=np.zeros((2,n),bool)
        members=np.zeros((2,n,3,3),bool)
        group_diagnostics=np.zeros((2,n,3,3,len(T.DIAGNOSTIC_CATEGORIES)),bool)
        pair_diagnostics=np.zeros((2,n,3,len(T.DIAGNOSTIC_CATEGORIES)),bool)
        checkpoint=args.output_dir/(mode+"-partial.npz")
        metadata=args.output_dir/(mode+"-provenance.json")
        expected=provenance(mode,args.draws)
        if args.resume:
            if not checkpoint.exists() or not metadata.exists() or json.loads(metadata.read_text(encoding="utf8"))!=expected:
                raise ValueError("Resume requires a checkpoint with matching input, code and settings provenance")
            s=np.load(checkpoint)
            assert s["points"].shape==vals.shape
            vals[:]=s["points"];flags[:]=s["flags"];params[:]=s["parameters"];done[:]=s["done"]
            group_diagnostics[:]=s["group_diagnostics"];pair_diagnostics[:]=s["pair_diagnostics"];members[:]=s["contributors"]
        else:
            # Start fresh, including when an old output directory exists.
            metadata.write_text(json.dumps(expected,indent=2),encoding="utf8")
            np.savez_compressed(checkpoint,points=vals,flags=flags,parameters=params,done=done,group_diagnostics=group_diagnostics,pair_diagnostics=pair_diagnostics,contributors=members)
        jobs=[(i,j) for j in range(n) for i in range(2) if not done[i,j]]
        print(mode,"scheduled fit jobs",len(jobs),flush=True)
        with ProcessPoolExecutor(args.workers,initializer=init,initargs=(populations,mode)) as pool:
            for i,j,v,f,e,p,gdiag,pdiag,member in pool.map(one,jobs,chunksize=2):
                vals[i,j]=v;flags[i,j]=f,e;params[i,j]=p;group_diagnostics[i,j]=gdiag;pair_diagnostics[i,j]=pdiag;members[i,j]=member;done[i,j]=True
                count=int(done.sum())
                if j==0 or count%100==0:
                    print(mode,count,"of",2*n,"elapsed seconds",round(time.monotonic()-begin), "point" if j==0 else "", v if j==0 else "",flush=True)
                if j==0 or count%50==0:np.savez_compressed(checkpoint,points=vals,flags=flags,parameters=params,done=done,group_diagnostics=group_diagnostics,pair_diagnostics=pair_diagnostics,contributors=members)
        assert done.all()
        np.savez_compressed(checkpoint,points=vals,flags=flags,parameters=params,done=done,group_diagnostics=group_diagnostics,pair_diagnostics=pair_diagnostics,contributors=members)
        report={};matched=vals.copy()
        for i,sheet in enumerate(T.SHEETS):
            matched[i],low,high,changed,valid=T.composition_intervals(vals[i],members[i])
            diagnostic_counts={}
            for j,(num,den) in enumerate(T.PAIRS):
                diagnostic_counts[f"{num}/{den}"]={name:{"point_flag":bool(pair_diagnostics[i,0,j,k]),"bootstrap_draws":int(pair_diagnostics[i,1:,j,k].sum())} for k,name in enumerate(T.DIAGNOSTIC_CATEGORIES)}
            report[sheet]=dict(point=vals[i,0].tolist(),lower=low.tolist(),upper=high.tolist(),point_contributors=members[i,0].tolist(),composition_changed_draws=changed.tolist(),valid_draws=valid.tolist(),
                insufficient_signal_draws_any_group=int((flags[i,1:,0]>0).sum()),unsuccessful_group_fits=int(flags[i,:,1].sum()),diagnostics=diagnostic_counts)
        (args.output_dir/(mode+"-table8.json")).write_text(json.dumps(report,indent=2),encoding="utf8")
        np.savez_compressed(args.output_dir/(mode+"-draws.npz"),points=matched,raw_points=vals,contributors=members,flags=flags,parameters=params,group_diagnostics=group_diagnostics,pair_diagnostics=pair_diagnostics,diagnostic_categories=T.DIAGNOSTIC_CATEGORIES)
        print(mode,"complete",json.dumps(report),flush=True)
if __name__=="__main__":main()
