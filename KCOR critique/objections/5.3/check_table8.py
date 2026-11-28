from pathlib import Path
import os,sys,io,json,tempfile,subprocess
import numpy as np
D=Path(os.environ["TEMP"])/"kcor-s4-approved/m105"
r=Path("KCOR critique/objections/5.3").resolve()
os.environ["KCOR_NPH_MODE"]=sys.argv[1]
os.environ["KCOR_BOOTSTRAP_DRAWS"]="0"
os.environ["DATASET"]="Czech"
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
sys.path.insert(0,str(D/"analysis/table8"))
import table8_bootstrap as T
import replay
pop=[T.read_population(io.BytesIO((D/"analysis/table8/input.xlsx").read_bytes()),s) for s in T.SHEETS]
with tempfile.TemporaryDirectory(prefix="kcor-s5-reference-") as scratch:
 source=Path(scratch)/"input.xlsx"; out=Path(scratch)/"result.xlsx"
 source.write_bytes((D/"analysis/table8/input.xlsx").read_bytes())
 subprocess.run([sys.executable,'-c','import replay,sys; from pathlib import Path; replay.write_workbook(Path(sys.argv[1]),Path(sys.argv[2]))',str(source),str(out)],cwd=D/'analysis/table8',check=True)
 values=T.workbook_points(out,pop)
 expected=np.load(D/"analysis/table8"/(sys.argv[1]+"-draws.npz"))["points"][:,0]
 err=float(np.max(np.abs(values-expected)))
 assert err<1e-9
 (r/(sys.argv[1]+"-table8-check.json")).write_text(json.dumps({"mode":sys.argv[1],"max_point_difference":err},indent=2))
 print("max point difference",err)
