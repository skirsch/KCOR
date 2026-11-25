"""Verify combined candidate and weekly survival through ordinary workbook entry."""
from pathlib import Path
import subprocess,sys,tempfile,json
import numpy as np
import replay as r
def populations(data):
    return [r.table.read_population(r.io.BytesIO(data),s) for s in r.table.SHEETS]
def child(source,out):
    r.init_table(populations(source.read_bytes()))
    r.table.K.OVERRIDE_DOSE_PAIRS=list(r.table.PAIRS)
    r.table.K.fit_theta0_gompertz=r.table.compare.joint_fit
    r.table.K.process_workbook(str(source),str(out))
def main():
    data=subprocess.check_output(["git","cat-file","blob",r.table.INPUT_BLOB],cwd=r.ROOT)
    pop=populations(data)
    r.init_table(pop)
    expected=np.array([r.table.estimate(i,0)[0] for i in range(2)])
    with tempfile.TemporaryDirectory(prefix="kcor-combined-wave-") as tmp:
        folder=Path(tmp);source=folder/"source.xlsx";out=folder/"combined.xlsx";source.write_bytes(data)
        childproc=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),str(source),str(out)],cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if childproc.returncode:raise RuntimeError(childproc.stdout.decode(errors="replace")[-6000:])
        actual=np.array(r.table.workbook_points(out,pop))
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)
        result=dict(expected=expected.tolist(),actual=actual.tolist(),max_abs_difference=float(np.max(abs(expected-actual))))
        (r.HERE/"workbook-verification.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
        print(result,flush=True)
if __name__=="__main__":
    if len(sys.argv)==3:child(Path(sys.argv[1]),Path(sys.argv[2]))
    else:main()
