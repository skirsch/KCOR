"""Check nonlinearly consistent fitter through the real ordinary Table 8 workbook."""
from pathlib import Path
import sys,tempfile,subprocess,json
import numpy as np
import replay as r

def pop(data):return [r.table.read_population(r.io.BytesIO(data),s) for s in r.table.SHEETS]
def child(src,dst):
    r.table_init(pop(src.read_bytes()))
    K=r.table.K
    lengths=[len(x[1]) for x in r.table.DATA]
    assert len(set(lengths))==2
    def dispatch(h,*args,**kwargs):
        index=lengths.index(len(h))
        return r.fits[index](h,*args,**kwargs)
    K.OVERRIDE_DOSE_PAIRS=list(r.table.PAIRS)
    K.fit_theta0_gompertz=dispatch
    K.process_workbook(str(src),str(dst))
def main():
    data=subprocess.check_output(["git","cat-file","blob",r.table.INPUT_BLOB],cwd=r.ROOT)
    populations=pop(data);r.table_init(populations)
    expected=np.array([r.table_task((i,0))[2] for i in range(2)])
    with tempfile.TemporaryDirectory(prefix="kcor-nonlinear-fit-") as d:
        folder=Path(d);src=folder/"input.xlsx";dst=folder/"output.xlsx";src.write_bytes(data)
        result=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),str(src),str(dst)],cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(result.stdout.decode(errors="replace")[-6000:])
        actual=np.array(r.table.workbook_points(dst,populations))
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)
        rec=dict(expected=expected.tolist(),actual=actual.tolist(),max_abs_difference=float(np.max(abs(expected-actual))))
        (r.HERE/"workbook-verification.json").write_text(json.dumps(rec,indent=2)+"\n",encoding="utf8")
        print(rec,flush=True)
if __name__=="__main__":
    if len(sys.argv)==3:child(Path(sys.argv[1]),Path(sys.argv[2]))
    else:main()
