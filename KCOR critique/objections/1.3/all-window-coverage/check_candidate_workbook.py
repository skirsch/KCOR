"""Compare candidate fast replay against actual MC workbook processing."""
from pathlib import Path
import sys,tempfile,subprocess,json
import numpy as np
import pandas as pd
import replay
import study
import importlib.util
spec=importlib.util.spec_from_file_location("coverage_workbook_check",replay.HERE.parent/"multiwindow-coverage/check_workbook.py")
check=importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
from experiment import candidate

def configure(scenario,stress):
    if stress:
        study.POPULATION=50000;study.WINDOWS=((2,5),(35,52));study.SEED=20261126
    else:
        study.POPULATION=1000000;study.WINDOWS=((2,26),(35,52));study.SEED=20261124
    study.population.cache_clear()
    model=check.configure(scenario,"joint",stress)
    model.fit_theta0_gompertz=study.compare.production_fit
    study.compare.joint_fit=candidate(model,"multistart")
    model.fit_theta0_gompertz=study.compare.joint_fit
    return model

def main():
    results=[]
    with tempfile.TemporaryDirectory(prefix="kcor-candidate-workbook-") as tmp:
        for label,scenario,rep,stress in (("no_wave","no_wave",0,False),("nonlinear_wave","nonlinear_wave",0,False),("stress","nonlinear_wave",34,True)):
            replay.initialize()
            configure(scenario,stress)
            # Rebuild namespace of fast estimator after configuration.
            replay.initialize()
            batch,truth=study.samples(scenario,rep,5)
            weights=study.pooling_weights(batch)
            expected=np.asarray([replay.estimate_one(c,weights,scenario)[0] for c in batch])
            folder=Path(tmp)/label;folder.mkdir()
            source=folder/"input.xlsx";output=folder/"output.xlsx"
            check.frame(batch).to_excel(source,sheet_name="2021_24",index=False)
            args=[sys.executable,"-B",str(Path(__file__).resolve()),str(source),str(output),scenario,str(int(stress))]
            result=subprocess.run(args,cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            if result.returncode:raise RuntimeError(result.stdout.decode(errors="replace")[-6000:])
            book=pd.read_excel(output,sheet_name="2021_24")
            final=book[(book.Date==study.DATES[-1])&(book.Dose_num==1)&(book.Dose_den==0)]
            maxdiff=0.
            for col,yob in enumerate((0,*study.STRATA)):
                rows=final[final.YearOfBirth==yob].sort_values("mc_id")
                np.testing.assert_array_equal(rows.mc_id,np.arange(len(batch)))
                np.testing.assert_allclose(rows.KCOR,expected[:,col],rtol=1e-10,atol=1e-12)
                maxdiff=max(maxdiff,float(np.max(abs(rows.KCOR-expected[:,col]))))
            log=result.stdout.decode(errors="replace").split("SUMMARY STATISTICS")[-1]
            for col,yob in enumerate(study.STRATA,1):
                lo,hi=np.quantile(expected[1:,col],[.025,.975])
                assert f"Percentile 95% CI: [{lo:.4f}, {hi:.4f}]" in log,log[-6000:]
            results.append(dict(scenario=label,rep=rep,original_and_draws=len(batch),max_abs_difference=maxdiff))
            print(label,"actual MC workbook matches",maxdiff,flush=True)
    (replay.HERE/"workbook-verification.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf8")
if __name__=="__main__":
    if len(sys.argv)==5:
        _,source,output,scenario,stress=sys.argv
        configure(scenario,bool(int(stress))).process_workbook(source,output)
    else: main()


