"""Run Table 9 on the archived workbook with the all-window joint fitter."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CODE = ROOT / "code"
INPUT = ROOT / "test/positive_control/data/KCOR_positive_control.xlsx"
SCENARIOS = ("benefit", "harm")
SEED = 33080
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.update(DATASET="positive_control", MONTE_CARLO="0", SA_DOSE_PAIRS="1,0",
                  KCOR_RISK_HISTORY_POLICY="allow_exits",
                  PYTHONIOENCODING="utf-8")
sys.path.insert(0, str(CODE))
import numpy as np
import pandas as pd
import KCOR as model


def workbook(scenario, output):
    # The scenario name belongs to the run; the sheet name identifies enrollment.
    frame = pd.read_excel(INPUT, sheet_name="2021_24_"+scenario)
    source = output.with_name("input.xlsx")
    frame.to_excel(source, sheet_name="2021_24", index=False)
    os.chdir(output.parent)
    model.OVERRIDE_DOSE_PAIRS = [(1,0)]
    fit = model.fit_theta0_gompertz
    calls=[]
    def joint_fit(*args, **kwargs):
        (k,theta),diag=fit(*args,**kwargs,joint_seed=True)
        calls.append(dict(k=float(k),theta=float(theta),status=diag["status"]))
        return (k,theta),diag
    model.fit_theta0_gompertz = joint_fit
    model.process_workbook(str(source), str(output))
    output.with_suffix(".json").write_text(json.dumps(calls),encoding="utf8")


def read_data():
    data = []
    for scenario in SCENARIOS:
        df = pd.read_excel(INPUT, sheet_name="2021_24_"+scenario, parse_dates=["DateDied"])
        weekly = df.groupby(["Dose","DateDied"])[["Alive","Dead"]].sum()
        dates = weekly.loc[0].index
        assert dates.equals(weekly.loc[1].index) and len(dates) == 150
        assert (dates.to_series().diff().dropna() == pd.Timedelta(weeks=1)).all()
        alive, dead = [weekly[c].to_numpy().reshape(2,-1) for c in ("Alive","Dead")]
        # Counts were rounded separately in the generator. Treat the residual loss
        # after each week's deaths as a nondeath exit before the following week.
        exits = alive[:,:-1] - dead[:,:-1] - alive[:,1:]
        cells = np.column_stack([dead, exits, alive[:,-1]-dead[:,-1]])
        assert cells.min() >= 0
        np.testing.assert_array_equal(cells.sum(axis=1), alive[:,0])
        data.append((alive,dead,cells))
    np.testing.assert_array_equal(data[0][2][0],data[1][2][0])
    return dates, data


def initialize(dates, data):
    global DATA, WEEKS, QUIET, CONFIG, GUARD
    DATA = data
    WEEKS = np.arange(len(dates))
    QUIET = model._build_theta_quiet_mask(dates.strftime("%G%V").astype(int).to_numpy(), WEEKS)[0]
    CONFIG = model.get_gompertz_theta_fit_config()
    GUARD = model.get_theta_degenerate_fit_config()
    assert not (model.get_covid_correction_config() or {}).get("apply_correction")


def estimates(draw):
    results=[]
    for s,(original_alive,original_dead,cells) in enumerate(DATA):
        curves=[]
        diagnostics=[]
        for d in (0,1):
            if draw:
                # The control data are identical in both workbooks. Use the same
                # control draw, and independent treated draws for the two scenarios.
                rng=np.random.default_rng([SEED,draw,0 if d==0 else s+1])
                sample=rng.multinomial(cells[d].sum(),cells[d]/cells[d].sum())
                dead, exits = sample[:len(WEEKS)],sample[len(WEEKS):-1]
                alive=cells[d].sum()-np.r_[0,np.cumsum(dead[:-1]+exits)]
            else:
                alive,dead=original_alive[d],original_dead[d]
            hazard=model.hazard_from_mr(dead/(alive+model.EPS))
            skip=model.DYNAMIC_HVE_SKIP_WEEKS
            effective=np.where(WEEKS>=skip,hazard,0.)
            (rate,theta),diag=model.fit_theta0_gompertz(effective,WEEKS-skip,QUIET,
                CONFIG["k_anchor_weeks"],CONFIG["gamma_per_week"],deaths_arr=dead,joint_seed=True)
            applied=theta if diag["success"] and np.isfinite(theta) and theta>=0 else 0.
            applied,_,_=model._apply_theta_degenerate_guard(applied,GUARD)
            total=effective.cumsum()
            historical=(-np.log1p(-dead/alive)).cumsum()
            adjusted=model.invert_gamma_frailty(total,applied)
            anchor=model.KCOR_NORMALIZATION_WEEKS_EFFECTIVE
            curves.append([historical[80]/historical[4],total[80]/total[anchor],
                           adjusted[80]/adjusted[anchor]])
            diagnostics.append(dict(k=float(rate), theta=float(applied), raw=float(diag["theta0_raw"]),
                status=diag["status"],success=bool(diag["success"]),flag=bool(diag["insufficient_signal"]),
                upper_bound=bool(diag["theta_hit_bound"]),gap_first_pass=diag.get("gap_first_pass"),gap_seed_status=diag.get("gap_seed_status","not_assessed")))
        results.append((np.divide(curves[1],curves[0]),diagnostics))
    return results


def ordinary_points(dates, original):
    points={}
    with TemporaryDirectory(prefix="kcor-table9-") as tmp:
        for s in SCENARIOS:
            directory=Path(tmp)/s
            directory.mkdir()
            path=directory/"fitted.xlsx"
            result=subprocess.run([sys.executable,"-B",str(Path(__file__).resolve()),"--workbook",s,str(path)],
                                  cwd=directory,capture_output=True)
            if result.returncode:
                raise RuntimeError(result.stderr.decode(errors="replace")[-3000:]+
                                   result.stdout.decode(errors="replace")[-1000:])
            df=pd.read_excel(path,sheet_name="dose_pairs")
            selected=df[(pd.to_datetime(df.Date)==dates[80]) & (df.YearOfBirth==1950) &
                        (df.Dose_num==1) & (df.Dose_den==0)]
            points[s]=float(selected.KCOR.item())
            calls=json.loads(path.with_suffix(".json").read_text(encoding="utf8"))
            values,diags=original[SCENARIOS.index(s)]
            np.testing.assert_allclose(points[s],values[-1],rtol=1e-10)
            for diag in diags:
                assert any(c["status"]==diag["status"] and
                           np.allclose([c["k"],c["theta"]],[diag["k"],diag["theta"]],rtol=1e-10,atol=1e-12)
                           for c in calls)
            shutil.copyfile(path,HERE/f"fitted-{s}.xlsx")
    return points


def main():
    dates,data=read_data()
    initialize(dates,data)
    original=estimates(0)
    ordinary=ordinary_points(dates,original)
    for s,(values,diag) in zip(SCENARIOS,original):
        print(s,values,diag,"workbook",ordinary[s],flush=True)
        np.testing.assert_allclose(ordinary[s],values[-1],rtol=1e-10)
    assert [round(item[0][0],3) for item in original] == [.825,1.107]
    with ProcessPoolExecutor(4,initializer=initialize,initargs=(dates,data)) as pool:
        draws=list(pool.map(estimates,range(1,1000),chunksize=8))
    report=dict(draws=999,seed=SEED,workbook_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
                method_sha256=hashlib.sha256((CODE/"KCOR.py").read_bytes()).hexdigest(),
                anchor=int(model.KCOR_NORMALIZATION_WEEKS_EFFECTIVE),skip=model.DYNAMIC_HVE_SKIP_WEEKS,
                quiet_indices=WEEKS[QUIET].tolist(),gamma=CONFIG["gamma_per_week"],week80=str(dates[80].date()),
                scenarios={})
    for i,s in enumerate(SCENARIOS):
        vals,diag=original[i]
        array=np.array([r[i][0] for r in draws])
        assert np.isfinite(array).all()
        record=dict(workbook=ordinary[s],fits=diag,rounding_exits=data[i][2][:,150:-1].sum(axis=1).tolist())
        for j,key in enumerate(("historical_raw","matched_raw","fitted")):
            record[key]=[vals[j],*np.quantile(array[:,j],[.025,.975])]
        for d in (0,1):
            sample=[r[i][1][d] for r in draws]
            diag[d]["theta_interval"]=np.quantile([x["theta"] for x in sample],[.025,.975]).tolist()
            diag[d]["k_interval"]=np.quantile([x["k"] for x in sample],[.025,.975]).tolist()
            diag[d]["flagged_draws"]=sum(x["flag"] for x in sample)
            diag[d]["unsuccessful_draws"]=sum(not x["success"] for x in sample)
        report["scenarios"][s]=record
    (HERE/"results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--workbook":
        workbook(sys.argv[2],Path(sys.argv[3]).resolve())
    else:
        main()
