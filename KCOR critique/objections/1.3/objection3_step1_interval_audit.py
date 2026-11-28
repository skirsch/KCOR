"""Read existing interval evidence and reproduce one retained coverage row.

No new coverage campaign or changes to production/validation code. The small
seed comparison checks whether existing scenario generators vary their data.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir',type=Path,required=True)
    args=ap.parse_args()
    sources=[ROOT/p for p in [
        'code/KCOR.py','code/KCOR_CMR.py','code/Makefile',
        'test/sim_grid/code/compute_bootstrap_coverage.py',
        'test/sim_grid/code/generate_sim_grid.py',
        'test/sim_grid/out/bootstrap_coverage.csv',
        'test/sim_grid/out/bootstrap_coverage_replicates.csv',
        'test/sim_grid/out/bootstrap_coverage_run.json',
        'KCOR critique/original-review/KCOR.pdf','KCOR critique/objections/1.1/step4_results_2026-11-09/manuscript_extracted.txt']]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    # Legacy source uses the removed NumPy trapz name for a separate survival
    # summary. Supply its trapezoid equivalent in this process only.
    compatibility_alias=not hasattr(np,'trapz')
    if compatibility_alias:
        np.trapz=np.trapezoid
    args.outdir.mkdir(parents=True,exist_ok=False)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_hashes=hashes,
                  script_sha256=sha(Path(__file__)),numpy_version=np.__version__,in_process_trapz_alias=compatibility_alias,
                  scope='Read-only inventory, three-seed generator comparison, and replay of one saved 200-bootstrap-draw row; not a new calibration campaign.')
    (args.outdir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT/'test/sim_grid/code'))
    grid=importlib.import_module('generate_sim_grid')
    coverage=importlib.import_module('compute_bootstrap_coverage')
    saved=pd.read_csv(ROOT/'test/sim_grid/out/bootstrap_coverage_replicates.csv',float_precision='round_trip')
    summary=pd.read_csv(ROOT/'test/sim_grid/out/bootstrap_coverage.csv',float_precision='round_trip')
    run=json.loads((ROOT/'test/sim_grid/out/bootstrap_coverage_run.json').read_text())
    inventory=[]
    for name,g in saved.groupby('scenario_name',sort=False):
        old=summary[summary.scenario==name].iloc[0]
        assert len(g)==old.n_valid
        assert not g.simulation_id.duplicated().any()
        assert (g.covered==((g.ci_lower<=g.true_kcor)&(g.true_kcor<=g.ci_upper))).all()
        np.testing.assert_allclose(g.covered.mean(),old.coverage_proportion,rtol=1e-12)
        inventory.append(dict(scenario=name,rows=len(g),distinct_point_estimates=int(g.point_estimate.nunique()),
            point_estimate_min=float(g.point_estimate.min()),point_estimate_max=float(g.point_estimate.max()),
            distinct_interval_lower_bounds=int(g.ci_lower.nunique()),reported_coverage_percent=float(old.coverage_percent),
            point_equals_reference_count=int((g.point_estimate==g.true_kcor).sum())))
    inv=pd.DataFrame(inventory); inv.to_csv(args.outdir/'saved_run_inventory.csv',index=False)

    comparisons=[]
    for ix,name in [(0,'Gamma-frailty null'),(1,'Injected effect (harm)'),(2,'Injected effect (benefit)'),(4,'Sparse events')]:
        generator=coverage.SCENARIO_RUNNERS[ix]
        previous=None; points=[]
        for seed in [0,42,43]:
            cfg=grid.SimConfig(seed=seed)
            scenario=generator(cfg)
            arrays=[np.asarray(c[key]) for c in scenario['cohorts'] for key in ['weeks','alive','dead']]
            if previous is not None:
                for a,b in zip(arrays,previous): np.testing.assert_array_equal(a,b)
            previous=arrays
            result=grid.compute_kcor_for_scenario(scenario,cfg,include_cox=False)
            point=float(result['kcor_trajectory'][int(run['target_week'])]); points.append(point)
            np.testing.assert_allclose(point,summary[summary.scenario==name].iloc[0].median_point_estimate,rtol=1e-6,atol=1e-8)
        assert len(set(points))==1
        comparisons.append(dict(scenario=name,seeds=[0,42,43],all_count_arrays_identical=True,all_point_estimates_identical=True,point_estimate=points[0]))
    (args.outdir/'seed_comparison.json').write_text(json.dumps(comparisons,indent=2),encoding='utf-8')

    old=saved[(saved.scenario_name=='Gamma-frailty null')&(saved.simulation_id==1)].iloc[0]
    replay=coverage._run_one_simulation_combined(scenario_runner_ix=0,config=grid.SimConfig(),sim_index=0,
        base_seed=int(run['base_mc_seed']),n_bootstrap=int(run['n_bootstrap']),target_week=int(run['target_week']),
        true_kcor=float(old.true_kcor),scenario_name='Gamma-frailty null',skip_theta=True,max_week_index=None,
        debug_kcor_call_count=True,min_boot_ok=min(int(run['n_bootstrap']),max(20,int(.5*run['n_bootstrap']))))
    assert replay['status']=='ok'
    r=replay['kcor_row']; checks=[]
    for key in ['point_estimate','ci_lower','ci_upper','ci_width','n_boot_valid','n_boot_failed']:
        np.testing.assert_allclose(r[key],old[key],rtol=1e-6,atol=1e-8)
        checks.append(dict(field=key,saved=float(old[key]),replayed=float(r[key]),absolute_difference=abs(float(r[key])-float(old[key]))))
    assert r['covered']==bool(old.covered)
    pd.DataFrame(checks).to_csv(args.outdir/'saved_row_replay.csv',index=False)
    for name,h in hashes.items(): assert sha(ROOT/name)==h,name
    (args.outdir/'verification.json').write_text(json.dumps(dict(saved_rows_checked=len(saved),generator_cases_checked=len(comparisons),seeds_per_case=3,
        saved_rows_replayed=1,bootstrap_draws_in_replay=int(run['n_bootstrap']),source_hashes_unchanged=True,
        limitation='One saved row replay establishes local numerical reproduction, not the complete historical execution environment or correct coverage.'),indent=2),encoding='utf-8')
    print(inv.to_string(index=False))
    print('Verified four deterministic outer generators across three seeds; reproduced one retained coverage row; sources unchanged.')


if __name__=='__main__': main()
