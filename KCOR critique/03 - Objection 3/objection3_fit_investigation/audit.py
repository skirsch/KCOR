"""Check retained fit-investigation evidence and frozen production provenance."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np

HERE=Path(__file__).resolve().parent
CAL=HERE.parent/'objection3_corrected_validation'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
p=read(HERE/'population_derivation.json')
s=read(HERE/'saved_dataset_refits_summary.json')
replay=read(HERE/'bootstrap_fit_replay.json')
manifest=read(CAL/'final_2026-11-10/manifest.json')
assert p['source_sha256']==hashlib.sha256((CAL/'source_snapshot/code/KCOR.py').read_bytes()).hexdigest()
assert p['design_sha256']==hashlib.sha256((CAL/'design.json').read_bytes()).hexdigest()
assert manifest['hashes'][str((CAL/'source_snapshot/code/KCOR.py').relative_to(CAL.parents[1]))]==p['source_sha256']
assert s['source_manifest_sha256']==hashlib.sha256((CAL/'final_2026-11-10/manifest.json').read_bytes()).hexdigest()
assert read(CAL/'final_2026-11-10/assessment/completion_audit.json')['checks']=='passed'
assert abs(p['model_truth']-1)<1e-12
assert abs(p['production_population_output']-1.0046145529025252)<1e-12
assert abs(p['oracle_inversion_at_generating_theta']-1)<1e-12
assert abs(p['inversion_at_joint_approximate_theta']-1)<1e-3
assert abs(p['inversion_at_joint_exact_discrete_theta']-1)<1e-12
assert all(a['exact_model_at_generating_parameters_max_hazard_error']<1e-12 for a in p['arms'])
assert all(a['joint_exact_discrete_fit']['max_abs_residual']<1e-12 for a in p['arms'])
assert s['datasets']==600
with (HERE/'saved_dataset_refits.csv').open(newline='',encoding='utf-8') as file:
    rows=list(csv.DictReader(file))
assert len(rows)==600 and len({int(row['repetition']) for row in rows})==600
for key in ('production','joint_approximate','joint_exact_discrete'):
    points=np.asarray([float(row[f'{key}_point']) for row in rows])
    np.testing.assert_allclose(points.mean(),s[key]['mean'],atol=1e-12)
    np.testing.assert_allclose(points.std(ddof=1),s[key]['sd'],atol=1e-12)
    np.testing.assert_allclose(np.sqrt(np.mean((points-1)**2)),s[key]['rmse'],atol=1e-12)
    if key!='production':assert s[key]['successful_both_arms']==600
assert [r['repetition'] for r in replay]==[0,7,171]
assert all(r['bootstrap_draws']==999 and r['max_replay_point_error']==0 for r in replay)
assert replay[0]['percentile_covers_truth'] and not replay[1]['percentile_covers_truth'] and not replay[2]['percentile_covers_truth']
evidence=dict(checks='passed',source_hashes_match=True,exact_population_model_verified=True,
    production_fixed_k_optimum_reproduced=True,saved_original_datasets_recomputed=len(rows),
    bootstrap_replay_datasets=len(replay),bootstrap_estimates_replayed=sum(r['bootstrap_draws'] for r in replay),
    prespecified_coverage_unchanged=True)
with (HERE/'audit.json').open('x',encoding='utf-8') as output:json.dump(evidence,output,indent=2)
print(json.dumps(evidence,indent=2))
