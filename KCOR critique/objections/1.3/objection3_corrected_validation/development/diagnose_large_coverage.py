"""Post-run diagnosis of the prespecified large-population coverage failure."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
run = HERE/'final_2026-11-10/unequal_large'
files = sorted(run.glob('[0-9][0-9][0-9][0-9].json'))
assert len(files) == 600
rows = [json.loads(path.read_text(encoding='utf-8')) for path in files]
truth = np.asarray([r['truth'] for r in rows])
assert np.all(truth == 1)
point = np.asarray([r['point'] for r in rows])
ci = np.asarray([r['intervals']['percentile'] for r in rows])
basic = np.asarray([r['intervals']['basic'] for r in rows])
prefix = np.asarray([r['prefix_intervals']['percentile'] for r in rows])
theta = np.asarray([[d['theta'] for d in r['diagnostics']] for r in rows])
bootstrap_means = []
bootstrap_medians = []
reconstructed = []
for path,r in zip(files,rows):
    with np.load(path.with_suffix('.npz')) as saved:
        draws=saved['draws']
        assert len(draws)==999 and np.isfinite(draws).all()
        assert int(saved['counts'].sum())==1000000
    bootstrap_means.append(float(np.mean(draws)))
    bootstrap_medians.append(float(np.median(draws)))
    q=np.quantile(draws,[.025,.975])
    assert np.allclose(q,r['intervals']['percentile'],rtol=0,atol=1e-12)
    reconstructed.append(float(np.max(np.abs(q-ci[len(reconstructed)]))))
bootstrap_means=np.asarray(bootstrap_means)
bootstrap_medians=np.asarray(bootstrap_medians)
miss_low=ci[:,1]<1
miss_high=ci[:,0]>1
miss=miss_low|miss_high
basic_miss=(basic[:,1]<1)|(basic[:,0]>1)
population_target=1.0046145529025252
pop_covered=(ci[:,0]<=population_target)&(population_target<=ci[:,1])
width=ci[:,1]-ci[:,0]
center=ci.mean(axis=1)
shift=bootstrap_medians-point
def quantiles(x):
    return np.quantile(x,[0,.025,.25,.5,.75,.975,1]).tolist()
recentered=ci-shift[:,None]
popbias=population_target-1
corrected_popbias=ci-popbias
result=dict(n=len(rows), truth=1., fitted_population_output=population_target,
    population_output_minus_truth=popbias,
    point_mean=float(point.mean()), point_sd=float(point.std(ddof=1)),
    point_quantiles=quantiles(point), point_minus_truth_mean=float(np.mean(point-1)),
    theta_means=theta.mean(axis=0).tolist(),theta_medians=np.median(theta,axis=0).tolist(),
    bootstrap_mean_minus_point_mean=float(np.mean(bootstrap_means-point)),
    bootstrap_median_minus_point_mean=float(np.mean(shift)),
    bootstrap_median_above_point=int(np.sum(shift>0)),
    bootstrap_median_minus_point_quantiles=quantiles(shift),
    percentile_center_minus_point_mean=float(np.mean(center-point)),
    interval_width_mean=float(np.mean(width)),interval_width_quantiles=quantiles(width),
    percentile_coverage=int(np.sum(~miss)), percentile_miss_below_truth=int(np.sum(miss_low)),
    percentile_miss_above_truth=int(np.sum(miss_high)),
    miss_gap_quantiles=quantiles(np.where(miss_high,ci[:,0]-1,np.where(miss_low,1-ci[:,1],0))[miss]),
    basic_coverage=int(np.sum(~basic_miss)),
    percentile_coverage_fitted_population=int(np.sum(pop_covered)),
    primary_misses_rescued_by_fitted_target=int(np.sum(miss&pop_covered)),
    percentile_coverage_after_subtracting_population_bias=int(np.sum((corrected_popbias[:,0]<=1)&(1<=corrected_popbias[:,1]))),
    percentile_coverage_after_recentering_bootstrap_median=int(np.sum((recentered[:,0]<=1)&(1<=recentered[:,1]))),
    prefix_coverage=int(np.sum((prefix[:,0]<=1)&(1<=prefix[:,1]))),
    all_bootstrap_estimates_finite=True,all_original_accepted=all(r['accepted'] for r in rows),
    all_bootstrap_fits_successful=all(r['bootstrap_fit_failures']==0 for r in rows),
    all_saved_intervals_reconstructed=True,max_endpoint_reconstruction_error=max(reconstructed))
(HERE/'development/diagnosis_unequal_large.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
