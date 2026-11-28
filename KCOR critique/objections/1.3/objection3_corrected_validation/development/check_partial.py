"""Read-only diagnostic of currently completed final records; never a final verdict."""
from pathlib import Path
import json
import numpy as np

HERE = Path(__file__).resolve().parents[1]
run = HERE/'final_2026-11-10'
design = json.loads((HERE/'design.json').read_text())
summary = {}
for scenario in design['scenarios']:
    sid = scenario['id']
    rows = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((run/sid).glob('[0-9][0-9][0-9][0-9].json'))]
    if not rows:
        continue
    truth = np.array([r['truth'] for r in rows])
    point = np.array([r['point'] for r in rows])
    ci = np.array([r['intervals']['percentile'] for r in rows])
    basic = np.array([r['intervals']['basic'] for r in rows])
    accepted = np.array([r['accepted'] for r in rows])
    fit_success = np.array([all(d['success'] for d in r['diagnostics']) for r in rows])
    valid = np.isfinite(ci).all(axis=1) & (ci[:,0] <= ci[:,1])
    covered = valid & (ci[:,0] <= truth) & (truth <= ci[:,1])
    basic_covered = np.isfinite(basic).all(axis=1) & (basic[:,0] <= truth) & (truth <= basic[:,1])
    statuses = {}
    for r in rows:
        for d in r['diagnostics']:
            statuses[d['status']] = statuses.get(d['status'],0)+1
    checks = []
    for p in sorted((run/sid).glob('[0-9][0-9][0-9][0-9].json'))[:5]:
        r = json.loads(p.read_text(encoding='utf-8'))
        with np.load(p.with_suffix('.npz')) as data:
            draws = data['draws']
            counts = data['counts']
        assert len(draws) == design['bootstrap_draws']
        assert int(counts.sum()) == scenario['n']
        assert np.isfinite(draws).sum() > 0
        q = np.quantile(draws[np.isfinite(draws)], [.025,.975])
        assert np.allclose(q, r['intervals']['percentile'], rtol=0, atol=1e-12)
        checks.append(p.name)
    summary[sid] = dict(completed=len(rows), target=float(truth[0]),
        point_mean=float(np.mean(point)), point_sd=float(np.std(point,ddof=1)) if len(rows)>1 else None,
        percentile_coverage_so_far=float(covered.mean()),
        basic_coverage_so_far=float(basic_covered.mean()),
        mean_percentile_width=float(np.mean(ci[:,1]-ci[:,0])),
        intervals_valid=int(valid.sum()), original_fits_successful=int(fit_success.sum()),
        accepted=int(accepted.sum()), fit_statuses=statuses,
        bootstrap_fit_failures=sum(r['bootstrap_fit_failures'] for r in rows),
        bootstrap_draws=len(rows)*design['bootstrap_draws'],
        interval_reconstructions_checked=checks)
print(json.dumps(summary,indent=2))
