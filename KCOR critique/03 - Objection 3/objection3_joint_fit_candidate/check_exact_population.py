"""Fixed deterministic gate before stochastic testing."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'objection3_corrected_validation'))
import numpy as np
from production_adapter import arrays, configure, estimate
from simulation import population
from conditional_fit import weekly_hazard
from candidate_estimator import estimate_candidate


def main():
    design = json.loads((HERE.parent/'objection3_corrected_validation'/'design.json').read_text())
    specific = next(s for s in design['scenarios'] if s['id'] == 'unequal_large')
    scenario = design['common'] | specific
    probabilities, truth = population(scenario)
    weights = np.asarray(scenario['arm_weights'])
    counts = probabilities * weights[:, None] * scenario['n']
    dead, alive = arrays(counts)
    cfg = configure(scenario['windows'])
    first, last = scenario['windows'][0]
    starts = ((.1,), (1.,), (5.,), (.1, 1., 5.))
    candidate = [estimate_candidate(counts, scenario['windows'], scenario['target'], s) for s in starts]
    original = estimate(counts, scenario['windows'], scenario['target'])
    checks = []
    for result in candidate:
        for arm in (0, 1):
            fit = result['diagnostics'][arm]
            h, _, _ = weekly_hazard(fit['k_effective'], fit['theta'],
                np.exp(cfg['gompertz']['gamma_per_week']*np.arange(last-first+1)))
            observed = -np.log1p(-dead[arm, first:last+1]/alive[arm, first:last+1])
            residual = float(np.max(np.abs(h-observed)))
            checks.append(dict(theta=fit['theta'], expected_theta=scenario['theta'][arm],
                max_hazard_error=residual, success=fit['success'], objective=fit['objective']))
            assert fit['success'] and abs(fit['theta']-scenario['theta'][arm]) < 2e-3
            assert residual < 1e-8
        assert abs(result['point']-truth) < 1e-8
    out = HERE/'exact_population'
    out.mkdir(exist_ok=False)
    payload = dict(scenario=scenario, truth=truth, candidate=candidate,
        original_point=original['point'], checks=checks,
        code_sha256=hashlib.sha256((HERE/'conditional_fit.py').read_bytes()).hexdigest(),
        passed=True)
    (out/'verification.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(dict(passed=True, truth=truth, original_point=original['point'],
        candidate_points=[r['point'] for r in candidate], checks=checks), indent=2))


if __name__ == '__main__':
    main()
