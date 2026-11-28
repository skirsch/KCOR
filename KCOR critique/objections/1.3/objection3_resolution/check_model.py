"""Checks the analytic reference and that outer datasets really vary."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
from production_adapter import K, arrays, configure, estimate
from simulation import population, sample_dataset


def main():
    here = Path(__file__).parent
    design = json.loads((here/'design.json').read_text())
    results = []
    for si, specific in enumerate(design['scenarios']):
        scenario = design['common'] | specific
        cfg = configure(scenario['windows'])
        probs, truth = population(scenario)
        np.testing.assert_allclose(probs.sum(axis=1), 1., atol=1e-14)
        assert np.all(probs >= 0)
        dead, alive = arrays(probs)
        Hobs = np.cumsum(np.where(np.arange(dead.shape[1]) >= cfg['skip'],
                                 -np.log1p(-dead/alive), 0.), axis=1)
        H = np.asarray([K.invert_gamma_frailty(Hobs[d], scenario['theta'][d]) for d in (0, 1)])
        reference = (H[1,scenario['target']]/H[0,scenario['target']])/(H[1,cfg['anchor']]/H[0,cfg['anchor']])
        np.testing.assert_allclose(reference, truth, atol=1e-12, rtol=1e-12)
        samples = [sample_dataset(scenario, np.random.default_rng(np.random.SeedSequence([415004,si,j])))[0]
                   for j in range(10)]
        assert len({c.tobytes() for c in samples}) == 10
        expected_counts = probs * np.asarray(scenario['arm_weights'])[:,None] * scenario['n']
        args = (expected_counts, scenario['windows'], scenario['target'])
        single = estimate(*args)
        production = estimate(*args, total_iterations=design['bootstrap_draws']+1)
        results.append(dict(scenario=scenario['id'], truth=truth,
                            exact_inversion_reference=float(reference),
                            single_population_point=single['point'],
                            production_population_point=production['point'],
                            independent_dataset_hashes=10))
    out = Path(sys.argv[1])
    with out.open('x', encoding='utf-8') as f:
        json.dump(dict(checks='passed', scenarios=results), f, indent=2)
    print('Analytic reference and independent outer samples: passed for all scenarios')


if __name__ == '__main__':
    main()
