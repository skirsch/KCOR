"""Post-run target sensitivity; never replaces the prespecified model reference."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    model = json.loads((HERE/'development/model_checks.json').read_text())
    rows = pd.read_csv(HERE/'final_2026-11-10/assessment/replicates.csv')
    results = []
    for case in model['scenarios']:
        group = rows[rows.scenario == case['scenario']]
        for key in ('truth', 'single_population_point', 'production_population_point'):
            target = case[key]
            for kind in ('percentile', 'basic'):
                lo, hi = group[kind+'_low'], group[kind+'_high']
                valid = np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
                covered = valid & (lo <= target) & (target <= hi)
                results.append(dict(scenario=case['scenario'], target_kind=key, target=target,
                                    interval=kind, coverage=float(covered.mean()), repetitions=len(group)))
    out = HERE/'final_2026-11-10/assessment/target_sensitivity.json'
    with out.open('x', encoding='utf-8') as f:
        json.dump(dict(purpose='Post-run diagnosis using population outputs recorded during development; not a new calibration target or a tuned procedure.',
                       results=results), f, indent=2)
    for row in results:
        if row['scenario'] == 'unequal_large':
            print(row)


if __name__ == '__main__':
    main()
