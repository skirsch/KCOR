"""Compare fourth- and fifth-week anchors on the published comorbidity sweep."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import t, ttest_ind

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2.7"))
from comorbidity_sweep import N, simulate, study


def replay_level(level):
    # Observe the same fitted curves used by the original simulation. It still
    # computes and records its own fifth-week endpoint in the usual way.
    historical_anchor = study.anchored_kcor
    fourth_endpoints = []

    def compare(a, b, skip, *, workbook_anchor=False):
        assert not workbook_anchor
        historical = historical_anchor(a, b, skip)
        fourth = skip + study.K.KCOR_NORMALIZATION_WEEKS - 1
        fifth = skip + 4
        assert (fourth, fifth) == (5, 6)
        assert np.all(np.isfinite(historical[[fourth, fifth]]))
        assert historical[fourth] > 0 and historical[fifth] == 1
        # Dividing by the fourth week's value cancels the fifth-week anchor.
        fourth_endpoints.append(study.kcor_asymptote_value(historical / historical[fourth], b))
        return historical

    with patch.object(study, "anchored_kcor", compare):
        runs = simulate(level)
    np.testing.assert_array_equal(runs.rep, np.arange(N))
    pairs = runs[["epsilon_comorb", "rep", "KCOR_asymptote"]].rename(columns={"KCOR_asymptote": "fifth"})
    pairs["fourth"] = fourth_endpoints
    pairs["change"] = pairs.fourth - pairs.fifth
    assert np.isfinite(pairs.to_numpy()).all()
    return pairs


def summarize(pairs):
    zero = pairs.loc[pairs.epsilon_comorb == 0]
    rows = []
    for level, group in pairs.groupby("epsilon_comorb"):
        for measure in ("fourth", "fifth", "change"):
            values = group[measure]
            mean = values.mean()
            margin = t.ppf(.975, len(values) - 1) * values.sem()
            row = dict(added_weekly_hazard=level, pairs=len(values), measure=measure,
                       mean=mean, lower95=mean-margin, upper95=mean+margin)
            if level:
                # The two anchors are paired within a dataset. The datasets
                # at different hazard levels use independent random streams.
                ci = ttest_ind(values, zero[measure], equal_var=False).confidence_interval()
                row.update(contrast=mean-zero[measure].mean(), contrast_lower95=ci.low,
                           contrast_upper95=ci.high)
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=5) as pool:
        pairs = pd.concat(pool.map(replay_level, study.EPSILON_COMORB), ignore_index=True)
    keys = ["epsilon_comorb", "rep"]
    saved = pd.read_csv(HERE.parent / "2.7/comorbidity_200_runs.csv").set_index(keys).sort_index()
    fifth = pairs.set_index(keys).sort_index().fifth
    pd.testing.assert_index_equal(fifth.index, saved.index)
    np.testing.assert_allclose(fifth, saved.KCOR_asymptote, atol=1e-8, rtol=0)
    print("Maximum absolute difference from archived endpoints:", (fifth-saved.KCOR_asymptote).abs().max())
    summary = summarize(pairs)
    pairs.to_csv(HERE / "anchor_week_pairs.csv", index=False)
    summary.to_csv(HERE / "anchor_week_summary.csv", index=False)
    print(summary.to_string(index=False))
