"""Exact gamma-frailty survival probabilities and independent person samples."""
from __future__ import annotations
import numpy as np
from production_adapter import configure


def population(scenario):
    cfg = configure(scenario.get('windows', ((6, 55),)))
    weeks = scenario.get('weeks', 100)
    t = np.arange(weeks)
    baseline = scenario.get('k', .001) * np.exp(cfg['gompertz']['gamma_per_week'] * (t-cfg['skip']))
    H0 = np.tile(np.cumsum(baseline), (2, 1))
    effect = scenario.get('effect', 1.)
    effect_window = scenario.get('effect_window', [60, 75])
    multiplier = np.where((t >= effect_window[0]) & (t <= effect_window[1]), effect, 1.)
    H0[1] = np.cumsum(baseline * multiplier)
    probabilities = []
    for d, theta in enumerate(scenario['theta']):
        survival = np.exp(-H0[d]) if theta == 0 else np.exp(-np.log1p(theta * H0[d]) / theta)
        probabilities.append(np.r_[np.r_[1., survival[:-1]] - survival, survival[-1]])
    probs = np.asarray(probabilities)
    anchor = cfg['anchor']
    target = scenario.get('target', 80)
    # The paper's post-skip cumulative baseline contrast, anchored at week 5.
    # Conditioning gamma frailty on survival to the skip boundary changes its
    # mean and variance; its arm-specific mean cancels under normalization.
    post_H0 = H0 - H0[:, cfg['skip']-1, None]
    truth = (post_H0[1, target] / post_H0[0, target]) / (post_H0[1, anchor] / post_H0[0, anchor])
    return probs, float(truth)


def sample_dataset(scenario, rng):
    probs, truth = population(scenario)
    # A random arm label and one lifetime category per independently sampled
    # person. This is exactly the multinomial distribution of those records.
    weights = np.asarray(scenario.get('arm_weights', [.5, .5]))
    cells = (weights[:, None] * probs).ravel()
    counts = rng.multinomial(scenario['n'], cells).reshape(probs.shape)
    return counts, truth


def bootstrap(counts, rng):
    # Pooled person bootstrap, NOT independent weekly Poisson draws and NOT
    # fixed arm sizes. Grouping identical records only compresses their storage.
    n = int(counts.sum())
    return rng.multinomial(n, counts.ravel() / n).reshape(counts.shape)
