"""Arithmetic check of the two distinct constant-scaling operations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def gompertz_cumulative(t, k, gamma):
    return k * np.expm1(gamma * t) / gamma


def observed_hazard(t, k, theta, gamma):
    g = gompertz_cumulative(t, k, gamma)
    return k * np.exp(gamma * t) / (1 + theta * g)


def observed_cumulative(t, k, theta, gamma):
    g = gompertz_cumulative(t, k, gamma)
    return np.log1p(theta * g) / theta if theta else g


def invert(h_obs, theta):
    return np.expm1(theta * h_obs) / theta if theta else h_obs


def main():
    gamma = .085 / 52
    k = .001
    theta = 4.
    t = np.arange(0, 101, dtype=float)
    base_h = observed_hazard(t, k, theta, gamma)
    base_H = observed_cumulative(t, k, theta, gamma)
    cases = []
    for c in (.5, 2.):
        marginal_scaled = c * base_h
        reparameterized = observed_hazard(t, c * k, theta / c, gamma)
        H_scaled = c * base_H
        H_reparameterized = observed_cumulative(t, c * k, theta / c, gamma)
        individual_scaled = observed_hazard(t, c * k, theta, gamma)
        rel_identity = float(np.max(np.abs(reparameterized - marginal_scaled) / marginal_scaled))
        rel_cumulative = float(np.max(np.abs(H_reparameterized[1:] - H_scaled[1:]) / H_scaled[1:]))
        neutral = invert(base_H, theta)
        neutral_scaled = invert(H_scaled, theta / c)
        rel_inversion = float(np.max(np.abs(neutral_scaled[1:] - c * neutral[1:]) / (c * neutral[1:])))
        assert rel_identity < 1e-12 and rel_cumulative < 1e-12 and rel_inversion < 1e-12
        cases.append(dict(multiplier=c, baseline_k_after_observed_scaling=c*k,
            frailty_theta_after_observed_scaling=theta/c,
            frailty_theta_after_individual_scaling=theta,
            max_relative_observed_hazard_identity_error=rel_identity,
            max_relative_cumulative_identity_error=rel_cumulative,
            max_relative_exact_inversion_error=rel_inversion,
            individual_scaled_over_marginal_scaled_week52=float(individual_scaled[52]/marginal_scaled[52])))
    source = REPO / "documentation/preprint/paper.md"
    result = dict(scope="exact Gompertz-gamma identities, not a finite-sample fit audit",
        original_k=k, original_theta=theta, gamma_per_week=gamma,
        time_weeks=[0, 100], cases=cases,
        paper_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    with (HERE / "results.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
