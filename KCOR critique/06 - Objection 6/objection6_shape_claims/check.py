"""Check the two cumulative-hazard shape statements under the paper's model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main() -> None:
    k = 0.001
    gamma = 0.085 / 52
    t = np.linspace(0, 52, 521)
    base = k * np.expm1(gamma * t) / gamma
    cases = []
    for theta in (0.0, gamma / k, 4.0):
        observed = base if theta == 0 else np.log1p(theta * base) / theta
        hazard = k * np.exp(gamma * t) / (1 + theta * base)
        predicted_curvature_sign = np.sign(gamma - theta * k)
        cases.append({
            "theta": float(theta),
            "curvature_sign": int(predicted_curvature_sign),
            "observed_hazard_start": float(hazard[0]),
            "observed_hazard_end": float(hazard[-1]),
            "observed_midpoint_deviation_from_chord": float(observed[260] - (observed[0] + observed[-1]) / 2),
        })
    midpoint = base[260]
    chord_midpoint = (base[0] + base[-1]) / 2
    assert cases[0]["observed_midpoint_deviation_from_chord"] < 0
    assert abs(cases[1]["observed_midpoint_deviation_from_chord"]) < 1e-14
    assert np.allclose(base, k * np.expm1(gamma * t) / gamma)
    report = {
        "model": "h0(t)=k exp(gamma t), Hobs=log(1+theta H0)/theta",
        "k": k,
        "gamma_per_week": gamma,
        "curvature_identity": "Hobs''(t) = k exp(gamma t) (gamma-theta*k)/(1+theta*H0(t))^2",
        "cases": cases,
        "gompertz_baseline_52_week_midpoint_deviation_from_chord": float(midpoint - chord_midpoint),
        "gompertz_baseline_52_week_midpoint_deviation_fraction_of_end_cumulative_hazard": float((midpoint - chord_midpoint) / base[-1]),
        "baseline_hazard_increase_52_weeks": float(np.exp(gamma * 52) - 1),
    }
    (HERE / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
