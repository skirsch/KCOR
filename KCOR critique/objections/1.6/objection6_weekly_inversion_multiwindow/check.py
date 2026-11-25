"""Compare the frozen KCOR fitter with an in-memory exact-weekly variant."""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parents[1] / "03 - Objection 3" / "objection3_corrected_validation"
sys.path.insert(0, str(VALIDATION))
from production_adapter import K, arrays, configure, estimate  # noqa: E402
from simulation import population  # noqa: E402


def exact_weekly_variant():
    original = K.fit_theta0_gompertz
    source = inspect.getsource(original)
    changes = (
        (
            "h0_i = hi * (1.0 + theta * H_running)",
            "h0_i = (1.0 + theta * H_running) * (np.expm1(theta * hi) / theta if theta != 0 else hi)",
        ),
        (
            "H_running += max(hi * (1.0 + theta * H_running), 0.0)",
            "H_running += max((1.0 + theta * H_running) * (np.expm1(theta * hi) / theta if theta != 0 else hi), 0.0)",
        ),
    )
    modified = source
    for old, new in changes:
        if modified.count(old) != 1:
            raise RuntimeError(f"Expected one occurrence of {old!r}; got {modified.count(old)}")
        modified = modified.replace(old, new)
    namespace = {}
    exec(compile(modified, "<exact-weekly variant>", "exec"), K.__dict__, namespace)
    return original, namespace[original.__name__], hashlib.sha256(source.encode()).hexdigest()


def compact(result):
    return {
        "point": result["point"],
        "accepted": result["accepted"],
        "diagnostics": result["diagnostics"],
    }


def raw_fit_diagnostics(counts, windows, fitter):
    cfg = configure(windows)
    dead, alive = arrays(counts)
    t = np.arange(dead.shape[1], dtype=float)
    quiet = np.zeros(len(t), bool)
    for a, b in windows:
        quiet |= (t >= a) & (t <= b)
    output = []
    for d in (0, 1):
        mr = np.divide(dead[d], alive[d] + K.EPS)
        h = K.hazard_from_mr(np.clip(mr, 0, .999))
        he = np.where(t >= cfg["skip"], h, 0.)
        (_, _), diag = fitter(he, t - cfg["skip"], quiet,
                              cfg["gompertz"]["k_anchor_weeks"],
                              cfg["gompertz"]["gamma_per_week"], deaths_arr=dead[d])
        output.append({key: diag.get(key) for key in (
            "theta0_init", "theta0_raw", "delta_raw_values", "delta_applied_values", "n_iter")})
    return output


def both_arm_wave_probabilities(scenario):
    cfg = configure(scenario["windows"])
    t = np.arange(scenario["weeks"])
    baseline = scenario.get("k", .001) * np.exp(cfg["gompertz"]["gamma_per_week"] * (t - cfg["skip"]))
    multiplier = np.where((t >= 26) & (t <= 40), scenario["effect"], 1.)
    H0 = np.cumsum(baseline * multiplier)
    output = []
    for theta in scenario["theta"]:
        survival = np.exp(-np.log1p(theta * H0) / theta)
        output.append(np.r_[np.r_[1., survival[:-1]] - survival, survival[-1]])
    return np.asarray(output), 1.0


def main() -> None:
    original, variant, source_hash = exact_weekly_variant()
    current_source = (HERE.parents[2] / "code" / "KCOR.py").read_text(encoding="utf-8")
    assert current_source.count("h0_i = hi * (1.0 + theta * H_running)") == 1
    assert current_source.count("H_running += max(hi * (1.0 + theta * H_running), 0.0)") == 1
    cases = []
    for name, windows, effect, both_arms in (
        ("one_window_control", ((6, 55),), 1.0, False),
        ("two_windows_no_wave", ((6, 25), (41, 55)), 1.0, False),
        ("two_windows_one_arm_wave", ((6, 25), (41, 55)), 1.3, False),
        ("two_windows_both_arms_wave", ((6, 25), (41, 55)), 1.3, True),
    ):
        scenario = {
            "theta": [4.0, 0.5], "n": 4_000_000, "weeks": 100,
            "windows": windows, "effect": effect, "effect_window": [26, 40],
            "target": 80,
        }
        probabilities, truth = both_arm_wave_probabilities(scenario) if both_arms else population(scenario)
        counts = probabilities * scenario["n"] * 0.5  # Expected counts, no sampling noise.
        K.fit_theta0_gompertz = original
        before = compact(estimate(counts, windows=windows, target=80))
        before_raw = raw_fit_diagnostics(counts, windows, original)
        K.fit_theta0_gompertz = variant
        after = compact(estimate(counts, windows=windows, target=80))
        after_raw = raw_fit_diagnostics(counts, windows, variant)
        cases.append({
            "name": name, "windows": windows, "effect": effect, "wave_in_both_arms": both_arms,
            "truth": truth, "first_order": before, "exact_weekly": after,
            "first_order_raw_fit": before_raw, "exact_weekly_raw_fit": after_raw,
            "point_change": after["point"] - before["point"],
            "theta_changes": [after["diagnostics"][i]["theta"] - before["diagnostics"][i]["theta"] for i in (0, 1)],
        })
    K.fit_theta0_gompertz = original
    assert cases[0]["point_change"] == 0
    assert cases[0]["theta_changes"] == [0, 0]
    report = {
        "design": "Exact expected weekly death counts from gamma-frailty model; both fitter versions receive identical input",
        "source_function_sha256": source_hash,
        "current_production_file_sha256": hashlib.sha256(current_source.encode()).hexdigest(),
        "current_production_contains_both_approximation_lines": True,
        "variant": "Only two internal first-order weekly reconstruction lines are replaced; final cumulative inversion and all other fitting rules are unchanged",
        "cases": cases,
    }
    (HERE / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for case in cases:
        print(case["name"], "point change", case["point_change"], "theta changes", case["theta_changes"],
              "accepted", case["first_order"]["accepted"], case["exact_weekly"]["accepted"])


if __name__ == "__main__":
    main()
