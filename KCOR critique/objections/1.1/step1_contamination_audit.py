"""Audit existing scenarios only; preserve the original generator and saved outputs.

Expected-count runs reproduce the original discrete hazard schedule, not an
exact continuous-time gamma model. Fixed-clean-fit results isolate refitting;
they are NOT an oracle or a treatment-effect target.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "test/quiet_window_contamination/run_contamination_test.py"
SAVED = SOURCE.parent / "out"


def load_experiment():
    os.environ["DATASET"] = "Czech"
    spec = importlib.util.spec_from_file_location("existing_contamination", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    # Fail rather than overwrite an earlier audit.
    args.outdir.mkdir(parents=True, exist_ok=False)
    E = load_experiment()
    K = E.K
    saved_meta = pd.read_csv(SAVED / "quiet_contamination_meta.csv", index_col=0).iloc[:, 0]
    k = float(saved_meta["k_simulator"])
    gamma = float(saved_meta["gamma_per_week"])
    assert int(saved_meta["N"]) == E.N_INITIAL
    assert int(saved_meta["n_weeks"]) == E.N_WEEKS
    skip = int(K.DYNAMIC_HVE_SKIP_WEEKS)
    cfg = K.get_gompertz_theta_fit_config()
    assert np.isclose(gamma, cfg["gamma_per_week"], rtol=0, atol=1e-16)
    anchor_weeks = int(cfg["k_anchor_weeks"])
    t, tr, quiet = E.build_time_arrays(E.N_WEEKS, skip)
    anchor = skip + 4  # Match this experiment, not another production convention.
    baseline_h = k * np.exp(gamma * np.maximum(0.0, tr))
    H_pre = np.concatenate(([0.0], np.cumsum(baseline_h)[:-1]))
    post = t >= skip

    def contamination(ed, ec):
        return np.where(quiet, ed * np.exp(-E.LAMBDA_DECAY * tr) + ec, 0.0)

    def expected(theta, g):
        h = baseline_h / (1 + theta * H_pre) + g
        alive = E.N_INITIAL * np.exp(-np.concatenate(([0.0], np.cumsum(h)[:-1])))
        dead = alive * -np.expm1(-h)
        np.testing.assert_allclose(dead.sum(), E.N_INITIAL * -np.expm1(-h.sum()), rtol=1e-12)
        np.testing.assert_allclose(alive[1:], alive[:-1] - dead[:-1], rtol=1e-12)
        return alive, dead, h

    def fit(theta, g):
        alive, dead, h = expected(theta, g)
        kh, th, diag, Hobs, H0 = E.cohort_pipeline(
            dead, alive, t, tr, quiet, anchor_weeks, gamma)
        np.testing.assert_allclose(Hobs, np.cumsum(np.where(post, h, 0)), rtol=1e-10, atol=1e-14)
        return dict(alive=alive, dead=dead, h=h, k=kh, theta=th, diag=diag, Hobs=Hobs, H0=H0)

    zero = np.zeros(E.N_WEEKS)
    clean_a = fit(E.THETA0_A, zero)
    clean_b = fit(E.THETA0_B, zero)
    agg = pd.read_csv(SAVED / "quiet_contamination_agg.csv")
    historical = pd.read_csv(SAVED / "quiet_contamination_long.csv")
    historical.groupby(["contamination_type", "fit_status_A", "fit_status_B"], dropna=False).size().rename("count").to_csv(args.outdir / "saved_fit_status_counts.csv")
    historical[(historical.epsilon_dyn == 0) & (historical.epsilon_comorb == 0)].groupby("contamination_type")["KCOR_asymptote"].agg(["count", "mean", "std", "min", "max"]).to_csv(args.outdir / "saved_no_contamination_summary.csv")

    rows, trajectory, replay_rows, diagnostics = [], [], [], []
    for _, old in agg.iterrows():
        mode = old.contamination_type.removesuffix("_agg")
        ed, ec = float(old.epsilon_dyn), float(old.epsilon_comorb)
        g = contamination(ed, ec)
        a = fit(E.THETA0_A, g)
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = a["H0"] / clean_b["H0"]
            fixed_raw = K.invert_gamma_frailty(a["Hobs"], clean_a["theta"]) / clean_b["H0"]
        anchored = E.anchored_kcor(a["H0"], clean_b["H0"], skip)
        np.testing.assert_allclose(anchored[post], raw[post] / raw[anchor], rtol=1e-12)
        treated_survival = np.exp(-a["h"].sum())
        clean_survival = np.exp(-clean_a["h"].sum())
        np.testing.assert_allclose(treated_survival / clean_survival, np.exp(-g.sum()), rtol=1e-12)
        assert g.min() >= 0
        assert treated_survival <= clean_survival + 1e-14
        d = a["diag"]
        row = dict(mode=mode, epsilon_dyn=ed, epsilon_constant=ec,
                   peak_added_hazard_over_href=float(g.max() / k),
                   added_cumulative_hazard=float(g.sum()),
                   extra_expected_deaths_vs_same_group_without_contamination=float(E.N_INITIAL * (clean_survival - treated_survival)),
                   theta_A=a["theta"], theta_B=clean_b["theta"],
                   raw_final=float(raw[-1]), raw_at_anchor=float(raw[anchor]),
                   anchored_final=float(anchored[-1]),
                   fixed_clean_fit_raw_final=float(fixed_raw[-1]),
                   fixed_clean_fit_anchored_final=float(fixed_raw[-1] / fixed_raw[anchor]),
                   saved_random_mean=float(old.KCOR_asymptote_mean),
                   fit_success_A=bool(d.get("success", False)),
                   fit_status_A=d.get("status"),
                   relative_fit_rmse_A=d.get("relRMSE_hazard_fit"),
                   weak_identification_A=d.get("identifiability_flag"),
                   bad_fit_A=d.get("bad_fit"),
                   insufficient_signal_A=d.get("insufficient_signal"),
                   insufficient_deaths_A=d.get("insufficient_deaths"))
        rows.append(row)
        diagnostics.append(dict(mode=mode, epsilon_dyn=ed, epsilon_constant=ec, A=d, B=clean_b["diag"]))
        for i in range(E.N_WEEKS):
            trajectory.append(dict(mode=mode, epsilon_dyn=ed, epsilon_constant=ec,
                                   week=i, rebased_week=tr[i], in_quiet_window=bool(quiet[i]),
                                   added_hazard=g[i], expected_alive_A=a["alive"][i],
                                   expected_dead_A=a["dead"][i], raw=raw[i], anchored=anchored[i],
                                   fixed_clean_fit_raw=fixed_raw[i]))

        # Replay the first saved repetition per existing setting. Historical metadata
        # did not retain a seed/version; base seed 42 is a reconstruction hypothesis.
        offset, code = {"dynamic": (0, 1), "comorbidity": (10000, 2), "combined": (20000, 3)}[mode]
        seed = np.random.SeedSequence([(42 + offset) % (2**32), code,
                                      int(round(ed * 1e12)) % (2**31),
                                      int(round(ec * 1e12)) % (2**31), 0])
        rng = np.random.default_rng(seed)
        replay = E.run_single_rep(rng, k, gamma, ed, ec, mode, t, tr, quiet, anchor_weeks, gamma)
        match = historical[(historical.contamination_type == mode)
                           & np.isclose(historical.epsilon_dyn, ed, rtol=0, atol=1e-14)
                           & np.isclose(historical.epsilon_comorb, ec, rtol=0, atol=1e-14)
                           & (historical.rep == 0)]
        assert len(match) == 1
        previous = match.iloc[0]
        replay_row = dict(mode=mode, epsilon_dyn=ed, epsilon_constant=ec, replay_returned=replay is not None)
        for field in ("KCOR_asymptote", "theta0_hat_A", "theta0_hat_B", "k_hat_A", "k_hat_B"):
            replay_row[field + "_saved"] = float(previous[field])
            replay_row[field + "_current"] = float(replay[field]) if replay else None
            replay_row[field + "_matches"] = bool(replay and np.isclose(replay[field], previous[field], rtol=1e-6, atol=1e-8))
        replay_rows.append(replay_row)

    pd.DataFrame(rows).to_csv(args.outdir / "expected_count_results.csv", index=False)
    pd.DataFrame(trajectory).to_csv(args.outdir / "expected_count_trajectories.csv", index=False)
    pd.DataFrame(replay_rows).to_csv(args.outdir / "saved_replay_comparison.csv", index=False)

    def serial(value):
        if isinstance(value, dict):
            return {str(k): serial(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [serial(v) for v in value]
        if isinstance(value, np.generic):
            return serial(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    sources = [SOURCE, ROOT / "code/KCOR.py", ROOT / "data/Czech/Czech.yaml",
               SAVED / "quiet_contamination_meta.csv", SAVED / "quiet_contamination_agg.csv",
               SAVED / "quiet_contamination_long.csv", Path(__file__)]
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    dataset="Czech", mortality_scale=k, gamma_per_week=gamma,
                    baseline_fit_config=cfg, diagnostic_config=K.get_theta_degenerate_fit_config(),
                    anchor_original_week=anchor, anchor_rebased_week=anchor-skip,
                    sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    limitations=["Existing discrete generator retained; no new timing or mechanisms.",
                                 "Expected counts remove random event noise; they are not a continuous-time exact gamma model.",
                                 "Fixed-clean-fit contrasts measure refitting sensitivity, not truth.",
                                 "Full workbook/NPH processing and paper-level acceptance claims are not evaluated.",
                                 "Historical base seed 42 is inferred; saved files lack original code hashes and seed.",
                                 "No causal or depletion-neutral target is declared by this first-step audit."])
    (args.outdir / "manifest.json").write_text(json.dumps(serial(manifest), indent=2, allow_nan=False), encoding="utf-8")
    (args.outdir / "fit_diagnostics.json").write_text(json.dumps(serial(diagnostics), indent=2, allow_nan=False), encoding="utf-8")
    result = pd.DataFrame(rows)
    print(result[["mode", "epsilon_dyn", "epsilon_constant", "extra_expected_deaths_vs_same_group_without_contamination", "theta_A", "raw_final", "raw_at_anchor", "anchored_final", "fit_status_A", "relative_fit_rmse_A"]].to_string(index=False))
    replay_df = pd.DataFrame(replay_rows)
    matching = replay_df[[c for c in replay_df if c.endswith("_matches")]].all(axis=1)
    print(f"Replay matches across all five numerical fields: {int(matching.sum())}/{len(matching)}")
    print(f"Saved audit to {args.outdir}")


if __name__ == "__main__":
    main()
