"""Paired Step 1 experiment. No changes are made to the selectable KCOR editions."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

for thread_setting in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_setting] = "1"
import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parents[2] / "iterative review"
OBJECTION = HERE.parent
STUDY = OBJECTION / "objection3_corrected_validation"
PRODUCTION = REVIEW / "revised/code/KCOR.py"
PAPER = OBJECTION / "objection3_paper_method"
sys.path[:0] = [str(PAPER), str(STUDY)]
import production_adapter as adapter
from simulation import bootstrap
from paper_step1 import fit_step1

ARCHIVED_FIT = adapter.K.fit_theta0_gompertz
DATASETS, DRAWS = 200, 199
design = json.loads((STUDY / "design.json").read_text())
windows, target = design["common"]["windows"], design["common"]["target"]


def initialize(weekly_update="archived"):
    global production_fit, joint_fit
    if weekly_update == "entry-moment":
        path = PRODUCTION
    elif weekly_update == "wave-week":
        path = REVIEW.parent / "objections/2.2/wave-reference/code/KCOR.py"
    elif weekly_update == "gap-refit":
        path = REVIEW.parent / "objections/3.1/code/KCOR.py"
    elif weekly_update == "wave-entry":
        path = REVIEW / "checkpoints/iteration-2/revised/code/KCOR.py"
    elif weekly_update == "gap-boundary":
        path = REVIEW.parent / "objections/2.3/code/KCOR.py"
    elif weekly_update == "iteration-2":
        path = REVIEW.parent / "objections/2.2/code/KCOR.py"
    else:
        path = REVIEW / "checkpoints/iteration-1/revised/code/KCOR.py"
    spec = importlib.util.spec_from_file_location("revised_kcor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter.K = module
    production_fit = module.fit_theta0_gompertz
    if weekly_update in ("iteration-2", "gap-boundary", "wave-entry", "gap-refit", "wave-week", "entry-moment"):
        joint_fit = partial(production_fit, joint_seed=True)
        return
    source = inspect.getsource(production_fit)
    if weekly_update == "archived":
        # Replay this study with its original weekly reconstruction and current module configuration.
        source = inspect.getsource(ARCHIVED_FIT)
        namespace = dict(vars(module))
        exec(compile(source, "<archived weekly reconstruction>", "exec"), namespace)
        production_fit = module.fit_theta0_gompertz = namespace["fit_theta0_gompertz"]
    elif weekly_update != "revised":
        raise ValueError(f"Unknown weekly update: {weekly_update}")
    original = "theta_seed, _, _, _, seed_ok = _fit_theta(first_quiet_mask, theta0_init, Delta=None)"
    candidate = """seed = fit_step1(h_arr[first_idx], t_r[first_idx], gamma, theta0_init)
        k_fixed, theta_seed, seed_ok = seed["k"], seed["theta"], seed["success"]
        H_gom_full = _build_H_gom_discrete(k_fixed)"""
    assert source.count(original) == 1
    namespace = dict(vars(module), fit_step1=fit_step1)
    exec(compile(source.replace(original, candidate), "<isolated Step 1 trial>", "exec"), namespace)
    joint_fit = namespace["fit_theta0_gompertz"]


def compare(i):
    folder = STUDY / "final_2026-11-10/unequal_large"
    record = json.loads((folder / f"{i:04d}.json").read_text())
    with np.load(folder / f"{i:04d}.npz") as archive:
        counts = archive["counts"]
        fixed = archive["draws"][:DRAWS]
    assert hashlib.sha256(counts.tobytes()).hexdigest() == record["counts_sha256"]
    rng = np.random.default_rng(np.random.SeedSequence([*record["seed_components"], 1]))
    samples = [bootstrap(counts, rng) for _ in range(DRAWS)]
    adapter.K.fit_theta0_gompertz = production_fit
    fixed_point = adapter.estimate(counts, windows, target)
    np.testing.assert_allclose(fixed_point["point"], record["point"], rtol=1e-10)
    # Verify the replayed sequence against the archived production-control arm.
    for j in (0, DRAWS // 2, DRAWS - 1):
        result = adapter.estimate(samples[j], windows, target)
        np.testing.assert_allclose(result["point"], fixed[j], rtol=1e-10)
    adapter.K.fit_theta0_gompertz = joint_fit
    joint_point = adapter.estimate(counts, windows, target)
    joint = []
    failures = 0
    for sample in samples:
        fit = adapter.estimate(sample, windows, target)
        failures += int(any(not d["success"] for d in fit["diagnostics"]))
        joint.append(fit["point"])
    assert np.isfinite(fixed).all() and np.isfinite(joint).all()
    def statistics(point, draws):
        lo, hi = np.quantile(draws, [.025, .975])
        return dict(point=point["point"], accepted=point["accepted"], low=lo, high=hi,
                    covered=bool(lo <= record["truth"] <= hi))
    return dict(dataset=i, truth=record["truth"], fixed=statistics(fixed_point, fixed),
                joint=statistics(joint_point, joint), joint_draw_fit_failures=failures)


def main():
    with ProcessPoolExecutor(max_workers=5, initializer=initialize) as pool:
        rows = []
        for row in pool.map(compare, range(DATASETS)):
            rows.append(row)
            if len(rows) % 20 == 0:
                print(f"{len(rows)}/{DATASETS}", flush=True)
    summary = {}
    for arm in ("fixed", "joint"):
        points = np.array([r[arm]["point"] for r in rows])
        widths = [r[arm]["high"] - r[arm]["low"] for r in rows]
        summary[arm] = dict(covered=sum(r[arm]["covered"] for r in rows),
                            accepted=sum(r[arm]["accepted"] for r in rows),
                            mean_width=float(np.mean(widths)), mean_point=float(np.mean(points)),
                            rmse=float(np.sqrt(np.mean((points-1)**2))))
    gains = sum(r["joint"]["covered"] and not r["fixed"]["covered"] for r in rows)
    losses = sum(r["fixed"]["covered"] and not r["joint"]["covered"] for r in rows)
    summary.update(joint_only=gains, fixed_only=losses,
                   paired_exact_p=binomtest(gains, gains+losses).pvalue if gains+losses else 1,
                   joint_draw_fit_failures=sum(r["joint_draw_fit_failures"] for r in rows))
    report = dict(datasets=DATASETS, draws_per_dataset=DRAWS, source=str(PRODUCTION.relative_to(REVIEW)),
                  revised_code_sha256=hashlib.sha256(PRODUCTION.read_bytes()).hexdigest(),
                  summary=summary, paired_results=rows)
    (HERE / "results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
