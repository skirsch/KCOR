#!/usr/bin/env python3
"""
Quiet-window synthetic contamination experiment.

Injects additive non-frailty hazard only in rebased weeks 4–56 on cohort A (dose 1,
lower theta0), runs production fit_theta0_gompertz + invert_gamma_frailty + anchored
KCOR (dose 1 / dose 0). Calibrates baseline k to Czech KCOR_CMR weekly hazard scale.

See repository plan: quiet-window contamination experiment.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
import KCOR as K  # noqa: E402

# --- Experiment constants (plan) ---
N_INITIAL = 50_000
N_WEEKS = 130
THETA0_A = 0.5  # dose 1 (lower frailty)
THETA0_B = 4.0  # dose 0 (higher frailty)
LAMBDA_DECAY = 0.05
EPSILON_DYNAMIC = np.array([0.0, 5e-4, 1e-3, 2e-3, 5e-3], dtype=float)
EPSILON_COMORB = np.array([0.0, 1e-4, 3e-4, 1e-3, 3e-3], dtype=float)
EPSILON_DYNAMIC_POS = np.array([5e-4, 1e-3, 2e-3, 5e-3], dtype=float)
EPSILON_COMORB_POS = np.array([1e-4, 3e-4, 1e-3, 3e-3], dtype=float)
N_REPS = 20
MIN_SUCCESS_REPS = 16
PLAUSIBLE_DYNAMIC_HVE_MAX = 0.001
PLAUSIBLE_COMORBIDITY_MAX = 0.0005
QUIET_T_REBASED_MIN = 4
QUIET_T_REBASED_MAX = 56
YOB_MIN = 1940
YOB_MAX = 1960


def parse_iso_label(label: str) -> tuple[int, int]:
    match = re.search(r"(\d{4})[-_]?(\d{1,2})", str(label).strip())
    if not match:
        raise ValueError(f"Bad ISO week label: {label!r}")
    return int(match.group(1)), int(match.group(2))


def enrollment_label_to_monday(label: str) -> pd.Timestamp:
    year, week = parse_iso_label(label)
    return pd.Timestamp.fromisocalendar(year, week, 1)


def standardize_sheet_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "YearOfBirth": ["YearOfBirth", "YoB", "YOB"],
        "Dose": ["Dose"],
        "ISOweekDied": ["ISOweekDied", "ISOWeekDied", "ISOweek", "ISOWeek"],
        "Dead": ["Dead", "Deaths"],
        "Alive": ["Alive"],
    }
    rename_map = {}
    for required, candidates in col_map.items():
        for name in candidates:
            if name in df.columns:
                rename_map[name] = required
                break
        if required not in rename_map.values():
            raise ValueError(f"Missing required column (one of): {required} / {candidates}")
    return df.rename(columns=rename_map)


def compute_h_ref_czech(
    cmr_path: Path,
    sheet_name: str,
    skip_weeks: int,
) -> dict:
    """
    Median weekly hazard (K.hazard_from_mr) over post-skip rebased weeks 0–3,
    dose 0, YearOfBirth in [YOB_MIN, YOB_MAX], enrollment sheet sheet_name.
    """
    if not cmr_path.is_file():
        raise FileNotFoundError(
            f"KCOR_CMR workbook not found: {cmr_path}\n"
            "Generate it with code/KCOR_CMR.py or make targets (see README), "
            "or pass --cmr-xlsx PATH."
        )
    raw = pd.read_excel(cmr_path, sheet_name=sheet_name)
    df = standardize_sheet_columns(raw)
    df["YearOfBirth"] = pd.to_numeric(df["YearOfBirth"], errors="coerce")
    df = df[(df["YearOfBirth"] >= YOB_MIN) & (df["YearOfBirth"] <= YOB_MAX)]
    df = df[df["Dose"] == 0].copy()
    parsed = df["ISOweekDied"].apply(lambda x: parse_iso_label(str(x)))
    df["iso_year"] = parsed.apply(lambda x: x[0])
    df["iso_week"] = parsed.apply(lambda x: x[1])
    df["week_monday"] = pd.to_datetime(
        [pd.Timestamp.fromisocalendar(y, w, 1) for y, w in zip(df["iso_year"], df["iso_week"])],
        utc=False,
    )
    enroll_monday = enrollment_label_to_monday(sheet_name)
    df["t_week"] = ((df["week_monday"] - enroll_monday).dt.days // 7).astype(int)
    df = df[df["t_week"] >= 0].copy()
    band = (df["t_week"] >= skip_weeks) & (df["t_week"] < skip_weeks + int(K.get_gompertz_theta_fit_config().get("k_anchor_weeks", 4)))
    df = df[band]
    if df.empty:
        raise RuntimeError("No Czech rows for h_ref (check sheet, YoB filter, anchor band).")
    alive = df["Alive"].to_numpy(dtype=float)
    dead = df["Dead"].to_numpy(dtype=float)
    mr = np.clip(dead / np.maximum(alive, 1.0), 0.0, 1.0 - 1e-12)
    hazards = K.hazard_from_mr(mr)
    h_ref = float(np.median(hazards[np.isfinite(hazards)]))
    return {
        "h_ref": h_ref,
        "cmr_path": str(cmr_path),
        "sheet_name": sheet_name,
        "yob_min": YOB_MIN,
        "yob_max": YOB_MAX,
        "dose_ref": 0,
        "n_rows_used": int(len(df)),
        "rule": "median(hazard_from_mr(Dead/Alive)) over t_week in [skip, skip+k_anchor), Dose=0",
    }


def build_time_arrays(n_weeks: int, skip: int) -> tuple[np.ndarray, np.ndarray]:
    t_week = np.arange(n_weeks, dtype=float)
    t_rebased = t_week - float(skip)
    quiet_mask = (t_rebased >= QUIET_T_REBASED_MIN) & (t_rebased <= QUIET_T_REBASED_MAX)
    return t_week, t_rebased, quiet_mask


def simulate_one_cohort(
    rng: np.random.Generator,
    n0: int,
    n_weeks: int,
    k: float,
    gamma_wk: float,
    theta0: float,
    skip: int,
    contamination_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gamma-frailty Gompertz DGP; optional contamination adds to cohort hazard in quiet
    window only. Latent H0_eff accumulates uncontaminated gf increments only.
    """
    alive = np.zeros(n_weeks + 1, dtype=np.int64)
    dead = np.zeros(n_weeks, dtype=np.int64)
    alive[0] = n0
    H0_eff_running = 0.0
    for t_week in range(n_weeks):
        t_r = float(t_week - skip)
        gf = k * np.exp(gamma_wk * max(0.0, t_r))
        h0_eff = gf
        H_pre = H0_eff_running
        h_frailty = h0_eff / (1.0 + theta0 * H_pre)
        g = 0.0
        if contamination_fn is not None:
            g = float(contamination_fn(t_r))
        h_total = h_frailty + g
        h_total = min(max(h_total, 0.0), 50.0)
        p_death = 1.0 - np.exp(-h_total)
        p_death = min(p_death, 1.0 - 1e-15)
        at_risk = int(alive[t_week])
        d = rng.binomial(at_risk, p_death) if at_risk > 0 else 0
        dead[t_week] = d
        alive[t_week + 1] = at_risk - d
        H0_eff_running += h0_eff
    return alive[:n_weeks], dead


def cohort_pipeline(
    dead: np.ndarray,
    alive: np.ndarray,
    t_week: np.ndarray,
    t_rebased: np.ndarray,
    quiet_mask: np.ndarray,
    k_anchor_weeks: int,
    gamma_per_week: float,
) -> tuple[float, float, dict, np.ndarray, np.ndarray]:
    skip = int(K.DYNAMIC_HVE_SKIP_WEEKS)
    alive_f = alive.astype(float)
    dead_f = dead.astype(float)
    mr = np.clip(dead_f / np.maximum(alive_f, 1.0), 0.0, 1.0 - 1e-12)
    hazard_obs = K.hazard_from_mr(mr)
    hazard_eff = np.where(t_week >= float(skip), hazard_obs, 0.0)
    H_obs = np.cumsum(hazard_eff)
    (k_hat, theta_hat), diag = K.fit_theta0_gompertz(
        hazard_eff,
        t_rebased,
        quiet_mask,
        k_anchor_weeks,
        gamma_per_week,
        deaths_arr=dead_f,
    )
    diag = dict(diag)
    ok = bool(diag.get("success", False)) and np.isfinite(theta_hat)
    th = float(theta_hat) if ok else 0.0
    H0 = K.invert_gamma_frailty(H_obs, th)
    return float(k_hat), th, diag, H_obs, H0


def anchored_kcor(H0_a: np.ndarray, H0_b: np.ndarray, skip: int) -> np.ndarray:
    norm_week = skip + 4
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = np.where(H0_b > 1e-12, H0_a / H0_b, np.nan)
    if norm_week < len(raw) and np.isfinite(raw[norm_week]) and raw[norm_week] > 0:
        return raw / raw[norm_week]
    return raw


def kcor_slope(kcor: np.ndarray, t_rebased: np.ndarray, post_start: float = 8.0) -> float:
    mask = t_rebased >= post_start
    x = t_rebased[mask].astype(float)
    y = kcor[mask].astype(float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2:
        return 0.0
    x_c = x - x.mean()
    if np.sum(x_c**2) < 1e-12:
        return 0.0
    return float((x_c @ (y - y.mean())) / (x_c @ x_c))


def kcor_asymptote_value(kcor: np.ndarray, H0_b: np.ndarray) -> float:
    """Last week with valid denominator on raw ratio scale; use final anchored index."""
    for i in range(len(kcor) - 1, -1, -1):
        if H0_b[i] > 1e-10 and np.isfinite(kcor[i]):
            return float(kcor[i])
    return float("nan")


def run_single_rep(
    rng: np.random.Generator,
    k: float,
    gamma_wk: float,
    eps_dyn: float,
    eps_com: float,
    mode: str,
    t_week: np.ndarray,
    t_rebased: np.ndarray,
    quiet_mask: np.ndarray,
    k_anchor_weeks: int,
    gamma_per_week: float,
) -> dict | None:
    skip = int(K.DYNAMIC_HVE_SKIP_WEEKS)

    def make_fn(ed: float, ec: float, m: str):
        if m == "dynamic":

            def fn(tr: float) -> float:
                if QUIET_T_REBASED_MIN <= tr <= QUIET_T_REBASED_MAX:
                    return ed * np.exp(-LAMBDA_DECAY * tr)
                return 0.0

            return fn if ed > 0 else None
        if m == "comorbidity":

            def fn2(tr: float) -> float:
                if QUIET_T_REBASED_MIN <= tr <= QUIET_T_REBASED_MAX:
                    return ec
                return 0.0

            return fn2 if ec > 0 else None
        if m == "combined":

            def fn3(tr: float) -> float:
                if QUIET_T_REBASED_MIN <= tr <= QUIET_T_REBASED_MAX:
                    return ed * np.exp(-LAMBDA_DECAY * tr) + ec
                return 0.0

            return fn3

        raise ValueError(mode)

    fn_a = make_fn(eps_dyn, eps_com, mode)
    alive_a, dead_a = simulate_one_cohort(
        rng, N_INITIAL, N_WEEKS, k, gamma_wk, THETA0_A, skip, fn_a
    )
    alive_b, dead_b = simulate_one_cohort(
        rng, N_INITIAL, N_WEEKS, k, gamma_wk, THETA0_B, skip, None
    )

    ka, tha, dia, _, H0a = cohort_pipeline(
        dead_a, alive_a, t_week, t_rebased, quiet_mask, k_anchor_weeks, gamma_per_week
    )
    kb, thb, dib, _, H0b = cohort_pipeline(
        dead_b, alive_b, t_week, t_rebased, quiet_mask, k_anchor_weeks, gamma_per_week
    )

    if not (bool(dia.get("success", False)) and np.isfinite(tha)):
        return None
    if not (bool(dib.get("success", False)) and np.isfinite(thb)):
        return None

    kcor = anchored_kcor(H0a, H0b, skip)
    kasy = kcor_asymptote_value(kcor, H0b)
    kslope = kcor_slope(kcor, t_rebased)

    assert tha >= 0.0, "theta0 must be non-negative"
    assert thb >= 0.0, "theta0 must be non-negative"
    assert np.isfinite(kasy) and kasy > 0.0, "KCOR must be positive"

    return {
        "theta0_hat_A": float(tha),
        "theta0_hat_B": float(thb),
        "k_hat_A": float(ka),
        "k_hat_B": float(kb),
        "KCOR_asymptote": float(kasy),
        "KCOR_slope": float(kslope),
        "sign_reversal_rep": int(kasy > 1.0),
        "fit_status_A": str(dia.get("status", "")),
        "fit_status_B": str(dib.get("status", "")),
    }


def sweep_mode(
    mode: str,
    eps_grid_dyn: np.ndarray,
    eps_grid_com: np.ndarray,
    k: float,
    gamma_wk: float,
    h_ref: float,
    t_week: np.ndarray,
    t_rebased: np.ndarray,
    quiet_mask: np.ndarray,
    k_anchor_weeks: int,
    gamma_per_week: float,
    base_seed: int,
) -> list[dict]:
    rows: list[dict] = []
    if mode == "combined":
        pairs = [(ed, ec) for ed in eps_grid_dyn for ec in eps_grid_com]
    elif mode == "dynamic":
        pairs = [(ed, 0.0) for ed in eps_grid_dyn]
    else:
        pairs = [(0.0, ec) for ec in eps_grid_com]

    mode_code = {"dynamic": 1, "comorbidity": 2, "combined": 3}[mode]
    for ed, ec in pairs:
        rep_results: list[dict] = []
        for rep in range(N_REPS):
            ed_i = int(np.clip(round(float(ed) * 1e12), -(2**63), 2**63 - 1)) % (2**31)
            ec_i = int(np.clip(round(float(ec) * 1e12), -(2**63), 2**63 - 1)) % (2**31)
            ss = np.random.SeedSequence([base_seed % (2**32), mode_code, ed_i, ec_i, rep])
            rng = np.random.default_rng(ss)
            out = run_single_rep(
                rng,
                k,
                gamma_wk,
                ed,
                ec,
                mode,
                t_week,
                t_rebased,
                quiet_mask,
                k_anchor_weeks,
                gamma_per_week,
            )
            if out is not None:
                rep_results.append(out)

        n_ok = len(rep_results)
        if n_ok < MIN_SUCCESS_REPS:
            raise RuntimeError(
                f"{mode} ed={ed} ec={ec}: only {n_ok}/{N_REPS} reps succeeded "
                f"(need >= {MIN_SUCCESS_REPS})."
            )
        mean_k = float(np.mean([r["KCOR_asymptote"] for r in rep_results]))
        std_k = float(np.std([r["KCOR_asymptote"] for r in rep_results], ddof=1)) if n_ok > 1 else 0.0
        mean_ea = float(np.mean([r["theta0_hat_A"] - THETA0_A for r in rep_results]))
        std_ea = float(np.std([r["theta0_hat_A"] - THETA0_A for r in rep_results], ddof=1)) if n_ok > 1 else 0.0
        mean_eb = float(np.mean([r["theta0_hat_B"] - THETA0_B for r in rep_results]))
        std_eb = float(np.std([r["theta0_hat_B"] - THETA0_B for r in rep_results], ddof=1)) if n_ok > 1 else 0.0
        frac_rev = float(np.mean([r["sign_reversal_rep"] for r in rep_results]))

        for rep, r in enumerate(rep_results):
            rows.append(
                {
                    "contamination_type": mode,
                    "epsilon_dyn": ed,
                    "epsilon_comorb": ec,
                    "rep": rep,
                    "theta0_hat_A": r["theta0_hat_A"],
                    "theta0_hat_B": r["theta0_hat_B"],
                    "theta0_err_A": r["theta0_hat_A"] - THETA0_A,
                    "theta0_err_B": r["theta0_hat_B"] - THETA0_B,
                    "k_hat_A": r["k_hat_A"],
                    "k_hat_B": r["k_hat_B"],
                    "KCOR_asymptote": r["KCOR_asymptote"],
                    "KCOR_slope": r["KCOR_slope"],
                    "sign_reversal_rep": r["sign_reversal_rep"],
                    "fit_status_A": r["fit_status_A"],
                    "fit_status_B": r["fit_status_B"],
                    "h_ref": h_ref,
                    "epsilon_dyn_over_href": ed / h_ref if h_ref > 0 else np.nan,
                    "epsilon_comorb_over_href": ec / h_ref if h_ref > 0 else np.nan,
                }
            )

        rows.append(
            {
                "contamination_type": f"{mode}_agg",
                "epsilon_dyn": ed,
                "epsilon_comorb": ec,
                "rep": -1,
                "theta0_hat_A": np.nan,
                "theta0_hat_B": np.nan,
                "theta0_err_A_mean": mean_ea,
                "theta0_err_A_std": std_ea,
                "theta0_err_B_mean": mean_eb,
                "theta0_err_B_std": std_eb,
                "KCOR_asymptote_mean": mean_k,
                "KCOR_asymptote_std": std_k,
                "n_success": n_ok,
                "frac_reps_kcor_gt_1": frac_rev,
            }
        )
    return rows


def first_reversal_1d(agg_rows: list[dict], mode: str, eps_key: str) -> float | None:
    sub = [r for r in agg_rows if r.get("contamination_type") == f"{mode}_agg"]
    sub.sort(key=lambda r: float(r[eps_key]))
    for r in sub:
        if r.get("KCOR_asymptote_mean", 0) > 1.0 and float(r[eps_key]) > 0:
            return float(r[eps_key])
    return None


def first_reversal_combined(agg_rows: list[dict]) -> tuple[float, float] | None:
    sub = [r for r in agg_rows if r.get("contamination_type") == "combined_agg"]
    sub.sort(key=lambda r: (float(r["epsilon_dyn"]), float(r["epsilon_comorb"])))
    for r in sub:
        if r.get("KCOR_asymptote_mean", 0) > 1.0:
            return float(r["epsilon_dyn"]), float(r["epsilon_comorb"])
    return None


def plot_figures(
    out_dir: Path,
    agg_dynamic: list[dict],
    agg_comorb: list[dict],
    agg_combined: list[dict],
    h_ref: float,
) -> None:
    def agg_only(rows, mode):
        return [r for r in rows if r.get("contamination_type") == f"{mode}_agg"]

    ad = agg_only(agg_dynamic, "dynamic")
    ac = agg_only(agg_comorb, "comorbidity")
    comb_agg = agg_only(agg_combined, "combined")

    # Figure A: theta0 error panels + combined heatmap
    fig_a, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax_i, data, title, xkey in zip(
        axes,
        [ad, ac, None],
        ["dynamic", "comorbidity", "combined frac reps KCOR>1"],
        ["epsilon_dyn", "epsilon_comorb", None],
    ):
        if data is not None:
            xs = [float(r[xkey]) for r in data]
            ys = [r["theta0_err_A_mean"] for r in data]
            yerr = [r["theta0_err_A_std"] for r in data]
            ax_i.errorbar(xs, ys, yerr=yerr, fmt="o-", capsize=3)
            ax_i.axhline(0.0, color="gray", lw=0.8)
            ax_i.set_xlabel(xkey)
            ax_i.set_ylabel(r"mean $\hat\theta_{0,A}-\theta_{0,A}$")
            ax_i.set_title(title)
        else:
            # Combined grid: frac. reps with anchored KCOR asymptote > 1 (sign-reversal proxy).
            # (|theta0_err| heatmap is flat when hat_theta hits 0 floor — not paper-useful.)
            mat = np.full((len(EPSILON_COMORB_POS), len(EPSILON_DYNAMIC_POS)), np.nan)
            for r in comb_agg:
                i = int(np.argmin(np.abs(EPSILON_COMORB_POS - float(r["epsilon_comorb"]))))
                j = int(np.argmin(np.abs(EPSILON_DYNAMIC_POS - float(r["epsilon_dyn"]))))
                mat[i, j] = float(r.get("frac_reps_kcor_gt_1", np.nan))
            im = ax_i.imshow(mat, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
            ax_i.set_xticks(range(len(EPSILON_DYNAMIC_POS)))
            ax_i.set_xticklabels([f"{x:.0e}" for x in EPSILON_DYNAMIC_POS], rotation=45, ha="right")
            ax_i.set_yticks(range(len(EPSILON_COMORB_POS)))
            ax_i.set_yticklabels([f"{x:.0e}" for x in EPSILON_COMORB_POS])
            ax_i.set_xlabel(r"$\epsilon_{\mathrm{dyn}}$")
            ax_i.set_ylabel(r"$\epsilon_{\mathrm{comorb}}$")
            cbar = fig_a.colorbar(im, ax=ax_i, fraction=0.046, pad=0.04)
            cbar.set_label(
                r"Frac. reps ($\mathrm{KCOR}_{\infty}>1$)",
                rotation=90,
                labelpad=12,
            )
            cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            cbar.ax.yaxis.get_offset_text().set_visible(False)
            ax_i.set_title("Combined: reversal frequency")
    fig_a.tight_layout()
    fig_a.savefig(out_dir / "fig_quiet_contam_theta_error.png", dpi=150)
    plt.close(fig_a)

    # Figure B: KCOR asymptote 1D + heatmap
    fig_b, axes_b = plt.subplots(1, 3, figsize=(14, 4))
    for ax_i, data, title, xkey in zip(
        axes_b,
        [ad, ac, None],
        ["dynamic", "comorbidity", "combined mean KCOR"],
        ["epsilon_dyn", "epsilon_comorb", None],
    ):
        if data is not None:
            xs = [float(r[xkey]) for r in data]
            ys = [r["KCOR_asymptote_mean"] for r in data]
            yerr = [r["KCOR_asymptote_std"] for r in data]
            ax_i.errorbar(xs, ys, yerr=yerr, fmt="o-", capsize=3)
            ax_i.axhline(1.0, color="black", ls="--", lw=1)
            ax_i.set_xlabel(xkey)
            ax_i.set_ylabel("mean KCOR asymptote (anchored)")
            ax_i.set_title(title)
        else:
            matk = np.full((len(EPSILON_COMORB_POS), len(EPSILON_DYNAMIC_POS)), np.nan)
            for r in comb_agg:
                i = int(np.argmin(np.abs(EPSILON_COMORB_POS - float(r["epsilon_comorb"]))))
                j = int(np.argmin(np.abs(EPSILON_DYNAMIC_POS - float(r["epsilon_dyn"]))))
                matk[i, j] = float(r.get("KCOR_asymptote_mean", np.nan))
            im = ax_i.imshow(matk, aspect="auto", origin="lower")
            ax_i.set_xticks(range(len(EPSILON_DYNAMIC_POS)))
            ax_i.set_xticklabels([f"{x:.0e}" for x in EPSILON_DYNAMIC_POS], rotation=45, ha="right")
            ax_i.set_yticks(range(len(EPSILON_COMORB_POS)))
            ax_i.set_yticklabels([f"{x:.0e}" for x in EPSILON_COMORB_POS])
            ax_i.set_xlabel(r"$\epsilon_{\mathrm{dyn}}$")
            ax_i.set_ylabel(r"$\epsilon_{\mathrm{comorb}}$")
            cbar_b = fig_b.colorbar(im, ax=ax_i, fraction=0.046, pad=0.04)
            cbar_b.set_label("Mean KCOR asymptote (anchored)", rotation=90, labelpad=12)
            cbar_b.ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
            cbar_b.ax.yaxis.get_offset_text().set_visible(False)
            ax_i.set_title("Combined: KCOR asymptote")
    fig_b.tight_layout()
    fig_b.savefig(out_dir / "fig_quiet_contam_kcor_asymptote.png", dpi=150)
    plt.close(fig_b)


def print_summary(
    thr_d: float | None,
    thr_c: float | None,
    thr_pair: tuple[float, float] | None,
    h_ref: float,
) -> None:
    z_lo, z_hi = 0.0003, 0.001
    print("\n--- Summary (quiet-window contamination) ---")
    print(f"Empirical h_ref (calibration) = {h_ref:.6f} (weekly hazard scale; typical Czech band ~{z_lo}–{z_hi} cited in plan).")
    if thr_d is not None:
        print(f"Type 1 (dynamic): sign reversal first at epsilon_dynamic > {thr_d} (mean KCOR asymptote > 1).")
        if thr_d < PLAUSIBLE_DYNAMIC_HVE_MAX:
            print(
                f"WARNING: Sign reversal at epsilon={thr_d} — within plausible dynamic HVE max ({PLAUSIBLE_DYNAMIC_HVE_MAX})."
            )
        else:
            print(
                f"OK: Sign reversal only at epsilon_dynamic={thr_d} — above plausible max ({PLAUSIBLE_DYNAMIC_HVE_MAX})."
            )
    else:
        print("Type 1 (dynamic): no sign reversal in grid (mean KCOR asymptote stayed <= 1).")
        print(f"OK: no reversal in grid — above plausible range for dynamic HVE (max tested vs {PLAUSIBLE_DYNAMIC_HVE_MAX}).")

    if thr_c is not None:
        print(f"Type 2 (comorbidity): sign reversal first at epsilon_comorb > {thr_c}.")
        if thr_c < PLAUSIBLE_COMORBIDITY_MAX:
            print(
                f"WARNING: Sign reversal at epsilon={thr_c} — within plausible comorbidity max ({PLAUSIBLE_COMORBIDITY_MAX})."
            )
        else:
            print(
                f"OK: Sign reversal only at epsilon_comorb={thr_c} — above plausible max ({PLAUSIBLE_COMORBIDITY_MAX})."
            )
    else:
        print("Type 2 (comorbidity): no sign reversal in grid.")
        print(f"OK: no reversal in grid — relative to plausible comorbidity max ({PLAUSIBLE_COMORBIDITY_MAX}).")

    if thr_pair is not None:
        xd, yc = thr_pair
        print(f"Combined (dyn + comorb) sign reversal at epsilon_dyn={xd}, epsilon_comorb={yc}")
    else:
        print("Combined (dyn + comorb): none in grid (mean KCOR asymptote <= 1 for all 4×4 pairs tested).")

    print(
        "Closing: individual 1D sweeps and the joint 4×4 grid bound how much quiet-window "
        "non-frailty contamination is needed for a harmful KCOR shift under known θ0 asymmetry; "
        "compare ε to h_ref for substantive plausibility."
    )
    print("---\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cmr-xlsx",
        type=Path,
        default=REPO_ROOT / "data" / "Czech" / "KCOR_CMR.xlsx",
        help="Path to KCOR_CMR.xlsx",
    )
    parser.add_argument(
        "--enrollment-sheet",
        type=str,
        default="2021_24",
        help="Workbook sheet name (must match KCOR_CMR.xlsx tab, e.g. 2021_24)",
    )
    parser.add_argument("--outdir", type=Path, default=REPO_ROOT / "test" / "quiet_window_contamination" / "out")
    parser.add_argument("--base-seed", type=int, default=42)
    args = parser.parse_args()

    gom = K.get_gompertz_theta_fit_config()
    gamma_per_week = float(gom.get("gamma_per_week", float(gom.get("gamma_per_year", 0.085)) / 52.0))
    k_anchor_weeks = int(gom.get("k_anchor_weeks", 4))
    skip = int(K.DYNAMIC_HVE_SKIP_WEEKS)

    hmeta = compute_h_ref_czech(args.cmr_xlsx, args.enrollment_sheet, skip)
    h_ref = float(hmeta["h_ref"])
    k_sim = h_ref

    t_week, t_rebased, quiet_mask = build_time_arrays(N_WEEKS, skip)

    args.outdir.mkdir(parents=True, exist_ok=True)

    rows_all: list[dict] = []

    rd = sweep_mode(
        "dynamic",
        EPSILON_DYNAMIC,
        np.array([0.0]),
        k_sim,
        gamma_per_week,
        h_ref,
        t_week,
        t_rebased,
        quiet_mask,
        k_anchor_weeks,
        gamma_per_week,
        args.base_seed,
    )
    rc = sweep_mode(
        "comorbidity",
        np.array([0.0]),
        EPSILON_COMORB,
        k_sim,
        gamma_per_week,
        h_ref,
        t_week,
        t_rebased,
        quiet_mask,
        k_anchor_weeks,
        gamma_per_week,
        args.base_seed + 10_000,
    )
    rx = sweep_mode(
        "combined",
        EPSILON_DYNAMIC_POS,
        EPSILON_COMORB_POS,
        k_sim,
        gamma_per_week,
        h_ref,
        t_week,
        t_rebased,
        quiet_mask,
        k_anchor_weeks,
        gamma_per_week,
        args.base_seed + 20_000,
    )
    rows_all = rd + rc + rx

    long_rows = [r for r in rows_all if not str(r.get("contamination_type", "")).endswith("_agg")]
    agg_rows = [r for r in rows_all if str(r.get("contamination_type", "")).endswith("_agg")]

    pd.DataFrame(long_rows).to_csv(args.outdir / "quiet_contamination_long.csv", index=False)
    pd.DataFrame(agg_rows).to_csv(args.outdir / "quiet_contamination_agg.csv", index=False)
    meta = {**hmeta, "k_simulator": k_sim, "gamma_per_week": gamma_per_week, "n_weeks": N_WEEKS, "N": N_INITIAL}
    pd.Series(meta).to_csv(args.outdir / "quiet_contamination_meta.csv", header=["value"])

    thr_d = first_reversal_1d(rows_all, "dynamic", "epsilon_dyn")
    thr_c = first_reversal_1d(rows_all, "comorbidity", "epsilon_comorb")
    thr_pair = first_reversal_combined(rows_all)

    plot_figures(args.outdir, rd, rc, rx, h_ref)
    print_summary(thr_d, thr_c, thr_pair, h_ref)


if __name__ == "__main__":
    main()
