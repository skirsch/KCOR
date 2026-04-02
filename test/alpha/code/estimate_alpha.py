from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.special import gammaln


EPS = 1e-12
MANUSCRIPT_ALPHA_PAIR = 1.19
MANUSCRIPT_ALPHA_COLLAPSE = 1.18
NEUTRALIZATION_REFERENCE = "reference_anchored"
NEUTRALIZATION_SYMMETRIC = "symmetric_all_cohorts"
SUPPORTED_NEUTRALIZATION_MODES = {
    NEUTRALIZATION_REFERENCE,
    NEUTRALIZATION_SYMMETRIC,
}
INVARIANCE_TOL = 1e-10
SYNTHETIC_DOSE_REFERENCE = 0
SYNTHETIC_DOSE_VACCINATED = 2


@dataclass(frozen=True)
class WaveWindow:
    start_label: str
    end_label: str
    start_monday: object
    end_monday: object


def log_milestone(label: str, start_ts: float, last_ts: float) -> float:
    now = time.perf_counter()
    stage_s = now - last_ts
    total_s = now - start_ts
    print(f"[MILESTONE] {label} | stage={stage_s:.2f}s | total={total_s:.2f}s", flush=True)
    return now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate pandemic heterogeneity alpha")
    parser.add_argument("--config", required=True, help="Path to alpha yaml config")
    parser.add_argument("--outdir", required=True, help="Output directory")
    return parser.parse_args()


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def iso_label_to_monday(label: str):
    year, week = parse_iso_label(label)
    return pd.Timestamp.fromisocalendar(year, week, 1)


def parse_iso_label(label: str) -> tuple[int, int]:
    match = re.search(r"(\d{4})[-_]?(\d{1,2})", str(label).strip())
    if not match:
        raise ValueError(f"Bad ISO week label: {label!r}")
    return int(match.group(1)), int(match.group(2))


def iso_to_int(year: int, week: int) -> int:
    return year * 100 + week


def enrollment_label_to_monday(label: str):
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
            raise ValueError(f"Missing required column: {required}")
    return df.rename(columns=rename_map)


def expand_alpha_grid(cfg: dict) -> np.ndarray:
    start = float(cfg["start"])
    stop = float(cfg["stop"])
    step = float(cfg["step"])
    values = []
    current = start
    while current <= stop + step * 1e-9:
        values.append(round(current, 10))
        current += step
    return np.asarray(values, dtype=float)


def get_identifiability_cfg(cfg: dict) -> dict:
    defaults = {
        "min_normalized_curvature": 0.01,
        "max_estimator_gap": 0.03,
        "max_leave_one_out_shift": 0.03,
        "max_bootstrap_iqr": 0.10,
        "min_bootstrap_finite_fraction": 0.80,
        "max_bootstrap_boundary_fraction": 0.20,
    }
    user_cfg = cfg.get("identifiability") or {}
    merged = dict(defaults)
    merged.update(user_cfg)
    return merged


def get_wave_forcing_multipliers(cfg: dict) -> list[float]:
    synth_cfg = cfg.get("synthetic") or {}
    raw = synth_cfg.get("wave_forcing_multipliers", [1.0])
    if not raw:
        return [1.0]
    return [float(x) for x in raw]


def get_theta_strength_multipliers(cfg: dict) -> list[float]:
    synth_cfg = cfg.get("synthetic") or {}
    raw = synth_cfg.get("theta_strength_multipliers", [1.0])
    if not raw:
        return [1.0]
    return [float(x) for x in raw]


def wave_identifiability_experiment_enabled(cfg: dict) -> bool:
    synth_cfg = cfg.get("synthetic") or {}
    block = synth_cfg.get("wave_identifiability_experiment") or {}
    return bool(block.get("enabled", False))


def strict_synthetic_identified(
    cfg: dict,
    identified_curve: int,
    alpha_hat_reported: float,
    bootstrap_finite_fraction: float,
    bootstrap_iqr: float,
    bootstrap_boundary_fraction: float,
) -> int:
    """1 iff curve reports alpha_hat_reported and bootstrap stability gates pass (per estimator)."""
    ident_cfg = get_identifiability_cfg(cfg)
    reported_ok = int(identified_curve) == 1 and np.isfinite(alpha_hat_reported)
    bootstrap_ok = bool(
        float(bootstrap_finite_fraction) >= float(ident_cfg["min_bootstrap_finite_fraction"])
        and (not np.isfinite(bootstrap_iqr) or float(bootstrap_iqr) <= float(ident_cfg["max_bootstrap_iqr"]))
        and (
            not np.isfinite(bootstrap_boundary_fraction)
            or float(bootstrap_boundary_fraction) <= float(ident_cfg["max_bootstrap_boundary_fraction"])
        )
    )
    return int(reported_ok and bootstrap_ok)


def assert_wave_identifiability_config_clean(cfg: dict) -> None:
    synth = cfg.get("synthetic") or {}
    ve_cfg = synth.get("synthetic_vaccine_effect") or {}
    if bool(ve_cfg.get("enabled", False)):
        raise AssertionError("wave identifiability experiment: set synthetic.synthetic_vaccine_effect.enabled to false")
    if bool(synth.get("conditional_VE_enabled", False)):
        raise AssertionError("wave identifiability experiment: set synthetic.conditional_VE_enabled to false")


def synthetic_wave_amplitude_and_delta_h(cfg: dict, wave_multiplier: float) -> tuple[np.ndarray, np.ndarray]:
    synth_cfg = cfg["synthetic"]
    n_weeks = int(synth_cfg["n_weeks"])
    peak = float(synth_cfg["wave_amplitude_peak"])
    wave_phase = np.linspace(0.1, 0.95, n_weeks)
    wave_amplitude = peak * np.exp(-((wave_phase - 0.45) ** 2) / 0.03) * float(wave_multiplier)
    delta_h_path = np.cumsum(0.5 * wave_amplitude + 0.002)
    return wave_amplitude, delta_h_path


def assert_synthetic_wave_theta_defaults_match_table(cfg: dict) -> None:
    """Regression: explicit wave_multiplier=1 and theta_multiplier=1 match default simulate_synthetic_table."""
    synth_cfg = cfg.get("synthetic") or {}
    if not bool(synth_cfg.get("enabled", True)):
        return
    alpha_true = float(synth_cfg["alpha_true_values"][0])
    seed = int(synth_cfg["seed"])
    noise_model = str(synth_cfg.get("noise_models", ["lognormal_fixed"])[0])
    base = simulate_synthetic_table(cfg, alpha_true, seed, noise_model=noise_model, ve_multiplier=1.0)
    explicit = simulate_synthetic_table(
        cfg,
        alpha_true,
        seed,
        noise_model=noise_model,
        ve_multiplier=1.0,
        wave_multiplier=1.0,
        theta_multiplier=1.0,
    )
    if not np.allclose(base["excess"].to_numpy(dtype=float), explicit["excess"].to_numpy(dtype=float), rtol=0.0, atol=1e-9):
        raise AssertionError("wave/theta default multipliers changed synthetic excess vs baseline")


def _append_unique_float(values: list[float], candidate: object) -> None:
    try:
        numeric = float(candidate)
    except (TypeError, ValueError):
        return
    for existing in values:
        if math.isclose(existing, numeric, rel_tol=0.0, abs_tol=1e-12):
            return
    values.append(numeric)


def get_synthetic_ve_multipliers(cfg: dict) -> list[float]:
    synth_cfg = cfg.get("synthetic") or {}
    ve_cfg = synth_cfg.get("synthetic_vaccine_effect") or {}
    if not bool(ve_cfg.get("enabled", False)):
        return [1.0]
    values: list[float] = []
    mode_map = {
        "none": 1.0,
        "vaccinated_half_hazard": 0.5,
    }
    for mode in ve_cfg.get("modes", []):
        if mode not in mode_map:
            raise ValueError(f"Unknown synthetic vaccine-effect mode: {mode}")
        _append_unique_float(values, mode_map[mode])
    for item in synth_cfg.get("synthetic_vaccine_effect_values", []):
        _append_unique_float(values, item)
    if not values:
        values.append(1.0)
    return values


def conditional_ve_enabled(cfg: dict) -> bool:
    synth_cfg = cfg.get("synthetic") or {}
    return bool(synth_cfg.get("conditional_VE_enabled", False))


def get_conditional_ve_values(cfg: dict) -> list[float]:
    synth_cfg = cfg.get("synthetic") or {}
    values_cfg = synth_cfg.get("conditional_VE_values", [])
    values: list[float] = []
    for item in values_cfg:
        _append_unique_float(values, item)
    if not values:
        values.append(1.0)
    return values


def get_conditional_ve_target_multiplier(cfg: dict) -> float:
    synth_cfg = cfg.get("synthetic") or {}
    target = synth_cfg.get("conditional_VE_target_multiplier", 0.5)
    try:
        numeric = float(target)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Bad conditional_VE_target_multiplier: {target!r}") from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"conditional_VE_target_multiplier must be positive finite, got {target!r}")
    return numeric


def synthetic_week_monday(week_idx: int) -> pd.Timestamp:
    return pd.Timestamp("2021-01-04") + pd.to_timedelta(int(week_idx) * 7, unit="D")


def gamma_moment_alpha(theta: np.ndarray | float, alpha: float) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    out = np.ones_like(theta_arr, dtype=float)
    if abs(float(alpha) - 1.0) <= INVARIANCE_TOL:
        return out
    mask = np.isfinite(theta_arr) & (theta_arr > 1e-8)
    if np.any(mask):
        th = theta_arr[mask]
        inv = 1.0 / th
        out[mask] = np.exp(gammaln(alpha + inv) - gammaln(inv) + alpha * np.log(th))
    return out


def propagate_theta(theta_start: np.ndarray | float, delta_h: np.ndarray | float) -> np.ndarray:
    theta_arr = np.asarray(theta_start, dtype=float)
    delta_arr = np.asarray(delta_h, dtype=float)
    denom = 1.0 + np.maximum(theta_arr, 0.0) * np.maximum(delta_arr, 0.0)
    return np.maximum(theta_arr, 0.0) / np.maximum(denom, EPS)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    values = np.asarray(values, dtype=float)
    total = np.sum(weights)
    if total <= 0:
        return float(np.nanmean(values))
    return float(np.sum(weights * values) / total)


def build_floor(values: pd.Series) -> float:
    positives = np.asarray(values[values > 0], dtype=float)
    if positives.size == 0:
        return 1e-8
    return float(max(1e-8, np.nanquantile(positives, 0.10) * 0.10))


def transform_excess(values: np.ndarray, mode: str, floor: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    valid = np.isfinite(arr)
    if mode == "exclude_nonpositive":
        mask = valid & (arr > 0.0)
        out[mask] = np.log(np.maximum(arr[mask], floor))
        return out, mask
    if mode == "truncate_zero":
        mask = valid
        out[mask] = np.log(np.maximum(arr[mask], floor))
        return out, mask
    if mode == "signed_stable":
        mask = valid & (np.abs(arr) > 0.0)
        out[mask] = np.sign(arr[mask]) * np.log(np.maximum(np.abs(arr[mask]), floor))
        return out, mask
    raise ValueError(f"Unknown excess mode: {mode}")


def pairwise_weight(x: float, y: float, mode: str) -> float:
    if mode == "deaths_min":
        return float(max(min(x, y), 0.0))
    if mode == "sqrt_deaths":
        return float(max(math.sqrt(max(x, 0.0) * max(y, 0.0)), 0.0))
    return 1.0


def collapse_weight(values: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if mode in {"deaths", "alive"}:
        return np.clip(arr, 0.0, None)
    return np.ones_like(arr, dtype=float)


def build_time_segment_map(table: pd.DataFrame) -> dict[str, set[int]]:
    unique_weeks = sorted(table["iso_int"].dropna().astype(int).unique())
    if not unique_weeks:
        return {"pooled": set(), "early_wave": set(), "late_wave": set()}
    midpoint = len(unique_weeks) // 2
    early = set(unique_weeks[:midpoint])
    late = set(unique_weeks[midpoint:])
    return {"pooled": set(unique_weeks), "early_wave": early, "late_wave": late}


def age_band_mask(df: pd.DataFrame, age_band: str) -> pd.Series:
    if age_band == "pooled":
        return pd.Series(True, index=df.index)
    decade = int(age_band[:4])
    return df["yob_decade"].eq(decade)


def prepare_imports(repo_root: Path, dataset: str):
    code_dir = repo_root / "code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    os.environ["DATASET"] = dataset
    import KCOR  # type: ignore

    return KCOR


def build_wave_window(dataset_cfg: dict) -> WaveWindow:
    covid_cfg = dataset_cfg.get("NPH_correction") or {}
    start_label = str(covid_cfg["startDate"]).replace("_", "-")
    end_label = str(covid_cfg["endDate"]).replace("_", "-")
    return WaveWindow(
        start_label=start_label,
        end_label=end_label,
        start_monday=iso_label_to_monday(start_label),
        end_monday=iso_label_to_monday(end_label),
    )


def get_dataset_default_alpha(dataset_cfg: dict) -> float | None:
    cfg = dataset_cfg.get("NPH_correction") or {}
    value = cfg.get("default_alpha")
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def get_nph_neutralization_modes(cfg: dict) -> list[str]:
    raw_modes = cfg.get("nph_neutralization_mode")
    if raw_modes is None:
        return [NEUTRALIZATION_REFERENCE]
    if isinstance(raw_modes, str):
        modes = [raw_modes]
    else:
        modes = [str(mode) for mode in raw_modes]
    if not modes:
        raise ValueError("nph_neutralization_mode must contain at least one mode")
    invalid = [mode for mode in modes if mode not in SUPPORTED_NEUTRALIZATION_MODES]
    if invalid:
        raise ValueError(f"Unsupported nph_neutralization_mode values: {invalid}")
    deduped = list(dict.fromkeys(modes))
    return deduped


def get_primary_nph_neutralization_mode(cfg: dict) -> str:
    modes = get_nph_neutralization_modes(cfg)
    requested = str(cfg.get("primary_nph_neutralization_mode", NEUTRALIZATION_REFERENCE))
    if requested not in modes:
        raise ValueError(
            "primary_nph_neutralization_mode must be included in nph_neutralization_mode; "
            f"got {requested!r} not in {modes!r}"
        )
    return requested


def should_write_manuscript_figures(cfg: dict) -> bool:
    return bool(cfg.get("write_manuscript_figures", False))


def neutralize_excess(
    excess: np.ndarray | float,
    theta_vals: np.ndarray | float,
    alpha: float,
    neutralization_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    factors = np.maximum(gamma_moment_alpha(theta_vals, alpha), EPS)
    excess_arr = np.asarray(excess, dtype=float)
    if neutralization_mode == NEUTRALIZATION_REFERENCE:
        return excess_arr.copy(), factors
    if neutralization_mode == NEUTRALIZATION_SYMMETRIC:
        return excess_arr / factors, factors
    raise ValueError(f"Unknown neutralization mode: {neutralization_mode}")


def build_neutralized_frame(
    df: pd.DataFrame,
    alpha: float,
    *,
    theta_column: str,
    neutralization_mode: str,
) -> pd.DataFrame:
    out = df.copy()
    neutralized_excess, factors = neutralize_excess(
        out["excess"].to_numpy(dtype=float),
        out[theta_column].to_numpy(dtype=float),
        alpha,
        neutralization_mode,
    )
    out["frailty_factor"] = factors
    out["excess_neutralized"] = neutralized_excess
    out["hazard_adjusted"] = out["href"].to_numpy(dtype=float) + neutralized_excess
    return out


def _nanmax_abs(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0
    return float(np.max(np.abs(finite)))


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(w) & (w > 0)
    if int(np.count_nonzero(mask)) < 2:
        return float("nan")
    vals = vals[mask]
    w = w[mask]
    mu = weighted_mean(vals, w)
    total = np.sum(w)
    if total <= 0:
        return float("nan")
    return float(np.sum(w * (vals - mu) ** 2) / total)


def fit_cohort_theta(
    cohort_df: pd.DataFrame,
    enrollment_label: str,
    K,
) -> tuple[float, float, dict, np.ndarray, np.ndarray]:
    gom_cfg = K.get_gompertz_theta_fit_config()
    gamma_per_week = float(gom_cfg.get("gamma_per_week", gom_cfg.get("gamma_per_year", 0.085) / 52.0))
    k_anchor_weeks = int(gom_cfg.get("k_anchor_weeks", 4))

    hazard_eff = cohort_df["hazard_eff"].to_numpy(dtype=float)
    deaths = cohort_df["Dead"].to_numpy(dtype=float)
    t_vals = cohort_df["t_week"].to_numpy(dtype=float)
    iso_int = cohort_df["iso_int"].to_numpy(dtype=float)
    quiet_mask, theta_source, quiet_window_label = K._build_theta_quiet_mask(iso_int, t_vals)
    t_rebased = t_vals - float(K.DYNAMIC_HVE_SKIP_WEEKS)
    (k_hat, theta_hat), diag = K.fit_theta0_gompertz(
        hazard_eff,
        t_rebased,
        quiet_mask,
        k_anchor_weeks,
        gamma_per_week,
        deaths_arr=deaths,
    )
    diag = dict(diag)
    diag["theta_source"] = theta_source
    diag["quiet_window_label"] = quiet_window_label
    return float(k_hat), float(theta_hat), diag, quiet_mask, t_rebased


def build_real_cohort_table(repo_root: Path, cfg: dict, K) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = str(cfg["dataset"])
    params = cfg["analysis"]
    dataset_cfg = load_yaml(repo_root / "data" / dataset / f"{dataset}.yaml")
    wave = build_wave_window(dataset_cfg)
    source_path = repo_root / "data" / dataset / "KCOR_CMR.xlsx"

    selected_decades = set(int(x) for x in params["yob_decades"])
    selected_doses = set(int(x) for x in params["doses"])

    wave_rows: list[dict] = []
    cohort_rows: list[dict] = []

    for enrollment_label in params["enrollment_dates"]:
        sheet_name = str(enrollment_label)
        print(f"[ALPHA] building cohort-week rows for {sheet_name}", flush=True)
        raw = pd.read_excel(source_path, sheet_name=sheet_name)
        df = standardize_sheet_columns(raw)
        df["YearOfBirth"] = df["YearOfBirth"].astype(int)
        df["yob_decade"] = (df["YearOfBirth"] // 10) * 10
        df = df[df["yob_decade"].isin(selected_decades)].copy()
        df = df[df["Dose"].isin(selected_doses)].copy()

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
        df["iso_int"] = [iso_to_int(y, w) for y, w in zip(df["iso_year"], df["iso_week"])]

        grouped = (
            df.groupby(["yob_decade", "Dose", "iso_year", "iso_week", "iso_int", "week_monday", "t_week"], as_index=False)
            .agg({"Dead": "sum", "Alive": "sum"})
            .sort_values(["yob_decade", "Dose", "t_week"])
        )
        grouped = grouped[grouped["Alive"] > 0].copy()
        grouped["MR"] = np.clip(grouped["Dead"] / grouped["Alive"], 0.0, 1.0 - 1e-12)
        grouped["hazard_obs"] = K.hazard_from_mr(grouped["MR"].to_numpy(dtype=float))
        grouped["hazard_eff"] = np.where(
            grouped["t_week"].to_numpy(dtype=float) >= float(K.DYNAMIC_HVE_SKIP_WEEKS),
            grouped["hazard_obs"].to_numpy(dtype=float),
            0.0,
        )

        for (yob_decade, dose), cohort_df in grouped.groupby(["yob_decade", "Dose"]):
            cohort_df = cohort_df.sort_values("t_week").reset_index(drop=True)
            cohort_id = f"{sheet_name}|{int(yob_decade)}|{int(dose)}"
            cohort_df["H_obs"] = np.cumsum(cohort_df["hazard_eff"].to_numpy(dtype=float))

            k_hat, theta_hat, diag, quiet_mask, t_rebased = fit_cohort_theta(cohort_df, sheet_name, K)
            theta_fit_ok = bool(diag.get("success")) and np.isfinite(theta_hat)
            H_gamma = K.invert_gamma_frailty(cohort_df["H_obs"].to_numpy(dtype=float), theta_hat if np.isfinite(theta_hat) else 0.0)
            H_raw = cohort_df["H_obs"].to_numpy(dtype=float)

            wave_mask = (
                (cohort_df["week_monday"] >= wave.start_monday)
                & (cohort_df["week_monday"] <= wave.end_monday)
            )
            wave_idx = np.where(wave_mask.to_numpy(dtype=bool))[0]
            wave_deaths = float(cohort_df.loc[wave_mask, "Dead"].sum())
            wave_weeks = int(np.count_nonzero(wave_mask))
            if wave_idx.size == 0:
                exclusion = "no_wave_overlap"
            elif not theta_fit_ok:
                exclusion = "bad_theta_fit"
            elif wave_deaths < float(params["min_wave_deaths"]):
                exclusion = "wave_deaths_too_low"
            elif wave_weeks < int(params["min_wave_weeks"]):
                exclusion = "wave_weeks_too_low"
            else:
                exclusion = ""

            if wave_idx.size > 0:
                start_idx = int(wave_idx[0])
                H_start_gamma = float(H_gamma[start_idx])
                H_start_raw = float(H_raw[start_idx])
                theta_w_gamma = float(propagate_theta(theta_hat if np.isfinite(theta_hat) else 0.0, H_start_gamma))
                theta_w_raw = float(propagate_theta(theta_hat if np.isfinite(theta_hat) else 0.0, H_start_raw))
                wave_mid_idx = start_idx + len(wave_idx) // 2
            else:
                start_idx = 0
                H_start_gamma = 0.0
                H_start_raw = 0.0
                theta_w_gamma = float("nan")
                theta_w_raw = float("nan")
                wave_mid_idx = 0

            cohort_rows.append(
                {
                    "cohort_id": cohort_id,
                    "enrollment_date": sheet_name,
                    "yob_decade": int(yob_decade),
                    "dose": int(dose),
                    "k_hat": k_hat,
                    "theta0_hat": theta_hat,
                    "theta_fit_ok": int(theta_fit_ok),
                    "theta_fit_status": diag.get("status"),
                    "theta_source": diag.get("theta_source"),
                    "quiet_window_label": diag.get("quiet_window_label"),
                    "wave_deaths": wave_deaths,
                    "wave_weeks": wave_weeks,
                    "theta_w_gamma": theta_w_gamma,
                    "theta_w_raw": theta_w_raw,
                    "eligible": int(exclusion == ""),
                    "exclusion_reason": exclusion or "eligible",
                }
            )

            wave_subset = cohort_df.loc[wave_mask].copy()
            for row in wave_subset.itertuples(index=True):
                idx = int(row.Index)
                delta_gamma = max(float(H_gamma[idx] - H_start_gamma), 0.0)
                delta_raw = max(float(H_raw[idx] - H_start_raw), 0.0)
                theta_t_gamma = float(propagate_theta(theta_w_gamma, delta_gamma)) if np.isfinite(theta_w_gamma) else float("nan")
                theta_t_raw = float(propagate_theta(theta_w_raw, delta_raw)) if np.isfinite(theta_w_raw) else float("nan")
                time_segment = "early_wave" if idx < wave_mid_idx else "late_wave"
                wave_rows.append(
                    {
                        "cohort_id": cohort_id,
                        "enrollment_date": sheet_name,
                        "yob_decade": int(yob_decade),
                        "dose": int(dose),
                        "iso_year": int(row.iso_year),
                        "iso_week": int(row.iso_week),
                        "iso_int": int(row.iso_int),
                        "week_monday": row.week_monday,
                        "t_week": int(row.t_week),
                        "Dead": float(row.Dead),
                        "Alive": float(row.Alive),
                        "hazard_obs": float(row.hazard_obs),
                        "hazard_eff": float(row.hazard_eff),
                        "H_obs": float(row.H_obs),
                        "H_gamma": float(H_gamma[idx]),
                        "theta0_hat": theta_hat,
                        "theta_w_gamma": theta_w_gamma,
                        "theta_w_raw": theta_w_raw,
                        "theta_t_gamma": theta_t_gamma,
                        "theta_t_raw": theta_t_raw,
                        "theta_fit_ok": int(theta_fit_ok),
                        "eligible": int(exclusion == ""),
                        "exclusion_reason": exclusion or "eligible",
                        "time_segment": time_segment,
                    }
                )

    wave_table = pd.DataFrame(wave_rows).sort_values(["week_monday", "enrollment_date", "yob_decade", "dose"])
    cohort_diag = pd.DataFrame(cohort_rows).sort_values(["enrollment_date", "yob_decade", "dose"])
    return wave_table, cohort_diag


def add_reference_excess(table: pd.DataFrame, anchor: str) -> pd.DataFrame:
    df = table.copy()
    stratum = ["enrollment_date", "yob_decade", "week_monday"]
    if anchor == "dose0":
        anchor_rows = df[df["dose"] == 0][stratum + ["hazard_obs"]].rename(columns={"hazard_obs": "href"})
        df = df.merge(anchor_rows, on=stratum, how="left")
    elif anchor == "dose1":
        anchor_rows = df[df["dose"] == 1][stratum + ["hazard_obs"]].rename(columns={"hazard_obs": "href"})
        df = df.merge(anchor_rows, on=stratum, how="left")
    elif anchor == "pooled_median":
        href = df.groupby(stratum)["hazard_obs"].median().reset_index().rename(columns={"hazard_obs": "href"})
        df = df.merge(href, on=stratum, how="left")
    else:
        raise ValueError(f"Unknown anchor: {anchor}")
    df["excess"] = df["hazard_obs"] - df["href"]
    df["anchor_mode"] = anchor
    return df


def evaluate_pairwise_objective(
    subset: pd.DataFrame,
    alpha_values: np.ndarray,
    excess_mode: str,
    weight_mode: str,
    theta_column: str,
    min_pairs: int,
    neutralization_mode: str,
) -> list[dict]:
    floor = build_floor(subset["excess"])
    grouped_weeks: list[dict[str, np.ndarray]] = []
    for _, group in subset.groupby("iso_int"):
        grouped_weeks.append(
            {
                "excess": group["excess"].to_numpy(dtype=float),
                "theta_vals": group[theta_column].to_numpy(dtype=float),
                "weights": group["Dead"].to_numpy(dtype=float),
            }
        )
    records: list[dict] = []
    for alpha in alpha_values:
        obj = 0.0
        n_pairs = 0
        n_weeks_used = 0
        for group in grouped_weeks:
            theta_vals = group["theta_vals"]
            if neutralization_mode == NEUTRALIZATION_REFERENCE:
                transformed, valid = transform_excess(group["excess"], excess_mode, floor)
                log_factor = np.log(np.maximum(gamma_moment_alpha(theta_vals, alpha), floor))
            else:
                neutralized_excess, _ = neutralize_excess(group["excess"], theta_vals, alpha, neutralization_mode)
                transformed, valid = transform_excess(neutralized_excess, excess_mode, floor)
                log_factor = np.zeros_like(theta_vals, dtype=float)
            theta_vals = group["theta_vals"]
            weights = group["weights"]
            idx = np.where(valid & np.isfinite(theta_vals))[0]
            if idx.size < 2:
                continue
            used_here = 0
            for i_pos, i in enumerate(idx[:-1]):
                for j in idx[i_pos + 1 :]:
                    w = pairwise_weight(weights[i], weights[j], weight_mode)
                    if w <= 0:
                        continue
                    observed = transformed[i] - transformed[j]
                    predicted = log_factor[i] - log_factor[j]
                    resid = observed - predicted
                    obj += w * resid * resid
                    n_pairs += 1
                    used_here += 1
            if used_here > 0:
                n_weeks_used += 1
        records.append(
            {
                "alpha": float(alpha),
                "objective": float(obj) if n_pairs >= min_pairs else np.nan,
                "n_pairs": int(n_pairs),
                "n_weeks_used": int(n_weeks_used),
            }
        )
    return records


def evaluate_collapse_objective(
    subset: pd.DataFrame,
    alpha_values: np.ndarray,
    excess_mode: str,
    weight_mode: str,
    theta_column: str,
    min_cohorts_per_week: int,
    neutralization_mode: str,
) -> list[dict]:
    floor = build_floor(subset["excess"])
    grouped_weeks: list[dict[str, np.ndarray]] = []
    for _, group in subset.groupby("iso_int"):
        grouped_weeks.append(
            {
                "excess": group["excess"].to_numpy(dtype=float),
                "theta_vals": group[theta_column].to_numpy(dtype=float),
                "weights": collapse_weight(group["Dead"].to_numpy(dtype=float), weight_mode),
            }
        )
    records: list[dict] = []
    for alpha in alpha_values:
        obj = 0.0
        n_points = 0
        n_weeks_used = 0
        for group in grouped_weeks:
            theta_vals = group["theta_vals"]
            neutralized_excess, _ = neutralize_excess(group["excess"], theta_vals, alpha, neutralization_mode)
            transformed, valid = transform_excess(neutralized_excess, excess_mode, floor)
            weights = group["weights"]
            mask = valid & np.isfinite(theta_vals)
            if int(np.count_nonzero(mask)) < min_cohorts_per_week:
                continue
            vals = transformed[mask]
            w = weights[mask]
            mu = weighted_mean(vals, w)
            obj += float(np.sum(w * (vals - mu) ** 2))
            n_points += int(np.count_nonzero(mask))
            n_weeks_used += 1
        records.append(
            {
                "alpha": float(alpha),
                "objective": float(obj) if n_points >= min_cohorts_per_week else np.nan,
                "n_pairs": int(n_points),
                "n_weeks_used": int(n_weeks_used),
            }
        )
    return records


def summarize_best_curve(curve_df: pd.DataFrame, cfg: dict) -> dict | None:
    curve_df = curve_df[np.isfinite(curve_df["objective"])].copy()
    if curve_df.empty:
        return None
    curve_df = curve_df.sort_values("alpha").reset_index(drop=True)
    best_idx = int(curve_df["objective"].idxmin())
    best = curve_df.iloc[best_idx]
    obj_vals = curve_df["objective"].to_numpy(dtype=float)
    alpha_vals = curve_df["alpha"].to_numpy(dtype=float)
    objective_range = float(np.nanmax(obj_vals) - np.nanmin(obj_vals)) if obj_vals.size else float("nan")
    at_boundary = best_idx == 0 or best_idx == (len(curve_df) - 1)
    curvature_metric = float("nan")
    if not at_boundary and np.isfinite(objective_range) and objective_range > EPS:
        neighbor_mean = 0.5 * (obj_vals[best_idx - 1] + obj_vals[best_idx + 1])
        curvature_metric = float((neighbor_mean - obj_vals[best_idx]) / objective_range)
    ident_cfg = get_identifiability_cfg(cfg)
    is_identified = (not at_boundary) and np.isfinite(curvature_metric) and curvature_metric >= float(ident_cfg["min_normalized_curvature"])
    status = "identified"
    if at_boundary:
        status = "boundary_seeking"
    elif not np.isfinite(curvature_metric) or curvature_metric < float(ident_cfg["min_normalized_curvature"]):
        status = "low_curvature"
    return {
        "alpha_hat": float(best["alpha"]),
        "alpha_hat_reported": float(best["alpha"]) if is_identified else np.nan,
        "objective": float(best["objective"]),
        "n_pairs": int(best["n_pairs"]),
        "n_weeks_used": int(best["n_weeks_used"]),
        "curvature_metric": curvature_metric,
        "objective_range": objective_range,
        "boundary_optimum": int(at_boundary),
        "identified_curve": int(is_identified),
        "identification_status": status,
        "alpha_min_grid": float(np.nanmin(alpha_vals)),
        "alpha_max_grid": float(np.nanmax(alpha_vals)),
    }


def evaluate_real_data(
    wave_table: pd.DataFrame,
    cfg: dict,
    alpha_values: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = cfg["analysis"]
    neutralization_modes = get_nph_neutralization_modes(cfg)
    time_map = build_time_segment_map(wave_table)
    theta_scales = {
        "gamma_primary": "theta_t_gamma",
        "raw_observed": "theta_t_raw",
    }

    all_curves: list[dict] = []
    best_rows: list[dict] = []

    for neutralization_mode in neutralization_modes:
        for anchor in params["anchor_choices"]:
            anchored = add_reference_excess(wave_table, anchor)
            anchored = anchored[np.isfinite(anchored["href"])].copy()
            anchored["weight_deaths"] = anchored["Dead"].clip(lower=0.0)
            for theta_scale, theta_column in theta_scales.items():
                scale_df = anchored[np.isfinite(anchored[theta_column])].copy()
                scale_df = scale_df[scale_df["eligible"] == 1].copy()
                for excess_mode in params["excess_modes"]:
                    for age_band in params["age_bands"]:
                        age_df = scale_df[age_band_mask(scale_df, age_band)].copy()
                        if age_df.empty:
                            continue
                        for time_segment in params["time_segments"]:
                            week_set = time_map.get(time_segment, set())
                            segment_df = age_df[age_df["iso_int"].isin(week_set)] if time_segment != "pooled" else age_df
                            if segment_df.empty:
                                continue
                            for estimator in ("pairwise", "collapse"):
                                if estimator == "pairwise":
                                    curve = evaluate_pairwise_objective(
                                        segment_df,
                                        alpha_values,
                                        excess_mode,
                                        params["weight_mode_pairwise"],
                                        theta_column,
                                        int(params["min_pairs_per_alpha"]),
                                        neutralization_mode,
                                    )
                                else:
                                    curve = evaluate_collapse_objective(
                                        segment_df,
                                        alpha_values,
                                        excess_mode,
                                        params["weight_mode_collapse"],
                                        theta_column,
                                        int(params["min_cohorts_per_week"]),
                                        neutralization_mode,
                                    )
                                curve_df = pd.DataFrame(curve)
                                if curve_df.empty:
                                    continue
                                curve_df["neutralization_mode"] = neutralization_mode
                                curve_df["anchor_mode"] = anchor
                                curve_df["theta_scale"] = theta_scale
                                curve_df["excess_mode"] = excess_mode
                                curve_df["age_band"] = age_band
                                curve_df["time_segment"] = time_segment
                                curve_df["estimator"] = estimator
                                all_curves.extend(curve_df.to_dict(orient="records"))
                                best = summarize_best_curve(curve_df, cfg)
                                if best is not None:
                                    best.update(
                                        {
                                            "neutralization_mode": neutralization_mode,
                                            "anchor_mode": anchor,
                                            "theta_scale": theta_scale,
                                            "excess_mode": excess_mode,
                                            "age_band": age_band,
                                            "time_segment": time_segment,
                                            "estimator": estimator,
                                            "n_cohorts": int(segment_df["cohort_id"].nunique()),
                                        }
                                    )
                                    best_rows.append(best)

    return pd.DataFrame(all_curves), pd.DataFrame(best_rows)


def build_primary_subset(wave_table: pd.DataFrame, cfg: dict, neutralization_mode: str) -> pd.DataFrame:
    params = cfg["analysis"]
    df = add_reference_excess(wave_table, params["primary_anchor"])
    df = df[df["eligible"] == 1].copy()
    df = df[np.isfinite(df["href"]) & np.isfinite(df["theta_t_gamma"])].copy()
    df["neutralization_mode"] = neutralization_mode
    return df


def build_theta_scale_summary(best_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    params = cfg["analysis"]
    primary_mode = get_primary_nph_neutralization_mode(cfg)
    summary = best_df[
        (best_df["neutralization_mode"] == primary_mode)
        & (best_df["anchor_mode"] == params["primary_anchor"])
        & (best_df["excess_mode"] == params["primary_excess_mode"])
        & (best_df["age_band"] == "pooled")
        & (best_df["time_segment"] == "pooled")
    ].copy()
    if summary.empty:
        return summary
    summary = summary.sort_values(["estimator", "theta_scale"]).reset_index(drop=True)
    return summary[
        [
            "neutralization_mode",
            "theta_scale",
            "estimator",
            "anchor_mode",
            "excess_mode",
            "alpha_hat",
            "alpha_hat_reported",
            "objective",
            "n_pairs",
            "n_weeks_used",
            "n_cohorts",
            "curvature_metric",
            "boundary_optimum",
            "identified_curve",
            "identification_status",
        ]
    ]


def evaluate_single_estimator_curve(
    primary_df: pd.DataFrame,
    cfg: dict,
    alpha_values: np.ndarray,
    neutralization_mode: str,
    estimator: str,
) -> pd.DataFrame:
    params = cfg["analysis"]
    if estimator == "pairwise":
        return pd.DataFrame(
            evaluate_pairwise_objective(
                primary_df,
                alpha_values,
                params["primary_excess_mode"],
                params["weight_mode_pairwise"],
                "theta_t_gamma",
                int(params["min_pairs_per_alpha"]),
                neutralization_mode,
            )
        )
    if estimator == "collapse":
        return pd.DataFrame(
            evaluate_collapse_objective(
                primary_df,
                alpha_values,
                params["primary_excess_mode"],
                params["weight_mode_collapse"],
                "theta_t_gamma",
                int(params["min_cohorts_per_week"]),
                neutralization_mode,
            )
        )
    raise ValueError(f"Unknown estimator: {estimator}")


def leave_one_out_analysis(
    primary_df: pd.DataFrame,
    cfg: dict,
    alpha_values: np.ndarray,
    neutralization_mode: str,
    estimator: str = "pairwise",
) -> pd.DataFrame:
    rows = []
    eligible_ids = sorted(primary_df["cohort_id"].dropna().unique())
    for cohort_id in eligible_ids:
        subset = primary_df[primary_df["cohort_id"] != cohort_id].copy()
        curve = evaluate_single_estimator_curve(subset, cfg, alpha_values, neutralization_mode, estimator)
        best = summarize_best_curve(curve, cfg)
        rows.append(
            {
                "neutralization_mode": neutralization_mode,
                "estimator": estimator,
                "left_out_cohort": cohort_id,
                "alpha_hat": np.nan if best is None else best["alpha_hat"],
                "alpha_hat_reported": np.nan if best is None else best["alpha_hat_reported"],
                "objective": np.nan if best is None else best["objective"],
                "n_pairs": 0 if best is None else best["n_pairs"],
                "curvature_metric": np.nan if best is None else best["curvature_metric"],
                "boundary_optimum": 0 if best is None else best["boundary_optimum"],
                "identified_curve": 0 if best is None else best["identified_curve"],
                "identification_status": "missing" if best is None else best["identification_status"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_alpha(
    primary_df: pd.DataFrame,
    cfg: dict,
    alpha_values: np.ndarray,
    neutralization_mode: str,
    estimator: str | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    boot_cfg = cfg["bootstrap"]
    estimator_name = str(estimator or boot_cfg["summary_estimator"])
    rng = np.random.default_rng(int(boot_cfg["seed"] if seed is None else seed))
    cohort_ids = np.asarray(sorted(primary_df["cohort_id"].unique()))
    rows = []
    for rep in range(int(boot_cfg["reps"])):
        sampled = rng.choice(cohort_ids, size=len(cohort_ids), replace=True)
        pieces = []
        for idx, cohort_id in enumerate(sampled):
            part = primary_df[primary_df["cohort_id"] == cohort_id].copy()
            part["cohort_id"] = f"{cohort_id}#b{rep}_{idx}"
            pieces.append(part)
        subset = pd.concat(pieces, ignore_index=True) if pieces else primary_df.iloc[0:0].copy()
        curve = evaluate_single_estimator_curve(subset, cfg, alpha_values, neutralization_mode, estimator_name)
        best = summarize_best_curve(curve, cfg)
        rows.append(
            {
                "neutralization_mode": neutralization_mode,
                "estimator": estimator_name,
                "bootstrap_rep": rep,
                "alpha_hat": np.nan if best is None else best["alpha_hat"],
                "alpha_hat_reported": np.nan if best is None else best["alpha_hat_reported"],
                "objective": np.nan if best is None else best["objective"],
                "curvature_metric": np.nan if best is None else best["curvature_metric"],
                "boundary_optimum": 0 if best is None else best["boundary_optimum"],
                "identified_curve": 0 if best is None else best["identified_curve"],
                "identification_status": "missing" if best is None else best["identification_status"],
            }
        )
    return pd.DataFrame(rows)


def simulate_synthetic_table(
    cfg: dict,
    alpha_true: float,
    seed: int,
    noise_model: str = "lognormal_fixed",
    ve_multiplier: float = 1.0,
    wave_multiplier: float = 1.0,
    theta_multiplier: float = 1.0,
) -> pd.DataFrame:
    synth_cfg = cfg["synthetic"]
    rng = np.random.default_rng(seed)
    n_cohorts = int(synth_cfg["n_cohorts"])
    n_weeks = int(synth_cfg["n_weeks"])
    theta_values = np.asarray(synth_cfg["theta_w_values"], dtype=float)
    peak = float(synth_cfg["wave_amplitude_peak"])
    baseline_weight = float(synth_cfg["baseline_weight"])
    lognormal_sigma = float(synth_cfg.get("lognormal_sigma", 0.03))
    heteroskedastic_scale = float(synth_cfg.get("heteroskedastic_scale", 0.35))

    wave_amplitude, delta_h_path = synthetic_wave_amplitude_and_delta_h(cfg, wave_multiplier)
    baseline_hazard = peak * 0.10
    cohort_strata = [
        (1930, SYNTHETIC_DOSE_REFERENCE),
        (1930, SYNTHETIC_DOSE_VACCINATED),
        (1940, SYNTHETIC_DOSE_REFERENCE),
        (1940, SYNTHETIC_DOSE_VACCINATED),
        (1950, SYNTHETIC_DOSE_REFERENCE),
        (1950, SYNTHETIC_DOSE_VACCINATED),
    ]

    rows = []
    for cohort_idx in range(n_cohorts):
        theta_w = float(theta_values[cohort_idx % len(theta_values)]) * float(theta_multiplier)
        yob_decade, dose = cohort_strata[cohort_idx % len(cohort_strata)]
        enrollment_group = cohort_idx // len(cohort_strata)
        cohort_id = f"synthetic_{yob_decade}_{dose}_{enrollment_group:02d}"
        enrollment_date = f"synthetic_{enrollment_group:02d}"
        dose_multiplier = 1.0 if int(dose) == SYNTHETIC_DOSE_REFERENCE else float(ve_multiplier)
        for week_idx in range(n_weeks):
            week_monday = synthetic_week_monday(week_idx)
            iso = week_monday.isocalendar()
            theta_t = float(propagate_theta(theta_w, delta_h_path[week_idx] - delta_h_path[0]))
            factor = float(gamma_moment_alpha(theta_t, alpha_true))
            expected_excess = wave_amplitude[week_idx] * factor
            deaths_proxy = max(expected_excess * baseline_weight, 1e-6)
            if noise_model == "lognormal_fixed":
                sigma = lognormal_sigma
                noise = float(np.exp(rng.normal(0.0, sigma)))
                observed_excess = expected_excess * noise
            elif noise_model == "heteroskedastic_lognormal":
                sigma = heteroskedastic_scale / math.sqrt(deaths_proxy)
                sigma = float(np.clip(sigma, 0.02, 0.35))
                noise = float(np.exp(rng.normal(0.0, sigma)))
                observed_excess = expected_excess * noise
            else:
                raise ValueError(f"Unknown synthetic noise_model: {noise_model}")
            excess = observed_excess * dose_multiplier
            hazard_obs = baseline_hazard + excess
            rows.append(
                {
                    "cohort_id": cohort_id,
                    "enrollment_date": enrollment_date,
                    "dose": int(dose),
                    "iso_int": week_idx,
                    "iso_year": int(iso.year),
                    "iso_week": int(iso.week),
                    "week_monday": week_monday,
                    "t_week": week_idx,
                    "time_segment": "early_wave" if week_idx < (n_weeks // 2) else "late_wave",
                    "yob_decade": yob_decade,
                    "Dead": baseline_weight * max(excess, 0.001),
                    "Alive": baseline_weight,
                    "hazard_obs": hazard_obs,
                    "excess": excess,
                    "noise_model": noise_model,
                    "ve_multiplier": float(ve_multiplier),
                    "theta_t_gamma": theta_t,
                    "theta_t_raw": theta_t,
                    "eligible": 1,
                }
            )
    return pd.DataFrame(rows)


def build_synthetic_primary_subset(table: pd.DataFrame, neutralization_mode: str) -> pd.DataFrame:
    df = table.copy()
    df = df[df["eligible"] == 1].copy()
    df = df[np.isfinite(df["theta_t_gamma"]) & np.isfinite(df["excess"])].copy()
    df["neutralization_mode"] = neutralization_mode
    return df


def apply_conditional_ve_adjustment(primary_df: pd.DataFrame, ve_assumed: float) -> pd.DataFrame:
    assumed = float(ve_assumed)
    if not np.isfinite(assumed) or assumed <= 0.0:
        raise ValueError(f"VE_assumed must be positive finite, got {ve_assumed!r}")
    adjusted = primary_df.copy()
    adjusted["excess_observed"] = adjusted["excess"].to_numpy(dtype=float)
    adjusted["VE_assumed"] = assumed
    vaccinated_mask = adjusted["dose"].to_numpy(dtype=float) > 0.0 if "dose" in adjusted.columns else np.zeros(len(adjusted), dtype=bool)
    adjusted["conditional_ve_adjusted"] = vaccinated_mask.astype(int)
    if np.any(vaccinated_mask):
        adjusted.loc[vaccinated_mask, "excess"] = adjusted.loc[vaccinated_mask, "excess_observed"] / assumed
    return adjusted


def validate_conditional_ve_adjustment(original_df: pd.DataFrame, adjusted_df: pd.DataFrame, ve_assumed: float) -> None:
    if len(original_df) != len(adjusted_df):
        raise ValueError("Conditional VE adjustment changed row count")
    ref_mask = original_df["dose"].to_numpy(dtype=float) <= 0.0 if "dose" in original_df.columns else np.ones(len(original_df), dtype=bool)
    vacc_mask = ~ref_mask
    orig_excess = original_df["excess"].to_numpy(dtype=float)
    adj_excess = adjusted_df["excess"].to_numpy(dtype=float)
    if np.any(ref_mask) and not np.allclose(orig_excess[ref_mask], adj_excess[ref_mask], equal_nan=True):
        raise ValueError("Conditional VE adjustment modified non-vaccinated cohorts")
    if np.any(vacc_mask):
        expected = orig_excess[vacc_mask] / float(ve_assumed)
        if not np.allclose(expected, adj_excess[vacc_mask], equal_nan=True):
            raise ValueError("Conditional VE adjustment did not match expected vaccinated-cohort scaling")


def evaluate_synthetic_primary(
    primary_df: pd.DataFrame,
    cfg: dict,
    alpha_values: np.ndarray,
    neutralization_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common = {
        "neutralization_mode": neutralization_mode,
        "anchor_mode": "synthetic_direct",
        "theta_scale": "gamma_primary",
        "excess_mode": cfg["analysis"]["primary_excess_mode"],
        "age_band": "pooled",
        "time_segment": "pooled",
        "n_cohorts": int(primary_df["cohort_id"].nunique()),
    }
    curve_rows: list[dict] = []
    best_rows: list[dict] = []
    for estimator in ("pairwise", "collapse"):
        curve_df = evaluate_single_estimator_curve(primary_df, cfg, alpha_values, neutralization_mode, estimator)
        curve_df["estimator"] = estimator
        for key, value in common.items():
            curve_df[key] = value
        curve_rows.extend(curve_df.to_dict(orient="records"))
        best = summarize_best_curve(curve_df, cfg)
        if best is None:
            continue
        best.update(common)
        best["estimator"] = estimator
        best_rows.append(best)
    return pd.DataFrame(curve_rows), pd.DataFrame(best_rows)


def evaluate_synthetic_best_with_diagnostics(
    primary_df: pd.DataFrame,
    cfg: dict,
    alpha_values: np.ndarray,
    neutralization_mode: str,
    seed_base: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, best_df = evaluate_synthetic_primary(primary_df, cfg, alpha_values, neutralization_mode)
    if best_df.empty or set(best_df["estimator"]) != {"pairwise", "collapse"}:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    pair_row = best_df[best_df["estimator"] == "pairwise"].iloc[0]
    collapse_row = best_df[best_df["estimator"] == "collapse"].iloc[0]
    estimator_gap = abs(float(pair_row["alpha_hat"]) - float(collapse_row["alpha_hat"]))
    metric_rows: list[dict] = []
    regression_rows: list[dict] = []
    for estimator in ("pairwise", "collapse"):
        boot_df = bootstrap_alpha(
            primary_df,
            cfg,
            alpha_values,
            neutralization_mode,
            estimator=estimator,
            seed=seed_base + (17 if estimator == "pairwise" else 29),
        )
        for column, value in {
            "anchor_mode": "synthetic_direct",
            "theta_scale": "gamma_primary",
            "excess_mode": cfg["analysis"]["primary_excess_mode"],
            "age_band": "pooled",
            "time_segment": "pooled",
        }.items():
            boot_df[column] = value
        boot_summary = build_bootstrap_summary(boot_df, cfg)
        loo_df = leave_one_out_analysis(
            primary_df,
            cfg,
            alpha_values,
            neutralization_mode,
            estimator=estimator,
        )
        for column, value in {
            "anchor_mode": "synthetic_direct",
            "theta_scale": "gamma_primary",
            "excess_mode": cfg["analysis"]["primary_excess_mode"],
            "age_band": "pooled",
            "time_segment": "pooled",
        }.items():
            loo_df[column] = value
        loo_summary = build_leave_one_out_summary(loo_df, best_df, cfg)
        best_row = pair_row if estimator == "pairwise" else collapse_row
        boot_row = boot_summary.iloc[0]
        loo_row = loo_summary.iloc[0]
        metric_rows.append(
            {
                "estimator": estimator,
                "alpha_hat_raw": float(best_row["alpha_hat"]),
                "alpha_hat_reported": np.nan if not np.isfinite(best_row["alpha_hat_reported"]) else float(best_row["alpha_hat_reported"]),
                "identified": int(best_row["identified_curve"]),
                "identification_status": str(best_row["identification_status"]),
                "curvature_metric": float(best_row["curvature_metric"]),
                "boundary_optimum": int(best_row["boundary_optimum"]),
                "objective": float(best_row["objective"]),
                "estimator_gap": float(estimator_gap),
                "bootstrap_iqr": float(boot_row["iqr_alpha_hat"]),
                "bootstrap_boundary_fraction": float(boot_row["boundary_fraction"]),
                "bootstrap_finite_fraction": float(boot_row["finite_fraction"]),
                "leave_one_out_max_shift": float(loo_row["max_abs_shift"]),
                "leave_one_out_boundary_count": int(loo_row["boundary_count"]),
            }
        )
        regression_rows.append(
            {
                "estimator": estimator,
                "alpha_hat_raw": float(best_row["alpha_hat"]),
            }
        )
    return best_df, pd.DataFrame(metric_rows), pd.DataFrame(regression_rows)


def synthetic_recovery(cfg: dict, alpha_values: np.ndarray) -> dict[str, object]:
    synth_cfg = cfg["synthetic"]
    if not bool(synth_cfg.get("enabled", True)):
        return {
            "legacy_df": pd.DataFrame(),
            "ve_recovery_df": pd.DataFrame(),
            "ve_summary_df": pd.DataFrame(),
            "ve_report": "",
        }
    base_seed = int(synth_cfg["seed"])
    reps = int(synth_cfg.get("reps", 8))
    noise_models = list(synth_cfg.get("noise_models", ["lognormal_fixed"]))
    ve_recovery_rows: list[dict] = []
    ve_multipliers = get_synthetic_ve_multipliers(cfg)
    regression_summary = []
    regression_check_rows = []
    for noise_idx, noise_model in enumerate(noise_models):
        for alpha_idx, alpha_true in enumerate(synth_cfg["alpha_true_values"]):
            print(
                f"[ALPHA] synthetic recovery for alpha_true={float(alpha_true):.3f} "
                f"(noise_model={noise_model})",
                flush=True,
            )
            for ve_idx, ve_multiplier in enumerate(ve_multipliers):
                for rep in range(reps):
                    combo_seed = (
                        base_seed
                        + rep
                        + 1000 * alpha_idx
                        + 10000 * noise_idx
                        + 100000 * ve_idx
                    )
                    table = simulate_synthetic_table(
                        cfg,
                        float(alpha_true),
                        combo_seed,
                        noise_model=noise_model,
                        ve_multiplier=float(ve_multiplier),
                    )
                    if math.isclose(float(ve_multiplier), 1.0, rel_tol=0.0, abs_tol=1e-12):
                        legacy_pair_curve = pd.DataFrame(
                            evaluate_pairwise_objective(
                                table,
                                alpha_values,
                                "exclude_nonpositive",
                                "deaths_min",
                                "theta_t_gamma",
                                8,
                                NEUTRALIZATION_REFERENCE,
                            )
                        )
                        legacy_collapse_curve = pd.DataFrame(
                            evaluate_collapse_objective(
                                table,
                                alpha_values,
                                "exclude_nonpositive",
                                "deaths",
                                "theta_t_gamma",
                                3,
                                NEUTRALIZATION_REFERENCE,
                            )
                        )
                        legacy_pair_best = summarize_best_curve(legacy_pair_curve, cfg)
                        legacy_collapse_best = summarize_best_curve(legacy_collapse_curve, cfg)
                    primary_df = build_synthetic_primary_subset(table, NEUTRALIZATION_REFERENCE)
                    _, best_df = evaluate_synthetic_primary(primary_df, cfg, alpha_values, NEUTRALIZATION_REFERENCE)
                    if best_df.empty or set(best_df["estimator"]) != {"pairwise", "collapse"}:
                        continue
                    pair_row = best_df[best_df["estimator"] == "pairwise"].iloc[0]
                    collapse_row = best_df[best_df["estimator"] == "collapse"].iloc[0]
                    estimator_gap = abs(float(pair_row["alpha_hat"]) - float(collapse_row["alpha_hat"]))
                    for estimator in ("pairwise", "collapse"):
                        boot_df = bootstrap_alpha(
                            primary_df,
                            cfg,
                            alpha_values,
                            NEUTRALIZATION_REFERENCE,
                            estimator=estimator,
                            seed=combo_seed + (17 if estimator == "pairwise" else 29),
                        )
                        for column, value in {
                            "anchor_mode": "synthetic_direct",
                            "theta_scale": "gamma_primary",
                            "excess_mode": cfg["analysis"]["primary_excess_mode"],
                            "age_band": "pooled",
                            "time_segment": "pooled",
                        }.items():
                            boot_df[column] = value
                        boot_summary = build_bootstrap_summary(boot_df, cfg)
                        loo_df = leave_one_out_analysis(
                            primary_df,
                            cfg,
                            alpha_values,
                            NEUTRALIZATION_REFERENCE,
                            estimator=estimator,
                        )
                        for column, value in {
                            "anchor_mode": "synthetic_direct",
                            "theta_scale": "gamma_primary",
                            "excess_mode": cfg["analysis"]["primary_excess_mode"],
                            "age_band": "pooled",
                            "time_segment": "pooled",
                        }.items():
                            loo_df[column] = value
                        loo_summary = build_leave_one_out_summary(loo_df, best_df, cfg)
                        best_row = pair_row if estimator == "pairwise" else collapse_row
                        boot_row = boot_summary.iloc[0]
                        loo_row = loo_summary.iloc[0]
                        ve_recovery_rows.append(
                            {
                                "noise_model": noise_model,
                                "alpha_true": float(alpha_true),
                                "ve_multiplier": float(ve_multiplier),
                                "rep": rep,
                                "estimator": estimator,
                                "alpha_hat_raw": float(best_row["alpha_hat"]),
                                "alpha_hat_reported": (
                                    np.nan
                                    if not np.isfinite(best_row["alpha_hat_reported"])
                                    else float(best_row["alpha_hat_reported"])
                                ),
                                "bias": float(best_row["alpha_hat"] - float(alpha_true)),
                                "absolute_error": float(abs(best_row["alpha_hat"] - float(alpha_true))),
                                "identified": int(best_row["identified_curve"]),
                                "identification_status": str(best_row["identification_status"]),
                                "curvature_metric": float(best_row["curvature_metric"]),
                                "boundary_optimum": int(best_row["boundary_optimum"]),
                                "objective": float(best_row["objective"]),
                                "estimator_gap": float(estimator_gap),
                                "bootstrap_iqr": float(boot_row["iqr_alpha_hat"]),
                                "bootstrap_boundary_fraction": float(boot_row["boundary_fraction"]),
                                "bootstrap_finite_fraction": float(boot_row["finite_fraction"]),
                                "leave_one_out_max_shift": float(loo_row["max_abs_shift"]),
                                "leave_one_out_boundary_count": int(loo_row["boundary_count"]),
                            }
                        )
                    if math.isclose(float(ve_multiplier), 1.0, rel_tol=0.0, abs_tol=1e-12):
                        regression_summary.append(
                            {
                                "noise_model": noise_model,
                                "alpha_true": float(alpha_true),
                                "rep": rep,
                                "pairwise_alpha_hat": float(pair_row["alpha_hat"]),
                                "collapse_alpha_hat": float(collapse_row["alpha_hat"]),
                            }
                        )
                        regression_check_rows.append(
                            {
                                "noise_model": noise_model,
                                "alpha_true": float(alpha_true),
                                "rep": rep,
                                "pairwise_delta": (
                                    np.nan
                                    if legacy_pair_best is None
                                    else float(pair_row["alpha_hat"] - legacy_pair_best["alpha_hat"])
                                ),
                                "collapse_delta": (
                                    np.nan
                                    if legacy_collapse_best is None
                                    else float(collapse_row["alpha_hat"] - legacy_collapse_best["alpha_hat"])
                                ),
                            }
                        )
    ve_recovery_df = pd.DataFrame(ve_recovery_rows)
    legacy_df = pd.DataFrame(regression_summary)
    regression_check_df = pd.DataFrame(regression_check_rows)
    ve_summary_df = summarize_synthetic_vaccine_effect(ve_recovery_df)
    ve_report = render_synthetic_vaccine_effect_report(ve_recovery_df, ve_summary_df)
    validate_synthetic_ve_regression(regression_check_df)
    return {
        "legacy_df": legacy_df,
        "ve_recovery_df": ve_recovery_df,
        "ve_summary_df": ve_summary_df,
        "ve_report": ve_report,
    }


def conditional_ve_alpha_identification(cfg: dict, alpha_values: np.ndarray) -> dict[str, object]:
    synth_cfg = cfg["synthetic"]
    if not bool(synth_cfg.get("enabled", True)) or not conditional_ve_enabled(cfg):
        return {
            "estimates_df": pd.DataFrame(),
            "summary_df": pd.DataFrame(),
            "report": "",
            "identity_df": pd.DataFrame(),
        }
    base_seed = int(synth_cfg["seed"])
    reps = int(synth_cfg.get("reps", 8))
    noise_models = list(synth_cfg.get("noise_models", ["lognormal_fixed"]))
    target_multiplier = get_conditional_ve_target_multiplier(cfg)
    assumed_values = get_conditional_ve_values(cfg)
    estimate_rows: list[dict] = []
    identity_rows: list[dict] = []
    neutralization_mode = NEUTRALIZATION_REFERENCE
    for noise_idx, noise_model in enumerate(noise_models):
        for alpha_idx, alpha_true in enumerate(synth_cfg["alpha_true_values"]):
            print(
                f"[ALPHA] conditional VE fit for alpha_true={float(alpha_true):.3f} "
                f"(noise_model={noise_model}, dgp_ve={target_multiplier:.2f})",
                flush=True,
            )
            for rep in range(reps):
                combo_seed = base_seed + rep + 1000 * alpha_idx + 10000 * noise_idx + 500000
                table = simulate_synthetic_table(
                    cfg,
                    float(alpha_true),
                    combo_seed,
                    noise_model=noise_model,
                    ve_multiplier=target_multiplier,
                )
                primary_df = build_synthetic_primary_subset(table, neutralization_mode)
                _, base_metrics_df, _ = evaluate_synthetic_best_with_diagnostics(
                    primary_df,
                    cfg,
                    alpha_values,
                    neutralization_mode,
                    combo_seed + 7000,
                )
                if base_metrics_df.empty:
                    continue
                for assumed_idx, ve_assumed in enumerate(assumed_values):
                    adjusted_df = apply_conditional_ve_adjustment(primary_df, float(ve_assumed))
                    validate_conditional_ve_adjustment(primary_df, adjusted_df, float(ve_assumed))
                    seed_offset = combo_seed + 7000 if math.isclose(float(ve_assumed), 1.0, rel_tol=0.0, abs_tol=1e-12) else combo_seed + 7000 + 1000 * (assumed_idx + 1)
                    _, metrics_df, _ = evaluate_synthetic_best_with_diagnostics(
                        adjusted_df,
                        cfg,
                        alpha_values,
                        neutralization_mode,
                        seed_offset,
                    )
                    if metrics_df.empty:
                        continue
                    for _, row in metrics_df.iterrows():
                        estimate_rows.append(
                            {
                                "noise_model": noise_model,
                                "alpha_true": float(alpha_true),
                                "dgp_ve_multiplier": float(target_multiplier),
                                "VE_assumed": float(ve_assumed),
                                "rep": rep,
                                "estimator": str(row["estimator"]),
                                "alpha_hat_raw": float(row["alpha_hat_raw"]),
                                "alpha_hat_reported": np.nan if not np.isfinite(row["alpha_hat_reported"]) else float(row["alpha_hat_reported"]),
                                "bias": float(row["alpha_hat_raw"] - float(alpha_true)),
                                "absolute_error": float(abs(row["alpha_hat_raw"] - float(alpha_true))),
                                "identified": int(row["identified"]),
                                "identification_status": str(row["identification_status"]),
                                "curvature_metric": float(row["curvature_metric"]),
                                "bootstrap_iqr": float(row["bootstrap_iqr"]),
                                "bootstrap_boundary_fraction": float(row["bootstrap_boundary_fraction"]),
                                "leave_one_out_max_shift": float(row["leave_one_out_max_shift"]),
                                "estimator_gap": float(row["estimator_gap"]),
                                "boundary_optimum": int(row["boundary_optimum"]),
                            }
                        )
                    if math.isclose(float(ve_assumed), 1.0, rel_tol=0.0, abs_tol=1e-12):
                        base_by_est = {str(row["estimator"]): row for _, row in base_metrics_df.iterrows()}
                        cond_by_est = {str(row["estimator"]): row for _, row in metrics_df.iterrows()}
                        for estimator in ("pairwise", "collapse"):
                            base_row = base_by_est[estimator]
                            cond_row = cond_by_est[estimator]
                            identity_rows.append(
                                {
                                    "noise_model": noise_model,
                                    "alpha_true": float(alpha_true),
                                    "rep": rep,
                                    "estimator": estimator,
                                    "alpha_hat_delta": float(cond_row["alpha_hat_raw"] - base_row["alpha_hat_raw"]),
                                    "curvature_delta": float(cond_row["curvature_metric"] - base_row["curvature_metric"]),
                                    "bootstrap_boundary_delta": float(
                                        cond_row["bootstrap_boundary_fraction"] - base_row["bootstrap_boundary_fraction"]
                                    ),
                                    "leave_one_out_shift_delta": float(
                                        cond_row["leave_one_out_max_shift"] - base_row["leave_one_out_max_shift"]
                                    ),
                                }
                            )
    estimates_df = pd.DataFrame(estimate_rows)
    identity_df = pd.DataFrame(identity_rows)
    summary_df = summarize_conditional_ve_estimates(estimates_df, target_multiplier)
    report = render_conditional_ve_report(estimates_df, summary_df, target_multiplier)
    validate_conditional_ve_identity(identity_df)
    return {
        "estimates_df": estimates_df,
        "summary_df": summary_df,
        "report": report,
        "identity_df": identity_df,
    }


THETA_EFFECTIVE_WARN_THRESHOLD = 3.5


def run_wave_identifiability_experiment(cfg: dict, alpha_values: np.ndarray) -> dict[str, object]:
    if not wave_identifiability_experiment_enabled(cfg):
        return {"df": pd.DataFrame(), "report": ""}
    synth_cfg = cfg["synthetic"]
    if not bool(synth_cfg.get("enabled", True)):
        return {"df": pd.DataFrame(), "report": ""}
    assert_wave_identifiability_config_clean(cfg)
    assert_synthetic_wave_theta_defaults_match_table(cfg)

    base_seed = int(synth_cfg["seed"])
    reps = int(synth_cfg.get("reps", 8))
    noise_models = list(synth_cfg.get("noise_models", ["lognormal_fixed"]))
    wave_mults = get_wave_forcing_multipliers(cfg)
    theta_mults = get_theta_strength_multipliers(cfg)
    theta_base_max = float(np.max(np.asarray(synth_cfg["theta_w_values"], dtype=float)))
    for tm in theta_mults:
        eff = theta_base_max * float(tm)
        if eff > THETA_EFFECTIVE_WARN_THRESHOLD:
            print(
                f"[WAVE-ID] warning: large effective baseline theta "
                f"(max theta_w * theta_mult = {eff:.3f}, theta_mult={float(tm):g})",
                flush=True,
            )

    neutralization_mode = NEUTRALIZATION_REFERENCE
    rows: list[dict] = []
    signal_logged: set[tuple[float, float]] = set()

    for noise_idx, noise_model in enumerate(noise_models):
        for alpha_idx, alpha_true in enumerate(synth_cfg["alpha_true_values"]):
            for wave_idx, wave_mult in enumerate(wave_mults):
                for theta_idx, theta_mult in enumerate(theta_mults):
                    for rep in range(reps):
                        combo_seed = (
                            base_seed
                            + rep
                            + 1000 * alpha_idx
                            + 10000 * noise_idx
                            + 100000 * wave_idx
                            + 1000000 * theta_idx
                        )
                        table = simulate_synthetic_table(
                            cfg,
                            float(alpha_true),
                            combo_seed,
                            noise_model=noise_model,
                            ve_multiplier=1.0,
                            wave_multiplier=float(wave_mult),
                            theta_multiplier=float(theta_mult),
                        )
                        wkey = (float(wave_mult), float(theta_mult))
                        if rep == 0 and noise_idx == 0 and alpha_idx == 0 and wkey not in signal_logged:
                            _, delta_h = synthetic_wave_amplitude_and_delta_h(cfg, float(wave_mult))
                            max_dh = float(np.max(delta_h))
                            max_ex = float(table["excess"].abs().max())
                            print(
                                f"[WAVE-ID] wave_mult={float(wave_mult):.4f} theta_mult={float(theta_mult):.4f} "
                                f"max_delta_h_path={max_dh:.6f} max_abs_excess={max_ex:.6f}",
                                flush=True,
                            )
                            signal_logged.add(wkey)

                        primary_df = build_synthetic_primary_subset(table, neutralization_mode)
                        _, metrics_df, _ = evaluate_synthetic_best_with_diagnostics(
                            primary_df,
                            cfg,
                            alpha_values,
                            neutralization_mode,
                            combo_seed + 1_200_000,
                        )
                        if metrics_df.empty:
                            continue
                        for _, mrow in metrics_df.iterrows():
                            identified_curve = int(mrow["identified"])
                            alpha_rep = float(mrow["alpha_hat_reported"])
                            ident_strict = strict_synthetic_identified(
                                cfg,
                                identified_curve,
                                alpha_rep,
                                float(mrow["bootstrap_finite_fraction"]),
                                float(mrow["bootstrap_iqr"]),
                                float(mrow["bootstrap_boundary_fraction"]),
                            )
                            rows.append(
                                {
                                    "noise_model": noise_model,
                                    "alpha_true": float(alpha_true),
                                    "wave_multiplier": float(wave_mult),
                                    "theta_multiplier": float(theta_mult),
                                    "rep": rep,
                                    "estimator": str(mrow["estimator"]),
                                    "alpha_hat_raw": float(mrow["alpha_hat_raw"]),
                                    "alpha_hat_reported": alpha_rep,
                                    "identified": int(ident_strict),
                                    "identified_curve": identified_curve,
                                    "identification_status": str(mrow["identification_status"]),
                                    "curvature_metric": float(mrow["curvature_metric"]),
                                    "bootstrap_boundary_fraction": float(mrow["bootstrap_boundary_fraction"]),
                                    "bootstrap_finite_fraction": float(mrow["bootstrap_finite_fraction"]),
                                    "bootstrap_iqr": float(mrow["bootstrap_iqr"]),
                                    "boundary_optimum": int(mrow["boundary_optimum"]),
                                }
                            )

    out_df = pd.DataFrame(rows)
    report = render_wave_identifiability_report(out_df)
    return {"df": out_df, "report": report}


def render_wave_identifiability_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Wave vs theta alpha identifiability\n\nNo experiment rows were produced.\n"
    lines = [
        "# Wave vs theta alpha identifiability",
        "",
        "Synthetic experiment: epidemic forcing multiplier (wave) vs frailty-strength multiplier (theta), "
        "VE off, same estimator and identifiability gates as the main alpha sandbox.",
        "",
        "## Key tests",
        "",
        "- Does **increasing wave magnitude alone** improve pairwise identification rate (strict: reported alpha + bootstrap gates)?",
        "- Does **increasing theta** (cross-cohort frailty spread) improve identification?",
        "",
        "## Expected outcome",
        "",
        "- Increasing `wave_multiplier` alone should have **limited** effect on pairwise identification rate.",
        "- Increasing `theta_multiplier` should **increase** identification rate and curvature.",
        "- If that pattern holds, alpha identifiability is driven by **cross-cohort divergence** (frailty spread), "
        "not absolute wave magnitude.",
        "",
        "## Definition of `identified` in CSV",
        "",
        "- `identified` = 1 iff `alpha_hat_reported` is finite and bootstrap stability gates pass "
        "(finite fraction, IQR, boundary fraction), per estimator row.",
        "- `identified_curve` is the objective-curve gate only (before bootstrap).",
        "",
        "## Run summary",
        "",
        f"- Rows: {len(df)}",
        f"- Noise models: {', '.join(sorted(df['noise_model'].astype(str).unique()))}",
        f"- Wave multipliers: {sorted(df['wave_multiplier'].unique().tolist())}",
        f"- Theta multipliers: {sorted(df['theta_multiplier'].unique().tolist())}",
        "",
    ]
    pair = df[df["estimator"] == "pairwise"].copy()
    if not pair.empty:
        g = (
            pair.groupby(["wave_multiplier", "theta_multiplier"], as_index=False)
            .agg(identification_rate=("identified", "mean"), mean_curvature=("curvature_metric", "mean"))
            .sort_values(["wave_multiplier", "theta_multiplier"])
        )
        lines.append("### Pairwise identification rate by (wave, theta)")
        lines.append("")
        lines.append("```text")
        lines.append(g.to_string(index=False))
        lines.append("```")
        lines.append("")
    lines.append(
        "## Conclusion template\n\nIf wave increases without improved identification, but theta increases do improve "
        "identification, then alpha is driven by cross-cohort differential scaling, not absolute wave size.\n"
    )
    return "\n".join(lines) + "\n"


def plot_wave_identifiability(df: pd.DataFrame, out_path: Path) -> None:
    pair = df[df["estimator"] == "pairwise"].copy()
    if pair.empty:
        print(f"[ALPHA] Skipping {out_path.name}: no pairwise wave-identifiability rows", flush=True)
        return
    plt = _import_matplotlib()
    if plt is None:
        return

    def _close_mask(series: pd.Series, target: float, rtol: float = 0.0, atol: float = 1e-6) -> pd.Series:
        return np.isclose(series.to_numpy(dtype=float), float(target), rtol=rtol, atol=atol)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    base_theta = 1.0
    part_a = pair[_close_mask(pair["theta_multiplier"], base_theta)].copy()
    if not part_a.empty:
        agg_a = (
            part_a.groupby(["wave_multiplier", "alpha_true"], as_index=False)
            .agg(identification_rate=("identified", "mean"))
            .sort_values(["alpha_true", "wave_multiplier"])
        )
        for alpha_val, sub in agg_a.groupby("alpha_true"):
            sub = sub.sort_values("wave_multiplier")
            ax.plot(
                sub["wave_multiplier"],
                sub["identification_rate"],
                marker="o",
                lw=2,
                label=f"alpha_true={float(alpha_val):.2f}",
            )
    ax.set_title("A. Identification rate vs wave (theta_mult = 1)")
    ax.set_xlabel("wave_multiplier")
    ax.set_ylabel("Mean identified (pairwise, strict)")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    ax = axes[1]
    target_wave = 2.0
    part_b = pair[_close_mask(pair["wave_multiplier"], target_wave)].copy()
    if part_b.empty:
        wmax = float(pair["wave_multiplier"].max())
        part_b = pair[np.isclose(pair["wave_multiplier"].to_numpy(dtype=float), wmax)].copy()
        ax.set_title(f"B. Identification rate vs theta (wave_mult = {wmax:g}, fallback max)")
    else:
        ax.set_title("B. Identification rate vs theta (wave_mult = 2)")
    if not part_b.empty:
        agg_b = (
            part_b.groupby(["theta_multiplier", "alpha_true"], as_index=False)
            .agg(identification_rate=("identified", "mean"))
            .sort_values(["alpha_true", "theta_multiplier"])
        )
        for alpha_val, sub in agg_b.groupby("alpha_true"):
            sub = sub.sort_values("theta_multiplier")
            ax.plot(
                sub["theta_multiplier"],
                sub["identification_rate"],
                marker="s",
                lw=2,
                label=f"alpha_true={float(alpha_val):.2f}",
            )
    ax.set_xlabel("theta_multiplier")
    ax.set_ylabel("Mean identified (pairwise, strict)")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def print_wave_identifiability_summary(df: pd.DataFrame) -> None:
    print("WAVE IDENTIFIABILITY TEST", flush=True)
    if df.empty:
        print("no wave identifiability rows", flush=True)
        return
    pair = df[df["estimator"] == "pairwise"].copy()
    if pair.empty:
        print("no pairwise rows", flush=True)
        return

    def _rate(sub: pd.DataFrame) -> float:
        return float(sub["identified"].mean()) if len(sub) else float("nan")

    scenarios = [
        (1.25, 1.0),
        (2.0, 1.0),
        (2.0, 0.5),
        (2.0, 2.0),
    ]
    for wm, tm in scenarios:
        mask = np.isclose(pair["wave_multiplier"].to_numpy(dtype=float), wm) & np.isclose(
            pair["theta_multiplier"].to_numpy(dtype=float), tm
        )
        sub = pair.loc[mask]
        print(
            f"wave={wm:.2f} theta={tm:.1f} identified_rate={_rate(sub):.3f} n={len(sub)}",
            flush=True,
        )


def _import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.ioff()
        return plt
    except Exception as exc:
        print(f"[ALPHA] Skipping plot generation for {exc}", flush=True)
        return None


def _save_figure(fig, out_path: Path) -> bool:
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        return True
    except Exception as exc:
        print(f"[ALPHA] Failed to save plot {out_path.name}: {exc}", flush=True)
        return False


def _manuscript_figures_dir(repo_root: Path) -> Path:
    return repo_root / "documentation" / "preprint" / "figures"


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required alpha figure inputs: {missing}")


def _resample_plot_grid(alpha_vals: np.ndarray, objective_vals: np.ndarray, start: float = 1.00, stop: float = 1.30, step: float = 0.005) -> tuple[np.ndarray, np.ndarray]:
    alpha_arr = np.asarray(alpha_vals, dtype=float)
    obj_arr = np.asarray(objective_vals, dtype=float)
    if alpha_arr.size == 0:
        return alpha_arr, obj_arr
    order = np.argsort(alpha_arr)
    alpha_arr = alpha_arr[order]
    obj_arr = obj_arr[order]
    diffs = np.diff(alpha_arr)
    if np.any(diffs <= 0):
        raise ValueError("Alpha grid must be strictly monotonic for plotting")
    target = np.arange(start, stop + step * 0.5, step, dtype=float)
    if np.allclose(alpha_arr[0], start) and np.allclose(alpha_arr[-1], stop) and np.allclose(diffs, diffs[0]):
        return alpha_arr, obj_arr
    return target, np.interp(target, alpha_arr, obj_arr)


def _filter_best_unique(
    best_df: pd.DataFrame,
    *,
    neutralization_mode: str,
    anchor_mode: str,
    theta_scale: str,
    excess_mode: str,
    age_band: str,
    time_segment: str,
    estimator: str,
) -> pd.Series:
    subset = best_df[
        (best_df["neutralization_mode"] == neutralization_mode)
        & (best_df["anchor_mode"] == anchor_mode)
        & (best_df["theta_scale"] == theta_scale)
        & (best_df["excess_mode"] == excess_mode)
        & (best_df["age_band"] == age_band)
        & (best_df["time_segment"] == time_segment)
        & (best_df["estimator"] == estimator)
    ]
    if len(subset) != 1:
        raise ValueError(
            "Expected exactly one alpha_best_estimates row for "
            f"neutralization_mode={neutralization_mode}, anchor={anchor_mode}, "
            f"theta_scale={theta_scale}, excess_mode={excess_mode}, "
            f"age_band={age_band}, time_segment={time_segment}, estimator={estimator}; got {len(subset)}"
        )
    return subset.iloc[0]


def _warn_if_manuscript_mismatch(pair_alpha: float, collapse_alpha: float) -> None:
    if abs(pair_alpha - MANUSCRIPT_ALPHA_PAIR) > 0.01 or abs(collapse_alpha - MANUSCRIPT_ALPHA_COLLAPSE) > 0.01:
        print(
            "[ALPHA] Warning: pooled manuscript alpha values differ from figure source "
            f"(pair={pair_alpha:.3f}, collapse={collapse_alpha:.3f})",
            flush=True,
        )


def build_alpha_run_artifact(
    cfg: dict,
    dataset_cfg: dict,
    alpha_values: np.ndarray,
    cohort_diag: pd.DataFrame,
    best_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    loo_summary: pd.DataFrame,
    neutralization_mode: str,
) -> dict:
    ident_cfg = get_identifiability_cfg(cfg)
    params = cfg["analysis"]
    primary = best_df[
        (best_df["neutralization_mode"] == neutralization_mode)
        & (best_df["anchor_mode"] == params["primary_anchor"])
        & (best_df["theta_scale"] == "gamma_primary")
        & (best_df["excess_mode"] == params["primary_excess_mode"])
        & (best_df["age_band"] == "pooled")
        & (best_df["time_segment"] == "pooled")
        & (best_df["estimator"].isin(["pairwise", "collapse"]))
    ].copy()
    primary_by_est = {row["estimator"]: row for _, row in primary.iterrows()}
    pair_row = primary_by_est.get("pairwise")
    collapse_row = primary_by_est.get("collapse")
    estimator_gap = (
        abs(float(pair_row["alpha_hat"]) - float(collapse_row["alpha_hat"]))
        if pair_row is not None and collapse_row is not None
        else float("nan")
    )
    estimators_agree = bool(np.isfinite(estimator_gap) and estimator_gap <= float(ident_cfg["max_estimator_gap"]))
    boot_vals = bootstrap_df["alpha_hat"].dropna().to_numpy(dtype=float)
    finite_fraction = float(np.isfinite(bootstrap_df["alpha_hat"]).mean()) if len(bootstrap_df) else 0.0
    bootstrap_iqr = float(np.subtract(*np.nanpercentile(boot_vals, [75, 25]))) if boot_vals.size else float("nan")
    boot_summary_row = bootstrap_summary.iloc[0] if not bootstrap_summary.empty else None
    bootstrap_boundary_fraction = (
        float(boot_summary_row["boundary_fraction"]) if boot_summary_row is not None else float("nan")
    )
    bootstrap_ok = bool(
        finite_fraction >= float(ident_cfg["min_bootstrap_finite_fraction"])
        and (not np.isfinite(bootstrap_iqr) or bootstrap_iqr <= float(ident_cfg["max_bootstrap_iqr"]))
        and (
            not np.isfinite(bootstrap_boundary_fraction)
            or bootstrap_boundary_fraction <= float(ident_cfg["max_bootstrap_boundary_fraction"])
        )
    )
    pair_identified = bool(pair_row is not None and int(pair_row.get("identified_curve", 0)) == 1)
    collapse_identified = bool(collapse_row is not None and int(collapse_row.get("identified_curve", 0)) == 1)
    primary_identified = pair_identified and collapse_identified and estimators_agree and bootstrap_ok
    primary_status = "identified" if primary_identified else "not_identified"
    failure_reasons = []
    if not pair_identified:
        failure_reasons.append("pairwise_curve_failed")
    if not collapse_identified:
        failure_reasons.append("collapse_curve_failed")
    if pair_row is not None and int(pair_row.get("boundary_optimum", 0)) == 1:
        failure_reasons.append("pairwise_boundary")
    if collapse_row is not None and int(collapse_row.get("boundary_optimum", 0)) == 1:
        failure_reasons.append("collapse_boundary")
    if not estimators_agree:
        failure_reasons.append("estimator_disagreement")
    if not bootstrap_ok:
        failure_reasons.append("bootstrap_instability")
    if np.isfinite(bootstrap_boundary_fraction) and bootstrap_boundary_fraction > float(ident_cfg["max_bootstrap_boundary_fraction"]):
        failure_reasons.append("bootstrap_boundary_seeking")
    loo_summary_row = loo_summary.iloc[0] if not loo_summary.empty else None
    default_alpha = get_dataset_default_alpha(dataset_cfg)
    pair_reported = None if pair_row is None or not np.isfinite(pair_row["alpha_hat_reported"]) else float(pair_row["alpha_hat_reported"])
    collapse_reported = None if collapse_row is None or not np.isfinite(collapse_row["alpha_hat_reported"]) else float(collapse_row["alpha_hat_reported"])
    if primary_identified and pair_reported is not None and collapse_reported is not None:
        chosen_alpha = float(0.5 * (pair_reported + collapse_reported))
        chosen_source = "identified_dose0_vs_dose2"
        chosen_status = "identified"
    elif default_alpha is not None:
        chosen_alpha = float(default_alpha)
        chosen_source = "dataset_default_alpha"
        chosen_status = "external_calibration"
    else:
        chosen_alpha = None
        chosen_source = "none"
        chosen_status = "unavailable"
    return {
        "dataset": cfg["dataset"],
        "alpha_interpretation_contract": {
            "alpha_is_not_causal_or_biological": True,
            "alpha_is_model_calibrated": True,
            "requires_identifiability_diagnostics": True,
            "if_diagnostics_fail_report_not_identified": True,
        },
        "configuration": {
            "analysis": cfg["analysis"],
            "neutralization_mode": neutralization_mode,
            "neutralization_modes": get_nph_neutralization_modes(cfg),
            "primary_nph_neutralization_mode": get_primary_nph_neutralization_mode(cfg),
            "alpha_grid": cfg["alpha_grid"],
            "alpha_grid_points": [float(x) for x in alpha_values.tolist()],
            "bootstrap": cfg.get("bootstrap", {}),
            "synthetic": cfg.get("synthetic", {}),
            "identifiability": ident_cfg,
            "excess_definition": "h_excess,d(t) = h_d(t) - h_ref(t)",
        },
        "cohort_selection": {
            "enrollment_dates": cfg["analysis"]["enrollment_dates"],
            "yob_decades": cfg["analysis"]["yob_decades"],
            "doses": cfg["analysis"]["doses"],
            "eligible_cohorts": int(cohort_diag["eligible"].sum()) if not cohort_diag.empty else 0,
            "total_cohorts": int(len(cohort_diag)),
        },
        "nph_window": {
            "startDate": (dataset_cfg.get("NPH_correction") or {}).get("startDate"),
            "endDate": (dataset_cfg.get("NPH_correction") or {}).get("endDate"),
            "default_alpha": default_alpha,
        },
        "seeds": {
            "bootstrap_seed": cfg.get("bootstrap", {}).get("seed"),
            "synthetic_seed": cfg.get("synthetic", {}).get("seed"),
        },
        "primary_identification": {
            "status": primary_status,
            "failure_reasons": failure_reasons,
            "pairwise_alpha_hat_raw": None if pair_row is None else float(pair_row["alpha_hat"]),
            "pairwise_alpha_hat_reported": None if pair_row is None or not np.isfinite(pair_row["alpha_hat_reported"]) else float(pair_row["alpha_hat_reported"]),
            "collapse_alpha_hat_raw": None if collapse_row is None else float(collapse_row["alpha_hat"]),
            "collapse_alpha_hat_reported": None if collapse_row is None or not np.isfinite(collapse_row["alpha_hat_reported"]) else float(collapse_row["alpha_hat_reported"]),
            "pairwise_curve_status": None if pair_row is None else str(pair_row["identification_status"]),
            "collapse_curve_status": None if collapse_row is None else str(collapse_row["identification_status"]),
            "estimator_gap": None if not np.isfinite(estimator_gap) else estimator_gap,
            "estimators_agree": estimators_agree,
            "bootstrap_finite_fraction": finite_fraction,
            "bootstrap_iqr": None if not np.isfinite(bootstrap_iqr) else bootstrap_iqr,
            "bootstrap_boundary_fraction": None if not np.isfinite(bootstrap_boundary_fraction) else bootstrap_boundary_fraction,
            "bootstrap_ok": bootstrap_ok,
            "leave_one_out_max_abs_shift": None
            if loo_summary_row is None or not np.isfinite(loo_summary_row["max_abs_shift"])
            else float(loo_summary_row["max_abs_shift"]),
            "leave_one_out_large_shift_count": None if loo_summary_row is None else int(loo_summary_row["large_shift_count"]),
        },
        "production_integration": {
            "recommended": primary_identified,
            "decision": "defer" if not primary_identified else "evaluate_after_phase4",
        },
        "calibration_choice": {
            "status": chosen_status,
            "alpha_value": chosen_alpha,
            "source": chosen_source,
            "uses_external_default_alpha": bool((not primary_identified) and default_alpha is not None),
            "identified_alpha_required_diagnostics_passed": bool(primary_identified),
        },
    }


def build_calibration_choice_summary(run_artifact: dict) -> pd.DataFrame:
    calibration = run_artifact["calibration_choice"]
    nph_window = run_artifact.get("nph_window", {})
    primary = run_artifact["primary_identification"]
    return pd.DataFrame(
        [
            {
                "neutralization_mode": run_artifact["configuration"]["neutralization_mode"],
                "calibration_status": calibration["status"],
                "alpha_value": calibration["alpha_value"],
                "alpha_source": calibration["source"],
                "default_alpha": nph_window.get("default_alpha"),
                "nph_startDate": nph_window.get("startDate"),
                "nph_endDate": nph_window.get("endDate"),
                "primary_identification_status": primary["status"],
                "primary_failure_reasons": "; ".join(primary["failure_reasons"]) if primary["failure_reasons"] else "",
            }
        ]
    )


def build_bootstrap_summary(bootstrap_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ident_cfg = get_identifiability_cfg(cfg)
    alpha_min = float(cfg["alpha_grid"]["start"])
    alpha_max = float(cfg["alpha_grid"]["stop"])
    estimator = bootstrap_df["estimator"].iloc[0] if "estimator" in bootstrap_df.columns and not bootstrap_df.empty else None
    if bootstrap_df.empty:
        return pd.DataFrame(
            [
                {
                    "neutralization_mode": bootstrap_df["neutralization_mode"].iloc[0] if "neutralization_mode" in bootstrap_df.columns and not bootstrap_df.empty else None,
                    "estimator": estimator,
                    "n_reps": 0,
                    "finite_reps": 0,
                    "finite_fraction": 0.0,
                    "median_alpha_hat": np.nan,
                    "q25_alpha_hat": np.nan,
                    "q75_alpha_hat": np.nan,
                    "iqr_alpha_hat": np.nan,
                    "boundary_low_count": 0,
                    "boundary_high_count": 0,
                    "boundary_fraction": np.nan,
                    "bootstrap_ok": 0,
                }
            ]
        )
    vals = bootstrap_df["alpha_hat"].to_numpy(dtype=float)
    finite_mask = np.isfinite(vals)
    finite_vals = vals[finite_mask]
    boundary_low = int(np.count_nonzero(np.isclose(finite_vals, alpha_min)))
    boundary_high = int(np.count_nonzero(np.isclose(finite_vals, alpha_max)))
    boundary_fraction = (
        float((boundary_low + boundary_high) / finite_vals.size) if finite_vals.size else float("nan")
    )
    iqr = float(np.subtract(*np.nanpercentile(finite_vals, [75, 25]))) if finite_vals.size else float("nan")
    finite_fraction = float(np.mean(finite_mask)) if vals.size else 0.0
    bootstrap_ok = bool(
        finite_fraction >= float(ident_cfg["min_bootstrap_finite_fraction"])
        and (not np.isfinite(iqr) or iqr <= float(ident_cfg["max_bootstrap_iqr"]))
        and (
            not np.isfinite(boundary_fraction)
            or boundary_fraction <= float(ident_cfg["max_bootstrap_boundary_fraction"])
        )
    )
    return pd.DataFrame(
        [
            {
                "neutralization_mode": bootstrap_df["neutralization_mode"].iloc[0] if "neutralization_mode" in bootstrap_df.columns else None,
                "estimator": estimator,
                "n_reps": int(len(bootstrap_df)),
                "finite_reps": int(np.count_nonzero(finite_mask)),
                "finite_fraction": finite_fraction,
                "median_alpha_hat": float(np.nanmedian(finite_vals)) if finite_vals.size else np.nan,
                "q25_alpha_hat": float(np.nanpercentile(finite_vals, 25)) if finite_vals.size else np.nan,
                "q75_alpha_hat": float(np.nanpercentile(finite_vals, 75)) if finite_vals.size else np.nan,
                "iqr_alpha_hat": iqr,
                "boundary_low_count": boundary_low,
                "boundary_high_count": boundary_high,
                "boundary_fraction": boundary_fraction,
                "bootstrap_ok": int(bootstrap_ok),
            }
        ]
    )


def build_leave_one_out_summary(loo_df: pd.DataFrame, best_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    ident_cfg = get_identifiability_cfg(cfg)
    neutralization_mode = (
        str(loo_df["neutralization_mode"].iloc[0])
        if "neutralization_mode" in loo_df.columns and not loo_df.empty
        else get_primary_nph_neutralization_mode(cfg)
    )
    estimator = (
        str(loo_df["estimator"].iloc[0])
        if "estimator" in loo_df.columns and not loo_df.empty
        else "pairwise"
    )
    anchor_mode = str(loo_df["anchor_mode"].iloc[0]) if "anchor_mode" in loo_df.columns and not loo_df.empty else cfg["analysis"]["primary_anchor"]
    theta_scale = str(loo_df["theta_scale"].iloc[0]) if "theta_scale" in loo_df.columns and not loo_df.empty else "gamma_primary"
    excess_mode = str(loo_df["excess_mode"].iloc[0]) if "excess_mode" in loo_df.columns and not loo_df.empty else cfg["analysis"]["primary_excess_mode"]
    age_band = str(loo_df["age_band"].iloc[0]) if "age_band" in loo_df.columns and not loo_df.empty else "pooled"
    time_segment = str(loo_df["time_segment"].iloc[0]) if "time_segment" in loo_df.columns and not loo_df.empty else "pooled"
    if loo_df.empty:
        return pd.DataFrame(
            [
                {
                    "neutralization_mode": neutralization_mode,
                    "estimator": estimator,
                    "pooled_alpha_hat": np.nan,
                    "n_omissions": 0,
                    "finite_omissions": 0,
                    "max_abs_shift": np.nan,
                    "median_abs_shift": np.nan,
                    "large_shift_count": 0,
                    "boundary_count": 0,
                    "stable_leave_one_out": 0,
                }
            ]
        )
    pair_row = _filter_best_unique(
        best_df,
        neutralization_mode=neutralization_mode,
        anchor_mode=anchor_mode,
        theta_scale=theta_scale,
        excess_mode=excess_mode,
        age_band=age_band,
        time_segment=time_segment,
        estimator=estimator,
    )
    pooled_alpha = float(pair_row["alpha_hat"])
    finite = loo_df[np.isfinite(loo_df["alpha_hat"])].copy()
    shifts = np.abs(finite["alpha_hat"].to_numpy(dtype=float) - pooled_alpha) if not finite.empty else np.asarray([], dtype=float)
    threshold = float(ident_cfg["max_leave_one_out_shift"])
    large_shift_count = int(np.count_nonzero(shifts > threshold)) if shifts.size else 0
    boundary_count = int(finite["boundary_optimum"].fillna(0).astype(int).sum()) if not finite.empty else 0
    stable = bool(large_shift_count == 0 and boundary_count == 0 and len(finite) == len(loo_df))
    return pd.DataFrame(
        [
            {
                "neutralization_mode": neutralization_mode,
                "estimator": estimator,
                "pooled_alpha_hat": pooled_alpha,
                "n_omissions": int(len(loo_df)),
                "finite_omissions": int(len(finite)),
                "max_abs_shift": float(np.max(shifts)) if shifts.size else np.nan,
                "median_abs_shift": float(np.median(shifts)) if shifts.size else np.nan,
                "large_shift_count": large_shift_count,
                "boundary_count": boundary_count,
                "stable_leave_one_out": int(stable),
            }
        ]
    )


def build_primary_sensitivity_slices(best_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    params = cfg["analysis"]
    primary_mode = get_primary_nph_neutralization_mode(cfg)
    rows: list[dict] = []

    def add_dimension(
        dimension: str,
        subset: pd.DataFrame,
        setting_col: str,
        primary_setting: str,
    ) -> None:
        if subset.empty:
            return
        for estimator in sorted(subset["estimator"].unique()):
            est_subset = subset[subset["estimator"] == estimator].copy()
            baseline = est_subset[est_subset[setting_col] == primary_setting]
            baseline_alpha = float(baseline.iloc[0]["alpha_hat"]) if not baseline.empty else float("nan")
            for _, row in est_subset.sort_values(setting_col).iterrows():
                rows.append(
                    {
                        "dimension": dimension,
                        "setting": str(row[setting_col]),
                        "setting_is_primary": int(str(row[setting_col]) == str(primary_setting)),
                        "estimator": estimator,
                        "alpha_hat": float(row["alpha_hat"]),
                        "alpha_hat_reported": np.nan
                        if not np.isfinite(row["alpha_hat_reported"])
                        else float(row["alpha_hat_reported"]),
                        "delta_from_primary": np.nan
                        if not np.isfinite(baseline_alpha)
                        else float(row["alpha_hat"] - baseline_alpha),
                        "curvature_metric": float(row["curvature_metric"]),
                        "boundary_optimum": int(row["boundary_optimum"]),
                        "identified_curve": int(row["identified_curve"]),
                        "identification_status": str(row["identification_status"]),
                        "n_pairs": int(row["n_pairs"]),
                        "n_weeks_used": int(row["n_weeks_used"]),
                        "n_cohorts": int(row["n_cohorts"]),
                    }
                )

    anchor_subset = best_df[
        (best_df["neutralization_mode"] == primary_mode)
        & (best_df["theta_scale"] == "gamma_primary")
        & (best_df["excess_mode"] == params["primary_excess_mode"])
        & (best_df["age_band"] == "pooled")
        & (best_df["time_segment"] == "pooled")
    ].copy()
    add_dimension("anchor_mode", anchor_subset, "anchor_mode", params["primary_anchor"])

    theta_subset = best_df[
        (best_df["neutralization_mode"] == primary_mode)
        & (best_df["anchor_mode"] == params["primary_anchor"])
        & (best_df["excess_mode"] == params["primary_excess_mode"])
        & (best_df["age_band"] == "pooled")
        & (best_df["time_segment"] == "pooled")
    ].copy()
    add_dimension("theta_scale", theta_subset, "theta_scale", "gamma_primary")

    excess_subset = best_df[
        (best_df["neutralization_mode"] == primary_mode)
        & (best_df["anchor_mode"] == params["primary_anchor"])
        & (best_df["theta_scale"] == "gamma_primary")
        & (best_df["age_band"] == "pooled")
        & (best_df["time_segment"] == "pooled")
    ].copy()
    add_dimension("excess_mode", excess_subset, "excess_mode", params["primary_excess_mode"])

    segment_subset = best_df[
        (best_df["neutralization_mode"] == primary_mode)
        & (best_df["anchor_mode"] == params["primary_anchor"])
        & (best_df["theta_scale"] == "gamma_primary")
        & (best_df["excess_mode"] == params["primary_excess_mode"])
        & (best_df["age_band"] == "pooled")
    ].copy()
    add_dimension("time_segment", segment_subset, "time_segment", "pooled")

    return pd.DataFrame(rows)


def build_decision_summary(
    best_df: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    loo_summary: pd.DataFrame,
    run_artifact: dict,
    cfg: dict,
) -> pd.DataFrame:
    ident_cfg = get_identifiability_cfg(cfg)
    neutralization_mode = str(run_artifact["configuration"]["neutralization_mode"])
    pair_row = _filter_best_unique(
        best_df,
        neutralization_mode=neutralization_mode,
        anchor_mode=cfg["analysis"]["primary_anchor"],
        theta_scale="gamma_primary",
        excess_mode=cfg["analysis"]["primary_excess_mode"],
        age_band="pooled",
        time_segment="pooled",
        estimator="pairwise",
    )
    collapse_row = _filter_best_unique(
        best_df,
        neutralization_mode=neutralization_mode,
        anchor_mode=cfg["analysis"]["primary_anchor"],
        theta_scale="gamma_primary",
        excess_mode=cfg["analysis"]["primary_excess_mode"],
        age_band="pooled",
        time_segment="pooled",
        estimator="collapse",
    )
    boot = bootstrap_summary.iloc[0]
    loo = loo_summary.iloc[0]
    interior_ok = int(pair_row["boundary_optimum"]) == 0 and int(collapse_row["boundary_optimum"]) == 0
    curvature_ok = (
        np.isfinite(pair_row["curvature_metric"])
        and np.isfinite(collapse_row["curvature_metric"])
        and float(pair_row["curvature_metric"]) >= float(ident_cfg["min_normalized_curvature"])
        and float(collapse_row["curvature_metric"]) >= float(ident_cfg["min_normalized_curvature"])
    )
    stability_ok = bool(
        run_artifact["primary_identification"]["estimators_agree"]
        and int(loo["stable_leave_one_out"]) == 1
    )
    bootstrap_boundary_ok = bool(
        int(boot["bootstrap_ok"]) == 1
        and (
            not np.isfinite(boot["boundary_fraction"])
            or float(boot["boundary_fraction"]) <= float(ident_cfg["max_bootstrap_boundary_fraction"])
        )
    )
    verdict = run_artifact["primary_identification"]["status"]
    return pd.DataFrame(
        [
            {
                "neutralization_mode": neutralization_mode,
                "question": "Is the optimum interior?",
                "answer": "yes" if interior_ok else "no",
                "status": "pass" if interior_ok else "fail",
                "observed": f"pairwise_boundary={int(pair_row['boundary_optimum'])}; collapse_boundary={int(collapse_row['boundary_optimum'])}",
                "threshold": "Both pooled optima must be interior",
            },
            {
                "neutralization_mode": neutralization_mode,
                "question": "Is curvature strong enough?",
                "answer": "yes" if curvature_ok else "no",
                "status": "pass" if curvature_ok else "fail",
                "observed": f"pairwise={float(pair_row['curvature_metric']):.6f}; collapse={float(collapse_row['curvature_metric']):.6f}",
                "threshold": f"Both >= {float(ident_cfg['min_normalized_curvature']):.3f}",
            },
            {
                "neutralization_mode": neutralization_mode,
                "question": "Are estimates stable?",
                "answer": "yes" if stability_ok else "no",
                "status": "pass" if stability_ok else "fail",
                "observed": (
                    f"pair_gap={float(run_artifact['primary_identification']['estimator_gap'] or np.nan):.6f}; "
                    f"loo_max_shift={float(loo['max_abs_shift']):.6f}"
                ),
                "threshold": (
                    f"pair_gap <= {float(ident_cfg['max_estimator_gap']):.3f}; "
                    f"loo_max_shift <= {float(ident_cfg['max_leave_one_out_shift']):.3f}"
                ),
            },
            {
                "neutralization_mode": neutralization_mode,
                "question": "Is bootstrap concentrated away from boundaries?",
                "answer": "yes" if bootstrap_boundary_ok else "no",
                "status": "pass" if bootstrap_boundary_ok else "fail",
                "observed": (
                    f"finite_fraction={float(boot['finite_fraction']):.3f}; "
                    f"iqr={float(boot['iqr_alpha_hat']):.3f}; boundary_fraction={float(boot['boundary_fraction']):.3f}"
                ),
                "threshold": (
                    f"finite_fraction >= {float(ident_cfg['min_bootstrap_finite_fraction']):.2f}; "
                    f"iqr <= {float(ident_cfg['max_bootstrap_iqr']):.2f}; "
                    f"boundary_fraction <= {float(ident_cfg['max_bootstrap_boundary_fraction']):.2f}"
                ),
            },
            {
                "neutralization_mode": neutralization_mode,
                "question": "Final verdict",
                "answer": verdict,
                "status": "pass" if verdict == "identified" else "fail",
                "observed": "; ".join(run_artifact["primary_identification"]["failure_reasons"]) or "identified",
                "threshold": "Report alpha only if all diagnostics pass",
            },
        ]
    )


def render_identifiability_report(
    run_artifact: dict,
    bootstrap_summary: pd.DataFrame,
    loo_summary: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    decision_summary: pd.DataFrame,
) -> str:
    primary = run_artifact["primary_identification"]
    calibration = run_artifact["calibration_choice"]
    neutralization_mode = run_artifact["configuration"]["neutralization_mode"]
    boot = bootstrap_summary.iloc[0]
    loo = loo_summary.iloc[0]

    def fmt(value: object, digits: int = 3) -> str:
        if value is None:
            return "NA"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(numeric):
            return "NA"
        return f"{numeric:.{digits}f}"

    lines = [
        "# Alpha Identifiability Report",
        "",
        f"- Neutralization mode: `{neutralization_mode}`",
        "",
        "## Primary Czech Run",
        f"- Pairwise raw estimate: `{fmt(primary['pairwise_alpha_hat_raw'])}`",
        f"- Collapse raw estimate: `{fmt(primary['collapse_alpha_hat_raw'])}`",
        f"- Pairwise curvature metric: `{fmt(sensitivity_df[(sensitivity_df['dimension'] == 'theta_scale') & (sensitivity_df['setting'] == 'gamma_primary') & (sensitivity_df['estimator'] == 'pairwise')]['curvature_metric'].iloc[0], 6)}`",
        f"- Collapse curvature metric: `{fmt(sensitivity_df[(sensitivity_df['dimension'] == 'theta_scale') & (sensitivity_df['setting'] == 'gamma_primary') & (sensitivity_df['estimator'] == 'collapse')]['curvature_metric'].iloc[0], 6)}`",
        f"- Boundary flag: `pairwise={int(sensitivity_df[(sensitivity_df['dimension'] == 'theta_scale') & (sensitivity_df['setting'] == 'gamma_primary') & (sensitivity_df['estimator'] == 'pairwise')]['boundary_optimum'].iloc[0])}`, `collapse={int(sensitivity_df[(sensitivity_df['dimension'] == 'theta_scale') & (sensitivity_df['setting'] == 'gamma_primary') & (sensitivity_df['estimator'] == 'collapse')]['boundary_optimum'].iloc[0])}`",
        (
            "- Bootstrap instability summary: "
            f"`finite_fraction={fmt(boot['finite_fraction'])}`, "
            f"`iqr={fmt(boot['iqr_alpha_hat'])}`, "
            f"`boundary_fraction={fmt(boot['boundary_fraction'])}`"
        ),
        f"- Final reported status: `{primary['status']}`",
        "",
        "## Why The Gate Failed",
        f"- Failure reasons: `{', '.join(primary['failure_reasons']) or 'none'}`",
        f"- Leave-one-out max shift from pooled pairwise alpha: `{fmt(loo['max_abs_shift'])}`",
        f"- Leave-one-out large-shift omissions: `{int(loo['large_shift_count'])}`",
        "",
        "## Calibration Choice",
        f"- Calibration status: `{calibration['status']}`",
        f"- Alpha used for correction branch: `{fmt(calibration['alpha_value'])}`",
        f"- Alpha source: `{calibration['source']}`",
        "",
        "## Sensitivity Slices",
    ]
    for dimension in ["anchor_mode", "theta_scale", "excess_mode", "time_segment"]:
        part = sensitivity_df[sensitivity_df["dimension"] == dimension].copy()
        if part.empty:
            continue
        lines.append(f"- `{dimension}`: see `alpha_primary_sensitivity_slices.csv` for {len(part)} rows covering this axis.")
    lines.extend(
        [
            "",
            "## Decision Table",
            "",
            "| Question | Answer | Observed | Threshold |",
            "| --- | --- | --- | --- |",
        ]
    )
    for _, row in decision_summary.iterrows():
        lines.append(
            f"| {row['question']} | {row['answer']} | {row['observed']} | {row['threshold']} |"
        )
    lines.extend(
        [
            "",
            "## Production Integration",
            "",
            "- Production integration remains deferred until a later gate is justified.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize_synthetic_vaccine_effect(recovery_df: pd.DataFrame) -> pd.DataFrame:
    if recovery_df.empty:
        return pd.DataFrame()

    def safe_nanmean(series: pd.Series) -> float:
        values = series.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else float("nan")

    rows: list[dict] = []
    for ve_multiplier in sorted(recovery_df["ve_multiplier"].dropna().unique(), reverse=True):
        ve_part = recovery_df[recovery_df["ve_multiplier"] == ve_multiplier].copy()
        for estimator, est_part in [("overall", ve_part), *[(name, ve_part[ve_part["estimator"] == name].copy()) for name in ("pairwise", "collapse")]]:
            if est_part.empty:
                continue
            reported_mask = np.isfinite(est_part["alpha_hat_reported"].to_numpy(dtype=float))
            rows.append(
                {
                    "ve_multiplier": float(ve_multiplier),
                    "estimator": estimator,
                    "n_rows": int(len(est_part)),
                    "mean_bias": safe_nanmean(est_part["bias"]),
                    "mean_absolute_error": safe_nanmean(est_part["absolute_error"]),
                    "identification_rate": float(np.mean(reported_mask)) if reported_mask.size else np.nan,
                    "non_identification_rate": float(1.0 - np.mean(reported_mask)) if reported_mask.size else np.nan,
                    "mean_curvature": safe_nanmean(est_part["curvature_metric"]),
                    "mean_bootstrap_boundary_fraction": safe_nanmean(est_part["bootstrap_boundary_fraction"]),
                    "mean_boundary_optimum_rate": safe_nanmean(est_part["boundary_optimum"]),
                    "mean_estimator_gap": safe_nanmean(est_part["estimator_gap"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["ve_multiplier", "estimator"], ascending=[False, True]).reset_index(drop=True)


def classify_synthetic_ve_severity(summary_df: pd.DataFrame, ve_multiplier: float) -> str:
    if summary_df.empty:
        return "unknown"
    baseline = summary_df[
        np.isclose(summary_df["ve_multiplier"].to_numpy(dtype=float), 1.0)
        & (summary_df["estimator"] == "overall")
    ]
    current = summary_df[
        np.isclose(summary_df["ve_multiplier"].to_numpy(dtype=float), float(ve_multiplier))
        & (summary_df["estimator"] == "overall")
    ]
    if baseline.empty or current.empty:
        return "unknown"
    base = baseline.iloc[0]
    cur = current.iloc[0]
    ident_drop = float(base["identification_rate"] - cur["identification_rate"])
    curvature_ratio = (
        float(cur["mean_curvature"] / base["mean_curvature"])
        if np.isfinite(base["mean_curvature"]) and abs(float(base["mean_curvature"])) > EPS and np.isfinite(cur["mean_curvature"])
        else np.nan
    )
    boundary_increase = float(cur["mean_bootstrap_boundary_fraction"] - base["mean_bootstrap_boundary_fraction"])
    if ident_drop >= 0.25 or (np.isfinite(curvature_ratio) and curvature_ratio <= 0.50) or boundary_increase >= 0.20:
        return "severe"
    if ident_drop >= 0.10 or (np.isfinite(curvature_ratio) and curvature_ratio <= 0.75) or boundary_increase >= 0.10:
        return "moderate"
    return "mild"


def render_synthetic_vaccine_effect_report(recovery_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    if recovery_df.empty or summary_df.empty:
        return "# Synthetic vaccine-effect stress test\n\nNo synthetic VE recovery rows were produced.\n"

    def fmt(value: object, digits: int = 3) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(numeric):
            return "NA"
        return f"{numeric:.{digits}f}"

    overall = summary_df[summary_df["estimator"] == "overall"].copy().sort_values("ve_multiplier", ascending=False)
    baseline = overall[np.isclose(overall["ve_multiplier"].to_numpy(dtype=float), 1.0)].iloc[0]
    strongest = overall.sort_values("ve_multiplier", ascending=True).iloc[0]
    bias_remains_small = bool(abs(float(strongest["mean_bias"])) <= max(0.02, abs(float(baseline["mean_bias"])) + 0.01))
    curvature_weakens = bool(float(strongest["mean_curvature"]) + 1e-12 < float(baseline["mean_curvature"]))
    non_ident_more_common = bool(float(strongest["identification_rate"]) + 1e-12 < float(baseline["identification_rate"]))
    boundary_increases = bool(
        float(strongest["mean_bootstrap_boundary_fraction"]) > float(baseline["mean_bootstrap_boundary_fraction"]) + 1e-12
    )

    lines = [
        "# Synthetic vaccine-effect stress test",
        "",
        "## Summary Table",
        "",
        "| VE | Severity | Mean bias | Mean absolute error | Identification rate | Mean curvature | Mean bootstrap boundary fraction |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"| {fmt(row['ve_multiplier'], 2)} | {classify_synthetic_ve_severity(summary_df, float(row['ve_multiplier']))} | "
            f"{fmt(row['mean_bias'])} | {fmt(row['mean_absolute_error'])} | {fmt(row['identification_rate'])} | "
            f"{fmt(row['mean_curvature'], 6)} | {fmt(row['mean_bootstrap_boundary_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Direct Answers",
            f"- Does alpha recovery remain unbiased? `{ 'mostly yes' if bias_remains_small else 'no, bias grows under stronger VE contamination' }`",
            f"- Does curvature weaken? `{ 'yes' if curvature_weakens else 'no clear weakening' }`",
            f"- Does non-identification become more common? `{ 'yes' if non_ident_more_common else 'no clear increase' }`",
            f"- Does boundary-seeking behavior increase? `{ 'yes' if boundary_increases else 'no clear increase' }`",
            "",
            "## Estimator Detail",
        ]
    )
    for estimator in ("pairwise", "collapse"):
        part = summary_df[summary_df["estimator"] == estimator].copy().sort_values("ve_multiplier", ascending=False)
        if part.empty:
            continue
        strongest_est = part.sort_values("ve_multiplier", ascending=True).iloc[0]
        lines.append(
            f"- `{estimator}`: strongest VE gives `mean_abs_error={fmt(strongest_est['mean_absolute_error'])}`, "
            f"`identification_rate={fmt(strongest_est['identification_rate'])}`, "
            f"`mean_curvature={fmt(strongest_est['mean_curvature'], 6)}`."
        )
    lines.extend(
        [
            "",
            "## Contamination Severity Conclusion",
            f"- Relative to `VE=1.0`, the strongest contamination level (`VE={fmt(strongest['ve_multiplier'], 2)}`) is classified as "
            f"`{classify_synthetic_ve_severity(summary_df, float(strongest['ve_multiplier']))}`.",
            "",
            "Interpretation:",
            "If stronger vaccine effects flatten objectives, increase boundary-seeking, or reduce identification rates, this supports the hypothesis that real cohort-specific treatment effects can make alpha harder to identify under the alpha-only working model.",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_synthetic_ve_regression(regression_check_df: pd.DataFrame, tolerance: float = 1e-9) -> None:
    if regression_check_df.empty:
        raise ValueError("Synthetic VE regression check produced no VE=1.0 rows")
    pair_max = float(np.nanmax(np.abs(regression_check_df["pairwise_delta"].to_numpy(dtype=float))))
    collapse_max = float(np.nanmax(np.abs(regression_check_df["collapse_delta"].to_numpy(dtype=float))))
    if pair_max > tolerance or collapse_max > tolerance:
        raise ValueError(
            "Synthetic VE regression gate failed: "
            f"pairwise_delta={pair_max:.6g}, collapse_delta={collapse_max:.6g}"
        )


def summarize_conditional_ve_estimates(estimates_df: pd.DataFrame, target_multiplier: float) -> pd.DataFrame:
    if estimates_df.empty:
        return pd.DataFrame()

    def safe_nanmean(series: pd.Series) -> float:
        values = series.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else float("nan")

    rows: list[dict] = []
    for ve_assumed in sorted(estimates_df["VE_assumed"].dropna().unique(), reverse=True):
        ve_part = estimates_df[estimates_df["VE_assumed"] == ve_assumed].copy()
        grouped_parts = [("overall", ve_part), *[(name, ve_part[ve_part["estimator"] == name].copy()) for name in ("pairwise", "collapse")]]
        for estimator, est_part in grouped_parts:
            if est_part.empty:
                continue
            reported_mask = np.isfinite(est_part["alpha_hat_reported"].to_numpy(dtype=float))
            rows.append(
                {
                    "dgp_ve_multiplier": float(target_multiplier),
                    "VE_assumed": float(ve_assumed),
                    "estimator": estimator,
                    "n_rows": int(len(est_part)),
                    "mean_alpha_hat_raw": safe_nanmean(est_part["alpha_hat_raw"]),
                    "mean_bias": safe_nanmean(est_part["bias"]),
                    "mean_absolute_error": safe_nanmean(est_part["absolute_error"]),
                    "identification_rate": float(np.mean(reported_mask)) if reported_mask.size else np.nan,
                    "mean_curvature": safe_nanmean(est_part["curvature_metric"]),
                    "mean_bootstrap_boundary_fraction": safe_nanmean(est_part["bootstrap_boundary_fraction"]),
                    "mean_leave_one_out_max_shift": safe_nanmean(est_part["leave_one_out_max_shift"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["VE_assumed", "estimator"], ascending=[False, True]).reset_index(drop=True)


def render_conditional_ve_report(estimates_df: pd.DataFrame, summary_df: pd.DataFrame, target_multiplier: float) -> str:
    if estimates_df.empty or summary_df.empty:
        return "# Conditional VE alpha fit\n\nNo conditional-VE estimate rows were produced.\n"

    def fmt(value: object, digits: int = 3) -> str:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(numeric):
            return "NA"
        return f"{numeric:.{digits}f}"

    overall = summary_df[summary_df["estimator"] == "overall"].copy().sort_values("VE_assumed", ascending=False)
    baseline = overall[np.isclose(overall["VE_assumed"].to_numpy(dtype=float), 1.0)].iloc[0]
    target = overall[np.isclose(overall["VE_assumed"].to_numpy(dtype=float), float(target_multiplier))].iloc[0]
    improves_mae = float(target["mean_absolute_error"]) + 1e-12 < float(baseline["mean_absolute_error"])
    improves_curvature = float(target["mean_curvature"]) > float(baseline["mean_curvature"]) + 1e-12
    improves_identification = float(target["identification_rate"]) > float(baseline["identification_rate"]) + 1e-12
    improves_boundary = float(target["mean_bootstrap_boundary_fraction"]) + 1e-12 < float(baseline["mean_bootstrap_boundary_fraction"])

    lines = [
        "# Conditional VE alpha fit",
        "",
        f"- Primary DGP VE multiplier: `{fmt(target_multiplier, 2)}`",
        "",
        "## Summary Table",
        "",
        "| VE_assumed | Mean alpha | Mean absolute error | Identification rate | Mean curvature | Mean bootstrap boundary fraction | Mean leave-one-out max shift |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in overall.iterrows():
        lines.append(
            f"| {fmt(row['VE_assumed'], 2)} | {fmt(row['mean_alpha_hat_raw'])} | {fmt(row['mean_absolute_error'])} | "
            f"{fmt(row['identification_rate'])} | {fmt(row['mean_curvature'], 6)} | "
            f"{fmt(row['mean_bootstrap_boundary_fraction'])} | {fmt(row['mean_leave_one_out_max_shift'])} |"
        )
    lines.extend(
        [
            "",
            "## Direct Answers",
            f"- Does fixing VE enable identification? `{ 'yes, identification improves for some assumed VE values' if overall['identification_rate'].max() > baseline['identification_rate'] + 1e-12 else 'no clear identification gain' }`",
            f"- How does alpha change with VE? `See the VE-alpha profile in the summary CSV and figure; mean alpha at VE={fmt(target_multiplier, 2)} is {fmt(target['mean_alpha_hat_raw'])}.`",
            f"- Is there a stable region? `{ 'yes, the target VE branch improves stability metrics' if improves_boundary or improves_curvature else 'no clearly stable VE region' }`",
            "",
            "## Primary Recovery Criterion",
            f"- Relative to `VE_assumed=1.0`, does `VE_assumed={fmt(target_multiplier, 2)}` improve mean absolute error? `{str(improves_mae).lower()}`",
            f"- Relative to `VE_assumed=1.0`, does `VE_assumed={fmt(target_multiplier, 2)}` improve curvature? `{str(improves_curvature).lower()}`",
            f"- Relative to `VE_assumed=1.0`, does `VE_assumed={fmt(target_multiplier, 2)}` improve identification rate? `{str(improves_identification).lower()}`",
            f"- Relative to `VE_assumed=1.0`, does `VE_assumed={fmt(target_multiplier, 2)}` improve bootstrap boundary fraction? `{str(improves_boundary).lower()}`",
            "",
            "## Interpretation",
            "- This conditional fit tests whether externally fixing VE makes alpha recoverable under the existing alpha-only estimator.",
            "- If the exact-match branch improves recovery and identifiability, VE was a major confounder.",
            "- If alpha still follows a broad VE-alpha tradeoff or remains non-identified, the structure is not separable without stronger external information.",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_conditional_ve_identity(identity_df: pd.DataFrame, tolerance: float = 1e-9) -> None:
    if identity_df.empty:
        raise ValueError("Conditional VE identity check produced no VE_assumed=1.0 rows")
    def finite_abs_max(series: pd.Series) -> float:
        values = np.abs(series.to_numpy(dtype=float))
        finite = values[np.isfinite(values)]
        return float(np.max(finite)) if finite.size else 0.0
    checks = {
        "alpha_hat_delta": finite_abs_max(identity_df["alpha_hat_delta"]),
        "curvature_delta": finite_abs_max(identity_df["curvature_delta"]),
        "bootstrap_boundary_delta": finite_abs_max(identity_df["bootstrap_boundary_delta"]),
        "leave_one_out_shift_delta": finite_abs_max(identity_df["leave_one_out_shift_delta"]),
    }
    failures = {name: value for name, value in checks.items() if value > tolerance}
    if failures:
        raise ValueError(f"Conditional VE identity check failed: {failures}")


def _primary_best_rows_for_mode(best_df: pd.DataFrame, cfg: dict, neutralization_mode: str) -> tuple[pd.Series, pd.Series]:
    params = cfg["analysis"]
    pair_row = _filter_best_unique(
        best_df,
        neutralization_mode=neutralization_mode,
        anchor_mode=params["primary_anchor"],
        theta_scale="gamma_primary",
        excess_mode=params["primary_excess_mode"],
        age_band="pooled",
        time_segment="pooled",
        estimator="pairwise",
    )
    collapse_row = _filter_best_unique(
        best_df,
        neutralization_mode=neutralization_mode,
        anchor_mode=params["primary_anchor"],
        theta_scale="gamma_primary",
        excess_mode=params["primary_excess_mode"],
        age_band="pooled",
        time_segment="pooled",
        estimator="collapse",
    )
    return pair_row, collapse_row


def _mode_contract_signature(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "cohort_id",
        "iso_int",
        "week_monday",
        "dose",
        "Dead",
        "Alive",
        "hazard_obs",
        "href",
        "excess",
        "theta_t_gamma",
        "theta_t_raw",
    ]
    available = [col for col in cols if col in df.columns]
    return df[available].sort_values(["cohort_id", "iso_int"]).reset_index(drop=True)


def validate_mode_contract(primary_frames: dict[str, pd.DataFrame], cfg: dict) -> dict:
    modes = list(primary_frames)
    if not modes:
        raise ValueError("No neutralization modes available for contract validation")
    base_mode = modes[0]
    base_sig = _mode_contract_signature(primary_frames[base_mode])
    checks = {
        "primary_anchor": cfg["analysis"]["primary_anchor"],
        "primary_excess_mode": cfg["analysis"]["primary_excess_mode"],
        "alpha_grid": cfg["alpha_grid"],
        "cohort_week_rows_by_mode": {},
        "contract_ok": True,
    }
    for mode, df in primary_frames.items():
        checks["cohort_week_rows_by_mode"][mode] = int(len(df))
        sig = _mode_contract_signature(df)
        if list(sig.columns) != list(base_sig.columns) or len(sig) != len(base_sig):
            raise ValueError(f"Neutralization mode contract drifted for {mode}: cohort-week row shape changed")
        if not base_sig.equals(sig):
            raise ValueError(f"Neutralization mode contract drifted for {mode}: cohort-week rows changed")
    return checks


def validate_alpha_one_invariance(
    curves: pd.DataFrame,
    primary_frames: dict[str, pd.DataFrame],
    cfg: dict,
) -> dict:
    if not np.any(np.isclose(curves["alpha"].to_numpy(dtype=float), 1.0)):
        raise ValueError("Alpha=1.0 is missing from the configured alpha grid; invariance check cannot run")
    reference_df = primary_frames[NEUTRALIZATION_REFERENCE]
    symmetric_df = primary_frames.get(NEUTRALIZATION_SYMMETRIC)
    if symmetric_df is None:
        return {"checked": False, "reason": "symmetric mode not configured"}
    ref_at_one = build_neutralized_frame(
        reference_df,
        1.0,
        theta_column="theta_t_gamma",
        neutralization_mode=NEUTRALIZATION_REFERENCE,
    )
    sym_at_one = build_neutralized_frame(
        symmetric_df,
        1.0,
        theta_column="theta_t_gamma",
        neutralization_mode=NEUTRALIZATION_SYMMETRIC,
    )
    neutralized_delta = _nanmax_abs(
        sym_at_one["excess_neutralized"].to_numpy(dtype=float) - symmetric_df["excess"].to_numpy(dtype=float)
    )
    hazard_delta = _nanmax_abs(
        sym_at_one["hazard_adjusted"].to_numpy(dtype=float) - symmetric_df["hazard_obs"].to_numpy(dtype=float)
    )
    ref_delta = _nanmax_abs(
        ref_at_one["hazard_adjusted"].to_numpy(dtype=float) - reference_df["hazard_obs"].to_numpy(dtype=float)
    )
    shared_cols = [
        "anchor_mode",
        "theta_scale",
        "excess_mode",
        "age_band",
        "time_segment",
        "estimator",
    ]
    alpha_one = curves[np.isclose(curves["alpha"], 1.0)].copy()
    ref_curves = alpha_one[alpha_one["neutralization_mode"] == NEUTRALIZATION_REFERENCE].sort_values(shared_cols)
    sym_curves = alpha_one[alpha_one["neutralization_mode"] == NEUTRALIZATION_SYMMETRIC].sort_values(shared_cols)
    if len(ref_curves) != len(sym_curves):
        raise ValueError("Alpha=1 invariance failed: mode objective slices are not aligned")
    if not ref_curves[shared_cols].reset_index(drop=True).equals(sym_curves[shared_cols].reset_index(drop=True)):
        raise ValueError("Alpha=1 invariance failed: mode objective slice labels differ")
    objective_delta = _nanmax_abs(
        ref_curves["objective"].to_numpy(dtype=float) - sym_curves["objective"].to_numpy(dtype=float)
    )
    pair_count_delta = _nanmax_abs(
        ref_curves["n_pairs"].to_numpy(dtype=float) - sym_curves["n_pairs"].to_numpy(dtype=float)
    )
    weeks_delta = _nanmax_abs(
        ref_curves["n_weeks_used"].to_numpy(dtype=float) - sym_curves["n_weeks_used"].to_numpy(dtype=float)
    )
    invariance_ok = (
        neutralized_delta <= INVARIANCE_TOL
        and hazard_delta <= INVARIANCE_TOL
        and ref_delta <= INVARIANCE_TOL
        and objective_delta <= INVARIANCE_TOL
        and pair_count_delta <= INVARIANCE_TOL
        and weeks_delta <= INVARIANCE_TOL
    )
    if not invariance_ok:
        raise ValueError(
            "Alpha=1 invariance failed: "
            f"neutralized_delta={neutralized_delta:.3e}, hazard_delta={hazard_delta:.3e}, "
            f"objective_delta={objective_delta:.3e}, pair_count_delta={pair_count_delta:.3e}, "
            f"weeks_delta={weeks_delta:.3e}"
        )
    return {
        "checked": True,
        "neutralized_excess_max_abs_delta": neutralized_delta,
        "hazard_adjusted_max_abs_delta": hazard_delta,
        "reference_hazard_identity_delta": ref_delta,
        "objective_max_abs_delta": objective_delta,
        "n_pairs_max_abs_delta": pair_count_delta,
        "n_weeks_used_max_abs_delta": weeks_delta,
        "tolerance": INVARIANCE_TOL,
        "passed": True,
    }


def compute_mode_downstream_metrics(
    primary_df: pd.DataFrame,
    cfg: dict,
    neutralization_mode: str,
    pair_alpha: float | None,
    collapse_alpha: float | None,
) -> dict:
    params = cfg["analysis"]
    alpha_for_adjustment = pair_alpha if pair_alpha is not None and np.isfinite(pair_alpha) else collapse_alpha
    if alpha_for_adjustment is None or not np.isfinite(alpha_for_adjustment):
        return {
            "cross_cohort_coherence_metric": np.nan,
            "post_neutralization_dispersion": np.nan,
            "raw_hazard_dispersion": np.nan,
            "dispersion_reduction_ratio": np.nan,
        }
    adjusted = build_neutralized_frame(
        primary_df,
        float(alpha_for_adjustment),
        theta_column="theta_t_gamma",
        neutralization_mode=neutralization_mode,
    )
    floor = build_floor(adjusted["excess"])
    coherence_terms = []
    hazard_dispersion_terms = []
    raw_dispersion_terms = []
    for _, group in adjusted.groupby("iso_int"):
        weights = collapse_weight(group["Dead"].to_numpy(dtype=float), params["weight_mode_collapse"])
        neutralized_vals = group["excess_neutralized"].to_numpy(dtype=float)
        transformed, valid = transform_excess(neutralized_vals, params["primary_excess_mode"], floor)
        mask = valid & np.isfinite(group["theta_t_gamma"].to_numpy(dtype=float))
        if int(np.count_nonzero(mask)) >= int(params["min_cohorts_per_week"]):
            coherence_terms.append(_weighted_variance(transformed[mask], weights[mask]))
            hazard_dispersion_terms.append(
                _weighted_variance(group["hazard_adjusted"].to_numpy(dtype=float)[mask], weights[mask])
            )
            raw_dispersion_terms.append(
                _weighted_variance(group["hazard_obs"].to_numpy(dtype=float)[mask], weights[mask])
            )
    coherence = float(np.nanmean(coherence_terms)) if coherence_terms else float("nan")
    hazard_dispersion = float(np.nanmean(hazard_dispersion_terms)) if hazard_dispersion_terms else float("nan")
    raw_dispersion = float(np.nanmean(raw_dispersion_terms)) if raw_dispersion_terms else float("nan")
    reduction_ratio = (
        float(hazard_dispersion / raw_dispersion)
        if np.isfinite(hazard_dispersion) and np.isfinite(raw_dispersion) and raw_dispersion > 0
        else float("nan")
    )
    return {
        "cross_cohort_coherence_metric": coherence,
        "post_neutralization_dispersion": hazard_dispersion,
        "raw_hazard_dispersion": raw_dispersion,
        "dispersion_reduction_ratio": reduction_ratio,
    }


def summarize_neutralization_mode(
    mode: str,
    cfg: dict,
    best_df: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    loo_summary: pd.DataFrame,
    run_artifact: dict,
    primary_df: pd.DataFrame,
) -> dict:
    pair_row, collapse_row = _primary_best_rows_for_mode(best_df, cfg, mode)
    pair_alpha = float(pair_row["alpha_hat"]) if np.isfinite(pair_row["alpha_hat"]) else None
    collapse_alpha = float(collapse_row["alpha_hat"]) if np.isfinite(collapse_row["alpha_hat"]) else None
    downstream = compute_mode_downstream_metrics(primary_df, cfg, mode, pair_alpha, collapse_alpha)
    sensitivity_part = best_df[best_df["neutralization_mode"] == mode].copy()
    sensitivity_identified_fraction = (
        float(np.nanmean(sensitivity_part["identified_curve"].to_numpy(dtype=float))) if not sensitivity_part.empty else float("nan")
    )
    sensitivity_boundary_fraction = (
        float(np.nanmean(sensitivity_part["boundary_optimum"].to_numpy(dtype=float))) if not sensitivity_part.empty else float("nan")
    )
    boot = bootstrap_summary.iloc[0]
    loo = loo_summary.iloc[0]
    primary = run_artifact["primary_identification"]
    row = {
        "mode": mode,
        "pairwise_alpha_raw": float(pair_row["alpha_hat"]),
        "collapse_alpha_raw": float(collapse_row["alpha_hat"]),
        "pairwise_alpha_reported": np.nan if not np.isfinite(pair_row["alpha_hat_reported"]) else float(pair_row["alpha_hat_reported"]),
        "collapse_alpha_reported": np.nan if not np.isfinite(collapse_row["alpha_hat_reported"]) else float(collapse_row["alpha_hat_reported"]),
        "pairwise_curvature": float(pair_row["curvature_metric"]),
        "collapse_curvature": float(collapse_row["curvature_metric"]),
        "pairwise_boundary_optimum": int(pair_row["boundary_optimum"]),
        "collapse_boundary_optimum": int(collapse_row["boundary_optimum"]),
        "estimator_gap": np.nan if primary["estimator_gap"] is None else float(primary["estimator_gap"]),
        "bootstrap_finite_fraction": float(boot["finite_fraction"]),
        "bootstrap_iqr": float(boot["iqr_alpha_hat"]),
        "bootstrap_boundary_fraction": float(boot["boundary_fraction"]),
        "leave_one_out_max_shift": float(loo["max_abs_shift"]),
        "leave_one_out_large_shift_count": int(loo["large_shift_count"]),
        "cross_cohort_coherence_metric": downstream["cross_cohort_coherence_metric"],
        "post_neutralization_dispersion": downstream["post_neutralization_dispersion"],
        "raw_hazard_dispersion": downstream["raw_hazard_dispersion"],
        "dispersion_reduction_ratio": downstream["dispersion_reduction_ratio"],
        "sensitivity_identified_fraction": sensitivity_identified_fraction,
        "sensitivity_boundary_fraction": sensitivity_boundary_fraction,
        "final_status": str(primary["status"]),
        "failure_reasons": "; ".join(primary["failure_reasons"]) if primary["failure_reasons"] else "",
    }
    return row


def determine_neutralization_recommendation(comparison_df: pd.DataFrame) -> tuple[str, dict]:
    if comparison_df.empty:
        return "keep_reference_anchored_baseline", {"reason": "comparison table is empty"}
    ref = comparison_df[comparison_df["mode"] == NEUTRALIZATION_REFERENCE]
    sym = comparison_df[comparison_df["mode"] == NEUTRALIZATION_SYMMETRIC]
    if ref.empty or sym.empty:
        return "keep_reference_anchored_baseline", {"reason": "both comparison modes were not available"}
    ref_row = ref.iloc[0]
    sym_row = sym.iloc[0]
    status_rank = {"not_identified": 0, "identified": 1}

    def safe_le(lhs: float, rhs: float) -> bool:
        if np.isfinite(lhs) and np.isfinite(rhs):
            return bool(lhs <= rhs + 1e-12)
        return not np.isfinite(lhs) and not np.isfinite(rhs)

    def safe_ge(lhs: float, rhs: float) -> bool:
        if np.isfinite(lhs) and np.isfinite(rhs):
            return bool(lhs + 1e-12 >= rhs)
        return not np.isfinite(lhs) and not np.isfinite(rhs)

    ident_not_worse = all(
        [
            safe_ge(float(sym_row["pairwise_curvature"]), float(ref_row["pairwise_curvature"])),
            safe_ge(float(sym_row["collapse_curvature"]), float(ref_row["collapse_curvature"])),
            safe_ge(float(sym_row["bootstrap_finite_fraction"]), float(ref_row["bootstrap_finite_fraction"])),
            safe_le(float(sym_row["bootstrap_iqr"]), float(ref_row["bootstrap_iqr"])),
            safe_le(float(sym_row["bootstrap_boundary_fraction"]), float(ref_row["bootstrap_boundary_fraction"])),
            safe_le(float(sym_row["leave_one_out_max_shift"]), float(ref_row["leave_one_out_max_shift"])),
            safe_le(float(sym_row["estimator_gap"]), float(ref_row["estimator_gap"])),
            safe_ge(float(sym_row["sensitivity_identified_fraction"]), float(ref_row["sensitivity_identified_fraction"])),
            safe_le(float(sym_row["sensitivity_boundary_fraction"]), float(ref_row["sensitivity_boundary_fraction"])),
        ]
    )
    coherence_improves = all(
        [
            safe_le(float(sym_row["cross_cohort_coherence_metric"]), float(ref_row["cross_cohort_coherence_metric"])),
            safe_le(float(sym_row["post_neutralization_dispersion"]), float(ref_row["post_neutralization_dispersion"])),
            safe_le(float(sym_row["dispersion_reduction_ratio"]), float(ref_row["dispersion_reduction_ratio"])),
        ]
    )
    no_new_instability = all(
        [
            safe_le(float(sym_row["pairwise_boundary_optimum"]), float(ref_row["pairwise_boundary_optimum"])),
            safe_le(float(sym_row["collapse_boundary_optimum"]), float(ref_row["collapse_boundary_optimum"])),
            safe_le(float(sym_row["bootstrap_boundary_fraction"]), float(ref_row["bootstrap_boundary_fraction"])),
            safe_le(float(sym_row["leave_one_out_large_shift_count"]), float(ref_row["leave_one_out_large_shift_count"])),
        ]
    )
    status_improves = status_rank.get(str(sym_row["final_status"]), -1) > status_rank.get(str(ref_row["final_status"]), -1)
    recommend_symmetric = ident_not_worse and coherence_improves and no_new_instability and status_improves
    detail = {
        "identifiability_not_worse": ident_not_worse,
        "coherence_improves": coherence_improves,
        "no_new_instability": no_new_instability,
        "status_improves": status_improves,
    }
    return (
        "recommend_symmetric_all_cohorts" if recommend_symmetric else "keep_reference_anchored_baseline",
        detail,
    )


def render_neutralization_comparison_report(
    comparison_df: pd.DataFrame,
    recommendation: str,
    recommendation_detail: dict,
    contract_check: dict,
    invariance_check: dict,
) -> str:
    def fmt(value: object, digits: int = 3) -> str:
        if value is None:
            return "NA"
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(numeric):
            return "NA"
        return f"{numeric:.{digits}f}"

    lines = [
        "# Sandbox-only NPH neutralization comparison",
        "Not integrated into production KCOR.py",
        "",
        "## Contract checks",
        f"- Neutralization modes compared: `{', '.join(comparison_df['mode'].tolist())}`",
        f"- Shared cohort-week rows by mode: `{contract_check['cohort_week_rows_by_mode']}`",
        f"- Alpha=1 invariance check: `{'passed' if invariance_check.get('passed') else 'not_run'}`",
    ]
    if invariance_check.get("checked"):
        lines.extend(
            [
                f"- Alpha=1 neutralized-excess max delta: `{fmt(invariance_check.get('neutralized_excess_max_abs_delta'), 6)}`",
                f"- Alpha=1 objective max delta: `{fmt(invariance_check.get('objective_max_abs_delta'), 6)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Mode comparison",
            "",
            "| Mode | Pair raw | Collapse raw | Pair curv | Collapse curv | Bootstrap boundary | LOO max shift | Coherence | Dispersion ratio | Status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in comparison_df.iterrows():
        lines.append(
            f"| {row['mode']} | {fmt(row['pairwise_alpha_raw'])} | {fmt(row['collapse_alpha_raw'])} | "
            f"{fmt(row['pairwise_curvature'], 6)} | {fmt(row['collapse_curvature'], 6)} | "
            f"{fmt(row['bootstrap_boundary_fraction'])} | {fmt(row['leave_one_out_max_shift'])} | "
            f"{fmt(row['cross_cohort_coherence_metric'], 6)} | {fmt(row['dispersion_reduction_ratio'], 6)} | {row['final_status']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            f"- Downstream coherence improves under symmetric mode: `{recommendation_detail.get('coherence_improves')}`",
            f"- Identifiability is not worse under symmetric mode: `{recommendation_detail.get('identifiability_not_worse')}`",
            f"- Symmetric mode avoids new instability: `{recommendation_detail.get('no_new_instability')}`",
            f"- Symmetric mode improves the pooled final status: `{recommendation_detail.get('status_improves')}`",
            "",
            "## Recommendation",
            f"- Recommendation: `{recommendation}`",
        ]
    )
    if recommendation == "recommend_symmetric_all_cohorts":
        lines.append("- Symmetric all-cohort neutralization is the stronger sandbox formulation on the current diagnostics.")
    else:
        lines.append("- Keep `reference_anchored` as the sandbox baseline; treat the symmetric path as exploratory unless later runs improve the diagnostics.")
    if np.allclose(
        comparison_df["collapse_alpha_raw"].to_numpy(dtype=float),
        comparison_df["collapse_alpha_raw"].iloc[0],
        equal_nan=True,
    ):
        lines.append("- The collapse estimator was effectively invariant across modes in this run, so the comparison signal comes mainly from pairwise and downstream coherence diagnostics.")
    return "\n".join(lines) + "\n"


def plot_neutralization_mode_comparison(
    curves: pd.DataFrame,
    best_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    out_path: Path,
    cfg: dict,
) -> None:
    plt = _import_matplotlib()
    if plt is None or comparison_df.empty:
        return
    params = cfg["analysis"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, mode, title in zip(
        axes[:2],
        [NEUTRALIZATION_REFERENCE, NEUTRALIZATION_SYMMETRIC],
        ["A. reference_anchored", "B. symmetric_all_cohorts"],
    ):
        subset = curves[
            (curves["neutralization_mode"] == mode)
            & (curves["anchor_mode"] == params["primary_anchor"])
            & (curves["theta_scale"] == "gamma_primary")
            & (curves["excess_mode"] == params["primary_excess_mode"])
            & (curves["age_band"] == "pooled")
            & (curves["time_segment"] == "pooled")
        ].copy()
        for estimator, color, linestyle in [("pairwise", "tab:blue", "-"), ("collapse", "tab:orange", "--")]:
            est_df = subset[subset["estimator"] == estimator].sort_values("alpha")
            if est_df.empty:
                continue
            alpha_grid, obj_grid = _resample_plot_grid(est_df["alpha"].to_numpy(), est_df["objective"].to_numpy())
            obj_grid = obj_grid - np.nanmin(obj_grid)
            ax.plot(alpha_grid, obj_grid, color=color, ls=linestyle, lw=2, label=estimator.capitalize())
            best = _filter_best_unique(
                best_df,
                neutralization_mode=mode,
                anchor_mode=params["primary_anchor"],
                theta_scale="gamma_primary",
                excess_mode=params["primary_excess_mode"],
                age_band="pooled",
                time_segment="pooled",
                estimator=estimator,
            )
            ax.axvline(float(best["alpha_hat"]), color=color, ls=linestyle, lw=1.2, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("alpha")
        ax.set_ylabel("Normalized objective")
        ax.grid(True, ls=":", alpha=0.4)
    axes[0].legend()
    ax = axes[2]
    plot_df = comparison_df.set_index("mode")
    x_pos = np.arange(len(plot_df.index), dtype=float)
    ax.bar(x_pos - 0.18, plot_df["cross_cohort_coherence_metric"], width=0.36, label="Coherence metric")
    ax.bar(x_pos + 0.18, plot_df["dispersion_reduction_ratio"], width=0.36, label="Dispersion ratio")
    ax.set_xticks(x_pos, plot_df.index.tolist(), rotation=10)
    ax.set_title("C. Downstream coherence")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_objectives(curves: pd.DataFrame, best_df: pd.DataFrame, out_path: Path, cfg: dict) -> None:
    plt = _import_matplotlib()
    if plt is None:
        return

    primary_mode = get_primary_nph_neutralization_mode(cfg)
    primary = curves[
        (curves["neutralization_mode"] == primary_mode)
        & (curves["anchor_mode"] == "dose0")
        & (curves["theta_scale"] == "gamma_primary")
        & (curves["excess_mode"] == "exclude_nonpositive")
        & (curves["age_band"] == "pooled")
        & (curves["time_segment"] == "pooled")
    ].copy()
    if primary.empty:
        print(f"[ALPHA] Skipping {out_path.name}: no primary objective rows", flush=True)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, estimator in zip(axes, ["pairwise", "collapse"]):
        est_df = primary[primary["estimator"] == estimator].sort_values("alpha")
        if est_df.empty:
            continue
        ax.plot(est_df["alpha"], est_df["objective"], lw=2)
        best = best_df[
            (best_df["neutralization_mode"] == primary_mode)
            & (best_df["anchor_mode"] == "dose0")
            & (best_df["theta_scale"] == "gamma_primary")
            & (best_df["excess_mode"] == "exclude_nonpositive")
            & (best_df["age_band"] == "pooled")
            & (best_df["time_segment"] == "pooled")
            & (best_df["estimator"] == estimator)
        ]
        if not best.empty:
            alpha_hat = float(best.iloc[0]["alpha_hat"])
            ax.axvline(alpha_hat, color="tab:red", ls="--", lw=1)
        ax.set_title(estimator)
        ax.set_xlabel("alpha")
        ax.set_ylabel("objective")
        ax.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_manuscript_synthetic_recovery(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        print(f"[ALPHA] Skipping {out_path.name}: synthetic recovery is empty", flush=True)
        return
    plt = _import_matplotlib()
    if plt is None:
        return

    title_map = {
        "lognormal_fixed": "A. Baseline synthetic branch",
        "heteroskedastic_lognormal": "B. Heteroskedastic branch",
    }
    preferred_order = ["lognormal_fixed", "heteroskedastic_lognormal"]
    observed_models = list(df["noise_model"].dropna().unique()) if "noise_model" in df.columns else ["synthetic"]
    noise_models = [name for name in preferred_order if name in observed_models]
    noise_models.extend(name for name in observed_models if name not in noise_models)
    fig, axes = plt.subplots(1, len(noise_models), figsize=(6.4 * len(noise_models), 4.8), sharex=True, sharey=True)
    if len(noise_models) == 1:
        axes = [axes]
    panel_summaries: list[pd.DataFrame] = []
    for ax, noise_model in zip(axes, noise_models):
        part = df[df["noise_model"] == noise_model] if "noise_model" in df.columns else df
        summary = (
            part.groupby("alpha_true", as_index=False)
            .agg(
                pairwise_mean=("pairwise_alpha_hat", "mean"),
                pairwise_sd=("pairwise_alpha_hat", "std"),
                collapse_mean=("collapse_alpha_hat", "mean"),
                collapse_sd=("collapse_alpha_hat", "std"),
            )
            .sort_values("alpha_true")
        )
        summary["pairwise_sd"] = summary["pairwise_sd"].fillna(0.0)
        summary["collapse_sd"] = summary["collapse_sd"].fillna(0.0)
        panel_summaries.append(summary.copy())
        ax.errorbar(
            summary["alpha_true"],
            summary["pairwise_mean"],
            yerr=summary["pairwise_sd"],
            fmt="o-",
            color="tab:blue",
            lw=2,
            capsize=3,
            label="Pairwise",
        )
        ax.errorbar(
            summary["alpha_true"],
            summary["collapse_mean"],
            yerr=summary["collapse_sd"],
            fmt="s--",
            color="tab:orange",
            lw=2,
            capsize=3,
            label="Collapse",
        )
        ax.plot(summary["alpha_true"], summary["alpha_true"], color="0.5", lw=1.5, ls=":", label="Identity")
        pairwise_mae = float(np.mean(np.abs(summary["pairwise_mean"] - summary["alpha_true"])))
        collapse_mae = float(np.mean(np.abs(summary["collapse_mean"] - summary["alpha_true"])))
        ax.text(
            0.03,
            0.97,
            f"Pairwise MAE = {pairwise_mae:.3f}\nCollapse MAE = {collapse_mae:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
        ax.set_title(title_map.get(noise_model, str(noise_model)), fontsize=12)
        ax.set_xlabel("True alpha")
        ax.grid(True, ls=":", alpha=0.4)
    x_vals = np.concatenate([summary["alpha_true"].to_numpy(dtype=float) for summary in panel_summaries])
    y_candidates = []
    for summary in panel_summaries:
        y_candidates.extend(summary["pairwise_mean"].to_numpy(dtype=float))
        y_candidates.extend((summary["pairwise_mean"] + summary["pairwise_sd"]).to_numpy(dtype=float))
        y_candidates.extend((summary["pairwise_mean"] - summary["pairwise_sd"]).to_numpy(dtype=float))
        y_candidates.extend(summary["collapse_mean"].to_numpy(dtype=float))
        y_candidates.extend((summary["collapse_mean"] + summary["collapse_sd"]).to_numpy(dtype=float))
        y_candidates.extend((summary["collapse_mean"] - summary["collapse_sd"]).to_numpy(dtype=float))
        y_candidates.extend(summary["alpha_true"].to_numpy(dtype=float))
    x_pad = 0.02
    y_vals = np.asarray(y_candidates, dtype=float)
    y_pad = 0.03
    for ax in axes:
        ax.set_xlim(float(np.nanmin(x_vals)) - x_pad, float(np.nanmax(x_vals)) + x_pad)
        ax.set_ylim(float(np.nanmin(y_vals)) - y_pad, float(np.nanmax(y_vals)) + y_pad)
    axes[0].set_ylabel("Estimated alpha")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.99))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_manuscript_czech_objective(curves: pd.DataFrame, best_df: pd.DataFrame, out_path: Path, cfg: dict) -> None:
    plt = _import_matplotlib()
    if plt is None:
        return

    primary_mode = get_primary_nph_neutralization_mode(cfg)
    primary = curves[
        (curves["neutralization_mode"] == primary_mode)
        & (curves["anchor_mode"] == "dose0")
        & (curves["theta_scale"] == "gamma_primary")
        & (curves["excess_mode"] == "exclude_nonpositive")
        & (curves["age_band"] == "pooled")
        & (curves["time_segment"] == "pooled")
    ].copy()
    if primary.empty:
        raise ValueError("Primary objective figure rows not found for pooled Czech specification")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pair_row = _filter_best_unique(
        best_df,
        neutralization_mode=primary_mode,
        anchor_mode="dose0",
        theta_scale="gamma_primary",
        excess_mode="exclude_nonpositive",
        age_band="pooled",
        time_segment="pooled",
        estimator="pairwise",
    )
    collapse_row = _filter_best_unique(
        best_df,
        neutralization_mode=primary_mode,
        anchor_mode="dose0",
        theta_scale="gamma_primary",
        excess_mode="exclude_nonpositive",
        age_band="pooled",
        time_segment="pooled",
        estimator="collapse",
    )
    _warn_if_manuscript_mismatch(float(pair_row["alpha_hat"]), float(collapse_row["alpha_hat"]))
    curve_max = 0.0
    for estimator, color, linestyle in [("pairwise", "tab:blue", "-"), ("collapse", "tab:orange", "--")]:
        est_df = primary[primary["estimator"] == estimator].sort_values("alpha")
        if est_df.empty:
            raise ValueError(f"Missing objective rows for estimator={estimator}")
        alpha_grid, obj_grid = _resample_plot_grid(est_df["alpha"].to_numpy(), est_df["objective"].to_numpy())
        obj_grid = obj_grid - np.nanmin(obj_grid)
        curve_max = max(curve_max, float(np.nanmax(obj_grid)))
        ax.plot(alpha_grid, obj_grid, color=color, ls=linestyle, lw=2, label=estimator.capitalize())
    ax.axvline(float(pair_row["alpha_hat"]), color="tab:blue", ls="-", lw=1.5, alpha=0.7)
    ax.axvline(float(collapse_row["alpha_hat"]), color="tab:orange", ls="--", lw=1.5, alpha=0.7)
    y_top = min(25.0, curve_max * 1.03 if curve_max > 0 else 25.0)
    ax.set_ylim(0.0, y_top)
    ax.text(
        float(pair_row["alpha_hat"]) + 0.002,
        y_top * 0.97,
        f"alpha_pair = {float(pair_row['alpha_hat']):.2f}",
        color="tab:blue",
        rotation=90,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.text(
        float(collapse_row["alpha_hat"]) - 0.002,
        y_top * 0.97,
        f"alpha_coll = {float(collapse_row['alpha_hat']):.2f}",
        color="tab:orange",
        rotation=90,
        va="top",
        ha="right",
        fontsize=9,
    )
    ax.set_xlabel("alpha")
    ax.set_ylabel("Normalized objective")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_manuscript_czech_diagnostics(
    loo_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    best_df: pd.DataFrame,
    out_path: Path,
    cfg: dict,
) -> None:
    plt = _import_matplotlib()
    if plt is None:
        return
    if loo_df.empty:
        raise ValueError("Leave-one-out diagnostics are empty")
    if bootstrap_df.empty:
        raise ValueError("Bootstrap diagnostics are empty")
    primary_mode = get_primary_nph_neutralization_mode(cfg)
    if "neutralization_mode" in loo_df.columns:
        loo_df = loo_df[loo_df["neutralization_mode"] == primary_mode].copy()
    if "neutralization_mode" in bootstrap_df.columns:
        bootstrap_df = bootstrap_df[bootstrap_df["neutralization_mode"] == primary_mode].copy()
    pair_row = _filter_best_unique(
        best_df,
        neutralization_mode=primary_mode,
        anchor_mode="dose0",
        theta_scale="gamma_primary",
        excess_mode="exclude_nonpositive",
        age_band="pooled",
        time_segment="pooled",
        estimator="pairwise",
    )
    pooled_pair = float(pair_row["alpha_hat"])
    seg = best_df[
        (best_df["neutralization_mode"] == primary_mode)
        & (best_df["anchor_mode"] == "dose0")
        & (best_df["theta_scale"] == "gamma_primary")
        & (best_df["excess_mode"] == "exclude_nonpositive")
        & (best_df["age_band"] == "pooled")
        & (best_df["time_segment"].isin(["pooled", "early_wave", "late_wave"]))
        & (best_df["estimator"].isin(["pairwise", "collapse"]))
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    loo_plot = loo_df.dropna(subset=["alpha_hat"]).reset_index(drop=True).copy()
    loo_plot["idx"] = np.arange(1, len(loo_plot) + 1)
    influential = np.abs(loo_plot["alpha_hat"] - pooled_pair) > 0.02
    ax.scatter(loo_plot.loc[~influential, "idx"], loo_plot.loc[~influential, "alpha_hat"], color="tab:blue", s=20)
    if influential.any():
        ax.scatter(loo_plot.loc[influential, "idx"], loo_plot.loc[influential, "alpha_hat"], color="tab:red", s=26)
    ax.axhline(pooled_pair, color="0.4", ls="--", lw=1.5)
    ax.set_title("A. Leave-one-out")
    ax.set_xlabel("Omitted cohort index")
    ax.set_ylabel("Estimated alpha")
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[1]
    boot_plot = bootstrap_df["alpha_hat"].dropna().to_numpy()
    if boot_plot.size == 0:
        raise ValueError("No finite bootstrap alpha estimates found")
    ax.hist(boot_plot, bins=min(12, max(6, boot_plot.size // 2)), color="tab:blue", alpha=0.75, edgecolor="white")
    ax.axvline(pooled_pair, color="0.4", ls="--", lw=1.5)
    ymax = ax.get_ylim()[1]
    ax.text(
        pooled_pair + 0.003,
        ymax * 0.95,
        f"alpha_pooled approx {pooled_pair:.2f}",
        color="0.35",
        rotation=90,
        va="top",
        ha="left",
        fontsize=9,
    )
    ax.set_title("B. Bootstrap")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Count")
    ax.grid(True, ls=":", alpha=0.4)

    ax = axes[2]
    seg["time_segment"] = pd.Categorical(seg["time_segment"], categories=["pooled", "early_wave", "late_wave"], ordered=True)
    seg = seg.sort_values(["time_segment", "estimator"])
    positions = np.arange(3, dtype=float)
    position_map = {"pooled": 0.0, "early_wave": 1.0, "late_wave": 2.0}
    for estimator, color, marker, offset in [("pairwise", "tab:blue", "o", -0.08), ("collapse", "tab:orange", "s", 0.08)]:
        part = seg[seg["estimator"] == estimator].sort_values("time_segment")
        if part.empty:
            continue
        x_vals = [position_map[str(name)] + offset for name in part["time_segment"].astype(str)]
        ax.plot(x_vals, part["alpha_hat"], color=color, ls="-" if estimator == "pairwise" else "--", marker=marker, lw=2, label=estimator.capitalize())
    ax.axhline(pooled_pair, color="0.4", ls=":", lw=1.5)
    ax.set_xticks(positions, ["pooled", "early_wave", "late_wave"])
    ax.set_title("C. Segmented estimates")
    ax.set_xlabel("Time segment")
    ax.set_ylabel("alpha")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_synthetic_recovery(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        print(f"[ALPHA] Skipping {out_path.name}: synthetic recovery is empty", flush=True)
        return
    plt = _import_matplotlib()
    if plt is None:
        return

    noise_models = list(df["noise_model"].dropna().unique()) if "noise_model" in df.columns else ["synthetic"]
    fig, axes = plt.subplots(1, len(noise_models), figsize=(6 * len(noise_models), 4), sharey=True)
    if len(noise_models) == 1:
        axes = [axes]
    for ax, noise_model in zip(axes, noise_models):
        part = df[df["noise_model"] == noise_model] if "noise_model" in df.columns else df
        summary = (
            part.groupby("alpha_true", as_index=False)
            .agg(
                pairwise_mean=("pairwise_alpha_hat", "mean"),
                collapse_mean=("collapse_alpha_hat", "mean"),
            )
            .sort_values("alpha_true")
        )
        ax.plot(summary["alpha_true"], summary["pairwise_mean"], "o-", label="pairwise")
        ax.plot(summary["alpha_true"], summary["collapse_mean"], "s-", label="collapse")
        ax.plot(summary["alpha_true"], summary["alpha_true"], ls="--", color="black", label="truth")
        ax.set_title(str(noise_model))
        ax.set_xlabel("alpha_true")
        ax.set_ylabel("recovered alpha")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend()
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_synthetic_vaccine_effect(recovery_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    if recovery_df.empty or summary_df.empty:
        print(f"[ALPHA] Skipping {out_path.name}: synthetic VE recovery is empty", flush=True)
        return
    plt = _import_matplotlib()
    if plt is None:
        return

    ve_levels = sorted(recovery_df["ve_multiplier"].dropna().unique(), reverse=True)
    cmap = plt.get_cmap("viridis")
    colors = {ve: cmap(idx / max(len(ve_levels) - 1, 1)) for idx, ve in enumerate(ve_levels)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    for ve in ve_levels:
        part = recovery_df[recovery_df["ve_multiplier"] == ve].copy()
        if part.empty:
            continue
        for estimator, ls, marker in [("pairwise", "-", "o"), ("collapse", "--", "s")]:
            est = part[part["estimator"] == estimator].copy()
            if est.empty:
                continue
            summary = (
                est.groupby("alpha_true", as_index=False)
                .agg(alpha_hat_mean=("alpha_hat_raw", "mean"))
                .sort_values("alpha_true")
            )
            ax.plot(
                summary["alpha_true"],
                summary["alpha_hat_mean"],
                linestyle=ls,
                marker=marker,
                color=colors[ve],
                label=f"{estimator} VE={ve:.2f}",
            )
    truth = (
        recovery_df.groupby("alpha_true", as_index=False)
        .agg(alpha_true=("alpha_true", "first"))
        .sort_values("alpha_true")
    )
    ax.plot(truth["alpha_true"], truth["alpha_true"], color="0.35", ls=":", lw=1.5, label="truth")
    ax.set_title("A. Recovery vs true alpha")
    ax.set_xlabel("True alpha")
    ax.set_ylabel("Estimated alpha")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    overall = summary_df[summary_df["estimator"] == "overall"].copy().sort_values("ve_multiplier", ascending=False)
    ax.plot(overall["ve_multiplier"], overall["identification_rate"], "o-", color="tab:blue", label="identification_rate")
    ax.set_title("B. Identifiability degradation")
    ax.set_xlabel("VE multiplier")
    ax.set_ylabel("Identification rate", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.grid(True, ls=":", alpha=0.4)
    ax2 = ax.twinx()
    ax2.plot(overall["ve_multiplier"], overall["mean_curvature"], "s--", color="tab:orange", label="mean_curvature")
    ax2.set_ylabel("Mean curvature", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax = axes[2]
    ax.plot(
        overall["ve_multiplier"],
        overall["mean_bootstrap_boundary_fraction"],
        "o-",
        color="tab:red",
        label="boundary_fraction",
    )
    ax.set_title("C. Boundary-seeking")
    ax.set_xlabel("VE multiplier")
    ax.set_ylabel("Mean bootstrap boundary fraction")
    ax.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def plot_conditional_ve_alpha(summary_df: pd.DataFrame, out_path: Path) -> None:
    if summary_df.empty:
        print(f"[ALPHA] Skipping {out_path.name}: conditional VE summary is empty", flush=True)
        return
    plt = _import_matplotlib()
    if plt is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    estimator_rows = summary_df[summary_df["estimator"].isin(["pairwise", "collapse"])].copy()

    ax = axes[0]
    for estimator, color, marker in [("pairwise", "tab:blue", "o"), ("collapse", "tab:orange", "s")]:
        part = estimator_rows[estimator_rows["estimator"] == estimator].copy().sort_values("VE_assumed", ascending=False)
        if part.empty:
            continue
        ax.plot(part["VE_assumed"], part["mean_alpha_hat_raw"], color=color, lw=2, label=estimator)
        identified_mask = part["identification_rate"].to_numpy(dtype=float) > 0.0
        ax.scatter(
            part.loc[identified_mask, "VE_assumed"],
            part.loc[identified_mask, "mean_alpha_hat_raw"],
            color=color,
            marker=marker,
            s=55,
        )
        ax.scatter(
            part.loc[~identified_mask, "VE_assumed"],
            part.loc[~identified_mask, "mean_alpha_hat_raw"],
            facecolors="white",
            edgecolors=color,
            marker=marker,
            s=55,
        )
    ax.set_title("A. Alpha vs assumed VE")
    ax.set_xlabel("VE_assumed")
    ax.set_ylabel("Mean alpha_hat_raw")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend()

    ax = axes[1]
    overall = summary_df[summary_df["estimator"] == "overall"].copy().sort_values("VE_assumed", ascending=False)
    ax.plot(overall["VE_assumed"], overall["mean_curvature"], "o-", color="tab:green", label="mean_curvature")
    ax.set_title("B. Curvature vs assumed VE")
    ax.set_xlabel("VE_assumed")
    ax.set_ylabel("Mean curvature")
    ax.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def generate_manuscript_figures(outdir: Path, repo_root: Path, cfg: dict) -> None:
    manuscript_dir = _manuscript_figures_dir(repo_root)
    required = [
        outdir / "alpha_synthetic_recovery.csv",
        outdir / "alpha_objective_curves.csv",
        outdir / "alpha_best_estimates.csv",
        outdir / "alpha_leave_one_out.csv",
        outdir / "alpha_bootstrap.csv",
    ]
    _require_files(required)
    synthetic_df = pd.read_csv(outdir / "alpha_synthetic_recovery.csv")
    curves_df = pd.read_csv(outdir / "alpha_objective_curves.csv")
    best_df = pd.read_csv(outdir / "alpha_best_estimates.csv")
    loo_df = pd.read_csv(outdir / "alpha_leave_one_out.csv")
    bootstrap_df = pd.read_csv(outdir / "alpha_bootstrap.csv")
    plot_manuscript_synthetic_recovery(synthetic_df, manuscript_dir / "fig_alpha_synthetic_recovery.png")
    plot_manuscript_czech_objective(curves_df, best_df, manuscript_dir / "fig_alpha_czech_objective.png", cfg)
    plot_manuscript_czech_diagnostics(loo_df, bootstrap_df, best_df, manuscript_dir / "fig_alpha_czech_diagnostics.png", cfg)


def write_outputs(
    outdir: Path,
    repo_root: Path,
    cfg: dict,
    cohort_diag: pd.DataFrame,
    wave_table: pd.DataFrame,
    curves: pd.DataFrame,
    best_df: pd.DataFrame,
    theta_scale_summary: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    loo_summary: pd.DataFrame,
    sensitivity_df: pd.DataFrame,
    decision_summary: pd.DataFrame,
    calibration_choice_summary: pd.DataFrame,
    loo_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    synthetic_ve_recovery_df: pd.DataFrame,
    synthetic_ve_summary_df: pd.DataFrame,
    synthetic_ve_report: str,
    conditional_ve_estimates_df: pd.DataFrame,
    conditional_ve_summary_df: pd.DataFrame,
    conditional_ve_report: str,
    run_artifact: dict,
    identifiability_report: str,
    neutralization_comparison_df: pd.DataFrame,
    neutralization_comparison_report: str,
    neutralization_comparison_artifact: dict,
    wave_identifiability_df: pd.DataFrame,
    wave_identifiability_report: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cohort_diag.to_csv(outdir / "alpha_cohort_diagnostics.csv", index=False)
    wave_table.to_csv(outdir / "alpha_wave_table.csv", index=False)
    curves.to_csv(outdir / "alpha_objective_curves.csv", index=False)
    best_df.to_csv(outdir / "alpha_best_estimates.csv", index=False)
    theta_scale_summary.to_csv(outdir / "alpha_theta_scale_summary.csv", index=False)
    bootstrap_summary.to_csv(outdir / "alpha_bootstrap_summary.csv", index=False)
    loo_summary.to_csv(outdir / "alpha_leave_one_out_summary.csv", index=False)
    sensitivity_df.to_csv(outdir / "alpha_primary_sensitivity_slices.csv", index=False)
    decision_summary.to_csv(outdir / "alpha_decision_summary.csv", index=False)
    calibration_choice_summary.to_csv(outdir / "alpha_calibration_choice.csv", index=False)
    loo_df.to_csv(outdir / "alpha_leave_one_out.csv", index=False)
    bootstrap_df.to_csv(outdir / "alpha_bootstrap.csv", index=False)
    synthetic_df.to_csv(outdir / "alpha_synthetic_recovery.csv", index=False)
    synthetic_ve_recovery_df.to_csv(outdir / "alpha_synthetic_vaccine_effect_recovery.csv", index=False)
    synthetic_ve_summary_df.to_csv(outdir / "alpha_synthetic_vaccine_effect_summary.csv", index=False)
    (outdir / "alpha_synthetic_vaccine_effect_report.md").write_text(synthetic_ve_report, encoding="utf-8")
    conditional_ve_estimates_df.to_csv(outdir / "alpha_conditional_VE_estimates.csv", index=False)
    conditional_ve_summary_df.to_csv(outdir / "alpha_conditional_VE_summary.csv", index=False)
    (outdir / "alpha_conditional_VE_report.md").write_text(conditional_ve_report, encoding="utf-8")
    (outdir / "alpha_run_artifact.json").write_text(json.dumps(run_artifact, indent=2) + "\n", encoding="utf-8")
    (outdir / "alpha_identifiability_report.md").write_text(identifiability_report, encoding="utf-8")
    neutralization_comparison_df.to_csv(outdir / "alpha_neutralization_mode_comparison.csv", index=False)
    (outdir / "alpha_neutralization_comparison_report.md").write_text(
        neutralization_comparison_report,
        encoding="utf-8",
    )
    (outdir / "alpha_neutralization_run_artifact.json").write_text(
        json.dumps(neutralization_comparison_artifact, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_objectives(curves, best_df, outdir / "fig_alpha_objectives.png", cfg)
    plot_synthetic_recovery(synthetic_df, outdir / "fig_alpha_synthetic_recovery.png")
    plot_synthetic_vaccine_effect(
        synthetic_ve_recovery_df,
        synthetic_ve_summary_df,
        outdir / "fig_alpha_synthetic_vaccine_effect.png",
    )
    plot_conditional_ve_alpha(
        conditional_ve_summary_df,
        outdir / "fig_alpha_vs_assumed_VE.png",
    )
    plot_neutralization_mode_comparison(
        curves,
        best_df,
        neutralization_comparison_df,
        outdir / "fig_alpha_neutralization_mode_comparison.png",
        cfg,
    )
    if not wave_identifiability_df.empty:
        wave_identifiability_df.to_csv(outdir / "alpha_wave_identifiability.csv", index=False)
        (outdir / "alpha_wave_identifiability_report.md").write_text(wave_identifiability_report, encoding="utf-8")
        plot_wave_identifiability(wave_identifiability_df, outdir / "fig_wave_vs_identifiability.png")
    if should_write_manuscript_figures(cfg):
        generate_manuscript_figures(outdir, repo_root, cfg)


def print_synthetic_vaccine_effect_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("SYNTHETIC VE STRESS TEST", flush=True)
        print("no synthetic VE summary rows", flush=True)
        return
    overall = summary_df[summary_df["estimator"] == "overall"].copy().sort_values("ve_multiplier", ascending=False)
    print("SYNTHETIC VE STRESS TEST", flush=True)
    for _, row in overall.iterrows():
        print(
            f"VE={float(row['ve_multiplier']):.2f} "
            f"identified_rate={float(row['identification_rate']):.3f} "
            f"mean_abs_error={float(row['mean_absolute_error']):.3f} "
            f"boundary_fraction={float(row['mean_bootstrap_boundary_fraction']):.3f}",
            flush=True,
        )


def print_conditional_ve_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("CONDITIONAL ALPHA FIT", flush=True)
        print("no conditional VE summary rows", flush=True)
        return
    overall = summary_df[summary_df["estimator"] == "overall"].copy().sort_values("VE_assumed", ascending=False)
    print("CONDITIONAL ALPHA FIT", flush=True)
    for _, row in overall.iterrows():
        print(
            f"VE={float(row['VE_assumed']):.2f} "
            f"alpha={float(row['mean_alpha_hat_raw']):.3f} "
            f"identified={float(row['identification_rate']):.3f} "
            f"curvature={float(row['mean_curvature']):.6f}",
            flush=True,
        )


def main() -> None:
    start_ts = time.perf_counter()
    last_ts = start_ts
    args = parse_args()
    repo_root = repo_root_from_script()
    config_path = Path(args.config).resolve()
    outdir = Path(args.outdir).resolve()
    cfg = load_yaml(config_path)
    dataset_cfg = load_yaml(repo_root / "data" / str(cfg["dataset"]) / f"{cfg['dataset']}.yaml")
    alpha_values = expand_alpha_grid(cfg["alpha_grid"])
    last_ts = log_milestone("config loaded", start_ts, last_ts)

    K = prepare_imports(repo_root, str(cfg["dataset"]))
    last_ts = log_milestone("KCOR helpers imported", start_ts, last_ts)
    wave_table, cohort_diag = build_real_cohort_table(repo_root, cfg, K)
    last_ts = log_milestone("real cohort-wave table built", start_ts, last_ts)
    curves, best_df = evaluate_real_data(wave_table, cfg, alpha_values)
    last_ts = log_milestone("alpha objective sweeps completed", start_ts, last_ts)
    neutralization_modes = get_nph_neutralization_modes(cfg)
    primary_mode = get_primary_nph_neutralization_mode(cfg)
    primary_frames = {mode: build_primary_subset(wave_table, cfg, mode) for mode in neutralization_modes}
    contract_check = validate_mode_contract(primary_frames, cfg)
    invariance_check = validate_alpha_one_invariance(curves, primary_frames, cfg)
    last_ts = log_milestone("primary subsets and contract checks built", start_ts, last_ts)
    theta_scale_summary = build_theta_scale_summary(best_df, cfg)
    last_ts = log_milestone("theta-scale summary built", start_ts, last_ts)
    mode_results: dict[str, dict] = {}
    for mode in neutralization_modes:
        mode_primary_df = primary_frames[mode]
        mode_loo_df = leave_one_out_analysis(mode_primary_df, cfg, alpha_values, mode)
        mode_bootstrap_df = bootstrap_alpha(mode_primary_df, cfg, alpha_values, mode)
        mode_bootstrap_summary = build_bootstrap_summary(mode_bootstrap_df, cfg)
        mode_loo_summary = build_leave_one_out_summary(mode_loo_df, best_df, cfg)
        mode_run_artifact = build_alpha_run_artifact(
            cfg,
            dataset_cfg,
            alpha_values,
            cohort_diag,
            best_df,
            mode_bootstrap_df,
            mode_bootstrap_summary,
            mode_loo_summary,
            mode,
        )
        mode_results[mode] = {
            "primary_df": mode_primary_df,
            "loo_df": mode_loo_df,
            "bootstrap_df": mode_bootstrap_df,
            "bootstrap_summary": mode_bootstrap_summary,
            "loo_summary": mode_loo_summary,
            "run_artifact": mode_run_artifact,
        }
    last_ts = log_milestone("leave-one-out and bootstrap completed", start_ts, last_ts)
    primary_df = mode_results[primary_mode]["primary_df"]
    loo_df = mode_results[primary_mode]["loo_df"]
    bootstrap_df = mode_results[primary_mode]["bootstrap_df"]
    bootstrap_summary = mode_results[primary_mode]["bootstrap_summary"]
    loo_summary = mode_results[primary_mode]["loo_summary"]
    sensitivity_df = build_primary_sensitivity_slices(best_df, cfg)
    synthetic_results = synthetic_recovery(cfg, alpha_values)
    synthetic_df = synthetic_results["legacy_df"]
    synthetic_ve_recovery_df = synthetic_results["ve_recovery_df"]
    synthetic_ve_summary_df = synthetic_results["ve_summary_df"]
    synthetic_ve_report = synthetic_results["ve_report"]
    conditional_ve_results = conditional_ve_alpha_identification(cfg, alpha_values)
    conditional_ve_estimates_df = conditional_ve_results["estimates_df"]
    conditional_ve_summary_df = conditional_ve_results["summary_df"]
    conditional_ve_report = conditional_ve_results["report"]
    wave_ident_results = run_wave_identifiability_experiment(cfg, alpha_values)
    wave_identifiability_df = wave_ident_results["df"]
    wave_identifiability_report = str(wave_ident_results["report"])
    last_ts = log_milestone("synthetic recovery completed", start_ts, last_ts)
    run_artifact = mode_results[primary_mode]["run_artifact"]
    decision_summary = build_decision_summary(best_df, bootstrap_summary, loo_summary, run_artifact, cfg)
    calibration_choice_summary = build_calibration_choice_summary(run_artifact)
    identifiability_report = render_identifiability_report(
        run_artifact,
        bootstrap_summary,
        loo_summary,
        sensitivity_df,
        decision_summary,
    )
    comparison_rows = [
        summarize_neutralization_mode(
            mode,
            cfg,
            best_df,
            mode_results[mode]["bootstrap_summary"],
            mode_results[mode]["loo_summary"],
            mode_results[mode]["run_artifact"],
            mode_results[mode]["primary_df"],
        )
        for mode in neutralization_modes
    ]
    neutralization_comparison_df = pd.DataFrame(comparison_rows).sort_values("mode").reset_index(drop=True)
    recommendation, recommendation_detail = determine_neutralization_recommendation(neutralization_comparison_df)
    neutralization_comparison_df["recommendation"] = recommendation
    neutralization_comparison_report = render_neutralization_comparison_report(
        neutralization_comparison_df,
        recommendation,
        recommendation_detail,
        contract_check,
        invariance_check,
    )
    comparison_output_files = [
        "alpha_neutralization_mode_comparison.csv",
        "alpha_neutralization_comparison_report.md",
        "alpha_neutralization_run_artifact.json",
        "fig_alpha_neutralization_mode_comparison.png",
    ]
    neutralization_comparison_artifact = {
        "dataset": cfg["dataset"],
        "config": {
            "analysis": cfg["analysis"],
            "alpha_grid": cfg["alpha_grid"],
            "neutralization_modes": neutralization_modes,
            "primary_nph_neutralization_mode": primary_mode,
            "write_manuscript_figures": should_write_manuscript_figures(cfg),
            "excess_definition": "h_excess,d(t) = h_d(t) - h_ref(t)",
        },
        "cohort_selection": run_artifact["cohort_selection"],
        "seeds": run_artifact["seeds"],
        "alpha_grid": [float(x) for x in alpha_values.tolist()],
        "contract_check": contract_check,
        "alpha_one_invariance": invariance_check,
        "neutralization_modes": [
            {
                "mode": mode,
                "summary": json.loads(
                    neutralization_comparison_df[neutralization_comparison_df["mode"] == mode].to_json(orient="records")
                )[0],
                "primary_identification": mode_results[mode]["run_artifact"]["primary_identification"],
            }
            for mode in neutralization_modes
        ],
        "recommendation": {
            "decision": recommendation,
            "detail": recommendation_detail,
        },
        "output_file_set": comparison_output_files,
    }

    write_outputs(
        outdir,
        repo_root,
        cfg,
        cohort_diag,
        wave_table,
        curves,
        best_df,
        theta_scale_summary,
        bootstrap_summary,
        loo_summary,
        sensitivity_df,
        decision_summary,
        calibration_choice_summary,
        loo_df,
        bootstrap_df,
        synthetic_df,
        synthetic_ve_recovery_df,
        synthetic_ve_summary_df,
        synthetic_ve_report,
        conditional_ve_estimates_df,
        conditional_ve_summary_df,
        conditional_ve_report,
        run_artifact,
        identifiability_report,
        neutralization_comparison_df,
        neutralization_comparison_report,
        neutralization_comparison_artifact,
        wave_identifiability_df,
        wave_identifiability_report,
    )
    last_ts = log_milestone("outputs and figures written", start_ts, last_ts)

    print(f"Wrote outputs to {outdir}")
    primary_status = run_artifact["primary_identification"]["status"]
    if primary_status == "identified":
        pair_alpha = run_artifact["primary_identification"]["pairwise_alpha_hat_reported"]
        collapse_alpha = run_artifact["primary_identification"]["collapse_alpha_hat_reported"]
        print(
            f"Primary alpha identified: pairwise={pair_alpha:.4f}, collapse={collapse_alpha:.4f}",
            flush=True,
        )
    else:
        reasons = ", ".join(run_artifact["primary_identification"]["failure_reasons"]) or "diagnostics_failed"
        print(f"[ALPHA] Primary alpha not identified ({reasons})", flush=True)
        calibration = run_artifact["calibration_choice"]
        if calibration["status"] == "external_calibration":
            print(
                f"[ALPHA] Using external default alpha = {float(calibration['alpha_value']):.4f} "
                f"(source={calibration['source']})",
                flush=True,
            )
    print("NPH NEUTRALIZATION COMPARISON", flush=True)
    for _, row in neutralization_comparison_df.iterrows():
        print(
            f"{row['mode']}: status={row['final_status']}, "
            f"pair={row['pairwise_alpha_raw']:.4f}, coll={row['collapse_alpha_raw']:.4f}, "
            f"boundary={row['bootstrap_boundary_fraction']:.3f}",
            flush=True,
        )
    print(f"recommendation={recommendation}", flush=True)
    print_synthetic_vaccine_effect_summary(synthetic_ve_summary_df)
    print_conditional_ve_summary(conditional_ve_summary_df)
    print_wave_identifiability_summary(wave_identifiability_df)
    log_milestone("alpha run finished", start_ts, last_ts)


if __name__ == "__main__":
    main()
