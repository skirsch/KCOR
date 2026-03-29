from __future__ import annotations

import argparse
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


def gamma_moment_alpha(theta: np.ndarray | float, alpha: float) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    out = np.ones_like(theta_arr, dtype=float)
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
    covid_cfg = dataset_cfg.get("covidCorrection") or {}
    start_label = str(covid_cfg["startDate"]).replace("_", "-")
    end_label = str(covid_cfg["endDate"]).replace("_", "-")
    return WaveWindow(
        start_label=start_label,
        end_label=end_label,
        start_monday=iso_label_to_monday(start_label),
        end_monday=iso_label_to_monday(end_label),
    )


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

            for idx, row in cohort_df.iterrows():
                if idx not in set(wave_idx.tolist()):
                    continue
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
                        "iso_year": int(row["iso_year"]),
                        "iso_week": int(row["iso_week"]),
                        "iso_int": int(row["iso_int"]),
                        "week_monday": row["week_monday"],
                        "t_week": int(row["t_week"]),
                        "Dead": float(row["Dead"]),
                        "Alive": float(row["Alive"]),
                        "hazard_obs": float(row["hazard_obs"]),
                        "hazard_eff": float(row["hazard_eff"]),
                        "H_obs": float(row["H_obs"]),
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
) -> list[dict]:
    floor = build_floor(subset["excess"])
    records: list[dict] = []
    for alpha in alpha_values:
        obj = 0.0
        n_pairs = 0
        n_weeks_used = 0
        for _, group in subset.groupby("iso_int"):
            transformed, valid = transform_excess(group["excess"].to_numpy(dtype=float), excess_mode, floor)
            theta_vals = group[theta_column].to_numpy(dtype=float)
            weights = group["Dead"].to_numpy(dtype=float)
            log_factor = np.log(np.maximum(gamma_moment_alpha(theta_vals, alpha), floor))
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
) -> list[dict]:
    floor = build_floor(subset["excess"])
    records: list[dict] = []
    for alpha in alpha_values:
        obj = 0.0
        n_points = 0
        n_weeks_used = 0
        for _, group in subset.groupby("iso_int"):
            theta_vals = group[theta_column].to_numpy(dtype=float)
            factors = np.maximum(gamma_moment_alpha(theta_vals, alpha), floor)
            a_hat = group["excess"].to_numpy(dtype=float) / factors
            transformed, valid = transform_excess(a_hat, excess_mode, floor)
            weights = collapse_weight(group["Dead"].to_numpy(dtype=float), weight_mode)
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


def summarize_best_curve(curve_df: pd.DataFrame) -> dict | None:
    curve_df = curve_df[np.isfinite(curve_df["objective"])].copy()
    if curve_df.empty:
        return None
    best = curve_df.sort_values(["objective", "alpha"]).iloc[0]
    return {
        "alpha_hat": float(best["alpha"]),
        "objective": float(best["objective"]),
        "n_pairs": int(best["n_pairs"]),
        "n_weeks_used": int(best["n_weeks_used"]),
    }


def evaluate_real_data(
    wave_table: pd.DataFrame,
    cfg: dict,
    alpha_values: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = cfg["analysis"]
    time_map = build_time_segment_map(wave_table)
    theta_scales = {
        "gamma_primary": "theta_t_gamma",
        "raw_observed": "theta_t_raw",
    }

    all_curves: list[dict] = []
    best_rows: list[dict] = []

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
                                )
                            else:
                                curve = evaluate_collapse_objective(
                                    segment_df,
                                    alpha_values,
                                    excess_mode,
                                    params["weight_mode_collapse"],
                                    theta_column,
                                    int(params["min_cohorts_per_week"]),
                                )
                            curve_df = pd.DataFrame(curve)
                            if curve_df.empty:
                                continue
                            curve_df["anchor_mode"] = anchor
                            curve_df["theta_scale"] = theta_scale
                            curve_df["excess_mode"] = excess_mode
                            curve_df["age_band"] = age_band
                            curve_df["time_segment"] = time_segment
                            curve_df["estimator"] = estimator
                            all_curves.extend(curve_df.to_dict(orient="records"))
                            best = summarize_best_curve(curve_df)
                            if best is not None:
                                best.update(
                                    {
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


def build_primary_subset(wave_table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    params = cfg["analysis"]
    df = add_reference_excess(wave_table, params["primary_anchor"])
    df = df[df["eligible"] == 1].copy()
    df = df[np.isfinite(df["href"]) & np.isfinite(df["theta_t_gamma"])].copy()
    return df


def leave_one_out_analysis(primary_df: pd.DataFrame, cfg: dict, alpha_values: np.ndarray) -> pd.DataFrame:
    params = cfg["analysis"]
    rows = []
    eligible_ids = sorted(primary_df["cohort_id"].dropna().unique())
    for cohort_id in eligible_ids:
        subset = primary_df[primary_df["cohort_id"] != cohort_id].copy()
        curve = pd.DataFrame(
            evaluate_pairwise_objective(
                subset,
                alpha_values,
                params["primary_excess_mode"],
                params["weight_mode_pairwise"],
                "theta_t_gamma",
                int(params["min_pairs_per_alpha"]),
            )
        )
        best = summarize_best_curve(curve)
        rows.append(
            {
                "left_out_cohort": cohort_id,
                "alpha_hat": np.nan if best is None else best["alpha_hat"],
                "objective": np.nan if best is None else best["objective"],
                "n_pairs": 0 if best is None else best["n_pairs"],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_alpha(primary_df: pd.DataFrame, cfg: dict, alpha_values: np.ndarray) -> pd.DataFrame:
    params = cfg["analysis"]
    boot_cfg = cfg["bootstrap"]
    estimator = str(boot_cfg["summary_estimator"])
    rng = np.random.default_rng(int(boot_cfg["seed"]))
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
        if estimator == "pairwise":
            curve = pd.DataFrame(
                evaluate_pairwise_objective(
                    subset,
                    alpha_values,
                    params["primary_excess_mode"],
                    params["weight_mode_pairwise"],
                    "theta_t_gamma",
                    int(params["min_pairs_per_alpha"]),
                )
            )
        else:
            curve = pd.DataFrame(
                evaluate_collapse_objective(
                    subset,
                    alpha_values,
                    params["primary_excess_mode"],
                    params["weight_mode_collapse"],
                    "theta_t_gamma",
                    int(params["min_cohorts_per_week"]),
                )
            )
        best = summarize_best_curve(curve)
        rows.append(
            {
                "bootstrap_rep": rep,
                "alpha_hat": np.nan if best is None else best["alpha_hat"],
                "objective": np.nan if best is None else best["objective"],
            }
        )
    return pd.DataFrame(rows)


def simulate_synthetic_table(cfg: dict, alpha_true: float, seed: int) -> pd.DataFrame:
    synth_cfg = cfg["synthetic"]
    rng = np.random.default_rng(seed)
    n_cohorts = int(synth_cfg["n_cohorts"])
    n_weeks = int(synth_cfg["n_weeks"])
    theta_values = np.asarray(synth_cfg["theta_w_values"], dtype=float)
    peak = float(synth_cfg["wave_amplitude_peak"])
    baseline_weight = float(synth_cfg["baseline_weight"])

    wave_phase = np.linspace(0.1, 0.95, n_weeks)
    wave_amplitude = peak * np.exp(-((wave_phase - 0.45) ** 2) / 0.03)
    delta_h_path = np.cumsum(0.5 * wave_amplitude + 0.002)

    rows = []
    for cohort_idx in range(n_cohorts):
        theta_w = float(theta_values[cohort_idx % len(theta_values)])
        cohort_id = f"synthetic_{cohort_idx:02d}"
        yob_decade = [1930, 1940, 1950][cohort_idx % 3]
        for week_idx in range(n_weeks):
            theta_t = float(propagate_theta(theta_w, delta_h_path[week_idx] - delta_h_path[0]))
            factor = float(gamma_moment_alpha(theta_t, alpha_true))
            noise = float(np.exp(rng.normal(0.0, 0.03)))
            excess = wave_amplitude[week_idx] * factor * noise
            rows.append(
                {
                    "cohort_id": cohort_id,
                    "iso_int": week_idx,
                    "time_segment": "early_wave" if week_idx < (n_weeks // 2) else "late_wave",
                    "yob_decade": yob_decade,
                    "Dead": baseline_weight * max(excess, 0.001),
                    "excess": excess,
                    "theta_t_gamma": theta_t,
                    "theta_t_raw": theta_t,
                    "eligible": 1,
                }
            )
    return pd.DataFrame(rows)


def synthetic_recovery(cfg: dict, alpha_values: np.ndarray) -> pd.DataFrame:
    synth_cfg = cfg["synthetic"]
    if not bool(synth_cfg.get("enabled", True)):
        return pd.DataFrame()
    base_seed = int(synth_cfg["seed"])
    reps = int(synth_cfg.get("reps", 8))
    rows = []
    for alpha_true in synth_cfg["alpha_true_values"]:
        print(f"[ALPHA] synthetic recovery for alpha_true={float(alpha_true):.3f}", flush=True)
        for rep in range(reps):
            table = simulate_synthetic_table(cfg, float(alpha_true), base_seed + rep + int(float(alpha_true) * 1000))
            pair_curve = pd.DataFrame(
                evaluate_pairwise_objective(
                    table,
                    alpha_values,
                    "exclude_nonpositive",
                    "deaths_min",
                    "theta_t_gamma",
                    8,
                )
            )
            collapse_curve = pd.DataFrame(
                evaluate_collapse_objective(
                    table,
                    alpha_values,
                    "exclude_nonpositive",
                    "deaths",
                    "theta_t_gamma",
                    3,
                )
            )
            pair_best = summarize_best_curve(pair_curve)
            collapse_best = summarize_best_curve(collapse_curve)
            rows.append(
                {
                    "alpha_true": float(alpha_true),
                    "rep": rep,
                    "pairwise_alpha_hat": np.nan if pair_best is None else pair_best["alpha_hat"],
                    "collapse_alpha_hat": np.nan if collapse_best is None else collapse_best["alpha_hat"],
                }
            )
    return pd.DataFrame(rows)


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
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        return True
    except Exception as exc:
        print(f"[ALPHA] Failed to save plot {out_path.name}: {exc}", flush=True)
        return False


def plot_objectives(curves: pd.DataFrame, best_df: pd.DataFrame, out_path: Path) -> None:
    plt = _import_matplotlib()
    if plt is None:
        return

    primary = curves[
        (curves["anchor_mode"] == "dose0")
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
            (best_df["anchor_mode"] == "dose0")
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


def plot_synthetic_recovery(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        print(f"[ALPHA] Skipping {out_path.name}: synthetic recovery is empty", flush=True)
        return
    plt = _import_matplotlib()
    if plt is None:
        return

    summary = (
        df.groupby("alpha_true", as_index=False)
        .agg(
            pairwise_mean=("pairwise_alpha_hat", "mean"),
            collapse_mean=("collapse_alpha_hat", "mean"),
        )
        .sort_values("alpha_true")
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(summary["alpha_true"], summary["pairwise_mean"], "o-", label="pairwise")
    ax.plot(summary["alpha_true"], summary["collapse_mean"], "s-", label="collapse")
    ax.plot(summary["alpha_true"], summary["alpha_true"], ls="--", color="black", label="truth")
    ax.set_xlabel("alpha_true")
    ax.set_ylabel("recovered alpha")
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, out_path)
    plt.close(fig)


def write_outputs(
    outdir: Path,
    cohort_diag: pd.DataFrame,
    wave_table: pd.DataFrame,
    curves: pd.DataFrame,
    best_df: pd.DataFrame,
    loo_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cohort_diag.to_csv(outdir / "alpha_cohort_diagnostics.csv", index=False)
    wave_table.to_csv(outdir / "alpha_wave_table.csv", index=False)
    curves.to_csv(outdir / "alpha_objective_curves.csv", index=False)
    best_df.to_csv(outdir / "alpha_best_estimates.csv", index=False)
    loo_df.to_csv(outdir / "alpha_leave_one_out.csv", index=False)
    bootstrap_df.to_csv(outdir / "alpha_bootstrap.csv", index=False)
    synthetic_df.to_csv(outdir / "alpha_synthetic_recovery.csv", index=False)
    plot_objectives(curves, best_df, outdir / "fig_alpha_objectives.png")
    plot_synthetic_recovery(synthetic_df, outdir / "fig_alpha_synthetic_recovery.png")


def main() -> None:
    start_ts = time.perf_counter()
    last_ts = start_ts
    args = parse_args()
    repo_root = repo_root_from_script()
    config_path = Path(args.config).resolve()
    outdir = Path(args.outdir).resolve()
    cfg = load_yaml(config_path)
    alpha_values = expand_alpha_grid(cfg["alpha_grid"])
    last_ts = log_milestone("config loaded", start_ts, last_ts)

    K = prepare_imports(repo_root, str(cfg["dataset"]))
    last_ts = log_milestone("KCOR helpers imported", start_ts, last_ts)
    wave_table, cohort_diag = build_real_cohort_table(repo_root, cfg, K)
    last_ts = log_milestone("real cohort-wave table built", start_ts, last_ts)
    curves, best_df = evaluate_real_data(wave_table, cfg, alpha_values)
    last_ts = log_milestone("alpha objective sweeps completed", start_ts, last_ts)
    primary_df = build_primary_subset(wave_table, cfg)
    last_ts = log_milestone("primary subset assembled", start_ts, last_ts)
    loo_df = leave_one_out_analysis(primary_df, cfg, alpha_values)
    last_ts = log_milestone("leave-one-out influence completed", start_ts, last_ts)
    bootstrap_df = bootstrap_alpha(primary_df, cfg, alpha_values)
    last_ts = log_milestone("bootstrap completed", start_ts, last_ts)
    synthetic_df = synthetic_recovery(cfg, alpha_values)
    last_ts = log_milestone("synthetic recovery completed", start_ts, last_ts)

    write_outputs(outdir, cohort_diag, wave_table, curves, best_df, loo_df, bootstrap_df, synthetic_df)
    last_ts = log_milestone("outputs and figures written", start_ts, last_ts)

    print(f"Wrote outputs to {outdir}")
    if not best_df.empty:
        primary = best_df[
            (best_df["anchor_mode"] == cfg["analysis"]["primary_anchor"])
            & (best_df["theta_scale"] == "gamma_primary")
            & (best_df["excess_mode"] == cfg["analysis"]["primary_excess_mode"])
            & (best_df["age_band"] == "pooled")
            & (best_df["time_segment"] == "pooled")
            & (best_df["estimator"] == "pairwise")
        ]
        if not primary.empty:
            print(f"Primary pairwise alpha_hat = {float(primary.iloc[0]['alpha_hat']):.4f}")
    log_milestone("alpha run finished", start_ts, last_ts)


if __name__ == "__main__":
    main()
