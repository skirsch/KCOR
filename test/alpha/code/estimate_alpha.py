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
MANUSCRIPT_ALPHA_PAIR = 1.19
MANUSCRIPT_ALPHA_COLLAPSE = 1.18


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
) -> list[dict]:
    floor = build_floor(subset["excess"])
    grouped_weeks: list[dict[str, np.ndarray]] = []
    for _, group in subset.groupby("iso_int"):
        transformed, valid = transform_excess(group["excess"].to_numpy(dtype=float), excess_mode, floor)
        grouped_weeks.append(
            {
                "transformed": transformed,
                "valid": valid,
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
            transformed = group["transformed"]
            valid = group["valid"]
            theta_vals = group["theta_vals"]
            weights = group["weights"]
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
            factors = np.maximum(gamma_moment_alpha(theta_vals, alpha), floor)
            a_hat = group["excess"] / factors
            transformed, valid = transform_excess(a_hat, excess_mode, floor)
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


def build_theta_scale_summary(best_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    params = cfg["analysis"]
    summary = best_df[
        (best_df["anchor_mode"] == params["primary_anchor"])
        & (best_df["excess_mode"] == params["primary_excess_mode"])
        & (best_df["age_band"] == "pooled")
        & (best_df["time_segment"] == "pooled")
    ].copy()
    if summary.empty:
        return summary
    summary = summary.sort_values(["estimator", "theta_scale"]).reset_index(drop=True)
    return summary[
        [
            "theta_scale",
            "estimator",
            "anchor_mode",
            "excess_mode",
            "alpha_hat",
            "objective",
            "n_pairs",
            "n_weeks_used",
            "n_cohorts",
        ]
    ]


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


def simulate_synthetic_table(cfg: dict, alpha_true: float, seed: int, noise_model: str = "lognormal_fixed") -> pd.DataFrame:
    synth_cfg = cfg["synthetic"]
    rng = np.random.default_rng(seed)
    n_cohorts = int(synth_cfg["n_cohorts"])
    n_weeks = int(synth_cfg["n_weeks"])
    theta_values = np.asarray(synth_cfg["theta_w_values"], dtype=float)
    peak = float(synth_cfg["wave_amplitude_peak"])
    baseline_weight = float(synth_cfg["baseline_weight"])
    lognormal_sigma = float(synth_cfg.get("lognormal_sigma", 0.03))
    heteroskedastic_scale = float(synth_cfg.get("heteroskedastic_scale", 0.35))

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
            expected_excess = wave_amplitude[week_idx] * factor
            deaths_proxy = max(expected_excess * baseline_weight, 1e-6)
            if noise_model == "lognormal_fixed":
                sigma = lognormal_sigma
                noise = float(np.exp(rng.normal(0.0, sigma)))
                excess = expected_excess * noise
            elif noise_model == "heteroskedastic_lognormal":
                sigma = heteroskedastic_scale / math.sqrt(deaths_proxy)
                sigma = float(np.clip(sigma, 0.02, 0.35))
                noise = float(np.exp(rng.normal(0.0, sigma)))
                excess = expected_excess * noise
            else:
                raise ValueError(f"Unknown synthetic noise_model: {noise_model}")
            rows.append(
                {
                    "cohort_id": cohort_id,
                    "iso_int": week_idx,
                    "time_segment": "early_wave" if week_idx < (n_weeks // 2) else "late_wave",
                    "yob_decade": yob_decade,
                    "Dead": baseline_weight * max(excess, 0.001),
                    "excess": excess,
                    "noise_model": noise_model,
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
    noise_models = list(synth_cfg.get("noise_models", ["lognormal_fixed"]))
    for noise_model in noise_models:
        for alpha_true in synth_cfg["alpha_true_values"]:
            print(
                f"[ALPHA] synthetic recovery for alpha_true={float(alpha_true):.3f} "
                f"(noise_model={noise_model})",
                flush=True,
            )
            for rep in range(reps):
                table = simulate_synthetic_table(
                    cfg,
                    float(alpha_true),
                    base_seed + rep + int(float(alpha_true) * 1000),
                    noise_model=noise_model,
                )
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
                        "noise_model": noise_model,
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
    anchor_mode: str,
    theta_scale: str,
    excess_mode: str,
    age_band: str,
    time_segment: str,
    estimator: str,
) -> pd.Series:
    subset = best_df[
        (best_df["anchor_mode"] == anchor_mode)
        & (best_df["theta_scale"] == theta_scale)
        & (best_df["excess_mode"] == excess_mode)
        & (best_df["age_band"] == age_band)
        & (best_df["time_segment"] == time_segment)
        & (best_df["estimator"] == estimator)
    ]
    if len(subset) != 1:
        raise ValueError(
            "Expected exactly one alpha_best_estimates row for "
            f"anchor={anchor_mode}, theta_scale={theta_scale}, excess_mode={excess_mode}, "
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


def plot_manuscript_czech_objective(curves: pd.DataFrame, best_df: pd.DataFrame, out_path: Path) -> None:
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
        raise ValueError("Primary objective figure rows not found for pooled Czech specification")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pair_row = _filter_best_unique(
        best_df,
        anchor_mode="dose0",
        theta_scale="gamma_primary",
        excess_mode="exclude_nonpositive",
        age_band="pooled",
        time_segment="pooled",
        estimator="pairwise",
    )
    collapse_row = _filter_best_unique(
        best_df,
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
) -> None:
    plt = _import_matplotlib()
    if plt is None:
        return
    if loo_df.empty:
        raise ValueError("Leave-one-out diagnostics are empty")
    if bootstrap_df.empty:
        raise ValueError("Bootstrap diagnostics are empty")
    pair_row = _filter_best_unique(
        best_df,
        anchor_mode="dose0",
        theta_scale="gamma_primary",
        excess_mode="exclude_nonpositive",
        age_band="pooled",
        time_segment="pooled",
        estimator="pairwise",
    )
    pooled_pair = float(pair_row["alpha_hat"])
    seg = best_df[
        (best_df["anchor_mode"] == "dose0")
        & (best_df["theta_scale"] == "gamma_primary")
        & (best_df["excess_mode"] == "exclude_nonpositive")
        & (best_df["age_band"] == "pooled")
        & (best_df["time_segment"].isin(["pooled", "early_wave", "late_wave"]))
        & (best_df["estimator"].isin(["pairwise", "collapse"]))
    ].copy()
    if len(seg) != 6:
        raise ValueError(f"Expected 6 segmented diagnostic rows, got {len(seg)}")

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
    for estimator, color, marker, offset in [("pairwise", "tab:blue", "o", -0.08), ("collapse", "tab:orange", "s", 0.08)]:
        part = seg[seg["estimator"] == estimator].sort_values("time_segment")
        ax.plot(positions + offset, part["alpha_hat"], color=color, ls="-" if estimator == "pairwise" else "--", marker=marker, lw=2, label=estimator.capitalize())
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


def generate_manuscript_figures(outdir: Path, repo_root: Path) -> None:
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
    plot_manuscript_czech_objective(curves_df, best_df, manuscript_dir / "fig_alpha_czech_objective.png")
    plot_manuscript_czech_diagnostics(loo_df, bootstrap_df, best_df, manuscript_dir / "fig_alpha_czech_diagnostics.png")


def write_outputs(
    outdir: Path,
    repo_root: Path,
    cohort_diag: pd.DataFrame,
    wave_table: pd.DataFrame,
    curves: pd.DataFrame,
    best_df: pd.DataFrame,
    theta_scale_summary: pd.DataFrame,
    loo_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    cohort_diag.to_csv(outdir / "alpha_cohort_diagnostics.csv", index=False)
    wave_table.to_csv(outdir / "alpha_wave_table.csv", index=False)
    curves.to_csv(outdir / "alpha_objective_curves.csv", index=False)
    best_df.to_csv(outdir / "alpha_best_estimates.csv", index=False)
    theta_scale_summary.to_csv(outdir / "alpha_theta_scale_summary.csv", index=False)
    loo_df.to_csv(outdir / "alpha_leave_one_out.csv", index=False)
    bootstrap_df.to_csv(outdir / "alpha_bootstrap.csv", index=False)
    synthetic_df.to_csv(outdir / "alpha_synthetic_recovery.csv", index=False)
    plot_objectives(curves, best_df, outdir / "fig_alpha_objectives.png")
    plot_synthetic_recovery(synthetic_df, outdir / "fig_alpha_synthetic_recovery.png")
    generate_manuscript_figures(outdir, repo_root)


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
    theta_scale_summary = build_theta_scale_summary(best_df, cfg)
    last_ts = log_milestone("theta-scale summary built", start_ts, last_ts)
    loo_df = leave_one_out_analysis(primary_df, cfg, alpha_values)
    last_ts = log_milestone("leave-one-out influence completed", start_ts, last_ts)
    bootstrap_df = bootstrap_alpha(primary_df, cfg, alpha_values)
    last_ts = log_milestone("bootstrap completed", start_ts, last_ts)
    synthetic_df = synthetic_recovery(cfg, alpha_values)
    last_ts = log_milestone("synthetic recovery completed", start_ts, last_ts)

    write_outputs(outdir, repo_root, cohort_diag, wave_table, curves, best_df, theta_scale_summary, loo_df, bootstrap_df, synthetic_df)
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
