#
#   GLM.py
#
# Runs a GLM on the variable-cohort tallies from KCOR_variable.py.
#
# Usage:
#   python GLM.py --input <input.xlsx> --sheet <sheet_name> --out-png <output.png> --out-xlsx <output.xlsx> --start-week <start_week> --end-week <end_week> --baseline-week <baseline_week> --plot <plot_type> --x-tick-step <x_tick_step> --grid-x-step <grid_x_step>
#
# this is a python remake of henjin_glm.R from the validation/GLM folder.
#
# This requires as input the variable-cohort tallies from KCOR_variable.py as noted in the KCOR_variable.py script and Makefile here.

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def iso_week_to_monday(iso_week: str) -> pd.Timestamp:
    return pd.to_datetime(f"{iso_week}-1", format="%G-%V-%u", errors="coerce")


def prepare_data(
    excel_path: str,
    sheet: str = "KCOR_variable",
    start_week: Optional[str] = None,
    end_week: Optional[str] = None,
    baseline_week: str = "2021-34",
) -> Tuple[pd.DataFrame, str, list]:
    """
    Load variable-cohort tallies and prepare cumulative exposure and deaths by (dose01, born, week).

    - Aggregates over Sex
    - Collapses doses >=1 to a single vaccinated group (dose01 = 1)
    - Restricts weeks to [start_week, end_week] if provided
    - Computes cumulative sums within (dose01, born) across calendar time
    - Adds a synthetic "base" week copied from baseline_week for reference-level coding
    """
    df = pd.read_excel(excel_path, sheet_name=sheet)

    required_cols = {"ISOweekDied", "YearOfBirth", "Sex", "Dose", "Alive", "Dead"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {excel_path}: {sorted(missing)}")

    df = df.rename(
        columns={
            "ISOweekDied": "week",
            "YearOfBirth": "born",
            "Dose": "dose",
            "Alive": "alive",
            "Dead": "dead",
        }
    )

    # Keep valid birth years
    df = df[pd.to_numeric(df["born"], errors="coerce").notna()].copy()
    df["born"] = df["born"].astype(int)

    # Binary vaccinated indicator: current dose >= 1
    df["dose01"] = (pd.to_numeric(df["dose"], errors="coerce").fillna(0).astype(int) >= 1).astype(int)

    # Aggregate over sex and original dose levels
    g = (
        df.groupby(["week", "born", "dose01"], as_index=False)[["alive", "dead"]]
        .sum()
        .reset_index(drop=True)
    )

    # Sort weeks by calendar order
    g["week_date"] = g["week"].map(iso_week_to_monday)
    g = g[g["week_date"].notna()].copy()

    # Restrict analysis window
    if start_week is not None:
        g = g[g["week_date"] >= iso_week_to_monday(start_week)]
    if end_week is not None:
        g = g[g["week_date"] <= iso_week_to_monday(end_week)]

    # Within (dose01, born), compute cumulative deaths and cumulative exposure (person-weeks)
    g = g.sort_values(["dose01", "born", "week_date"])  # stable ordering
    g["dead_cum"] = g.groupby(["dose01", "born"], sort=False)["dead"].cumsum()
    g["exp_cum"] = g.groupby(["dose01", "born"], sort=False)["alive"].cumsum()

    # Add baseline pseudo-week copied from the specified baseline_week
    if baseline_week not in set(g["week"]):
        # If requested baseline missing, fallback to earliest observed week
        baseline_week = (
            g.sort_values("week_date")["week"].iloc[0]
            if not g.empty
            else "base_missing"
        )

    base_rows = g[g["week"] == baseline_week].copy()
    base_rows = base_rows.assign(
        week="base",
        week_date=iso_week_to_monday(baseline_week),
    )

    g = pd.concat([g, base_rows], ignore_index=True)

    # Encode ordered weeks for plotting
    ordered_weeks = [
        w
        for w in (
            g.sort_values("week_date")["week"].drop_duplicates().tolist()
        )
        if w != "base"
    ]

    return g, baseline_week, ordered_weeks


def fit_glm(g: pd.DataFrame) -> sm.GLM:
    """
    Fit Poisson GLM on cumulative counts with exposure offset:
        dead_cum ~ dose01 * C(week, Treatment(reference='base')) + C(born)
    """
    g = g.copy()
    g["week"] = g["week"].astype(str)
    g["born"] = g["born"].astype(int)
    g["dose01"] = g["dose01"].astype(int)

    model = smf.glm(
        formula="dead_cum ~ dose01 * C(week, Treatment(reference='base')) + C(born)",
        data=g,
        family=sm.families.Poisson(),
        exposure=g["exp_cum"].replace(0, np.nan).fillna(1.0),
    )
    result = model.fit()
    return result


def extract_weekly_effects(result: sm.GLM, ordered_weeks: list) -> pd.DataFrame:
    """
    Extract exp(coef) and 95% CI for the interaction terms dose01:week[w],
    which represent the weekly vaccination effect relative to baseline week.

    Also compute absolute effect (includes main effect of dose01) with proper
    variance propagation.
    """
    params = result.params
    cov = result.cov_params()

    # Try both possible naming conventions from patsy/statsmodels
    def interaction_name(w: str) -> Optional[str]:
        c1 = f"dose01:C(week, Treatment(reference='base'))[T.{w}]"
        c2 = f"C(week, Treatment(reference='base'))[T.{w}]:dose01"
        if c1 in params:
            return c1
        if c2 in params:
            return c2
        return None

    dose_main = "dose01"

    rows = []
    for w in ordered_weeks:
        name = interaction_name(w)
        if name is None:
            continue

        b_int = params[name]
        se_int = np.sqrt(cov.loc[name, name])

        # Relative to baseline (interaction only)
        rr_rel = np.exp(b_int)
        rr_rel_lo = np.exp(b_int - 1.96 * se_int)
        rr_rel_hi = np.exp(b_int + 1.96 * se_int)

        # Absolute vaccinated vs unvaccinated effect for week w
        if dose_main in params:
            b_abs = params[dose_main] + b_int
            var_abs = cov.loc[dose_main, dose_main] + cov.loc[name, name] + 2.0 * cov.loc[dose_main, name]
            se_abs = np.sqrt(max(var_abs, 0.0))
            rr_abs = np.exp(b_abs)
            rr_abs_lo = np.exp(b_abs - 1.96 * se_abs)
            rr_abs_hi = np.exp(b_abs + 1.96 * se_abs)
        else:
            rr_abs = rr_abs_lo = rr_abs_hi = np.nan

        rows.append(
            {
                "week": w,
                "rr_relative": rr_rel,
                "rr_relative_lo": rr_rel_lo,
                "rr_relative_hi": rr_rel_hi,
                "rr": rr_abs,
                "rr_lo": rr_abs_lo,
                "rr_hi": rr_abs_hi,
            }
        )

    out = pd.DataFrame(rows).sort_values("week").reset_index(drop=True)
    return out


def plot_effects(
    effects: pd.DataFrame,
    out_png: str,
    metric: str = "relative",
    x_tick_step: int = 4,
    grid_x_step: int = 0,
):
    plt.figure(figsize=(12, 5))
    x = np.arange(len(effects))
    if metric == "absolute":
        y = effects["rr"].values
        lo = effects["rr_lo"].values
        hi = effects["rr_hi"].values
        title = "GLM weekly RR (vaccinated vs unvaccinated)"
        ylabel = "Relative risk"
    else:
        y = effects["rr_relative"].values
        lo = effects["rr_relative_lo"].values
        hi = effects["rr_relative_hi"].values
        title = "GLM weekly RR relative to baseline week"
        ylabel = "Relative risk (vs baseline)"

    plt.plot(x, y, label="RR vaccinated vs unvaccinated")
    plt.fill_between(x, lo, hi, color="C0", alpha=0.2, label="95% CI")
    plt.axhline(1.0, color="black", lw=1, ls=":")
    # Thin x-axis tick labels to avoid crowding
    step = max(int(x_tick_step), 1)
    ticks = x[::step]
    labels = effects["week"].iloc[::step].tolist()
    plt.xticks(ticks, labels, rotation=90)
    plt.margins(x=0.01)
    plt.tick_params(axis='x', labelsize=8)

    # Vertical grid lines: draw in data coordinates so they span full height
    gx = step if grid_x_step in (None, 0) else max(int(grid_x_step), 1)
    if gx:
        ax = plt.gca()
        grid_ticks = x[::gx]
        y0, y1 = ax.get_ylim()
        ax.vlines(grid_ticks, y0, y1, colors="#e0e0e0", linewidth=0.8, zorder=1, alpha=0.4)
    plt.ylabel(ylabel)
    plt.xlabel("ISO week")
    plt.title(title)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Henjin-style GLM on variable cohorts (Czech)")
    parser.add_argument("--input", default="../../data/Czech/KCOR_variable.xlsx", help="Path to KCOR_variable.xlsx")
    parser.add_argument("--sheet", default="KCOR_variable", help="Excel sheet name")
    parser.add_argument("--out-png", default="out/GLM_plot_Czech_data.png", help="Output PNG path")
    parser.add_argument("--out-xlsx", default="out/GLM_output.xlsx", help="Output Excel path for effects table")
    parser.add_argument("--start-week", default="2021-24", help="Start ISO week (inclusive)")
    parser.add_argument("--end-week", default="2024-26", help="End ISO week (inclusive)")
    parser.add_argument("--baseline-week", default="2021-34", help="Baseline ISO week used as reference level")
    parser.add_argument("--plot", choices=["relative", "absolute"], default="relative", help="Plot interaction-only RR (relative) or absolute RR")
    parser.add_argument("--x-tick-step", type=int, default=4, help="Show every Nth week label on x-axis")
    parser.add_argument("--grid-x-step", type=int, default=0, help="Vertical grid every N weeks (0 uses x-tick-step; set <=0 to disable)")
    args = parser.parse_args()

    g, chosen_baseline, ordered_weeks = prepare_data(
        excel_path=args.input,
        sheet=args.sheet,
        start_week=args.start_week,
        end_week=args.end_week,
        baseline_week=args.baseline_week,
    )

    result = fit_glm(g)
    effects = extract_weekly_effects(result, ordered_weeks)

    # Save outputs
    os.makedirs(os.path.dirname(args.out_xlsx), exist_ok=True)
    with pd.ExcelWriter(args.out_xlsx) as writer:
        effects.to_excel(writer, sheet_name="effects", index=False)

    plot_effects(
        effects,
        args.out_png,
        metric=args.plot,
        x_tick_step=args.__dict__["x_tick_step"],
        grid_x_step=args.__dict__["grid_x_step"],
    )

    # Convenience printout
    print("Baseline week:", chosen_baseline)
    print(effects.head())


if __name__ == "__main__":
    main()


