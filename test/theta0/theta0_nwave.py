"""
theta0_nwave.py -- Exact theta0 recovery under N epidemic waves
               -- Sweep over theta0 values and baseline mortality rates

Implements the Delta-Iteration estimator:

    1. k = h_obs(0)                          [exact, no fitting]
    2. theta_0 from first quiet window       [initial estimate]
    3. Iterate:
       a. Reconstruct H0_eff from h_obs via frailty inversion
       b. delta_i = H0_eff(wave_end_i) - H_gom(wave_end_i)  [incremental, direct subtraction]
       c. Refit theta on ALL quiet windows with accumulated deltas

Recovers theta0 to <0.1% error regardless of number of waves,
wave sizes, or wave timing -- with no knowledge of wave multipliers.

Usage
-----
    python theta0_nwave.py [--config params_nwave.yaml] [--outdir out/]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

WEEKS_PER_YEAR = 52.1775


# ---------------------------------------------------------------------------
# Range expansion
# ---------------------------------------------------------------------------

def expand_range(val):
    """
    Expand "start:end:step" string into a list of floats.
    If val is already a list, return it unchanged.
    """
    if isinstance(val, list):
        return [float(v) for v in val]
    s = str(val).strip()
    if ':' in s:
        parts = s.split(':')
        if len(parts) != 3:
            raise ValueError(f"Range must be start:end:step, got: {s!r}")
        start, stop, step = float(parts[0]), float(parts[1]), float(parts[2])
        values = []
        v = start
        while v <= stop + step * 1e-9:
            values.append(round(v, 10))
            v += step
        return values
    return [float(s)]



# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_cohort(cfg: dict) -> dict:
    """
    Simulate a fixed cohort under gamma frailty with N epidemic waves.
    cfg must have scalar cohort.annual_death_rate and cohort.theta0_true.
    """
    n0          = cfg['cohort']['n_initial']
    annual_rate = float(cfg['cohort']['annual_death_rate'])
    theta0      = float(cfg['cohort']['theta0_true'])
    gamma_yr    = cfg['gompertz']['gamma_per_year']
    gamma_wk    = gamma_yr / WEEKS_PER_YEAR
    k           = annual_rate / WEEKS_PER_YEAR
    n_weeks     = cfg['simulation']['n_weeks']
    waves       = cfg['epidemic_waves']

    wave_mult = np.ones(n_weeks)
    for w in waves:
        ws   = w['start_week']
        we   = ws + w['duration_weeks']
        for t in range(ws, min(we, n_weeks)):
            wave_mult[t] = w['mortality_multiplier']

    alive    = np.zeros(n_weeks + 1)
    dead     = np.zeros(n_weeks)
    h0_eff   = np.zeros(n_weeks)
    H0_eff   = np.zeros(n_weeks)
    H_gom    = np.zeros(n_weeks)
    h_cohort = np.zeros(n_weeks)

    alive[0]       = n0
    H0_eff_running = 0.0

    for t in range(n_weeks):
        gf           = k * np.exp(gamma_wk * t)
        H_gom[t]     = (k / gamma_wk) * (np.exp(gamma_wk * t) - 1.0)
        h0_eff[t]    = gf * wave_mult[t]
        H0_eff[t]    = H0_eff_running
        h_cohort[t]  = h0_eff[t] / (1.0 + theta0 * H0_eff_running)
        p_death      = 1.0 - np.exp(-h_cohort[t])
        dead[t]      = alive[t] * p_death
        alive[t + 1] = alive[t] - dead[t]
        H0_eff_running += h0_eff[t]

    h_obs = -np.log(1.0 - dead / np.maximum(alive[:n_weeks], 1.0))

    return dict(
        n_weeks     = n_weeks,
        alive       = alive[:n_weeks],
        dead        = dead,
        h_obs       = h_obs,
        h_cohort    = h_cohort,
        H0_eff      = H0_eff,
        H_gom       = H_gom,
        wave_mult   = wave_mult,
        k_true      = k,
        gamma_wk    = gamma_wk,
        theta0_true = theta0,
        waves       = waves,
        annual_rate = annual_rate,
    )


# ---------------------------------------------------------------------------
# Delta-Iteration Estimator
# ---------------------------------------------------------------------------

def build_H_gom_discrete(k: float, gamma_wk: float, n: int) -> np.ndarray:
    """Discrete running cumulative sum of Gompertz baseline hazard."""
    H = np.zeros(n)
    for t in range(1, n):
        H[t] = H[t - 1] + k * np.exp(gamma_wk * (t - 1))
    return H


def reconstruct_H0_eff(h_obs: np.ndarray, theta: float) -> np.ndarray:
    """
    Reconstruct H0_eff from observed h_obs via gamma-frailty inversion:
        h0_eff(t)   = h_obs(t) * (1 + theta * H0_eff(t))
        H0_eff(t+1) = H0_eff(t) + h0_eff(t)
    Passes through ALL weeks (wave and quiet) with no special cases.
    """
    n    = len(h_obs)
    H0   = np.zeros(n)
    H0_t = 0.0
    for t in range(n):
        H0[t]  = H0_t
        h0_t   = h_obs[t] * (1.0 + theta * H0_t)
        H0_t  += h0_t
    return H0


def build_quiet_mask(n: int, waves: list) -> np.ndarray:
    """Boolean mask: True for quiet weeks (outside all wave periods)."""
    mask = np.ones(n, dtype=bool)
    for w in waves:
        ws = w['start_week']
        we = ws + w['duration_weeks']
        mask[ws:min(we, n)] = False
    return mask


def estimate_theta0(sim: dict, cfg: dict, verbose: bool = False) -> dict:
    """
    Delta-Iteration estimator. Recovers theta0 from quiet windows
    under N epidemic waves without knowledge of wave multipliers.
    """
    h_obs    = sim['h_obs']
    gamma_wk = sim['gamma_wk']
    n        = sim['n_weeks']
    waves    = sim['waves']
    tol      = float(cfg['estimation']['convergence_tol'])
    max_iter = int(cfg['estimation']['max_iterations'])
    t        = np.arange(n)

    wave_ends = [w['start_week'] + w['duration_weeks'] for w in waves]
    quiet     = build_quiet_mask(n, waves)

    # Step 1: k = h_obs(0) exactly
    k  = float(h_obs[0])
    Hg = build_H_gom_discrete(k, gamma_wk, n)

    # Step 2: initial theta from pre-wave quiet window only
    first_wave_start = min(w['start_week'] for w in waves)
    t_pre = t[(t < first_wave_start) & quiet]

    def res_pre(params):
        th = params[0]
        return h_obs[t_pre] - k * np.exp(gamma_wk * t_pre) / (1.0 + th * Hg[t_pre])

    r0    = least_squares(res_pre, [1.0], bounds=([0.0], [cfg['estimation']['theta0_max']]))
    theta = float(r0.x[0])

    history = []

    # Step 3: iterate
    for iteration in range(max_iter):

        # 3a: reconstruct H0_eff through all weeks via frailty inversion
        H0_eff_recon = reconstruct_H0_eff(h_obs, theta)

        # 3b: incremental delta per wave
        # H0_eff_recon[we] - Hg[we] = cumulative excess up to wave i
        # incremental delta_i = cumulative_i - cumulative_{i-1}
        deltas          = []
        cumulative_prev = 0.0
        for we in wave_ends:
            idx          = min(we, n - 1)
            cumulative_i = float(H0_eff_recon[idx] - Hg[idx])
            delta_i      = max(cumulative_i - cumulative_prev, 0.0)
            deltas.append(delta_i)
            cumulative_prev = cumulative_i

        # Accumulated Delta(t) = sum of delta_i for all waves ending at or before t
        Delta = np.zeros(n)
        for we, di in zip(wave_ends, deltas):
            Delta[min(we, n - 1):] += di

        # 3c: single-parameter fit on ALL quiet windows with accumulated deltas
        t_quiet = t[quiet]

        def res_all(params):
            th    = params[0]
            H_eff = Hg[t_quiet] + Delta[t_quiet]
            return h_obs[t_quiet] - k * np.exp(gamma_wk * t_quiet) / (1.0 + th * H_eff)

        r         = least_squares(res_all, [theta],
                                  bounds=([0.0], [cfg['estimation']['theta0_max']]))
        theta_new = float(r.x[0])
        err       = 100.0 * (theta_new - sim['theta0_true']) / sim['theta0_true']
        history.append(dict(iter=iteration + 1, theta=theta_new, err=err, deltas=deltas[:]))

        if abs(theta_new - theta) < tol:
            theta = theta_new
            break
        theta = theta_new

    return dict(
        k_hat     = k,
        theta_hat = theta,
        deltas    = deltas,
        Delta     = Delta,
        Hg        = Hg,
        quiet     = quiet,
        history   = history,
    )


# ---------------------------------------------------------------------------
# Plot (single run)
# ---------------------------------------------------------------------------

def plot_results(sim: dict, est: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n         = sim['n_weeks']
    t         = np.arange(n, dtype=float)
    gamma_wk  = sim['gamma_wk']
    theta0    = sim['theta0_true']
    k_true    = sim['k_true']
    waves     = sim['waves']
    h_obs     = sim['h_obs']
    H0_eff    = sim['H0_eff']
    k_hat     = est['k_hat']
    theta_hat = est['theta_hat']
    deltas    = est['deltas']
    Delta     = est['Delta']
    Hg        = est['Hg']
    quiet     = est['quiet']
    history   = est['history']
    err_pct   = 100.0 * (theta_hat - theta0) / theta0

    wave_colors = ['#ff6b6b', '#ffa94d', '#a9e34b', '#74c0fc']

    fig, axes = plt.subplots(4, 1, figsize=(12, 18))
    fig.patch.set_facecolor('#0f0f1a')
    for ax in axes:
        ax.set_facecolor('#161625')
        ax.tick_params(colors='#aaaacc')
        ax.xaxis.label.set_color('#aaaacc')
        ax.yaxis.label.set_color('#aaaacc')
        ax.title.set_color('#e0e0ff')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    def shade_waves(ax):
        for i, w in enumerate(waves):
            ws = w['start_week']
            we = ws + w['duration_weeks']
            ax.axvspan(ws, we, color=wave_colors[i % len(wave_colors)],
                       alpha=0.15, label=f'Wave {i+1} (×{w["mortality_multiplier"]})')

    # (a) h_obs and fitted curves
    ax = axes[0]
    shade_waves(ax)
    ax.scatter(t[quiet],  h_obs[quiet],  color='#74c0fc', s=12, zorder=5,
               label='h_obs quiet', alpha=0.9)
    ax.scatter(t[~quiet], h_obs[~quiet], color='#ff6b6b', s=12, zorder=5,
               marker='x', label='h_obs wave', alpha=0.7)
    h_true = k_true * np.exp(gamma_wk * t) / (1.0 + theta0 * H0_eff)
    ax.plot(t, h_true, color='#a9e34b', lw=1.5, ls='--',
            label=f'true model (θ={theta0})', zorder=4)
    h_fit = k_hat * np.exp(gamma_wk * t) / (1.0 + theta_hat * (Hg + Delta))
    ax.plot(t, h_fit, color='#ffd43b', lw=2,
            label=f'fitted (θ̂={theta_hat:.6f})', zorder=6)
    ax.set_ylabel('Weekly hazard h(t)')
    ax.set_title('(a) Observed hazard and fitted model', pad=10)
    ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='#ccccee', framealpha=0.8)
    ax.grid(True, ls=':', alpha=0.3, color='#444466')

    # (b) H0_eff reconstruction
    ax = axes[1]
    shade_waves(ax)
    H0_eff_recon = reconstruct_H0_eff(h_obs, theta_hat)
    ax.plot(t, H0_eff,       color='#a9e34b', lw=2,   ls='--', label='true H0_eff(t)')
    ax.plot(t, H0_eff_recon, color='#ffd43b', lw=1.5, ls=':',  label='reconstructed H0_eff(t)')
    ax.plot(t, Hg,           color='#74c0fc', lw=1.5,          label='H_gom(t)')
    ax.plot(t, Hg + Delta,   color='#ff6b6b', lw=1.5, ls='--', label='H_gom(t) + Δ(t)')
    for i, (we, di) in enumerate(
            zip([w['start_week'] + w['duration_weeks'] for w in waves], deltas)):
        y_pos = Hg[min(we, n-1)] + sum(deltas[:i+1])
        ax.annotate(f'δ_{i+1}={di:.5f}',
                    xy=(we, y_pos),
                    xytext=(we + 3, y_pos + (Hg[-1] * 0.02)),
                    color='#ffa94d', fontsize=8,
                    arrowprops=dict(arrowstyle='->', color='#ffa94d', lw=0.8))
    ax.set_ylabel('Cumulative hazard H(t)')
    ax.set_title('(b) H0_eff reconstruction and delta computation', pad=10)
    ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='#ccccee', framealpha=0.8)
    ax.grid(True, ls=':', alpha=0.3, color='#444466')

    # (c) convergence
    ax  = axes[2]
    ax2 = ax.twinx()
    iters  = [h['iter']  for h in history]
    thetas = [h['theta'] for h in history]
    errs   = [h['err']   for h in history]
    ax.plot(iters, thetas, 'o-', color='#ffd43b', lw=2, ms=8, label='theta_hat')
    ax.axhline(theta0, color='#a9e34b', lw=1.5, ls='--', label=f'true θ={theta0}')
    ax2.semilogy(iters, [max(abs(e), 1e-12) for e in errs], 's--',
                 color='#ff6b6b', lw=1.5, ms=6, label='|error %|')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('theta_hat', color='#ffd43b')
    ax2.set_ylabel('|error %|', color='#ff6b6b')
    ax2.tick_params(colors='#ff6b6b')
    ax.set_title('(c) Convergence of delta-iteration', pad=10)
    ax.set_xticks(iters)
    l1, b1 = ax.get_legend_handles_labels()
    l2, b2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, b1 + b2, fontsize=8,
              facecolor='#1a1a2e', labelcolor='#ccccee', framealpha=0.8)
    ax.grid(True, ls=':', alpha=0.3, color='#444466')

    # (d) residuals on quiet windows
    ax = axes[3]
    H_eff_q   = Hg[quiet] + Delta[quiet]
    h_mod_q   = k_hat * np.exp(gamma_wk * t[quiet]) / (1.0 + theta_hat * H_eff_q)
    residuals = h_obs[quiet] - h_mod_q
    max_res   = np.max(np.abs(residuals))
    ax.scatter(t[quiet], residuals, color='#74c0fc', s=15, alpha=0.9, zorder=5)
    ax.axhline(0, color='#a9e34b', lw=1.5, ls='--')
    shade_waves(ax)
    ax.set_ylabel('Residual h_obs − h_model')
    ax.set_xlabel('Week from enrollment')
    ax.set_title(
        f'(d) Quiet-window residuals — max={max_res:.2e} | '
        f'θ̂={theta_hat:.8f} | err={err_pct:+.6f}%', pad=10)
    ax.grid(True, ls=':', alpha=0.3, color='#444466')
    ax.legend(handles=[Patch(color=wave_colors[i % len(wave_colors)], alpha=0.5,
                             label=f'Wave {i+1}') for i in range(len(waves))],
              fontsize=8, facecolor='#1a1a2e', labelcolor='#ccccee', framealpha=0.8)

    annual_pct = sim['annual_rate'] * 100
    fig.suptitle(
        f'Delta-Iteration Estimator  |  {len(waves)}-Wave Simulation\n'
        f'Baseline mortality {annual_pct:.1f}%/yr  |  '
        f'True θ={theta0}  |  θ̂={theta_hat:.8f}  |  err={err_pct:+.6f}%',
        color='#e0e0ff', fontsize=13, fontweight='bold', y=1.005,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'  Saved: {out_path.name}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Summary heatmap
# ---------------------------------------------------------------------------

def _plot_heatmap(results: list, outdir: Path, suffix: str = "") -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    rates  = sorted(set(r['rate']       for r in results))
    thetas = sorted(set(r['theta_true'] for r in results))

    err_grid = np.zeros((len(thetas), len(rates)))
    for r in results:
        ri = rates.index(r['rate'])
        ti = thetas.index(r['theta_true'])
        err_grid[ti, ri] = r['err_pct']

    fig, ax = plt.subplots(figsize=(max(10, len(rates)*0.55),
                                    max(7,  len(thetas)*0.45)))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')

    vmax = max(abs(err_grid).max(), 0.1)
    im = ax.imshow(err_grid, aspect='auto', cmap='RdYlGn_r',
                   vmin=-vmax, vmax=vmax, origin='lower')

    ax.set_xticks(range(len(rates)))
    ax.set_yticks(range(len(thetas)))
    ax.set_xticklabels([f'{r*100:.0f}%' for r in rates],
                       rotation=45, ha='right', color='#aaaacc', fontsize=8)
    ax.set_yticklabels([f'{th:.0f}' for th in thetas],
                       color='#aaaacc', fontsize=8)
    ax.set_xlabel('Annual mortality rate', color='#aaaacc', fontsize=10)
    ax.set_ylabel('True theta0', color='#aaaacc', fontsize=10)

    # Annotate cells if grid is small enough
    if len(rates) * len(thetas) <= 400:
        for ti in range(len(thetas)):
            for ri in range(len(rates)):
                val = err_grid[ti, ri]
                color = 'white' if abs(val) > vmax * 0.6 else '#cccccc'
                ax.text(ri, ti, f'{val:+.2f}%', ha='center', va='center',
                        fontsize=max(5, 8 - len(rates)//5), color=color)

    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label('theta_hat error %', color='#aaaacc')
    cb.ax.yaxis.set_tick_params(color='#aaaacc')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='#aaaacc')

    ax.set_title('Delta-Iteration Estimator — theta0 recovery error across parameter space',
                 color='#e0e0ff', fontsize=12, pad=12)
    ax.tick_params(colors='#aaaacc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    plt.tight_layout()
    hmap_path = outdir / f'summary_heatmap{suffix}.png'
    plt.savefig(hmap_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f'  Heatmap saved: {hmap_path.name}')
    plt.close(fig)


def run_sweep(cfg: dict, outdir: Path, individual_plots: bool = True) -> list:
    """
    Iterate over all (annual_death_rate, theta0) combinations in cfg.
    Supports "start:end:step" range syntax in yaml for both lists.
    Produces one .png per combination plus a summary heatmap.
    """
    rates  = expand_range(cfg['cohort']['annual_death_rates'])
    thetas = expand_range(cfg['cohort']['theta0_values'])
    n_waves = len(cfg['epidemic_waves'])
    total   = len(rates) * len(thetas)

    print('=' * 65)
    print('DELTA-ITERATION SWEEP')
    print(f'  {len(rates)} mortality rates  x  {len(thetas)} theta values  =  {total} runs')
    print(f'  {n_waves} epidemic wave(s)  |  {cfg["simulation"]["n_weeks"]} weeks')
    print('=' * 65)

    header = f"  {'mort%':>6}  {'theta_true':>10}  {'theta_hat':>12}  {'err%':>9}  {'iters':>5}"
    print('\n' + header)
    print('  ' + '-' * (len(header) - 2))

    results = []
    for rate in rates:
        for theta_true in thetas:
            run_cfg = copy.deepcopy(cfg)
            run_cfg['cohort']['annual_death_rate'] = rate
            run_cfg['cohort']['theta0_true']       = theta_true

            sim = simulate_cohort(run_cfg)
            est = estimate_theta0(sim, run_cfg, verbose=False)

            err_pct = 100.0 * (est['theta_hat'] - theta_true) / theta_true
            n_iter  = len(est['history'])

            print(f"  {rate*100:>6.1f}  {theta_true:>10.2f}  "
                  f"{est['theta_hat']:>12.8f}  {err_pct:>+9.5f}%  {n_iter:>5}")

            fname = (f"theta0_mort{rate*100:.0f}pct_"
                     f"theta{str(theta_true).replace('.','p')}.png")
            if individual_plots:
                plot_results(sim, est, outdir / fname)

            results.append(dict(
                rate=rate, theta_true=theta_true,
                theta_hat=est['theta_hat'], err_pct=err_pct, n_iter=n_iter,
            ))

    print(f'\nAll {total} plots saved to: {outdir}')

    # Summary heatmap
    _plot_heatmap(results, outdir)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description='Delta-iteration theta0 sweep')
    parser.add_argument('--config',  default='params_nwave.yaml')
    parser.add_argument('--outdir',  default='out/')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip individual PNGs, only produce heatmap')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_sweep(cfg, outdir, individual_plots=not args.no_plots)


if __name__ == '__main__':
    main()
