"""Replay Step 6 draws with additional exploratory diagnostics, separately.

Single-check cutoffs are calibrated on Step 6's separate null calibration draws.
Their intersection is descriptive: it has no promised joint false-alarm rate.
No original code or earlier audit outputs are modified.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from step1_contamination_audit import ROOT, SOURCE, load_experiment
from step6_random_detection import sha, wilson


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--step6',type=Path,required=True)
    ap.add_argument('--outdir',type=Path,required=True)
    args=ap.parse_args()
    meta=json.loads((args.step6/'design.json').read_text())
    for name,h in meta['source_hashes'].items():
        assert sha(ROOT/name)==h,name
    assert sha(Path(__file__).with_name('step6_random_detection.py'))==meta['script_sha256']
    prior=json.loads((ROOT/'KCOR critique/01 - Objection 1/step5_results_2026-11-09/manifest.json').read_text())
    d=prior['original_design']
    saved=pd.read_csv(args.step6/'evaluation_runs.csv').set_index(['rep','case'])
    cal_saved=pd.read_csv(args.step6/'calibration.csv').set_index('rep')
    residual_cut=json.loads((args.step6/'frozen_threshold.json').read_text())['score_cutoff']
    E=load_experiment(); K=E.K
    k,gamma=d['mortality_scale'],d['gamma_per_week']
    skip=d['skip_weeks']; t=np.arange(d['n_weeks'],dtype=float)
    qlo,qhi=d['quiet_start_rebased']+skip,d['quiet_end_rebased']+skip
    anchor_weeks=int(K.get_gompertz_theta_fit_config()['k_anchor_weeks'])
    variants=[('nominal',0,0,skip,1.)]
    variants += [(f'window_{a}_{b}',a,b,skip,1.) for a in [-4,0,4] for b in [-4,0,4] if a or b]
    variants += [(f'skip_{s}',0,0,s,1.) for s in [0,4]]
    variants += [(f'gamma_{g}',0,0,skip,g) for g in [.95,1.05]]
    metrics=['window_change','skip_change','gamma_change','linearity_loss']
    inputs=[SOURCE,ROOT/'code/KCOR.py',ROOT/'data/Czech/Czech.yaml',
            Path(__file__).with_name('step1_contamination_audit.py'),
            Path(__file__).with_name('step6_random_detection.py'),
            args.step6/'design.json',args.step6/'evaluation_runs.csv',args.step6/'calibration.csv',
            args.step6/'frozen_threshold.json',ROOT/'KCOR critique/01 - Objection 1/step4_results_2026-11-09/diagnostic_mapping.csv',
            ROOT/'KCOR critique/01 - Objection 1/step5_results_2026-11-09/manifest.json']
    design=dict(created_utc=datetime.now(timezone.utc).isoformat(),variants=variants,
        source_hashes={str(p.resolve().relative_to(ROOT)):sha(p) for p in inputs},script_sha256=sha(Path(__file__)),
        metrics={'window_change':'max abs(variant raw / nominal raw - 1) over eight boundary shifts',
                 'skip_change':'max abs(variant raw / nominal raw - 1) for skip 0 and 4',
                 'gamma_change':'max abs(variant raw / nominal raw - 1) for gamma times 0.95 and 1.05',
                 'linearity_loss':'max(1 - R squared) across both nominal corrected cumulative hazards versus Gompertz time'},
        cutoff_rule='Each metric independently uses the higher empirical 95th percentile of all 200 separate null calibration runs, before evaluation replay. Failed variant fits force rejection for their sensitivity family.',
        combined_rule='Step 6 residual acceptance AND all four additional checks. No promised joint 5% false-alarm rate; report observed null rejection.',
        scope='Exploratory diagnostic comparison on the existing draws, not a complete manuscript protocol or fresh validation. No thresholds tuned to harm outcomes.',
        caveats=['Skip/gamma perturbation magnitudes are exploratory, inherited from Step 4.',
                 'Moving the lower window bound earlier includes four harm weeks in the baseline-clean case; no affected-baseline harm enters any shifted quiet window.',
                 'Skip changes alter the included follow-up and target: treat this as stability, not as error against the nominal target.',
                 'One quiet window cannot test omit-window or between-window coherence.',
                 'Near-zero frailty frequency is reported descriptively, with no new rejection rule.',
                 'Uncertainty intervals are simulation-rate intervals conditional on calibration.'])
    args.outdir.mkdir(parents=True,exist_ok=False)
    (args.outdir/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')

    def generate(theta,start,phase,rep,cohort):
        rng=np.random.default_rng(np.random.SeedSequence([meta['seed'],phase,rep,cohort]))
        amount=meta['profiles']['baseline_affected']['added_weekly_hazard']
        fn=None if start is None else lambda w:amount if start<=w<start+53 else 0.
        return E.simulate_one_cohort(rng,d['n_initial'],d['n_weeks'],k,gamma,theta,skip,fn)

    def fit_all(counts):
        alive,dead=counts
        obs=K.hazard_from_mr(dead.astype(float)/np.maximum(alive,1))
        out={}
        for name,ds,de,s,gs in variants:
            tr=t-s; quiet=(t>=qlo+ds)&(t<=qhi+de)
            eff=np.where(t>=s,obs,0.)
            (kh,th),diag=K.fit_theta0_gompertz(eff,tr,quiet,anchor_weeks,gamma*gs,deaths_arr=dead.astype(float))
            ok=bool(diag.get('success',False)) and np.isfinite(th)
            eligible=ok and all(not bool(diag.get(f,False)) for f in ['identifiability_flag','insufficient_signal','insufficient_deaths','bad_fit'])
            H0=K.invert_gamma_frailty(np.cumsum(eff),float(th) if ok else 0.)
            x=np.expm1(gamma*gs*tr[quiet])/(gamma*gs); y=H0[quiet]
            pred=np.polyval(np.polyfit(x,y,1),x)
            loss=float(np.sum((y-pred)**2)/np.sum((y-y.mean())**2))
            out[name]=dict(final=float(H0[-1]),theta=float(th) if ok else 0.,k=float(kh),eligible=eligible,linearity_loss=loss,status=diag.get('status'))
        return out

    def compare(a,b,phase,rep,case,original):
        nominal=a['nominal']['final']/b['nominal']['final']
        if phase==1:
            np.testing.assert_allclose([nominal,a['nominal']['theta'],b['nominal']['theta'],a['nominal']['k'],b['nominal']['k']],
                [original.raw_final,original.theta_A,original.theta_B,original.k_A,original.k_B],rtol=1e-9,atol=1e-11)
        else:
            np.testing.assert_allclose([a['nominal']['theta'],b['nominal']['theta']],
                [original.theta_A,original.theta_B],rtol=1e-9,atol=1e-11)
        rows=[]
        for name,*_ in variants:
            ratio=a[name]['final']/b[name]['final']
            rows.append(dict(phase=phase,rep=rep,case=case,variant=name,raw_final=ratio,
                relative_change=abs(ratio/nominal-1),eligible=a[name]['eligible'] and b[name]['eligible'],
                theta_A=a[name]['theta'],theta_B=b[name]['theta'],status_A=a[name]['status'],status_B=b[name]['status']))
        frame=pd.DataFrame(rows)
        append(frame,'variant_runs.csv')
        result=dict(phase=phase,rep=rep,case=case,raw_final=nominal,
                    residual_accepted=bool(original.accepted) if phase==1 else bool(original.eligible and original.score<=residual_cut),
                    nominal_near_zero=bool(min(a['nominal']['theta'],b['nominal']['theta'])<1e-8))
        for family in ['window','skip','gamma']:
            g=frame[frame.variant.str.startswith(family)]
            result[family+'_change']=float(g.relative_change.max()) if g.eligible.all() else float('inf')
        result['linearity_loss']=max(a['nominal']['linearity_loss'],b['nominal']['linearity_loss']) if frame.iloc[0].eligible else float('inf')
        if phase==1:
            result['large_error']=bool(original.absolute_error_above_10pct)
            result['target']=float(original.target)
        return result

    def append(frame,name):
        dest=args.outdir/name
        frame.to_csv(dest,index=False,mode='a',header=not dest.exists())

    calibration=[]
    for rep in range(meta['calibration_reps']):
        a=fit_all(generate(d['theta_A'],None,0,rep,0)); b=fit_all(generate(d['theta_B'],None,0,rep,1))
        row=compare(a,b,0,rep,'no_effect',cal_saved.loc[rep]); calibration.append(row)
        append(pd.DataFrame([row]),'calibration_metrics.csv')
        if (rep+1)%25==0: print(f'Calibration diagnostics {rep+1}/200',flush=True)
    cal=pd.DataFrame(calibration)
    cutoffs={m:float(np.quantile(cal[m],.95,method='higher')) for m in metrics}
    assert all(np.isfinite(list(cutoffs.values())))
    (args.outdir/'frozen_cutoffs.json').write_text(json.dumps(cutoffs,indent=2),encoding='utf-8')
    print('Additional cutoffs frozen: '+json.dumps(cutoffs),flush=True)
    rows=[]
    for rep in range(meta['evaluation_reps']):
        b=fit_all(generate(d['theta_B'],None,1,rep,1))
        for case,profile in meta['profiles'].items():
            a=fit_all(generate(d['theta_A'],profile['start_rebased'],1,rep,0))
            row=compare(a,b,1,rep,case,saved.loc[(rep,case)])
            for m in metrics: row['pass_'+m]=row[m]<=cutoffs[m]
            row['combined_accepted']=row['residual_accepted'] and all(row['pass_'+m] for m in metrics)
            rows.append(row); append(pd.DataFrame([row]),'evaluation_metrics.csv')
        if (rep+1)%25==0: print(f'Evaluation diagnostics {rep+1}/200 paired repetitions',flush=True)
    runs=pd.DataFrame(rows)
    summaries=[]
    for case,g in runs.groupby('case',sort=False):
        gates={'residual_only':g.residual_accepted}
        gates.update({m:g.residual_accepted & g['pass_'+m] for m in metrics})
        gates['all_additional']=g.combined_accepted
        for label,mask in gates.items():
            accepted=g[mask]; n=len(accepted); bad=int(accepted.large_error.sum())
            lo,hi=wilson(n,len(g))
            summaries.append(dict(case=case,check=label,accepted=n,total=len(g),acceptance_percent=100*n/len(g),
                acceptance_wilson95_low=lo,acceptance_wilson95_high=hi,
                accepted_raw_median=accepted.raw_final.median(),accepted_large_error_count=bad,
                accepted_large_error_percent=100*bad/n if n else np.nan,
                joint_accept_large_error_percent=100*bad/len(g)))
    summary=pd.DataFrame(summaries); summary.to_csv(args.outdir/'summary.csv',index=False)
    runs.groupby('case')[metrics].quantile([.05,.5,.95]).to_csv(args.outdir/'metric_quantiles.csv')
    fig,axes=plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    labels={'no_effect':'No added harm','baseline_clean':'Harm after baseline','baseline_affected':'Harm includes baseline'}
    colors=['#777777','#157f60','#9348a6']
    titles=['Quiet-window boundary sensitivity','Skipped-week sensitivity','Mortality-trend sensitivity','Corrected-curve nonlinearity']
    for ax,m,title in zip(axes.flat,metrics,titles):
        for (case,g),color in zip(runs.groupby('case',sort=False),colors):
            values=np.sort(g[m]); ax.step(values,np.arange(1,len(values)+1)/len(values),where='post',label=labels[case],color=color)
        ax.axvline(cutoffs[m],color='black',ls='--',lw=1)
        ax.set_title(title); ax.set_xlabel('Diagnostic score (larger is worse)'); ax.set_ylabel('Fraction of runs')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Exploratory checks: each dashed cutoff calibrated separately on no-harm runs')
    fig.savefig(args.outdir/'additional_diagnostics.png',dpi=150); plt.close(fig)
    for name,h in design['source_hashes'].items(): assert sha(ROOT/name)==h,name
    (args.outdir/'completion.json').write_text(json.dumps(dict(nominal_evaluation_replays=len(runs),nominal_calibration_replays=len(cal),source_hashes_unchanged=True,script_sha256=sha(Path(__file__))),indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__': main()
