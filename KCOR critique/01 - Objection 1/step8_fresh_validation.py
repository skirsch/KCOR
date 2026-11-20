"""Fresh independent draws under the unchanged Step 6/7 diagnostic rules.

This extension imports the existing simulator and production fitting functions.
It reads fixed cutoffs, verifies its adapters against earlier saved runs, and
never recalibrates or modifies prior code/results.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
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
    ap.add_argument('--step7',type=Path,required=True)
    ap.add_argument('--outdir',type=Path,required=True)
    args=ap.parse_args()
    m6=json.loads((args.step6/'design.json').read_text())
    m7=json.loads((args.step7/'design.json').read_text())
    for meta in [m6,m7]:
        for name,h in meta['source_hashes'].items(): assert sha(ROOT/name)==h,name
    for num,meta in [(6,m6),(7,m7)]:
        script='step6_random_detection.py' if num==6 else 'step7_additional_diagnostics.py'
        assert sha(Path(__file__).with_name(script))==meta['script_sha256']
    d=json.loads((ROOT/'KCOR critique/01 - Objection 1/step5_results_2026-11-09/manifest.json').read_text())['original_design']
    cutoffs=json.loads((args.step7/'frozen_cutoffs.json').read_text())
    residual_cut=json.loads((args.step6/'frozen_threshold.json').read_text())['score_cutoff']
    variants=m7['variants']; metrics=list(cutoffs)
    reps=200; phase=2  # Earlier calibration/evaluation used branches 0 and 1.
    inputs=[SOURCE,ROOT/'code/KCOR.py',ROOT/'data/Czech/Czech.yaml',
            Path(__file__).with_name('step1_contamination_audit.py'),Path(__file__).with_name('step6_random_detection.py'),
            Path(__file__).with_name('step7_additional_diagnostics.py'),
            ROOT/'KCOR critique/01 - Objection 1/step5_results_2026-11-09/manifest.json']
    inputs += [args.step6/f for f in ['design.json','frozen_threshold.json','evaluation_runs.csv']]
    inputs += [args.step7/f for f in ['design.json','frozen_cutoffs.json','evaluation_metrics.csv','variant_runs.csv','summary.csv']]
    design=dict(created_utc=datetime.now(timezone.utc).isoformat(),seed=m6['seed'],fresh_phase=phase,
                repetitions=reps,profiles=m6['profiles'],targets=m6['targets'],cohort_size=d['n_initial'],
                original_design=d,variants=variants,additional_cutoffs=cutoffs,residual_cutoff=residual_cut,
                rule='Unchanged Step 6 residual acceptance AND all four Step 7 checks. No recalibration, optimization, early stopping, or post-outcome exclusions.',
                metrics=m7['metrics'],large_error_rule='Absolute relative error >10% in nominal raw contrast versus its matching mechanistic reference.',
                pairing='Within each fresh repetition the comparator is shared and affected-group cases share seed streams; branch 2 is distinct from earlier branches 0 and 1.',
                replay='Before fresh draws, reproduce all three cases and all 13 variants for earlier evaluation repetitions 0, 99, and 199, including residual scores and acceptance.',
                limits=['Validation is conditional on the same synthetic generator, parameters, cohort size, and fixed cutoffs.',
                        'These exploratory numerical checks are not the complete manuscript procedure.',
                        'No real-data plausibility or attribution claim; no guaranteed combined false-alarm rate.',
                        'Window shifts can include four harm weeks in the baseline-clean case; skip changes alter included follow-up.',
                        'Intervals measure simulation-rate uncertainty conditional on the cutoffs.'],
                source_hashes={str(p.resolve().relative_to(ROOT)):sha(p) for p in inputs},script_sha256=sha(Path(__file__)))
    args.outdir.mkdir(parents=True,exist_ok=False)
    (args.outdir/'design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    E=load_experiment(); K=E.K
    k,gamma,skip=d['mortality_scale'],d['gamma_per_week'],d['skip_weeks']
    t=np.arange(d['n_weeks'],dtype=float)
    qlo,qhi=d['quiet_start_rebased']+skip,d['quiet_end_rebased']+skip
    anchor_weeks=int(K.get_gompertz_theta_fit_config()['k_anchor_weeks'])

    def generate(theta,start,branch,rep,cohort):
        rng=np.random.default_rng(np.random.SeedSequence([m6['seed'],branch,rep,cohort]))
        amount=m6['profiles']['baseline_affected']['added_weekly_hazard']
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
            row=dict(final=float(H0[-1]),theta=float(th) if ok else 0.,k=float(kh),eligible=eligible,linearity_loss=loss,status=diag.get('status'))
            if name=='nominal':
                assert not diag.get('delta_applied_values')
                Hgom=np.zeros_like(t); post=tr>=0
                Hgom[post]=np.r_[0.,np.cumsum(kh*np.exp(gamma*tr[post]))[:-1]]
                prediction=kh*np.exp(gamma*tr)/(1+float(diag.get('theta0_raw',th))*Hgom)
                p=-np.expm1(-prediction[quiet])
                z=(dead[quiet]-alive[quiet]*p)/np.sqrt(np.maximum(alive[quiet]*p*(1-p),1e-12))
                u=tr[quiet]-tr[quiet].mean()
                row['score']=float(max(np.mean(z*z),np.sum(z)**2/len(z),np.dot(u,z)**2/np.dot(u,u)))
            out[name]=row
        return out

    def compare(a,b,rep,case):
        nominal=a['nominal']['final']/b['nominal']['final']
        variant_rows=[]
        for name,*_ in variants:
            ratio=a[name]['final']/b[name]['final']
            variant_rows.append(dict(rep=rep,case=case,variant=name,raw_final=ratio,relative_change=abs(ratio/nominal-1),
                eligible=a[name]['eligible'] and b[name]['eligible'],theta_A=a[name]['theta'],theta_B=b[name]['theta'],
                status_A=a[name]['status'],status_B=b[name]['status']))
        frame=pd.DataFrame(variant_rows)
        score=max(a['nominal']['score'],b['nominal']['score'])
        result=dict(rep=rep,case=case,target=m6['targets'][case],raw_final=nominal,residual_score=score,
                    residual_accepted=bool(frame.iloc[0].eligible and score<=residual_cut),
                    nominal_near_zero=bool(min(a['nominal']['theta'],b['nominal']['theta'])<1e-8),
                    relative_error_percent=100*(nominal/m6['targets'][case]-1),
                    large_error=abs(nominal/m6['targets'][case]-1)>.10)
        for family in ['window','skip','gamma']:
            block=frame[frame.variant.str.startswith(family)]
            result[family+'_change']=float(block.relative_change.max()) if block.eligible.all() else float('inf')
        result['linearity_loss']=max(a['nominal']['linearity_loss'],b['nominal']['linearity_loss']) if frame.iloc[0].eligible else float('inf')
        for metric in metrics: result['pass_'+metric]=bool(result[metric]<=cutoffs[metric])
        result['combined_accepted']=result['residual_accepted'] and all(result['pass_'+m] for m in metrics)
        assert np.isfinite(nominal) and nominal>0
        return result,frame

    old6=pd.read_csv(args.step6/'evaluation_runs.csv',float_precision='round_trip').set_index(['rep','case'])
    old7=pd.read_csv(args.step7/'evaluation_metrics.csv',float_precision='round_trip').set_index(['rep','case'])
    oldv=pd.read_csv(args.step7/'variant_runs.csv',float_precision='round_trip')
    oldv=oldv[oldv.phase==1].set_index(['rep','case','variant'])
    replays=[]
    for rep in [0,99,199]:
        b=fit_all(generate(d['theta_B'],None,1,rep,1))
        for case,profile in m6['profiles'].items():
            a=fit_all(generate(d['theta_A'],profile['start_rebased'],1,rep,0))
            row,frame=compare(a,b,rep,case)
            o6=old6.loc[(rep,case)]; o7=old7.loc[(rep,case)]
            np.testing.assert_allclose([row['raw_final'],row['residual_score']],[o6.raw_final,o6.score],rtol=1e-9,atol=1e-11)
            np.testing.assert_allclose([row[m] for m in metrics],[o7[m] for m in metrics],rtol=1e-9,atol=1e-11)
            for key in ['residual_accepted','combined_accepted','large_error']: assert row[key]==o7[key]
            for rr in frame.itertuples():
                ov=oldv.loc[(rep,case,rr.variant)]
                np.testing.assert_allclose([rr.raw_final,rr.theta_A,rr.theta_B],[ov.raw_final,ov.theta_A,ov.theta_B],rtol=1e-9,atol=1e-11)
                assert rr.eligible==ov.eligible
            replays.append(dict(rep=rep,case=case,variants_matched=len(frame),metrics_and_decisions_matched=True))
    pd.DataFrame(replays).to_csv(args.outdir/'prior_replay_verification.csv',index=False)
    print('Verified nine earlier cases, 117 variants, residual scores, and fixed decisions. Starting fresh branch 2.',flush=True)

    def append(frame,name):
        dest=args.outdir/name
        frame.to_csv(dest,index=False,mode='a',header=not dest.exists())

    rows=[]
    for rep in range(reps):
        b=fit_all(generate(d['theta_B'],None,phase,rep,1))
        for case,profile in m6['profiles'].items():
            a=fit_all(generate(d['theta_A'],profile['start_rebased'],phase,rep,0))
            row,frame=compare(a,b,rep,case)
            rows.append(row); append(frame,'variant_runs.csv'); append(pd.DataFrame([row]),'evaluation_metrics.csv')
        if (rep+1)%25==0: print(f'Fresh validation {rep+1}/{reps} paired repetitions',flush=True)
    runs=pd.DataFrame(rows); summaries=[]
    for case,g in runs.groupby('case',sort=False):
        masks={'residual_only':g.residual_accepted,'all_additional':g.combined_accepted}
        masks.update({m:g.residual_accepted & g['pass_'+m] for m in metrics})
        for label,mask in masks.items():
            accepted=g[mask]; n=len(accepted); bad=int(accepted.large_error.sum())
            lo,hi=wilson(n,len(g)); blo,bhi=wilson(bad,n)
            summaries.append(dict(case=case,check=label,accepted=n,total=len(g),acceptance_percent=100*n/len(g),
                acceptance_wilson95_low=lo,acceptance_wilson95_high=hi,target=m6['targets'][case],
                raw_median=g.raw_final.median(),accepted_raw_median=accepted.raw_final.median(),
                accepted_raw_p05=accepted.raw_final.quantile(.05),accepted_raw_p95=accepted.raw_final.quantile(.95),
                accepted_median_relative_error_percent=accepted.relative_error_percent.median(),
                accepted_large_error_count=bad,accepted_large_error_percent=100*bad/n if n else np.nan,
                accepted_large_error_wilson95_low=blo,accepted_large_error_wilson95_high=bhi,
                joint_accept_large_error_percent=100*bad/len(g)))
    summary=pd.DataFrame(summaries); summary.to_csv(args.outdir/'summary.csv',index=False)
    wide=runs.pivot(index='rep',columns='case',values='raw_final')
    wide['affected_minus_clean']=wide.baseline_affected-wide.baseline_clean
    wide.to_csv(args.outdir/'paired_contrasts.csv')
    old_summary=pd.read_csv(args.step7/'summary.csv'); old_summary['sample']='Earlier evaluation'
    summary['sample']='Fresh validation'
    comparison=pd.concat([old_summary,summary],ignore_index=True)
    comparison=comparison[comparison.check=='all_additional']
    comparison.to_csv(args.outdir/'old_new_comparison.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    labels=['No harm','Harm after\nbaseline','Harm includes\nbaseline']; cases=list(m6['profiles']); x=np.arange(3)
    for off,sample,color in [(-.18,'Earlier evaluation','#999999'),(.18,'Fresh validation','#157f60')]:
        block=comparison[comparison['sample']==sample].set_index('case').loc[cases]
        axes[0].bar(x+off,block.acceptance_percent,width=.36,label=sample,color=color)
        axes[1].bar(x+off,block.accepted_raw_median/np.array([m6['targets'][c] for c in cases]),width=.36,color=color)
    axes[0].set_ylabel('Passed all implemented checks (%)'); axes[0].set_ylim(0,100); axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Median estimate / reference among passing runs'); axes[1].axhline(1,color='black',ls='--',lw=1)
    for ax in axes: ax.set_xticks(x,labels); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Fresh independent simulations with unchanged diagnostic cutoffs')
    fig.savefig(args.outdir/'fresh_validation.png',dpi=160); plt.close(fig)
    for name,h in design['source_hashes'].items(): assert sha(ROOT/name)==h,name
    (args.outdir/'completion.json').write_text(json.dumps(dict(fresh_case_runs=len(runs),fresh_phase=phase,prior_cases_reproduced=len(replays),source_hashes_unchanged=True,cutoffs_unchanged=True,script_sha256=sha(Path(__file__))),indent=2),encoding='utf-8')
    print(summary[summary.check=='all_additional'].to_string(index=False),flush=True)


if __name__=='__main__': main()
