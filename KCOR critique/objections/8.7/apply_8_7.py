"""Apply measured matched frailty-misspecification results; no estimator changes."""
from pathlib import Path
import sys,json
r=Path(sys.argv[1]);e=Path(sys.argv[2]);results=json.loads((e/'summary.json').read_text(encoding='utf8'))
assert len(results)==6 and all(x['datasets']==100 for x in results)
p=r/'documentation/preprint/paper.md';s=p.read_text(encoding='utf8')
start=s.index('To assess robustness to departures from the gamma frailty assumption, simulations were conducted under alternative frailty distributions')
end=s.index('#### 3.3.2 Quiet-window contamination stress test',start)
new="""To assess robustness to departures from the gamma frailty assumption, a matched null study compared gamma, lognormal, two-point, continuous bimodal-mixture and within-cluster correlated gamma frailty using the current fitter. The cohorts share an individual baseline and the frailty laws have matched means and variances across scenarios. SI S9 reports the mean anchored KCOR (bias relative to 1), empirical sampling variance, percentile-interval coverage, and numerical-fit and residual-screen flags across 100 independent datasets per scenario. For correlated frailty, cluster resampling is compared with person resampling on the same datasets.

These results are conditional on the tested distributions and event rates; a diagnostic pass is not proof that the frailty law is correct. Under misspecification the normalized contrast may move in either direction. The historical tail-mixture experiment in SI S4.5 provides a separate adversarial stress test.

"""
s=s[:start]+new+s[end:];p.write_text(s,encoding='utf8')
p=r/'documentation/preprint/supplement.md';s=p.read_text(encoding='utf8')
a='This setting stress-tests whether the revised $\\theta_{0,d}$ estimator and depletion normalization remain effective when frailty heterogeneity is concentrated in the tails rather than smoothly distributed.'
b='This historical single-quiet-window experiment stress-tests depletion normalization when frailty heterogeneity is concentrated in the tails rather than smoothly distributed.'
assert s.count(a)==1;s=s.replace(a,b)
a='The evaluation includes (i) KCOR drift, (ii) fit quality under the revised estimator, (iii) post-normalization linearity, and (iv) parameter stability under window perturbation.'
b='The archived results report KCOR drift, fitted parameters, cumulative-hazard fit error and linearity diagnostics under the earlier fitter. They do not provide replicated coverage or parameter-perturbation stability under the current estimator; the matched current-fitter frailty study is in SI S9.'
assert s.count(a)==1;s=s.replace(a,b)
s+='\n\n## S9. Matched frailty-misspecification validation\n\n'
s+='The five scenarios use 100,000 people per cohort, 120 weeks, a common Gompertz baseline (weekly initial rate 0.002, annual slope 0.085), mean-one frailty at enrollment and variances 1.0 and 0.3. The stabilization skip is two weeks, the reference is the fourth accumulating week, and the quiet window is enrollment weeks 20--80 (zero-based). Gamma, lognormal, two-point and continuous bimodal-mixture frailties are sampled independently across people. For the two-point law, the low support is 0.2 and the high support and weights match the stated moments; the bimodal law is a mixture of two narrow gamma distributions with within-component relative variance 0.02, adjusted to match the same marginal moments. Correlated gamma frailty is shared by clusters of 20 people, with independent clusters. The target anchored endpoint is 1 under the specified fixed-frailty null.\n\n'
s+='Each scenario uses 100 independent datasets and 199 matched-unit bootstrap replicates of the current fitter. Independent scenarios resample persons; the dependent scenario resamples entire clusters, with a separate person-resampling comparison. Means have 95% Monte Carlo confidence intervals; sampling variance is calculated across independent datasets. Coverage is the proportion of all 100 attempted datasets whose complete percentile interval contains 1, with Wilson uncertainty for coverage. Non-OK numerical-fit and exploratory residual-screen flags are reported separately; neither implies a calibrated identification test.\n\n'
s+='Table: Matched frailty-misspecification null study at the endpoint. Bias is mean KCOR minus 1. The variance is across independent point estimates, not across time. {#tbl:si_misspecification}\n\n'
s+='| Frailty / resampling | Mean KCOR [95% MC CI] | Sampling variance | Coverage [95% Wilson CI] | Non-OK fit / 100 | Residual flag / 100 |\n|---|---:|---:|---:|---:|---:|\n'
names={'gamma':'Gamma','lognormal':'Lognormal','two_point':'Two-point','bimodal':'Bimodal','correlated_gamma':'Correlated gamma'}
for q in results:
 m,lo,hi=q['mean_kcor_ci'];cl,ch=q['coverage_ci'];label=names[q['scenario']]+' / '+q['bootstrap']
 s+=f"| {label} | {m:.3f} [{lo:.3f}, {hi:.3f}] | {q['variance']:.5f} | {q['coverage']}% [{cl*100:.1f}, {ch*100:.1f}] | {q['fit_flag']} | {q['residual_flag']} |\n"
s+='\n'
available=[q['intervals_available'] for q in results];failed=[q['failed'] for q in results]
s+=f"Full percentile intervals were available for {min(available)}--{max(available)} of 100 datasets per row; numerical fitting failures ranged from {min(failed)} to {max(failed)} per row. "
exact=json.loads((e/'expected-counts.json').read_text());v=[x['point'] for x in exact]
s+=f"With expected (noiseless) counts, fitted anchored endpoints ranged from {min(v):.5f} to {max(v):.5f}; the fitted variance need not equal the true variance for non-gamma laws. "
cluster=next(x for x in results if x['scenario']=='correlated_gamma' and x['bootstrap']=='cluster')
person=next(x for x in results if x['scenario']=='correlated_gamma' and x['bootstrap']=='person')
s+=f"For correlated frailty, median interval widths were {cluster['median_interval_width']:.3f} with clusters and {person['median_interval_width']:.3f} with persons. "
s+='These are design-specific results, not a general coverage guarantee under arbitrary misspecification or dependence. Source: `analysis/misspecification/` (simulation generator, replicate-level results and summary).\n'
p.write_text(s,encoding='utf8')
