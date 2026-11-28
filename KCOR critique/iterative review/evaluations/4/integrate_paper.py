from pathlib import Path
import os,shutil,csv,json,re
R=Path(__file__).resolve().parents[3]
D=Path(os.environ["TEMP"])/"kcor-s4-approved/m94"
P=D/"documentation/preprint"
def put(path,s):
 path.unlink(missing_ok=True);path.write_text(s,encoding="utf8",newline="\n")
p=P/"paper.md";s=p.read_text(encoding="utf-8-sig")
a=s.index("### 2.5 Estimating"); b=s.index("### 2.6 ",a)
s=s[:a]+r"""### 2.5 Joint baseline and frailty estimation across quiet windows

For each cohort independently, estimate one common Gompertz baseline scale $k_d$ and enrollment-time frailty variance $\theta_d\ge0$, with fixed slope $\gamma$. Weekly observed hazards use the stabilization skip only. Let $G(s;k)=k(e^{\gamma s}-1)/\gamma$ denote cumulative baseline exposure at the start of rebased week $s$. For a proportional offset $a$ the exact integrated weekly prediction is

$$
q(s;k,\theta,a)=\frac{1}{\theta}\log\left(1+\frac{\theta[G(s+1;k)-G(s;k)]}{1+\theta[G(s;k)+a]}\right).
$$

At $\theta=0$ use the continuous limit. The first-window joint fit provides an initialization, not a fixed final baseline rate. For each candidate pair, reconstruct proportional cumulative exposure from observed survival, $H_{\mathrm{eff}}(s)=\operatorname{expm1}(\theta H_{\mathrm{obs}}(s))/\theta$, using the linear limit at zero. At quiet re-entry, including entry to the first quiet window if preceded by a gap, reconstruct raw cumulative offset $H_{\mathrm{eff}}(s)-G(s;k)$; the incremental offset relative to the preceding raw offset is constrained nonnegative. Record signed deficits diagnostically. At initial entry following proportional shock, the effective rate $r$ and observed survival identify the entry-aware seed rate through $k=r\exp(\theta H_{\rm obs}(s))/(1+\theta r F(s))$ with $F(s)=G(s;1)$.

The objective jointly refits both parameters:

$$
(\hat k_d,\hat\theta_d)=\arg\min_{k>0,\theta\ge0}
\sum_{s\in\mathcal Q_d}\{h_d^{\rm eff}(s)-q_d(s;k,\theta,a_d(s;k,\theta))\}^2 .
$$
{#eq:nls-objective}

Here $\mathcal Q_d$ includes the designated quiet and early fitting-anchor rows used by the implementation. The offsets are reconstructed inside the objective, not treated as fixed values. Optimize in log $k$ and nonnegative $\theta$; compare the first-window initialization with a second start at $\theta=0.5$. The implementation retains event-count, parameter-bound and residual diagnostics. No-common-signal and failed fits are flagged; zero-variance fallback is not evidence that frailty is absent.

When a specified nonlinear wave is active, use the full survival law $L(B,W)=E[\exp(-ZB-Z^\alpha W)]$ of §2.7.3 for both fitting and inversion. For every candidate $k,\theta$, carry proportional gap exposures and the marked wave exposure through the history, predict quiet-week increments $\log L(B_s,W_s)-\log L(B_{s+1},W_{s+1})$, and minimize the same weekly least-squares objective. Marked wave rows reconstruct exposure rather than contributing quiet residuals, even if they overlap the early fitting anchors. Unmarked nonlinear shocks cannot be distinguished from other mortality changes by counts alone. This route assumes a supplied exponent; the proportional route is used when the module is off or $\alpha=1$.

"""+s[b:]
s=s.replace("refits $\theta_{0,d}$ jointly across all prespecified quiet windows","jointly refits $k_d$ and $\theta_{0,d}$ across all prespecified quiet windows")
s=s.replace("Wave-period NPH adjustment is applied after inversion only to **frailty-neutral cumulative hazard**, not to raw observed hazard (§2.7.3).","For a specified nonlinear exponent, the fitting and correction use the same survivor law; the raw observed hazard is not rescaled (§2.7.3).")
a=s.index("#### 2.7.3 ");b=s.index("#### 2.7.4 ",a)
s=s[:a]+r"""#### 2.7.3 Application of the NPH correction once $\alpha$ is specified

Let $Z$ have the starting mean-one gamma law, and define $L(B,W)=E[\exp(-ZB-Z^\alpha W)]$ for proportional exposure $B$ and additional wave exposure $W$. At $\alpha=1$ ordinary gamma inversion returns $B+W$ exactly. At wave entry, use the preceding inverted cumulative hazard for $B$ and set $W=0$.

For each modeled wave week, let $q$ be its observed integrated hazard and $b=G(s+1;\hat k)-G(s;\hat k)$. Solve the monotone scalar equation

$$
\frac{L(B+b,W+w)}{L(B,W)}=e^{-q}
$$

for a nonnegative wave increment $w$. If mortality is below the fitted conditional baseline, no such nonnegative increment exists at the planned $b$. In those weeks keep prior wave exposure, assign no additional wave exposure, and solve instead for a smaller nonnegative proportional increment from the observed survival. Count these deficit weeks and report their frequency. This is an attribution convention, not evidence of negative wave exposure or an estimator of both unknown hazards without a baseline assumption.

Update the cumulative exposures week by week and output $H_{\rm corr}=B+W$. After wave exit $e$, hold $W_e$ fixed and solve $L(B_e+u,W_e)/L(B_e,W_e)=\exp[-(H_{\rm obs}(t)-H_{\rm obs}(e))]$ for the subsequent proportional exposure $u(t)$. This retains total reference-frailty cumulative mortality while accounting for changing survivor frailty. Numerical exactness is conditional on the working survival model, fitted inputs, dates, and attribution rules. The same law is used to fit quiet mortality following nonlinear selection.

"""+s[b:]
s=s.replace("During the NPH window, construct the discrete Gompertz reference path and divide strictly positive inverted excess by $\widetilde F_d(t;\alpha)$ as in §2.7.3. After wave exit, apply the conditional survival equation to subsequent weeks.","During the NPH window solve weekly conditional survival for exposure increments as in §2.7.3. After wave exit, hold wave exposure fixed and solve the conditional survival equation.")
s=s.replace("Estimate cohort-specific depletion geometry from prespecified quiet windows to obtain $\hat\theta_{0,d}$ and $\hat k_d$.","Initialize the cohort-specific baseline and frailty fit. If a nonlinear exponent is supplied, refit the quiet-window model consistently with that exponent.")
a=s.index("Table @tbl:neg_control_summary provides numeric summaries.");b=s.index("<!--",a)
s=s[:a]+"""Table @tbl:neg_control_summary reports the original v4.3 estimates and the jointly fitted all-window survivor-model estimates, using the same reporting date and input workbook. The configured Czech exponent 1.18 is a sensitivity value, not an identified parameter. Each of 999 person-bootstrap draws per enrollment reconstructs risk sets, refits each group, and recalculates pooling weights. All six sensitivity intervals include one; the 2022 widths decrease relative to the earlier first-window joint fitter, though they increase compared with the all-window gamma-working fit with exact inversion. The empirical age-shift contrasts have no known exact target; intervals including one do not establish equivalence. For the full survivor-model run all 18,000 contributing fits succeeded without insufficient-signal flags.

"""+s[b:]
rows=list(csv.DictReader((R/"objections/3.2/nonlinear-survivor-fit/table8-comparison.csv").open()))
vals={(x["enrollment"],x["pair"].replace("/"," vs ")):x for x in rows if x["fitter"]=="nonlinear_survivor_fit"}
a=s.index("Table: Example end-of-window");b=s.index("Table: Week-80",a)
block=s[a:b].replace("Joint fit [95% CI]","All-window survivor fit [95% CI]").replace("The joint-fit estimates and person-bootstrap intervals","The all-window survivor-model sensitivity estimates and person-bootstrap intervals")
for line in block.splitlines():
 if line.startswith("| 20"):
  cells=line.split("|");v=vals[(cells[1].strip(),cells[2].strip())]
  cells[-2]=f" {float(v['point']):.4f} [{float(v['lower95']):.3f}, {float(v['upper95']):.3f}] "
  block=block.replace(line,"|".join(cells))
s=s[:a]+block+s[b:]
# Explicitly distinguish prior evidence from new trial results.
a=s.index("#### 3.4.2 ");b=s.index("#### 3.4.3 ",a)
s=s[:a]+r"""#### 3.4.2 Empirical pooled Czech registry application

The historical illustrative pooled objectives (not supplied an external intervention-effect scale) had minima 1.190 and 1.185. Replaying the same data and objective with the all-window fit and with the nonlinear survivor fit refitted at each candidate exponent yields both minima at the lower boundary 1.00, on the original 1.00–1.30 grid. The wider diagnostic grid 0.70–1.30 gives both exact-survivor minima at its lower boundary. Forty-eight eligible cohort histories amount to 10.36 million overlapping enrollment memberships, but the positive raw-excess filter retains only 62 cohort-week rows across eight cohorts. The historical raw excess is the observed hazard difference from dose zero, not independently measured wave exposure.

![Comparison of historical and updated illustrative objective profiles; lower-boundary optima do not identify an exponent.](figures/fig_alpha_czech_objective.png){#fig:alpha_czech_objective}

"""+s[b:]
s=s.replace("Bootstrap resampling shows substantial dispersion","In the historical replay, bootstrap resampling shows substantial dispersion")
s=s.replace("We therefore report $\alpha$ as **not identified** for this dataset. The observed region near $\alpha \approx 1.18$--1.19 should be interpreted as a model-dependent preference rather than a reliably estimated parameter.","We therefore report $\alpha$ as **not identified**. The historical preference near 1.18–1.19 is not reproduced by the improved fitter. A value of 1.18 is retained only for sensitivity analyses; the primary correction is inactive absent an identified or externally specified exponent.")
s=s.replace("Refit $\hat{\theta}_{0,d}$ across all quiet windows","Jointly refit $\hat{k}_d,\hat{\theta}_{0,d}$ across all quiet windows")
# Add information results and measured finite-sample findings at uncertainty section.
a=s.index("### 2.9 ")
b=s.index("### 2.10 ",a)
info=r"""
#### Model-conditional information and precision

For integrated hazard $q_i$, conditional event probability $p_i=1-e^{-q_i}$ and expected risk set $Y_i$, the conditional binomial information for $\beta=(\log k,\theta,\text{unknown gap exposures})$ is
$$
I(\beta)=\sum_i Y_i\frac{1-p_i}{p_i}\nabla q_i\nabla q_i^T.
$$
The model-conditional regular variance bound for log KCOR follows by propagating the gradient through $I^{-1}$, including nuisance covariance. Include all quiet weeks and survival transitions across gaps, also before the first quiet window; treat alpha as fixed. This is not an individual-dataset CI or a boundary-valid coverage guarantee. In a contiguous low-mortality window of $L$ weeks with approximately constant weekly rate $k$, profiling the unknown baseline gives $\operatorname{SE}(\theta)\gtrsim\sqrt{12/[Nk^3 L(L^2-1)]}$. Low volume and short windows impede frailty resolution; exact multi-window calculations account for their timing and gaps. Frailty precision and endpoint precision need not coincide. In a matched short-window 50,000-person two-stratum simulation, the theoretical KCOR interval was [0.877,1.140]; the all-window fitter median person-bootstrap width was 0.315, compared with 213.056 for the first-window joint fit. The local approximation does not capture nonnegative boundary bias. Overlapping empirical strata require a dependence model for a pooled bound.

"""
s=s[:b]+info+s[b:]
a=s.index("Additional validation results—including")
s=s[:a]+"""**Population-size experiment.** Keeping the original contamination generator, timing and fifth accumulating-week reference, 200 simulations at each level for 50,000, 500,000 and 5,000,000 people per cohort used the all-window fit. Zero-addition mean KCOR was respectively 1.0494 [1.0305,1.0684], 1.0090 [1.0027,1.0153], and 1.0013 [0.9993,1.0033]. At 5M the smallest tested addition (0.0001) shifted the mean by +0.01969 [0.01698,0.02240]. These Monte Carlo intervals concern means, not individual endpoints. The nonnegative variance boundary and sampled reference contribute to the smaller population bias; this benchmark does not locate a universal contamination threshold. Earlier figures and the selected grid above refer to the historical smaller-population fitter.

"""+s[a:]
s=s.replace("Full grid in `KCOR critique/2.8/planned-anchor-grid.csv`","Full grid in `analysis/contamination/historical-grid.csv`")
put(p,s)
p=P/"supplement.md";s=p.read_text(encoding="utf-8-sig")
a=s.index("**Post-inversion correction.**");b=s.index("\n\n",a)
s=s[:a]+"""**Cumulative correction.** With a supplied nonlinear exponent, fit quiet mortality and invert the observed wave using the same survival law. Solve weekly survival for added exposure with the fitted baseline; when a week falls below that baseline, retain prior wave exposure and solve for a smaller proportional increment. At exit, hold the recovered wave exposure fixed. Ordinary gamma inversion is exact for alpha = 1. See main text §2.7.3 for the equation and limitation of attribution."""+s[b:]
s=s.replace("Seed Gompertz fit + delta iteration + pooled refit","All-window joint baseline and frailty fit with profiled gaps; nonlinear survivor law for supplied nonlinear exponent")
s=s.replace("The weekly prediction is defined in Step 1 of main-text §2.5.","The integrated weekly prediction is defined in main-text §2.5.")
s+="\n\n## S8. Joint-fit and nonlinear survivor-model validation\n\n"
s+="The main sampled nonlinear-wave experiment (100 independent datasets, 199 resamples) gives mean KCOR 1.00093 [0.99596,1.00591] with the consistent model, versus 0.98982 [0.98493,0.99471] with the gamma working fit and exact inversion; coverage is 90/100 [82.4%,95.1%] for both. In the stress design (50 datasets, 99 resamples), coverage is 48/50 [86.3%,99.5%]; mean with consistent model is 1.04019 [1.01789,1.06249]. In the noiseless labeled controls both original baseline and frailty are recovered. The proportional entry correction changes sampled mean from 1.20386 [1.19425,1.21347] to 0.99774 [0.99321,1.00228]. The actual pre-entry and wave allocation is unknown in registry controls. Earlier validation figures and tables not rerun under the updated estimator illustrate their stated historical configurations.\n"
put(p,s)
def cp(src,dst):
 q=D/dst;q.parent.mkdir(parents=True,exist_ok=True);q.unlink(missing_ok=True);shutil.copy2(R/src,q)
cp("objections/1.4/alpha-pooled-replay/pooled-alpha-objectives.png","documentation/preprint/figures/fig_alpha_czech_objective.png")
cp("objections/2.8/planned-anchor-grid.csv","analysis/contamination/historical-grid.csv")
for f in ("table8-comparison.csv","table8-draws.npz","main-full-results.json","stress-results.json","main-full-intervals.csv","stress-intervals.csv","mixed-results.json","workbook-verification.json"):
 cp("objections/3.2/nonlinear-survivor-fit/"+f,"analysis/survivor_validation/"+f)
for f in ("information-limits.csv","ratio-limits.csv","synthetic-pooled-comparison.csv","planning-grid.csv"):
 cp("objections/2.7/low-frailty/theoretical-limits/"+f,"analysis/information/"+f)
for f in ("runs.csv","README.md"):
 if f!="README.md":cp("objections/2.7/all-window-contamination/"+f,"analysis/contamination/"+f)
for f in ("summary.csv","wider-grid-summary.csv","objective-curves.csv"):
 cp("objections/1.4/alpha-pooled-replay/"+f,"analysis/alpha/"+f)
print("Paper and supplement updated; matched evidence copied.",flush=True)