from pathlib import Path
import sys
p=Path(sys.argv[1])/'documentation/preprint/supplement.md'
s=p.read_text(encoding='utf8')
a='Across these strata, the trajectories remain qualitatively stable after depletion normalization, which is consistent with the observed behavior not being driven solely by age aggregation.'
b='The trajectories and pointwise intervals illustrate how the normalized contrasts vary across age strata under the current estimator; they are descriptive rather than a formal validation of stability.'
assert s.count(a)==1;s=s.replace(a,b)
a='A brief illustrative application is included to demonstrate end-to-end KCOR behavior on real registry mortality data in a setting that minimizes timing-driven shocks and window-tuning sensitivity. Cohorts were enrolled in ISO week 2022-26, and evaluation was restricted to calendar year 2023, yielding a 26-week post-enrollment buffer before slope estimation and a prespecified assessment period. Frailty parameters were estimated using the revised quiet-window settings and diagnostics described above to minimize wave-related hazard variation.'
b='A brief illustrative application demonstrates end-to-end KCOR behavior on real registry mortality data. Cohorts enrolled in ISO week 2022-26 are followed through 2024-41 (October 7, 2024); the displayed curves begin at the reference week after the two-week stabilization skip. Frailty parameters are estimated using the current all-window joint fitter, with the configured quiet windows within follow-up (2023-19 through 2023-40 and 2024-11 through 2024-14), early fitting anchors, and NPH module off. The figures in S5.6 and S6.1.3 use the same specification and current product-restricted CMR extract. Shading shows pointwise 95% person-bootstrap percentile intervals (999 draws, refitting each draw). All 12 point fits were numerically accepted; residual flags occurred for the 1940-1944 dose-3 cohort and the all-ages dose-0 and dose-3 cohorts. The trajectories and diagnostic results are archived in `analysis/s6-figures/`.'
assert s.count(a)==1;s=s.replace(a,b)
a='and evaluated over calendar year 2023.';b='and followed through October 7, 2024.'
assert s.count(a)==4;s=s.replace(a,b)
a='does not support causal inference. X-axis units are weeks since enrollment.';b='does not support causal inference. The horizontal axis shows calendar dates; shaded bands are pointwise 95% person-bootstrap intervals (999 draws).'
assert s.count(a)==4;s=s.replace(a,b)
p.write_text(s,encoding='utf8')
