from pathlib import Path
import sys
p=Path(sys.argv[1])/'documentation/preprint/paper.md';s=p.read_text(encoding='utf8')
pairs=[
(r'In the KCOR framework, this manifests as differing enrollment-time frailty variance $\theta_{0,d}$ and is the primary target of frailty normalization.',r'In the KCOR framework, differences in relative frailty heterogeneity manifest as differing enrollment-time frailty variance $\theta_{0,d}$ and are the primary target of frailty normalization. Differences in mean frailty may instead enter the cohort-specific reference baseline rate and need not change $\theta_{0,d}$.'),
(r'at reference frailty $Z=1$ under the mean-one working model, obtained',r'at reference frailty $Z=1$ (the mean frailty within cohort $d$ at the rebased post-skip origin) under the mean-one working model, obtained'),
('it highlights residual differences that cannot be attributed to frailty-driven selection under the working model, flagging regimes that demand substantive explanation beyond selection artifacts.','it highlights residual differences after adjustment for modeled depletion geometry. Selection differences that change the cohort-specific mean frailty may remain in the reference baseline rates.'),
('This choice is intentional: cumulative outcome accumulation is invariant to transient hazard-ratio oscillations induced purely by depletion geometry.','This choice is intentional: under the working model, frailty inversion removes depletion-induced curvature from the cumulative comparison; accumulation alone does not remove it.'),
('under a true null effect, KCOR should remain approximately flat at 1 after normalization, subject to sampling variability and model adequacy;', 'under equal post-skip reference cumulative hazards, unanchored KCOR should remain approximately flat at 1; for proportional reference cumulative hazards, anchored KCOR should remain approximately flat at 1, subject to sampling variability and model adequacy. Equal intrinsic hazards before the skip need not imply equal post-skip reference hazards;'),
('The synthetic negative control also checks whether KCOR remains near 1 under a true null effect.','The synthetic negative control also checks whether KCOR remains near 1 under its stated normalized null benchmark.')
]
for a,b in pairs:
 assert s.count(a)==1,a
 s=s.replace(a,b)
p.write_text(s,encoding='utf8')
