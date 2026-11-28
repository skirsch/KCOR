from pathlib import Path
import sys
r=Path(sys.argv[1])/'documentation/preprint'
def edit(name,pairs):
 p=r/name;s=p.read_text(encoding='utf8')
 for a,b in pairs:
  assert s.count(a)==1,a[:100]
  s=s.replace(a,b)
 p.write_text(s,encoding='utf8')
edit('paper.md',[
(r'while $\alpha$ is not separately identifiable from cohort-specific multiplicative effects if that external scale is not fixed.',r'while the proposed $\alpha$-localization procedure requires that external scale to distinguish frailty-dependent amplification from cohort-specific multiplicative effects.'),
(r'The optional NPH exponent $\alpha$ is not separately identifiable from cohort-specific multiplicative structure in minimal aggregated data',r'The optional NPH localization procedure requires a specified intervention-effect scale to separate $\alpha$ from cohort-specific multiplicative structure'),
(r'Because $\alpha$ is not separately identifiable from cohort-specific multiplicative intervention effects in minimal aggregated data, localizing $\alpha$ with this module requires',r'Because a single cross-cohort excess-hazard contrast can confound $\alpha$ with an unknown intervention multiplier, the proposed localization procedure requires'),
(r'Under minimal aggregated data, the NPH exponent $\alpha$ is not separately identifiable from cohort-specific multiplicative intervention effects if both are left free. Accordingly, when the optional NPH module is used to localize $\alpha$,',r'The proposed localization procedure does not jointly estimate the NPH exponent $\alpha$ and cohort-specific intervention multipliers. When this procedure is used,'),
(r'KCOR does not identify that scale from the same aggregated wave-period contrasts used to localize $\alpha$.' ,r'KCOR does not identify that scale from the same aggregated wave-period contrasts used to localize $\alpha$. This is a requirement of the procedure, not a universal non-identifiability theorem: with known baseline and frailty parameters, a known common wave exposure and a constant unknown multiplier, repeated survivor observations can in some cases locally identify both parameters. Joint longitudinal localization is not evaluated here.'),
(r'Identification of $\alpha$ requires sufficient cross-cohort variation',r'In the proposed localization procedure, identification of $\alpha$ requires sufficient cross-cohort variation'),
(r'In minimal aggregated data, the NPH exponent $\alpha$ is not separately identifiable from cohort-specific multiplicative effects (e.g., intervention-associated hazard reduction).',r'Without a specified effect scale, the proposed localization procedure does not separate the NPH exponent $\alpha$ from cohort-specific multiplicative effects (e.g., intervention-associated hazard reduction).')
])
edit('supplement.md',[
(r'When the optional NPH module is used, identifiability of $\alpha$ is likewise conditional:',r'When the proposed optional NPH localization procedure is used, identifiability of $\alpha$ is likewise conditional:'),
(r'$\alpha$ is not separately identifiable from cohort-specific multiplicative effects under minimal data if that scale is not fixed externally',r'the procedure does not jointly identify $\alpha$ and cohort-specific multiplicative effects if that scale is not fixed externally'),
(r'Under minimal aggregated data, $\alpha$ is not separately identifiable from cohort-specific multiplicative intervention effects unless **externally supplied VE or other prespecified intervention-effect scale** is fixed and not estimated from the same wave-period contrasts used to localize $\alpha$',r'The proposed localization procedure requires **externally supplied VE or other prespecified intervention-effect scale**, fixed rather than estimated jointly with $\alpha$ from the wave-period contrasts')
])
