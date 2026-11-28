from pathlib import Path
import sys
p=Path(sys.argv[1])/'documentation/preprint/paper.md'
s=p.read_text(encoding='utf8')
pairs=[
('biasing standard survival estimands in retrospective cohort studies using registry and administrative data.', 'which can distort fixed-frailty contrasts inferred from conventional survival summaries in retrospective cohort studies using registry and administrative data.'),
('it is a **depletion-adjusted marginal estimand**, defined as','it is a **depletion-adjusted cohort-level estimand on the reference-frailty scale**, defined as'),
('where $\\tilde H_{0,d}(t)$ is cohort $d$\'s **depletion-neutralized baseline cumulative hazard** obtained','where $\\tilde H_{0,d}(t)$ is cohort $d$\'s **depletion-neutralized baseline cumulative hazard** at reference frailty $Z=1$ under the mean-one working model, obtained'),
('| Conditions on survival              | Yes          | Yes               | No                      |','| Final hazard target conditional on current survival | Yes | Yes | No (reference-frailty cumulative contrast) |'),
('Note: KCOR is reported here as a cumulative hazard ratio for comparability; alternative post-normalization estimands are admissible within the framework.','Note: KCOR uses hazards calculated in surviving risk sets as input, but its final target is a cumulative contrast on the reference-frailty scale. Alternative post-normalization estimands are admissible within the framework.')
]
for a,b in pairs:
 assert s.count(a)==1,a
 s=s.replace(a,b)
p.write_text(s,encoding='utf8')
