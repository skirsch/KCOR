from pathlib import Path
import sys
old,new,red,out=map(Path,sys.argv[1:])
s=red.read_text(encoding='utf8')
mark=r'\textsuperscript{\hyperlink{issue-7.11}{7.11}} '
def strike(x):return ' '.join(r'\oldword{'+w+'}' for w in x.split())
def change(a,b):
 global s
 assert s.count(a)==1,a
 s=s.replace(a,mark+strike(a)+r' {\color{red} '+b+'}',1)
change('biasing standard survival estimands','which can distort fixed-frailty contrasts inferred from conventional survival summaries')
a=r'\textbf{depletion-adjusted marginal estimand}'
b=r'\textbf{depletion-adjusted cohort-level estimand on the reference-frailty scale}'
assert s.count(a)==1
s=s.replace(a,mark+r'\oldword{\textbf{depletion-adjusted marginal estimand}} {\color{red} '+b+'}',1)
a=r"cohort \(d\)'s \textbf{depletion-neutralized baseline cumulative hazard} obtained"
assert s.count(a)==1
s=s.replace(a,a[:-len('obtained')]+mark+r'{\color{red} at reference frailty \(Z=1\) under the mean-one working model,} obtained',1)
a='Conditions on survival & Yes & Yes & No'
assert s.count(a)==1
s=s.replace(a,mark+strike('Conditions on survival')+r' {\color{red} Final hazard target conditional on current survival} & Yes & Yes & '+strike('No')+r' {\color{red} No (reference-frailty cumulative contrast)}',1)
change('Note: KCOR is reported here as a cumulative hazard ratio for comparability;','Note: KCOR uses hazards calculated in surviving risk sets as input, but its final target is a cumulative contrast on the reference-frailty scale.')
s=s.replace(r'\end{document}',r'\par\hypertarget{issue-7.11}{}\textbf{7.11} Qualify survival-estimand comparisons and distinguish survivor-risk-set input from the reference-frailty cumulative target.'+'\n'+r'\end{document}')
out.write_text(s,encoding='utf8')
