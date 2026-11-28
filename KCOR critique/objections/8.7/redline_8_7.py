"""Cumulative redline for validation evidence and completed ceiling-500 refresh."""
from pathlib import Path
import re,sys
old,new,red,out=map(Path,sys.argv[1:])
a=old.read_text(encoding='utf8');b=new.read_text(encoding='utf8');s=red.read_text(encoding='utf8')
def mark(n='8.7'):return r'\textsuperscript{\hyperlink{issue-'+n+'}{'+n+'}} '
def strike(x):
    return ' '.join(r'\oldword{'+w+'}' for w in re.findall(r'\\\(.*?\\\)|\\textbf\{[^{}]*\}|\\texttt\{[^{}]*\}|\S+',x))
def change(x,y,n='8.7'):
 global s
 assert x in s,x[:100]
 s=s.replace(x,mark(n)+strike(x)+r' {\color{red} '+y+'}',1)
# The old 3.3.1 section has no previous edits. Strike its text, keeping list syntax intact.
start='To assess robustness to departures from the gamma frailty assumption,'
end=r'\paragraph{3.3.2 Quiet-window contamination stress test}'
i=a.index(start);j=a.index(end,i)
oldsec=a[i:j];assert oldsec in s
ni=b.index(start);nj=b.index(end,ni);newsec=b[ni:nj]
struck=[]
for line in oldsec.splitlines():
 if not line.strip() or line.strip() in (r'\begin{itemize}',r'\end{itemize}',r'\tightlist',r'\item'):struck.append(line)
 else:struck.append(strike(line))
s=s.replace(oldsec,mark()+ '\n'.join(struck)+'\n\n'+r'{\color{red} '+newsec.strip()+'}\n\n',1)
change(r'This setting stress-tests whether the revised \(\theta_{0,d}\) estimator and depletion normalization remain effective when frailty heterogeneity is concentrated in the tails rather than smoothly distributed.','This historical single-quiet-window experiment stress-tests depletion normalization when frailty heterogeneity is concentrated in the tails rather than smoothly distributed.')
change('The evaluation includes (i) KCOR drift, (ii) fit quality under the revised estimator, (iii) post-normalization linearity, and (iv) parameter stability under window perturbation.','The archived results report KCOR drift, fitted parameters, cumulative-hazard fit error and linearity diagnostics under the earlier fitter. They do not provide replicated coverage or parameter-perturbation stability under the current estimator; the matched current-fitter frailty study is in SI S9.')
# Remove the temporary earlier-ceiling caveat now that forced alpha is fully refitted.
x=mark('8.4')+r'\oldword{current forced-alpha sensitivity estimates} {\color{red} archived forced-alpha sensitivity estimates (previous numerical ceiling of 100)}'
assert x in s;s=s.replace(x,r'{\color{red} current forced-alpha sensitivity estimates}',1)
x=next(z for z in a.splitlines() if z.endswith(r'\label{tbl:neg_control_summary}\tabularnewline'))
y=next(z for z in b.splitlines() if z.endswith(r'\label{tbl:neg_control_summary}\tabularnewline'))
assert x in s;s=s.replace(x,y,1)
x='Table 8 primary column refitted at ceiling 500; the existing forced-alpha sensitivity is labeled as the preceding ceiling-100 replay pending completion of the fresh replay.'
assert x in s;s=s.replace(x,'Both Table 8 columns have been refitted at ceiling 500, with all printed estimates and intervals unchanged.',1)
i=s.index(mark('8.4')+r'{\color{red} The S6 ',s.index('Scientific analyses use their separate runners'))
j=s.index(' The historical S8 draws',i)
s=s[:i]+mark('8.4')+r'{\color{red} The S6 table and trajectories, both Table 8 columns, age-shift figures, quiet-window scan and sensitivity grid have been refitted with numerical ceiling 500.}'+s[j:]
# Append the new supplemental validation section before the objection index.
i=b.index(r'\subsection{S9. Matched frailty-misspecification validation}')
j=b.index(r'\end{document}',i)
section=b[i:j].strip()
idx=s.index(r'\clearpage\section*{Objections referenced in this redline}')
s=s[:idx]+mark()+r'\begingroup\color{red}'+'\n'+section+'\n'+r'\endgroup'+'\n\n'+s[idx:]
s=s.replace(r'\end{document}',r'\par\hypertarget{issue-8.7}{}\textbf{8.7} Complete the matched five-structure validation and report actual sampling variance, coverage and diagnostic outcomes; preserve historical tail evidence.'+'\n'+r'\end{document}')
out.write_text(s,encoding='utf8')
