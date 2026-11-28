from pathlib import Path
import sys,re
old,new,red,out=map(Path,sys.argv[1:])
a=old.read_text(encoding='utf8').splitlines();b=new.read_text(encoding='utf8').splitlines();s=red.read_text(encoding='utf8')
mark=r'\textsuperscript{\hyperlink{issue-8.4}{8.4}} '
def change(x,y):
 global s
 assert x in s,x[:110]
 if r"\(" in x:
  before,rest=x.split(r"\(",1)
  math,after=rest.split(r"\)",1)
  words=before.split()+[r"\("+math+r"\)"]+after.split()
 else: words=x.split()
 s=s.replace(x,mark+' '.join(r'\oldword{'+w+'}' for w in words)+r' {\color{red} '+y+'}',1)
change(r'and nonnegative \(\theta\); compare',r'and \(0\le\theta\le500\) by default; the ceiling is numerical, not a biological upper limit. Fits near the ceiling are flagged and the constrained value is retained for diagnostic output rather than reset to zero. Compare')
# Existing redline may already present the preceding wording as red text.
x=r'{\color{red} current}{\color{red} forced-alpha sensitivity estimates}'
assert x in s
s=s.replace(x,mark+r'\oldword{current forced-alpha sensitivity estimates} {\color{red} archived forced-alpha sensitivity estimates (previous numerical ceiling of 100)}',1)
change(r'KCOR does not impose \(\theta_0 = 0\) for any cohort.',r'KCOR does not fix \(\theta_0\) at zero a priori for any cohort; a flagged zero-variance fallback for insufficient or failed fitting data does not establish that the cohort is homogeneous.')
x=a[1761];y=b[1761];assert x in s
pos=s.index(x);start=s.rfind(r'\begin{longtable}',0,pos)
s=s[:start]+mark+r'{\color{red} Table 8 primary column refitted at ceiling 500; the existing forced-alpha sensitivity is labeled as the preceding ceiling-100 replay pending completion of the fresh replay.}'+'\n\n'+s[start:]
s=s.replace(x,y,1)
for x,y in [(r'63.87 {[}29.60, 96.51{]}',r'63.87 {[}29.60, 96.50{]}'),(r'Poorly resolved {[}0, 100{]}',r'Poorly resolved {[}0, 167.90{]}'),(r'1.507 {[}0.824, 2.712{]}',r'1.507 {[}0.960, 2.822{]}')]:change(x,y)
change('The 40--49 dose-2 fitted value is poorly resolved across the entire allowed range. KCOR intervals retain the existing upper-bound zero-variance fallback (287 of 999 dose-2 draws at ages 40--49).','The 40--49 dose-2 frailty remains poorly resolved. With the numerical ceiling increased to 500, none of the 999 draws reaches the upper-bound flag threshold in either dose group. The upper tail is no longer truncated at 100; the broad raw variance range is a statistical limitation, not evidence that variance is zero.')
change(r'\texttt{analysis/solver\_scaling/s6-all-window.json}',r'\texttt{analysis/ceiling500/s6-all-window.json}')
change('its matched draws; the earlier first-window study is archived separately.','its matched draws; the previous ceiling and first-window studies are archived separately.')
x=r'\texttt{analysis/README.md}). The historical S8 draws'
y=r'\texttt{analysis/README.md}). '+mark+r'{\color{red} The S6 frailty-resolution table and primary Table 8 column have been refitted at the numerical ceiling of 500; other stored empirical figures and the forced-alpha sensitivity column retain the preceding ceiling-100 replays.} The historical S8 draws'
assert x in s;s=s.replace(x,y,1)
s=s.replace(r'\end{document}',r'\par\hypertarget{issue-8.4}{}\textbf{8.4} Increase numerical ceiling to 500; flag and retain near-ceiling fits rather than treating proximity as zero frailty. Update verified S6 and primary Table 8 results.'+'\n'+r'\end{document}')
out.write_text(s,encoding='utf8')
