from pathlib import Path
import sys,re,difflib
old,new,red,out=map(Path,sys.argv[1:])
a=old.read_text(encoding='utf8').splitlines();b=new.read_text(encoding='utf8').splitlines();s=red.read_text(encoding='utf8')
mark=r'\textsuperscript{\hyperlink{issue-8.6}{8.6}} '
def strike(x):return ' '.join(r'\oldword{'+w+'}' for w in re.findall(r'\\\(.*?\\\)|\S+',x))
for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
 if tag=='equal':continue
 assert j-i==l-k==1
 if i==321:
  x=r'at reference frailty \(Z=1\) under the mean-one working model,'
  assert x in s
  s=s.replace(x,r'at reference frailty \(Z=1\) '+mark+r'{\color{red} (the mean frailty within cohort \(d\) at the rebased post-skip origin)} under the mean-one working model,',1)
  continue
 x,y=a[i].split(),b[k].split();pre=0
 while pre<min(len(x),len(y)) and x[pre]==y[pre]:pre+=1
 suf=0
 while suf<min(len(x)-pre,len(y)-pre) and x[-1-suf]==y[-1-suf]:suf+=1
 op=' '.join(x[pre:len(x)-suf]);np=' '.join(y[pre:len(y)-suf])
 assert op and op in s,op
 s=s.replace(op,mark+strike(op)+r' {\color{red} '+np+'}',1)
s=s.replace(r'\end{document}',r'\par\hypertarget{issue-8.6}{}\textbf{8.6} Clarify the cohort-relative mean-one reference after skip and the distinct anchored/unanchored null benchmarks; no estimator change.'+'\n'+r'\end{document}')
out.write_text(s,encoding='utf8')
