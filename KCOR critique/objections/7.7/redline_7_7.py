from pathlib import Path
import sys,difflib
old,new,red,out=map(Path,sys.argv[1:])
a=old.read_text(encoding='utf8').splitlines();b=new.read_text(encoding='utf8').splitlines();s=red.read_text(encoding='utf8')
mark=r'\textsuperscript{\hyperlink{issue-7.7}{7.7}} '
def strike(x):return ' '.join(r'\oldword{'+w+'}' for w in x.split())
for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
 if tag=='equal':continue
 if j-i==2:
  for x,y in zip(a[i:j],b[k:l]): 
   assert x in s,x[:100]
   s=s.replace(x,y,1)
  pos=s.index(b[k]);start=s.rfind(r'\begin{figure}',0,pos)
  note=mark+strike('Evaluated over calendar year 2023. X-axis units are weeks since enrollment.')+r' {\color{red} Regenerated with the current fitter through October 7, 2024; calendar dates and pointwise 95\% person-bootstrap intervals.}'+'\n\n'
  s=s[:start]+note+s[start:]
 else:
  assert j-i==l-k==1
  x,y=a[i].split(),b[k].split()
  pre=0
  while pre<min(len(x),len(y)) and x[pre]==y[pre]:pre+=1
  suf=0
  while suf<min(len(x)-pre,len(y)-pre) and x[-1-suf]==y[-1-suf]:suf+=1
  op=' '.join(x[pre:len(x)-suf]);np=' '.join(y[pre:len(y)-suf])
  assert op in s,op[:100]
  s=s.replace(op,mark+strike(op)+r' {\color{red} '+np+'}',1)
s=s.replace(r'\end{document}',r'\par\hypertarget{issue-7.7}{}\textbf{7.7} Replace four historical Czech plots with reproducible current-fitter trajectories, correct dates and include bootstrap uncertainty.'+'\n'+r'\end{document}')
out.write_text(s,encoding='utf8')
