from pathlib import Path
import sys,difflib
old,new,red,out=map(Path,sys.argv[1:])
a=old.read_text(encoding='utf8').splitlines();b=new.read_text(encoding='utf8').splitlines();s=red.read_text(encoding='utf8')
ids={307:'8.12',311:'8.12',417:'8.9',556:'8.11',905:'8.10',1196:'8.9',2028:'8.9',2034:'8.9',2546:'8.13'}
def mark(n):return r'\textsuperscript{\hyperlink{issue-'+n+'}{'+n+'}} '
def strike(s):return ' '.join(r'\oldword{'+w+'}' for w in s.split())
def change(x,y,n):
 global s
 assert x in s,x[:100]
 s=s.replace(x,mark(n)+strike(x)+r' {\color{red} '+y+'}',1)
for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
 if tag=='equal':continue
 n=ids[i];assert j-i==l-k==1
 if i==905:
  change(a[i].split(' The manuscript')[0],b[k].split(' The manuscript')[0],n)
  continue
 x,y=a[i].split(),b[k].split()
 pre=0
 while pre<min(len(x),len(y)) and x[pre]==y[pre]:pre+=1
 suf=0
 while suf<min(len(x)-pre,len(y)-pre) and x[-1-suf]==y[-1-suf]:suf+=1
 change(' '.join(x[pre:len(x)-suf]),' '.join(y[pre:len(y)-suf]),n)
for n,desc in [('8.9','Consistent diagnostic interpretation and clear decision-table logic.'),('8.10','Clarify actual reproduction scope; fresh versus archived replay.'),('8.11','Consistent early fitting-anchor definition.'),('8.12','Qualify survival-estimand benchmark.'),('8.13','Correct zero-frailty Gompertz interpretation.')]:
 s=s.replace(r'\end{document}',r'\par\hypertarget{issue-'+n+r'}{}\textbf{'+n+'} '+desc+'\n'+r'\end{document}')
out.write_text(s,encoding='utf8')
