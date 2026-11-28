from pathlib import Path
import sys,difflib
old,new,red,out=map(Path,sys.argv[1:])
a=old.read_text(encoding='utf8').splitlines();b=new.read_text(encoding='utf8').splitlines();s=red.read_text(encoding='utf8')
mark=r'\textsuperscript{\hyperlink{issue-7.6}{7.6}} '
for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
 if tag=='equal':continue
 assert j-i==l-k==1
 x,y=a[i].split(),b[k].split()
 pre=0
 while pre<min(len(x),len(y)) and x[pre]==y[pre]:pre+=1
 suf=0
 while suf<min(len(x)-pre,len(y)-pre) and x[-1-suf]==y[-1-suf]:suf+=1
 oldpart=' '.join(x[pre:len(x)-suf]);newpart=' '.join(y[pre:len(y)-suf])
 assert oldpart in s,oldpart
 strike=' '.join(r'\oldword{'+w+'}' for w in oldpart.split())
 s=s.replace(oldpart,mark+strike+r' {\color{red} '+newpart+'}',1)
s=s.replace(r'\end{document}',r'\par\hypertarget{issue-7.6}{}\textbf{7.6} Report measured diagnostic outcomes and window-omission sensitivity; distinguish numerical acceptance from substantive identification.'+'\n'+r'\end{document}')
out.write_text(s,encoding='utf8')
