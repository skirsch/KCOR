from pathlib import Path
import sys,difflib
old,new,red,out=map(Path,sys.argv[1:])
a=old.read_text(encoding='utf8').splitlines();b=new.read_text(encoding='utf8').splitlines();s=red.read_text(encoding='utf8')
def mark(n):return r'\textsuperscript{\hyperlink{issue-'+n+'}{'+n+'}} '
def strike(x):return ' '.join(r'\oldword{'+w+'}' for w in x.split())
for tag,i,j,k,l in difflib.SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
 if tag=='equal':continue
 if j-i==l-k==1:
  x,y=a[i],b[k];assert x in s,x[:100]
  # The original sentence contains math rendered as inline LaTeX; use a single balanced strike.
  s=s.replace(x,mark('8.3')+mark('8.8')+r'\oldword{'+x+r'} {\color{red} '+y+'}',1)
 elif j-i==l-k==2:
  for x,y in zip(a[i:j],b[k:l]):
   assert x in s,x[:100]
   s=s.replace(x,y,1)
  pos=s.index(b[k]);start=s.rfind(r'\begin{figure}',0,pos)
  note=mark('8.8')+strike('Across all comparisons, KCOR varies smoothly and modestly with no qualitative changes in sign or magnitude.')+r' {\color{red} Regenerated using the corrected ordinary all-ages path and reported actual sensitivity; common color scale.}'+'\n\n'
  s=s[:start]+note+s[start:]
 else:raise ValueError((i,j,k,l))
for n,desc in [('8.3','Use the ordinary all-ages population, calendar, fitter and normalization in sensitivity analysis.'),('8.8','Regenerate the selected sensitivity figure with a genuinely shared color scale and report the full configured grid results.')]:
 s=s.replace(r'\end{document}',r'\par\hypertarget{issue-'+n+r'}{}\textbf{'+n+'} '+desc+'\n'+r'\end{document}')
out.write_text(s,encoding='utf8')
