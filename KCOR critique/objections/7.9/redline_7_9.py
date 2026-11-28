from pathlib import Path
import sys,difflib
old,new,red,out=map(Path,sys.argv[1:])
a=old.read_text(encoding='utf8');b=new.read_text(encoding='utf8');s=red.read_text(encoding='utf8')
sentence='The inclusion of early fitting-anchor weeks assumes that their mortality follows the baseline working model, even when they fall outside the designated quiet windows.'
anchor='early fitting-anchor rows used by the implementation.'
assert b==a.replace(anchor,anchor+' '+sentence,1)
assert anchor in s
s=s.replace(anchor,anchor+r' \textsuperscript{\hyperlink{issue-7.9}{7.9}} {\color{red} '+sentence+'}',1)
s=s.replace(r'\end{document}',r'\par\hypertarget{issue-7.9}{}\textbf{7.9} State the baseline-model assumption for early fitting-anchor weeks; no estimator change.'+'\n'+r'\end{document}')
out.write_text(s,encoding='utf8')
