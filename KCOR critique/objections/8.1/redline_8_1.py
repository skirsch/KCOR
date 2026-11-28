from pathlib import Path
import sys
old,new,red,out=map(Path,sys.argv[1:])
a=old.read_text(encoding='utf8');b=new.read_text(encoding='utf8');s=red.read_text(encoding='utf8')
sentence='Source birth-year, sex and comorbidity components are checked before aggregation; a truncated component with people remaining at risk is incomplete, not exhausted.'
anchor='The workbook records the normalization status, and affected pooled contrasts are not silently reweighted.'
assert b==a.replace(anchor,sentence+' '+anchor,1)
assert anchor in s
s=s.replace(anchor,r'\textsuperscript{\hyperlink{issue-8.1}{8.1}} {\color{red} '+sentence+'} '+anchor,1)
s=s.replace(r'\end{document}',r'\par\hypertarget{issue-8.1}{}\textbf{8.1} Validate source fixed-cohort histories before aggregation; distinguish missing history from true exhaustion.'+'\n'+r'\end{document}')
out.write_text(s,encoding='utf8')
