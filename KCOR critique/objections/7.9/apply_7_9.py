from pathlib import Path
import sys
p=Path(sys.argv[1])/'documentation/preprint/paper.md'
s=p.read_text(encoding='utf8')
a=r'Here $\mathcal Q_d$ includes the designated quiet and early fitting-anchor rows used by the implementation.'
b=a+' The inclusion of early fitting-anchor weeks assumes that their mortality follows the baseline working model, even when they fall outside the designated quiet windows.'
assert s.count(a)==1
p.write_text(s.replace(a,b),encoding='utf8')
