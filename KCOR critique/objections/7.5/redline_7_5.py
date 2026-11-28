"""Keep prior cumulative redlines; link each 7.5 manuscript change."""
from pathlib import Path
import sys,re
old=Path(sys.argv[1]).read_text(encoding="utf8")
new=Path(sys.argv[2]).read_text(encoding="utf8")
red=Path(sys.argv[3]).read_text(encoding="utf8")
out=Path(sys.argv[4])
mark=r"\textsuperscript{\hyperlink{issue-7.5}{7.5}} "
def replace(a,b):
    global red
    assert a in red,a[:100]
    red=red.replace(a,b,1)
def strike(s):
    return " ".join(r"\oldword{"+w+"}" for w in s.split())
a="The Czech application deliberately restricts its study population"
b=next(x for x in new.splitlines() if x.startswith("The age-shift figures and Table 8 use"))
prefix=b[:b.index(" to individuals with")]
replace(a,mark+strike(a)+r" {\color{red} "+prefix+"}")
a="complete strata within the restricted study population rather than selectively sampled subcohorts"
b="complete strata within the archived September 2025 extract, before the product restriction used for SI S6, rather than selectively sampled subcohorts"
replace(a,mark+strike(a)+r" {\color{red} "+b+"}")
a=r"\textbf{Czech population eligibility.} The converter"
b=r"\textbf{Czech population eligibility for the S6 application.} Unlike the archived age-shift controls and Table 8, this application uses the current product-restricted CMR extract. The converter"
replace(a,mark+r"\oldword{\textbf{Czech population eligibility.}} {\color{red} "+b+"}")
# Moving captions stay clean; the tracked change is immediately beside the figure.
for label,name in [("fig:neg_control_10yr","fig2_neg_control_10yr_age_diff.png"),("fig:neg_control_20yr","fig3_neg_control_20yr_age_diff.png")]:
    x=next(l for l in old.splitlines() if l.endswith(r"\label{"+label+"}"))
    y=next(l for l in new.splitlines() if l.endswith(r"\label{"+label+"}"))
    pos=red.index(x);start=red.rfind(r"\begin{figure}",0,pos)
    note=mark+strike("Data source: Czech registry age-shift control.")+r" {\color{red} Data source: archived September 2025 Czech registry control, before the product restriction.}"+"\n\n"
    red=red[:start]+note+red[start:]
    replace(x,y)
    x=next(l for l in old.splitlines() if "includegraphics" in l and name in l)
    y=next(l for l in new.splitlines() if "includegraphics" in l and name in l)
    replace(x,y)
x=next(l for l in old.splitlines() if l.endswith(r"\label{tbl:neg_control_summary}\tabularnewline"))
y=next(l for l in new.splitlines() if l.endswith(r"\label{tbl:neg_control_summary}\tabularnewline"))
pos=red.index(x);start=red.rfind(r"\begin{longtable}",0,pos)
red=red[:start]+mark+r"{\color{red} The input is the September 2025 extract, before the product restriction.}"+"\n\n"+red[start:]
replace(x,y)
red=red.replace(r"\end{document}",r"\par\hypertarget{issue-7.5}{}\textbf{7.5} Identify the archived pre-restriction age-shift extract separately from the current restricted Czech S6 application; neither analysis changes."+"\n"+r"\end{document}")
out.write_text(red,encoding="utf8")
