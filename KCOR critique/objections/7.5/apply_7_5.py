"""Apply the approved manuscript-only population provenance correction."""
from pathlib import Path
import sys
root=Path(sys.argv[1])
def edit(path,replacements):
    s=path.read_text(encoding="utf8")
    for old,new in replacements:
        assert s.count(old)==1,(path,old,s.count(old))
        s=s.replace(old,new,1)
    path.write_text(s,encoding="utf8")
def main():
    p=root/"documentation/preprint/paper.md"
    edit(p,[
      ("The Czech application deliberately restricts its study population",
       "The age-shift figures and Table 8 use the archived September 2025 Czech registry extract (Git commit `bb2d6824`), which predates the introduction of the full-history vaccine-product eligibility restriction in November 2025. The separate Czech registry application in SI S6 deliberately restricts its study population"),
      ("Data source: Czech registry age-shift control. X-axis:", "Data source: archived September 2025 Czech registry control, before the product restriction. X-axis:"),
      ("these groups overlap). Historical v4.3 estimates", "these groups overlap). The input is the September 2025 extract, before the product restriction. Historical v4.3 estimates"),
    ])
    p=root/"documentation/preprint/supplement.md"
    edit(p,[
      ("complete strata within the restricted study population rather than selectively sampled subcohorts",
       "complete strata within the archived September 2025 extract, before the product restriction used for SI S6, rather than selectively sampled subcohorts"),
      ("Data source: Czech registry age-shift control. X-axis:", "Data source: archived September 2025 Czech registry control, before the product restriction. X-axis:"),
      ("**Czech population eligibility.** The converter", "**Czech population eligibility for the S6 application.** Unlike the archived age-shift controls and Table 8, this application uses the current product-restricted CMR extract. The converter"),
    ])
if __name__=="__main__":main()
