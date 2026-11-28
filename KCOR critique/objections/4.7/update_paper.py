from pathlib import Path
import os,json,shutil,difflib
R=Path(__file__).resolve().parent
root=Path(os.environ["TEMP"])/"kcor-s4-approved"
D=root/"m100"
off=json.loads((R/"identified_only-table8.json").read_text())
on=json.loads((R/"forced_alpha-table8.json").read_text())
p=D/"documentation/preprint/paper.md";old=p.read_text(encoding="utf8");s=old
a=s.index("Table: Example end-of-window")
b=s.index("\n\nTable: Week-80",a)
block=s[a:b]
lines=block.splitlines()
lines[0]="Table: Empirical negative-control KCOR at December 26, 2022 (pooled/ASMR). Historical v4.3 estimates and analytical intervals are retained for comparison. The current primary analysis uses the all-window fit with the optional nonlinear module inactive; the sensitivity analysis uses the consistent survivor fit at forced alpha=1.18. Both current columns use 999 matched person-bootstrap draws per enrollment. Sources: `analysis/table8/identified_only-table8.json` and `analysis/table8/forced_alpha-table8.json`; historical source `test/negative_control/out/KCOR_summary.log`. {#tbl:neg_control_summary}"
lines[2]="| Enrollment | Dose comparison | Historical KCOR | Historical 95% CI | Core [95% CI] | Sensitivity 1.18 [95% CI] |"
lines[3]="|---|---|---:|---|---|---|"
for i,line in enumerate(lines):
 if not line.startswith("| 20"):continue
 cells=line.split("|");en=cells[1].strip();pair=cells[2].strip();j=("1 vs 0","2 vs 0","2 vs 1").index(pair)
 def fmt(r):
  q=r[en];return f"{q['point'][j]:.4f} [{q['lower'][j]:.3f}, {q['upper'][j]:.3f}]"
 cells[-2]=" "+fmt(off)+" ";cells.insert(-1," "+fmt(on)+" ")
 lines[i]="|".join(cells)
s=s[:a]+"\n".join(lines)+s[b:]
a=s.index("Table @tbl:neg_control_summary reports")
b=s.index("<!--",a)
old_para=s[a:b]
s=s[:a]+"""Table @tbl:neg_control_summary reports the historical v4.3 estimates, the current module-off primary estimates, and the forced-alpha sensitivity estimates, on the same input, reporting date and 999 matched person-bootstrap draws per enrollment. The configured Czech exponent 1.18 is a sensitivity value, not an identified parameter. The primary Czech configuration leaves the optional module inactive. The historical configuration documented a preference for the 2023 period because it showed the least vaccine harm; the chronology of that choice is unknown. The dates are retained, and the empirical illustrations should not be interpreted as outcome-independent validation of window selection. """+(
"All twelve current intervals include one. " if all(v["lower"][j]<=1<=v["upper"][j] for xx in (off,on) for v in xx.values() for j in range(3)) else ""
)+"""The empirical age-shift contrasts have no known exact target; intervals including one do not establish equivalence. """+(
"No draws had insufficient-signal flags or unsuccessful contributing fits in either run." if all(v["flagged_draws"]==v["failed"]==0 for xx in (off,on) for v in xx.values()) else "Fit flags and unsuccessful contributing fits are reported in the accompanying result files."
)+"\n\n"+s[b:]
p.write_text(s,encoding="utf8")
(R/"paper.patch").write_text("".join(difflib.unified_diff((root/"m99/documentation/preprint/paper.md").read_text(encoding="utf8").splitlines(True),s.splitlines(True),"a/documentation/preprint/paper.md","b/documentation/preprint/paper.md")),encoding="utf8")
for mode in ("identified_only","forced_alpha"):
 for suffix in ("-table8.json","-draws.npz"):
  shutil.copy2(R/(mode+suffix),D/"analysis/table8"/(mode+suffix))
print("Table 8 and narrative updated.")