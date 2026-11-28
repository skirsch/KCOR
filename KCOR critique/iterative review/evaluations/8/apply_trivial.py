from pathlib import Path
import sys
r=Path(sys.argv[1]);p=r/'documentation/preprint/paper.md';s=p.read_text(encoding='utf8')
pairs=[
('When diagnostics indicate non-identifiability, the analysis is treated as not identified and results are not reported. Checks include:', 'When required diagnostics indicate non-identifiability, substantive interpretation is withheld; explicitly flagged numerical outputs may be retained for descriptive illustration or audit. Checks include:'),
('if diagnostics fail, contrasts are not reported.', 'if required diagnostics fail, substantive interpretation is withheld. Explicitly flagged numerical outputs may be retained for audit, while undefined computations remain unavailable.'),
('All figures, tables, and simulations can be reproduced from the accompanying code repository. The manuscript is built from `documentation/preprint/paper.md` using the root `Makefile` build target `make paper-pdf`.','The accompanying repository provides the manuscript, figure assets, executable analyses and archived results. The reproduction scope of each analysis, including historical studies verified from stored draws, is described in SI S7 and `analysis/README.md`. The manuscript is built from `documentation/preprint/paper.md` using the root `Makefile` build target `make paper-pdf`.'),
('Frailty parameters are identified using only bins whose corresponding calendar weeks lie inside prespecified quiet windows (defined in ISO-week space).','Frailty parameters are fitted using bins within prespecified calendar quiet windows (ISO-week space) and the early post-skip fitting-anchor weeks, which are assumed to follow the baseline working model (section 2.5). Marked wave rows are excluded from quiet residuals when the optional wave model is active.'),
('curvature that can bias common survival estimands;','curvature that can distort fixed-frailty contrasts inferred from conventional survival summaries;'),
('but as a prerequisite normalization step that addresses a source of bias arising prior to model fitting in many retrospective cohort studies.','but as a normalization step for studies targeting depletion-adjusted cumulative contrasts, rather than the observed marginal survival itself.')
]
for a,b in pairs:
 assert s.count(a)==1,a
 s=s.replace(a,b)
p.write_text(s,encoding='utf8')
p=r/'documentation/preprint/supplement.md';s=p.read_text(encoding='utf8')
pairs=[
('| I1 Diagnostic sufficiency | All required diagnostics pass. | KCOR interpretable |','| I1 Diagnostic sufficiency | All required diagnostics pass. | Substantive interpretability not established |'),
('| I7 Conservative failure rule | A required diagnostic fails. | Substantive interpretability not established; flagged outputs may be shown for audit |','| I7 Conservative failure rule | Required identification diagnostics are satisfied. | Substantive interpretability not established; flagged outputs may be shown for audit |'),
('Near-zero values indicate weak identifiability or effectively linear cumulative hazards under the working model.','Near-zero values indicate little fitted frailty-induced depletion; precision and identification must be assessed separately. At zero variance, the working-model cumulative hazard follows the Gompertz baseline.')
]
for a,b in pairs:
 assert s.count(a)==1,a
 s=s.replace(a,b)
p.write_text(s,encoding='utf8')
p=r/'code/KCOR.py';s=p.read_text(encoding='utf8')
start=s.index('            # Pool only over valid, finite entries; ignore NaNs/inf entirely')
end=s.index('    # -------- All Ages ASMR rows',start)
block=s[start:end]
block=block.replace('                # Create ASMR row for this date if we have valid pooling data','            # Create ASMR row for this date if we have valid pooling data',1)
s=s[:start]+''.join('    '+line if line.strip() else line for line in block.splitlines(keepends=True))+s[end:]
start=s.index('            # If no ASMR rows at exact report_date, find closest ASMR date and include those rows')
end=s.index('            # Get dose pairs for this specific sheet',start)
s=s[:start]+'            # Never substitute a pooled estimate from a different reporting date.\n\n'+s[end:]
p.write_text(s,encoding='utf8')
