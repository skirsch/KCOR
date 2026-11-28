from pathlib import Path
import sys
r=Path(sys.argv[1])/'documentation/preprint'
def patch(file,pairs):
    p=r/file;s=p.read_text(encoding='utf8')
    for a,b in pairs:
        assert s.count(a)==1,a
        s=s.replace(a,b)
    p.write_text(s,encoding='utf8')
patch('paper.md',[
('When diagnostics fail, contrasts are not reported and results are treated as non-identifiable rather than as substantive cumulative effects.', 'When required diagnostics fail, substantive interpretability is not established; explicitly flagged numerical outputs may be retained for descriptive illustration or audit.'),
('Because KCOR defines explicit diagnostic failure modes'+chr(8212)+'instability, dose reversals, age incoherence, or absence of asymptotic convergence'+chr(8212)+'the absence of such failures in the Czech 2021_24 Dose 0 versus Dose 2 cohorts provides stronger validation than goodness-of-fit alone.', 'In the Czech 2021_24 Dose 0 versus Dose 2 example, all 14 subgroup fits were numerically accepted, but six triggered exploratory residual flags and window-omission sensitivity differed by group (SI S6.1.2). These checks illustrate diagnostic behavior rather than establish that all identification criteria were satisfied.')
])
patch('supplement.md',[
('Analysis treated as not identified; results not reported','Substantive interpretability not established; flagged outputs may be shown for audit'),
('When diagnostics indicate non-identifiability, the analysis is treated as not identified and results are not reported;', 'When diagnostics indicate non-identifiability, substantive interpretation is withheld; explicitly flagged numerical outputs may be retained for descriptive illustration or audit;'),
('The Czech results do not validate KCOR; they represent an application that satisfies all pre-specified diagnostic criteria.', 'The Czech example illustrates numerical fits and diagnostic outcomes; it does not establish that all pre-specified identification criteria were satisfied.'),
('Window-omission stability was not evaluated in this replay.', 'Omitting one configured quiet window at a time while retaining early fitting anchors yielded anchored KCOR point estimates of 1.336'+chr(8211)+'1.555 for ages 40'+chr(8211)+'49 and 1.115'+chr(8211)+'1.134 for all ages (full-window estimates and bootstrap intervals in Table @tbl:si_frailty_variance). These ranges describe point sensitivity, not confidence intervals. Group-specific results are archived in `analysis/s6/window-omission.csv`.')])

