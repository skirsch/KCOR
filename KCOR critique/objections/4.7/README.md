# 4.7 — Czech alpha configuration and reporting

**High for interpretation — continuing 1.4 with specific new config issue**

The Czech YAML has an identified_alpha field 1.18 as well as forced_alpha 1.18, contrary to non-identification; the comment about choosing 2023 for least vaccine harm warrants provenance clarification. The new module-off 999-draw Table 8 comparison is recorded; all six intervals contain one.

**Root cause / related objections:** Related 1.4 and 2.10. Do not infer actual chronology or causal effects from the comment. Alpha identification remains deferred.

**Source review:** [repeat-3/chain-current.md](<../../iterative review/evaluations/4/repeat-3/chain-current.md>) · [repeat-1/original-handoff.md](<../../iterative review/evaluations/4/repeat-1/original-handoff.md>)

[Series 4 comparison](<../../iterative review/evaluations/4/convergence.md>) · [Independent verification](<../../iterative review/evaluations/4/verification.md>) · [Counterexamples](<../../iterative review/evaluations/4/counterexamples.json>)

**Corrected in candidate m100.** Czech primary configuration uses identified_only, with identified_alpha null; forced_alpha remains 1.18 exclusively for sensitivity. KCOR_NPH_MODE=forced_alpha explicitly selects the existing sensitivity mode without changing the stored estimate. Both ordinary workbook paths confirmed and all six point estimates in each mode matched the matched replay to numerical precision. Full 999-draw person-bootstrap replays per enrollment were completed with the current fitter and identical seeds in both modes. All twelve current intervals contain one; zero flagged draws and unsuccessful fits. Table 8 and its narrative now distinguish the historical column, module-off primary column, and forced-alpha sensitivity column, with the actual result paths. The original Czech quiet-window dates remain unchanged; the historical outcome-related rationale is disclosed without inferring chronology. Further alpha identification remains deferred.

[Primary results](identified_only-table8.json) · [Sensitivity results](forced_alpha-table8.json) · [Matched replay](table8-replay.py) · [Configuration patch](Czech.yaml.patch) · [Paper patch](paper.patch). This record concerns the hypothetical integrated edition, not a claim that the actual draft PR has been merged.
