# Round 10 post-review revisions

- m152 frozen evaluated checkpoint preserved.
- m153 straightforward approved batch: 10.1, 10.2, 10.5, 10.6, 10.7, 10.9, 10.10. [Patches and verification](post-review-batch-1/). All three implementation regressions passed; all 12 Table 8 ordinary points match to 2.23e-16 and archived diagnostic/interval arrays remain byte-identical. No full refit needed because the published positive-risk histories, default grouping and estimator unchanged.
- Remaining for individual discussion: 10.3 seed ceiling and diagnostic adequacy, 10.4 actual gap indexing, 10.8 pooling-weight definition/intent.

- m154: 10.3 implemented; shared configured seed ceiling and unavailable/bounded seed diagnostic status. Primary Table 8 999-draw replay: printed estimates and intervals unchanged, 30 gap intervals assessed without deficit, 24 unassessed. Forced-alpha smoke 49 draws; full forced-alpha and broader archive refresh pending. 10.4 and 10.8 remain for discussion.
  S6 999-draw replay confirms identical printed values and intervals.

- m155: 10.4 implemented; baseline-anchor/quiet union defines actual gap transitions, seed and quiet-death eligibility unchanged. Adjacent/overlapping and genuine first/internal gap regressions pass, as does 10.3. Table 8 and S6 configured re-entry indices unchanged. 10.8 remains for discussion.

- m156: 10.8 documentation only; verified original death-burden weighting intent, specified rate averaging, window, fallback and actual Table 8 weights. No estimator change. All ten round-10 objections now have an implemented response; independent assessment remains pending.
