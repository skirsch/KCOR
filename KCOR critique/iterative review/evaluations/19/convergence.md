# Round 19 - m183 (review interrupted by usage limit)

Two fresh blind and one targeted GPT-6.2 Astra High reviews used the same importance-focused question. Both blind reports were saved before the agents reached their usage limit; the targeted reviewer sent substantive progress but did not deliver its final report. All 1,383 frozen candidate files verified unchanged. Do not describe this as three completed independent reviews.

## Previous scope

18.1-18.4 independently reproduced and corrected according to targeted progress; ordinary S6 frame exact (1,192 rows x 46 fields). 18.5 wording is implemented and verified by parent, absent from blind findings, but targeted independent disposition was not delivered. No new fundamental mathematical flaw found in the available reports. No full bootstrap rerun.

## New concrete findings

- [19.1](../../../objections/19.1/README.md): high for MC summary intervals; date mixing and unavailable draw omission. Verified by all three reviewers in scope.
- [19.2](../../../objections/19.2/README.md): high for changed MC range; converter truncates selected birth years.
- [19.3](../../../objections/19.3/README.md): moderate; absent sampled dose aborts production.
- [19.4](../../../objections/19.4/README.md): moderate; stale MC reuse after configuration change.
- [19.5](../../../objections/19.5/README.md): moderate malformed input; overlapping Table 8 point validation does not enforce nested feasibility. Actual archived sheets passed.
- [19.6](../../../objections/19.6/README.md): low; raw comparator tiny positive reference threshold.
- [19.7](../../../objections/19.7/README.md): low; MC source and capped-dose metadata.

19.1-19.4 and 19.7 concern the separate record-level MC producer and summary workflow. They complement the prior fixes to MC workbook aggregation; they are not new defects of the ordinary person-bootstrap estimator. 19.5 is an independent feasibility edge case. Neither report demonstrated errors in the archived primary Table 8 values. The conditional external-MC future-birth issue is preserved in targeted progress, not an active objection to synthetic conventions already explicitly retained in 13.2. Existing scope/interpretation choices remain intact. No corrections applied.

[Blind A](blind-a.md) | [Blind B](blind-b.md) | [Targeted progress](resolution-progress.md) | [Input verification](input-verification.json).

## Approved corrections in m184

19.1-19.7 implemented, preserving the source population and dose-cap design. [Producer and Makefile tests](producer-fixes/README.md), [reader and interval tests](reader-fixes/README.md), [Table 8 overlap tests](overlap-fix/README.md), [actual producer-reader integration](integration.md), [candidate verification](m184-verification.json). Ordinary S6 point output exactly unchanged across 1192 rows x 46 fields; all manuscript, figure and numerical archives unchanged. Single-worker actual converter tested on Windows; multi-worker branch not independently exercised. No full bootstrap rerun and no independent round 20 launched. From round 20, use GPT-6.2 Sol High as instructed.
