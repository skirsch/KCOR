# Round 18 convergence - m181

Two fresh blind and one targeted resolution GPT-6.2 Astra High reviews; unchanged importance-focused question from round 17. All three reports saved independently.

## Previous objections

17.2 and 17.3 resolved. 17.1 original mixed-replicate row omission resolved, but its extension to per-dose reconstruction introduces a regression when the reader filters out a source age (18.1). No change to core fitting or inversion.

## New and related findings

| ID | Finding | Severity and origin |
|---|---|---|
| [18.1](../../../objections/18.1/README.md) | Supplied and reconstructed MC total can cover different ages | High for affected MC outputs; regression from 17.1 |
| [18.2](../../../objections/18.2/README.md) | MC bypasses validation of exhaustive source totals | Moderate, malformed-input handling |
| [18.3](../../../objections/18.3/README.md) | Raw all-ages comparator pools or loses replicate IDs | Moderate, secondary output |
| [18.4](../../../objections/18.4/README.md) | Partial MC aggregate labeled All Ages with missing coverage | Moderate, population reporting |
| [18.5](../../../objections/18.5/README.md) | Optional nonlinear survivor-law fitting versus post-inversion wording | Minor manuscript inconsistency |

Root cause shared by 18.1-18.4: MC-specific aggregation/validation/reporting paths do not consistently carry the same replicate-specific source population through all computations. The scope is distinct from the ordinary person-bootstrap pipeline and the independent Table 8 calculator. A single coherent correction and an input-layout matrix would prevent iterative local patches from introducing another mismatch. No change adopted yet.

## Scientific convergence

No new foundational flaw. The ordinary S6 replay is identical across 1,192 rows and 46 fields between m180 and m181. All twelve Table 8 points and 24 archived percentile bounds independently match the reported values at displayed precision. The earlier S6 slope sensitivity is reproduced, not a new defect. All numerical archives and figures remain unchanged. No expensive full bootstrap rerun. Previous intentional choices and editorial dismissals remain respected; [scientific context](scientific-context/README.md).

[Blind A](blind-a.md) | [Blind B](blind-b.md) | [Resolution](resolution.md) | [Frozen input manifest](inputs.json) | [Input verification](input-verification.json).

## Approved corrections

18.1-18.4 implemented together in persistent m182. [Verification](mc-source-consistency/README.md) and [candidate manifest](m182-verification.json). All targeted tests pass and the ordinary S6 replay is exactly unchanged. One unrelated legacy controlled-failure assertion fails identically in m181 and m182. No manuscript or numerical archive changes. 18.5 remains open for author discussion. No new evaluation launched.

18.5 implemented in persistent m183. Equivalent obsolete descriptions in paper and supplement corrected, all numerical files and code identical to m182; clean and redline PDFs regenerated and links verified. [Exact edits](18.5-wording-edits.json). All five corrections complete; round 19 authorized.
