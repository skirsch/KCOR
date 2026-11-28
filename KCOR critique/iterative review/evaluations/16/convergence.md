# Round 16 convergence - m177

Two independent blind GPT-6.2 Astra High reviews and one targeted Astra High resolution review. Same importance-focused question as round 15: "Examine KCOR closely. What, if anything, is wrong that matters to the validity of the method or its results, and how would you fix it? Distinguish substantive flaws from minor corrections, and do not propose unnecessary redesigns."

## Result

Both blind reviewers found the proportional estimator mathematically coherent under the specified working model. No new fundamental mathematical flaw was established. Published Table 8 intervals and S6 points independently reproduced; the targeted review found exact parity of 1,192 S6 rows and 40 previous fields between m173 and m177, 26 checked numerical archives unchanged, and all 14 supplied Czech total/dose combinations valid. No full bootstrap rerun.

15.1, 15.2 and 15.5 resolved in original scope. 15.4 closed by confirmed original intent. 15.3 original integer examples resolved, but two specific associated edge cases remain. 15.6-15.10 deliberately dismissed; do not reopen solely for stylistic preferences.

## New active findings

- **[16.1](../../../objections/16.1/README.md) - Partial aggregate may silently compare different age-group coverage across doses** (Moderate for affected partial ordinary inputs; B). An omitted age-by-dose group changes the partial aggregate endpoint and still has an OK status and complete draws labeled All available ages. This is not a requirement for equal age proportions in fully observed cohorts; the issue is undisclosed missing component support.
- **[16.2](../../../objections/16.2/README.md) - Monte Carlo pooling weights depend on other replicates in the same workbook** (Moderate for the batched Monte Carlo branch; B). A fixed simulated replicate has a different pooled endpoint when another replicate is appended to its workbook. Sheet-wide weights are computed across mc_id before per-replicate pooling. Ordinary production person bootstrap is distinct; primary Table 8 and S6 archives not implicated.
- **[16.3](../../../objections/16.3/README.md) - Nested-total validator rejects valid fractional counts from floating-point roundoff** (Low numerical edge case; real point availability regression; R). Dividing a valid nested integer fixture by ten gives fractional expected counts accepted by the source-history validator. Complement residual is 2.27e-13 person; exact loss!=0 now withholds both valid points. Actual Czech integer counts unaffected.
- **[16.4](../../../objections/16.4/README.md) - Partial hidden population can have more cause-specific deaths than all deaths** (Low input-integrity defect; no demonstrated all-cause numerical error; R). A partial supplied-minus-displayed remainder has one COVID death among zero all-cause deaths while all supplied histories individually pass; full-minus-young complement also passes. Joint intervals already withheld.
- **[16.5](../../../objections/16.5/README.md) - Window-omission ancillary CSV contains superseded full-fit intervals** (Minor artifact provenance; B). S6 window-omission CSV reads older summary for full-fit bounds; age 40-49 old [0.823855,2.711896] differs from current reported [0.960387,2.821651]. Omission ranges are point sensitivity and the paper/main interval archive are correct.


## Scientific convergence, not endless restyling

The two moderate software findings affect specific ordinary partial-extract or batched Monte Carlo features, not the reproduced primary numerical evidence. The other three concern fractional-count availability, input integrity, and stale ancillary columns. The previously rejected additional S6 table and negative-control wording returned in blind A, but neither constitutes a new demonstrated error; preserve the confirmed design and decisions. The independent sensitivity results are retained as [scientific context](scientific-context/README.md), not new active objections.

The review supports scientific convergence of the method within stated assumptions. Further corrections can be limited to these five concrete findings. No correction is adopted by this evaluation and no subsequent round is authorized.

[Blind A](blind-a.md) | [Blind B](blind-b.md) | [Resolution](resolution.md) | [Frozen input manifest](inputs.json).

16.3-16.5 implemented and verified in m178, without paper changes. [Correction record](straightforward-batch/README.md). Remaining active: 16.1 and 16.2.

16.1 addressed in m179: per-dose source coverage and recorded-zero flags visible, same original numerical contrast and interval, actual S6 point parity. [Evidence](age-coverage/README.md). Remaining active: 16.2.

16.2 addressed in m180: independent MC weighting and replicate-specific flag lookup verified; standalone/batched/reordered rows identical, ordinary S6 parity. [Evidence](mc-pooling/README.md). All five round-16 findings implemented; no active findings remain, independent review pending. No new evaluation launched.
