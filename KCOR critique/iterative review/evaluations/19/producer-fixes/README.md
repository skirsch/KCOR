# Round 19 producer corrections: candidate m184

## Implemented scope

- **19.2:** Build the MC grid from the actual eligible sampled records, without the former second 1930–1960 restriction. Reconcile each opening-risk cell and each weekly death cell with the underlying sampled records before constructing aggregate rows. Keep the original default birth range (1930–1960), earlier 1920–2005 eligibility filter, existing product restrictions and enrollment dose cap. The metadata bounds describe the selected birth-frame policy, not the per-draw observed extrema.
- **19.3:** Reindex each enrollment pivot explicitly to all configured doses, preserving legitimate zero counts in highest or interior categories. Do not alter the capped enrollment dose rule.
- **19.4:** Use one active configuration stamp, content-compared on every invocation, plus two per-stage completion stamps. Content comparison also forces a rebuild when file timestamps are too coarse to distinguish consecutive generation stages. Update each completion mark only after that generation succeeds; retain existing completed outputs on a failed producer run without treating them as fresh for its new configuration. Include the YAML dependency and track the enrollment, iteration count, both bounds, aggregate switch, and preexisting freeze override. Unchanged configuration does not regenerate either workbook.
- **19.7 / producer side of 19.1:** Persist `MCSourceYOBStart`, `MCSourceYOBEnd`, `MCDoseCap` and `MCPlannedDraws` on every emitted row. The planned draw count excludes the original (`mc_id=0`). All resamples use the last recorded date of the original eligible master, rather than the date extremes of their own draws, so missing the last event does not truncate the observation calendar. The source does not establish a later observation cutoff; none is invented. If the master has no recorded follow-up on/after enrollment, fail explicitly. A parallel run with a missing completion marker now fails instead of combining partial iterations. The existing fork path is retained for parallel processing; `MC_THREADS=1` runs directly, permitting the real converter to run on Windows too.

## Verification

From the repository root:

```powershell
python work/kcor-review-candidates/m184/test/test_round19_mc_producer.py
```

Completed successfully on Windows/Python 3.14, pandas 3.0.2, GNU Make 4.4.1 (POSIX shell from Git for Windows):

```
PASS actual converter: nondefault years, default scope, zero highest/interior doses, capped doses, shared calendar, exact sampled records and aggregates
PASS actual Makefile: unchanged run is fresh; changed range, aggregates, enrollment and iteration count rebuild both stages; restoring every setting rebuilds; failed producer retries
```

The actual converter was run on three independent 53-column record fixtures using four iterations including the original. Fixture records included 1940, 1960 and 1970 births, a pre-enrollment death, a later death outside the requested birth range, and missing dose categories. For 1930–1970, the original output had six people and three deaths, including one 1970 death. With the default 1930–1960 range, the original output had four people and two deaths. Every individual age/dose/week risk/death cell was independently reconciled with the exact original or `random_state=iteration` sampled people; aggregate rows were compared to constituent sums. The missing top-dose category (2) and a separate missing interior category (1) had explicit zero risks/deaths without losing other groups; dose 3 was correctly pooled at cap 2 in the second test. Draw 2 omitted the last-death person yet retained the fixed 2021-40 observation week with zero deaths there. The variant disabling aggregates emitted no negative-code totals.

An additional actual converter invocation on the bundled `records_10k.csv` sample with `MC_YOB_START=1930`, `MC_YOB_END=1970`, `MC_ENROLLMENT_DATE=2021_24`, `MC_ITERATIONS=2`, and `MC_THREADS=1` completed with 11,484 workbook rows from two iterations, 174 shared weeks per iteration. This was a scratch workbook, not a replacement for an archived result.

The Make test copies the exact candidate `code/Makefile` into a scratch tree and uses two minimal generation probes, not simplified mock dependency rules. It verifies actual process invocation and transmitted settings; the independent converter tests verify real data generation. The test changes range, aggregate flag, enrollment and iteration count individually and back, checks unchanged invocation does not modify the active stamp or call either stage, and verifies a deliberate failing producer is rerun on retry. Initial testing also demonstrated why both stage completion markers are needed: a bare content-checked active-stamp rule on this GNU Make/Windows installation could regenerate the producer yet skip the consumer because of equal-second timestamps.

## Boundaries

- No changes to ordinary bootstrap, primary estimator, manuscript, archived source/results, consumer (`KCOR.py`) or external workbook conventions in this producer patch.
- Native Windows does not support the Linux `fork` start method; the actual converter tests exercised the one-worker direct execution path and did not independently execute the multi-worker branch. Completion checking is applied after either path.
- A complete 999-draw analysis was not executed, and no published primary result is asserted to have changed.

Candidate files changed: `code/KCOR_CMR.py`, `code/Makefile`, `test/test_round19_mc_producer.py`.
