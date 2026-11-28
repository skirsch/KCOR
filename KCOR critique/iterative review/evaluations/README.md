# Iterative LLM evaluation of KCOR

Each cycle asks whether independent reviewers find the same specific flaws in a given edition of KCOR, and whether they still find those flaws after a revision. The three paths vary what the reviewer knows about the earlier objections. Use the original question verbatim at every indicated turn.

> Please examine KCOR closely. What is wrong, if anything, and how would you fix it?

## Prepare the two repo editions

For this series, **baseline** means KCOR before our revisions. **Revised** means the repository with our present paper and method changes applied. **The changes** means the actual cumulative difference between those two editions. PR #5 supplied the objections and supporting studies. Its merge alone does not identify our revised edition.

For each cycle, freeze both repo inputs and the baseline-to-revised diff before starting any path. Record the common source revision, the exact files overlaid (including working-tree changes), and hashes or immutable copies of the material actually supplied. The [preserved baseline](../baseline/manifest.json) identifies the common source and archived files. The [candidate directory](../revised/) contains selected replacements. Assemble each edition as a usable repo with the paper, supplement, code, configuration, data and relevant run scripts at unambiguous paths. Include the actual fitter and analysis entry point used for reported results, including the Table 8 joint-fit replay when applicable.

Give reviewers of each edition the same access to its scientific materials. For a blind run, expose a single edition with its matching clean manuscript. Omit Git history, other editions, redlines, critique responses, decision logs and study write-ups that reveal the objections. Preserve scientific inputs or executable dependencies that happen to live beside those write-ups. For a handoff, supply the precise diff, including code, plus the same revised repo available to the blind reviewer. Identify any inputs the reviewer could not inspect.

## Run the three paths

| Path | First review | Review of the revised repo |
|---|---|---|
| A. Revised, blind | Give a fresh instance the revised repo and the original question. | Its first answer is its verdict. |
| B. Same conversation | Give a fresh instance the baseline repo and the original question. Capture its objections verbatim. | In that conversation, supply the changes and the revised repo, identify which edition is now under review, and ask the original question again. Its own earlier reasoning remains in context. |
| C. Fresh handoff | Use the exact baseline answer from B in this repetition. | Start a **new instance**. Give it that baseline reviewer's complete objection output, the changes, and the revised repo. Ask the original question. Its only account of the earlier review is the supplied output. |

Run several independent repetitions. Use fresh instances for each first review and each C handoff. Keep the repo versions, question, model settings and available tools comparable within a set, and record any differences. Keep other paths' answers out of each conversation until all its answers have been captured. The separately recorded follow-up question is available for a later targeted discussion:

> What do you think about this critique of KCOR? Do you have any other critique? (Focus on what is really important, not design choices)

## Compare the answers

Make one row per distinct, checkable claim. Group equivalent objections by the passage, equation, code path or result they challenge. Two reviewers can describe the same flaw in different words, or challenge different claims in the same section. Include every concrete error they identify, regardless of size.

| Finding and location | Baseline B / C | Revised A, blind | Revised B, same context | Revised C, fresh handoff | Evidence and conclusion |
|---|---|---|---|---|---|
| One row per claim | Who raised it? | Did it surface independently? | Did the reviewer retain, narrow or withdraw it? | Did the new reviewer uphold, narrow or dismiss it? | What do the two editions and any reproducible check show? |

Save the open-ended responses first. If a reviewer with baseline objections leaves one unmentioned on re-review, follow up on that specific objection in the corresponding context. Ask "Does this specific objection still apply to the revised version? What in the paper or code supports your answer?" Keep prompted answers separate from spontaneous findings. Record silence as **unmentioned**. Mark an objection **resolved** when the baseline evidence supports it and the revised text, code or result demonstrably removes the problem. Use **persists**, **partly resolved**, **unsupported**, or **uncertain** when the evidence warrants it. For a newly raised finding, check both editions to establish whether it was already present or was introduced by the revision. Capture proposed fixes separately from the diagnosis.

Judge convergence at two levels. **Within one frozen edition**, compare substantive conclusions across independent repetitions and across A, B and C. If the handoff paths repeatedly uphold a verified flaw that blind reviewers miss, record that dependence on prior knowledge. If reviewers agree that a flaw persists, their diagnosis has converged and the flaw is still open. **Across revision cycles**, compare the verified open findings with the preceding cycle. Report which disappeared, persisted, first appeared, or remain disputed. Report convergence only where repeated reviews and the underlying evidence support a stable conclusion. A repeated omission or a majority vote cannot establish correctness.

## Keep the chain of evidence

Save each completed comparison in its numbered review-series folder here and record the date in its README. Each objection has a matching x.y folder at the critique root. Record the input manifests and diff reference, the model and settings, the prompts, and the final responses under their path, repetition and turn. Keep the precise baseline objection text passed to C. The comparison should link each finding to its source responses, verification and [revision decision](../decisions.md). Exploratory intermediate outputs can be discarded. For a subsequent approved change, record what changed, freeze the new candidate, and repeat the three paths against the same original baseline so cycles remain comparable.

[First three-path evaluation](2/) uses three independent repetitions with GPT-6.2 Astra at high reasoning. The [earlier targeted review](../../objections/1.1/fresh-review-evaluation.md) concerns one objection.

## Reviews after a second or later revision

Freeze the current edition, the original edition, the exact last evaluated edition and the last author checkpoint. Compare the current edition blind, directly against the original, and directly against the last evaluated edition. For each direct comparison, take an independent review of the starting edition, continue the same conversation with the new edition and its diff, and give a separate fresh reviewer the same starting answer verbatim, the same diff and the new edition. In each repetition, also run one conversation through original, last evaluated and current in order. Give that reviewer only the next edition and its incremental diff at each transition. Save its intermediate answers.

Compare the checkpoint to current by file and claim to attribute changes. For example, the first checkpoint incorporated the known-frailty disclosure after the prior evaluation. A fresh handoff receives only the specified materials. A targeted review of an archived objection follows the open answer and is recorded separately. Record where a flaw first appears, where a reviewer first raises it, where the relevant change lands, and what independently supports its current status. Keep the scientific question verbatim throughout the open turns. [Multi-edition evaluation](3/README.md)


[Series 4](4/README.md) evaluates the hypothetical fully integrated PR #6 candidate without altering the actual PR. [Convergence](4/convergence.md) and [specific checks](4/verification.md) distinguish remaining assumptions from verified new implementation defects.

[Series 5](5/convergence.md) and [Series 6](6/convergence.md) use the reduced protocol of two independent blind Astra High reviews and one Astra Medium targeted resolution review. Series 6 evaluates candidate m117 after the round-5 corrections. The frozen input manifest and verification confirm that all 1,168 files remained unchanged. Findings are indexed as 6.1–6.14 in the numbered objection directory.

[Series 7](7/convergence.md) uses two blind and one targeted resolution GPT-6.2 Astra High reviews of m128, the frozen post-round-6 candidate. All 1,192 input files remained unchanged. Six new and five returning issues are indexed as 7.1-7.11.

Maintain the [cross-round objection lineage](../../OBJECTION-LINEAGE.md) and its machine-readable companion `objection-lineage.json` after every review and approved correction. Record direct returns separately from related root causes and implementation separately from independent closure.

[Series 9](9/convergence.md) reviews m142 using two blind and one targeted Astra High reviews. Nine round-8 issues resolved, four partial, eleven new specific findings, and one recurring disclosed application limitation. See the numbered records 9.1-9.16.

[Series 10](10/convergence.md) reviews m152 using two blind and one targeted Astra High reviews. Thirteen round-9 substantive issues resolved, two partial, seven new specific findings and one narrow recurring application wording issue. See numbered records 10.1-10.10.

[Series 11](11/convergence.md) reviews m156 using two blind and one targeted Astra High reviews. All ten round-10 issues resolved in original scope; eighteen consolidated new or related findings are indexed as 11.1-11.18.

[Series 12](12/convergence.md) reviews m161 with two blind and one targeted Astra High reviews. Fifteen round-11 issues resolved, three partial; three direct carryovers and eight new or related findings are indexed as 12.1-12.11.

[Series 13](13/convergence.md) reviews m167 with two blind and one targeted Astra High reviews. Ten round-12 issues resolved, one partial; one direct carryover and seven new or related findings are indexed as 13.1-13.8.

[Series 14](14/convergence.md) reviews persistent m169 with two blind and one targeted Astra High reviews. All eight round-13 issues resolved in original scope; twelve new or related findings indexed as 14.1-14.12.

The post-round-14 straightforward corrections (14.5-14.12) are in persistent m170; m169 remains unchanged. [Corrections and verification](14/post-review-revisions.md). Four findings remain for individual discussion.

Redline maintenance (m172 onward): every manuscript change or changed table group must carry a visible blue bracketed objection number linked to its entry in the redline index. Use the robust `\reviewtag{X.Y}` macro and add a matching `\hypertarget{issue-X.Y}` in the index. Preserve older checkpoint redlines; code-only corrections remain documented in the numbered objection record rather than creating artificial paper edits. Verify the PDF link destinations and rendered page for each new edit.

Round 14 correction batch completed in persistent m173 (m170, m171 and m172 retain the intermediate steps). [Corrections](14/post-review-revisions.md). All twelve responses implemented; independent review remains pending.

[Series 15](15/convergence.md) uses two blind and one targeted Astra High reviews of m173 with the updated importance-focused question. Nine round-14 objections fully resolved, three partial; ten consolidated findings indexed 15.1-15.10, with scientific sensitivities retained separately. No fundamental new mathematical defect identified.

Round-15 triage: 15.6-15.10 dismissed as editorial findings, 15.1-15.5 remain active. See [convergence](15/convergence.md#author-side-triage-after-review).

Round-15 corrections complete in m177: 15.1-15.3 implemented; 15.4 original anchored purpose confirmed and stated, 15.5 unassessed diagnostic disclosed; 15.6-15.10 dismissed. No active objections remain from round 15; independent verification pending.

[Series 16](16/convergence.md) uses two blind and one targeted Astra High reviews of m177. Original round-15 findings closed except two narrower validation issues related to 15.3. Five concrete new or related defects indexed 16.1-16.5. Repeated design/wording preferences remain dismissed; independent numerical results again reproduced, no fundamental mathematical flaw.

Round-16 corrections complete in persistent m180: 16.3-16.5 in m178, 16.1 in m179 and 16.2 in m180. All targeted tests passed; independent review pending. Scientific results reproduced, no fundamental estimator change.

[Series 17](17/convergence.md) reviews m180 using two blind and one targeted Astra High reviews. All five round-16 objections resolved in original scope. Two pre-existing software availability defects and one incorrect section reference are indexed as 17.1?17.3; repeated scientific sensitivities and the Table 8 anchor diagnostic are retained separately. No new fundamental mathematical defect.

Round-17 corrections complete in persistent m181; two workbook regressions and the corrected PDF reference verified. [Correction record](17/post-review-revisions.md). Round 18 authorized.

[Series 18](18/convergence.md) reviews m181: 17.2 and 17.3 resolved; 17.1 original row loss resolved, but a new one-dose reconstruction regression was identified. Four related MC findings and one manuscript inconsistency indexed 18.1-18.5. No new foundational flaw, and the main published results reproduced.

Round-18 MC corrections 18.1-18.4 implemented together in persistent m182; complete input-layout matrix and ordinary S6 exact parity verified. [Evidence](18/mc-source-consistency/README.md). 18.5 remains open; no manuscript changes or new review launched.

Round-18 corrections complete in persistent m183: 18.1-18.4 in m182, 18.5 paper/supplement wording in m183, no code changes after m182. Round 19 authorized.

[Series 19](19/convergence.md) reviews m183. Two blind reports delivered, targeted progress recovered; all three agents hit usage limit before final turns. 18.1-18.4 resolved in original scope per targeted progress; 18.5 implemented but independent closure not explicit. Seven new concrete findings indexed 19.1-19.7, mostly separate MC producer/summary bugs. Frozen inputs verified unchanged.

Model choice from round 20 onward (user decision after round 19): retain the two fresh blind reviews and one targeted resolution review, using GPT-6.2 Sol High instead of Astra High. Record actual model and reasoning in each input manifest; do not retroactively relabel earlier reviews.

Round-19 corrections 19.1-19.7 complete in persistent m184. [Verification](19/convergence.md#approved-corrections-in-m184). Ordinary S6 exact parity, actual producer-reader integration and Table 8 overlap tests passed. No manuscript changes or further evaluation launched.

[Series 20](20/convergence.md) is the first Sol High round. All seven round-19 findings and 18.5 independently resolved. Two MC defects and one nomenclature issue indexed 20.1-20.3. All three reports delivered after a successful usage-limit retry. Ordinary S6 exactly reproduced, Table 8 input cells verified.

Round-20 corrections completed in m185, with exact ordinary and valid MC point parity, new flag/calendar regressions and linked manuscript correction. [Verification](20/post-review-revisions.md). Round 21 authorized.

[Series 21](21/convergence.md) reviews m185 with three Sol High reviewers. All three round-20 findings independently resolved. Four specific new/related reporting issues indexed 21.1-21.4; ordinary and valid MC numerical results unchanged.
