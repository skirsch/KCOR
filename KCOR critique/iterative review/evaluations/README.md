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
