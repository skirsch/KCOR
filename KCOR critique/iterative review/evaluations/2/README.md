# Three-path KCOR evaluation

Review series 2 was run on November 21, 2026. The resulting findings are filed as [2.1 through 2.10](../../../README.md). This was the first full three-path evaluation.

[Convergence and objection-by-objection assessment](convergence.md) · [Independent checks](checks.md)

The cycle uses GPT-6.2 Astra with high reasoning and three independent repetitions. Each starts with a blind review of the revised edition and an independent review of the baseline. B continues the baseline conversation with the revised edition and the diff. C receives the same baseline answer verbatim, the same revised edition and the same diff in a fresh context. The scientific question for all four turns is:

> Please examine KCOR closely. What is wrong, if anything, and how would you fix it?

After the open responses are saved, a separate follow-up asks B or C about baseline objections it left unanswered. These prompted judgments are distinguished from spontaneous findings.

| Repetition | Shared baseline | A · blind | B · same conversation | C · fresh handoff | Follow-up on omissions | Exact prompts |
|---|---|---|---|---|---|---|
| 1 | [Baseline](repeat-1/baseline.md) | [A](repeat-1/A.md) | [B](repeat-1/B.md) | [C](repeat-1/C.md) | [B](repeat-1/B-follow.md) · [C](repeat-1/C-follow.md) | [Prompts](repeat-1/prompts.json) |
| 2 | [Baseline](repeat-2/baseline.md) | [A](repeat-2/A.md) | [B](repeat-2/B.md) | [C](repeat-2/C.md) | [B](repeat-2/B-follow.md) · [C](repeat-2/C-follow.md) | [Prompts](repeat-2/prompts.json) |
| 3 | [Baseline](repeat-3/baseline.md) | [A](repeat-3/A.md) | [B](repeat-3/B.md) | [C](repeat-3/C.md) | [B](repeat-3/B-follow.md) · [C](repeat-3/C-follow.md) | [Prompts](repeat-3/prompts.json) |

Nine separate Astra High reviewer instances produced the scored responses. Every A, baseline and C instance started without parent conversation history. B resumed its corresponding baseline instance. The fresh C instance received the baseline Markdown file as a verbatim handoff. Each reviewer could inspect the scoped scientific files using the same local tools. Early CLI launches could not read the repos and were discarded before scoring. The frozen copies and file references supplied to the scored reviewers are preserved in this archive. [Response hashes and reviewer assignments](responses.json)

## Inputs

The common scientific files come from Git commit `82f546cd5cc1af2ea99e4501a1aa92d0c9335f78`. The review editions include the method code, data, validation, analyses, examples, tests and reference results. The combined manuscript and its Markdown sources appear under `documentation/preprint/`. The supplied repos have a neutral navigation README and omit the repository history, earlier paper drafts, and historical reviewer discussions. The blind instances receive one edition only.

The baseline includes 1,119 files. The revised edition includes 1,128. Four shared files changed: the primary Python method, manuscript, supplement and combined PDF. The revised edition also supplies the joint-fit replay, archived input and results for Table 8, plus person-bootstrap validation outputs. The Table 8 runner uses the stored workbook in place of a Git object lookup and uses relocated copies of the estimator functions. The default fitter in the method code is preserved as reviewed. Binary artifacts are available directly in the applicable edition.

[File hashes](inputs.json) · [Preserved edition overlays](edition-overlays.zip) · [Cumulative text diff](changes.diff) · [Materialization script](freeze.py)

To reconstruct the review inputs from the repository, run `py -3.11 freeze.py <new-empty-directory>` in this folder. The original directory labels were `m17` for the baseline and `m42` for the revised edition. The labels identify the material supplied in the archived prompts.
