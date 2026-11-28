# 4.5 — First-window seed eligibility

**Moderate — verified remaining restriction**

Although the all-window objective uses later quiet observations, a two-week first window returns insufficient_data before pooled optimization, even with 31 later noiseless quiet weeks.

**Root cause / related objections:** Related 1.3 and 3.2. Decide whether the first-window minimum is scientifically necessary or an initialization limitation.

**Source review:** [repeat-1/blind-targeted.md](<../../iterative review/evaluations/4/repeat-1/blind-targeted.md>)

[Series 4 comparison](<../../iterative review/evaluations/4/convergence.md>) · [Independent verification](<../../iterative review/evaluations/4/verification.md>) · [Counterexamples](<../../iterative review/evaluations/4/counterexamples.json>)

**Corrected in candidate m98.** Use the first usable contiguous quiet window for initialization. If no individual window has three weeks or the seed optimization fails, fall back to the early rate and initial frailty and attempt the unchanged pooled objective. The total-data and death-count checks are retained. The original counterexample recovers k=0.002 and theta=2 with 33 quiet weeks (35 fitted rows including early anchors); all-short-window case also recovers truth; two total quiet weeks still fail. A forced seed optimizer exception also recovered k=0.002, theta=2, status ok. Table 8 points and first paired draw match m97. Full bootstrap rerun was not performed.

[Patch](initialization.patch) · [Paper patch](paper.patch) · [Regression](check.py) · [Results](verification.json). This record concerns the hypothetical integrated edition, not a claim that the actual draft PR has been merged.
