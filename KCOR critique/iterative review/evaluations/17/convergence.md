# Round 17 convergence - m180

Two independent blind GPT-6.2 Astra High reviews and one targeted Astra High resolution review. Same importance-focused question as round 16: ?Examine KCOR closely. What, if anything, is wrong that matters to the validity of the method or its results, and how would you fix it? Distinguish substantive flaws from minor corrections, and do not propose unnecessary redesigns.?

## Result

All five round-16 findings independently resolved in their original scope. Both blind reviewers found the core proportional inversion coherent under its working assumptions and reproduced the main numerical evidence. All six Table 8 primary percentile intervals match the archived draws; all 999 draws per comparison contain a contributing residual flag, already correctly disclosed. The targeted S6 replay matched all 1,192 rows and 41 prior fields exactly. Thirty numerical/figure files checked across candidates; only the intended ancillary summary changed. No full bootstrap rerun.

## New active findings

- [17.1](../../../objections/17.1/README.md) ? Moderate, mixed Monte Carlo input availability. A total supplied in one replicate removes another replicate?s direct all-ages result. Pre-existing, not a regression.
- [17.2](../../../objections/17.2/README.md) ? Low, fractional-count availability. Exact equality rejects a valid exhaustive total due to floating-point roundoff. Pre-existing, not a regression.
- [17.3](../../../objections/17.3/README.md) ? Minor, wrong section reference (3.4 instead of 3.3).

## Convergence and scope

No new foundational mathematical defect or demonstrated error in the current main numerical results. The remaining concrete software findings concern edge-case output availability, not the core estimator. The existing intentional anchored S6 example and previous editorial dismissals remain respected. Repeated eligibility and slope sensitivities and the newly quantified Table 8 wave-anchor sensitivity are retained as [scientific context](scientific-context/README.md), not active redesign proposals. The archived primary and sensitivity intervals were reproduced, but alternate specifications have only point refits. No correction adopted by this evaluation.

[Blind A](blind-a.md) | [Blind B](blind-b.md) | [Resolution](resolution.md) | [Frozen input manifest](inputs.json) | [Input verification](input-verification.json).

All three corrections implemented in m181. [Verification](post-review-revisions.md). Independent round 18 follows at user request.
