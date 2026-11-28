# Round 10 convergence - m152

Two independent blind GPT-6.2 Astra High reviews and one independent targeted Astra High review of m142 against m152. Original question verbatim: "Please examine KCOR closely. What is wrong, if anything, and how would you fix it?" All three found proportional gamma inversion sound. No large simulation reruns or corrections adopted.

## Direct carryovers from round 9 (two)

- **[10.1](../../../objections/10.1/README.md) - Verified exhaustion suffix rejected by overlapping bootstrap** (Moderate to high; R). An entire age/dose genuinely exhausts; explicit zero suffix gives unaffected 1950 point 1 and three valid draws [0.850521,1.083847], omitted zeros give same point but no valid draws. Both dedicated reader and bootstrap reject the omitted form, ordinary validator accepts.
- **[10.2](../../../objections/10.2/README.md) - Remaining main-text near-zero frailty identification claims** (Low to moderate; A, B, R). Main Section 2.6.2 item 4 and opening of Section 5 still treat small theta as weak identification; Section 2.6.1 also confuses detectable curvature and fitted depletion. SI wording is correct.

## New specific findings (seven)

- **[10.3](../../../objections/10.3/README.md) - Hard-coded seed ceiling falsely flags a baseline gap** (High; A, B). Seed fit capped at 100 while final fit ceiling is 500. True variance 200, exact zero-excess gap: final theta 200 but first-pass offsets -0.011288 and -0.0534085 in two independent designs; sampled 49-draw intervals entirely negative despite correct model.
- **[10.4](../../../objections/10.4/README.md) - Anchor and quiet union can create a nonexistent gap** (Moderate; A). Identical 55 fitted rows declared all quiet vs quiet starting immediately after four anchors yield theta 4.184287 vs 4.181767 and a phantom offset 0.0000614. Re-entry built from quiet-only segments rather than union.
- **[10.5](../../../objections/10.5/README.md) - Nonlinear fit inherits proportional initializer gap diagnostic** (Moderate for optional module; A, B). Exact nonlinear wave with every other week quiet has no ordinary gap, yet forwarded first-pass offsets at wave exit +0.221537 and +0.0701172 in two independent designs.
- **[10.6](../../../objections/10.6/README.md) - Spawned bootstrap workers lose effective age grouping** (High for affected workbook users; A). Windows five-year AGE_RANGE=5 point for 1940 receives finite draws from ten-year merged 1940+1945 group; 1945 gets missing draws. Dedicated S6 archive unaffected.
- **[10.7](../../../objections/10.7/README.md) - Direction of depletion distortion overgeneralized** (Moderate scientific wording; A). Section 4.3 says depletion exaggerates vaccinated-unvaccinated difference. Example reference hazards .05/.10 and variances 0/5: reference ratio .5, observed ratio .617, attenuation not exaggeration.
- **[10.8](../../../objections/10.8/README.md) - Pooling weight formula not fully specified** (Moderate for reproducibility; B). Implemented weights average positive first-four-enrollment-week dose-week rates unweighted, then multiply by total person-weeks; not pooled deaths/person-time. Actual 2022 normalized weights .17343,.17560,.65096 vs death-share .16244,.17864,.65893.
- **[10.9](../../../objections/10.9/README.md) - Diagnostics cannot guarantee failure in unsuitable designs** (Low to moderate scientific wording; B). Section 2.14 says required diagnostics will fail in unsuitable designs. A homogeneous time-varying hazard can reproduce exactly the gamma-Gompertz marginal survival; observed checks cannot distinguish mechanisms.

## Previously disclosed application limitation: narrow wording (one)

- **[10.10](../../../objections/10.10/README.md) - Fixed-cohort wording still overstates avoidance of immortal-time artifacts** (Low wording; known application limitation; A, R). Section 2.2 says fixed design avoids immortal-time artifacts, while later eligibility disclaimer and S6 disclose retrospective selection. 363,033 excluded solely at/after enrollment, effect size/direction unknown.

## Round 9 substantive objections independently resolved (thirteen)

[9.2](../../../objections/9.2/README.md), [9.3](../../../objections/9.3/README.md), [9.5](../../../objections/9.5/README.md), [9.6](../../../objections/9.6/README.md), [9.7](../../../objections/9.7/README.md), [9.8](../../../objections/9.8/README.md), [9.9](../../../objections/9.9/README.md), [9.10](../../../objections/9.10/README.md), [9.11](../../../objections/9.11/README.md), [9.12](../../../objections/9.12/README.md), [9.13](../../../objections/9.13/README.md), [9.14](../../../objections/9.14/README.md), [9.15](../../../objections/9.15/README.md). The 9.7 protective-gap diagnostic resolves the reported protective case; the new 10.3, 10.4 and 10.5 counterexamples concern the seed, gap location and nonlinear path, not failure of the original targeted regression.

The targeted review independently ran nine regression scripts in scratch, verified all 12 Table 8 ordinary points within 2.23e-16, recomputed stored intervals, checked all 1,323 sensitivity cells and all 120 diagnostic category/pair combinations of point flags and draw counts, and verified all 300 forced-alpha fresh-replay records against the archive. Full 999-draw fits were not rerun.

[Blind A](blind-a.md) | [Blind B](blind-b.md) | [Resolution](resolution.md). No reviewer recommendation is adopted automatically.

Frozen input verification: 1309 files; unchanged=True. [Manifest](inputs.json), [Verification](input-verification.json).
