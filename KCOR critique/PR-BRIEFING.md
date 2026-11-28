# KCOR revision - lead-author briefing

## Why we investigated

- **Table 8:** Two original 2022 negative-control intervals excluded 1: 1.0756 [1.055, 1.097] and 1.0911 [1.070, 1.112]. Recalculated intervals included 1 but were sometimes very wide. Low frailty was difficult to estimate.
- **Contamination:** With no added mortality, mean KCOR was 1.050 [1.034, 1.066], despite a known true value of 1.

Two fundamental causes emerged: insufficient statistical information about frailty and numerical instability caused by carrying a weak initial fit into the later fit. We also found specific mistakes in wave equations, uncertainty calculations, validation experiments and software. The objective of KCOR is unchanged.

**Review status:** m185 is the hypothetical integrated candidate for lead-author approval, not an approved replacement for production. Round 21 independently closed the previous three findings; four output/reporting issues remain open. The changes below are ordered from scientific importance to minor wording and labeling. IDs link to representative evidence, not a one-to-one count of all 190 historical objections.

## A. Fundamental algorithm and scientific propositions

| Category | Before | Proposed change and what it addresses |
|---|---|---|
| A1. All-window joint fitter | Estimate k early, then theta with k fixed; the intermediate joint fit used only the first window for both. | Jointly fit one baseline scale k and enrollment variance theta across all quiet windows, recomputing gap offsets for each candidate pair. Reduces instability and propagation of weak seed estimates. [2.7](objections/2.7/README.md) |
| A2. Theoretical precision limits | Empirical uncertainty without a mathematical benchmark. | Relate attainable frailty and KCOR precision to population size, deaths, variance and quiet-window timing using Fisher information with gap survival. Separates irreducible uncertainty from fitter failure. [Precision evidence](PR-SCIENTIFIC-EVIDENCE.md#2-theoretical-limits-of-frailty-and-kcor-precision) |
| A3. Nonlinear wave inversion | Approximate cumulative correction assuming gamma-like survivors. | Solve weekly conditional survival for wave exposure and carry both cumulative exposures forward. Accounts for the non-gamma survivor distribution; retains total reference-frailty exposure B+W. [1.2](objections/1.2/README.md) |
| A4. Consistent wave-period fitter | Fit quiet mortality with a gamma model after nonlinear selection. | Use the same specified survivor law for fitting and inversion. Corrects inconsistent frailty estimates; exact recovery with known model inputs. [1.2](objections/1.2/README.md) |
| A5. Pre-entry correction | Mortality before the first quiet window can distort the fitted baseline. | Recover preceding proportional exposure from survival; align the first window as well as later gaps. Unmarked nonlinear exposure remains unidentified. [3.2](objections/3.2/README.md) |
| A6. Person-bootstrap uncertainty | Resample raw week blocks; former analytical interval omitted fitting uncertainty. | Resample people with their complete histories, refit on every draw, and recompute the anchor and pooling; retire the legacy analytic interval. [1.3](objections/1.3/README.md) |
| A7. Integrated weekly hazard | Fit an instantaneous model value to an entire week's mortality. | Fit exact integrated weekly predictions and invert observed log survival consistently. [2.2](objections/2.2/README.md) |
| A8. Higher-information contamination experiment | Small-cohort biased null confused added contamination with sampling/fitting bias. | Retain the original stress experiment and add larger cohorts; isolate finite-information bias from excess-mortality effects. [2.7](objections/2.7/README.md) |
| A9. Alpha identification | Historical pooled objective near 1.18-1.19 could be misconstrued as an identified exponent. | Retain alpha=1.18 as sensitivity only; the new machinery does not establish identification in Czech data. No redesign or new exponent proposed. [1.4](objections/1.4/README.md) |

**Most important numerical findings**

- All-window fitting reduced the 2022 Table 8 interval widths by 73-90% in the matched fitter comparison. In a known-null stress case, the point changed from 812.675 [0.848, 744.878] to 1.016 [0.875, 1.131].
- The theoretical model-based KCOR interval width was 0.263 versus observed median bootstrap width 0.315 in the matched stress design. KCOR can be much better resolved than frailty itself.
- Consistent survivor fitting moved the sampled known-null mean from 0.98982 [0.98493, 0.99471] to 1.00093 [0.99596, 1.00591]; coverage was unchanged. Proportional entry correction moved a separate sampled mean from 1.20386 [1.19425, 1.21347] to 0.99774 [0.99321, 1.00228].
- Zero-contamination mean with the improved fitter: 1.0494 [1.0305, 1.0684] at 50,000 per cohort and 1.0013 [0.9993, 1.0033] at five million. The larger study explains, rather than hides, the low-information limitation.

These are specific matched studies, not universal performance guarantees. Simulation-average intervals describe Monte Carlo uncertainty; stress-case intervals are person-bootstrap intervals. The empirical Table 8 control has no known exact null. Approve the proposed scientific/algorithmic choices separately, including the attribution rule for below-baseline wave weeks (keep prior wave exposure, add zero wave exposure, recover a smaller proportional increment).

## B. Mathematical and numerical correctness

| Category | Before | Correction and purpose |
|---|---|---|
| B1. Unit-exponent identity | Extra wave adjustment even at alpha=1. | Restore ordinary proportional gamma inversion; distinguish inversion identity from wave-marked fitting selection. [1.2](objections/1.2/README.md), [11.5](objections/11.5/README.md) |
| B2. Wave-entry frailty | Use the wrong survivor state at wave entry. | Carry the actual entry state and cumulative exposure into the correction. [2.4](objections/2.4/README.md) |
| B3. Gap boundaries | Final gap week omitted or nonexistent gap created. | Use complete calendar boundaries and the union of fitting weeks. [2.3](objections/2.3/README.md), [10.4](objections/10.4/README.md) |
| B4. Wave and deficit attribution | First wave week or small numerical deficits handled inconsistently. | Include the actual first week; handle identity cases and signed/rounded deficits consistently, retaining diagnostic counts. [4.3](objections/4.3/README.md), [12.2](objections/12.2/README.md), [13.1](objections/13.1/README.md) |
| B5. Gap safeguards | Negative offset could erase a valid positive frailty fit. | Preserve fitted variance; flag material negative gaps relative to uncertainty without silently changing the estimator. [3.1](objections/3.1/README.md), [9.7](objections/9.7/README.md) |
| B6. Seed eligibility and boundaries | Unusable first seed or hard-coded bound could stop/alter later fitting. | Separate initialization from pooled eligibility, match configured bounds, retain flags. [4.5](objections/4.5/README.md), [5.2](objections/5.2/README.md), [10.3](objections/10.3/README.md) |
| B7. Solver accuracy | Absolute tolerance or ceiling could erase accurate rare-event fit. | Scale the objective, check bounds and retain numerical diagnostics; a ceiling is not a biological limit. [7.3](objections/7.3/README.md), [8.4](objections/8.4/README.md) |
| B8. Survival numerical integration | Branch switch or clipping could distort survival/inversion. | Use a consistent monotone, converged integral; report unavailable when outside representable range. [6.12](objections/6.12/README.md), [5.11](objections/5.11/README.md) |
| B9. Reference and ratio arithmetic | Undefined anchor became 1; positive ratios clipped or genuine zeros dropped. | Require valid reference support, preserve representable positive/zero outcomes and unanchored support separately. [5.3](objections/5.3/README.md), [12.8](objections/12.8/README.md), [13.3](objections/13.3/README.md), [13.6](objections/13.6/README.md), [19.6](objections/19.6/README.md) |

## C. Cohort, data and processing correctness

| Category | Before | Correction and purpose |
|---|---|---|
| C1. Fixed risk-set accounting | Unexplained losses or incomplete component histories could pass. | Reconcile deaths and risk sets, require complete histories unless truly exhausted, and preserve declared exit policy. [8.1](objections/8.1/README.md), [9.5](objections/9.5/README.md) |
| C2. Exhaustion and calendar support | Finite results after risk exhaustion, or valid exhausted suffix rejected. | Withhold unsupported portions and recognize valid exhausted groups. [9.6](objections/9.6/README.md), [10.1](objections/10.1/README.md), [11.2](objections/11.2/README.md) |
| C3. Consistent all-ages population | Fit, normalization or supplied totals used different populations. | Reconcile exhaustive sources before aggregation; preserve genuine wider totals and reject contradictions. [6.1](objections/6.1/README.md), [11.1](objections/11.1/README.md), [14.4](objections/14.4/README.md) |
| C4. Nested and overlapping populations | Impossible complementary populations or double-counted overlapping controls. | Validate nested weekly deaths/survivors and reconstruct disjoint resampling cells; preserve intentional overlap in reported control statistic. [7.4](objections/7.4/README.md), [15.3](objections/15.3/README.md), [19.5](objections/19.5/README.md) |
| C5. Partial extraction and ingestion | Incomplete supplied groups could be mistaken for a full population; rounding rejected correct fractional totals. | Preserve available-age support and recorded zeros separately by dose; tolerate only representational fractional rounding. [15.2](objections/15.2/README.md), [16.1](objections/16.1/README.md), [17.2](objections/17.2/README.md) |
| C6. Subgroup pooling | Incorrect weight formula or silent dropping of contributors. | Retain the documented expected-death weights and full support; propagate flags and recompute weights for refits. [2.9](objections/2.9/README.md), [9.8](objections/9.8/README.md), [10.8](objections/10.8/README.md) |
| C7. Independent MC replicates | One replicate affected another's totals, weights or raw comparator. | Per-replicate aggregate source, validation, weighting, raw comparator and population coverage; no cross-replicate mixing. [16.2](objections/16.2/README.md), [17.1](objections/17.1/README.md), [18.1](objections/18.1/README.md), [18.3](objections/18.3/README.md) |
| C8. MC converter and build | Hard-coded birth range, absent dose crash, stale configuration reuse. | Respect configured source range, emit explicit zero categories, rebuild on changed configuration and preserve complete planned draws. [19.2](objections/19.2/README.md), [19.3](objections/19.3/README.md), [19.4](objections/19.4/README.md) |
| C9. Date and interval support | Inconsistent week labels, changed population or differing follow-up dates could enter a summary. | Validate MC ISO dates, common reporting date, original population and complete draw roster; retain correctly dated trajectories. [19.1](objections/19.1/README.md), [20.2](objections/20.2/README.md) |

## D. Validation and scientific evidence

| Category | Before | Correction and purpose |
|---|---|---|
| D1. Cox comparison target | Cox deviations described as bias without adequately distinguishing targets and provided frailty. | Identify fixed-frailty benchmark, known variance supplied, and nominal Cox intervals; separate inversion from fitted performance. [1.5](objections/1.5/README.md), [2.1](objections/2.1/README.md), [11.10](objections/11.10/README.md) |
| D2. Positive-control injection | Description and generator did not match intervention applied. | Match descriptions to actual injection and report the benefit gap-assumption failure as audit-only. [2.6](objections/2.6/README.md), [3.3](objections/3.3/README.md), [13.4](objections/13.4/README.md) |
| D3. Empirical Table 8 controls | Near-unit estimates overinterpreted and historical interval procedure insufficient. | Recalculate matching intervals; distinguish pseudo-null, model discrepancy and absence of exact target. [1.4](objections/1.4/README.md), [9.10](objections/9.10/README.md) |
| D4. Quiet-window robustness | Scan discarded failed windows and used inconsistent fitter. | Use the corresponding fitter, count all attempted windows and distinguish genuine instability from insufficient signal. [2.5](objections/2.5/README.md) |
| D5. Frailty recovery and stress designs | Deterministic or historical simulations overstated what current fitted KCOR validated. | Identify generating inputs, distinguish deterministic from sampled coverage, preserve original stress cases. [6.5](objections/6.5/README.md), [8.7](objections/8.7/README.md), [12.5](objections/12.5/README.md) |
| D6. Empirical example eligibility | Selection and retrospective manufacturer restriction not sufficiently identified. | Preserve the intended restricted population, disclose retrospective selection and distinguish the older age-shift extract. [2.10](objections/2.10/README.md), [7.5](objections/7.5/README.md), [9.16](objections/9.16/README.md) |
| D7. Manufacturer analysis | Report used unadjusted ratios or inconsistent enrollment cohort. | Reuse ordinary KCOR, reconcile fixed manufacturer groups, replace superseded artifacts. [14.1](objections/14.1/README.md), [14.2](objections/14.2/README.md), [15.1](objections/15.1/README.md) |
| D8. Archived result reproducibility | Resume settings, figures or references could use mismatched configurations. | Match reported anchors, horizons, ceilings, results and figure data to their calculation; preserve reproducible sources. [7.7](objections/7.7/README.md), [9.3](objections/9.3/README.md), [11.12](objections/11.12/README.md), [16.5](objections/16.5/README.md) |
| D9. Interpretation of diagnostic evidence | Numerical success or near-zero frailty could be confused with identification. | Distinguish residual/gap assessment, fit bounds, insufficient information and causal limitations. Preserve numerical fallback for audit rather than resetting it. [6.3](objections/6.3/README.md), [10.2](objections/10.2/README.md), [10.9](objections/10.9/README.md) |

## E. Reporting, labels and minor corrections

| Category | Before | Correction and purpose |
|---|---|---|
| E1. Diagnostic flags in MC intervals | Fallback fit could be reported as valid with unqualified intervals. | Preserve finite values and count flagged/unassessed fits; qualify audit-only results. [20.1](objections/20.1/README.md) |
| E2. Standalone ordinary/negative-control summaries | Diagnostic fields omitted from auxiliary outputs. | **Open:** propagate point/draw residual, gap and fit qualifications without changing the estimator. [21.1](objections/21.1/README.md), [21.2](objections/21.2/README.md) |
| E3. Ordinary date labels | Contradictory ISO week may persist in output even though actual dates drive fitting. | **Open:** canonicalize output dates and disclose contradiction; numerical fit unchanged. [21.3](objections/21.3/README.md) |
| E4. Current uncertainty documentation | Old slope-removal and analytical-method descriptions persist in code/summary default. | **Open:** describe current refitted person-bootstrap and mark absent method unavailable. [21.4](objections/21.4/README.md) |
| E5. Population and dose labels | Age band, partial aggregate, top-coded dose or selected MC birth frame were unclear. | Identify actual birth band, supplied ages, selected range and capped-dose meaning; do not change cohort definition. [6.10](objections/6.10/README.md), [18.4](objections/18.4/README.md), [19.7](objections/19.7/README.md) |
| E6. Output parameter and exposure labels | Pooled row copied one stratum's parameter; cumulative person-time was weekly exposure. | Use pooled-appropriate labels/status and cumulative exposure. [11.13](objections/11.13/README.md), [11.16](objections/11.16/README.md), [12.9](objections/12.9/README.md) |
| E7. Anchored/unanchored and finite-endpoint notation | Anchored result called unanchored; finite simulation deviation called asymptotic bias. | Correct the notation while preserving the intended anchored illustration and finite endpoint. [13.7](objections/13.7/README.md), [13.8](objections/13.8/README.md), [15.4](objections/15.4/README.md) |
| E8. Method and interpretation wording | Outdated post-inversion-only wave description; k called gamma frailty parameter; some claims overstated scope. | Match current survivor-law fit and exact estimand; distinguish baseline level from frailty curvature and conditional descriptive from causal claims. [18.5](objections/18.5/README.md), [20.3](objections/20.3/README.md), [14.3](objections/14.3/README.md) |
| E9. Editorial and artifact consistency | Incorrect cross-reference, stale TeX, missing figure intervals or superseded summary labels. | Correct specific references, captions, source alignment and linked redline; retain intended designs. [6.8](objections/6.8/README.md), [14.11](objections/14.11/README.md), [17.3](objections/17.3/README.md) |

## Decision and next steps

The foundational propositions need lead-author consideration; routine corrections are implemented in the hypothetical candidate except the four explicitly open items. No further LLM evaluation is authorized yet. The latest review found no new core mathematical flaw under the stated working model. It did not establish universal coverage or causal identification.

[Current state](CURRENT-STATE.md) | [Full scientific presentation](PR-PRESENTATION.md) | [Detailed numerical and mathematical evidence](PR-SCIENTIFIC-EVIDENCE.md) | [Complete objection lineage](OBJECTION-LINEAGE.md) | [Candidate, clean paper and redline](<iterative review/candidates/m185/README.md>)
