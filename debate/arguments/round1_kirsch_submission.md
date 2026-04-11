# Round 1 Kirsch Submission: US mRNA Net Mortality Through 2022

Debate question: In the United States, did the mRNA COVID vaccines likely net save lives or cost lives through the end of 2022?

Position: the more defensible answer is that net benefit is not established and negative net benefit remains at least as plausible as net lives saved. My case is not that any single dataset is a perfect causal proof. It is that the most relevant real-world measurements do not show the large net benefit that Saar's position requires, while multiple US-specific and international safety signals make meaningful harm difficult to dismiss.

## Proposed Judge Categories

These categories are intended to be fair to both sides:

1. Direct efficacy evidence: how strong is the evidence that mRNA vaccination reduced severe COVID outcomes under trial or near-trial conditions?
2. Causal validity of observational VE-death estimates: do the studies adequately handle healthy-vaccinee effects, frailty, prior infection, testing/coding differences, and NCACM negative controls?
3. Population-level benefit fingerprint: if vaccination saved many lives, is the expected mortality pattern visible in real-world COVID-death and ACM measurements?
4. US-specific harm evidence: are there credible US mRNA safety signals that remain unresolved after reasonable confounding objections?
5. US net-mortality reconciliation: can the pro-net-benefit side quantitatively reconcile large claimed lives saved with persistent US mortality elevation?
6. Counterfactual completeness: which side gives the more complete and falsifiable explanation of what should have happened without vaccination?

## Primary Exhibits

1. US UCOD year-over-year divergence: [ucod_year_over_year_divergence_summary.csv](../../test/US/out/ucod_year_over_year_divergence_summary.csv)
2. US UCOD after excluding selected obvious categories: [ucod_icd_excluding_selected_causes_summary.csv](../../test/US/out/ucod_icd_excluding_selected_causes_summary.csv)
3. US mRNA brand-comparison harm signal: [Levi/Ladapo Florida Pfizer-Moderna mortality preprint](../docs/levi_ladapo_florida_pfizer_moderna_12_month_mortality_preprint.pdf)
4. FDA internal safety context: [FDA Prasad memo](../docs/FDA_Prasad_memo.pdf)
5. OWID population-level slope screen figure set: [all-country slope screen](../figures/owid_slopes/owid_wave_slopes_midpoint_r2ge99_dotplot.png) and [top25/bottom25 vaccination-contrast slope screen](../figures/owid_slopes/owid_wave_slopes_vaccination_top25_bottom25.png)

## Claim 1: The US Mortality Record Does Not Look Like Large Net Lives Saved

The US all-cause crude mortality rate was still materially elevated after vaccine rollout. In the UCOD year-over-year summary, all-age mortality was 878.0 per 100k in 2019, 1052.6 in 2021, and 992.7 in 2022. That is +19.9% in 2021 and +13.1% in 2022 versus 2019.

The elevation was not restricted to the oldest age group. Versus 2019, the 2021 increases were +30.7% for ages 0-39, +31.7% for ages 40-59, +22.2% for ages 60-79, and +10.8% for ages 80+. In 2022, the respective increases were +20.7%, +13.4%, +12.9%, and +3.8%.

Removing obvious categories such as external, drug, and alcohol causes did not make the elevation disappear. In the exclusion summary, excluding those categories, ages 35-54 rose from 183.7 per 100k in 2019 to 259.8 in 2021 and 206.0 in 2022. Ages 55-74 rose from 1122.6 in 2019 to 1415.8 in 2021 and 1266.4 in 2022. Ages 75+ rose from 3918.5 in 2019 to 4693.8 in 2021 and 4248.6 in 2022.

Causal interpretation: if mRNA vaccines saved large numbers of net US lives and caused no meaningful harm, the mortality record should be easier to reconcile. Saar needs a quantitative decomposition showing where the claimed net lives saved appear in the actual US mortality record, not just a list of possible confounders.

Caveat: this is descriptive mortality evidence, not proof that vaccines caused the elevation. COVID waves, delayed care, demographic shifts, coding, and other causes matter. But because the debate is net US lives, the pro-net-benefit side must quantitatively reconcile those explanations with its claimed vaccine benefit.

Falsifier: a transparent US decomposition that explains the persistent elevation while preserving large mRNA net lives saved and no meaningful vaccine harm would weaken this claim.

## Claim 2: There Is a Serious US mRNA Harm Signal That Saar Must Answer

The Florida Levi/Ladapo analysis is central because it is a US mRNA active-comparator design: Pfizer versus Moderna among vaccinated recipients, with exact matching. It is not the usual vaccinated-versus-unvaccinated comparison, so generic healthy-vaccinee objections are less responsive. If two mRNA products were equally safe, strictly matched Pfizer and Moderna recipients should have similar NCACM.

The reported brand difference is therefore a serious unresolved harm signal. It does not by itself prove the no-vaccine counterfactual, and it may still have residual confounding, but it directly challenges the claim that US mRNA vaccination was safe enough that harm can be treated as negligible.

The FDA Prasad memo is relevant context because it weakens the argument from agency silence. Institutional non-admission is not proof of no vaccine deaths or no unresolved risk-benefit problem.

Causal interpretation: the US harm case does not depend on one anecdote or passive report. It is a convergence argument: persistent US mortality elevation, an active-comparator mRNA brand signal, official safety-signal context, and broader passive-surveillance/pathology/insurance signals.

Caveat: Levi/Ladapo is not a randomized trial and is not a direct estimate of total mRNA deaths. It should be treated as a high-priority signal, not a complete count.

Falsifier: Saar should produce a comparable or stronger US Pfizer-versus-Moderna strictly matched study showing NCACM equality, plus a transparent safety-signal adjudication showing why the FDA/VAERS concerns are non-causal.

## Claim 3: Large Durable Benefit Is Not Visible in Population-Level COVID-Death Measurements

The OWID slope screen asks a simple question: after vaccination and accumulating natural immunity, do later COVID-death waves show the downward slope fingerprint expected from large durable vaccine benefit?

The all-country screen fits high-linearity cumulative COVID-death wave segments with R^2 >= 0.99 across all OWID countries. The fitted trend in log10 slope goes upward, not downward, with a fitted end/start ratio of about 1.68x and low explanatory power. This is not what a clean large-benefit story would make easy to see.

The top25/bottom25 vaccination-contrast screen is even more direct. The top 25 and bottom 25 countries differ by about 13x in mean dose exposure, yet their fitted COVID-death wave-slope trends are nearly identical, about 0.66x versus 0.68x. Extreme vaccination exposure did not produce an obvious slope fingerprint.

Causal interpretation: this is a falsification screen, not a direct individual VE estimate. But if large VE-death claims were robust at the population level, they should not be this hard to see in broad, cross-country mortality measurements.

Caveat: countries differ in age structure, variant timing, testing/coding, prior infection, healthcare capacity, and wave intensity. Those can explain away a slope screen only if Saar quantifies them and shows they plausibly cancel the expected vaccine signal.

Falsifier: a pre-specified model or transparent stratified analysis showing the expected downward slope fingerprint after accounting for those variables would weaken this claim.

## Prebuttals to Saar's Likely Strongest Points

RCT and severe-COVID efficacy evidence does not answer the net US mortality question. The trials can support short-term biologic protection against symptomatic or severe disease, but they were not powered or followed long enough to settle ACM/NCACM net mortality through the end of 2022.

The observational VE-death literature is not automatically decisive because many studies share the same vulnerability: healthy-vaccinee effects, frailty differences, non-COVID mortality differences, prior infection, and testing/coding differences. A pile of studies with the same negative-control weakness is not equivalent to one clean study.

Lives-saved models are assumption engines unless they are validated against actual measurements. If a model says large net lives were saved, it must explain why the US mortality record stayed elevated, why high-vaccination countries do not show a clean downward COVID-death slope fingerprint, and why the US Florida mRNA brand-comparison NCACM signal is non-causal.

## Bottom Line

Under a 55% standard, Saar should not win by pointing to short-term efficacy or observational VE estimates unless he also reconciles the US mortality record and the US harm signals. My position is narrower but more measurement-grounded: large net lives saved is not visible in the real-world mortality record, and meaningful US mRNA harm remains unresolved. That makes negative net benefit at least as plausible as net lives saved through the end of 2022.
