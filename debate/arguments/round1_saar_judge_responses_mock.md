# Round 1 Saar Judge Responses

Debate question: In the United States, did the mRNA COVID vaccines likely net save lives or cost lives through the end of 2022?

This document answers only the judge questions relevant to Saar, in the same neutral-Claude framing used for the mock debate.

## 1. Strongest US age-stratified estimate of lives saved through end-2022

The cleanest age-stratified US estimate in the packet is the Medicare-beneficiary model summarized in `SW argument 6-16 full.pdf`.

- Point estimate: about `330,000-370,000` lives saved in 2021 among Medicare beneficiaries, expanded in the packet to about `656,000` lives saved for the 2021-2022 period after scaling `+50%` for the full period and `+25%` for the broader population.
- Main assumptions:
  - the Medicare-beneficiary effect generalizes to the wider elderly US population;
  - the VE inputs used by the model are roughly correct;
  - the infection and herd-immunity counterfactuals are correctly specified;
  - later variant and waning dynamics do not destroy the modeled benefit.
- One observation that would most weaken it:
  - there is no obvious downward population fingerprint of that magnitude in the US mortality record or the broad OWID slope screens; if a benefit that large were real, it should be easier to see in measured cumulative mortality than it is.

I do not think this estimate should be treated as a direct all-cause mortality measurement. It is a model-based lives-saved estimate that inherits the model's assumptions.

## 2. Reconcile that estimate with the US UCOD record

The honest reconciliation is that UCOD does not give a clean all-cause subtraction of vaccine benefit. It gives a raw mortality record that still looks elevated versus 2019.

- All-age crude mortality in the UCOD summary was `878.0` per 100k in 2019, `1052.6` in 2021, and `992.7` in 2022.
- That is `+19.9%` in 2021 and `+13.1%` in 2022 versus 2019.
- The elevation was present even in younger groups:
  - ages 0-39: `+30.7%` in 2021 and `+20.7%` in 2022 versus 2019;
  - ages 40-59: `+31.7%` in 2021 and `+13.4%` in 2022 versus 2019;
  - ages 60-79: `+22.2%` in 2021 and `+12.9%` in 2022 versus 2019;
  - ages 80+: `+10.8%` in 2021 and `+3.8%` in 2022 versus 2019.

Using the rate deltas as a rough scale, the raw excess mortality relative to 2019 is on the order of about `0.58 million` in 2021 and `0.38 million` in 2022, or roughly `0.96 million` across the two years. I would not subtract the `656,000` model estimate one-for-one from that, because the model estimate and the UCOD record are not the same estimand.

So my answer is: the UCOD record still shows a large residual elevation, but that does not by itself prove vaccine harm. It does mean the pro-benefit case has to explain why the all-cause record stayed elevated even under a claimed large lives-saved model.

## 4. What exact fingerprint should have shown up by late 2022?

If large durable net benefit were real, I would expect a clear downward fingerprint in the measured population record.

- In the OWID slope screens, the fitted cumulative COVID-death wave-slope trend should have been visibly negative, not flat or upward.
- In highly vaccinated countries, later wave slopes should have been materially lower than earlier wave slopes after rollout and accumulating natural immunity.
- In the US mortality record, the post-rollout all-cause record should have bent downward enough that the large-benefit story was easy to see without heavy modeling.

If I have to give one pre-specified directional prediction, it is this: the fitted late-2022 trend should have gone down, not up.

## 5. Single best observational VE-death paper in the packet, and why it is not just the usual bias pattern

The strongest observational VE-death paper in the packet is the Hong Kong MMWR mortality report, not Kaiser.

Why I think it is the strongest:

- it is a real death endpoint, not just infection;
- Hong Kong was close to a zero-COVID / low-prior-immunity setting before Omicron;
- the analysis is age-specific and focused on the elderly, where the mortality signal matters most;
- the effect sizes are large, including a `21.3x` higher COVID-death risk among unvaccinated people aged 60+ versus recipients of 2-3 doses.

Why I do not think it is merely the same bias pattern as the weaker VE-death literature:

- the setting had much less room for prior-infection leakage than a typical Western cohort;
- the elderly strata were not an extreme tiny-unvaccinated fringe; the age groups closest to balance were around `48%` and `52%` unvaccinated in the packet's summary;
- the outbreak was a defined Omicron wave, so the endpoint is tied to a specific mortality surge rather than a long drifting calendar window.

That said, I do not treat it as bias-free. The missing negative-control check is still the biggest limitation, and the report does not by itself prove the no-vaccine counterfactual. I am saying only that, of the observational death papers in the packet, Hong Kong is the hardest one to dismiss as just ordinary HVE noise.

## Bottom line

My strongest answer remains narrow: the packet contains real evidence for meaningful short-term severe-COVID protection, but the US mortality record still does not give me a clean all-cause reconciliation. The best observational death paper is Hong Kong, yet it still does not fully settle the negative-control problem. That is why I would not claim more certainty than the evidence allows.
