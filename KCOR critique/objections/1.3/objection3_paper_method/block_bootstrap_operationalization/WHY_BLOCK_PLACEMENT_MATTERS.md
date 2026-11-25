# Why moving blocks of raw weeks changes the KCOR signal

The paper requires the resampled deaths and risk sets to obey the same survivor accounting as the original data. That is necessary, but it does not determine which weekly death pattern a bootstrap replicate represents.

## Mathematical argument

Write $\hat p_t=D_t/Y_t$ for the observed death fraction at week $t$ in one cohort, where $D_t$ is the death count and $Y_t$ is the number at risk. If the resampling rule moves source week $s$ to target week $t$, it sets the resampled death count to $Y_t^*\hat p_s$, up to rounding. Dividing by the reconstructed risk set $Y_t^*$ gives $\hat p_s$ again. Consequently the expected resampled fraction is

$$
E_*\!\left[\hat p_t^*\right]=\sum_s w(t,s)\hat p_s,
$$

where $w(t,s)$ is the probability of assigning source week $s$ to target week $t$, and the expectation is conditional on the original data. The difference from the rate at the target week is $\sum_s w(t,s)(\hat p_s-\hat p_t)$. Keeping source weeks close to $t$ can reduce this difference. Equality requires the weighted average to match the target.

Even with no waves, the KCOR model generally gives different death probabilities at different weeks. Let $B(t)$ be baseline cumulative hazard and let starting gamma frailty have mean one and variance $\theta>0$. Survival at time $t$ is $S(t)=[1+\theta B(t)]^{-1/\theta}$. The probability of dying during week $t$ conditional on being alive at its start is

$$
p_t=1-\frac{S(t+1)}{S(t)}
   =1-\left(\frac{1+\theta B(t+1)}{1+\theta B(t)}\right)^{-1/\theta}.
$$

The weekly baseline hazard and earlier depletion both enter this probability. The survivor distribution can stay gamma throughout and its mean still changes. Thus quiet weeks are generally different even under the working model.

With fixed follow-up and more independently sampled people, the observed fraction at each week approaches its own $p_t$. A raw-block rule with fixed placement probabilities still substitutes $p_s$ whenever it maps a source week $s$ to $t$. If it selects rates that differ, this variation persists as ordinary sampling noise shrinks. Correct survivor accounting leaves the substituted rate in place. Person resampling preserves each sampled person's death week or end-of-follow-up survival, so it varies the counts while preserving the original time structure.

## Numerical corroboration

On [exact expected counts](exact_counts_block_check.json), the paper's joint fit is essentially 1 before resampling, while this module's broad-segment four-week block bootstrap is centered near 1.10. In an exploratory [placement check](exact_counts_placement_check.json), restricting source blocks to roughly 12-week calendar segments reduced the center shift to about 0.011. These checks illustrate the mathematical mechanism. The [paired coverage study](../paired_resampling/ASSESSMENT.md) measures how it affected intervals under the tested rule.
