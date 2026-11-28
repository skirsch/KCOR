# Author proposal: an analytic companion to the Cox simulation

The author proposed adding a mathematical demonstration of the Cox departure alongside, or in place of, the simulation in §2.11.1. This is an optional manuscript addition originating with the author. The wording edits that address PR #5 are recorded separately in the review decisions.

Candidate mathematical companion to §2.11.1. The calculation below uses the two cohorts, equal enrollment sizes and common administrative censoring time in the [simulation generator](../../../test/sim_grid/code/generate_cox_bias_sim.py). The exposure indicator is the sole covariate in the ordinary Cox regression.

## Hazard ratio at a given time

Let both cohorts have individual hazard \(Z h_0(t)\), with common \(H_0(t)=\int_0^t h_0(s)\,ds\). The conditional treatment hazard ratio is 1. Cohort A has \(Z=1\). In B, \(Z\) is gamma with mean 1 and variance \(\theta>0\). A person with frailty \(z\) survives to \(t\) with probability \(\exp[-zH_0(t)]\). Taking its expectation over the enrollment distribution gives

$$
S_A(t)=e^{-H_0(t)},\qquad S_B(t)=(1+\theta H_0(t))^{-1/\theta}.
$$

Differentiating the negative log survival gives \(h_A=h_0\) and \(h_B=h_0/(1+\theta H_0)\). Thus the marginal B-to-A hazard ratio is

$$
R(t)=\frac{1}{1+\theta H_0(t)}.
$$

At enrollment \(R(0)=1\). It falls below 1 as soon as baseline hazard accumulates. Higher-frailty members of B die earlier on average. The mean frailty of survivors in B falls to \(1/(1+\theta H_0(t))\), while every survivor in A has frailty 1. The result applies to any common baseline hazard, including a Gompertz baseline. It uses the same gamma-mixture identity as the paper's cumulative-hazard inversion. [Hougaard, 1984](https://www.jstor.org/stable/2336399)

## Single ratio fitted across follow-up

Write \(q=e^\beta\) for a candidate B-to-A Cox ratio. In a risk set with equal initial enrollment and survival fractions \(S_A,S_B\), the Cox model assigns fraction \(qS_B/(S_A+qS_B)\) of events to B. The actual event rates per initial cohort member are \(S_A h_A\) and \(S_B h_B\). Subtracting the model-assigned B events from the actual B events yields the population partial-likelihood score, up to a positive constant. The limiting fitted ratio is the unique root of

$$
0=\int_0^T\frac{S_A(t)S_B(t)}{S_A(t)+qS_B(t)}
                \big[h_B(t)-q h_A(t)\big]\,dt .
$$

With \(\theta>0\) and positive baseline hazard over follow-up, the score is negative at \(q=1\) and positive at \(q=R(T)\). The score decreases strictly with \(q\), since the model-assigned fraction of B events increases in every risk set containing both groups. Hence \(R(T)<q<1\). This proves the direction of the ordinary Cox departure without drawing any random data.

For the checked-in settings, \(h_0=0.002\) per week and \(T=52\) weeks. The population Cox column below is from deterministic numerical integration and scalar root finding. The last column is the finite-sample fit in the [archived simulation results](../../../test/sim_grid/out/cox_bias_results.csv).

| Frailty variance in B | Week-52 hazard ratio \(R(T)\) | Population Cox ratio \(q\) | Simulated Cox ratio |
|---:|---:|---:|---:|
| 0 | 1.000 | 1.000 | 0.988 |
| 0.5 | 0.951 | 0.975 | 0.965 |
| 1 | 0.906 | 0.952 | 0.944 |
| 2 | 0.828 | 0.910 | 0.902 |
| 5 | 0.658 | 0.808 | 0.804 |
| 10 | 0.490 | 0.689 | 0.701 |
| 20 | 0.325 | 0.544 | 0.551 |

With the generating \(\theta\), inverting the population cumulative hazard in B gives \(\{\exp[\theta(-\log S_B)]-1\}/\theta=H_0\), so the corresponding KCOR population ratio is 1 for \(H_0(t)>0\). The simulation's KCOR calculation supplies the generating frailty variance. That calculation illustrates finite-sample variability around the inversion benchmark. Assessing the estimated-frailty pipeline also requires fitting the variance from the data.
