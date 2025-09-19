# KCOR v4.4 - Kirsch Cumulative Outcomes Ratio Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [⚠️Limitations](#️-limitations)
- [🔬 Methodology](#-methodology)
  - [🎯 Core Concept](#-core-concept)
  - [⚙️ KCOR algorithm](#️-kcor-algorithm)
  - [Key Assumptions](#key-assumptions)
- [🏆 KCOR vs. Traditional Epidemiological Methods](#-kcor-vs-traditional-epidemiological-methods)
- [🏗️ Repository Structure](#️-repository-structure)
- [📦 Installation & Dependencies](#-installation--dependencies)
- [🚀 Usage](#-usage)
- [⚙️ Configuration](#️-configuration)
- [📊 Interpretation](#-interpretation)
- [🔧 Advanced Features](#-advanced-features)
- [🚨 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)
- [📚 Citation](#-citation)
- [Version History](#version-history)
  - [🆕 Version 4.3 Enhancements](#-version-43-enhancements)
  - [🆕 Version 4.2 Enhancements](#-version-42-enhancements)
  - [🆕 Version 4.1 Enhancements](#-version-41-enhancements)
- [📊 Results Using Czech Data](#-results-using-czech-data)
- [🔬 Validation](#-validation)
  - [Independent Validation Methods](#independent-validation-methods)
  - [Negative-Control Tests](#negative-control-tests)
  - [Sensitivity Analysis](#sensitivity-analysis)
- [Peer review](#peer-review)
- [📄 License](#-license)
- [📞 Contact](#-contact)

## Overview

KCOR (Kirsch Cumulative Outcomes Ratio) is a robust statistical methodology for analyzing relative mortality risk between different vaccination groups while accounting for underlying mortality rate time trend differences. This repository contains the complete analysis pipeline for computing KCOR values from mortality data.

KCOR enables us, for the first time, to objectively answer critically important questions such as, "Was the COVID vaccine net beneficial by the end of 2022?" KCOR tells you whether the benefits (e.g., lives saved during COVID) outweighed the risks (e.g., people who who were killed by the vaccine) at any time t.

This is important because not a single epidemiologist in the entire world has been able to take any record level dataset (such as the Czech data) and answer that crucial question. That is an epic failure of epidemiology.

Grok wrote, "KCOR addresses a real gap: traditional epidemiology often struggles with net benefit assessments in retrospective data without randomization." That's why KCOR is so important.

Suppose you could take any two cohorts, regardless of age, sex, frailty mix, etc. and normalize their baseline mortality rates so that if there is no external signal applied that might *differentially* impact their mortality, both cohorts would die over time with identical mortality rates.

That's what KCOR does. Once the cohorts are precisely matched from a mortality rate point of view, we can simply cumulate the adjusted hazards and see which cohort had more cumulative deaths as a function of t. This means, given a specific time, we can show if an intervention was net harm, net benefit, or neutral as of that time. 

Therefore, any methodolgy that can accurately make such an risk/benefit assessment is a threat to the field because it will reveal that previous assessments done with traditional epidemiology methods were all flat out wrong (e.g., studies claiming the COVID vaccine were beneficial). This is why KCOR is either being ignored or denigrated by those in the field, despite high praise by people such as prominent US epidemiologist Yale Professor Harvey Risch (h-index 119).

KCOR is similar to running a randomized trial with respect to the death outcome, using retrospective observational data. It's certainly not perfect, and certainly not as good as a randomized trial, but with retrospective observational data, it's about the best you can do. No 1:1 cohort matching is required. No cause of death is needed. KCOR uses just 3 dates per person: birth, death, and dates of vaccination(s). That's it.

KCOR is objective. It does not have a bias. It doesn't have a belief in vaccine safety. It simply compares mortality rates between mortality rate matched cohorts and tells you which cohort died less.

There is also little ability to game the result since the parameters (enrollment dates, slope start/end dates) are determined by the data itself.

### Grok review

Grok (expert mode), after being shown that it's review of KCOR was flawed, finally admitted, "I would now recommend it as a legitimate method for assessing vaccine harm/benefit in observational data. It's particularly valuable in settings like the Czech or Qatar datasets, where traditional matching/Cox struggles with residual HVE and unobserved confounders, as KCOR's empirical measurement and normalization offer a transparent, minimal-assumption alternative that aligns with survival analysis principles while being robust to time-varying trends." 

Here is [the full discussion with Grok](https://grok.com/share/c2hhcmQtMg%3D%3D_c35e85ae-f70c-409d-b91f-96193d23e381).

The [results section](#-results-using-czech-data) shows that the COVID vaccines caused significant net harm regardless of age. Boosters were even worse than primary vaccination. The summary here is for all ages, but the statistically significant harms were present for all age groups individually. 

Those who claimed that the COVID shots were beneficial for the elderly were speculating based on no data; they shouldn't have done that. The Czech data clearly shows that such advice was deadly (see the one dose response with the earlier enrollment date). 

The [validation section](#-validation) covers the sensitivity tests, negative control tests, and validation of the results using three different methods: DS-CMRR, GLM, and Kaplan-Meier survival plots. In fact, the DS-CMRR and GLM plots are very similar in shape to the KCOR plots. 

The [Czech Republic record level dataset](https://www.nzip.cz/data/2135-covid-19-prehled-populace) is the most comprehensive publicly available dataset for the COVID vaccine in the world. Yet not a single epidemiologist has ever published an analysis of this data. KCOR reveals why.

You can see the [full summary of the results for the Czech data here](data/Czech/KCOR_summary.log).

There isn't a legitimate critique of KCOR that I'm aware of. See the [Peer Review section](#peer-review) for details.

The bottom line is that KCOR works extremely well with real world cohorts of sufficient size like the Czech Republic 11M record level dataset. It is very easy to validate the key KCOR assumption of an exponential mortality rate before applying the method.

I would be delighted to public debate any qualified scientist who believes KCOR is flawed. This would end the debate. No takers unfortunately. 

Martin Kulldorff wrote, "When there are different scientific views, only trust scientists who are willing to engage with and publicly debate the scientists with other views.”

## ⚠️ Limitations
There are 6 key limitations of the method that users should be aware of. 

In general, these limitations cause KCOR to be a **conservative estimator of harm**. This means if KCOR finds a harm signal, like it does in the Czech dataset, the actual harm is actually greater than KCOR indicates because KCOR tends to *understate* harms and *overstate* benefits as specifically described below.

1. **Exponential fit assumption:** Cohorts aged 90 and older with significant frailty will not be as accurate as cohorts of younger ages because the core assumption of a single exponential mortality rate starts to become less true. Estimates from these cohorts may be inaccurate by more than 1%. Grok did a compuation for age 90 with a frailty mix of 1-4 and [found less than a 1.6% per year error from the exponential assumption](https://grok.com/share/c2hhcmQtMg%3D%3D_924f6b7d-543f-4ebb-bc82-7cfd8eef297c).

2. **Fixed cohort asssumption:** KCOR uses fixed cohorts defined at specific enrollment dates. All of those cohorts may change their vaccine status over time and that is not reflected in the analysis. The enrollment dates are generally chosen after 80% to 90% of the people likely to die have been vaccinated to minimize this impact. For a vaccine which reduces risk of death, this has the effect of reducing the magnitude of the harm or benefit because the cohorts will not be as differentiated later in time. So KCOR will *understate* the harm and *understate* the benefits.

3. **Non-proportional hazards:** KCOR, in its current form, does not yet adjust for non-proportional hazards where the mortality differences in people with the SAME age (e.g., 5 year age band for the Czech data) are not proportional to their baseline mortality. This is particularly important for the COVID vaccine where the mortality increase in response to a virus wave is extremely sensitive to unmeasurable confounders. KCOR will *overstate* the net benefit during virus periods giving 100% credit to the vaccine when in fact, the protective effect could be 100% due to selection bias causing the unvaccinated to have higher fraility than would be assumed from the DCCI values. For example, the relative mortality increase of two 90 year olds during COVID waves, one vaccinated, the other unvaccinated, is be remarkably different. The percentage of that differential mortality increase from COVID credited to the COVID vaccine vs. differential frailty created by the selection bias is the subject of disagreement. Scientists supportive of the COVID vaccine are unwilling to actually debate this topic in a public forum so this remains unresolved. It is an "untouchable" subject because resolving the issue could cause the public to distrust mainstream scientists. There are ways to assess this, e.g., by looking at whole population cumulative deaths during vaccine rollout which was during a COVID wave to look for a "knee" in the curve. If the vaccine really worked, there will be a knee at the time the shots rolled out. If the COVID benefit was all selection bias, there will be no knee.

4. **Harm during baselien period:** KCOR needs a baseline period when there is no COVID to assess the relative mortality rates of the cohorts under study when there is not an external intervention that is supposed to cause a differential response.  But if the vaccine is unsafe, it will increase mortality in this period to an artifically high level. This will always cause KCOR to *understate* the true harm of the vaccine.

5. **Late enrollment:** If an enrollment date is chosen that is relatively distant from after most people a cohort have been vaccinated and the vaccine significantly increases non-COVID ACM (NCACM) that then plateaus as with the COVID shots, KCOR will miss this for those older cohorts and show a neutral or even a net benefit. It's important to interpret the results in light of this, e.g., for older cohorts, the earlier enrollment dates will be more reliable indicators of risk/benefit.

6. **Dynamic HVE:** In general, because KCOR uses a calendar time-series (the x axis is a calendar date) rather than event time-series (where the x axis is time since the injection), dynamic HVE is virtually non-existent because most all the people got vaccinated well before the enrollment date. Dynamic HVE is caused when people avoid getting vaccinated because they are going to die. This transfers deaths from the vaccinated cohort to the less vaccinated cohort, e.g., 3 dose to 2 dose. We can show by plotting deaths per week that the Dose 2,1, and 0 cohorts all track each other post booster enrollment. This falsifies claims of dynamic HVE. Another way to test for this is to run the algorithm with DYNAMIC_HVE_SKIP_WEEKS set to 1 or 2 and see if it materially change the results. If there is an effect, increasing DYNAMIC_HVE_SKIP_WEEKS will make the vaccine look safer because dynamic HVE would set an artifically low baseline for the vaccinated. The default for DYNAMIC_HVE_SKIP_WEEKS is 0 because dynamic HVE is negligible. For event time-series, HVE is generally insigificant by week 3, so using a value of 2 is a reasonable sensitivity test to assess this effect. However, if the vaccine increases NCACM for a period post-shot like the COVID vaccine does, increasing this value will likely result in artifically increasing vaccine safety (lowering all the KCOR numbers). Therefore, testing the dynamic HVE effect by inspection of the deaths/week curves of the cohorts post enrollment is the best way and the clearest is post-booster rather than post-primary two shots because there is only the 1 cohort that would accept the deferred deaths and there are two cohorts to compare to for what "baseline" should look like (dose 1 and 0 groups).

## 🔬 Methodology

### 🎯 Core Concept

KCOR represents the ratio of cumulative hazard functions between two groups (e.g., vaccinated vs. unvaccinated), normalized to 1 at a baseline period. This approach provides interpretable estimates of relative mortality risk that account for:

 - **Time-varying trends** in mortality rates through slope correction
 - **Mathematical exactness** through discrete hazard function transformation
 - **Baseline differences** between groups through normalization
 - **Statistical uncertainty** in the estimates through proper variance propagation

The KCOR algorithm uses fixed cohorts defined by their vaccine status (# of shots) on an enrollment date and tracks their mortality over time. It relies on Gompertz mortality with depletion which is industry standard. It turns out that any large group of people will die with a net mortality rate that can be approximated by a single exponential with high accuracy (this is the "engineering approximation" epidemiologist Harvey Risch refers to in his [review](#peer-review)). 

KCOR relies on a very simple engineering approximation that can be easily validated using Gompertz mortality with depletion and frailty: over a two year period, even a 90 year old cohort with frailty 2 will die on nearly a straight line (less than 1.6% deviation over a year). If you now mix together cohorts with different frailties, the mortality rate of the combined cohort (e.g., an unvaccinated cohort of 90 year olds) is well-approximated by a single exponential—and KCOR's slope-normalization behave as intended. The accuracy increases with cohorts younger than 90 years old. KCOR simply can't be invalidated using Gompertz mortality. So unless Gompertz mortality with depletion is overturned, KCOR is not invalidated.

A [concise, easy to understand, visual guide to KCOR](documentation/KCOR_Visual_Guide.pdf) describes each of the KCOR steps using a concrete example. The document was prepared by an honest epidemiologist who chooses to remain confidential for fear of being fired for not supporting the "safe and effective" narrative. 

There is also the latest draft of the [KCOR paper](documentation/KCOR_Method_Paper.docx) for submission to medical journals.

 The core steps are:
 1. Decide on the enrollment date(s), slope start/end dates. The enrollment dates are chosen when most of a cohort under interest has been vaccinated. The two slope dates are two widely separated quiet periods when the smoothed mortality (smoothing is done with a centered window) is in a trough (quiet periods with no COVID that might affect differential mortality). 

 2. Run the algorithm.

 These 3 parameters above are largely dictated by the data itself. There can be multiple choices for each of these parameters, but generally, the data itself determines them. A future version of KCOR will be able to make these decisions independently from the data. For now, they are made manually. 

 The algorithm does 3 things to process the data:
 1. Slope normalizes the weekly mortality rates of the cohorts being studied using the slope start/end dates to assess baseline mortality slope of the cohort. Week 0 (enrollment week) is left unscaled; mortality rate slope normalization is applied from week 1 onward.
 2. Computes the ratio of the cumulative hazards of the cohorts relative to each other as a function of time which provides a net/harm benefit readout at any point in time t. KCOR uses the discrete hazard function transform to enable this.
 3. Normalizes the final ratio to the ratio at the end of a 4‑week baseline period (week 4). 

 The algorithm depends on only three dates: birth, death, vaccination(s). 
 
 Weekly resolution is fine for vaccination and deaths; 5 or 10 year age ranges for the year of birth are fine. This avoids triggering privacy excuses for not providing the data. The algorithm can also be used on summary files created by aggregating the data for specific enrollment dates, for example, as done in the KCOR_CMR.py script. Such data summaries do not violate any privacy laws. There is no excuse for not providing these.

 Note: An optional "baseline correction" addition to the algorithm was made to adjust the baseline for cohorts where the people got vaccinated well before the enrollment date. This is disabled by default so that the results are truly unbiased.

### ⚙️ KCOR algorithm

#### 1. Data Preprocessing
- **Enrollment Date Filtering**: Data processing starts from the enrollment date derived from sheet names (e.g., "2021_24" = 2021, week 24, "2022_06" = 2022, week 6)
- **Sex Aggregation**: Mortality data is aggregated across sexes for each (YearOfBirth, Dose, DateDied. DCCI) combination
- **Smoothing**: 8-week centered moving average applied to raw mortality rates to reduce noise

#### 2. Slope Calculation (Dynamic Quiet-Period Anchors)
- **Quiet-Period Calendar Candidates**: `2022-25`, `2023-28`, `2024-15` (ISO year-week)
- **Automatic Anchor Selection**:
  - First anchor = first candidate date that is at least `MIN_ANCHOR_GAP_WEEKS` (default: 26) after enrollment
  - Second anchor = first candidate date at least `MIN_ANCHOR_SEPARATION_WEEKS` (default: 39) after the first
  - Each anchor snaps forward to the next available data date on or after the candidate
- **Window Approach**: Around each anchor, use a ±`SLOPE_WINDOW_SIZE` week window (default: 2) for stability
- **Geometric Mean on Smoothed MR**: Compute geometric mean over `MR_smooth` in each window; slope is `r = (1/Δt) × ln(B̃/Ã)`
- **Logging**: Chosen anchor dates and Δt are printed for each enrollment cohort in the log

**Slope Formula:**

$$r = \frac{1}{\Delta t} \ln\left(\frac{\tilde{B}}{\tilde{A}}\right)$$

Where:
- **Ã** = Geometric mean of MR values in window around first anchor: $$\tilde{A} = \text{GM}(MR_{t \in [t_0-w, t_0+w]})$$
- **B̃** = Geometric mean of MR values in window around second anchor: $$\tilde{B} = \text{GM}(MR_{t \in [t_1-w, t_1+w]})$$
- **Δt** = Time difference between anchor points (in weeks)
- **w** = Window size (default: 2 weeks)

**Geometric Mean Calculation:**

$$\text{GM}(x_1, x_2, \ldots, x_n) = e^{\frac{1}{n} \sum_{i=1}^{n} \ln(x_i)}$$

- **Consistency**: Same anchor points used for all doses for a given enrollment date to ensure comparability
- **Quiet Periods**: Anchor dates chosen during periods with minimal differential events (COVID waves, policy changes, etc.)

#### 3. Mortality Rate Adjustment Using the Computed Slopes (r)
- **Individual MR Adjustment**: Apply slope correction to each mortality rate for a given enrollment, age, dose combination to create an adjusted mortality rate: 

$$\text{MR}_{\text{adj}}(t) = \text{MR}(t) \times e^{-r(t - t_0)}$$

- **Anchoring**: tₑ = enrollment week index (tₑ = 0)
- **Dose-Specific Slopes**: For a given enrollment date, each dose-age combination gets its own slope for adjustment

#### 4. KCOR Computation 
**Three-Step Process:**

1. **Hazard Transform**: Convert adjusted mortality rates to discrete hazard functions for mathematical exactness  
2. **Cumulative Hazard**: Compute CH as cumulative sum of hazard functions
3. **Ratio Calculation**: Compute KCOR as ratio of cumulative hazards, normalized to baseline

**Step 1: Discrete Hazard Function Transform**

$$\text{hazard}(t) = -\ln(1 - \text{MR}_{\text{adj}}(t))$$

Where MR_adj is clipped to 0.999 to avoid log(0).

> **📚 Mathematical Reasoning**: For a detailed explanation of why KCOR uses discrete hazard functions and the mathematical derivation behind this approach, see [Hazard Function Methodology](documentation/hazard_function.md).

**Step 2: Cumulative Hazard (CH)**

$$\text{CH}(t) = \sum_{i=0}^{t} \text{hazard}(i)$$

**Step 3: KCOR as Hazard Ratio (Baseline at Week 4)**
By default, KCOR cumulate hazards for 5 weeks (week 0 to week 4) and uses the cumulated hazard ratio at that time to establish a reference hazard ratio where KCOR=1. Increasing this parameter will reduce the CI's (which are largely determine by the number of weeks used to establish the baseline ratio), but it will also result in the method missing vaccine harms (the baseline is done during low to no COVID so it won't miss any benefits). So 5 was a reasonable compromise. 

KCOR starts accumulating hazards on the enrollment date to capture a baseline mortality as close to vaccination as possible. The Czech data had no signs of dynamic HVE with KCOR fixed cohort enrollment dates since the enrollment dates are chosen after 80% of the cohort being studied has been vaccinated. 

Dynamic HVE is caused by people who are going to die shortly declining to be vaccinated. It looks like two highways merging if you look at a plot of deaths per week.

If examination of the deaths/week data shows signs of dynamic HVE, then you can either shift the enrollment date later, or set DYNAMIC_HVE_SKIP_WEEKS to a value other than 0. Setting DYNAMIC_HVE_SKIP_WEEKS >3 would be highly unusual since event time-series plots for vaccines rarely (if ever) have dynamic HVE lasting over 3 weeks. In the case of COVID, if anything, HVE would be very small since even people who were dying wanted to see their familty and the familty would demand vaccination.

**KCOR Formula:**

$$\text{KCOR}(t) = \frac{\text{CH}_v(t) / \text{CH}_u(t)}{\text{CH}_v(t_0) / \text{CH}_u(t_0)}$$

Where:
- **r** = Calculated slope for the specific dose-age combination
- **MR(t)** = Raw mortality rate at time t
- **tₑ** = Enrollment week index (0)
- **t₀** = Baseline for normalization (week 4; KCOR(t₀) = 1)
- **CH(t)** = Cumulative hazard at time t (sum of discrete hazards)
- **Mathematical Enhancement**: Discrete cumulative-hazard transform provides more exact CH calculation than simple summation
- **Interpretation**: KCOR = 1 at baseline, showing relative risk evolution over time

#### 5. Uncertainty Quantification
**95% Confidence Interval Calculation:**

The variance of KCOR is calculated using proper uncertainty propagation for the hazard ratio:

$$\text{Var}[\ln(\text{KCOR}(t))] = \frac{\text{Var}[\text{CH}_v(t)]}{\text{CH}_v(t)^2} + \frac{\text{Var}[\text{CH}_u(t)]}{\text{CH}_u(t)^2} + \frac{\text{Var}[\text{CH}_v(t_0)]}{\text{CH}_v(t_0)^2} + \frac{\text{Var}[\text{CH}_u(t_0)]}{\text{CH}_u(t_0)^2}$$

**Confidence Interval Bounds:**

$$\text{CI}_{\text{lower}}(t) = \text{KCOR}(t) \times e^{-1.96 \sqrt{\text{Var}[\ln(\text{KCOR}(t))]}}$$

$$\text{CI}_{\text{upper}}(t) = \text{KCOR}(t) \times e^{1.96 \sqrt{\text{Var}[\ln(\text{KCOR}(t))]}}$$

Where:
- **Var[CH] ≈ CH**: Using Poisson variance approximation for cumulative hazard (sum of hazards)
- **Var[ln(KCOR)]**: Variance on log scale for proper uncertainty propagation of hazard ratio
- **1.96**: 95% confidence level multiplier (standard normal distribution)
- **Log-Scale Calculation**: CI bounds calculated on log scale then exponentiated for proper asymmetry

#### 6. KCOR Normalization Fine-Tuning (v4.3+): Disabled by default
**Optional Baseline Correction for Unsafe Vaccine Effects:**

When unsafe vaccines create artificially high baseline mortality rates during the normalization period, KCOR assumes this is just normal mortality for the vaccinated. This may cause KCOR values at the end of the study period to be less than 1, making it appear that that an unsafe vaccine saved lives when in reality what was happening is that the mortality increase caused by the vaccine was just ephemeral and the enrollment date happened to correspond to peak mortality. 

This situation happens when the vaccine is unsafe and the enrollment date is many weeks after most people in that age group got their shots. So the KCOR value for those age groups are artifically low.

This optional feature (which is disabled by default) corrects for this bias by adjusting the KCOR scaling  to end up at a sensible final value, e.g., 1 for an unsafe vaccine.

This feature is disabled by default since it creates a bias. It can be enabled if you want to get more accurate ASMR final values that would not be affected by this limitation. 

So for most accurate results, use multiple enrollment dates to determine the KCOR values for different aged cohorts when the cohorts are vaccinated over a wide calendar range. 

The parameter is just a quick way to get more realistic KCOR values without having to examine multiple cohorts.

In the future, we'll automatically create fixed cohorts enrollment date for every 10 year age group to account for calendar staggered vaccine rollouts.

**Detection Logic:**
1. Compute KCOR values normally using enrollment-based baseline normalization (week 1)
2. Check KCOR value at specified final date (default: April 1, 2024)
3. If final KCOR < threshold (= FINAL_KCOR_MIN; default: 0 disables scaling), adjust scale factor = 1/final_KCOR
4. Apply adjusted scale factor to all KCOR computations

**Scaling Formula:**

$$\text{KCOR}(t) = \frac{\text{Kraw}(t)}{\text{baseline\\_kraw}} \times \text{scale\\_factor}$$

Where:
- **KCOR_final_date** = KCOR value at the specified final date with original normalization
- **Scale factor** = 1/KCOR_final_date when KCOR_final_date < threshold
- **Applied to**: Scale factor only (preserves all K_raw relationships)

**Configuration Parameters:**
```python
FINAL_KCOR_MIN = 0              # Setting to 0 DISABLES scaling based on the final value
FINAL_KCOR_DATE = "4/1/24"      # Date to check for scaling (MM/DD/YY format)
```

**Key Benefits:**
- **Corrects Baseline Bias**: Fixes artificially low KCOR values due to unsafe vaccine effects
- **Automatic Detection**: Only applies scaling when needed (KCOR < threshold)
- **Preserves Relationships**: Maintains relative differences between all time points
- **Transparent Process**: BEFORE/AFTER anchor values logged for full transparency
- **Conservative Approach**: Only scales when clear evidence of baseline bias exists

#### 7. Age Standardization 
**Expected-Deaths Weighting for ASMR Pooling:**

The age-standardized KCOR uses expected-deaths weights that properly reflect actual mortality burden:

$$\text{KCOR}_{\text{ASMR}}(t) = e^{\frac{\sum_i w_i \ln(\text{KCOR}_i(t))}{\sum_i w_i}}$$

Where:
- **wᵢ** = Expected-deaths weight for age group i
- **KCORᵢ(t)** = KCOR value for age group i at time t
- **ln(KCORᵢ(t))** = Natural logarithm of KCOR for age group i

**Expected-Deaths Weight Calculation:**

$$w_i = \frac{h_i \times \text{PT}_i(W)}{\sum_j h_j \times \text{PT}_j(W)}$$

Where:
- **hᵢ** = Smoothed mean mortality rate for age group i in quiet window W
- **PTᵢ(W)** = Person-time for age group i in quiet window W
- **W** = Quiet baseline window (first 4 distinct weeks)
- **Normalization**: Weights sum to 1.0 across all age groups

**Key Improvements (v4.2):**
- **Death Burden Focus**: Weights based on expected deaths (hazard × person-time) rather than just person-time
- **Elderly Properly Weighted**: Age groups with higher death rates get appropriate weight
- **Young Under-Weighted**: Age groups with low death rates get reduced weight
- **Mathematical Correctness**: ASMR now reflects actual mortality impact, not population size
- **Robust Implementation**: Uses pooled quiet baseline window with smoothed mortality rates

### Key Assumptions

- Mortality rates follow exponential trends during the observation period
- No differential events affect dose groups differently during anchor periods
- Baseline period (week 4) represents "normal" conditions
- Person-time = Alive (survivor function approximation)
- Discrete hazard function transformation provides accurate cumulative hazard estimation
- Hazard ratios are appropriate for comparing mortality risk between groups

## 🏆 KCOR vs. Traditional Epidemiological Methods

KCOR represents a groundbreaking advancement in epidemiological methodology, offering unique advantages over traditional approaches for comparing mortality between cohorts:

#### Traditional Methods vs. KCOR

| Aspect | Traditional Methods | KCOR |
|------------|------------------------|----------|
| Time-Varying Trends | ❌ Assume static baseline rates | ✅ Dynamic slope correction |
| Mathematical Rigor | ❌ Often use approximations | ✅ Discrete hazard functions |
| Baseline Control | ❌ Compare absolute rates | ✅ Normalized to matched baseline |
| Observational Data | ❌ Require randomized trials | ✅ Creates "virtual randomization" |
| Policy Questions | ❌ Limited applicability | ✅ Direct policy evaluation |

#### Why KCOR is Superior

🎯 Unique Problem Solving:
- Traditional SMR: Assumes static reference population rates → fails with time-varying trends
- KCOR: Dynamically adjusts for secular changes, seasonal effects, and policy impacts

🔬 Mathematical Excellence:
- Traditional Methods: Use approximations or assume proportional hazards
- KCOR: Uses exact discrete hazard transformation: `hazard(t) = -ln(1 - MR_adj(t))`

⚖️ Baseline Matching:
- Traditional Methods: Compare absolute rates between potentially different cohorts
- KCOR: Normalizes to baseline period where cohorts are "matched" from mortality perspective

🌍 Real-World Applicability:
- Traditional Methods: Require controlled conditions or make unrealistic assumptions
- KCOR: Works with observational data to answer policy-relevant questions

#### KCOR's Unique Value Proposition

KCOR is the only method that can:
- ✅ Create "virtual randomization" from observational data
- ✅ Dynamically adjust for time-varying trends affecting both cohorts  
- ✅ Provide mathematically exact hazard-based comparisons
- ✅ Answer policy-relevant questions using real-world data
- ✅ Handle COVID-era complexity with multiple confounding factors

Result: KCOR can objectively answer questions like "Did COVID vaccines kill more people than they saved?" using observational data—something no traditional epidemiological method can achieve.

#### Limitations of Traditional Epidemiological Methods

📊 Standardized Mortality Ratio (SMR)
- ❌ Assumes static reference population rates
- ❌ Doesn't account for time-varying trends  
- ❌ Vulnerable to secular changes in mortality
- ❌ Cannot handle COVID-era policy impacts

📈 Age-Period-Cohort (APC) Analysis
- ❌ Complex identifiability issues
- ❌ Requires large datasets
- ❌ Doesn't provide direct cohort comparisons
- ❌ Difficult to interpret for policy questions

⚖️ Proportional Hazards Models
- ❌ Assumes proportional hazards (often violated)
- ❌ Doesn't handle time-varying effects well
- ❌ Requires sophisticated statistical modeling
- ❌ Vulnerable to model misspecification

📋 Life Table Analysis
- ❌ Doesn't account for external time-varying factors
- ❌ Assumes stable mortality patterns
- ❌ Less suitable for policy evaluation
- ❌ Cannot handle rapid changes in mortality

🎯 Competing Risks Analysis
- ❌ Focuses on cause-specific mortality
- ❌ Requires detailed cause-of-death data
- ❌ Doesn't address overall mortality differences
- ❌ Complex interpretation for policy makers

#### The KCOR Advantage in Practice

🔬 Scientific Rigor:
- KCOR provides mathematically exact comparisons using discrete hazard functions
- Traditional methods rely on approximations that can introduce bias
- KCOR's approach is more robust to violations of common statistical assumptions

🌍 Real-World Relevance:
- KCOR works with the messy, complex data of real-world policy implementation
- Traditional methods require idealized conditions that rarely exist in practice
- KCOR can handle the rapid changes and multiple confounding factors of the COVID era

📊 Policy Impact:
- KCOR directly answers policy-relevant questions using observational data
- Traditional methods often require randomized trials that are impossible for policy evaluation
- KCOR provides interpretable results that policymakers can understand and act upon

⚡ Practical Implementation:
- KCOR requires only basic demographic and mortality data (birth, death, vaccination dates)
- Traditional methods often require extensive additional data (cause of death, detailed covariates)
- KCOR can be applied to existing datasets without additional data collection

## 🏗️ Repository Structure

```
KCOR/
├── README.md                           # This file
├── code/
│   ├── KCOR.py                      # Main analysis script (v4.4)
│   ├── KCOR_CMR.py                    # Data aggregation script
│   ├── Makefile                        # Build automation (Windows/Linux/Mac)
│   ├── debug/                          # Helper scripts for development/verification
│   └── old/                            # Archived scripts
├── data/                               # Outputs organized by country (e.g., Czech)
│   └── [country]/                     # Country-specific outputs (KCOR.xlsx, KCOR_summary.xlsx, KCOR_CMR.xlsx)
├── analysis/                           # Analysis artifacts and plots
│   └── [country]/                     # e.g., analysis/Czech/KCOR_analysis.xlsx, KCOR_ASMR_dose2.png
├── documentation/                      # Detailed methodology documentation
│   └── hazard_function.md             # Mathematical reasoning for hazard functions
├── validation/                         # Independent validation suites
│   ├── DS-CMRR/                       # Discrete Survival CMRR method
│   ├── GLM/                           # Generalized Linear Models validation
│   └── kaplan_meier/                  # Kaplan–Meier survival analysis
├── test/                               # Tests orchestrated by root Makefile
│   ├── negative_control/              # Synthetic no-signal tests
│   └── sensitivity/                   # Parameter sweep sensitivity tests
├── reference_results/                  # Frozen reference outputs for comparison
│   ├── KCOR/                          # Reference KCOR outputs
│   ├── GLM/                           # Reference GLM plots
│   ├── DS-CMRR/                       # Reference DS-CMRR plots
│   └── negative_control_tests/        # Reference negative-control outputs
└── peer_review/                        # Peer review materials
```

### Build structure

- Root `Makefile` orchestrates common tasks:
  - `make` → runs variable-cohort aggregation, analysis, validation, and tests
  - `make run` → main KCOR pipeline (delegates to `code/Makefile KCOR`)
  - `make validation` → DS-CMRR + KM + GLM validation (delegates to `validation/DS-CMRR/`, `validation/kaplan_meier/`, and `validation/GLM/`)
  - `make test` → runs both negative-control and sensitivity tests (delegates to `test/Makefile`)
- Important: Always run these targets from the repository root so environment and output paths are consistent.
- Subdirectory Makefiles (`code/`, `validation/DS-CMRR/`, `validation/kaplan_meier/`) are for advanced use only; invoking them directly may bypass root defaults and write outputs to different locations.

### Negative-Control Tests

Builds synthetic no-signal cohorts to ensure no false positives.
- Run: `make test`
- Outputs: `test/negative_control/out/` (e.g., `KCOR_processed_neg_control.xlsx`, `KCOR_summary.xlsx`)
- References: `reference_results/negative_control_tests/`

The analysis directory has human analysis of the data that shows that KCOR picks out real signals in the data that most people would have thought was perfect negative control data.

For the negative control tests, KCOR is called with baseline minimum set to 0 so that the KCOR baseline is not adjusted since we aren't dealing with a known net harmful vaccine.

### Sensitivity Analysis

Verifies that reasonable parameter choices do not change KCOR's conclusions by sweeping user-specified parameters.

How to run from repo root:
```bash
make sensitivity
```

Defaults (see `test/sensitivity/Makefile`):
- `SA_COHORTS=2021_24`
- `SA_DOSE_PAIRS=1,0;2,0`
- `SA_SLOPE_START=53,53,1`
- `SA_SLOPE_LENGTH=61,61,1`
- `SA_YOB=0` (ASMR only)

Key parameters:
- `SA_COHORTS`: comma-separated sheet names (e.g., `2021_24,2022_06`)
- `SA_DOSE_PAIRS`: semicolon-separated dose pairs (e.g., `1,0;2,0`)
- `SA_SLOPE_START`: `start,end,step` for offset1 (e.g., `52,60,2`)
- `SA_SLOPE_LENGTH`: `start,end,step` for Δt (e.g., `48,70,2`)
- `SA_YOB`: `0` (ASMR) | `start,end,step` | explicit `list`

Output:
- `test/sensitivity/out/KCOR_SA.xlsx` (sheet `sensitivity`)
- `test/sensitivity/out/KCOR_summary_SA.log`

## 📦 Installation & Dependencies

### Requirements
- Python 3.8 or higher
- pandas
- numpy
- openpyxl (for Excel output)

### Setup
```bash
# Clone the repository
git clone https://github.com/skirsch/KCOR
cd KCOR

# Install dependencies
pip install pandas numpy openpyxl

# That's it — Czech data is included under data/Czech. Run:
make
```

## 🚀 Usage

### Quick Start

#### Using Make (Cross-Platform)
Root Makefile orchestrates both the KCOR pipeline and the validation suite.
```bash
# From repo root
make                    # runs analysis (run) + validation + tests
make run                # main KCOR pipeline
make validation         # DS-CMRR + KM validation
make test               # negative-control and sensitivity tests (see test/)

# Dataset targeting (default DATASET=Czech)
make DATASET=Czech
make run DATASET=USA
make sensitivity DATASET=Czech
```

Notes:
- `make run` delegates to `code/Makefile KCOR`.
- `make validation` delegates to `validation/DS-CMRR/Makefile run`.
- Subdirectory Makefiles remain runnable directly; use `make -C <dir> <target>`.

 

#### Direct Python Execution
```bash
cd code
# Step 1: Data aggregation
python KCOR_CMR.py [input_file] [output_file]

# Step 2: KCOR analysis
python KCOR.py [aggregated_file] [analysis_output] [mode] [log_filename]
# Notes:
# - mode (e.g., "Primary Analysis" | "Sensitivity Analysis") is required
# - log_filename is optional (defaults to "KCOR_summary.log")
# Output appears both on console and in the specified log file
```

### Data Requirements

The Czech data files needed for running examples and validation are already included in this repository under `data/Czech/`. No additional downloads are required to run the default pipeline and validations.

#### Input Data Format

The script expects Excel workbooks with the following schema per sheet:

| Column | Description | Example |
|--------|-------------|---------|
| `ISOweekDied` | ISO week number of death | 24 |
| `DateDied` | Date of death | 2021-06-14 |
| `YearOfBirth` | Birth year | 1940 |
| `Sex` | Gender (M/F) | M |
| `Dose` | Vaccination dose | 0, 1, 2, 3 |
| `Alive` | Person-time (survivors) | 1500 |
| `Dead` | Death count | 25 |

### Output Files

The analysis produces Excel workbooks with comprehensive methodology transparency:

#### Main Output Files

**`KCOR_analysis.xlsx`** - Complete analysis with all enrollment periods combined
This file enables users to visualize results for any cohort combination and contains:

**`KCOR_summary.xlsx`** - Console-style summary by enrollment date
This file provides one sheet per enrollment period (e.g., 2021_24, 2022_06) formatted like the console output, with dose combination headers and final KCOR values for each age group.

#### Main Analysis Sheets
- **`dose_pairs`**: KCOR values for all dose comparisons with complete methodology transparency
- **Columns**: Sheet, ISOweekDied, Date, YearOfBirth, Dose_num, Dose_den, KCOR, CI_lower, CI_upper, 
  MR_num, MR_adj_num, CH_num, CH_actual_num, hazard_num, slope_num, scale_factor_num, MR_smooth_num, t_num,
  MR_den, MR_adj_den, CH_den, CH_actual_den, hazard_den, slope_den, scale_factor_den, MR_smooth_den, t_den,
  KCOR_o (optional death-based cumulative-deaths ratio, normalized at week 1)

#### Debug & Details Sheets
- **`by_dose`**: Individual dose curves with complete methodology transparency
- **Columns**: Date, YearOfBirth, Dose, ISOweek, Dead, Alive, MR, MR_adj, Cum_MR, Cum_MR_Actual, Hazard, 
  Slope, Scale_Factor, Cumu_Adj_Deaths, Cumu_Unadj_Deaths, Cumu_Person_Time, 
  Smoothed_Raw_MR, Smoothed_Adjusted_MR, Time_Index

- **`dose_pair_deaths`**: Per-pair weekly and cumulative death details supporting KCOR_o
- **Columns**: EnrollmentDate, ISOweekDied, Date, YearOfBirth, Dose_num, Dose_den,
  Dead_num, Dead_adj_num, cumD_num, Dead_den, Dead_adj_den, cumD_den, K_raw_o, KCOR_o

#### About Sheet
- **Metadata**: Version information, methodology overview, and analysis parameters
- **Documentation**: Complete explanation of the KCOR methodology and output columns

#### Visualization Capabilities

**`KCOR.xlsx`** - Complete analysis file:
- **Filter by Cohort**: Use Excel filters to examine specific dose combinations (e.g., 2 vs 0, 3 vs 0)
- **Filter by Age**: Focus on specific birth years or age groups
- **Time Series Analysis**: Plot KCOR values over time for any cohort combination
- **Confidence Intervals**: Visualize uncertainty bounds alongside point estimates
- **Methodology Validation**: Examine all intermediate calculations for transparency

**`KCOR_summary.xlsx`** - Console-style summary format:
- **One Sheet Per Enrollment**: Easy comparison across different enrollment periods (2021_24, 2022_06, etc.)
- **Console Format**: Structured like the console output with dose combination headers
- **Final Values**: Shows the latest KCOR values and confidence intervals for each age group
- **Easy Reading**: Clean format with dose combination headers and age group results
- **Cross-Period Analysis**: Compare final KCOR values across different enrollment cohorts

## ⚙️ Configuration

### Key Parameters

```python
# Core methodology
ANCHOR_WEEKS = 4                    # Baseline week for KCOR normalization (Note that week # 0 is the first week)
SLOPE_WINDOW_SIZE = 2               # Window size for slope calculation (±2 weeks)
MA_TOTAL_LENGTH = 8                 # Moving average length (8 weeks)
CENTERED = True                     # Use centered moving average
DYNAMIC_HVE_SKIP_WEEKS = 0           # Start accumulating hazards/statistics at this week index (0 = from enrollment)

# Analysis scope
YEAR_RANGE = (1920, 2000)          # Birth year range to process. Deaths outside the extremes are NOT combined.
# the following dates correspond to  3/29/21, 6/15/21, 2/7/22, 11/21/22
ENROLLMENT_DATES = ["2021_13", "2021_24", "2022_06", "2022_47"]  # ISO Year-week Enrollment dates (sheet names to process

# Dynamic slope anchors (quiet-period calendar picks)
QUIET_ANCHOR_ISO_WEEKS = ["2022-25", "2023-28", "2024-15"]
MIN_ANCHOR_GAP_WEEKS = 26             # min weeks after enrollment for first anchor
MIN_ANCHOR_SEPARATION_WEEKS = 39      # min weeks between first and second anchors

```

### Sheet-Specific Configuration

The script automatically determines dose pairs based on sheet names:

- **2021-13**: Doses 0, 1, 2 → Comparisons: (1,0), (2,0), (2,1)
- **2021_24**: Doses 0, 1, 2 → Comparisons: (1,0), (2,0), (2,1)
- **2022_06**: Doses 0, 1, 2, 3 → Comparisons: (1,0), (2,0), (2,1), (3,2), (3,0)
 - **2022_47**: Doses 0, 1, 2, 3, 4+ → Comparisons: (4,3), (4,2), (4,1), (4,0)

## 📊 Interpretation

### KCOR Values

- **KCOR = 1.0**: No difference in mortality risk between groups
- **KCOR > 1.0**: Higher mortality risk in numerator group (e.g., vaccinated)
- **KCOR < 1.0**: Lower mortality risk in numerator group
- **Confidence Intervals**: Provide statistical uncertainty around the point estimate

### Example Output

```
Dose combination: 2 vs 0 [2021_24]
--------------------------------------------------
            YoB | KCOR [95% CI]
--------------------------------------------------
  ASMR (pooled) | 1.2579 [1.232, 1.285]
           1940 | 1.2554 [1.194, 1.320]
           1955 | 1.5021 [1.375, 1.640]

Dose combination: 3 vs 2 [2022_06]
--------------------------------------------------
            YoB | KCOR [95% CI]
--------------------------------------------------
  ASMR (pooled) | 1.4941 [1.464, 1.525]
           1940 | 1.6489 [1.570, 1.731]
           1955 | 1.4619 [1.350, 1.583]
```

This shows that for dose 2 vs. dose 0 (2021_24 cohort):
- **ASMR**: 25.8% higher mortality risk (95% CI: 23.2% to 28.5%)
- **Age 1940**: 25.5% higher risk (95% CI: 19.4% to 32.0%)
- **Age 1955**: 50.2% higher risk (95% CI: 37.5% to 64.0%)

And for dose 3 vs. dose 2 (2022_06 cohort):
- **ASMR**: 49.4% higher mortality risk (95% CI: 46.4% to 52.5%)
- **Age 1940**: 64.9% higher risk (95% CI: 57.0% to 73.1%)
- **Age 1955**: 46.2% higher risk (95% CI: 35.0% to 58.3%)

## 🔧 Advanced Features

### Complete Methodology Transparency (v4.1)
- **Full Traceability**: Every step of the calculation is visible in output
- **Mathematical Relationships**: All intermediate values (slope, scale_factor, hazard) included
- **Validation Ready**: Users can verify every mathematical relationship
- **Debug Friendly**: Easy to spot-check individual values and calculations

### Discrete Hazard Function Transform (v4.1)
See [Hazard Function Methodology](documentation/hazard_function.md) for detailed derivation but basically the concept is you can cumulate deaths, but not hazard probabilities. Using the log transform eanbles you to cumulative mortality rate hazard via simple addition, the same way you'd cumulate deaths. Because mortality rate is slight more stable than deaths, using the discrete hazard function transform on mortality rates gives you a more accurate result.
- **Mathematical Enhancement**: More exact cumulative hazard (CH) calculation than simple summation of mortality rates (which is mathematically incorrect)
- **Hazard Function**: `hazard(t) = -ln(1 - MR_adj(t))` with proper clipping to avoid log(0)
- **Cumulative Process**: `CH(t) = sum(hazard(i))` for i=0 to t (cumulative hazard)
- **Numerical Stability**: Handles edge cases with proper bounds and clipping
- **Hazard Ratio**: KCOR computed as ratio of cumulative hazards, normalized to baseline


### Error Handling & User Experience
- **File Access Protection**: Automatic retry when Excel files are open
- **Clean Console Output**: Professional column headings and formatting
- **Version Documentation**: Complete change history in code
- **Cross-Platform**: Windows-compatible Makefile and scripts

### Moving Average Smoothing
- **8-week centered MA**: Reduces noise while preserving trend information
- **Configurable**: Window size and centering can be adjusted
- **Pre-slope**: Applied before slope calculation for stability

### Window-Based Slope Estimation
- **Robust Anchoring**: Uses multiple time points around each anchor
- **Geometric Mean**: Appropriate for multiplicative processes like mortality
- **Consistent Comparison**: Same anchor points across all dose groups

### Confidence Interval Calculation
- **Proper Propagation**: Accounts for uncertainty in both baseline and current estimates
- **Asymmetric Bounds**: Reflects the non-symmetric nature of ratio estimates
- **Binomial Variance**: Appropriate for count data

## 🚨 Troubleshooting

### Common Issues

1. **File Access Errors**: If Excel files are open, the script will prompt you to close them and retry
2. **Missing Data**: Ensure all required columns are present in input files
3. **Date Formats**: Verify dates are in proper datetime format
4. **Memory Issues**: Large datasets may require processing in smaller chunks
5. **Slope Calculation**: Check that anchor points fall within available data range
6. **Makefile Dependencies**: Ensure input files exist before running `make KCOR`

### Debug Mode

Enable detailed debugging by setting:
```python
DEBUG_VERBOSE = True
ENROLLMENT_DATES = ["sheet_name"]  # Limit to specific enrollment dates
YEAR_RANGE = (1940, 1945)          # Limit to specific age range
```

## 🤝 Contributing

We welcome contributions to improve the KCOR methodology and implementation. Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request with detailed description

## 📚 Citation

If you use KCOR in your research, please cite:

**KCOR v4.1 - Kirsch Cumulative Outcomes Ratio Analysis**  
[Your paper title]  
[Authors]  
[Journal/Conference]  
[Year]

That is, if I'm lucky enough to get this published. It's ground breaking, but people seem uninterested in methods that expose the truth about the COVID vaccines for some reason.

## 🆕 Version 4.3 Enhancements

### Major Improvements
- **Improved KCOR Scaling Logic**: Fixed baseline normalization adjustment to properly scale the scale factor
- **Corrected Scaling Application**: Scaling now applied to the scale factor, not individual KCOR values
- **Enhanced Parameter Management**: Single ENROLLMENT_DATES parameter replaces DEBUG_SHEET_ONLY
- **Streamlined Processing**: Removed 2021_13 enrollment date, focusing on 2021_24 and 2022_06 cohorts
- **Transparent Logging**: Original and adjusted scale factors logged when scaling is applied
- **Preserved Relationships**: All K_raw relationships maintained while correcting baseline bias
- **Updated Examples**: README examples updated with latest KCOR values from current analysis

### KCOR Scaling Fix (v4.3)
- **Before (v4.2)**: Scaling applied to individual KCOR values after computation
- **After (v4.3)**: Scaling applied to the scale factor during computation
- **Logic**: Check final KCOR value, if < threshold, adjust scale factor by 1/final_KCOR
- **Result**: Proper baseline correction while preserving all relative relationships
- **Transparency**: Original and adjusted scale factors logged for full methodology transparency

## 🆕 Version 4.2 Enhancements

### Major Improvements
- **Option 2+ Expected-Deaths Weighting**: Fixed ASMR pooling to properly reflect death burden
- **Corrected ASMR Values**: ASMR now reflects actual mortality impact, not population size
- **Dose-Dependent Pattern Discovery**: Revealed accelerating mortality pattern (1→2→3 doses)
- **Mathematical Correctness**: Elderly properly weighted, young under-weighted in ASMR
- **Robust Implementation**: Uses pooled quiet baseline window with smoothed mortality rates
- **Enhanced Documentation**: Complete explanation of Option 2+ methodology
- **KCOR Normalization Fine-Tuning**: Automatic baseline correction for unsafe vaccine effects

### ASMR Pooling Fix (Option 2+)
- **Before (v4.1)**: Weights = person-time only → over-weighted young people
- **After (v4.2)**: Weights = hazard × person-time → properly weighted by death burden
- **Formula**: `w_a ∝ h_a × PT_a(W)` where h_a = smoothed mean MR in quiet window
- **Result**: ASMR values now reflect actual mortality impact rather than population size

### KCOR Normalization Fine-Tuning
- **Automatic Detection**: Checks KCOR values on specified final date (default: April 1, 2024)
- **Baseline Correction**: Scales all KCOR values when KCOR_final < FINAL_KCOR_MIN (default: 0 disables scaling)
- **Unsafe Vaccine Fix**: Corrects for artificially high baseline mortality rates during normalization
- **Transparent Process**: Scaling factor is logged for full methodology transparency
- **Conservative Approach**: Only applies when clear evidence of baseline bias exists

### New Results Pattern
- **Dose 1 vs 0 (2021_24)**: KCOR = 1.05 (5.2% increased mortality risk)
- **Dose 2 vs 0 (2021_24)**: KCOR = 1.26 (25.8% increased mortality risk)
- **Dose 1 vs 0 (2022_06)**: KCOR = 1.12 (11.9% increased mortality risk)
- **Dose 2 vs 0 (2022_06)**: KCOR = 1.05 (5.0% increased mortality risk)
- **Dose 3 vs 0 (2022_06)**: KCOR = 1.55 (54.9% increased mortality risk)
- **Pattern**: Dose-dependent accelerating mortality with cohort-specific effects

## 🆕 Version 4.1 Enhancements

### Major Improvements
- **Discrete Hazard Function Transform**: Enhanced mathematical exactness in CH calculation using hazard functions
- **Hazard Ratio Methodology**: KCOR computed as ratio of cumulative hazards with proper normalization
- **Complete Methodology Transparency**: All intermediate values included in output
- **Error Handling**: Automatic retry when Excel files are open
- **Clean Console Output**: Professional formatting with column headings
- **Cross-Platform Build**: Windows-compatible Makefile and scripts
- **Version Documentation**: Complete change history in code

### New Output Columns
- **Hazard Values**: `hazard_num/den` - Discrete hazard function results
- **Slope Values**: `slope_num/den` - Slope used for each cohort
- **Scale Factors**: `scale_factor_num/den` - `exp(-slope × (t - t0))` values
- **Time Indices**: `t_num/den` - Time index (weeks from enrollment)
- **Smoothed MR**: `MR_smooth_num/den` - Smoothed MR values used for slope calculation

### Mathematical Enhancements
- **Four-Step Process**: MR_adj → hazard → cumsum(hazard) → hazard ratio for KCOR
- **Hazard Function Transform**: `hazard(t) = -ln(1 - MR_adj(t))` with proper clipping
- **Cumulative Hazard**: `CH(t) = sum(hazard(i))` for mathematical exactness
- **Hazard Ratio**: `KCOR(t) = (CH_v(t)/CH_u(t)) / (CH_v(t0)/CH_u(t0))`
- **Numerical Stability**: Proper clipping to avoid log(0) and overflow
- **Validation Ready**: All mathematical relationships visible in output

## 📊 Results Using Czech Data

### Summary of Age-Standardized Mortality Ratio (ASMR) Results

The KCOR analysis of Czech vaccination and mortality data reveals significant findings across all dose levels compared to unvaccinated individuals. **Version 4.2 uses corrected expected-deaths weighting** that properly reflects actual mortality burden:

| **DOSE** | **KCOR** | **95% CI** | **Risk Increase** | **Enrollment** |
|----------|----------|------------|-------------------|----------------|
| **1** | 1.0453 | [1.016, 1.075] | +4.5% | 2021_24 |
| **2** | 1.2091 | [1.184, 1.235] | +20.9% | 2021_24 |
| **3** | 1.6354 | [1.603, 1.668] | +63.5% | 2022_06 |

### Key Findings

- **Dose 1 shows small harm** - 4.5% increased mortality (2021_24; significant). The 2022_06 cohort estimate (1.0156) is not statistically significant (CI includes 1.0) because most of the harm happens within 6 months of the shot which was long ago in that cohort. Looking at Dose 1 much closer to vaccination gives a very large signal (over 40%) See the [summary log](data/czech/KCOR_summary.log) for details.
- **Dose 2 shows significant harm** with 20.9% increased mortality (2021_24) and 6.5% (2022_06)
- **Dose 3 shows severe harm** with 53.5% increased mortality vs dose 2 and 63.5% vs dose 0 (2022_06), both highly significant
- **Dose-dependent accelerating mortality** - risk increases with additional doses

### ⚠️ Important Note on Dose 1 Harm Estimates

**The Dose 1 harm estimates are likely CONSERVATIVE (underestimated)** due to the enrollment period timing:

- **Enrollment starts months after first doses**: The analysis begins enrollment periods (2021_24, 2022_06) many months after the first COVID-19 vaccine doses were administered to elderly populations
- **Early harm missed**: Any immediate or early-term mortality effects from Dose 1 that occurred before the enrollment periods are not captured in this analysis
- **Baseline period protection**: The enrollment period was deliberately chosen to start after major COVID-19 waves to ensure accurate baseline mortality rate calculations
- **Conservative interpretation**: This means the true harm from Dose 1 is likely higher than the neutral effect (KCOR ≈ 1.0) shown in these results

This conservative bias is particularly important for understanding the true cumulative impact of COVID-19 vaccination on mortality risk.

### 🎯 Dose-Dependent Accelerating Mortality Pattern

The results reveal a **dose-dependent accelerating mortality pattern**:

| **Dose** | **KCOR (2021_24)** | **Risk Increase** | **Pattern** |
|----------|-------------------|-------------------|-------------|
| **1 vs 0** | 1.0453 | +4.5% | **Small harm (significant)** |
| **2 vs 0** | 1.2091 | +20.9% | **Significant harm** |
| **2 vs 1** | 1.1568 | +15.7% | **Significant harm** |

| **Dose** | **KCOR (2022_06)** | **Risk Increase** | **Pattern** |
|----------|-------------------|-------------------|-------------|
| **1 vs 0** | 1.0156 | +1.6% | **Small (not significant)** |
| **2 vs 0** | 1.0654 | +6.5% | **Moderate harm** |
| **2 vs 1** | 1.0490 | +4.9% | **Small–moderate harm** |
| **3 vs 2** | 1.5350 | +53.5% | **Severe harm** |
| **3 vs 0** | 1.6354 | +63.5% | **Severe harm** |

**Key Insights**:

1. **First dose shows significant harm** - 5.2% (2021_24) to 11.9% (2022_06) increased mortality risk
2. **Second dose shows varying harm** - 5.0% to 25.8% increased mortality risk depending on cohort
3. **Third dose shows severe harm** - 49.4% to 54.9% increased mortality risk
4. **Cohort-dependent effects** - 2021_24 cohort shows stronger dose 2 effects than 2022_06
5. **Accelerating pattern** - each additional dose causes exponentially more harm
6. **No protective effect** - all doses show statistically significant harm

**Biological Interpretation**: This pattern suggests that:
- **Dose 1**: Causes immediate and significant harm with no protective benefit
- **Dose 2**: Causes additional harm, with effects varying by timing of administration
- **Dose 3**: Causes severe cumulative damage with accelerating mortality risk

This **dose-dependent accelerating mortality pattern** provides compelling evidence of cumulative vaccine toxicity rather than simple linear dose-response effects.

### Complete Results

For detailed results including age-specific analyses and all dose combinations, see the comprehensive analysis files:

- **📈 Complete Analysis**: [`data/Czech/KCOR_summary.xlsx`](data/Czech/KCOR_summary.xlsx) - Age-standardized and age-specific results by enrollment cohort
- **📊 Full Dataset**: [`data/Czech/KCORv4.xlsx`](data/Czech/KCORv4.xlsx) - Complete analysis with all intermediate calculations
- **📋 Console Output**: [`data/Czech/KCOR_summary.log`](data/Czech/KCOR_summary.log) - Detailed console output from analysis (dual output: console + file)
 - **🧮 Interactive Plotting Workbook**: [`analysis/Czech/KCOR_analysis.xlsx`](analysis/Czech/KCOR_analysis.xlsx) - Excel workbook for plotting KCOR(t) curves for any cohort/dose mix

### Interpretation

These results demonstrate a **dose-dependent accelerating mortality pattern** that provides compelling evidence of cumulative vaccine toxicity:

1. **First dose (1 vs 0)**: Significant harm with 5.2% to 11.9% increased mortality risk
2. **Second dose (2 vs 0)**: Additional harm with 5.0% to 25.8% increased mortality risk  
3. **Third dose (3 vs 0)**: Severe harm with 54.9% increased mortality risk

This pattern shows that **all doses cause statistically significant harm** with **no protective benefit** at any dose level. The analysis reveals **cumulative damage with accelerating mortality risk** for each additional dose. This finding is consistent across different enrollment cohorts and age groups, providing robust evidence of dose-dependent vaccine toxicity in the Czech population.

## 🔬 Validation

The KCOR methodology has been independently validated using multiple approaches to ensure the robustness and accuracy of the results:

### Independent Validation Methods

The [`validation/`](validation/) directory contains four independent validation approaches to analyzing the Czech data: GLM, DS-CMRR, Kaplan-Meier, looking at cumulative deaths of naturally matched cohorts.

Here are the KCOR results for direct comparison with other methods (such as GLM and DS-CMRR) that produce similar style curves

   ![KCOR Results](analysis/Czech/KCOR_ASMR_dose2.png)
   
1. **GLM Validation**: A **completely different methodology** using Generalized Linear Models with **variable cohorts** rather than fixed cohorts. This approach creates output that looks nearly identical to KCOR results, providing strong independent validation. Defaults use 4‑week ticks with vertical grid lines.

   ![GLM Validation Results](validation/GLM/out/GLM_plot_Czech_data.png)
   
   *GLM validation results showing remarkable consistency with KCOR methodology*
   
2. **DS-CMRR Validation**: Discrete Survival Cumulative Mortality Rate Ratio method for independent verification

This method can be used with either fixed or variable cohorts. I chose to run it against fixed cohorts because that is the more meaningful outcome, but others are free to run it against variable cohorts. 

Question answered: "Between two groups defined at baseline, who accumulated more death risk over the window?" 

Readout: DS-CMRR is the ratio of cumulative hazards between two pre-specified groups—closest to a trial-like contrast.

   ![DS-CMRR dose 2 vs 0 (ASMR case)](validation/DS-CMRR/out/DS-CMRR_ASMR.png)

   *DS-CMRR output KCOR(t) for Czech data, dose 2 vs unvaccinated (single-sheet 2021_24)*

3. **Kaplan–Meier Validation**: Traditional survival analysis on naturally matched cohorts (equalized initial population at enrollment) using `validation/kaplan_meier/`.

   ![Kaplan–Meier survival (YoB 1940–1995, 2021_24)](validation/kaplan_meier/out/KM_2021_24_1940_1995.png)

   *Observation: With naturally matched cohorts, the curves diverge with the unvaccinated cohort exhibiting lower mortality over time.*

4. **Naturally Matched Cohorts**: I also validated using naturally matched cohorts where the cohorts are defined such that they had very similar deaths/week during the baseline and next COVID wave to demonstrate that matched cohort will diverge when a booster shot is given to a subset of the vaccinated group (which, if the vaccine was safe, should cause deaths to decrease, not increase). 

So this plot finds net harm, but possibly a modest mortality benefit. KCOR, GLM, DS-CMRR, and KM (properly interpreted) reflects the same thing as this raw data.

![Naturally matched cohorts](reference_results/analysis/naturally_matched_cohorts.png)

This plot shows that with naturally matched cohorts, the curves remain aligned but when people got the boosters, it prevented their mortality from returning to baseline levels (slope of the cumulative death curve). This is why there was a negative net harm.

5. **Aarstad Correlation Analysis**: Independent [correlation analysis of CDC  deaths data by county](https://jarle.substack.com/p/the-covid-19-vaccine-caused-almost), providing external validation of KCOR findings.

   ![Aarstad Correlation Analysis](validation/aarstad/aarstad.png)

   *Aarstad correlation analysis showing consistent patterns with KCOR methodology*

### Negative-Control and Sensitivity Tests

In addition to the validation suite, the repository includes:

- **Negative-Control Tests** (`test/negative_control/`): Builds synthetic no-signal cohorts to ensure no false positives.
  - Run: `make test`
  - Outputs: `test/negative_control/out/` (e.g., `KCOR_processed_neg_control.xlsx`, `KCOR_summary.xlsx`)
  - References: `reference_results/negative_control_tests/`

  These two graphs below show even with 10 and 20 year age differences between the cohorts, KCOR is able to accurately normalize the mortality and find neglible differences. Only when there is a real signal will there be a difference. Do you know of any other epidemiology tool that will find no signal in these groups which have dramatically different composition? All the methods I'm aware of require you to do 1:1 matching.

  ![Negative control (10-year age difference)](test/negative_control/analysis/neg_control_10yr_age_diff.png)

  ![Negative control (20-year age difference)](test/negative_control/analysis/neg_control_20yr_age_diff.png)

- **Sensitivity Tests** (`test/sensitivity/`): Sweeps key parameters (cohorts, anchors, ages) to check stability.
  - Run: `make test` or `make sensitivity`
  - Outputs: `test/sensitivity/out/` (e.g., `KCOR_SA.xlsx`, `KCOR_summary_SA.log`)
  - References: `reference_results/sensitivity/`

### Validation Objectives

- **Methodological Robustness**: Verify KCOR results using alternative statistical approaches
- **Cross-Validation**: Ensure consistency across different analytical methods
- **Sensitivity Analysis**: Test the stability of results under different assumptions
- **Reproducibility**: Independent verification of KCOR computations

### Validation Results

The validation studies confirm:
- **Consistent Findings**: KCOR results are robust across different analytical approaches
- **Methodological Soundness**: The discrete hazard function approach is mathematically valid
- **Statistical Reliability**: Confidence intervals and uncertainty quantification are appropriate
- **Reproducibility**: Results can be independently replicated using different implementations

For detailed validation results and methodology comparisons, see the [`validation/`](validation/) directory.

## Peer review

As you can imagine, it's like pulling teeth to get any credible epidemiologist to look at this. Harvey Risch, with an h-index of 119, is arguably one of the top epidemiologists in the world. He reviewed an earlier version of KCOR and I made a transcript of the Zoom call. The bottom line is he didn't find any flaws in the methodology but noted that it would be hard to convince the epidemiology community because it is an engineering approach to making the harm/benefit assessment (he used the term "heuristic").

- Yale Professor Harvey Risch review (PDF): [`peer_review/KCOR_Risch_review.pdf`](peer_review/KCOR_Risch_review.pdf)
- Grok assessment: [Grok validated](https://grok.com/share/c2hhcmQtMg%3D%3D_6bda87e0-f4b7-49b7-a6b7-3cf48fc453b2) the methodology, the documentation, and the implementation. It said the math was sound, but it didn't think people actually died per Gompertz mortality. It didn't have a real world counterexample.

"The KCOR method is a transparent and reproducible way to assess vaccine safety using only the most essential data. By relying solely on date of birth, vaccination, and death, it avoids the covariate manipulation and opaque modeling that plague conventional epidemiology, while slope normalization directly accounts for baseline mortality differences between groups. Applied to the Czech registry data, KCOR revealed a consistent net harm across all age groups. Given the strength and clarity of this signal, vaccine promoters will have no choice but to fall back on ideology rather than evidence in their response."

 — Nicolas Hulscher, MPH  
Epidemiologist and Administrator  
McCullough Foundation

### Grok review

Grok claimed KCOR is mathematically sound, but in a later discussion claimed that KCOR is fundamentally flawed because relies on assumptions about the data that are clearly wrong, such as that you can normalize slope mortality for frail cohorts. 

At first, Grok tried to argue that even for ideal Gompertz cohorts, KCOR would fail. So I had it compute the mortality rate change for age 80 with frailty 3 (which is more extreme than real life) and it realized there was only a 1.6% differential after 2 years. I pointed out KCOR is averaged out over that period so the error would be half that amount. I also pointed out that the slope computation is done over a year period in the center so the net error would be even smaller since it would error one way early and the other way later which further reduces the error.

 Next Grok asked about the flu and other mortality disruptions. I said that virtually all such disruptions are proportional to baseline mortality and would be normalized out. The ONLY calendar disruption that is non-linear and large enough to impact the KCOR curves was COVID and it gave the vaccine the benefit of the doubt because KCOR assumes proportional hazards which are violated by the COVID virus; COVID kills the frail disproportionally more than would be predicted by the mortality ratio as you can [instantly demostrate for yourself using the official Czech data](https://csu.gov.cz/produkty/obypz_cr) by looking at different age groups in 2020 vs. 2019. You'll see no mortality increase under 45 at the end of 2020, but a >2X mortality increase for those over 85. The bottom line; KCOR is conservative since it assumes proportional hazards so it gives the vaccine the credit for COVID that it doesn't deserve. 

Then the final argument was there is no possible way a 80-90 year old unvaccinated cohort could be slope normalized with an exponential because of the massive mix of frailty and depletion, even though we just went over that case. So I showed it that [the actual Czech data](analysis/Czech/KCOR_80_90_uvax_slope.png) could be fit with a straight line and that decimated its set of arguments. So much for the debunk attempts. 

Grok could have said something like "before using KCOR, validate that the cohorts under study can be accurately adjusted with a simple exponential." If people did that, they'd be pleasantly surprised that there is no flaw. But instead, Grok simply dismissed KCOR as flawed because it assumes KCOR must be flawed because it conflicts with the "safe and effective" narrative. 

In the real world, the underlying mortality trend is nearly linear over a 2 year period for most all cohorts of interest and most all hazards are proportional to mortality (COVID being the rare exception where KCOR will give a conservative harm estimate). If needed, KCOR can easily be adapted to highly frail cohorts using piecewise slope normalization if it is necessary to get extreme accuracy for these cases. Grok seemed more interested in trashing the method than pointing out how it's objections can be easily overcome.

Here is [the full discussion with Grok showing it lacked actual evidence that falsifies KCOR](https://grok.com/share/c2hhcmQtMg%3D%3D_c35e85ae-f70c-409d-b91f-96193d23e381) and finally said that it would recommend KCOR ([see Grok Review](#grok-review))

## 📄 License

This project is licensed under the MIT License — see https://opensource.org/licenses/MIT for details.

## 📞 Contact

For questions about the methodology or implementation, please open an issue on GitHub or contact the development team.

---

**Note**: This software is designed for research purposes. Users should carefully validate results and consider the specific context of their data and research questions.
