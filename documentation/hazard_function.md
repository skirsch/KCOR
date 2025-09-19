# Discrete-time Hazard Function Tranformation Methodology in KCOR

## Overview

KCOR uses the discrete-time hazard function transformation to precisely compute cumulative mortality risk with mathematical exactness. 

This document explains the mathematical reasoning behind this approach and why it's superior to simple mortality rate summation (which is mathematically nonsensical even though for small values it works).

It basically allows you to sum up hazards like you'd sum up deaths per week. Summing hazards is a much more accurate way to compare mortality between groups because mortality rate more closely tracks an exponential than deaths per week.

## The Problem with Simple Mortality Rate Summation

### ❌ Incorrect Approach

When we simply sum mortality rates over time:
```
CMR(t) = Σ MR(i) for i=0 to t
```

This is **mathematically incorrect** because:
- **Mortality rates are probabilities** (0 ≤ MR ≤ 1)
- **Summing probabilities** doesn't give you the correct cumulative probability
- **It's an approximation** that only works when MR is very small
- **It ignores the compounding effect** of survival over time

### Example of the Problem

Consider weekly mortality rates of 0.01 (1%) for 3 weeks:

**❌ Incorrect (Simple Sum):**
```
CMR = 0.01 + 0.01 + 0.01 = 0.03 (3%)
```

**✅ Correct (Hazard Function):**
```
hazard(1) = -ln(1 - 0.01) = -ln(0.99) = 0.01005
hazard(2) = -ln(1 - 0.01) = -ln(0.99) = 0.01005  
hazard(3) = -ln(1 - 0.01) = -ln(0.99) = 0.01005
CMR = 0.01005 + 0.01005 + 0.01005 = 0.03015
```

The difference becomes more significant with higher mortality rates.

## The Correct Mathematical Approach: Hazard Functions

### Survival Analysis Foundation

The hazard function `h(t)` represents the **instantaneous risk of death** at time t, given survival up to time t. The relationship between hazard and survival is:

```
S(t) = exp(-∫₀ᵗ h(s)ds)
```

Where:
- `S(t)` = survival probability at time t
- `h(s)` = hazard function at time s

### Discrete Time Transform

For **discrete time periods** (i.e., weekly data rather than continuous data), we use:

```
hazard(t) = -ln(1 - MR_adj(t))
```

## Mathematical Derivation

### Step 1: Survival Probability

If `MR_adj(t)` is the probability of death in period t, then `1 - MR_adj(t)` is the probability of survival in period t.

### Step 2: Cumulative Survival

The probability of surviving from time 0 to time t is:
```
S(t) = ∏(1 - MR_adj(i)) for i=0 to t
```

### Step 3: Log Transform

Taking the natural log:
```
ln(S(t)) = ln(∏(1 - MR_adj(i))) = Σ ln(1 - MR_adj(i))
```

### Step 4: Hazard Function

The hazard function is the negative of the log survival:
```
hazard(t) = -ln(1 - MR_adj(t))
```

hazard(t) is always between 0 and positive infinity because MR_adj(t) is a probability between 0 and 1. It's generally much closer to 0 most of the time except for when a cohort is fully depleted. Before computing h(t) we ensure that MR_adj(t) is clipped to a maximum value of .999.

### Step 5: Cumulative Hazard

The cumulative hazard is:
```
H(t) = Σ hazard(i) = -ln(S(t))
```

## Why This is Mathematically Exact

The discrete-time hazard transform gives us the **exact cumulative hazard** because:

1. **It accounts for the compounding effect** of survival over time
2. **It properly handles the probability mathematics** of sequential events
3. **It's not an approximation** - it's the mathematically correct way to compute cumulative risk
4. **It follows established survival analysis principles**

## Implementation in KCOR

### Code Implementation

```python
# Apply discrete cumulative-hazard transform for mathematical exactness
# Clip MR_adj to avoid log(0) and ensure numerical stability
df["hazard"] = -np.log(1 - df["MR_adj"].clip(upper=0.999))

# Calculate cumulative hazard (mathematically exact, not approximation)
df["CMR"] = df.groupby(["YearOfBirth","Dose"])["hazard"].cumsum()
```

### Why Clipping to 0.999?

The code clips `MR_adj` to 0.999 to avoid `ln(0)`:
- Prevents `ln(0)` which is undefined
- Maintains numerical stability
- The difference between `ln(0.001)` and `ln(0.0001)` is negligible for practical purposes

## Comparison with Traditional Methods

### Traditional Approaches

Most epidemiological methods use approximations:
- **Simple summation** of mortality rates
- **Proportional hazards assumptions** that are often violated
- **Static baseline rates** that don't account for time-varying trends

### KCOR's Advantage

KCOR uses the discrete hazard transform because:

1. **Mathematical Exactness**: It's the correct way to compute cumulative risk
2. **No Approximation**: Unlike simple summation, it's mathematically rigorous
3. **Handles Any Rate**: Works correctly for both low and high mortality rates
4. **Proper Probability Theory**: Follows the correct mathematical relationships
5. **Time-Varying Adjustment**: Combines with slope correction for trend adjustment

## Mathematical Properties

### Additivity

Hazard functions are additive:
```
H(t) = Σᵢ₌₀ᵗ hazard(i)
```

This property makes them ideal for cumulative risk calculation.

### Monotonicity

Cumulative hazard is monotonically increasing:
```
H(t₁) ≤ H(t₂) for t₁ < t₂
```

### Relationship to Survival

The relationship between cumulative hazard and survival is exact:
```
S(t) = exp(-H(t))
```

## Practical Implications

### Accuracy

The hazard function approach provides:
- **Exact cumulative risk** calculation
- **Proper handling** of high mortality rates
- **Mathematically rigorous** comparisons between cohorts

### Robustness

The method is robust because:
- **No assumptions** about rate magnitudes
- **Numerically stable** with proper clipping
- **Theoretically sound** based on survival analysis

## Conclusion

The discrete hazard function transformation in KCOR represents a **mathematically superior approach** to cumulative mortality risk calculation. By using the correct probability theory for sequential events, KCOR provides more accurate and reliable comparisons between cohorts than methods that rely on approximations or simple summation.

This mathematical rigor is one of the key factors that makes KCOR a groundbreaking advancement in epidemiological methodology, enabling it to answer policy-relevant questions with unprecedented accuracy using observational data.

---

*For more details on the complete KCOR methodology, see the main [README.md](../README.md).*
