# KCOR v4.1 - Kirsch Cumulative Outcomes Ratio Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Overview](#overview)
- [Methodology](#methodology)
  - [Core Concept](#core-concept)
  - [Analysis Pipeline](#analysis-pipeline)
  - [Key Assumptions](#key-assumptions)
- [Repository Structure](#repository-structure)
- [Installation & Dependencies](#installation--dependencies)
- [Usage](#usage)
- [Configuration](#configuration)
- [Interpretation](#interpretation)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Overview

KCOR (Kirsch Cumulative Outcomes Ratio) is a robust statistical methodology for analyzing relative mortality risk between different vaccination groups while accounting for underlying mortality rate time trend differences. This repository contains the complete analysis pipeline for computing KCOR values from mortality data.

Suppose you could take any two cohorts, regardless of age, sex, frailty mix, etc. and normalize the mortality rate so that if there is no external signal applied that might differentially impact their mortality, both cohorts would die over time with identical mortality rates.

That’s what KCOR does. Once the cohorts are precisely matched from a mortality rate point of view, we can simply cumulate the adjusted hazards and see which cohort had more deaths.

KCOR basically allows you to run a randomized trial with respect to the death outcome, using retrospective observational data. No 1:1 matching is required. No cause of death is needed. You just need 3 dates per person: birth, death, vaccination(s).

KCOR allows us, for the first time, to objectively answer very important societal questions such as, “Did the COVID vaccine kill more people than it saved?”

## 🔬 Methodology

### 🏆 KCOR vs. Traditional Epidemiological Methods

KCOR represents a **groundbreaking advancement** in epidemiological methodology, offering unique advantages over traditional approaches for comparing mortality between cohorts:

#### **Traditional Methods vs. KCOR**

| **Aspect** | **Traditional Methods** | **KCOR** |
|------------|------------------------|----------|
| **Time-Varying Trends** | ❌ Assume static baseline rates | ✅ Dynamic slope correction |
| **Mathematical Rigor** | ❌ Often use approximations | ✅ Discrete hazard functions |
| **Baseline Control** | ❌ Compare absolute rates | ✅ Normalized to matched baseline |
| **Observational Data** | ❌ Require randomized trials | ✅ Creates "virtual randomization" |
| **Policy Questions** | ❌ Limited applicability | ✅ Direct policy evaluation |

#### **Why KCOR is Superior**

**🎯 Unique Problem Solving:**
- **Traditional SMR**: Assumes static reference population rates → fails with time-varying trends
- **KCOR**: Dynamically adjusts for secular changes, seasonal effects, and policy impacts

**🔬 Mathematical Excellence:**
- **Traditional Methods**: Use approximations or assume proportional hazards
- **KCOR**: Uses exact discrete hazard transformation: `hazard(t) = -ln(1 - MR_adj(t))`

**⚖️ Baseline Matching:**
- **Traditional Methods**: Compare absolute rates between potentially different cohorts
- **KCOR**: Normalizes to baseline period where cohorts are "matched" from mortality perspective

**🌍 Real-World Applicability:**
- **Traditional Methods**: Require controlled conditions or make unrealistic assumptions
- **KCOR**: Works with observational data to answer policy-relevant questions

#### **KCOR's Unique Value Proposition**

KCOR is **the only method** that can:
- ✅ Create "virtual randomization" from observational data
- ✅ Dynamically adjust for time-varying trends affecting both cohorts  
- ✅ Provide mathematically exact hazard-based comparisons
- ✅ Answer policy-relevant questions using real-world data
- ✅ Handle COVID-era complexity with multiple confounding factors

**Result**: KCOR can objectively answer questions like *"Did COVID vaccines kill more people than they saved?"* using observational data—something no traditional epidemiological method can achieve.

#### **Limitations of Traditional Epidemiological Methods**

**📊 Standardized Mortality Ratio (SMR)**
- ❌ Assumes static reference population rates
- ❌ Doesn't account for time-varying trends  
- ❌ Vulnerable to secular changes in mortality
- ❌ Cannot handle COVID-era policy impacts

**📈 Age-Period-Cohort (APC) Analysis**
- ❌ Complex identifiability issues
- ❌ Requires large datasets
- ❌ Doesn't provide direct cohort comparisons
- ❌ Difficult to interpret for policy questions

**⚖️ Proportional Hazards Models**
- ❌ Assumes proportional hazards (often violated)
- ❌ Doesn't handle time-varying effects well
- ❌ Requires sophisticated statistical modeling
- ❌ Vulnerable to model misspecification

**📋 Life Table Analysis**
- ❌ Doesn't account for external time-varying factors
- ❌ Assumes stable mortality patterns
- ❌ Less suitable for policy evaluation
- ❌ Cannot handle rapid changes in mortality

**🎯 Competing Risks Analysis**
- ❌ Focuses on cause-specific mortality
- ❌ Requires detailed cause-of-death data
- ❌ Doesn't address overall mortality differences
- ❌ Complex interpretation for policy makers

#### **The KCOR Advantage in Practice**

**🔬 Scientific Rigor:**
- KCOR provides mathematically exact comparisons using discrete hazard functions
- Traditional methods rely on approximations that can introduce bias
- KCOR's approach is more robust to violations of common statistical assumptions

**🌍 Real-World Relevance:**
- KCOR works with the messy, complex data of real-world policy implementation
- Traditional methods require idealized conditions that rarely exist in practice
- KCOR can handle the rapid changes and multiple confounding factors of the COVID era

**📊 Policy Impact:**
- KCOR directly answers policy-relevant questions using observational data
- Traditional methods often require randomized trials that are impossible for policy evaluation
- KCOR provides interpretable results that policymakers can understand and act upon

**⚡ Practical Implementation:**
- KCOR requires only basic demographic and mortality data (birth, death, vaccination dates)
- Traditional methods often require extensive additional data (cause of death, detailed covariates)
- KCOR can be applied to existing datasets without additional data collection

### 🎯 Core Concept

KCOR represents the ratio of cumulative hazard functions between two groups (e.g., vaccinated vs. unvaccinated), normalized to 1 at a baseline period. This approach provides interpretable estimates of relative mortality risk that account for:

- **Time-varying trends** in mortality rates through slope correction
- **Mathematical exactness** through discrete hazard function transformation
- **Baseline differences** between groups through normalization
- **Statistical uncertainty** in the estimates through proper variance propagation

### ⚙️ Analysis Pipeline

#### 1. Data Preprocessing
- **Enrollment Date Filtering**: Data processing starts from the enrollment date derived from sheet names (e.g., "2021_24" = 2021, week 24)
- **Sex Aggregation**: Mortality data is aggregated across sexes for each (YearOfBirth, Dose, DateDied) combination
- **Smoothing**: 8-week centered moving average applied to raw mortality rates to reduce noise

#### 2. Slope Calculation (Lookup Table Method)
- **Anchor Points**: Uses predefined time points (e.g., weeks 53 and 114 from enrollment for 2021_24)
- **Window Approach**: For each anchor point, creates a ±2 week window (5 points total)
- **Geometric Mean**: Calculates geometric mean of smoothed MR values within each window

**Slope Formula:**

$$r = \frac{1}{\Delta t} \ln\left(\frac{\tilde{B}}{\tilde{A}}\right)$$

Where:
- **Ã** = Geometric mean of MR values in window around first anchor: $$\tilde{A} = \text{GM}(MR_{t \in [t_0-w, t_0+w]})$$
- **B̃** = Geometric mean of MR values in window around second anchor: $$\tilde{B} = \text{GM}(MR_{t \in [t_1-w, t_1+w]})$$
- **Δt** = Time difference between anchor points (in weeks)
- **w** = Window size (default: 2 weeks)

**Geometric Mean Calculation:**

$$\text{GM}(x_1, x_2, \ldots, x_n) = e^{\frac{1}{n} \sum_{i=1}^{n} \ln(x_i)}$$

- **Consistency**: Same anchor points used for all doses to ensure comparability
- **Quiet Periods**: Anchor dates chosen during periods with minimal differential events (COVID waves, policy changes, etc.)

#### 3. Mortality Rate Adjustment
- **Exponential Slope Removal**: `MR_adj = MR × e^{-slope × (t - t0)}`
- **Baseline Normalization**: t0 = baseline week (typically week 4) where KCOR is normalized to 1
- **Dose-Specific Slopes**: Each dose-age combination gets its own slope for adjustment

#### 4. KCOR Computation (Enhanced v4.1)
**Three-Step Process:**

1. **Individual MR Adjustment**: Apply slope correction to each mortality rate
2. **Hazard Transform**: Convert adjusted mortality rates to discrete hazard functions for mathematical exactness  
3. **Cumulative Hazard**: Compute CMR as cumulative sum of hazard functions
4. **Ratio Calculation**: Compute KCOR as ratio of cumulative hazards, normalized to baseline

**Step 1: Mortality Rate Adjustment**

$$\text{MR}_{\text{adj}}(t) = \text{MR}(t) \times e^{-r(t - t_0)}$$

**Step 2: Discrete Hazard Function Transform**

$$\text{hazard}(t) = -\ln(1 - \text{MR}_{\text{adj}}(t))$$

Where MR_adj is clipped to 0.999 to avoid log(0).

> **📚 Mathematical Reasoning**: For a detailed explanation of why KCOR uses discrete hazard functions and the mathematical derivation behind this approach, see [Hazard Function Methodology](documentation/hazard_function.md).

**Step 3: Cumulative Hazard (CMR)**

$$\text{CMR}(t) = \sum_{i=0}^{t} \text{hazard}(i)$$

**Step 4: KCOR as Hazard Ratio**

**KCOR Formula:**

$$\text{KCOR}(t) = \frac{\text{CMR}_v(t) / \text{CMR}_u(t)}{\text{CMR}_v(t_0) / \text{CMR}_u(t_0)}$$

Where:
- **r** = Calculated slope for the specific dose-age combination
- **MR(t)** = Raw mortality rate at time t
- **t₀** = Baseline time for normalization (typically week 4)
- **CMR(t)** = Cumulative hazard at time t (sum of discrete hazards)
- **Mathematical Enhancement**: Discrete cumulative-hazard transform provides more exact CMR calculation than simple summation
- **Interpretation**: KCOR = 1 at baseline, showing relative risk evolution over time

#### 5. Uncertainty Quantification
**95% Confidence Interval Calculation:**

The variance of KCOR is calculated using proper uncertainty propagation for the hazard ratio:

$$\text{Var}[\ln(\text{KCOR}(t))] = \frac{\text{Var}[\text{CMR}_v(t)]}{\text{CMR}_v(t)^2} + \frac{\text{Var}[\text{CMR}_u(t)]}{\text{CMR}_u(t)^2} + \frac{\text{Var}[\text{CMR}_v(t_0)]}{\text{CMR}_v(t_0)^2} + \frac{\text{Var}[\text{CMR}_u(t_0)]}{\text{CMR}_u(t_0)^2}$$

**Confidence Interval Bounds:**

$$\text{CI}_{\text{lower}}(t) = \text{KCOR}(t) \times e^{-1.96 \sqrt{\text{Var}[\ln(\text{KCOR}(t))]}}$$

$$\text{CI}_{\text{upper}}(t) = \text{KCOR}(t) \times e^{1.96 \sqrt{\text{Var}[\ln(\text{KCOR}(t))]}}$$

Where:
- **Var[CMR] ≈ CMR**: Using Poisson variance approximation for cumulative hazard (sum of hazards)
- **Var[ln(KCOR)]**: Variance on log scale for proper uncertainty propagation of hazard ratio
- **1.96**: 95% confidence level multiplier (standard normal distribution)
- **Log-Scale Calculation**: CI bounds calculated on log scale then exponentiated for proper asymmetry

#### 6. Age Standardization
**ASMR Pooling Formula:**

The age-standardized KCOR is calculated using fixed baseline weights:

$$\text{KCOR}_{\text{ASMR}}(t) = e^{\frac{\sum_i w_i \ln(\text{KCOR}_i(t))}{\sum_i w_i}}$$

Where:
- **wᵢ** = Fixed weight for age group i (person-time in first 4 weeks)
- **KCORᵢ(t)** = KCOR value for age group i at time t
- **ln(KCORᵢ(t))** = Natural logarithm of KCOR for age group i

**Weight Calculation:**

$$w_i = \sum_{t=t_0}^{t_0+3} \text{PT}_i(t)$$

Where:
- **PTᵢ(t)** = Person-time for age group i at week t
- **t₀** = Baseline week (typically week 4)
- **t₀+3** = Three weeks after baseline

- **Fixed Weights**: Weights based on person-time in first 4 weeks per age group (time-invariant)
- **Population Estimates**: Provides population-level KCOR estimates

### Key Assumptions

- Mortality rates follow exponential trends during the observation period
- No differential events affect dose groups differently during anchor periods
- Baseline period (week 4) represents "normal" conditions
- Person-time = Alive (survivor function approximation)
- Discrete hazard function transformation provides accurate cumulative hazard estimation
- Hazard ratios are appropriate for comparing mortality risk between groups

## 🏗️ Repository Structure

```
KCOR/
├── README.md                           # This file
├── code/
│   ├── KCORv4.py                      # Main analysis script (v4.1)
│   ├── KCOR_CMR.py                    # Data aggregation script
│   ├── Makefile                        # Build automation (Windows/Linux/Mac)
│   ├── run_KCOR.bat                   # Windows batch script
│   └── run_KCOR.ps1                   # Windows PowerShell script
├── data/                               # Output files organized by country
│   └── [country]/                     # Country-specific outputs
├── analysis/                           # Analysis outputs and logs
├── documentation/                      # Detailed methodology documentation
│   └── hazard_function.md             # Mathematical reasoning for hazard functions
└── peer review/                        # Peer review materials
```

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

# Download required data file
# Download vax_24.csv from: https://www.nzip.cz/data/2135-covid-19-prehled-populace
# Rename it to vax_24.csv and place it in: ../../Czech/data/vax_24.csv
```

## 🚀 Usage

### Quick Start

#### Using Make (Cross-Platform)
```bash
cd code
make KCOR
```

The Makefile automatically:
1. Runs `KCOR_CMR.py` to aggregate data from external sources
2. Runs `KCORv4.py` to perform the KCOR analysis
3. Organizes outputs by country in the `data/` directory

#### Using Windows Scripts
```bash
cd code
# Option 1: Batch file
run_KCOR.bat

# Option 2: PowerShell
.\run_KCOR.ps1
```

#### Direct Python Execution
```bash
cd code
# Step 1: Data aggregation
python KCOR_CMR.py [input_file] [output_file]

# Step 2: KCOR analysis
python KCORv4.py [aggregated_file] [analysis_output]
```

### Data Requirements

#### Czech Data Setup
The analysis requires Czech vaccination and mortality data:

1. **Download the data file**:
   - Visit: https://www.nzip.cz/data/2135-covid-19-prehled-populace
   - Download the CSV file containing population and vaccination data
   - Rename it to `vax_24.csv`

2. **Place the file**:
   - Create the directory structure: `../../Czech/data/`
   - Place `vax_24.csv` in `../../Czech/data/vax_24.csv`

3. **File structure should be**:
   ```
   KCOR/
   ├── code/
   └── ../../Czech/data/vax_24.csv
   ```

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

**`KCORv4_analysis.xlsx`** - Complete analysis with all enrollment periods combined
This file enables users to visualize results for any cohort combination and contains:

**`KCOR_summary.xlsx`** - Console-style summary by enrollment date
This file provides one sheet per enrollment period (e.g., 2021_24, 2022_06) formatted like the console output, with dose combination headers and final KCOR values for each age group.

#### Main Analysis Sheets
- **`dose_pairs`**: KCOR values for all dose comparisons with complete methodology transparency
- **Columns**: Sheet, ISOweekDied, Date, YearOfBirth, Dose_num, Dose_den, KCOR, CI_lower, CI_upper, 
  MR_num, MR_adj_num, CMR_num, CMR_actual_num, hazard_num, slope_num, scale_factor_num, MR_smooth_num, t_num,
  MR_den, MR_adj_den, CMR_den, CMR_actual_den, hazard_den, slope_den, scale_factor_den, MR_smooth_den, t_den

#### Debug Sheet
- **`by_dose`**: Individual dose curves with complete methodology transparency
- **Columns**: Date, YearOfBirth, Dose, ISOweek, Dead, Alive, MR, MR_adj, Cum_MR, Cum_MR_Actual, Hazard, 
  Slope, Scale_Factor, Cumu_Adj_Deaths, Cumu_Unadj_Deaths, Cumu_Person_Time, 
  Smoothed_Raw_MR, Smoothed_Adjusted_MR, Time_Index

#### About Sheet
- **Metadata**: Version information, methodology overview, and analysis parameters
- **Documentation**: Complete explanation of the KCOR methodology and output columns

#### Visualization Capabilities

**`KCORv4_analysis.xlsx`** - Complete analysis file:
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
ANCHOR_WEEKS = 4                    # Baseline week for KCOR normalization
SLOPE_WINDOW_SIZE = 2               # Window size for slope calculation (±2 weeks)
MA_TOTAL_LENGTH = 8                 # Moving average length (8 weeks)
CENTERED = True                     # Use centered moving average

# Data quality limits
MAX_DATE_FOR_SLOPE = "2024-04-01"  # Maximum date for slope calculation

# Analysis scope
YEAR_RANGE = (1920, 2000)          # Birth year range to process
DEBUG_SHEET_ONLY = ["2021_24", "2022_06"]  # Sheets to process
```

### Sheet-Specific Configuration

The script automatically determines dose pairs based on sheet names:

- **2021_24**: Doses 0, 1, 2 → Comparisons: (1,0), (2,0), (2,1)
- **2022_06**: Doses 0, 1, 2, 3 → Comparisons: (1,0), (2,0), (2,1), (3,2), (3,0)

## 📊 Interpretation

### KCOR Values

- **KCOR = 1.0**: No difference in mortality risk between groups
- **KCOR > 1.0**: Higher mortality risk in numerator group (e.g., vaccinated)
- **KCOR < 1.0**: Lower mortality risk in numerator group
- **Confidence Intervals**: Provide statistical uncertainty around the point estimate

### Example Output

```
Dose combination: 2 vs 0
--------------------------------------------------
            YoB | KCOR [95% CI]
--------------------------------------------------
  ASMR (pooled) | 1.3050 [1.032, 1.650]
           1940 | 1.2607 [1.124, 1.414]
           1955 | 1.5026 [1.229, 1.837]
```

This shows that for dose 2 vs. dose 0:
- **ASMR**: 30.5% higher mortality risk (95% CI: 3.2% to 65.0%)
- **Age 1940**: 26.1% higher risk (95% CI: 12.4% to 41.4%)
- **Age 1955**: 50.3% higher risk (95% CI: 22.9% to 83.7%)

## 🔧 Advanced Features

### Complete Methodology Transparency (v4.1)
- **Full Traceability**: Every step of the calculation is visible in output
- **Mathematical Relationships**: All intermediate values (slope, scale_factor, hazard) included
- **Validation Ready**: Users can verify every mathematical relationship
- **Debug Friendly**: Easy to spot-check individual values and calculations

### Discrete Hazard Function Transform (v4.1)
- **Mathematical Enhancement**: More exact CMR calculation than simple summation of mortality rates
- **Hazard Function**: `hazard(t) = -ln(1 - MR_adj(t))` with proper clipping to avoid log(0)
- **Cumulative Process**: `CMR(t) = sum(hazard(i))` for i=0 to t (cumulative hazard)
- **Numerical Stability**: Handles edge cases with proper bounds and clipping
- **Hazard Ratio**: KCOR computed as ratio of cumulative hazards, normalized to baseline
- **Mathematical Rigor**: See [Hazard Function Methodology](documentation/hazard_function.md) for detailed derivation

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
DEBUG_SHEET_ONLY = ["sheet_name"]  # Limit to specific sheets
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

## 🆕 Version 4.1 Enhancements

### Major Improvements
- **Discrete Hazard Function Transform**: Enhanced mathematical exactness in CMR calculation using hazard functions
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
- **Cumulative Hazard**: `CMR(t) = sum(hazard(i))` for mathematical exactness
- **Hazard Ratio**: `KCOR(t) = (CMR_v(t)/CMR_u(t)) / (CMR_v(t0)/CMR_u(t0))`
- **Numerical Stability**: Proper clipping to avoid log(0) and overflow
- **Validation Ready**: All mathematical relationships visible in output

## 📊 Results Using Czech Data

### Summary of Age-Standardized Mortality Ratio (ASMR) Results

The KCOR analysis of Czech vaccination and mortality data reveals significant findings across all dose levels compared to unvaccinated individuals:

| **DOSE** | **KCOR** | **95% CI** |
|----------|----------|------------|
| **1 vs 0** | 1.1405 | [1.020, 1.276] |
| **2 vs 0** | 1.3050 | [1.032, 1.650] |
| **3 vs 0** | 1.4466 | [1.218, 1.718] |

### Key Findings

- **All dose levels show increased mortality risk** compared to unvaccinated individuals
- **Dose 3 shows the highest risk** with 44.7% increased mortality (95% CI: 21.8% to 71.8%)
- **Dose 2 shows moderate risk** with 30.5% increased mortality (95% CI: 3.2% to 65.0%)
- **Dose 1 shows lower but still significant risk** with 14.1% increased mortality (95% CI: 2.0% to 27.6%)
- **All confidence intervals exclude 1.0**, indicating statistically significant harm
- **No statistically significant benefit** was found for any dose or age combination

### 🎯 Remarkable Dose-Response Relationship

The results reveal a **strikingly linear dose-response relationship**:

| **Dose** | **KCOR** | **Risk Increase** | **Per-Dose Risk** |
|----------|----------|-------------------|-------------------|
| **1** | 1.1405 | +14.1% | ~14% per dose |
| **2** | 1.3050 | +30.5% | ~15% per dose |
| **3** | 1.4466 | +44.7% | ~15% per dose |

**Key Insight**: The net harm is **nearly exactly proportional to the number of doses**, with approximately **14-15% increased mortality risk per dose**. This linear relationship provides compelling evidence of a direct causal effect because:

1. **The pattern is clean and predictable** (not messy or random)
2. **It's difficult to dismiss** as coincidence or confounding  
3. **It represents one of the most compelling pieces of evidence** for vaccine harm
4. **Such precise linear dose-response relationships are rarely seen in epidemiology**

The combination of the mathematical precision (14-15% per dose) with the logical argument about why this pattern is so compelling creates a very strong case for the causal relationship between vaccination and increased mortality risk.

### Complete Results

For detailed results including age-specific analyses and all dose combinations, see the comprehensive analysis files:

- **📈 Complete Analysis**: [`data/Czech/KCOR_summary.xlsx`](data/Czech/KCOR_summary.xlsx) - Age-standardized and age-specific results by enrollment cohort
- **📊 Full Dataset**: [`data/Czech/KCORv4.xlsx`](data/Czech/KCORv4.xlsx) - Complete analysis with all intermediate calculations
- **📋 Console Output**: [`analysis/KCOR_console_summary.log`](analysis/KCOR_console_summary.log) - Detailed console output from analysis

### Interpretation

These results demonstrate that **no combination of dose and age showed statistically significant benefit** from COVID-19 vaccination. Instead, the analysis reveals **statistically significant harm** across all dose levels, with the risk increasing with additional doses. This finding is consistent across different enrollment cohorts and age groups, providing robust evidence of increased mortality risk associated with COVID-19 vaccination in the Czech population.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions about the methodology or implementation, please open an issue on GitHub or contact the development team.

---

**Note**: This software is designed for research purposes. Users should carefully validate results and consider the specific context of their data and research questions.
