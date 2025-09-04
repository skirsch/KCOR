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

KCOR basically allows you to run a randomized trial with respect to the death outcome, using observational data.

KCOR allows us, for the first time, to objectively answer very important societal questions such as, “Did the COVID vaccine kill more people than it saved?”

## 🔬 Methodology

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

## Output when used on the Czech data
There was no combination of dose and age where there was a statistically significant benefit. It was pretty much all statistically significant harm. All the CI's had a high CI that was >1. See the [log file](analysis/KCOR_console_summary.log) for the full data.
```
Dose combination: 2 vs 0
--------------------------------------------------
  ASMR (pooled)   | KCOR [95% CI]:   1.3050 [1.032, 1.650]
  Age 1920        | KCOR [95% CI]:   0.8667 [0.642, 1.171]
  Age 1925        | KCOR [95% CI]:   0.8732 [0.746, 1.023]
  Age 1930        | KCOR [95% CI]:   0.9621 [0.856, 1.082]
  Age 1935        | KCOR [95% CI]:   1.2485 [1.110, 1.404]
  Age 1940        | KCOR [95% CI]:   1.2607 [1.124, 1.414]
  Age 1945        | KCOR [95% CI]:   1.3354 [1.182, 1.508]
  Age 1950        | KCOR [95% CI]:   1.5983 [1.376, 1.856]
  Age 1955        | KCOR [95% CI]:   1.5026 [1.229, 1.837]
  Age 1960        | KCOR [95% CI]:   1.3343 [1.034, 1.721]
  Age 1965        | KCOR [95% CI]:   1.7859 [1.174, 2.717]
  Age 1970        | KCOR [95% CI]:   1.0115 [0.662, 1.547]
  Age 1975        | KCOR [95% CI]:   1.7949 [0.894, 3.602]
  Age 1980        | KCOR [95% CI]:   0.8737 [0.418, 1.825]
  Age 1985        | KCOR [95% CI]:   1.1823 [0.354, 3.950]
  Age 1990        | KCOR [95% CI]:   1.9061 [0.428, 8.497]
  Age 1995        | KCOR [95% CI]:   0.7569 [0.165, 3.470]
  Age 2000        | KCOR [95% CI]:   0.5809 [0.121, 2.782]

  Dose combination: 3 vs 0
--------------------------------------------------
  ASMR (pooled)   | KCOR [95% CI]:   1.4466 [1.218, 1.718]
  Age 1920        | KCOR [95% CI]:   1.2191 [0.874, 1.700]
  Age 1925        | KCOR [95% CI]:   1.3087 [1.111, 1.541]
  Age 1930        | KCOR [95% CI]:   1.4961 [1.331, 1.682]
  Age 1935        | KCOR [95% CI]:   1.4936 [1.327, 1.682]
  Age 1940        | KCOR [95% CI]:   1.6118 [1.434, 1.811]
  Age 1945        | KCOR [95% CI]:   1.6706 [1.483, 1.882]
  Age 1950        | KCOR [95% CI]:   1.7781 [1.548, 2.042]
  Age 1955        | KCOR [95% CI]:   1.4656 [1.229, 1.748]
  Age 1960        | KCOR [95% CI]:   1.3396 [1.062, 1.690]
  Age 1965        | KCOR [95% CI]:   1.1809 [0.883, 1.580]
  Age 1970        | KCOR [95% CI]:   1.3752 [0.960, 1.970]
  Age 1975        | KCOR [95% CI]:   1.9399 [1.178, 3.193]
  Age 1980        | KCOR [95% CI]:   1.1662 [0.639, 2.127]
  Age 1985        | KCOR [95% CI]:   0.9350 [0.424, 2.064]
  Age 1990        | KCOR [95% CI]:   2.3429 [0.671, 8.178]
  Age 1995        | KCOR [95% CI]:   1.1485 [0.306, 4.304]
  Age 2000        | KCOR [95% CI]:   1.3872 [0.353, 5.453]
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions about the methodology or implementation, please open an issue on GitHub or contact the development team.

---

**Note**: This software is designed for research purposes. Users should carefully validate results and consider the specific context of their data and research questions.
