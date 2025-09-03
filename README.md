# KCOR v4.0 - Kirsch Cumulative Outcomes Ratio Analysis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

KCOR (Kirsch Cumulative Outcomes Ratio) is a robust statistical methodology for analyzing relative mortality risk between different vaccination groups while accounting for underlying time trends. This repository contains the complete analysis pipeline for computing KCOR values from mortality data.

## Methodology

### Core Concept

KCOR represents the ratio of cumulative mortality rates between two groups (e.g., vaccinated vs. unvaccinated), normalized to 1 at a baseline period. This approach provides interpretable estimates of relative mortality risk that account for:

- **Time-varying trends** in mortality rates
- **Baseline differences** between groups
- **Statistical uncertainty** in the estimates

### Analysis Pipeline

#### 1. Data Preprocessing
- **Enrollment Date Filtering**: Data processing starts from the enrollment date derived from sheet names (e.g., "2021_24" = 2021, week 24)
- **Sex Aggregation**: Mortality data is aggregated across sexes for each (YearOfBirth, Dose, DateDied) combination
- **Smoothing**: 8-week centered moving average applied to raw mortality rates to reduce noise

#### 2. Slope Calculation (Lookup Table Method)
- **Anchor Points**: Uses predefined time points (e.g., weeks 53 and 114 from enrollment for 2021_24)
- **Window Approach**: For each anchor point, creates a ±2 week window (5 points total)
- **Geometric Mean**: Calculates geometric mean of smoothed MR values within each window

**Slope Formula:**
```
r = (1/Δt) ln(B̃/Ã)
```

Where:
- **Ã** = Geometric mean of MR values in window around first anchor: `Ã = GM(MR_{t∈[t₀-w, t₀+w]})`
- **B̃** = Geometric mean of MR values in window around second anchor: `B̃ = GM(MR_{t∈[t₁-w, t₁+w]})`
- **Δt** = Time difference between anchor points (in weeks)
- **w** = Window size (default: 2 weeks)

**Geometric Mean Calculation:**
```
GM(x₁, x₂, ..., xₙ) = exp(1/n × Σ ln(xᵢ))
```

- **Consistency**: Same anchor points used for all doses to ensure comparability
- **Quiet Periods**: Anchor dates chosen during periods with minimal differential events (COVID waves, policy changes, etc.)

#### 3. Mortality Rate Adjustment
- **Exponential Slope Removal**: `MR_adj = MR × exp(-slope × (t - t0))`
- **Baseline Normalization**: t0 = baseline week (typically week 4) where KCOR is normalized to 1
- **Dose-Specific Slopes**: Each dose-age combination gets its own slope for adjustment

#### 4. KCOR Computation
**KCOR Formula:**
```
KCOR(t) = [CMR_num(t) / CMR_den(t)] / [CMR_num(t₀) / CMR_den(t₀)]
```

Where:
- **CMR(t)** = Cumulative adjusted mortality rate at time t: `CMR(t) = cumD_adj(t) / cumPT(t)`
- **t₀** = Baseline time (typically week 4) where KCOR is normalized to 1
- **cumD_adj(t)** = Cumulative adjusted deaths up to time t
- **cumPT(t)** = Cumulative person-time up to time t

**Mortality Rate Adjustment:**
```
MR_adj(t) = MR(t) × exp(-r × (t - t₀))
```

Where:
- **r** = Calculated slope for the specific dose-age combination
- **MR(t)** = Raw mortality rate at time t
- **t₀** = Baseline time for normalization

- **Baseline Reference**: Baseline values taken at week 4 (or first available week)
- **Interpretation**: KCOR = 1 at baseline, showing relative risk evolution over time

#### 5. Uncertainty Quantification
**95% Confidence Interval Calculation:**

The variance of KCOR is calculated using proper uncertainty propagation:

```
Var[KCOR(t)] = KCOR(t)² × [Var[cumD_num(t)]/cumD_num(t)² + Var[cumD_den(t)]/cumD_den(t)² + Var[cumD_num(t₀)]/cumD_num(t₀)² + Var[cumD_den(t₀)]/cumD_den(t₀)²]
```

**Confidence Interval Bounds:**
```
CI_lower(t) = KCOR(t) × exp(-1.96 × √Var[ln(KCOR(t))])
CI_upper(t) = KCOR(t) × exp(1.96 × √Var[ln(KCOR(t))])
```

Where:
- **Var[D] ≈ D**: Using binomial variance approximation for death counts
- **Var[ln(KCOR)]**: Variance on log scale for proper uncertainty propagation
- **1.96**: 95% confidence level multiplier (standard normal distribution)

- **Log-Scale Calculation**: CI bounds calculated on log scale then exponentiated for proper asymmetry

#### 6. Age Standardization
**ASMR Pooling Formula:**

The age-standardized KCOR is calculated using fixed baseline weights:

```
KCOR_ASMR(t) = exp(Σ wᵢ × ln(KCORᵢ(t)) / Σ wᵢ)
```

Where:
- **wᵢ** = Fixed weight for age group i (person-time in first 4 weeks)
- **KCORᵢ(t)** = KCOR value for age group i at time t
- **ln(KCORᵢ(t))** = Natural logarithm of KCOR for age group i

**Weight Calculation:**
```
wᵢ = Σ PTᵢ(t) for t ∈ [t₀, t₀+3]
```

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

## Repository Structure

```
KCOR/
├── README.md                           # This file
├── code/
│   ├── KCORv4.py                      # Main analysis script
│   ├── Makefile                        # Build automation (Linux/Mac)
│   ├── run_KCOR.bat                   # Windows batch script
│   └── run_KCOR.ps1                   # Windows PowerShell script
├── analysis/                           # Analysis outputs and logs
├── documentation/                      # Detailed methodology documentation
└── peer review/                        # Peer review materials
```

## Installation & Dependencies

### Requirements
- Python 3.8 or higher
- pandas
- numpy
- statsmodels
- openpyxl (for Excel output)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd KCOR

# Install dependencies
pip install pandas numpy statsmodels openpyxl
```

## Usage

### Quick Start

#### Using Make (Linux/Mac)
```bash
cd code
make KCOR
```

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
python KCORv4.py ../../Czech/data/KCOR_output.xlsx ../../Czech/analysis/KCOR_analysis.xlsx
```

### Input Data Format

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

The analysis produces Excel workbooks with multiple sheets:

#### Main Analysis Sheets
- **Individual Sheets**: KCOR values over time for each input sheet
- **ALL Sheet**: Combined results from all processed sheets
- **Columns**: Sheet, ISOweekDied, Date, YearOfBirth, Dose_num, Dose_den, KCOR, CI_lower, CI_upper, MR_num, MR_adj_num, CMR_num, MR_den, MR_adj_den, CMR_den

#### Debug Sheet
- **Individual Dose Curves**: Raw and adjusted mortality rates for each dose-age combination
- **Columns**: Date, ISOweekDied, YearOfBirth, Dose, Dead, Alive, MR, MR_adj, CMR, MR_smooth, Smoothed_Raw_MR, Smoothed_Adjusted_MR

## Configuration

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

## Interpretation

### KCOR Values

- **KCOR = 1.0**: No difference in mortality risk between groups
- **KCOR > 1.0**: Higher mortality risk in numerator group (e.g., vaccinated)
- **KCOR < 1.0**: Lower mortality risk in numerator group
- **Confidence Intervals**: Provide statistical uncertainty around the point estimate

### Example Output

```
Dose combination: 2 vs 0
--------------------------------------------------
  ASMR (pooled)   | KCOR [95% CI]:   1.3050 [1.032, 1.650]
  Age 1940        | KCOR [95% CI]:   1.2607 [1.124, 1.414]
  Age 1955        | KCOR [95% CI]:   1.5026 [1.229, 1.837]
```

This shows that for dose 2 vs. dose 0:
- **ASMR**: 30.5% higher mortality risk (95% CI: 3.2% to 65.0%)
- **Age 1940**: 26.1% higher risk (95% CI: 12.4% to 41.4%)
- **Age 1955**: 50.3% higher risk (95% CI: 22.9% to 83.7%)

## Advanced Features

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

## Troubleshooting

### Common Issues

1. **Missing Data**: Ensure all required columns are present in input files
2. **Date Formats**: Verify dates are in proper datetime format
3. **Memory Issues**: Large datasets may require processing in smaller chunks
4. **Slope Calculation**: Check that anchor points fall within available data range

### Debug Mode

Enable detailed debugging by setting:
```python
DEBUG_VERBOSE = True
DEBUG_SHEET_ONLY = ["sheet_name"]  # Limit to specific sheets
YEAR_RANGE = (1940, 1945)          # Limit to specific age range
```

## Contributing

We welcome contributions to improve the KCOR methodology and implementation. Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Submit a pull request with detailed description

## Citation

If you use KCOR in your research, please cite:

```
KCOR v4.0 - Kirsch Cumulative Outcomes Ratio Analysis
[Your paper title]
[Authors]
[Journal/Conference]
[Year]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions about the methodology or implementation, please open an issue on GitHub or contact the development team.

---

**Note**: This software is designed for research purposes. Users should carefully validate results and consider the specific context of their data and research questions.
