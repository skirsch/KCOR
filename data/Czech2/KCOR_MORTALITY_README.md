# KCOR Mortality Analysis Pipeline

This directory contains the implementation of the KCOR mortality analysis pipeline for person-month survival analysis on Czech event-level data.

## Czech2 Dataset
Download here: 
https://www.nzip.cz/data/2547-reprodukcni-zdravotni-udalosti-otevrena-data

Stanislav Veselý wrote:

```Czech officials released a data item series on births and mothers' vaccination status. There is more information there, but this is the most interesting. Until now, we only had aggregated data, which clearly showed that vaccinated women have about 30% fewer children than would correspond to their numbers.  The row dataset could show much more than agregate data. The dataset in on address https://www.nzip.cz/data/2547-reprodukcni-zdravotni-udalosti-otevrena-data. This is official data from Czech goverment.```

This dataset contains the ICD10 code for the underlying cause of death.

Note: This is NOT as useful as the MCOD dataset maintained by the CDC which is MULTIPLE causes of death.


## Overview

The KCOR mortality analysis estimates whether **vaccination increased or decreased the probability of death** for recipients using:
1. Fixed cohorts based on vaccination status at enrollment date
2. Person-month survival data from enrollment forward
3. Cohort-specific hazard curves
4. KCOR slope-normalization to remove age/health structure bias
5. Adjusted cumulative hazards and KCOR ratios

## Files

### Core Pipeline
- `code/kcor_mortality.py` - Main pipeline implementation
- `code/kcor_mortality_sensitivity.py` - Sensitivity analysis automation
- `code/kcor_mortality_age_stratified.py` - Age-stratified analysis
- `code/kcor_mortality_stats.py` - Statistical inference (CIs, hypothesis tests)
- `code/kcor_mortality_plots.py` - Enhanced visualizations
- `code/kcor_mortality_config.py` - Configuration management

### Configuration
- `config/kcor_mortality_config.yaml` - Centralized configuration file

### Documentation
- `data/Czech2/kcor_mortality_analysis.md` - Original pipeline instructions
- `data/Czech2/KCOR_MORTALITY_README.md` - This file

## Quick Start

### Basic Usage

Run the core pipeline with default parameters:

```bash
python code/kcor_mortality.py data/Czech2/data.csv data/Czech2/kcor_mortality_output \
    --enroll-year 2021 --enroll-month 7 --max-fu-months 24
```

### With Custom Parameters

```bash
python code/kcor_mortality.py data/Czech2/data.csv output_dir \
    --enroll-year 2021 \
    --enroll-month 7 \
    --max-fu-months 24 \
    --quiet-min 3 \
    --quiet-max 10 \
    --separate-doses
```

### Sensitivity Analysis

Run multiple configurations automatically:

```bash
python code/kcor_mortality_sensitivity.py data/Czech2/data.csv output_dir
```

With custom configuration file:

```bash
python code/kcor_mortality_sensitivity.py data/Czech2/data.csv output_dir \
    --config-file config/kcor_mortality_config.yaml
```

### Age-Stratified Analysis

```bash
python code/kcor_mortality_age_stratified.py data/Czech2/data.csv output_dir \
    --enroll-year 2021 --enroll-month 7
```

With custom age bands:

```bash
python code/kcor_mortality_age_stratified.py data/Czech2/data.csv output_dir \
    --age-bands "65-74:65:74" "75-84:75:84" "85+:85:200"
```

### Statistical Inference

Add confidence intervals and hypothesis tests:

```bash
python code/kcor_mortality_stats.py \
    output_dir/raw/kcor_hazard_adjusted.csv \
    output_dir/results/kcor_ratios_with_ci.csv \
    --alpha 0.05
```

### Enhanced Visualizations

Create all diagnostic plots:

```bash
python code/kcor_mortality_plots.py \
    output_dir/raw/kcor_hazard_adjusted.csv \
    output_dir/plots \
    --slopes-csv output_dir/raw/kcor_slopes.csv \
    --kcor-csv output_dir/results/kcor_ratios.csv \
    --plot-type all
```

## Input Data Format

The pipeline expects a CSV file with Czech event-level data. Required columns (case-insensitive):

- `id_zeny` - Person ID
- `rok_narozeni` - Birth year
- `udalost` - Event type (values include `umrti`, `covid ockovani`, etc.)
- `rok_udalosti` - Year of event
- `mesic_udalosti` - Month of event (1-12)
- `covid_ocko_poradi_davky` - Dose number for vaccination events
- ICD column (contains "diag" in name) - ICD-10 code for death events

## Output Structure

```
output_dir/
├── raw/
│   ├── kcor_hazard_raw.csv          # Raw hazards per cohort and month
│   ├── kcor_slopes.csv               # Gompertz slopes per cohort
│   └── kcor_hazard_adjusted.csv     # Slope-normalized hazards
├── results/
│   ├── kcor_ratios.csv               # KCOR ratios over time
│   └── kcor_summary.csv              # Summary statistics
└── plots/
    ├── kcor_ratio_plot.png           # KCOR ratio plot
    └── hazard_curves.png              # Hazard curves
```

For sensitivity analysis:

```
output_dir/
└── sensitivity/
    ├── {config_id}/
    │   └── [same structure as above]
    ├── summary_all_configs.csv       # Summary across all configurations
    └── comparison_plots/
        └── [comparison visualizations]
```

For age-stratified analysis:

```
output_dir/
└── age_stratified/
    ├── {age_band}/
    │   └── [same structure as above]
    └── summary_by_age_band.csv       # Summary across age bands
```

## Key Parameters

### Enrollment Date
- **Purpose**: Defines the baseline time point for cohort assignment
- **Typical values**: `2021-01`, `2021-07`, `2022-01`
- **Effect**: Determines which vaccinations count toward baseline status

### Quiet Period
- **Purpose**: Time window for estimating Gompertz slopes (should avoid COVID waves)
- **Typical values**: Months 3-10, 6-15, or 9-18 after enrollment
- **Effect**: Affects slope normalization quality

### Follow-up Horizon
- **Purpose**: Maximum months to track after enrollment
- **Typical values**: 12, 18, or 24 months
- **Effect**: Determines analysis window length

### Cohort Definitions
- **Grouped**: Doses 3+ grouped together as `dose3plus`
- **Separate**: Doses 3, 4, 5, 6 analyzed separately
- **Effect**: Affects granularity of dose-specific analysis

## Interpretation

### KCOR Ratios
- **KCOR > 1**: Higher adjusted cumulative hazard in vaccinated cohort (potential harm)
- **KCOR < 1**: Lower adjusted cumulative hazard in vaccinated cohort (potential benefit)
- **KCOR ≈ 1**: Neutral effect

### Confidence Intervals
- 95% CI computed using Nelson-Aalen variance estimator
- If CI excludes 1.0, effect is statistically significant (p < 0.05)

### Sensitivity Analysis
- **Consistent patterns**: Same direction (harm/benefit) across ≥80% of configurations
- **Mixed results**: No clear pattern across configurations
- **Divergent configurations**: Results that deviate significantly from median

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy pyyaml
```

Or on WSL:

```bash
apt install python3-pandas python3-numpy python3-matplotlib python3-seaborn python3-scipy python3-yaml
```

## Examples

### Example 1: Basic Analysis

```python
from kcor_mortality import run_kcor_pipeline

results = run_kcor_pipeline(
    csv_path="data/Czech2/data.csv",
    output_dir="output/basic",
    enroll_year=2021,
    enroll_month=7,
    max_fu_months=24,
    quiet_t_min=3,
    quiet_t_max=10
)

# Access results
kcor_df = results["kcor"]
print(kcor_df.head())
```

### Example 2: Sensitivity Analysis

```python
from kcor_mortality_sensitivity import run_sensitivity_analysis

summary_df = run_sensitivity_analysis(
    csv_path="data/Czech2/data.csv",
    base_output_dir="output/sensitivity"
)

# Analyze consistency
for cohort in summary_df["cohort"].unique():
    cohort_data = summary_df[summary_df["cohort"] == cohort]
    above_1 = (cohort_data["final_kcor"] > 1.0).mean()
    print(f"{cohort}: {above_1*100:.1f}% of configs with KCOR > 1")
```

### Example 3: Age-Stratified Analysis

```python
from kcor_mortality_age_stratified import run_age_stratified_analysis

results = run_age_stratified_analysis(
    csv_path="data/Czech2/data.csv",
    output_dir="output/age_stratified",
    enroll_year=2021,
    enroll_month=7
)

# Access age-specific results
for age_band, result in results.items():
    print(f"{age_band}: {result['n_persons']} persons")
    print(result["kcor"].head())
```

## Troubleshooting

### Common Issues

1. **"No vaccination events found"**
   - Check that `udalost` column contains "covid ockovani" (case-insensitive)
   - Verify column names are correctly normalized

2. **"No slopes could be fitted"**
   - Check quiet period: ensure it's within follow-up window
   - Verify sufficient data points in quiet period (need ≥2 per cohort)
   - Check for zero hazards in quiet period

3. **"Reference cohort not found"**
   - Verify cohort labeling function produces expected names
   - Check that unvaccinated persons exist in data

4. **Memory errors with large datasets**
   - Use chunked processing for very large CSV files
   - Reduce follow-up horizon or filter data before analysis

### Performance Tips

- For large datasets (>1GB), consider filtering to specific age ranges first
- Use `--separate-doses` only if needed (increases computation)
- Reduce number of sensitivity configurations for faster runs

## Validation

The pipeline has been validated against:
- Existing KCOR.py methodology (weekly aggregation)
- Known test cases with expected outcomes
- Numerical stability checks (edge cases)

## References

- Original pipeline instructions: `data/Czech2/kcor_mortality_analysis.md`
- KCOR methodology: `documentation/hazard_function.md`
- Main KCOR implementation: `code/KCOR.py`

## License

See main project LICENSE file.

