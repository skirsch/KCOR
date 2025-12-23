# Makefile for KCOR analysis pipeline. This is not the root Makefile.

# Configuration - Dataset namespace (inherited from root Makefile)
DATASET ?= Czech
# Python executable (defaults to system python3, but can be overridden with venv path)
PYTHON ?= python3
# Paths for inputs
RECORDS_CSV      = ../data/$(DATASET)/records.csv
# Optional KRF input produced by dataset converters
RECORDS_KRF      = ../data/$(DATASET)/krf.csv
# Raw Czech CSV (compressed asset) — used only for datasets that ship this file
RAW_CSV_GZ       = ../data/$(DATASET)/Otevrena-data-NR-26-30-COVID-19-prehled-populace-2024-01.csv.gzip
# Keep the same base name when unzipped (…csv.gzip -> …csv)
RAW_CSV          = $(RAW_CSV_GZ:.gzip=)
# Always target the standardized records.csv as input; provide a single rule to build it
INPUT_DATA       = $(RECORDS_CSV)
CMR_OUTPUT       = ../data/$(DATASET)/KCOR_CMR.xlsx
KCOR_OUTPUT      = ../data/$(DATASET)/KCOR.xlsx
VARIABLE_OUTPUT  = ../data/$(DATASET)/KCOR_variable.xlsx
TS_OUTPUT        = ../data/$(DATASET)/KCOR_ts.xlsx
MC_OUTPUT        = ../data/$(DATASET)/KCOR_CMR_MC.xlsx
KCOR_MC_OUTPUT   = ../data/$(DATASET)/KCOR_MC.xlsx


.PHONY: clean clean-all help sensitivity monte_carlo mortality mortality_sensitivity mortality_age mortality_stats mortality_plots mortality_all

# Default target - run complete KCOR pipeline
KCOR: $(KCOR_OUTPUT)

# KCOR analysis depends on CMR output and KCOR.py script
$(KCOR_OUTPUT): $(CMR_OUTPUT) KCOR.py
	@echo "Running KCOR analysis..."
	$(PYTHON) KCOR.py $(CMR_OUTPUT) $(KCOR_OUTPUT) "Primary Analysis" KCOR_summary.log
	@echo "KCOR analysis complete!"

# CMR aggregation depends on input data and KCOR_CMR.py script
TMP_CMR := /tmp/KCOR_CMR.$(DATASET).xlsx

$(CMR_OUTPUT): $(INPUT_DATA) KCOR_CMR.py | ../data/$(DATASET)
	@echo "Running KCOR CMR aggregation... (writing to $(TMP_CMR) then moving)"
	@rm -f $(TMP_CMR)
	$(PYTHON) KCOR_CMR.py $(INPUT_DATA) $(TMP_CMR)
	@echo "Wrote temporary: $(TMP_CMR)"
	@ls -lh $(TMP_CMR)
	@# Guard: do not move a zero-byte temp file
	@test -s $(TMP_CMR) || (echo "ERROR: Temp CMR file is zero bytes. Skipping move and failing target." && exit 1)
	@echo "Moving $(TMP_CMR) -> $(CMR_OUTPUT)"
	@mv -f $(TMP_CMR) $(CMR_OUTPUT)
	@ls -lh $(CMR_OUTPUT)
	@echo "CMR aggregation complete!"

# Build standardized records.csv from available inputs (prefer krf.csv)
$(RECORDS_CSV): | ../data/$(DATASET)
	@set -e; \
	if [ -f "$(RECORDS_KRF)" ]; then \
	  echo "Adapting KRF CSV to Czech-like input..."; \
	  $(PYTHON) krf_adapter.py $(RECORDS_KRF) $(RECORDS_CSV); \
	  echo "Wrote: $(RECORDS_CSV) from $(RECORDS_KRF)"; \
	elif [ -f "$(RAW_CSV)" ] || [ -f "$(RAW_CSV_GZ)" ]; then \
	  if [ -f "$(RAW_CSV_GZ)" ]; then \
	    echo "Decompressing $(RAW_CSV_GZ) -> $(RAW_CSV) ..."; \
	    gzip -dc "$(RAW_CSV_GZ)" > "$(RAW_CSV)"; \
	  fi; \
	  echo "Using raw CSV as standardized input (dataset ships Czech-format raw file)"; \
	  cp "$(RAW_CSV)" "$(RECORDS_CSV)"; \
	else \
	  echo "ERROR: No input found. Provide $(RECORDS_KRF) or $(RAW_CSV_GZ)."; \
	  exit 1; \
	fi

# If krf.csv is missing, build it via the dataset converter
$(RECORDS_KRF):
	@echo "Building KRF via dataset converter in data/$(DATASET)..."
	@$(MAKE) -C ../data/$(DATASET) convert

# Keep convenience target to run CMR on KRF explicitly
.PHONY: KCOR_from_krf
CMR_from_krf: | ../data/$(DATASET)
	@echo "Adapting KRF CSV to Czech-like input..."
	@$(PYTHON) krf_adapter.py ../data/$(DATASET)/krf.csv ../data/$(DATASET)/records.csv
	@echo "Running KCOR CMR aggregation on adapted input..."
	$(MAKE) CMR DATASET=$(DATASET)

# Convenience target for CMR only
CMR: $(CMR_OUTPUT)

# Monte Carlo mode - generates bootstrap samples
MC_ITERATIONS ?= 100
TMP_MC := /tmp/KCOR_CMR_MC.$(DATASET).xlsx

monte_carlo: $(KCOR_MC_OUTPUT)

$(MC_OUTPUT): $(INPUT_DATA) KCOR_CMR.py | ../data/$(DATASET)
	@echo "Running KCOR CMR Monte Carlo aggregation... (writing to $(TMP_MC) then moving)"
	@echo "  Monte Carlo iterations: $(MC_ITERATIONS)"
	@rm -f $(TMP_MC)
	MONTE_CARLO=1 MC_ITERATIONS=$(MC_ITERATIONS) $(PYTHON) KCOR_CMR.py $(INPUT_DATA) $(TMP_MC)
	@echo "Wrote temporary: $(TMP_MC)"
	@ls -lh $(TMP_MC)
	@# Guard: do not move a zero-byte temp file
	@test -s $(TMP_MC) || (echo "ERROR: Temp MC file is zero bytes. Skipping move and failing target." && exit 1)
	@echo "Moving $(TMP_MC) -> $(MC_OUTPUT)"
	@mv -f $(TMP_MC) $(MC_OUTPUT)
	@ls -lh $(MC_OUTPUT)
	@echo "Monte Carlo CMR aggregation complete!"

$(KCOR_MC_OUTPUT): $(MC_OUTPUT) KCOR.py | ../data/$(DATASET)
	@echo "Running KCOR analysis on Monte Carlo CMR data..."
	@echo "  Input: $(MC_OUTPUT)"
	@echo "  Output: $(KCOR_MC_OUTPUT)"
	MONTE_CARLO=1 $(PYTHON) KCOR.py $(MC_OUTPUT) $(KCOR_MC_OUTPUT) "Monte Carlo Analysis" KCOR_MC_summary.log
	@echo "Monte Carlo KCOR analysis complete!"

# Variable cohort aggregation (single-sheet; current dose per week)
KCOR_variable: $(VARIABLE_OUTPUT)

$(VARIABLE_OUTPUT): $(INPUT_DATA) KCOR_variable.py | ../data/$(DATASET)
	@echo "Running KCOR variable-cohort aggregation..."
	$(PYTHON) KCOR_variable.py $(INPUT_DATA) $(VARIABLE_OUTPUT)
	@echo "KCOR_variable aggregation complete!"

# Time series aggregation (single-sheet; weeks after each dose independently)
ts: $(TS_OUTPUT)

KCOR_ts: $(TS_OUTPUT)

$(TS_OUTPUT): $(INPUT_DATA) KCOR_ts.py | ../data/$(DATASET)
	@echo "Running KCOR time series aggregation..."
	$(PYTHON) KCOR_ts.py $(INPUT_DATA) $(TS_OUTPUT)
	@echo "KCOR time series aggregation complete!"

# Ensure output directory exists (real directory target)
../data/$(DATASET):
	@$(PYTHON) -c "import os; os.makedirs('$@', exist_ok=True)"

# Raw CSV inflation handled inside the unified records.csv rule above

# Clean up all output files for current country
clean:
	@echo "Cleaning up output files for $(DATASET)..."
	@$(PYTHON) -c "import os; [os.remove(f) for f in ['$(CMR_OUTPUT)', '$(KCOR_OUTPUT)'] if os.path.exists(f)]"
	@if [ "$(DATASET)" = "Czech2" ]; then \
		echo "Cleaning up mortality analysis outputs..."; \
		$(PYTHON) -c "import shutil, os; shutil.rmtree('$(MORTALITY_OUTPUT_DIR)') if os.path.exists('$(MORTALITY_OUTPUT_DIR)') else None"; \
	fi
	@echo "Cleanup complete!"

# Clean up all output files for all countries
clean-all:
	@echo "Cleaning up all output files..."
	@$(PYTHON) -c "import shutil, os; shutil.rmtree('../data') if os.path.exists('../data') else None"
	@echo "Cleanup complete!"

# Help target
help:
	@echo "Available targets:"
	@echo "  KCOR      - Run complete KCOR pipeline (CMR + analysis)"
	@echo "  CMR       - Run only CMR aggregation step"
	@echo "  CMR_from_krf - Run CMR aggregation step on KRF data; specify DATASET=japan to use Japan KRF data"
	@echo "  monte_carlo - Run Monte Carlo bootstrap sampling (100 iterations by default, override with MC_ITERATIONS=N)"
	@echo "  sensitivity - Smoke test sensitivity plumbing (echo SA_* and run analysis)"
	@echo ""
	@echo "KCOR Mortality Analysis (Czech2 dataset):"
	@echo "  mortality - Run basic KCOR mortality analysis pipeline"
	@echo "  mortality_sensitivity - Run sensitivity analysis (multiple configurations)"
	@echo "  mortality_age - Run age-stratified analysis"
	@echo "  mortality_stats - Add statistical inference (CIs, p-values) to results"
	@echo "  mortality_plots - Create enhanced visualizations"
	@echo "  mortality_all - Run ALL mortality analyses (recommended for complete analysis)"
	@echo ""
	@echo "Mortality analysis variables (override on CLI):"
	@echo "  ENROLL_YEAR=<year>     - Enrollment year (default: 2021)"
	@echo "  ENROLL_MONTH=<month>   - Enrollment month 1-12 (default: 7)"
	@echo "  MAX_FU_MONTHS=<months> - Maximum follow-up months (default: 24)"
	@echo "  QUIET_MIN=<month>      - Quiet period start (default: 3)"
	@echo "  QUIET_MAX=<month>      - Quiet period end (default: 10)"
	@echo ""
	@echo "  clean     - Remove output files for current dataset ($(DATASET))"
	@echo "  clean-all - Remove all output files for all countries"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Current configuration:"
	@echo "Current configuration:"
	@echo "  Dataset: $(DATASET)"
	@echo "  Input:   $(INPUT_DATA)"
	@echo "  CMR:     $(CMR_OUTPUT)"
	@echo "  KCOR:    $(KCOR_OUTPUT)"
	@echo "  MC:      $(MC_OUTPUT)"

## Minimal sensitivity plumbing target (kept for convenience)
# Runs KCOR6 sensitivity analysis varying baseline weeks × quiet-start offsets.
# Outputs grid tables (offsets as columns, baseline weeks as rows) per cohort with KCOR values.
sensitivity: $(CMR_OUTPUT) KCOR.py | ../data/$(DATASET)
	@echo "[sensitivity] Environment:"
	@echo "  SA_COHORTS           = $(SA_COHORTS)"
	@echo "  SA_DOSE_PAIRS        = $(SA_DOSE_PAIRS)"
	@echo "  SA_BASELINE_WEEKS    = $(SA_BASELINE_WEEKS)"
	@echo "  SA_QUIET_START_OFFSETS = $(SA_QUIET_START_OFFSETS)"
	@echo "Running KCOR sensitivity analysis ..."
	@echo "  Computing for age=-2 cohort (all ages aggregated) only"
	@$(PYTHON) -c "import os; os.makedirs('../test/sensitivity/out', exist_ok=True)"
	SENSITIVITY_ANALYSIS=1 SA_COHORTS="$(SA_COHORTS)" SA_DOSE_PAIRS="$(SA_DOSE_PAIRS)" \
		SA_BASELINE_WEEKS="$(SA_BASELINE_WEEKS)" SA_QUIET_START_OFFSETS="$(SA_QUIET_START_OFFSETS)" \
		$(PYTHON) KCOR.py $(CMR_OUTPUT) ../test/sensitivity/out/KCOR_SA.xlsx "Sensitivity Analysis" KCOR_summary_SA.log
	@echo "Sensitivity run complete! Output: test/sensitivity/out/KCOR_SA.xlsx"

# KCOR Mortality Analysis Pipeline
# Input data for Czech2 dataset
MORTALITY_INPUT_CSV = ../data/Czech2/data.csv
MORTALITY_OUTPUT_DIR = ../data/Czech2/kcor_mortality_output
MORTALITY_CONFIG = ../config/kcor_mortality_config.yaml

# Default parameters (can be overridden)
ENROLL_YEAR ?= 2021
ENROLL_MONTH ?= 7
MAX_FU_MONTHS ?= 24
QUIET_MIN ?= 3
QUIET_MAX ?= 10

# Basic mortality analysis
mortality: kcor_mortality.py | ../data/Czech2
	@echo "Running KCOR mortality analysis pipeline..."
	@ENROLL_YEAR_VAL=$${ENROLL_YEAR:-2021}; \
	ENROLL_MONTH_VAL=$${ENROLL_MONTH:-7}; \
	MAX_FU_VAL=$${MAX_FU_MONTHS:-24}; \
	QUIET_MIN_VAL=$${QUIET_MIN:-3}; \
	QUIET_MAX_VAL=$${QUIET_MAX:-10}; \
	echo "  Enrollment: $$ENROLL_YEAR_VAL-$$ENROLL_MONTH_VAL"; \
	echo "  Follow-up: $$MAX_FU_VAL months"; \
	echo "  Quiet period: [$$QUIET_MIN_VAL, $$QUIET_MAX_VAL]"; \
	$(PYTHON) kcor_mortality.py $(MORTALITY_INPUT_CSV) $(MORTALITY_OUTPUT_DIR) \
		--enroll-year $$ENROLL_YEAR_VAL \
		--enroll-month $$ENROLL_MONTH_VAL \
		--max-fu-months $$MAX_FU_VAL \
		--quiet-min $$QUIET_MIN_VAL \
		--quiet-max $$QUIET_MAX_VAL
	@echo "KCOR mortality analysis complete! Output: $(MORTALITY_OUTPUT_DIR)"

# Sensitivity analysis (multiple configurations)
mortality_sensitivity: kcor_mortality_sensitivity.py | ../data/Czech2
	@echo "Running KCOR mortality sensitivity analysis..."
	@if [ -f "$(MORTALITY_CONFIG)" ]; then \
		echo "  Using config: $(MORTALITY_CONFIG)"; \
		$(PYTHON) kcor_mortality_sensitivity.py $(MORTALITY_INPUT_CSV) $(MORTALITY_OUTPUT_DIR) \
			--config-file $(MORTALITY_CONFIG); \
	else \
		echo "  Using default configuration"; \
		$(PYTHON) kcor_mortality_sensitivity.py $(MORTALITY_INPUT_CSV) $(MORTALITY_OUTPUT_DIR); \
	fi
	@echo "KCOR mortality sensitivity analysis complete! Output: $(MORTALITY_OUTPUT_DIR)/sensitivity"

# Age-stratified analysis
mortality_age: kcor_mortality_age_stratified.py | ../data/Czech2
	@echo "Running KCOR mortality age-stratified analysis..."
	@ENROLL_YEAR_VAL=$${ENROLL_YEAR:-2021}; \
	ENROLL_MONTH_VAL=$${ENROLL_MONTH:-7}; \
	MAX_FU_VAL=$${MAX_FU_MONTHS:-24}; \
	QUIET_MIN_VAL=$${QUIET_MIN:-3}; \
	QUIET_MAX_VAL=$${QUIET_MAX:-10}; \
	echo "  Enrollment: $$ENROLL_YEAR_VAL-$$ENROLL_MONTH_VAL"; \
	$(PYTHON) kcor_mortality_age_stratified.py $(MORTALITY_INPUT_CSV) $(MORTALITY_OUTPUT_DIR) \
		--enroll-year $$ENROLL_YEAR_VAL \
		--enroll-month $$ENROLL_MONTH_VAL \
		--max-fu-months $$MAX_FU_VAL \
		--quiet-min $$QUIET_MIN_VAL \
		--quiet-max $$QUIET_MAX_VAL
	@echo "KCOR mortality age-stratified analysis complete! Output: $(MORTALITY_OUTPUT_DIR)/age_stratified"

# Statistical inference (adds CIs and p-values)
mortality_stats: kcor_mortality_stats.py | ../data/Czech2
	@echo "Adding statistical inference to KCOR mortality results..."
	@if [ -f "$(MORTALITY_OUTPUT_DIR)/raw/kcor_hazard_adjusted.csv" ]; then \
		$(PYTHON) kcor_mortality_stats.py \
			$(MORTALITY_OUTPUT_DIR)/raw/kcor_hazard_adjusted.csv \
			$(MORTALITY_OUTPUT_DIR)/results/kcor_ratios_with_ci.csv \
			--alpha 0.05; \
		echo "Statistical inference complete! Output: $(MORTALITY_OUTPUT_DIR)/results/kcor_ratios_with_ci.csv"; \
	else \
		echo "ERROR: Run 'make mortality' first to generate hazard data"; \
		exit 1; \
	fi

# Enhanced visualizations
mortality_plots: kcor_mortality_plots.py | ../data/Czech2
	@echo "Creating enhanced visualizations..."
	@QUIET_MIN_VAL=$${QUIET_MIN:-3}; \
	QUIET_MAX_VAL=$${QUIET_MAX:-10}; \
	if [ -f "$(MORTALITY_OUTPUT_DIR)/raw/kcor_hazard_adjusted.csv" ]; then \
		$(PYTHON) kcor_mortality_plots.py \
			$(MORTALITY_OUTPUT_DIR)/raw/kcor_hazard_adjusted.csv \
			$(MORTALITY_OUTPUT_DIR)/plots \
			--slopes-csv $(MORTALITY_OUTPUT_DIR)/raw/kcor_slopes.csv \
			--kcor-csv $(MORTALITY_OUTPUT_DIR)/results/kcor_ratios.csv \
			--quiet-min $$QUIET_MIN_VAL \
			--quiet-max $$QUIET_MAX_VAL \
			--plot-type all; \
		echo "Enhanced visualizations complete! Output: $(MORTALITY_OUTPUT_DIR)/plots"; \
	else \
		echo "ERROR: Run 'make mortality' first to generate hazard data"; \
		exit 1; \
	fi

# Run all mortality analyses (Czech2 dataset)
# This runs: basic analysis, sensitivity, age-stratified, stats, and plots
mortality_all: kcor_mortality.py kcor_mortality_sensitivity.py kcor_mortality_age_stratified.py kcor_mortality_stats.py kcor_mortality_plots.py | ../data/Czech2
	@echo "=========================================="
	@echo "Running Complete KCOR Mortality Analysis"
	@echo "=========================================="
	@echo ""
	@echo "Step 1/5: Basic mortality analysis..."
	@if [ -n "$(ENROLL_YEAR)" ]; then \
		$(MAKE) mortality ENROLL_YEAR=$(ENROLL_YEAR) ENROLL_MONTH=$(ENROLL_MONTH) MAX_FU_MONTHS=$(MAX_FU_MONTHS) QUIET_MIN=$(QUIET_MIN) QUIET_MAX=$(QUIET_MAX); \
	else \
		$(MAKE) mortality; \
	fi
	@echo ""
	@echo "Step 2/5: Sensitivity analysis..."
	@$(MAKE) mortality_sensitivity
	@echo ""
	@echo "Step 3/5: Age-stratified analysis..."
	@if [ -n "$(ENROLL_YEAR)" ]; then \
		$(MAKE) mortality_age ENROLL_YEAR=$(ENROLL_YEAR) ENROLL_MONTH=$(ENROLL_MONTH) MAX_FU_MONTHS=$(MAX_FU_MONTHS) QUIET_MIN=$(QUIET_MIN) QUIET_MAX=$(QUIET_MAX); \
	else \
		$(MAKE) mortality_age; \
	fi
	@echo ""
	@echo "Step 4/5: Statistical inference..."
	@$(MAKE) mortality_stats || echo "Warning: Statistical inference failed (may need basic analysis output)"
	@echo ""
	@echo "Step 5/5: Enhanced visualizations..."
	@if [ -n "$(QUIET_MIN)" ]; then \
		$(MAKE) mortality_plots QUIET_MIN=$(QUIET_MIN) QUIET_MAX=$(QUIET_MAX) || echo "Warning: Enhanced visualizations failed (may need basic analysis output)"; \
	else \
		$(MAKE) mortality_plots || echo "Warning: Enhanced visualizations failed (may need basic analysis output)"; \
	fi
	@echo ""
	@echo "=========================================="
	@echo "Complete KCOR Mortality Analysis Finished"
	@echo "=========================================="
	@echo "Output directory: $(MORTALITY_OUTPUT_DIR)"

# Ensure Czech2 data directory exists
../data/Czech2:
	@$(PYTHON) -c "import os; os.makedirs('$@', exist_ok=True)"
