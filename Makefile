# Root Makefile for KCOR

# Suppress Entering/Leaving directory noise from sub-makes
MAKEFLAGS += --no-print-directory

CODE_DIR := code
VALIDATION_DSCMRR_DIR := validation/DS-CMRR
VALIDATION_KM_DIR := validation/kaplan_meier
VALIDATION_GLM_DIR := validation/GLM
VALIDATION_HVE_DIR := validation/HVE
VALIDATION_ASMR_DIR := validation/ASMR_analysis

.PHONY: all KCOR CMR CMR_from_krf convert validation test clean sensitivity KCOR_variable HVE ASMR ts icd10 icd_population_shift mortality mortality_sensitivity mortality_age mortality_stats mortality_plots mortality_all install install-debian help

# Dataset namespace (override on CLI: make DATASET=USA)
DATASET ?= Czech

# Virtual environment path
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python3
VENV_PIP := $(VENV_DIR)/bin/pip3

# Install dependencies using pip in virtual environment
install: $(VENV_DIR)
	@echo "Installing dependencies from requirements.txt in virtual environment..."
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -r requirements.txt
	@echo "Installation complete!"

# Create virtual environment if it doesn't exist
$(VENV_DIR):
	@echo "Creating virtual environment in $(VENV_DIR)..."
	python3 -m venv $(VENV_DIR)
	@echo "Virtual environment created!"

# Install dependencies using Debian packages (alternative to pip)
install-debian:
	@echo "Installing dependencies using Debian packages..."
	@echo "This requires sudo privileges."
	sudo apt-get update
	sudo apt-get install -y \
		python3-pandas \
		python3-numpy \
		python3-openpyxl \
		python3-statsmodels \
		python3-scipy \
		python3-matplotlib \
		python3-seaborn
	@echo "Debian package installation complete!"
	@echo "Note: Package versions may differ from requirements.txt. For exact versions, use 'make install' with pip."

# Default: build everything (variable-cohort + analysis + validation + tests)
all: KCOR_variable KCOR validation test

# KCOR analysis pipeline (delegates to code/Makefile target KCOR)
KCOR: $(VENV_DIR)
	@echo "Checking for cvxpy in virtual environment..."
	@$(VENV_PYTHON) -c "import cvxpy" 2>/dev/null || (echo "ERROR: cvxpy not found in virtual environment. Run 'make install' to install dependencies." && exit 1)
	@echo "cvxpy is available."
	$(MAKE) -C $(CODE_DIR) KCOR DATASET=$(DATASET) PYTHON=$(abspath $(VENV_PYTHON))

# CMR aggregation only (delegates to code/Makefile target CMR)
CMR:
	$(MAKE) -C $(CODE_DIR) CMR DATASET=$(DATASET)

# Run CMR on KRF input by adapting to Czech-like format first
CMR_from_krf:
	$(MAKE) -C $(CODE_DIR) CMR_from_krf DATASET=$(DATASET)

# Dataset converter (delegates to data/<DATASET>/Makefile)
convert:
	$(MAKE) -C data/$(DATASET) convert

# Variable-cohort aggregation (delegates to code/Makefile target KCOR_variable)
KCOR_variable:
	$(MAKE) -C $(CODE_DIR) KCOR_variable DATASET=$(DATASET)

# Time series aggregation (delegates to code/Makefile target ts)
ts:
	$(MAKE) -C $(CODE_DIR) ts DATASET=$(DATASET)

# Validation suite (DS-CMRR, Kaplan–Meier, GLM)
validation:
	$(MAKE) -C $(VALIDATION_DSCMRR_DIR) run DATASET=$(DATASET)
	$(MAKE) -C $(VALIDATION_KM_DIR) run DATASET=$(DATASET)
	$(MAKE) -C $(VALIDATION_GLM_DIR) run DATASET=$(DATASET)

# Convenience target to run only Kaplan–Meier
km:
	$(MAKE) -C $(VALIDATION_KM_DIR) run DATASET=$(DATASET)

# Convenience target to run only GLM
glm:
	$(MAKE) -C $(VALIDATION_GLM_DIR) run DATASET=$(DATASET)

glm-compare:
	$(MAKE) -C $(VALIDATION_GLM_DIR) compare DATASET=$(DATASET)

# Negative-control test (delegates to test/Makefile)
test:
	$(MAKE) -C test all

clean:
	-$(MAKE) -C $(CODE_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_DSCMRR_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_KM_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_GLM_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_HVE_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_ASMR_DIR) clean DATASET=$(DATASET)


sensitivity:
	$(MAKE) -C test/sensitivity all DATASET=$(DATASET)

# HVE simulator (not part of default all)
HVE:
	$(MAKE) -C $(VALIDATION_HVE_DIR) run DATASET=$(DATASET)

# ASMR (fixed-cohort) analysis from KCOR_CMR.xlsx (not part of default all)
ASMR:
	$(MAKE) -C $(VALIDATION_ASMR_DIR) run DATASET=$(DATASET)

# ICD-10 cause of death analysis (Czech2 dataset)
icd10: $(VENV_DIR)
	@echo "Running ICD-10 cause of death analysis..."
	cd $(CODE_DIR) && $(abspath $(VENV_PYTHON)) icd_analysis.py ../data/Czech2/data.csv ../data/Czech2/
	@echo "ICD-10 analysis complete!"

# ICD-10 population structural shift analysis (Czech2 dataset)
icd_population_shift: $(VENV_DIR)
	@echo "Running ICD-10 population structural shift analysis..."
	cd $(CODE_DIR) && $(abspath $(VENV_PYTHON)) icd_population_shift.py ../data/Czech2/data.csv ../data/Czech2/
	@echo "ICD-10 population shift analysis complete!"

# KCOR Mortality Analysis Pipeline (Czech2 dataset)
mortality:
	$(MAKE) -C $(CODE_DIR) mortality ENROLL_YEAR=$(ENROLL_YEAR) ENROLL_MONTH=$(ENROLL_MONTH) MAX_FU_MONTHS=$(MAX_FU_MONTHS) QUIET_MIN=$(QUIET_MIN) QUIET_MAX=$(QUIET_MAX)

mortality_sensitivity:
	$(MAKE) -C $(CODE_DIR) mortality_sensitivity

mortality_age:
	$(MAKE) -C $(CODE_DIR) mortality_age ENROLL_YEAR=$(ENROLL_YEAR) ENROLL_MONTH=$(ENROLL_MONTH) MAX_FU_MONTHS=$(MAX_FU_MONTHS) QUIET_MIN=$(QUIET_MIN) QUIET_MAX=$(QUIET_MAX)

mortality_stats:
	$(MAKE) -C $(CODE_DIR) mortality_stats

mortality_plots:
	$(MAKE) -C $(CODE_DIR) mortality_plots QUIET_MIN=$(QUIET_MIN) QUIET_MAX=$(QUIET_MAX)

# Run all mortality analyses (Czech2 dataset)
mortality_all:
	$(MAKE) -C $(CODE_DIR) mortality_all ENROLL_YEAR=$(ENROLL_YEAR) ENROLL_MONTH=$(ENROLL_MONTH) MAX_FU_MONTHS=$(MAX_FU_MONTHS) QUIET_MIN=$(QUIET_MIN) QUIET_MAX=$(QUIET_MAX)

# Help target
help:
	@echo "Available targets:"
	@echo "  install         - Create .venv virtual environment and install dependencies from requirements.txt"
	@echo "  KCOR_variable   - Build variable-cohort aggregation (code/)"
	@echo "  ts              - Build time series aggregation (code/)"
	@echo "  KCOR            - Run main KCOR pipeline (code/)"
	@echo "  CMR             - Run only CMR aggregation step (code/)"
	@echo "  CMR_from_krf    - Adapt KRF CSV to Czech-like and run CMR (code/)"
	@echo "  convert         - Run dataset converter (data/$(DATASET)/)"
	@echo "  validation      - Run DS-CMRR, Kaplan–Meier, and GLM validation"
	@echo "  km              - Run only Kaplan–Meier validation"
	@echo "  glm             - Run only GLM validation"
	@echo "  glm-compare     - Compare GLM outputs"
	@echo "  test            - Run negative-control and sensitivity tests (test/)"
	@echo "  sensitivity     - Run parameter sweep (test/sensitivity)"
	@echo "  HVE             - Run Healthy Vaccinee Effect simulation (validation/HVE)"
	@echo "  ASMR            - Run ASMR analysis from KCOR_CMR.xlsx (validation/ASMR_analysis)"
	@echo "  icd10           - Run ICD-10 cause of death analysis (data/Czech2/)"
	@echo "  icd_population_shift - Run ICD-10 population structural shift analysis (data/Czech2/)"
	@echo ""
	@echo "KCOR Mortality Analysis (Czech2 dataset):"
	@echo "  mortality       - Run basic KCOR mortality analysis pipeline"
	@echo "  mortality_sensitivity - Run sensitivity analysis (multiple configurations)"
	@echo "  mortality_age   - Run age-stratified analysis"
	@echo "  mortality_stats - Add statistical inference (CIs, p-values) to results"
	@echo "  mortality_plots - Create enhanced visualizations"
	@echo "  mortality_all   - Run ALL mortality analyses (recommended for complete analysis)"
	@echo ""
	@echo "  clean           - Clean outputs"
	@echo ""
	@echo "Variables:"
	@echo "  DATASET=<name>        - Dataset namespace (default: Czech)"
	@echo "  ENROLL_YEAR=<year>    - Enrollment year for mortality analysis (default: 2021)"
	@echo "  ENROLL_MONTH=<month>  - Enrollment month 1-12 for mortality analysis (default: 7)"
	@echo "  MAX_FU_MONTHS=<n>     - Maximum follow-up months (default: 24)"
	@echo "  QUIET_MIN=<month>    - Quiet period start month (default: 3)"
	@echo "  QUIET_MAX=<month>     - Quiet period end month (default: 10)"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Create .venv virtual environment and install dependencies from requirements.txt"
	@echo "  make install-debian - Install dependencies using Debian packages (requires sudo)"
	@echo ""
	@echo "Note: KCOR v5.0+ requires cvxpy which is best installed in a virtual environment."
	@echo "      The 'make install' target automatically creates and uses .venv."
	@echo "      All Python commands run through the Makefile use the virtual environment automatically."

