# Root Makefile for KCOR

# Suppress Entering/Leaving directory noise from sub-makes
MAKEFLAGS += --no-print-directory

CODE_DIR := code
VALIDATION_DSCMRR_DIR := validation/DS-CMRR
VALIDATION_KM_DIR := validation/kaplan_meier
VALIDATION_GLM_DIR := validation/GLM
VALIDATION_HVE_DIR := validation/HVE
VALIDATION_ASMR_DIR := validation/ASMR_analysis

# Paper build (Pandoc)
PAPER_DIR ?= documentation/preprint
PAPER_MD ?= paper.md
PAPER_DOCX ?= paper.docx
PAPER_PDF ?= paper.pdf
PAPER_BIB ?= refs.bib
PAPER_CSL ?= american-medical-association.csl
PAPER_REFERENCE_DOC ?= reference.docx
# PDF engine (override on CLI if needed). Default: xelatex.
PAPER_PDF_ENGINE ?= xelatex
PAPER_PDF_GEOMETRY ?= margin=1in
PAPER_PDF_MAINFONT ?= TeX Gyre Termes
PAPER_PDF_MATHFONT ?= TeX Gyre Termes Math

.PHONY: all KCOR CMR CMR_from_krf monte_carlo convert validation test clean sensitivity KCOR_variable HVE ASMR ts icd10 icd_population_shift mortality mortality_sensitivity mortality_age mortality_stats mortality_plots mortality_all install install-debian slope-test paper paper-docx paper-pdf sim_grid cox-bias cox-bias-figures copy-cox-bias-figures skip-weeks cohort-size rollout help

# Dataset namespace (override on CLI: make DATASET=USA)
DATASET ?= Czech

# Monte Carlo iterations (override on CLI: make monte_carlo MC_ITERATIONS=50)
MC_ITERATIONS ?= 4
# Monte Carlo enrollment cohort (ISO week label; accepts YYYY_WW or YYYY-WW)
MC_ENROLLMENT_DATE ?= 2021_24

# Virtual environment path
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python3
VENV_PIP := $(VENV_DIR)/bin/pip

# Install dependencies using pip in virtual environment
install: $(VENV_PIP)
	@echo "Installing dependencies from requirements.txt in virtual environment..."
	@echo "Upgrading pip..."
	@$(VENV_PIP) install --upgrade pip
	@echo "Installing requirements..."
	@$(VENV_PIP) install -r requirements.txt
	@echo "Installation complete!"

# Ensure pip exists in virtual environment
$(VENV_PIP): $(VENV_PYTHON)
	@if [ ! -f "$(VENV_PIP)" ]; then \
		echo "Installing pip in virtual environment..."; \
		$(VENV_PYTHON) -m ensurepip --upgrade; \
	fi

# Create virtual environment directory and Python executable
$(VENV_PYTHON):
	@if [ ! -d "$(VENV_DIR)" ] || [ ! -f "$(VENV_PYTHON)" ]; then \
		echo "Creating virtual environment in $(VENV_DIR)..."; \
		python3 -m venv $(VENV_DIR); \
		echo "Virtual environment created!"; \
	fi

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
KCOR: $(VENV_PYTHON)
	$(MAKE) -C $(CODE_DIR) KCOR DATASET=$(DATASET) PYTHON=$(abspath $(VENV_PYTHON))

# CMR aggregation only (delegates to code/Makefile target CMR)
CMR:
	$(MAKE) -C $(CODE_DIR) CMR DATASET=$(DATASET)

# Monte Carlo mode (delegates to code/Makefile target monte_carlo)
monte_carlo:
	$(MAKE) -C $(CODE_DIR) monte_carlo DATASET=$(DATASET) MC_ITERATIONS=$(MC_ITERATIONS) MC_ENROLLMENT_DATE=$(MC_ENROLLMENT_DATE)

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

# Slope normalization test
slope-test: $(VENV_PYTHON)
	@echo "Running slope normalization test..."
	cd test/slope_normalization && $(abspath $(VENV_PYTHON)) test.py
	@echo "Slope normalization test complete!"

# Simulation grid (operating characteristics and failure-mode diagnostics)
sim_grid: $(VENV_PYTHON)
	$(MAKE) -C test/sim_grid all PYTHON=$(abspath $(VENV_PYTHON))

# Cox bias demonstration (Cox regression bias under frailty heterogeneity)
cox-bias: $(VENV_PYTHON)
	$(MAKE) -C test/sim_grid cox-bias PYTHON=$(abspath $(VENV_PYTHON))

cox-bias-figures: $(VENV_PYTHON)
	$(MAKE) -C test/sim_grid cox-bias-figures PYTHON=$(abspath $(VENV_PYTHON))

copy-cox-bias-figures: $(VENV_PYTHON)
	$(MAKE) -C test/sim_grid copy-cox-bias-figures PYTHON=$(abspath $(VENV_PYTHON))

# Skip-weeks sensitivity figure
skip-weeks: $(VENV_PYTHON)
	$(MAKE) -C test/sim_grid skip-weeks PYTHON=$(abspath $(VENV_PYTHON))

# Cohort size sensitivity analysis
cohort-size: $(VENV_PYTHON)
	$(MAKE) -C test/sim_grid cohort-size PYTHON=$(abspath $(VENV_PYTHON))

# Rollout design-mimic simulation
rollout: $(VENV_PYTHON)
	$(MAKE) -C test/sim_grid rollout PYTHON=$(abspath $(VENV_PYTHON))

clean:
	-$(MAKE) -C $(CODE_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_DSCMRR_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_KM_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_GLM_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_HVE_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_ASMR_DIR) clean DATASET=$(DATASET)


sensitivity:
	$(MAKE) -C test/sensitivity all DATASET=$(DATASET)

# Build methods paper (Pandoc → Word)
#
# Default inputs live in documentation/preprint/:
# - paper.md (combined main paper + supplement, Pandoc-crossref markup)
# - refs.bib
# - american-medical-association.csl
#
# Outputs:
# - paper.docx (single Word document)
# - paper.pdf (PDF)
#
# Usage:
#   make paper
#   make paper PAPER_MD=paper_v7.md PAPER_DOCX=paper_v7.docx
paper: $(PAPER_DIR)/$(PAPER_DOCX) $(PAPER_DIR)/$(PAPER_PDF)
	@echo "Copying paper files to website..."
	@scp $(PAPER_DIR)/$(PAPER_DOCX) truenas:/mnt/main/www/skirsch.com/covid/KCOR
	@scp $(PAPER_DIR)/$(PAPER_PDF) truenas:/mnt/main/www/skirsch.com/covid/KCOR/KCOR.pdf
	@echo "Paper files copied to website."

paper-docx: $(PAPER_DIR)/$(PAPER_DOCX)
paper-pdf: $(PAPER_DIR)/$(PAPER_PDF)

# Build paper Word file (single document)
$(PAPER_DIR)/$(PAPER_DOCX): $(PAPER_DIR)/$(PAPER_MD) $(PAPER_DIR)/$(PAPER_BIB) $(PAPER_DIR)/$(PAPER_CSL) $(PAPER_DIR)/$(PAPER_REFERENCE_DOC) $(wildcard $(PAPER_DIR)/figures/*)
	@echo "Building paper: $(PAPER_DIR)/$(PAPER_MD) -> $(PAPER_DIR)/$(PAPER_DOCX)"
	@cd $(PAPER_DIR) && \
		pandoc $(PAPER_MD) \
			--to=docx \
			--filter pandoc-crossref \
			--lua-filter pagebreak-tables.lua \
			--metadata-file pandoc-crossref-docx.yaml \
			--citeproc \
			--bibliography=$(PAPER_BIB) \
			--csl=$(PAPER_CSL) \
			-o $(PAPER_DOCX).new
	@cd $(PAPER_DIR) && \
		mv -f $(PAPER_DOCX).new $(PAPER_DOCX) 2>/dev/null || ( \
			echo "ERROR: could not overwrite $(PAPER_DIR)/$(PAPER_DOCX) (is it open in Word?)."; \
			echo "Leaving: $(PAPER_DIR)/$(PAPER_DOCX).new"; \
			exit 1; \
		)

# Build PDF
$(PAPER_DIR)/$(PAPER_PDF): $(PAPER_DIR)/$(PAPER_MD) $(PAPER_DIR)/$(PAPER_BIB) $(PAPER_DIR)/$(PAPER_CSL) $(PAPER_DIR)/header.tex $(wildcard $(PAPER_DIR)/figures/*)
	@echo "Building combined PDF: $(PAPER_DIR)/$(PAPER_MD) -> $(PAPER_DIR)/$(PAPER_PDF)"
	@cd $(PAPER_DIR) && \
		if ! command -v "$(PAPER_PDF_ENGINE)" >/dev/null 2>&1; then \
			echo "ERROR: PDF engine '$(PAPER_PDF_ENGINE)' not found. Install it (default expects xelatex) or override PAPER_PDF_ENGINE=<engine>."; \
			exit 1; \
		fi; \
		pandoc $(PAPER_MD) \
			--to=pdf \
			--pdf-engine="$(PAPER_PDF_ENGINE)" \
			--filter pandoc-crossref \
			--lua-filter pagebreak-tables.lua \
			--metadata-file pandoc-crossref.yaml \
			--citeproc \
			-V geometry:$(PAPER_PDF_GEOMETRY) \
			-V mainfont="$(PAPER_PDF_MAINFONT)" \
			-V mathfont="$(PAPER_PDF_MATHFONT)" \
			-H header.tex \
			--bibliography=$(PAPER_BIB) \
			--csl=$(PAPER_CSL) \
			-o $(PAPER_PDF).new
	@cd $(PAPER_DIR) && \
		mv -f $(PAPER_PDF).new $(PAPER_PDF) 2>/dev/null || ( \
			echo "ERROR: could not overwrite $(PAPER_DIR)/$(PAPER_PDF) (is it open?)."; \
			echo "Leaving: $(PAPER_DIR)/$(PAPER_PDF).new"; \
			exit 1; \
		)

# Create a default Pandoc reference.docx if missing (customize in Word as needed)
$(PAPER_DIR)/$(PAPER_REFERENCE_DOC):
	@echo "Creating default Pandoc reference doc: $(PAPER_DIR)/$(PAPER_REFERENCE_DOC)"
	@cd $(PAPER_DIR) && \
		pandoc --print-default-data-file reference.docx > $(PAPER_REFERENCE_DOC).new && \
		mv -f $(PAPER_REFERENCE_DOC).new $(PAPER_REFERENCE_DOC)

# HVE simulator (not part of default all)
HVE:
	$(MAKE) -C $(VALIDATION_HVE_DIR) run DATASET=$(DATASET)

# ASMR (fixed-cohort) analysis from KCOR_CMR.xlsx (not part of default all)
ASMR:
	$(MAKE) -C $(VALIDATION_ASMR_DIR) run DATASET=$(DATASET)

# ICD-10 cause of death analysis (Czech2 dataset)
icd10: $(VENV_PYTHON)
	@echo "Running ICD-10 cause of death analysis..."
	cd $(CODE_DIR) && $(abspath $(VENV_PYTHON)) icd_analysis.py ../data/Czech2/data.csv ../data/Czech2/
	@echo "ICD-10 analysis complete!"

# ICD-10 population structural shift analysis (Czech2 dataset)
icd_population_shift: $(VENV_PYTHON)
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
	@echo "  monte_carlo     - Run Monte Carlo bootstrap sampling (4 iterations by default, override with MC_ITERATIONS=N)"
	@echo "  convert         - Run dataset converter (data/$(DATASET)/)"
	@echo "  validation      - Run DS-CMRR, Kaplan–Meier, and GLM validation"
	@echo "  km              - Run only Kaplan–Meier validation"
	@echo "  glm             - Run only GLM validation"
	@echo "  glm-compare     - Compare GLM outputs"
	@echo "  test            - Run negative-control and sensitivity tests (test/)"
	@echo "  sensitivity     - Run parameter sweep (test/sensitivity)"
	@echo "  sim_grid        - Run simulation grid for operating characteristics (test/sim_grid)"
	@echo "  cox-bias        - Run Cox bias demonstration simulation (test/sim_grid)"
	@echo "  cox-bias-figures - Generate Cox bias demonstration figures"
	@echo "  copy-cox-bias-figures - Copy Cox bias figures to preprint directory"
	@echo "  skip-weeks      - Generate skip-weeks sensitivity figure"
	@echo "  cohort-size     - Run cohort size sensitivity analysis"
	@echo "  rollout         - Run rollout design-mimic simulation"
	@echo ""
	@echo "Sensitivity quick-test examples (optional overrides):"
	@echo "  make sensitivity DATASET=Czech SA_COHORTS=2022_06 SA_DOSE_PAIRS=1,0 SA_BASELINE_WEEKS=4 SA_QUIET_START_OFFSETS=0"
	@echo "  make sensitivity DATASET=Czech SA_BASELINE_WEEKS=2,8,1 SA_QUIET_START_OFFSETS=-12,12,4"
	@echo "  slope-test      - Run slope normalization test on booster_d0_slope.csv"
	@echo "  HVE             - Run Healthy Vaccinee Effect simulation (validation/HVE)"
	@echo "  ASMR            - Run ASMR analysis from KCOR_CMR.xlsx (validation/ASMR_analysis)"
	@echo "  icd10           - Run ICD-10 cause of death analysis (data/Czech2/)"
	@echo "  icd_population_shift - Run ICD-10 population structural shift analysis (data/Czech2/)"
	@echo "  paper           - Build documentation/preprint/paper.md -> paper.docx + paper.pdf (Pandoc + crossref + citeproc)"
	@echo "  paper-docx      - Build only documentation/preprint/paper.docx"
	@echo "  paper-pdf       - Build only documentation/preprint/paper.pdf"
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
	@echo "  MC_ITERATIONS=<n>     - Number of Monte Carlo iterations (default: 4)"
	@echo "  MC_ENROLLMENT_DATE=<YYYY_WW> - (Monte Carlo) Enrollment cohort used for MC CMR + analysis (default: 2021_24)"
	@echo "  SA_COHORTS=<list>     - (Sensitivity) Restrict cohorts, e.g. 2022_06 or 2021_24,2022_06"
	@echo "  SA_DOSE_PAIRS=<pairs> - (Sensitivity) Restrict dose pairs, e.g. 1,0 or 1,0;2,0"
	@echo "  SA_BASELINE_WEEKS=<vals> - (Sensitivity) Baseline weeks list/range, e.g. 4,6,8 or 2,8,1"
	@echo "  SA_QUIET_START_OFFSETS=<vals> - (Sensitivity) Quiet-start offsets (weeks) list/range, e.g. -12,-8,-4,0,4,8,12 or -12,12,4"
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
	@echo "Note: KCOR v5.1+ uses slope7 mode and no longer requires cvxpy (it was only needed for legacy quadratic mode)."
	@echo "      The 'make install' target automatically creates and uses .venv."
	@echo "      All Python commands run through the Makefile use the virtual environment automatically."

