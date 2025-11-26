# Root Makefile for KCOR

# Suppress Entering/Leaving directory noise from sub-makes
MAKEFLAGS += --no-print-directory

CODE_DIR := code
VALIDATION_DSCMRR_DIR := validation/DS-CMRR
VALIDATION_KM_DIR := validation/kaplan_meier
VALIDATION_GLM_DIR := validation/GLM
VALIDATION_HVE_DIR := validation/HVE
VALIDATION_ASMR_DIR := validation/ASMR_analysis

.PHONY: all KCOR CMR CMR_from_krf convert validation test clean sensitivity KCOR_variable HVE ASMR ts icd10 icd_population_shift help

# Dataset namespace (override on CLI: make DATASET=USA)
DATASET ?= Czech

# Default: build everything (variable-cohort + analysis + validation + tests)
all: KCOR_variable KCOR validation test

# KCOR analysis pipeline (delegates to code/Makefile target KCOR)
KCOR:
	$(MAKE) -C $(CODE_DIR) KCOR DATASET=$(DATASET)

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
icd10:
	@echo "Running ICD-10 cause of death analysis..."
	cd $(CODE_DIR) && python3 icd_analysis.py ../data/Czech2/data.csv ../data/Czech2/
	@echo "ICD-10 analysis complete!"

# ICD-10 population structural shift analysis (Czech2 dataset)
icd_population_shift:
	@echo "Running ICD-10 population structural shift analysis..."
	cd $(CODE_DIR) && python3 icd_population_shift.py ../data/Czech2/data.csv ../data/Czech2/
	@echo "ICD-10 population shift analysis complete!"

# Help target
help:
	@echo "Available targets:"
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
	@echo "  clean           - Clean outputs"
	@echo ""
	@echo "Variables:"
	@echo "  DATASET=<name>        - Dataset namespace (default: Czech)"

