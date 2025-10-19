# Root Makefile for KCOR

# Suppress Entering/Leaving directory noise from sub-makes
MAKEFLAGS += --no-print-directory

CODE_DIR := code
VALIDATION_DSCMRR_DIR := validation/DS-CMRR
VALIDATION_KM_DIR := validation/kaplan_meier
VALIDATION_GLM_DIR := validation/GLM
VALIDATION_HVE_DIR := validation/HVE
VALIDATION_ASMR_DIR := validation/ASMR_analysis

.PHONY: all KCOR CMR CMR_from_krf convert validation test clean sensitivity KCOR_variable HVE ASMR help

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

# Help target
help:
	@echo "Available targets:"
	@echo "  KCOR_variable   - Build variable-cohort aggregation (code/)"
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
	@echo "  clean           - Clean outputs"
	@echo ""
	@echo "Variables:"
	@echo "  DATASET=<name>        - Dataset namespace (default: Czech)"

