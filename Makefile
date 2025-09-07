# Root Makefile for KCOR

# Suppress Entering/Leaving directory noise from sub-makes
MAKEFLAGS += --no-print-directory

CODE_DIR := code
VALIDATION_DSCMRR_DIR := validation/DS-CMRR
VALIDATION_KM_DIR := validation/kaplan_meier

.PHONY: all run validation test clean sensitivity

# Default: build everything (analysis + validation + tests)
all: run validation test

# KCOR analysis pipeline (delegates to code/Makefile target KCOR)
run:
	$(MAKE) -C $(CODE_DIR) KCOR

# Validation suite (DS-CMRR and Kaplan–Meier)
validation:
	$(MAKE) -C $(VALIDATION_DSCMRR_DIR) run
	$(MAKE) -C $(VALIDATION_KM_DIR) run

# Convenience target to run only Kaplan–Meier
km:
	$(MAKE) -C $(VALIDATION_KM_DIR) run

# Negative-control test (delegates to test/Makefile)
test:
	$(MAKE) -C test all

clean:
	-$(MAKE) -C $(CODE_DIR) clean
	-$(MAKE) -C $(VALIDATION_DSCMRR_DIR) clean
	-$(MAKE) -C $(VALIDATION_KM_DIR) clean


sensitivity:
	$(MAKE) -C test/sensitivity all

