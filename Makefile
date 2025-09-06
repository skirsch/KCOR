# Root Makefile for KCOR

# Suppress Entering/Leaving directory noise from sub-makes
MAKEFLAGS += --no-print-directory

CODE_DIR := code
VALIDATION_DSCMRR_DIR := validation/DS-CMRR
VALIDATION_KM_DIR := validation/kaplan_meier

.PHONY: all run validation test clean

# Default: build everything (analysis + validation)
all: run validation

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

# Alias: `make test` maps to validation (keep unit tests separate if added later)
test: validation

clean:
	-$(MAKE) -C $(CODE_DIR) clean
	-$(MAKE) -C $(VALIDATION_DSCMRR_DIR) clean
	-$(MAKE) -C $(VALIDATION_KM_DIR) clean


