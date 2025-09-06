# Root Makefile for KCOR

# Suppress Entering/Leaving directory noise from sub-makes
MAKEFLAGS += --no-print-directory

CODE_DIR := code
VALIDATION_DSCMRR_DIR := validation/DS-CMRR

.PHONY: all run validation test clean

# Default: build everything (analysis + validation)
all: run validation

# KCOR analysis pipeline (delegates to code/Makefile target KCOR)
run:
	$(MAKE) -C $(CODE_DIR) KCOR

# Validation suite (currently DS-CMRR; extend as needed)
validation:
	$(MAKE) -C $(VALIDATION_DSCMRR_DIR) run

# Alias: `make test` maps to validation (keep unit tests separate if added later)
test: validation

clean:
	-$(MAKE) -C $(CODE_DIR) clean
	-$(MAKE) -C $(VALIDATION_DSCMRR_DIR) clean


