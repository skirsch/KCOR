# Root Makefile for KCOR

# Suppress Entering/Leaving directory noise from sub-makes
MAKEFLAGS += --no-print-directory

CODE_DIR := code
VALIDATION_DSCMRR_DIR := validation/DS-CMRR
VALIDATION_KM_DIR := validation/kaplan_meier

.PHONY: all run validation test clean sensitivity

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


## ---------------- Sensitivity analysis plumbing (Makefile-only) ----------------
# Defaults (override on command line):
#   SA_SLOPE_START  = start,end,step  for slope anchor offset1
#   SA_SLOPE_LENGTH = start,end,step  for slope window length Δt (offset2 = offset1 + Δt)
#   SA_YOB          = 0 | start,end,step | list (e.g., 1940,1950,1960)
#   SA_COHORTS      = comma-separated cohorts (e.g., 2021_24,2022_06)
# Defaults tuned to match current code settings for 2021_24:
# SLOPE_LOOKUP_TABLE['2021_24'] = (53, 114) => offset1=53, Δt=61
SA_SLOPE_START ?= 53,53,1
SA_SLOPE_LENGTH ?= 61,61,1
SA_YOB ?= 0
SA_COHORTS ?= 2021_24
SA_DOSE_PAIRS ?= 1,0;2,0

# Minimal wiring: delegate to code/Makefile sensitivity target
sensitivity:
	@echo "Running sensitivity (plumbing check) with:"
	@echo "  SA_SLOPE_START  = $(SA_SLOPE_START)"
	@echo "  SA_SLOPE_LENGTH = $(SA_SLOPE_LENGTH)"
	@echo "  SA_YOB          = $(SA_YOB)"
	@echo "  SA_COHORTS      = $(SA_COHORTS)"
	@echo "  SA_DOSE_PAIRS   = $(SA_DOSE_PAIRS)"
	$(MAKE) -C $(CODE_DIR) sensitivity SENSITIVITY_ANALYSIS=1 SA_SLOPE_START="$(SA_SLOPE_START)" SA_SLOPE_LENGTH="$(SA_SLOPE_LENGTH)" SA_YOB="$(SA_YOB)" SA_COHORTS="$(SA_COHORTS)" SA_DOSE_PAIRS="$(SA_DOSE_PAIRS)"

