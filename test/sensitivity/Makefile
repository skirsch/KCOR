# Sensitivity analysis Makefile
# Run from the root Makefile as make test
# 
# Sensitivity analysis computes KCOR for age=-2 cohort (all ages aggregated) 
# varying SLOPE8_QUANTILE_TAU (0.1, 0.2, 0.3, 0.4, 0.5) and 
# KCOR_NORMALIZATION_WEEKS (2, 3, 4, 5, 6, 7, 8).
# Outputs grid tables (tau as columns, normalization weeks as rows) per cohort.
#
# Quick test overrides:
# - SA_TAU_VALUES: e.g. "0.5" or "0.1,0.5,0.1" (start,end,step)
# - SA_NORM_WEEKS: e.g. "4" or "2,8,1" (start,end,step)

# Suppress Entering/Leaving directory noise from sub-makes
MAKEFLAGS += --no-print-directory

ROOT := ../..
PY := python3
DATASET ?= Czech

CMR := $(ROOT)/data/$(DATASET)/KCOR_CMR.xlsx
OUT_DIR := $(ROOT)/test/sensitivity/out
OUT_XLSX := $(OUT_DIR)/KCOR_SA.xlsx

# Optional overrides (if not provided, processes all cohorts and all dose pairs)
SA_COHORTS ?= 
SA_DOSE_PAIRS ?= 
SA_TAU_VALUES ?=
SA_NORM_WEEKS ?=

.PHONY: all clean

all: $(OUT_XLSX)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Build sensitivity analysis only if output is missing or stale
$(OUT_XLSX): $(CMR) $(ROOT)/code/KCOR.py | $(OUT_DIR)
	SENSITIVITY_ANALYSIS=1 \
	SA_COHORTS="$(SA_COHORTS)" SA_DOSE_PAIRS="$(SA_DOSE_PAIRS)" \
	SA_TAU_VALUES="$(SA_TAU_VALUES)" SA_NORM_WEEKS="$(SA_NORM_WEEKS)" \
	$(PY) $(ROOT)/code/KCOR.py $(CMR) $(OUT_XLSX) "Sensitivity Analysis" KCOR_summary_SA.log

clean:
	rm -f $(OUT_DIR)/KCOR_SA.xlsx
	rm -f $(OUT_DIR)/KCOR_summary_SA.log
	rm -f $(OUT_DIR)/KCOR_summary.xlsx


