# Sensitivity analysis Makefile
# Run from the root Makefile as make test
# 
# Sensitivity analysis computes KCOR for age=-2 cohort (all ages aggregated)
# varying:
# - SA_BASELINE_WEEKS (baseline normalization week count, e.g. 2..8)
# - SA_QUIET_START_OFFSETS (quiet-window start offsets in weeks, relative to 2022-24)
# Quiet-window end remains fixed at 2024-16.
# Outputs grid tables (offsets as columns, baseline weeks as rows) per cohort.
#
# Quick test overrides:
# - SA_BASELINE_WEEKS: e.g. "4" or "2,8,1" (start,end,step)
# - SA_QUIET_START_OFFSETS: e.g. "-12,-8,-4,0,4,8,12" or "-12,12,4"

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
SA_BASELINE_WEEKS ?=
SA_QUIET_START_OFFSETS ?=

.PHONY: all clean

all: $(OUT_XLSX)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Build sensitivity analysis only if output is missing or stale
$(OUT_XLSX): $(CMR) $(ROOT)/code/KCOR.py | $(OUT_DIR)
	SENSITIVITY_ANALYSIS=1 \
	SA_COHORTS="$(SA_COHORTS)" SA_DOSE_PAIRS="$(SA_DOSE_PAIRS)" \
	SA_BASELINE_WEEKS="$(SA_BASELINE_WEEKS)" SA_QUIET_START_OFFSETS="$(SA_QUIET_START_OFFSETS)" \
	$(PY) $(ROOT)/code/KCOR.py $(CMR) $(OUT_XLSX) "Sensitivity Analysis" KCOR_summary_SA.log

clean:
	rm -f $(OUT_DIR)/KCOR_SA.xlsx
	rm -f $(OUT_DIR)/KCOR_summary_SA.log
	rm -f $(OUT_DIR)/KCOR_summary.xlsx


