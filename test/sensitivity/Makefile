# Sensitivity analysis Makefile
# Run from the root Makefile as make test
# Run with: make run SA_SLOPE_START=53,53,1 SA_SLOPE_LENGTH=21,61,5 SA_YOB=0 SA_COHORTS=2021_24 SA_DOSE_PAIRS=1,0;2,0 SA_ANCHOR_WEEKS=4 SA_MA_TOTAL_LENGTH=8 SA_CENTERED=1 SA_FINAL_KCOR_MIN=0,1,1 SA_FINAL_KCOR_DATE=4/1/24

# Suppress Entering/Leaving directory noise from sub-makes
MAKEFLAGS += --no-print-directory

ROOT := ../..
PY := python3
DATASET ?= Czech

CMR := $(ROOT)/data/$(DATASET)/KCOR_CMR.xlsx
OUT_DIR := $(ROOT)/test/sensitivity/out
OUT_XLSX := $(OUT_DIR)/KCOR_SA.xlsx

# Defaults for sensitivity analysis (overridable from CLI)
SA_SLOPE_START ?= 53,53,1
SA_SLOPE_LENGTH ?= 21,61,10
SA_YOB ?= 0
SA_COHORTS ?= 2021_24
SA_DOSE_PAIRS ?= 1,0;2,0
SA_ANCHOR_WEEKS ?= 4
SA_MA_TOTAL_LENGTH ?= 8
SA_CENTERED ?= 1
## SLOPE_WINDOW_SIZE obsolete in slope2
SA_FINAL_KCOR_MIN ?= 0,1,1
SA_FINAL_KCOR_DATE ?= 4/1/24

.PHONY: all clean

all: $(OUT_XLSX)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Build sensitivity analysis only if output is missing or stale
$(OUT_XLSX): $(CMR) | $(OUT_DIR)
	SENSITIVITY_ANALYSIS=1 \
	SA_SLOPE_START="$(SA_SLOPE_START)" SA_SLOPE_LENGTH="$(SA_SLOPE_LENGTH)" \
	SA_YOB="$(SA_YOB)" SA_COHORTS="$(SA_COHORTS)" SA_DOSE_PAIRS="$(SA_DOSE_PAIRS)" \
	SA_ANCHOR_WEEKS="$(SA_ANCHOR_WEEKS)" SA_MA_TOTAL_LENGTH="$(SA_MA_TOTAL_LENGTH)" \
	SA_CENTERED="$(SA_CENTERED)" \
	SA_FINAL_KCOR_MIN="$(SA_FINAL_KCOR_MIN)" SA_FINAL_KCOR_DATE="$(SA_FINAL_KCOR_DATE)" \
	$(PY) $(ROOT)/code/KCOR.py $(CMR) $(OUT_XLSX) "Sensitivity Analysis" KCOR_summary_SA.log

clean:
	rm -f $(OUT_DIR)/KCOR_SA.xlsx
	rm -f $(OUT_DIR)/KCOR_summary_SA.log
	rm -f $(OUT_DIR)/KCOR_summary.xlsx


