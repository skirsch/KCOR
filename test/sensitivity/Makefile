ROOT := ../..
PY := python3

CMR := $(ROOT)/data/Czech/KCOR_CMR.xlsx
OUT_DIR := $(ROOT)/test/sensitivity/out
OUT_XLSX := $(OUT_DIR)/KCORv4_sensitivity_smoke.xlsx

.PHONY: all run clean

all: run

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# Run sensitivity analysis locally; outputs live under test/sensitivity/out
run: | $(OUT_DIR)
	SENSITIVITY_ANALYSIS=1 \
	SA_SLOPE_START="$(SA_SLOPE_START)" SA_SLOPE_LENGTH="$(SA_SLOPE_LENGTH)" \
	SA_YOB="$(SA_YOB)" SA_COHORTS="$(SA_COHORTS)" SA_DOSE_PAIRS="$(SA_DOSE_PAIRS)" \
	SA_ANCHOR_WEEKS="$(SA_ANCHOR_WEEKS)" SA_MA_TOTAL_LENGTH="$(SA_MA_TOTAL_LENGTH)" \
	SA_CENTERED="$(SA_CENTERED)" SA_SLOPE_WINDOW_SIZE="$(SA_SLOPE_WINDOW_SIZE)" \
	SA_FINAL_KCOR_MIN="$(SA_FINAL_KCOR_MIN)" SA_FINAL_KCOR_DATE="$(SA_FINAL_KCOR_DATE)" \
	$(PY) $(ROOT)/code/KCORv4.py $(CMR) $(OUT_XLSX) KCOR_summary_SA.log

clean:
	rm -f $(OUT_DIR)/KCORv4_sensitivity_smoke.xlsx
	rm -f $(OUT_DIR)/KCOR_summary_SA.log
	rm -f $(OUT_DIR)/KCOR_summary.xlsx


