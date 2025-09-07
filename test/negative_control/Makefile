ROOT := ../..
PY := python3

SRC_CMR := $(ROOT)/data/Czech/KCOR_CMR.xlsx
DATA_DIR := $(ROOT)/test/negative_control/data
OUT_DIR := $(ROOT)/test/negative_control/out
CODE_DIR := $(ROOT)/test/negative_control/code

SHEETS ?= 2021_24,2022_06
SYNTH := $(DATA_DIR)/KCOR_synthetic_neg_control.xlsx
OUT_XLSX := $(OUT_DIR)/KCOR_processed_neg_control.xlsx
SYNTH_OUT := $(OUT_DIR)/KCOR_processed_synthetic_test.xlsx

.PHONY: all run clean

all: $(SYNTH) run

$(DATA_DIR) $(OUT_DIR):
	mkdir -p $@

$(SYNTH): | $(DATA_DIR)
	$(PY) $(CODE_DIR)/generate_negative_control.py $(SRC_CMR) $(SYNTH) both $(SHEETS)

run: | $(OUT_DIR)
	$(PY) $(ROOT)/code/KCORv4.py $(SYNTH) $(OUT_XLSX) KCOR_summary_neg_control.log

clean:
	rm -f $(SYNTH)
	rm -f $(OUT_XLSX)
	rm -f $(OUT_DIR)/KCOR_summary_neg_control.log


