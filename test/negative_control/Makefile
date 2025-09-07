ROOT := ../..
PY := python3

SRC_CMR := $(ROOT)/data/Czech/KCOR_CMR.xlsx
DATA_DIR := $(ROOT)/test/negative_control/data
OUT_DIR := $(ROOT)/test/negative_control/out
CODE_DIR := $(ROOT)/test/negative_control/code

SYNTH := $(DATA_DIR)/KCOR_synthetic_test.xlsx
SYNTH_OUT := $(OUT_DIR)/KCOR_processed_synthetic_test.xlsx

.PHONY: all run clean

all: $(SYNTH) run

$(DATA_DIR) $(OUT_DIR):
	mkdir -p $@

$(SYNTH): | $(DATA_DIR)
	$(PY) $(CODE_DIR)/generate_negative_control.py $(SRC_CMR) $(SYNTH)

run: | $(OUT_DIR)
	$(PY) $(ROOT)/code/KCORv4.py $(SYNTH) $(SYNTH_OUT) KCOR_summary.log

clean:
	rm -f $(SYNTH)
	rm -f $(SYNTH_OUT)
	rm -f $(OUT_DIR)/KCOR_summary.log


