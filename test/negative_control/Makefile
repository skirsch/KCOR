ROOT := ../..
PY := python3
DATASET ?= Czech

SRC_CMR := $(ROOT)/data/$(DATASET)/KCOR_CMR.xlsx
DATA_DIR := $(ROOT)/test/negative_control/data
OUT_DIR := $(ROOT)/test/negative_control/out
CODE_DIR := $(ROOT)/test/negative_control/code

SHEETS ?= 2021_24,2022_06
SYNTH := $(DATA_DIR)/KCOR_synthetic_neg_control.xlsx
OUT_XLSX := $(OUT_DIR)/KCOR_processed_neg_control.xlsx

.PHONY: all clean

all: $(OUT_XLSX)

$(DATA_DIR) $(OUT_DIR):
	mkdir -p $@

$(SYNTH): $(SRC_CMR) $(CODE_DIR)/generate_negative_control.py | $(DATA_DIR)
	$(PY) $(CODE_DIR)/generate_negative_control.py $(SRC_CMR) $(SYNTH) $(SHEETS)

# Build negative control output only if stale/missing; turn off baseline scaling since 
# we are not dealing with a known net harmful vaccine.
$(OUT_XLSX): $(SYNTH) | $(OUT_DIR)
	SA_FINAL_KCOR_MIN=0 $(PY) $(ROOT)/code/KCOR.py $(SYNTH) $(OUT_XLSX) KCOR_summary_neg_control.log

clean:
	rm -f $(SYNTH)
	rm -f $(OUT_XLSX)
	rm -f $(OUT_DIR)/KCOR_summary_neg_control.log


