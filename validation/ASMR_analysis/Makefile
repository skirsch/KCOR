ROOT := ../..
PY := python3

DATASET ?= Czech
CMR := $(ROOT)/data/$(DATASET)/KCOR_CMR.xlsx
OUT_DIR := $(ROOT)/validation/ASMR_analysis/out

.PHONY: all run clean

all: run

$(OUT_DIR):
	mkdir -p $@

LAST_DATE ?= 2024-04-01

run: | $(OUT_DIR)
	$(PY) $(ROOT)/validation/ASMR_analysis/code/ASMR_analysis.py --input $(CMR) --sheet 2022_06 --out_dir $(OUT_DIR) --last_date $(LAST_DATE)

clean:
	rm -f $(OUT_DIR)/ASMR_*.csv


