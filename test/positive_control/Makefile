# Positive control test Makefile
# Run from root Makefile as: make test
#
# This test validates that KCOR can detect injected effects:
# - Harm scenario: hazard multiplier r=1.2, expect KCOR > 1
# - Benefit scenario: hazard multiplier r=0.8, expect KCOR < 1
#
# The test generates synthetic data with known injected effects and
# verifies that KCOR deviates from 1 in the expected direction.

MAKEFLAGS += --no-print-directory

ROOT := ../..
PY := python3
DATASET ?= positive_control

# Paths
DATA_DIR := $(ROOT)/test/positive_control/data
OUT_DIR := $(ROOT)/test/positive_control/out
ANALYSIS_DIR := $(ROOT)/test/positive_control/analysis
CODE_DIR := $(ROOT)/test/positive_control/code

# Input/output files
POS_CTRL_DATA := $(DATA_DIR)/KCOR_positive_control.xlsx
POS_CTRL_OUT := $(OUT_DIR)/KCOR_processed_positive_control.xlsx
POS_CTRL_LOG := $(OUT_DIR)/KCOR_summary_positive_control.log
POS_CTRL_FIG := $(ANALYSIS_DIR)/fig_pos_control_injected.png

.PHONY: all clean generate run figure

all: $(POS_CTRL_FIG)

# Generate synthetic positive control data
$(POS_CTRL_DATA): $(CODE_DIR)/generate_positive_control.py
	@mkdir -p $(DATA_DIR)
	$(PY) $(CODE_DIR)/generate_positive_control.py --both --output $(POS_CTRL_DATA)

# Run KCOR analysis on positive control data
$(POS_CTRL_OUT): $(POS_CTRL_DATA) $(ROOT)/code/KCOR.py
	@mkdir -p $(OUT_DIR)
	cd $(ROOT)/code && $(PY) KCOR.py \
		../test/positive_control/data/KCOR_positive_control.xlsx \
		../test/positive_control/out/KCOR_processed_positive_control.xlsx \
		"Positive Control" \
		KCOR_summary_positive_control.log

# Generate figure from processed data
$(POS_CTRL_FIG): $(POS_CTRL_OUT) $(CODE_DIR)/plot_positive_control.py
	@mkdir -p $(ANALYSIS_DIR)
	$(PY) $(CODE_DIR)/plot_positive_control.py \
		--input $(POS_CTRL_OUT) \
		--output $(POS_CTRL_FIG)

generate: $(POS_CTRL_DATA)
	@echo "Generated positive control data: $(POS_CTRL_DATA)"

run: $(POS_CTRL_OUT)
	@echo "Processed positive control data: $(POS_CTRL_OUT)"

figure: $(POS_CTRL_FIG)
	@echo "Generated figure: $(POS_CTRL_FIG)"

clean:
	rm -f $(DATA_DIR)/KCOR_positive_control.xlsx
	rm -f $(OUT_DIR)/KCOR_processed_positive_control.xlsx
	rm -f $(OUT_DIR)/KCOR_summary_positive_control.log
	rm -f $(OUT_DIR)/KCOR_slope_debug.csv
	rm -f $(ANALYSIS_DIR)/fig_pos_control_injected.png

