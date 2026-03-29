MAKEFLAGS += --no-print-directory

ROOT := ../..
PYTHON ?= python3
PY := $(PYTHON)
DATASET ?= Czech

CODE_DIR := $(ROOT)/test/alpha/code
OUT_DIR := $(ROOT)/test/alpha/out
CONFIG := $(ROOT)/test/alpha/params_alpha.yaml

BEST_EST := $(OUT_DIR)/alpha_best_estimates.csv
SYNTH_RECOVERY := $(OUT_DIR)/alpha_synthetic_recovery.csv
FIG_OBJECTIVES := $(OUT_DIR)/fig_alpha_objectives.png

.PHONY: all clean

all: $(BEST_EST) $(SYNTH_RECOVERY) $(FIG_OBJECTIVES)

$(OUT_DIR):
	mkdir -p $@

$(BEST_EST) $(SYNTH_RECOVERY) $(FIG_OBJECTIVES): $(CODE_DIR)/estimate_alpha.py $(CONFIG) | $(OUT_DIR)
	DATASET=$(DATASET) $(PY) $(CODE_DIR)/estimate_alpha.py --config $(CONFIG) --outdir $(OUT_DIR)

clean:
	rm -f $(OUT_DIR)/*.csv
	rm -f $(OUT_DIR)/*.png
