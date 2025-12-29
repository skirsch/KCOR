# Simulation Grid Makefile
# Run from root Makefile as: make sim_grid
#
# This test generates a prespecified simulation grid demonstrating:
# - Correct null behavior under selection-induced curvature
# - Detection of injected effects (hazard increase/decrease)
# - Failure modes and diagnostics under misspecification
#
# Six scenarios are implemented:
# 1. Gamma-frailty null (strong selection)
# 2. Injected hazard increase (r=1.2)
# 3. Injected hazard decrease (r=0.8)
# 4. Non-gamma frailty null (lognormal)
# 5. Quiet-window contamination
# 6. Sparse-events regime

MAKEFLAGS += --no-print-directory

ROOT := ../..
# Use passed PYTHON if provided, otherwise default to python3
PYTHON ?= python3
PY := $(PYTHON)

# Paths
CODE_DIR := $(ROOT)/test/sim_grid/code
DATA_DIR := $(ROOT)/test/sim_grid/data
OUT_DIR := $(ROOT)/test/sim_grid/out
FIG_DIR := $(ROOT)/documentation/preprint/figures

# Output files
SIM_DATA := $(DATA_DIR)/sim_grid_data.xlsx
SIM_RESULTS := $(OUT_DIR)/sim_grid_results.xlsx
SIM_DIAGNOSTICS := $(OUT_DIR)/sim_grid_diagnostics.csv
FIG_OVERVIEW := $(OUT_DIR)/fig_sim_grid_overview.png
FIG_DIAGNOSTICS := $(OUT_DIR)/fig_sim_grid_diagnostics.png

# S7 simulation files
S7_DATA := $(DATA_DIR)/s7_data.xlsx
S7_RESULTS := $(OUT_DIR)/s7_results.xlsx
S7_DIAGNOSTICS := $(OUT_DIR)/s7_diagnostics.csv
FIG_S7_OVERVIEW := $(OUT_DIR)/fig_s7_overview.png
FIG_S7_DIAGNOSTICS := $(OUT_DIR)/fig_s7_diagnostics.png

.PHONY: all clean generate run figures copy-figures s7 s7-figures copy-s7-figures

all: copy-figures copy-s7-figures

# Generate simulation data and run KCOR fitting
$(SIM_RESULTS) $(SIM_DIAGNOSTICS): $(CODE_DIR)/generate_sim_grid.py
	@mkdir -p $(DATA_DIR) $(OUT_DIR)
	$(PY) $(CODE_DIR)/generate_sim_grid.py \
		--output-data $(SIM_DATA) \
		--output-results $(SIM_RESULTS) \
		--output-diagnostics $(SIM_DIAGNOSTICS)

# Generate figures from results
$(FIG_OVERVIEW) $(FIG_DIAGNOSTICS): $(SIM_RESULTS) $(SIM_DIAGNOSTICS) $(CODE_DIR)/plot_sim_grid.py
	$(PY) $(CODE_DIR)/plot_sim_grid.py \
		--input-results $(SIM_RESULTS) \
		--input-diagnostics $(SIM_DIAGNOSTICS) \
		--output-overview $(FIG_OVERVIEW) \
		--output-diagnostics $(FIG_DIAGNOSTICS)

# Copy figures to preprint figures directory
copy-figures: $(FIG_OVERVIEW) $(FIG_DIAGNOSTICS)
	@mkdir -p $(FIG_DIR)
	cp $(FIG_OVERVIEW) $(FIG_DIR)/fig_sim_grid_overview.png
	cp $(FIG_DIAGNOSTICS) $(FIG_DIR)/fig_sim_grid_diagnostics.png
	@echo "Copied figures to $(FIG_DIR)"

# S7 simulation generation
$(S7_RESULTS) $(S7_DIAGNOSTICS): $(CODE_DIR)/generate_s7_sim.py
	@mkdir -p $(DATA_DIR) $(OUT_DIR)
	$(PY) $(CODE_DIR)/generate_s7_sim.py \
		--output-data $(S7_DATA) \
		--output-results $(S7_RESULTS) \
		--output-diagnostics $(S7_DIAGNOSTICS)

# S7 figures
$(FIG_S7_OVERVIEW) $(FIG_S7_DIAGNOSTICS): $(S7_RESULTS) $(S7_DIAGNOSTICS) $(CODE_DIR)/plot_s7_sim.py
	$(PY) $(CODE_DIR)/plot_s7_sim.py \
		--input-results $(S7_RESULTS) \
		--input-diagnostics $(S7_DIAGNOSTICS) \
		--output-overview $(FIG_S7_OVERVIEW) \
		--output-diagnostics $(FIG_S7_DIAGNOSTICS)

# Copy S7 figures
copy-s7-figures: $(FIG_S7_OVERVIEW) $(FIG_S7_DIAGNOSTICS)
	@mkdir -p $(FIG_DIR)
	cp $(FIG_S7_OVERVIEW) $(FIG_DIR)/fig_s7_overview.png
	cp $(FIG_S7_DIAGNOSTICS) $(FIG_DIR)/fig_s7_diagnostics.png
	@echo "Copied S7 figures to $(FIG_DIR)"

s7: $(S7_RESULTS)
	@echo "Generated S7 simulation data and results"

s7-figures: $(FIG_S7_OVERVIEW) $(FIG_S7_DIAGNOSTICS)
	@echo "Generated S7 figures: $(FIG_S7_OVERVIEW) $(FIG_S7_DIAGNOSTICS)"

generate: $(SIM_RESULTS)
	@echo "Generated simulation data and results"

figures: $(FIG_OVERVIEW) $(FIG_DIAGNOSTICS)
	@echo "Generated figures: $(FIG_OVERVIEW) $(FIG_DIAGNOSTICS)"

clean:
	rm -f $(DATA_DIR)/*.xlsx
	rm -f $(OUT_DIR)/*.xlsx
	rm -f $(OUT_DIR)/*.csv
	rm -f $(OUT_DIR)/*.png
	rm -f $(FIG_DIR)/fig_sim_grid_overview.png
	rm -f $(FIG_DIR)/fig_sim_grid_diagnostics.png
	rm -f $(FIG_DIR)/fig_s7_overview.png
	rm -f $(FIG_DIR)/fig_s7_diagnostics.png

