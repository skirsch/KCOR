ROOT := ../..
PY := python3

OUT_DIR := $(ROOT)/validation/HVE/out
CSV := $(OUT_DIR)/hve_output.csv
PNG := $(OUT_DIR)/hve_plot.png
XLSX ?= 

.PHONY: all run clean

all: run

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

run: | $(OUT_DIR)
	$(PY) $(ROOT)/validation/HVE/code/simulate_hve.py \
	  --N 1000000 \
	  --uptake 0.8 \
	  --frac_d2_within_vax 0.9 \
	  --baseline_rate 0.0005 \
	  --max_effect 0.75 \
	  --half_life 10 \
	  --t0_d2 5 \
	  --t0_d1 5 \
	  --weeks 30 \
	  --out_csv $(CSV) \
	  --out_png $(PNG) \
	  --out_xlsx "$(XLSX)" \
	  --show

clean:
	rm -f $(CSV) $(PNG) $(OUT_DIR)/hve_output.xlsx


