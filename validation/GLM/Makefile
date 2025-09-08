# GLM Makefile

PYTHON ?= python3
SCRIPT := code/GLM.py
COMPARE := code/compare_pngs.py
INPUT  := ../../data/$(DATASET)/KCOR_variable.xlsx
OUTDIR := out
OUTPNG := $(OUTDIR)/GLM_plot_Czech_data.png
OUTXLSX := $(OUTDIR)/GLM_output.xlsx
REFPNG := ../../reference_results/GLM/GLM_plot_Czech_data.png

.PHONY: ls run clean outdir compare

ls:
	@echo "Directory: $(shell pwd)"
	@ls -alh

outdir:
	@mkdir -p $(OUTDIR)

# If the KCOR_variable.xlsx input is missing or stale vs. the script, build it via the code Makefile
$(INPUT): ../../code/KCOR_variable.py
	$(MAKE) -C ../../code KCOR_variable DATASET=$(DATASET)

# Build the outputs when inputs change
$(OUTPNG): $(INPUT) $(SCRIPT) | outdir
	$(PYTHON) $(SCRIPT) --input $(INPUT) --out-png $(OUTPNG) --out-xlsx $(OUTXLSX)

# Convenience alias; noop if $(OUTPNG) is up-to-date
run: $(OUTPNG)

clean:
	rm -rf $(OUTDIR)

compare: $(OUTPNG) $(COMPARE) | outdir
	$(PYTHON) $(COMPARE) --a $(OUTPNG) --b $(REFPNG) --out $(OUTDIR)/GLM_compare.png --report $(OUTDIR)/GLM_compare.txt


