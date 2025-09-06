# DS-CMRR Makefile

PYTHON ?= python3
SCRIPT := DS-CMRR.py
INPUT  := ../../data/Czech/KCOR_CMR.xlsx
OUTDIR := out
OUTPUT := $(OUTDIR)/DS-CMRR_output.xlsx
LOG    := DS_CMRR.log
LOG_PATH := $(OUTDIR)/$(LOG)

.PHONY: ls run clean outdir by-age-only

ls:
	@echo "Directory: $(shell pwd)"
	@ls -alh

outdir:
	@mkdir -p $(OUTDIR)

# Build the main output only when inputs change
$(OUTPUT): $(INPUT) $(SCRIPT) | outdir
	$(PYTHON) $(SCRIPT) $(INPUT) $(OUTPUT) --log $(LOG) --balance-report --export-dose-pairs --sheet 2021_24 --by-age --dose-num-min 2 --dose-den-exact 0 --variance-method kish --save-png

# Convenience alias; noop if $(OUTPUT) is up-to-date
run: $(OUTPUT)


$(OUTDIR)/.by_age_only.stamp: $(INPUT) $(SCRIPT) | outdir
	$(PYTHON) $(SCRIPT) $(INPUT) $(OUTPUT) --log $(LOG) --sheet 2021_24 --by-age-only --dose-num-min 2 --dose-den-exact 0 --variance-method kish
	@touch $@

by-age-only: $(OUTDIR)/.by_age_only.stamp

clean:
	rm -rf $(OUTDIR) $(LOG)


