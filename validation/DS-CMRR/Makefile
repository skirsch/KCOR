# DS-CMRR Makefile

PYTHON ?= python3
SCRIPT := DS-CMRR.py
INPUT  := ../../data/Czech/KCOR_CMR.xlsx
OUTDIR := out
OUTPUT := $(OUTDIR)/DS-CMRR_output.xlsx
LOG    := DS_CMRR.log

.PHONY: ls run clean outdir

ls:
	@echo "Directory: $(shell pwd)"
	@ls -alh | cat

outdir:
	@mkdir -p $(OUTDIR)

run: outdir
	$(PYTHON) $(SCRIPT) $(INPUT) $(OUTPUT) --log $(LOG) --balance-report --export-dose-pairs --sheet 2021_24 --by-age --dose-num-min 2 --dose-den-exact 0 | cat

clean:
	rm -rf $(OUTDIR) $(LOG)


