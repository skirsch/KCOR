# KM Makefile

PYTHON ?= python3
SCRIPT := code/KM.py
INPUT  := ../../data/Czech/KCOR_CMR.xlsx
OUTDIR := out

# Defaults (can be overridden: make run SHEET=2021_24 START=1940 END=2000 GROUPA="0" GROUPB="1,2")
# note that 1940-2999 is good, but 1950-1970 is tighter and more obvious
SHEET  ?= 2021_24
START  ?= 1950
END    ?= 1970
GROUPA ?= 0
GROUPB ?= 1,2
OUTPNG ?= $(OUTDIR)/KM_$(SHEET)_$(START)_$(END).png

.PHONY: all run clean outdir ls

# Default target
all: run

ls:
	@echo "Directory: $(shell pwd)"
	@ls -alh

outdir:
	@mkdir -p $(OUTDIR)

run: $(OUTPNG)

$(OUTPNG): $(INPUT) $(SCRIPT) | outdir
	$(PYTHON) $(SCRIPT) --sheet $(SHEET) --birth-years $(START) $(END) --groups "$(GROUPA)" "$(GROUPB)" --out $(OUTPNG) --input $(INPUT)

clean:
	rm -rf $(OUTDIR)


