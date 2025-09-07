# KM Makefile
# Run from the root Makefile as make validation
# Example:
#     make -C validation/kaplan_meier run EQUAL_POPULATION=0

PYTHON ?= python3
SCRIPT := code/KM.py
INPUT  := ../../data/Czech/KCOR_CMR.xlsx
OUTDIR := out

# Defaults (can be overridden: make run SHEET=2021_24 START=1940 END=2000 PLOT="0,2,1")
# note that 1940-2999 is good, but 1940 to 1995 is amazingly tight. is tighter and more obvious
SHEET  ?= 2021_24
START  ?= 1950
END    ?= 1950
PLOT   ?= 0,2,1
# PLOT syntax: start,end[,step] → plots Dose-start vs combined [start+step..end].
# Example: 0,2,1 ⇒ 0 vs (1+2). If step omitted, defaults to 1.
EQUAL_POPULATION ?= 0
# EQUAL_POPULATION: 1 = equalize initial populations at enrollment (use reference cohort's N0),
#                   0 = use source populations as supplied (no equalization).
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
	$(PYTHON) $(SCRIPT) --sheet $(SHEET) --birth-years $(START) $(END) --plot "$(PLOT)" --out $(OUTPNG) --input $(INPUT) --equal-population $(EQUAL_POPULATION)

clean:
	rm -rf $(OUTDIR)


