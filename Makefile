# Root Makefile for KCOR

# Suppress Entering/Leaving directory noise from sub-makes
MAKEFLAGS += --no-print-directory

CODE_DIR := code
VALIDATION_DSCMRR_DIR := validation/DS-CMRR
VALIDATION_KM_DIR := validation/kaplan_meier
VALIDATION_GLM_DIR := validation/GLM
VALIDATION_HVE_DIR := validation/HVE
VALIDATION_ASMR_DIR := validation/ASMR_analysis

# Paper build (Pandoc)
PAPER_DIR ?= documentation/preprint
PAPER_MD ?= paper.md
PAPER_PDF ?= paper.pdf
PAPER_TEX ?= paper.tex
PAPER_BIB ?= KCOR_references.json
PAPER_BIB_PATH := $(PAPER_DIR)/$(PAPER_BIB)
PAPER_CSL ?= american-medical-association.csl
# PDF engine (override on CLI if needed). Default: xelatex.
PAPER_PDF_ENGINE ?= xelatex
PAPER_PDF_GEOMETRY ?= margin=1in
PAPER_PDF_MAINFONT ?= TeX Gyre Termes
PAPER_PDF_MATHFONT ?= TeX Gyre Termes Math

.PHONY: all KCOR CMR CMR_from_krf monte_carlo convert validation test clean sensitivity KCOR_variable HVE ASMR ts icd10 icd_population_shift mortality mortality_sensitivity mortality_age mortality_stats mortality_plots mortality_all install install-debian slope-test quiet-window paper paper-tex paper-pdf sim_grid cox-bias cox-bias-figures copy-cox-bias-figures skip-weeks cohort-size rollout help identifiability

# Dataset namespace (override on CLI: make DATASET=USA)
DATASET ?= Czech

# Monte Carlo iterations (override on CLI: make monte_carlo MC_ITERATIONS=50)
MC_ITERATIONS ?= 4
# Monte Carlo enrollment cohort (ISO week label; accepts YYYY_WW or YYYY-WW)
MC_ENROLLMENT_DATE ?= 2021_24

# Virtual environment path
VENV_DIR := .venv
VENV_PYTHON := $(VENV_DIR)/bin/python3
VENV_PIP := $(VENV_DIR)/bin/pip

# Install dependencies using pip in virtual environment
install: $(VENV_PIP)
	@echo "Installing dependencies from requirements.txt in virtual environment..."
	@echo "Upgrading pip..."
	@$(VENV_PIP) install --upgrade pip
	@echo "Installing requirements..."
	@$(VENV_PIP) install -r requirements.txt
	@echo "Installation complete!"

# Ensure pip exists in virtual environment
$(VENV_PIP): $(VENV_PYTHON)
	@if [ ! -f "$(VENV_PIP)" ]; then \
		echo "Installing pip in virtual environment..."; \
		$(VENV_PYTHON) -m ensurepip --upgrade; \
	fi

# Create virtual environment directory and Python executable
$(VENV_PYTHON):
	@if [ ! -d "$(VENV_DIR)" ] || [ ! -f "$(VENV_PYTHON)" ]; then \
		echo "Creating virtual environment in $(VENV_DIR)..."; \
		python3 -m venv $(VENV_DIR); \
		echo "Virtual environment created!"; \
	fi

# Install dependencies using Debian packages (alternative to pip)
install-debian:
	@echo "Installing dependencies using Debian packages..."
	@echo "This requires sudo privileges."
	sudo apt-get update
	sudo apt-get install -y \
		python3-pandas \
		python3-numpy \
		python3-openpyxl \
		python3-statsmodels \
		python3-scipy \
		python3-matplotlib \
		python3-seaborn
	@echo "Debian package installation complete!"
	@echo "Note: Package versions may differ from requirements.txt. For exact versions, use 'make install' with pip."

# Default: build everything (variable-cohort + analysis + validation + tests)
all: KCOR_variable KCOR validation test

# KCOR analysis pipeline (delegates to code/Makefile target KCOR)
KCOR: $(VENV_PYTHON)
	$(MAKE) -C $(CODE_DIR) KCOR DATASET=$(DATASET) PYTHON=$(abspath $(VENV_PYTHON))

# CMR aggregation only (delegates to code/Makefile target CMR)
CMR:
	$(MAKE) -C $(CODE_DIR) CMR DATASET=$(DATASET)

# Monte Carlo mode (delegates to code/Makefile target monte_carlo)
monte_carlo:
	$(MAKE) -C $(CODE_DIR) monte_carlo DATASET=$(DATASET) MC_ITERATIONS=$(MC_ITERATIONS) MC_ENROLLMENT_DATE=$(MC_ENROLLMENT_DATE)

# Run CMR on KRF input by adapting to Czech-like format first
CMR_from_krf:
	$(MAKE) -C $(CODE_DIR) CMR_from_krf DATASET=$(DATASET)

# Dataset converter (delegates to data/<DATASET>/Makefile)
convert:
	$(MAKE) -C data/$(DATASET) convert

# Variable-cohort aggregation (delegates to code/Makefile target KCOR_variable)
KCOR_variable:
	$(MAKE) -C $(CODE_DIR) KCOR_variable DATASET=$(DATASET)

# Time series aggregation (delegates to code/Makefile target ts)
ts:
	$(MAKE) -C $(CODE_DIR) ts DATASET=$(DATASET)

# Identifiability analysis: weekly incident booster (Czech 2021)
#
# Usage:
#   make identifiability
#   make identifiability IDENT_MAX_ROWS=1000000 IDENT_OUTDIR=/tmp/kcor_ident_1m
#
IDENT_DIR := identifiability/Czech/code
IDENT_SCRIPT := $(IDENT_DIR)/build_weekly_emulation.py
IDENT_ANALYZE_SCRIPT := $(IDENT_DIR)/analyze_identifiability.py
IDENT_INPUT ?= data/$(DATASET)/records.csv
# Default to a repo-local output directory (permanent, Windows-backed in WSL).
# Put outputs under identifiability/<DATASET>/booster/ to avoid redundant naming.
IDENT_OUTDIR ?= identifiability/$(DATASET)/booster
IDENT_ENROLLMENT_START ?= 2021-10-18
IDENT_N_ENROLLMENTS ?= 16
IDENT_LOOKBACK_DAYS ?= 7
# Dose3 incident window length (weeks before enrollment)
IDENT_DOSE3_INCIDENT_LOOKBACK_WEEKS ?= 4
IDENT_DOSE3_BIN_WEEKS ?= 4
IDENT_FOLLOWUP_WEEKS ?= 26
# Birth-year filter (optional). Leave blank for all ages.
IDENT_BIRTH_YEAR_MIN ?=
IDENT_BIRTH_YEAR_MAX ?=
IDENT_MAX_ROWS ?=
IDENT_FILTER_NON_MRNA ?= 1
IDENT_ANALYZE ?= 1

IDENT_ARGS := \
	--input $(IDENT_INPUT) \
	--outdir $(IDENT_OUTDIR) \
	--enrollment-start $(IDENT_ENROLLMENT_START) \
	--n-enrollments $(IDENT_N_ENROLLMENTS) \
	--lookback-days $(IDENT_LOOKBACK_DAYS) \
	--dose3-incident-lookback-weeks $(IDENT_DOSE3_INCIDENT_LOOKBACK_WEEKS) \
	--dose3-bin-weeks $(IDENT_DOSE3_BIN_WEEKS) \
	--followup-weeks $(IDENT_FOLLOWUP_WEEKS)

# Common args for multi-variant runs (outdir set per variant)
IDENT_COMMON_ARGS := \
	--input $(IDENT_INPUT) \
	--enrollment-start $(IDENT_ENROLLMENT_START) \
	--n-enrollments $(IDENT_N_ENROLLMENTS) \
	--lookback-days $(IDENT_LOOKBACK_DAYS) \
	--dose3-incident-lookback-weeks $(IDENT_DOSE3_INCIDENT_LOOKBACK_WEEKS) \
	--dose3-bin-weeks $(IDENT_DOSE3_BIN_WEEKS) \
	--followup-weeks $(IDENT_FOLLOWUP_WEEKS)

ifneq ($(strip $(IDENT_BIRTH_YEAR_MIN)),)
ifneq ($(strip $(IDENT_BIRTH_YEAR_MAX)),)
IDENT_ARGS += --birth-year-min $(IDENT_BIRTH_YEAR_MIN) --birth-year-max $(IDENT_BIRTH_YEAR_MAX)
IDENT_COMMON_ARGS += --birth-year-min $(IDENT_BIRTH_YEAR_MIN) --birth-year-max $(IDENT_BIRTH_YEAR_MAX)
endif
endif

ifneq ($(strip $(IDENT_MAX_ROWS)),)
IDENT_ARGS += --max-rows $(IDENT_MAX_ROWS)
IDENT_COMMON_ARGS += --max-rows $(IDENT_MAX_ROWS)
endif

ifeq ($(IDENT_FILTER_NON_MRNA),0)
IDENT_ARGS += --no-filter-non-mrna
IDENT_COMMON_ARGS += --no-filter-non-mrna
endif

identifiability: $(VENV_PYTHON)
	@echo "Running identifiability emulation (age-stratified outputs; single read of records.csv)..."
	@echo "  Script: $(IDENT_SCRIPT)"
	@echo "  Input:  $(IDENT_INPUT)"
	@echo "  Base:   $(IDENT_OUTDIR)"
	@set -e; \
	common_args="--input $(IDENT_INPUT) --enrollment-start $(IDENT_ENROLLMENT_START) --n-enrollments $(IDENT_N_ENROLLMENTS) --lookback-days $(IDENT_LOOKBACK_DAYS) --dose3-incident-lookback-weeks $(IDENT_DOSE3_INCIDENT_LOOKBACK_WEEKS) --dose3-bin-weeks $(IDENT_DOSE3_BIN_WEEKS) --followup-weeks $(IDENT_FOLLOWUP_WEEKS)"; \
	if [ -n "$(strip $(IDENT_MAX_ROWS))" ]; then common_args="$$common_args --max-rows $(IDENT_MAX_ROWS)"; fi; \
	if [ "$(IDENT_FILTER_NON_MRNA)" = "0" ]; then common_args="$$common_args --no-filter-non-mrna"; fi; \
	$(abspath $(VENV_PYTHON)) $(IDENT_SCRIPT) --outdir "$(IDENT_OUTDIR)" $$common_args \
		--strata all_ages \
		--strata born_193x:1930:1939 \
		--strata born_194x:1940:1949 \
		--strata born_195x:1950:1959; \
	if [ "$(IDENT_ANALYZE)" = "1" ]; then \
		for name in all_ages born_193x born_194x born_195x; do \
			echo ""; \
			echo "Post-run analysis (peak locking): $$name"; \
			$(abspath $(VENV_PYTHON)) $(IDENT_ANALYZE_SCRIPT) --outdir "$(IDENT_OUTDIR)/$$name" > "$(IDENT_OUTDIR)/$$name/analysis_report.txt"; \
			echo "  Wrote: $(IDENT_OUTDIR)/$$name/analysis_report.txt"; \
		done; \
	fi

.PHONY: identifiability-locking
IDENT_LOCKING_OUTDIR ?= $(IDENT_OUTDIR)/all_ages
identifiability-locking: $(VENV_PYTHON)
	@echo "Identifiability locking summary (from series.csv in $(IDENT_LOCKING_OUTDIR))"
	@tmp="/tmp/kcor_ident_locking_$$RANDOM.txt"; \
	$(abspath $(VENV_PYTHON)) $(IDENT_ANALYZE_SCRIPT) --outdir $(IDENT_LOCKING_OUTDIR) > $$tmp; \
	awk 'BEGIN{f=0} /^=== Locking summary ===/{f=1} f{print}' $$tmp; \
	rm -f $$tmp

.PHONY: identifiability-falsify
IDENT_FALSIFY_DIR ?= $(IDENT_OUTDIR)/falsification
IDENT_PLACEBO_START_WEEKS ?= 4
IDENT_PLACEBO_END_WEEKS ?= 8
IDENT_EVENTUAL_HORIZON_WEEKS ?= 12
IDENT_EXCLUDE_RECENT_INFECTION_WEEKS ?= 8
# Time-since-dose2 bins in DAYS. Format: "min-max min-max ..." where max is exclusive.
IDENT_TSD2_BINS ?= 0-90 90-180 180-270 270-360 360-100000

# Selection/eligibility falsification knobs
IDENT_TT_TSD2_MIN_DAYS ?= 180
IDENT_TT_TSD2_MAX_DAYS ?= 360
IDENT_TT_TREATED_WINDOW_WEEKS ?= 1
IDENT_LEAD_MAX_WEEKS ?= 8
IDENT_PERM_REPS ?= 200
IDENT_PERM_SEED ?= 1

identifiability-falsify: $(VENV_PYTHON)
	@echo "Running identifiability falsification suite..."
	@echo "  Base outdir: $(IDENT_FALSIFY_DIR)"
	@mkdir -p $(IDENT_FALSIFY_DIR)
	@set -e; \
	run_variant() { \
		name="$$1"; shift; \
		vdir="$(IDENT_FALSIFY_DIR)/$$name"; \
		mkdir -p "$$vdir"; \
		echo ""; \
		echo "=== Variant: $$name ==="; \
		$(abspath $(VENV_PYTHON)) $(IDENT_SCRIPT) --outdir "$$vdir" $(IDENT_COMMON_ARGS) "$$@"; \
		$(abspath $(VENV_PYTHON)) $(IDENT_ANALYZE_SCRIPT) --outdir "$$vdir" --auc-weeks 8 --label "$$name" --metrics-csv "$$vdir/analysis_metrics.csv" > "$$vdir/analysis_report.txt"; \
	}; \
	run_variant baseline; \
	run_variant placebo_future$(IDENT_PLACEBO_START_WEEKS)_$(IDENT_PLACEBO_END_WEEKS) --dose3-future-start-weeks $(IDENT_PLACEBO_START_WEEKS) --dose3-future-end-weeks $(IDENT_PLACEBO_END_WEEKS); \
	run_variant eventual_booster$(IDENT_EVENTUAL_HORIZON_WEEKS) --restrict-dose2-eventual-dose3-weeks $(IDENT_EVENTUAL_HORIZON_WEEKS); \
	run_variant exclude_recent_infection$(IDENT_EXCLUDE_RECENT_INFECTION_WEEKS) --exclude-recent-infection-weeks $(IDENT_EXCLUDE_RECENT_INFECTION_WEEKS); \
	run_variant selection_suite_tsd2$(IDENT_TT_TSD2_MIN_DAYS)_$(IDENT_TT_TSD2_MAX_DAYS) \
		--selection-suite \
		--tt-tsd2-min-days $(IDENT_TT_TSD2_MIN_DAYS) \
		--tt-tsd2-max-days $(IDENT_TT_TSD2_MAX_DAYS) \
		--tt-treated-window-weeks $(IDENT_TT_TREATED_WINDOW_WEEKS) \
		--lead-max-weeks $(IDENT_LEAD_MAX_WEEKS) \
		--perm-reps $(IDENT_PERM_REPS) \
		--perm-seed $(IDENT_PERM_SEED); \
	for bin in $(IDENT_TSD2_BINS); do \
		min="$${bin%-*}"; max="$${bin#*-}"; \
		run_variant "tsd2_$${min}_$${max}" --restrict-tsd2-min-days "$$min" --restrict-tsd2-max-days "$$max"; \
	done; \
	$(abspath $(VENV_PYTHON)) identifiability/Czech/code/aggregate_falsification.py --base "$(IDENT_FALSIFY_DIR)" --out "$(IDENT_FALSIFY_DIR)/falsification_summary.csv"

.PHONY: identifiability-falsify-selection
identifiability-falsify-selection: $(VENV_PYTHON)
	@echo "Running identifiability selection/eligibility falsification suite..."
	@echo "  Base outdir: $(IDENT_FALSIFY_DIR)"
	@mkdir -p $(IDENT_FALSIFY_DIR)
	@set -e; \
	name="selection_suite_tsd2$(IDENT_TT_TSD2_MIN_DAYS)_$(IDENT_TT_TSD2_MAX_DAYS)"; \
	vdir="$(IDENT_FALSIFY_DIR)/$$name"; \
	mkdir -p "$$vdir"; \
	echo ""; \
	echo "=== Variant: $$name ==="; \
	$(abspath $(VENV_PYTHON)) $(IDENT_SCRIPT) --outdir "$$vdir" $(IDENT_COMMON_ARGS) \
		--selection-suite \
		--tt-tsd2-min-days $(IDENT_TT_TSD2_MIN_DAYS) \
		--tt-tsd2-max-days $(IDENT_TT_TSD2_MAX_DAYS) \
		--tt-treated-window-weeks $(IDENT_TT_TREATED_WINDOW_WEEKS) \
		--lead-max-weeks $(IDENT_LEAD_MAX_WEEKS) \
		--perm-reps $(IDENT_PERM_REPS) \
		--perm-seed $(IDENT_PERM_SEED); \
	$(abspath $(VENV_PYTHON)) $(IDENT_ANALYZE_SCRIPT) --outdir "$$vdir" --auc-weeks 8 --label "$$name" --metrics-csv "$$vdir/analysis_metrics.csv" > "$$vdir/analysis_report.txt"

# Validation suite (DS-CMRR, Kaplan–Meier, GLM)
validation:
	$(MAKE) -C $(VALIDATION_DSCMRR_DIR) run DATASET=$(DATASET)
	$(MAKE) -C $(VALIDATION_KM_DIR) run DATASET=$(DATASET)
	$(MAKE) -C $(VALIDATION_GLM_DIR) run DATASET=$(DATASET)

# Convenience target to run only Kaplan–Meier
km:
	$(MAKE) -C $(VALIDATION_KM_DIR) run DATASET=$(DATASET)

# Convenience target to run only GLM
glm:
	$(MAKE) -C $(VALIDATION_GLM_DIR) run DATASET=$(DATASET)

glm-compare:
	$(MAKE) -C $(VALIDATION_GLM_DIR) compare DATASET=$(DATASET)

# Negative-control test (delegates to test/Makefile)
test:
	$(MAKE) -C test all

# Slope normalization test
slope-test: $(VENV_PYTHON)
	@echo "Running slope normalization test..."
	cd test/slope_normalization && $(abspath $(VENV_PYTHON)) test.py
	@echo "Slope normalization test complete!"

# Quiet-window sensitivity scan (Czech 2021_24)
quiet-window: $(VENV_PYTHON)
	@echo "Running quiet-window scan (Czech 2021_24)..."
	$(abspath $(VENV_PYTHON)) test/quiet_window/code/quiet_window_scan_theta_czech_2021_24.py
	@echo "Quiet-window scan complete!"

# Simulation grid (operating characteristics and failure-mode diagnostics)
sim_grid: $(VENV_PYTHON)
	$(MAKE) -C test/sim_grid all PYTHON=$(abspath $(VENV_PYTHON))

# Cox bias demonstration (Cox regression bias under frailty heterogeneity)
cox-bias: $(VENV_PYTHON)
	$(MAKE) -C test/sim_grid cox-bias PYTHON=$(abspath $(VENV_PYTHON))

cox-bias-figures: $(VENV_PYTHON)
	$(MAKE) -C test/sim_grid cox-bias-figures PYTHON=$(abspath $(VENV_PYTHON))

copy-cox-bias-figures: $(VENV_PYTHON)
	$(MAKE) -C test/sim_grid copy-cox-bias-figures PYTHON=$(abspath $(VENV_PYTHON))

# Skip-weeks sensitivity figure
skip-weeks: $(VENV_PYTHON)
	$(MAKE) -C test/sim_grid skip-weeks PYTHON=$(abspath $(VENV_PYTHON))

# Cohort size sensitivity analysis
cohort-size: $(VENV_PYTHON)
	$(MAKE) -C test/sim_grid cohort-size PYTHON=$(abspath $(VENV_PYTHON))

# Rollout design-mimic simulation
rollout: $(VENV_PYTHON)
	$(MAKE) -C test/sim_grid rollout PYTHON=$(abspath $(VENV_PYTHON))

clean:
	-$(MAKE) -C $(CODE_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_DSCMRR_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_KM_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_GLM_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_HVE_DIR) clean DATASET=$(DATASET)
	-$(MAKE) -C $(VALIDATION_ASMR_DIR) clean DATASET=$(DATASET)


sensitivity:
	$(MAKE) -C test/sensitivity all DATASET=$(DATASET)

# Build methods paper (Pandoc → LaTeX/PDF)
#
# Default inputs live in documentation/preprint/:
# - paper.md (main manuscript)
# - supplement.md (Supplementary Information)
# - KCOR_references.json
# - american-medical-association.csl
#
# Outputs:
# - paper.pdf (PDF)
# - paper.tex (LaTeX)
#
# Usage:
#   make paper
#   make paper PAPER_MD=paper_v7.md PAPER_PDF=paper_v7.pdf PAPER_TEX=paper_v7.tex
paper: $(PAPER_DIR)/$(PAPER_PDF) $(PAPER_DIR)/$(PAPER_TEX)
	@echo "Copying paper files to website..."
	@scp $(PAPER_DIR)/$(PAPER_PDF) truenas:/mnt/main/www/skirsch.com/covid/KCOR/KCOR.pdf
	@echo "Paper files copied to website."

paper-pdf: $(PAPER_DIR)/$(PAPER_PDF)
paper-tex: $(PAPER_DIR)/$(PAPER_TEX)

# Optional: include Supplementary Information when building the combined paper outputs.
PAPER_SUPP_MD ?= supplement.md

$(PAPER_DIR)/$(PAPER_TEX): $(PAPER_DIR)/$(PAPER_MD) $(PAPER_DIR)/$(PAPER_SUPP_MD) $(PAPER_BIB_PATH) $(PAPER_DIR)/$(PAPER_CSL) $(PAPER_DIR)/header.tex $(wildcard $(PAPER_DIR)/figures/*)
	@echo "Building LaTeX: $(PAPER_DIR)/$(PAPER_MD) -> $(PAPER_DIR)/$(PAPER_TEX)"
	@cd $(PAPER_DIR) && \
		if grep -n -E '\\\\n\\+|<<<<<<<|=======|>>>>>>>' "$(PAPER_MD)" "$(PAPER_SUPP_MD)"; then \
			echo ""; \
			echo "ERROR: Found suspicious merge/diff artifacts in paper sources (e.g., literal '\\n+' or conflict markers)."; \
			echo "Fix the indicated lines in $(PAPER_MD) / $(PAPER_SUPP_MD) before building."; \
			exit 1; \
		fi
	@cd $(PAPER_DIR) && \
		pandoc $(PAPER_MD) $(PAPER_SUPP_MD) \
			--to=latex \
			--filter pandoc-crossref \
			--lua-filter pagebreak-tables.lua \
			--metadata-file pandoc-crossref.yaml \
			--citeproc \
			-V geometry:$(PAPER_PDF_GEOMETRY) \
			-V mainfont="$(PAPER_PDF_MAINFONT)" \
			-V mathfont="$(PAPER_PDF_MATHFONT)" \
			-H header.tex \
			--bibliography=$(PAPER_BIB) \
			--csl=$(PAPER_CSL) \
			-o $(PAPER_TEX).new
	@cd $(PAPER_DIR) && \
		mv -f $(PAPER_TEX).new $(PAPER_TEX) 2>/dev/null || ( \
			echo "ERROR: could not overwrite $(PAPER_DIR)/$(PAPER_TEX) (is it open?)."; \
			echo "Leaving: $(PAPER_DIR)/$(PAPER_TEX).new"; \
			exit 1; \
		)

# Split outputs (main manuscript only, SI only)
MAIN_MD ?= paper.md
MAIN_TEX ?= main.tex
MAIN_PDF ?= main.pdf
SUPP_MD ?= supplement.md
SUPP_TEX ?= supplement.tex
SUPP_PDF ?= supplement.pdf

.PHONY: paper-all main-pdf main-tex supplement-pdf supplement-tex

paper-all: $(PAPER_DIR)/$(PAPER_PDF) $(PAPER_DIR)/$(PAPER_TEX) $(PAPER_DIR)/$(MAIN_PDF) $(PAPER_DIR)/$(MAIN_TEX) $(PAPER_DIR)/$(SUPP_PDF) $(PAPER_DIR)/$(SUPP_TEX)
	@echo "Built: $(PAPER_DIR)/$(PAPER_PDF), $(PAPER_DIR)/$(MAIN_PDF), $(PAPER_DIR)/$(SUPP_PDF)"

main-pdf: $(PAPER_DIR)/$(MAIN_PDF)
main-tex: $(PAPER_DIR)/$(MAIN_TEX)
supplement-pdf: $(PAPER_DIR)/$(SUPP_PDF)
supplement-tex: $(PAPER_DIR)/$(SUPP_TEX)

$(PAPER_DIR)/$(MAIN_TEX): $(PAPER_DIR)/$(MAIN_MD) $(PAPER_BIB_PATH) $(PAPER_DIR)/$(PAPER_CSL) $(PAPER_DIR)/header.tex $(wildcard $(PAPER_DIR)/figures/*)
	@echo "Building LaTeX: $(PAPER_DIR)/$(MAIN_MD) -> $(PAPER_DIR)/$(MAIN_TEX)"
	@cd $(PAPER_DIR) && \
		if grep -n -E '\\\\n\\+|<<<<<<<|=======|>>>>>>>' "$(MAIN_MD)"; then \
			echo ""; \
			echo "ERROR: Found suspicious merge/diff artifacts in $(MAIN_MD) (e.g., literal '\\n+' or conflict markers)."; \
			exit 1; \
		fi
	@cd $(PAPER_DIR) && \
		pandoc $(MAIN_MD) \
			--to=latex \
			--filter pandoc-crossref \
			--lua-filter pagebreak-tables.lua \
			--metadata-file pandoc-crossref.yaml \
			--citeproc \
			-V geometry:$(PAPER_PDF_GEOMETRY) \
			-V mainfont="$(PAPER_PDF_MAINFONT)" \
			-V mathfont="$(PAPER_PDF_MATHFONT)" \
			-H header.tex \
			--bibliography=$(PAPER_BIB) \
			--csl=$(PAPER_CSL) \
			-o $(MAIN_TEX).new
	@cd $(PAPER_DIR) && \
		mv -f $(MAIN_TEX).new $(MAIN_TEX) 2>/dev/null || ( \
			echo "ERROR: could not overwrite $(PAPER_DIR)/$(MAIN_TEX) (is it open?)."; \
			echo "Leaving: $(PAPER_DIR)/$(MAIN_TEX).new"; \
			exit 1; \
		)

$(PAPER_DIR)/$(SUPP_TEX): $(PAPER_DIR)/$(SUPP_MD) $(PAPER_BIB_PATH) $(PAPER_DIR)/$(PAPER_CSL) $(PAPER_DIR)/header.tex $(wildcard $(PAPER_DIR)/figures/*)
	@echo "Building LaTeX: $(PAPER_DIR)/$(SUPP_MD) -> $(PAPER_DIR)/$(SUPP_TEX)"
	@cd $(PAPER_DIR) && \
		if grep -n -E '\\\\n\\+|<<<<<<<|=======|>>>>>>>' "$(SUPP_MD)"; then \
			echo ""; \
			echo "ERROR: Found suspicious merge/diff artifacts in $(SUPP_MD) (e.g., literal '\\n+' or conflict markers)."; \
			exit 1; \
		fi
	@cd $(PAPER_DIR) && \
		pandoc $(SUPP_MD) \
			--to=latex \
			--filter pandoc-crossref \
			--lua-filter pagebreak-tables.lua \
			--metadata-file pandoc-crossref.yaml \
			--citeproc \
			-V geometry:$(PAPER_PDF_GEOMETRY) \
			-V mainfont="$(PAPER_PDF_MAINFONT)" \
			-V mathfont="$(PAPER_PDF_MATHFONT)" \
			-H header.tex \
			--bibliography=$(PAPER_BIB) \
			--csl=$(PAPER_CSL) \
			-o $(SUPP_TEX).new
	@cd $(PAPER_DIR) && \
		mv -f $(SUPP_TEX).new $(SUPP_TEX) 2>/dev/null || ( \
			echo "ERROR: could not overwrite $(PAPER_DIR)/$(SUPP_TEX) (is it open?)."; \
			echo "Leaving: $(PAPER_DIR)/$(SUPP_TEX).new"; \
			exit 1; \
		)

$(PAPER_DIR)/$(MAIN_PDF): $(PAPER_DIR)/$(MAIN_TEX)
	@echo "Building PDF: $(PAPER_DIR)/$(MAIN_TEX) -> $(PAPER_DIR)/$(MAIN_PDF)"
	@cd $(PAPER_DIR) && \
		if ! command -v "$(PAPER_PDF_ENGINE)" >/dev/null 2>&1; then \
			echo "ERROR: PDF engine '$(PAPER_PDF_ENGINE)' not found. Install it (default expects xelatex) or override PAPER_PDF_ENGINE=<engine>."; \
			exit 1; \
		fi; \
		"$(PAPER_PDF_ENGINE)" -interaction=nonstopmode -halt-on-error -shell-escape "$(MAIN_TEX)" >/dev/null; \
		"$(PAPER_PDF_ENGINE)" -interaction=nonstopmode -halt-on-error -shell-escape "$(MAIN_TEX)" >/dev/null

$(PAPER_DIR)/$(SUPP_PDF): $(PAPER_DIR)/$(SUPP_TEX)
	@echo "Building PDF: $(PAPER_DIR)/$(SUPP_TEX) -> $(PAPER_DIR)/$(SUPP_PDF)"
	@cd $(PAPER_DIR) && \
		if ! command -v "$(PAPER_PDF_ENGINE)" >/dev/null 2>&1; then \
			echo "ERROR: PDF engine '$(PAPER_PDF_ENGINE)' not found. Install it (default expects xelatex) or override PAPER_PDF_ENGINE=<engine>."; \
			exit 1; \
		fi; \
		"$(PAPER_PDF_ENGINE)" -interaction=nonstopmode -halt-on-error -shell-escape "$(SUPP_TEX)" >/dev/null; \
		"$(PAPER_PDF_ENGINE)" -interaction=nonstopmode -halt-on-error -shell-escape "$(SUPP_TEX)" >/dev/null

# Build PDF
#
# Important: LaTeX cross-references (e.g., Figure/Table refs) require at least
# two LaTeX passes to resolve. Pandoc's direct --to=pdf path can emit warnings
# about undefined references because it runs a single pass. We therefore build
# the PDF from the generated .tex and run the engine twice.
$(PAPER_DIR)/$(PAPER_PDF): $(PAPER_DIR)/$(PAPER_TEX)
	@echo "Building PDF: $(PAPER_DIR)/$(PAPER_TEX) -> $(PAPER_DIR)/$(PAPER_PDF)"
	@cd $(PAPER_DIR) && \
		if ! command -v "$(PAPER_PDF_ENGINE)" >/dev/null 2>&1; then \
			echo "ERROR: PDF engine '$(PAPER_PDF_ENGINE)' not found. Install it (default expects xelatex) or override PAPER_PDF_ENGINE=<engine>."; \
			exit 1; \
		fi; \
		"$(PAPER_PDF_ENGINE)" -interaction=nonstopmode -halt-on-error -shell-escape "$(PAPER_TEX)" >/dev/null; \
		"$(PAPER_PDF_ENGINE)" -interaction=nonstopmode -halt-on-error -shell-escape "$(PAPER_TEX)" >/dev/null; \
		if [ "$(PAPER_PDF)" != "paper.pdf" ]; then mv -f "paper.pdf" "$(PAPER_PDF)"; fi

# HVE simulator (not part of default all)
HVE:
	$(MAKE) -C $(VALIDATION_HVE_DIR) run DATASET=$(DATASET)

# ASMR (fixed-cohort) analysis from KCOR_CMR.xlsx (not part of default all)
ASMR:
	$(MAKE) -C $(VALIDATION_ASMR_DIR) run DATASET=$(DATASET)

# ICD-10 cause of death analysis (Czech2 dataset)
icd10: $(VENV_PYTHON)
	@echo "Running ICD-10 cause of death analysis..."
	cd $(CODE_DIR) && $(abspath $(VENV_PYTHON)) icd_analysis.py ../data/Czech2/data.csv ../data/Czech2/
	@echo "ICD-10 analysis complete!"

# ICD-10 population structural shift analysis (Czech2 dataset)
icd_population_shift: $(VENV_PYTHON)
	@echo "Running ICD-10 population structural shift analysis..."
	cd $(CODE_DIR) && $(abspath $(VENV_PYTHON)) icd_population_shift.py ../data/Czech2/data.csv ../data/Czech2/
	@echo "ICD-10 population shift analysis complete!"

# KCOR Mortality Analysis Pipeline (Czech2 dataset)
mortality:
	$(MAKE) -C $(CODE_DIR) mortality ENROLL_YEAR=$(ENROLL_YEAR) ENROLL_MONTH=$(ENROLL_MONTH) MAX_FU_MONTHS=$(MAX_FU_MONTHS) QUIET_MIN=$(QUIET_MIN) QUIET_MAX=$(QUIET_MAX)

mortality_sensitivity:
	$(MAKE) -C $(CODE_DIR) mortality_sensitivity

mortality_age:
	$(MAKE) -C $(CODE_DIR) mortality_age ENROLL_YEAR=$(ENROLL_YEAR) ENROLL_MONTH=$(ENROLL_MONTH) MAX_FU_MONTHS=$(MAX_FU_MONTHS) QUIET_MIN=$(QUIET_MIN) QUIET_MAX=$(QUIET_MAX)

mortality_stats:
	$(MAKE) -C $(CODE_DIR) mortality_stats

mortality_plots:
	$(MAKE) -C $(CODE_DIR) mortality_plots QUIET_MIN=$(QUIET_MIN) QUIET_MAX=$(QUIET_MAX)

# Run all mortality analyses (Czech2 dataset)
mortality_all:
	$(MAKE) -C $(CODE_DIR) mortality_all ENROLL_YEAR=$(ENROLL_YEAR) ENROLL_MONTH=$(ENROLL_MONTH) MAX_FU_MONTHS=$(MAX_FU_MONTHS) QUIET_MIN=$(QUIET_MIN) QUIET_MAX=$(QUIET_MAX)

# Help target
help:
	@echo "Available targets:"
	@echo "  install         - Create .venv virtual environment and install dependencies from requirements.txt"
	@echo "  KCOR_variable   - Build variable-cohort aggregation (code/)"
	@echo "  ts              - Build time series aggregation (code/)"
	@echo "  KCOR            - Run main KCOR pipeline (code/)"
	@echo "  CMR             - Run only CMR aggregation step (code/)"
	@echo "  CMR_from_krf    - Adapt KRF CSV to Czech-like and run CMR (code/)"
	@echo "  monte_carlo     - Run Monte Carlo bootstrap sampling (4 iterations by default, override with MC_ITERATIONS=N)"
	@echo "  convert         - Run dataset converter (data/$(DATASET)/)"
	@echo "  identifiability - Weekly incident booster identifiability emulation (writes $(IDENT_OUTDIR)/all_ages plus born_193x/ born_194x/ born_195x/)"
	@echo "  identifiability-falsify - Run multi-variant falsification suite (writes $(IDENT_FALSIFY_DIR))"
	@echo "  identifiability-falsify-selection - Run only the selection/eligibility suite variant"
	@echo "  validation      - Run DS-CMRR, Kaplan–Meier, and GLM validation"
	@echo "  km              - Run only Kaplan–Meier validation"
	@echo "  glm             - Run only GLM validation"
	@echo "  glm-compare     - Compare GLM outputs"
	@echo "  test            - Run negative-control and sensitivity tests (test/)"
	@echo "  sensitivity     - Run parameter sweep (test/sensitivity)"
	@echo "  sim_grid        - Run simulation grid for operating characteristics (test/sim_grid)"
	@echo "  quiet-window    - Run quiet-window scan (test/quiet_window/code/quiet_window_scan_theta_czech_2021_24.py)"
	@echo "  cox-bias        - Run Cox bias demonstration simulation (test/sim_grid)"
	@echo "  cox-bias-figures - Generate Cox bias demonstration figures"
	@echo "  copy-cox-bias-figures - Copy Cox bias figures to preprint directory"
	@echo "  skip-weeks      - Generate skip-weeks sensitivity figure"
	@echo "  cohort-size     - Run cohort size sensitivity analysis"
	@echo "  rollout         - Run rollout design-mimic simulation"
	@echo ""
	@echo "Sensitivity quick-test examples (optional overrides):"
	@echo "  make sensitivity DATASET=Czech SA_COHORTS=2022_06 SA_DOSE_PAIRS=1,0 SA_BASELINE_WEEKS=4 SA_QUIET_START_OFFSETS=0"
	@echo "  make sensitivity DATASET=Czech SA_BASELINE_WEEKS=2,8,1 SA_QUIET_START_OFFSETS=-12,12,4"
	@echo "  slope-test      - Run slope normalization test on booster_d0_slope.csv"
	@echo "  HVE             - Run Healthy Vaccinee Effect simulation (validation/HVE)"
	@echo "  ASMR            - Run ASMR analysis from KCOR_CMR.xlsx (validation/ASMR_analysis)"
	@echo "  icd10           - Run ICD-10 cause of death analysis (data/Czech2/)"
	@echo "  icd_population_shift - Run ICD-10 population structural shift analysis (data/Czech2/)"
	@echo "  paper           - Build documentation/preprint/paper.md -> paper.docx + paper.pdf (Pandoc + crossref + citeproc)"
	@echo "  paper-all       - Build documentation/preprint/{paper,main,supplement}.{pdf,tex}"
	@echo "  paper-docx      - Build only documentation/preprint/paper.docx"
	@echo "  paper-pdf       - Build only documentation/preprint/paper.pdf"
	@echo "  main-pdf        - Build only documentation/preprint/main.pdf"
	@echo "  supplement-pdf  - Build only documentation/preprint/supplement.pdf"
	@echo ""
	@echo "KCOR Mortality Analysis (Czech2 dataset):"
	@echo "  mortality       - Run basic KCOR mortality analysis pipeline"
	@echo "  mortality_sensitivity - Run sensitivity analysis (multiple configurations)"
	@echo "  mortality_age   - Run age-stratified analysis"
	@echo "  mortality_stats - Add statistical inference (CIs, p-values) to results"
	@echo "  mortality_plots - Create enhanced visualizations"
	@echo "  mortality_all   - Run ALL mortality analyses (recommended for complete analysis)"
	@echo ""
	@echo "  clean           - Clean outputs"
	@echo ""
	@echo "Variables:"
	@echo "  DATASET=<name>        - Dataset namespace (default: Czech)"
	@echo "  MC_ITERATIONS=<n>     - Number of Monte Carlo iterations (default: 4)"
	@echo "  MC_ENROLLMENT_DATE=<YYYY_WW> - (Monte Carlo) Enrollment cohort used for MC CMR + analysis (default: 2021_24)"
	@echo "  SA_COHORTS=<list>     - (Sensitivity) Restrict cohorts, e.g. 2022_06 or 2021_24,2022_06"
	@echo "  SA_DOSE_PAIRS=<pairs> - (Sensitivity) Restrict dose pairs, e.g. 1,0 or 1,0;2,0"
	@echo "  SA_BASELINE_WEEKS=<vals> - (Sensitivity) Baseline weeks list/range, e.g. 4,6,8 or 2,8,1"
	@echo "  SA_QUIET_START_OFFSETS=<vals> - (Sensitivity) Quiet-start offsets (weeks) list/range, e.g. -12,-8,-4,0,4,8,12 or -12,12,4"
	@echo "  ENROLL_YEAR=<year>    - Enrollment year for mortality analysis (default: 2021)"
	@echo "  ENROLL_MONTH=<month>  - Enrollment month 1-12 for mortality analysis (default: 7)"
	@echo "  MAX_FU_MONTHS=<n>     - Maximum follow-up months (default: 24)"
	@echo "  QUIET_MIN=<month>    - Quiet period start month (default: 3)"
	@echo "  QUIET_MAX=<month>     - Quiet period end month (default: 10)"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Create .venv virtual environment and install dependencies from requirements.txt"
	@echo "  make install-debian - Install dependencies using Debian packages (requires sudo)"
	@echo ""
	@echo "Identifiability falsification knobs (optional overrides):"
	@echo "  IDENT_OUTDIR=<path>                    - Base identifiability outdir (default: identifiability/$(DATASET)/booster)"
	@echo "  IDENT_FALSIFY_DIR=<path>               - Base falsification outdir (default: $(IDENT_OUTDIR)/falsification)"
	@echo "  IDENT_PLACEBO_START_WEEKS=<int>        - Future-booster placebo start (default: $(IDENT_PLACEBO_START_WEEKS))"
	@echo "  IDENT_PLACEBO_END_WEEKS=<int>          - Future-booster placebo end (default: $(IDENT_PLACEBO_END_WEEKS))"
	@echo "  IDENT_EVENTUAL_HORIZON_WEEKS=<int>     - Eventual-booster restriction horizon (default: $(IDENT_EVENTUAL_HORIZON_WEEKS))"
	@echo "  IDENT_EXCLUDE_RECENT_INFECTION_WEEKS=<int> - Recent infection exclusion lookback (default: $(IDENT_EXCLUDE_RECENT_INFECTION_WEEKS))"
	@echo "  IDENT_TSD2_BINS=\"min-max ...\"          - tsd2 bin sweep for falsify target (default: $(IDENT_TSD2_BINS))"
	@echo "  IDENT_TT_TSD2_MIN_DAYS=<int>           - Selection suite tsd2 min days (default: $(IDENT_TT_TSD2_MIN_DAYS))"
	@echo "  IDENT_TT_TSD2_MAX_DAYS=<int>           - Selection suite tsd2 max days (default: $(IDENT_TT_TSD2_MAX_DAYS))"
	@echo "  IDENT_TT_TREATED_WINDOW_WEEKS=<int>    - Selection suite incident treated window weeks (default: $(IDENT_TT_TREATED_WINDOW_WEEKS))"
	@echo "  IDENT_LEAD_MAX_WEEKS=<int>             - Selection suite lead max k (default: $(IDENT_LEAD_MAX_WEEKS))"
	@echo "  IDENT_PERM_REPS=<int>                  - Selection suite permutation reps (default: $(IDENT_PERM_REPS))"
	@echo "  IDENT_PERM_SEED=<int>                  - Selection suite permutation RNG seed (default: $(IDENT_PERM_SEED))"
	@echo ""
	@echo "Note: KCOR v5.1+ uses slope7 mode and no longer requires cvxpy (it was only needed for legacy quadratic mode)."
	@echo "      The 'make install' target automatically creates and uses .venv."
	@echo "      All Python commands run through the Makefile use the virtual environment automatically."

