# Makefile for KCOR analysis pipeline. This is not the root Makefile.

# Configuration - Dataset namespace (inherited from root Makefile)
DATASET ?= Czech
# Paths for inputs
RECORDS_CSV      = ../data/$(DATASET)/records.csv
# Optional KRF input produced by dataset converters
RECORDS_KRF      = ../data/$(DATASET)/krf.csv
# Raw Czech CSV (compressed asset) — used only for datasets that ship this file
RAW_CSV_GZ       = ../data/$(DATASET)/Otevrena-data-NR-26-30-COVID-19-prehled-populace-2024-01.csv.gzip
# Keep the same base name when unzipped (…csv.gzip -> …csv)
RAW_CSV          = $(RAW_CSV_GZ:.gzip=)
# Always target the standardized records.csv as input; provide a single rule to build it
INPUT_DATA       = $(RECORDS_CSV)
CMR_OUTPUT       = ../data/$(DATASET)/KCOR_CMR.xlsx
KCOR_OUTPUT      = ../data/$(DATASET)/KCOR.xlsx
VARIABLE_OUTPUT  = ../data/$(DATASET)/KCOR_variable.xlsx
TS_OUTPUT        = ../data/$(DATASET)/KCOR_ts.xlsx


.PHONY: clean clean-all help sensitivity

# Default target - run complete KCOR pipeline
KCOR: $(KCOR_OUTPUT)

# KCOR analysis depends on CMR output and KCOR.py script
$(KCOR_OUTPUT): $(CMR_OUTPUT) KCOR.py
	@echo "Running KCORv4 analysis..."
	python3 KCOR.py $(CMR_OUTPUT) $(KCOR_OUTPUT) "Primary Analysis" KCOR_summary.log
	@echo "KCOR analysis complete!"

# CMR aggregation depends on input data and KCOR_CMR.py script
TMP_CMR := /tmp/KCOR_CMR.$(DATASET).xlsx

$(CMR_OUTPUT): $(INPUT_DATA) KCOR_CMR.py | ../data/$(DATASET)
	@echo "Running KCOR CMR aggregation... (writing to $(TMP_CMR) then moving)"
	@rm -f $(TMP_CMR)
	python3 KCOR_CMR.py $(INPUT_DATA) $(TMP_CMR)
	@echo "Wrote temporary: $(TMP_CMR)"
	@ls -lh $(TMP_CMR)
	@# Guard: do not move a zero-byte temp file
	@test -s $(TMP_CMR) || (echo "ERROR: Temp CMR file is zero bytes. Skipping move and failing target." && exit 1)
	@echo "Moving $(TMP_CMR) -> $(CMR_OUTPUT)"
	@mv -f $(TMP_CMR) $(CMR_OUTPUT)
	@ls -lh $(CMR_OUTPUT)
	@echo "CMR aggregation complete!"

# Build standardized records.csv from available inputs (prefer krf.csv)
$(RECORDS_CSV): | ../data/$(DATASET)
	@set -e; \
	if [ -f "$(RECORDS_KRF)" ]; then \
	  echo "Adapting KRF CSV to Czech-like input..."; \
	  python3 krf_adapter.py $(RECORDS_KRF) $(RECORDS_CSV); \
	  echo "Wrote: $(RECORDS_CSV) from $(RECORDS_KRF)"; \
	elif [ -f "$(RAW_CSV)" ] || [ -f "$(RAW_CSV_GZ)" ]; then \
	  if [ -f "$(RAW_CSV_GZ)" ]; then \
	    echo "Decompressing $(RAW_CSV_GZ) -> $(RAW_CSV) ..."; \
	    gzip -dc "$(RAW_CSV_GZ)" > "$(RAW_CSV)"; \
	  fi; \
	  echo "Using raw CSV as standardized input (dataset ships Czech-format raw file)"; \
	  cp "$(RAW_CSV)" "$(RECORDS_CSV)"; \
	else \
	  echo "ERROR: No input found. Provide $(RECORDS_KRF) or $(RAW_CSV_GZ)."; \
	  exit 1; \
	fi

# If krf.csv is missing, build it via the dataset converter
$(RECORDS_KRF):
	@echo "Building KRF via dataset converter in data/$(DATASET)..."
	@$(MAKE) -C ../data/$(DATASET) convert

# Keep convenience target to run CMR on KRF explicitly
.PHONY: KCOR_from_krf
CMR_from_krf: | ../data/$(DATASET)
	@echo "Adapting KRF CSV to Czech-like input..."
	@python3 krf_adapter.py ../data/$(DATASET)/krf.csv ../data/$(DATASET)/records.csv
	@echo "Running KCOR CMR aggregation on adapted input..."
	$(MAKE) CMR DATASET=$(DATASET)

# Convenience target for CMR only
CMR: $(CMR_OUTPUT)

# Variable cohort aggregation (single-sheet; current dose per week)
KCOR_variable: $(VARIABLE_OUTPUT)

$(VARIABLE_OUTPUT): $(INPUT_DATA) KCOR_variable.py | ../data/$(DATASET)
	@echo "Running KCOR variable-cohort aggregation..."
	python3 KCOR_variable.py $(INPUT_DATA) $(VARIABLE_OUTPUT)
	@echo "KCOR_variable aggregation complete!"

# Time series aggregation (single-sheet; weeks after each dose independently)
ts: $(TS_OUTPUT)

KCOR_ts: $(TS_OUTPUT)

$(TS_OUTPUT): $(INPUT_DATA) KCOR_ts.py | ../data/$(DATASET)
	@echo "Running KCOR time series aggregation..."
	python3 KCOR_ts.py $(INPUT_DATA) $(TS_OUTPUT)
	@echo "KCOR time series aggregation complete!"

# Ensure output directory exists (real directory target)
../data/$(DATASET):
	@python3 -c "import os; os.makedirs('$@', exist_ok=True)"

# Raw CSV inflation handled inside the unified records.csv rule above

# Clean up all output files for current country
clean:
	@echo "Cleaning up output files for $(DATASET)..."
	@python3 -c "import os; [os.remove(f) for f in ['$(CMR_OUTPUT)', '$(KCOR_OUTPUT)'] if os.path.exists(f)]"
	@echo "Cleanup complete!"

# Clean up all output files for all countries
clean-all:
	@echo "Cleaning up all output files..."
	@python3 -c "import shutil, os; shutil.rmtree('../data') if os.path.exists('../data') else None"
	@echo "Cleanup complete!"

# Help target
help:
	@echo "Available targets:"
	@echo "  KCOR      - Run complete KCOR pipeline (CMR + analysis)"
	@echo "  CMR       - Run only CMR aggregation step"
	@echo "  CMR_from_krf - Run CMR aggregation step on KRF data; specify DATASET=japan to use Japan KRF data"
	@echo "  sensitivity - Smoke test sensitivity plumbing (echo SA_* and run analysis)"
	@echo "  clean     - Remove output files for current dataset ($(DATASET))"
	@echo "  clean-all - Remove all output files for all countries"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Current configuration:"
	@echo "  Dataset: $(DATASET)"
	@echo "  Input:   $(INPUT_DATA)"
	@echo "  CMR:     $(CMR_OUTPUT)"
	@echo "  KCOR:    $(KCOR_OUTPUT)"

## Minimal sensitivity plumbing target (kept for convenience)
# Echos SA_* variables and runs analysis to the unified SA artifact under test/sensitivity/out
sensitivity: $(CMR_OUTPUT) KCOR.py | ../data/$(DATASET)
	@echo "[sensitivity] Environment:"
	@echo "  SENSITIVITY_ANALYSIS = $(SENSITIVITY_ANALYSIS)"
	@echo "  SA_SLOPE_START       = $(SA_SLOPE_START)"
	@echo "  SA_SLOPE_LENGTH      = $(SA_SLOPE_LENGTH)"
	@echo "  SA_YOB               = $(SA_YOB)"
	@echo "  SA_COHORTS           = $(SA_COHORTS)"
	@echo "  SA_DOSE_PAIRS        = $(SA_DOSE_PAIRS)"
	@echo "Running KCOR sensitivity analysis ..."
	@python3 -c "import os; os.makedirs('../test/sensitivity/out', exist_ok=True)"
	SENSITIVITY_ANALYSIS=$(SENSITIVITY_ANALYSIS) SA_SLOPE_START="$(SA_SLOPE_START)" SA_SLOPE_LENGTH="$(SA_SLOPE_LENGTH)" SA_YOB="$(SA_YOB)" SA_COHORTS="$(SA_COHORTS)" SA_DOSE_PAIRS="$(SA_DOSE_PAIRS)" \
		python3 KCOR.py $(CMR_OUTPUT) ../test/sensitivity/out/KCOR_SA.xlsx "Sensitivity Analysis" KCOR_summary_SA.log
	@echo "Sensitivity run complete! Output: test/sensitivity/out/KCOR_SA.xlsx"
