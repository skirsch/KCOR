# Makefile for KCOR analysis pipeline

# Configuration - Edit these paths for different countries/datasets
COUNTRY = Czech
# Raw CSV (uncompressed) and compressed asset stored in repo
RAW_CSV          = ../data/$(COUNTRY)/Czech-NZIP-NR-26-30-release-2024-01.csv
RAW_CSV_GZ       = ../data/$(COUNTRY)/Otevrena-data-NR-26-30-COVID-19-prehled-populace-2024-01.csv.gzip
INPUT_DATA       = $(RAW_CSV)
CMR_OUTPUT       = ../data/$(COUNTRY)/KCOR_CMR.xlsx
KCOR_OUTPUT      = ../data/$(COUNTRY)/KCORv4.xlsx

.PHONY: clean clean-all help sensitivity

# Default target - run complete KCOR pipeline
KCOR: $(KCOR_OUTPUT)

# KCOR analysis depends on CMR output and KCOR.py script
$(KCOR_OUTPUT): $(CMR_OUTPUT) KCOR.py
	@echo "Running KCORv4 analysis..."
	python KCOR.py $(CMR_OUTPUT) $(KCOR_OUTPUT) "Primary Analysis" KCOR_summary.log
	@echo "KCOR analysis complete!"

# CMR aggregation depends on input data and KCOR_CMR.py script
$(CMR_OUTPUT): $(INPUT_DATA) KCOR_CMR.py | ../data/$(COUNTRY)
	@echo "Running KCOR CMR aggregation..."
	python KCOR_CMR.py $(INPUT_DATA) $(CMR_OUTPUT)
	@echo "CMR aggregation complete!"

# Convenience target for CMR only
CMR: $(CMR_OUTPUT)

# Ensure output directory exists (real directory target)
../data/$(COUNTRY):
	@python -c "import os; os.makedirs('$@', exist_ok=True)"

# Rule to inflate the raw CSV from the compressed LFS asset
$(RAW_CSV): $(RAW_CSV_GZ) | ../data/$(COUNTRY)
	@echo "Decompressing $(RAW_CSV_GZ) -> $(RAW_CSV) ..."
	@gzip -dc "$<" > "$@"
	@echo "Done."

# Clean up all output files for current country
clean:
	@echo "Cleaning up output files for $(COUNTRY)..."
	@python -c "import os; [os.remove(f) for f in ['$(CMR_OUTPUT)', '$(KCOR_OUTPUT)'] if os.path.exists(f)]"
	@echo "Cleanup complete!"

# Clean up all output files for all countries
clean-all:
	@echo "Cleaning up all output files..."
	@python -c "import shutil, os; shutil.rmtree('../data') if os.path.exists('../data') else None"
	@echo "Cleanup complete!"

# Help target
help:
	@echo "Available targets:"
	@echo "  KCOR      - Run complete KCOR pipeline (CMR + analysis)"
	@echo "  CMR       - Run only CMR aggregation step"
	@echo "  sensitivity - Smoke test sensitivity plumbing (echo SA_* and run analysis)"
	@echo "  clean     - Remove output files for current country ($(COUNTRY))"
	@echo "  clean-all - Remove all output files for all countries"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Current configuration:"
	@echo "  Country: $(COUNTRY)"
	@echo "  Input:   $(INPUT_DATA)"
	@echo "  CMR:     $(CMR_OUTPUT)"
	@echo "  KCOR:    $(KCOR_OUTPUT)"

## Minimal sensitivity plumbing target (no Python changes yet)
# Echos SA_* variables and runs normal pipeline to a separate output file to verify wiring.
sensitivity: $(CMR_OUTPUT) KCOR.py | ../data/$(COUNTRY)
	@echo "[sensitivity] Environment:"
	@echo "  SENSITIVITY_ANALYSIS = $(SENSITIVITY_ANALYSIS)"
	@echo "  SA_SLOPE_START       = $(SA_SLOPE_START)"
	@echo "  SA_SLOPE_LENGTH      = $(SA_SLOPE_LENGTH)"
	@echo "  SA_YOB               = $(SA_YOB)"
	@echo "  SA_COHORTS           = $(SA_COHORTS)"
	@echo "  SA_DOSE_PAIRS        = $(SA_DOSE_PAIRS)"
	@echo "Running KCORv4 analysis (smoke) ..."
	SENSITIVITY_ANALYSIS=$(SENSITIVITY_ANALYSIS) SA_SLOPE_START="$(SA_SLOPE_START)" SA_SLOPE_LENGTH="$(SA_SLOPE_LENGTH)" SA_YOB="$(SA_YOB)" SA_COHORTS="$(SA_COHORTS)" SA_DOSE_PAIRS="$(SA_DOSE_PAIRS)" \
		python KCOR.py $(CMR_OUTPUT) ../data/$(COUNTRY)/KCORv4_sensitivity_smoke.xlsx "Sensitivity Analysis" KCOR_summary_SA.log
	@echo "Sensitivity plumbing check complete!"
