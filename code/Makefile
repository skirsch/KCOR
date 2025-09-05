# Makefile for KCOR analysis pipeline

# Configuration - Edit these paths for different countries/datasets
COUNTRY = Czech
INPUT_DATA = ../../$(COUNTRY)/data/vax_24.csv
CMR_OUTPUT = ../data/$(COUNTRY)/KCOR_CMR.xlsx
KCOR_OUTPUT = ../data/$(COUNTRY)/KCORv4.xlsx

.PHONY: clean clean-all help create_dirs

# Default target - run complete KCOR pipeline
KCOR: $(KCOR_OUTPUT)
	@echo "KCOR analysis complete!"

# KCOR analysis depends on CMR output and KCORv4.py script
$(KCOR_OUTPUT): $(CMR_OUTPUT) KCORv4.py
	@echo "Running KCORv4 analysis..."
	python KCORv4.py $(CMR_OUTPUT) $(KCOR_OUTPUT)
	@echo "KCOR analysis complete!"

# CMR aggregation depends on input data and KCOR_CMR.py script
$(CMR_OUTPUT): $(INPUT_DATA) KCOR_CMR.py | create_dirs
	@echo "Running KCOR CMR aggregation..."
	python KCOR_CMR.py $(INPUT_DATA) $(CMR_OUTPUT)
	@echo "CMR aggregation complete!"

# Convenience target for CMR only
CMR: $(CMR_OUTPUT)

# Create output directories if they don't exist
create_dirs:
	@python -c "import os; os.makedirs('../data/$(COUNTRY)', exist_ok=True)"

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
	@echo "  clean     - Remove output files for current country ($(COUNTRY))"
	@echo "  clean-all - Remove all output files for all countries"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Current configuration:"
	@echo "  Country: $(COUNTRY)"
	@echo "  Input:   $(INPUT_DATA)"
	@echo "  CMR:     $(CMR_OUTPUT)"
	@echo "  KCOR:    $(KCOR_OUTPUT)"
