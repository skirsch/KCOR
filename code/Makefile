# Makefile for KCOR analysis

.PHONY: KCOR clean

KCOR:
	@echo "Running KCOR analysis..."
	python KCORv4.py ../../Czech/data/KCOR_output.xlsx KCOR_processed_REAL.xlsx
	@echo "KCOR analysis complete!"

clean:
	@echo "Cleaning up output files..."
	rm -f KCOR_processed_REAL.xlsx
	@echo "Cleanup complete!"

# Help target
help:
	@echo "Available targets:"
	@echo "  KCOR    - Run KCOR analysis on Czech data"
	@echo "  clean   - Remove output files"
	@echo "  help    - Show this help message"
