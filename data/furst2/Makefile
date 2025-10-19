ROOT := ../..
PY ?= python3
ifeq ($(OS),Windows_NT)
    PY := py -3
endif

# Defaults mirror japan's makefile
INPUT ?= .
OUTPUT ?= furst_krf.csv
SIDECAR ?= furst_krf.sidecar.yaml
OBS_END ?= 2025-12-31
BIRTH_BAND_YEARS ?= 1
REFERENCE_DATE ?= 2021-01-01
CONFIG ?=

.PHONY: convert help clean

convert:
	$(PY) convert.py $(if $(CONFIG),-c $(CONFIG)) -i $(INPUT) -o $(OUTPUT) --sidecar $(SIDECAR) --observation-end $(OBS_END) --birth-band-years $(BIRTH_BAND_YEARS) --reference-date $(REFERENCE_DATE)

help:
	@echo "make convert [INPUT=path_or_dir OUTPUT=furst_krf.csv SIDECAR=furst_krf.sidecar.yaml OBS_END=YYYY-MM-DD BIRTH_BAND_YEARS=1 REFERENCE_DATE=2021-01-01 CONFIG=path.yaml]"

clean:
	rm -f $(OUTPUT) $(SIDECAR)



