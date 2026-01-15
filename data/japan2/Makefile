ROOT := ../..
PY ?= python3
ifeq ($(OS),Windows_NT)
    PY := py -3
endif

INPUT ?= source
OUTPUT ?= krf.csv  # default output file name so that the CMR_from_krf.py script works
# Sidecar is analysis metadata; keep distinct from converter CONFIG to avoid overwriting
SIDECAR ?= japan2_krf.sidecar.yaml
OBS_END ?= 2025-12-31
BIRTH_BAND_YEARS ?= 5
START_DATE ?=
END_DATE ?=
CONFIG ?= japan2.yaml

.PHONY: convert help clean

# convert target depends on convert.py and optionally config file
# Note: Input files are checked dynamically by convert.py, so we can't easily list them as dependencies
# However, we can make krf.csv depend on convert.py so it rebuilds when the converter changes
$(OUTPUT): convert.py $(if $(CONFIG),$(CONFIG))
	$(PY) convert.py $(if $(CONFIG),-c $(CONFIG)) -i $(INPUT) -o $(OUTPUT) --sidecar $(SIDECAR) --observation-end $(OBS_END) --birth-band-years $(BIRTH_BAND_YEARS) $(if $(START_DATE),--start-date $(START_DATE)) $(if $(END_DATE),--end-date $(END_DATE))

# Keep convert as a phony target for convenience
convert: $(OUTPUT)

help:
	@echo "make convert [INPUT=source OUTPUT=krf.csv SIDECAR=japan2_krf.sidecar.yaml OBS_END=YYYY-MM-DD BIRTH_BAND_YEARS=5 START_DATE=YYYY-MM-DD END_DATE=YYYY-MM-DD CONFIG=path.yaml]"

clean:
	rm -f $(OUTPUT) $(SIDECAR)

