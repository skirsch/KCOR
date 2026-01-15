ROOT := ../..
PY ?= python3
ifeq ($(OS),Windows_NT)
    PY := py -3
endif

INPUT ?= hamamatsu.csv.gz
OUTPUT ?= krf.csv  # default output file name so that the CMR_from_krf.py script works
# Sidecar is analysis metadata; keep distinct from converter CONFIG to avoid overwriting
SIDECAR ?= japan_krf.sidecar.yaml
OBS_END ?= 2025-12-31
BIRTH_BAND_YEARS ?= 5
REFERENCE_DATE ?= earliest_vdate
START_DATE ?=
END_DATE ?=
CONFIG ?= japan.yaml

.PHONY: convert help clean

$(OUTPUT): convert.py $(if $(CONFIG),$(CONFIG))
	$(PY) convert.py $(if $(CONFIG),-c $(CONFIG)) -i $(INPUT) -o $(OUTPUT) --sidecar $(SIDECAR) --observation-end $(OBS_END) --birth-band-years $(BIRTH_BAND_YEARS) --reference-date $(REFERENCE_DATE) $(if $(START_DATE),--start-date $(START_DATE)) $(if $(END_DATE),--end-date $(END_DATE))

convert: $(OUTPUT)

help:
	@echo "make convert [INPUT=hamamatsu.csv.gz OUTPUT=japan_krf.csv SIDECAR=japan_krf.sidecar.yaml OBS_END=YYYY-MM-DD BIRTH_BAND_YEARS=5 REFERENCE_DATE=earliest_vdate START_DATE=YYYY-MM-DD END_DATE=YYYY-MM-DD CONFIG=path.yaml]"

clean:
	rm -f $(OUTPUT) $(SIDECAR)


