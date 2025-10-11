ROOT := ../..
PY ?= python3
ifeq ($(OS),Windows_NT)
    PY := py -3
endif

INPUT ?= hamamatsu.csv.gz
OUTPUT ?= japan_krf.csv
# Sidecar is analysis metadata; keep distinct from converter CONFIG to avoid overwriting
SIDECAR ?= japan_krf.sidecar.yaml
OBS_END ?= 2024-12-31
BIRTH_BAND_YEARS ?= 5
REFERENCE_DATE ?= earliest_vdate
START_DATE ?=
END_DATE ?=
CONFIG ?=

.PHONY: convert help clean

convert:
	$(PY) convert.py $(if $(CONFIG),-c $(CONFIG)) -i $(INPUT) -o $(OUTPUT) --sidecar $(SIDECAR) --observation-end $(OBS_END) --birth-band-years $(BIRTH_BAND_YEARS) --reference-date $(REFERENCE_DATE) $(if $(START_DATE),--start-date $(START_DATE)) $(if $(END_DATE),--end-date $(END_DATE))

help:
	@echo "make convert [INPUT=hamamatsu.csv.gz OUTPUT=japan_krf.csv SIDECAR=japan_krf.sidecar.yaml OBS_END=YYYY-MM-DD BIRTH_BAND_YEARS=5 REFERENCE_DATE=earliest_vdate START_DATE=YYYY-MM-DD END_DATE=YYYY-MM-DD CONFIG=path.yaml]"

clean:
	rm -f $(OUTPUT) $(SIDECAR)


