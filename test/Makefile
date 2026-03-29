ROOT := ..
DATASET ?= Czech

.PHONY: all negative_control sensitivity theta0 alpha clean help

all: negative_control sensitivity

help:
	@echo "Available targets:"
	@echo "  all              Run negative_control and sensitivity tests"
	@echo "  negative_control Run negative_control tests"
	@echo "  sensitivity      Run sensitivity tests"
	@echo "  theta0           Run theta0 ID test script"
	@echo "  alpha            Run alpha estimation sandbox"
	@echo "  clean            Clean generated test artifacts"

negative_control:
	$(MAKE) -C $(ROOT)/test/negative_control all DATASET=$(DATASET)

sensitivity:
	$(MAKE) -C $(ROOT)/test/sensitivity all DATASET=$(DATASET)

theta0:
	python $(ROOT)/test/theta0/theta0_nwave.py --config $(ROOT)/test/theta0/params_nwave.yaml --outdir $(ROOT)/test/theta0/out/

alpha:
	$(MAKE) -C $(ROOT)/test/alpha all DATASET=$(DATASET)

clean:
	-$(MAKE) -C $(ROOT)/test/negative_control clean DATASET=$(DATASET)
	-$(MAKE) -C $(ROOT)/test/sensitivity clean DATASET=$(DATASET)


