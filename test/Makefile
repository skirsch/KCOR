ROOT := ..
DATASET ?= Czech

.PHONY: all negative_control sensitivity clean

all: negative_control sensitivity

negative_control:
	$(MAKE) -C $(ROOT)/test/negative_control all DATASET=$(DATASET)

sensitivity:
	$(MAKE) -C $(ROOT)/test/sensitivity all DATASET=$(DATASET)

clean:
	-$(MAKE) -C $(ROOT)/test/negative_control clean DATASET=$(DATASET)
	-$(MAKE) -C $(ROOT)/test/sensitivity clean DATASET=$(DATASET)


