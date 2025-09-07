ROOT := ..

.PHONY: all negative_control sensitivity clean

all: negative_control sensitivity

negative_control:
	$(MAKE) -C $(ROOT)/test/negative_control all

sensitivity:
	$(MAKE) -C $(ROOT)/test/sensitivity all

clean:
	-$(MAKE) -C $(ROOT)/test/negative_control clean
	-$(MAKE) -C $(ROOT)/test/sensitivity clean


