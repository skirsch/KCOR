ROOT := ../..

.PHONY: all run clean

all: run

# Delegate to code/Makefile sensitivity target using root SA_* defaults
run:
	$(MAKE) -C $(ROOT)/code sensitivity

clean:
	# no local files to remove; sensitivity outputs are under data/Czech
	true


