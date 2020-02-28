MODULE_TOPDIR = ../../..

include $(MODULE_TOPDIR)/include/Make/Other.make
include $(MODULE_TOPDIR)/include/Make/Python.make

MODULES = raster utils indexing statistics

ETCDIR = $(ETC)/r.learn.ml

PYFILES := $(patsubst %,$(ETCDIR)/%.py,$(MODULES))
PYCFILES := $(patsubst %,$(ETCDIR)/%.pyc,$(MODULES))

default: $(PYFILES) $(PYCFILES)

$(ETCDIR):
	$(MKDIR) $@

$(ETCDIR)/%: % | $(ETCDIR)
	$(INSTALL_DATA) $< $@

install:
	cp -r $(ETCDIR) $(INST_DIR)/etc
