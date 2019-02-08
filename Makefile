MODULE_TOPDIR = ../..

PGM = r.learn.ml

ETCFILES = raster utils model_selection

include $(MODULE_TOPDIR)/include/Make/Script.make
include $(MODULE_TOPDIR)/include/Make/Python.make

default: script
