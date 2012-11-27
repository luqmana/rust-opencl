RUSTC = rustc

OPENCL_SRC = \
	OpenCL.rc \
	CL.rs \
	hl.rs \

.PHONY: all
all: libOpenCL test test2

.PHONY: libOpenCL
libOpenCL : $(OPENCL_SRC)
	$(RUSTC) -O --lib OpenCL.rc

test : libOpenCL test.rs
	$(RUSTC) -L . test.rs

test2 : libOpenCL test2.rs
	$(RUSTC) -L . test2.rs

