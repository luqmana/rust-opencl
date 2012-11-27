RUSTC = rustc

.PHONY: all
all: libOpenCL test

.PHONY: libOpenCL
libOpenCL : CL.rs OpenCL.rc
	$(RUSTC) -O --lib OpenCL.rc

test : libOpenCL test.rs
	$(RUSTC) -L . test.rs
