
ifndef RUSTC
	RUSTC = rustc
endif

OPENCL_SRC = \
	OpenCL.rc \
	CL.rs \
	hl.rs \
	error.rs \

.PHONY: all
all: libOpenCL test

.PHONY: libOpenCL
libOpenCL : $(OPENCL_SRC)
	$(RUSTC) -O --lib OpenCL.rc

test : libOpenCL test.rs
	$(RUSTC) -L . test.rs
