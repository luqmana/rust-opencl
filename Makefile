
ifndef RUSTC
	RUSTC = rustc
endif

ifndef RUSTPKG
	RUSTPKG = rustpkg
endif

OPENCL_SRC = \
	lib.rs \
	CL.rs \
	error.rs \
	hl.rs \
	util.rs \
	mem.rs \
	array.rs

.PHONY: all
all: libOpenCL opencl-test

.PHONY: libOpenCL
libOpenCL :
	$(RUSTPKG) build OpenCL

.PHONY: check
check:
	$(RUSTPKG) test OpenCL

.PHONY: clean
clean:
	$(RUSTPKG) clean OpenCL

.PHONY: docs
docs:
	rustdoc src/OpenCL/lib.rs

.PHONY: install
install:
	$(RUSTPKG) install OpenCL
