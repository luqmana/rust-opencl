
ifndef RUSTC
	RUSTC = rustc
endif

OPENCL_SRC = \
	lib.rs \
	CL.rs \
	error.rs \
	hl.rs \
	util.rs \
	mem.rs

.PHONY: all
all: libOpenCL opencl-test

.PHONY: libOpenCL
libOpenCL : $(OPENCL_SRC)
	$(RUSTC) -O --lib lib.rs

.PHONY: check
check: opencl-test
	RUST_THREADS=1 ./opencl-test

opencl-test : libOpenCL test.rs
	$(RUSTC) -L . test.rs
	$(RUSTC) -O --test --cfg test lib.rs -o opencl-test
