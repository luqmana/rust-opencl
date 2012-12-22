
ifndef RUSTC
	RUSTC = rustc
endif

OPENCL_SRC = \
	OpenCL.rc \
	CL.rs \
	error.rs \
	hl.rs \
	vector.rs \

.PHONY: all
all: libOpenCL test

.PHONY: libOpenCL
libOpenCL : $(OPENCL_SRC)
	$(RUSTC) -O --lib OpenCL.rc

.PHONY: check
check: opencl-test
	./opencl-test

test : libOpenCL test.rs
	$(RUSTC) -L . test.rs

opencl-test: $(OPENCL_SRC)
	$(RUSTC) -O --test --cfg test OpenCL.rc -o opencl-test
