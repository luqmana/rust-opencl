
ifndef RUSTC
	RUSTC = rustc
endif

ifndef TARGET_DIR
	TARGET_DIR = ./build
endif

RUSTC_OPTS = -L$(TARGET_DIR) --out-dir $(TARGET_DIR) -O

OPENCL_SRC = \
	lib.rs \
	CL.rs \
	error.rs \
	hl.rs \
	util.rs \
	mem.rs \
	array.rs

.PHONY: all
all: libOpenCL

build:
	mkdir -p build

.PHONY: libOpenCL
libOpenCL : build
	rustc $(RUSTC_OPTS) src/OpenCL/lib.rs

.PHONY: check
check: libOpenCL
	rustc $(RUSTC_OPTS) --test src/OpenCL/test.rs
	./build/test

.PHONY: clean
clean:
	rm -rf build

.PHONY: docs
docs:
	rustdoc src/OpenCL/lib.rs
