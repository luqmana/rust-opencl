
ifndef RUSTC
	RUSTC = rustc
endif

ifndef TARGET_DIR
	TARGET_DIR = target/
endif

RUSTC_OPTS = -g -L $(TARGET_DIR) --out-dir $(TARGET_DIR)

OPENCL_SRC = \
	lib.rs \
	CL.rs \
	error.rs \
	hl.rs \
	util.rs \
	mem.rs \
	array.rs

.PHONY: all
all: lib


.PHONY: debug
debug: lib demo
	gdb --cd=./ target/demo


.PHONY: lib
lib: target_dir $(TARGET_DIR)libopencl.rlib

$(TARGET_DIR)libopencl.rlib: src/*
	rustc $(RUSTC_OPTS) src/lib.rs


.PHONY: target_dir
target_dir: $(TARGET_DIR)

$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)


.PHONY: demo
demo: target_dir $(TARGET_DIR)demo

$(TARGET_DIR)demo: $(TARGET_DIR)libopencl.rlib test/demo.rs
	rustc $(RUSTC_OPTS) test/demo.rs


.PHONY: check
check: lib
	rustc $(RUSTC_OPTS) --test test/test.rs
	$(TARGET_DIR)/test


.PHONY: clean
clean:
	rm -rf $(TARGET_DIR)

.PHONY: docs
docs:
	rustdoc src/lib.rs
