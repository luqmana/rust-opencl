# rust-opencl

OpenCL bindings and high-level interface for Rust.

## Installation

Add the following to your `Cargo.toml` file:

```rust
[dependencies]
rust-opencl = "0.5.0"
```

## Example

From `examples/addition.rs`:

```rust
extern crate opencl;

use opencl::{Buffer, Program, Kernel, MemoryAccess};

// The kernel sources.
const KERNEL_SRC: &'static str = include_str!("addition.ocl");

fn main() {
    let vec_a = [ 0isize, 1, 2, -3, 4, 5, 6, 7 ];
    let vec_b = [ -7isize, -6, 5, -4, 0, -1, 2, 3 ];
    let mut vec_c = [ 0isize; 8 ];

    // Use a context for the first device of the first platform.
    // Profiling is disabled.
    let (device, ctx, queue) = opencl::create_compute_context(false).unwrap();

    // Create the pre-initialized buffer objects.
    let a = Buffer::new(&ctx, &vec_a[..], MemoryAccess::ReadOnly);
    let b = Buffer::new(&ctx, &vec_b[..], MemoryAccess::ReadOnly);
    let c = Buffer::<isize>::new_uninitialized(&ctx, 8, MemoryAccess::WriteOnly);

    // Create and build the program.
    let program = Program::new(&ctx, KERNEL_SRC);
    program.build(&device).ok().expect("Couldn't build program.");

    // Retrieve the kernel.
    let kernel = Kernel::new(&program, "vector_add");

    // Set the kernel arguments.
    kernel.set_arg(0, &a);
    kernel.set_arg(1, &b);
    kernel.set_arg(2, &c);

    // Run the kernel.
    let event = queue.enqueue_async_kernel(&kernel, vec_a.len(), None, None);

    // Synchronously read the result.
    queue.read(&c, &mut vec_c[..], &event);

    println!("  {:?}", vec_a);
    println!("+ {:?}", vec_b);
    println!("= {:?}", vec_c);
}
```

With the kernel `addition.ocl`:
```opencl
__kernel void vector_add(__global const long *A, __global const long *B, __global long *C) {
    int i = get_global_id(0);
    C[i] = A[i] + B[i];
}
```
