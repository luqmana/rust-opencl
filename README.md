# rust-opencl

OpenCL bindings and high-level interface for Rust.

## Installation

Add the following to your `Cargo.toml` file:

```rust
[dependencies] 
rust-opencl = "0.5.0"
```

## Example 

From 'examples/demo/main.rs':

```rust
extern crate opencl;

use opencl::{CLBuffer, Program};

// The kernel sources.
const KERNEL_SRC: &'static str = include_str!("demo.ocl");

fn main()
{
    let vec_a = [ 0isize, 1, 2, -3, 4, 5, 6, 7 ];
    let vec_b = [ -7isize, -6, 5, -4, 0, -1, 2, 3 ];
    let mut vec_c = [ 0isize; 8 ];

    // Use a context for the first device of the first platform.
    // Profiling is disabled.
    let (device, ctx, queue) = opencl::create_compute_context(false).unwrap();

    // Create the pre-initialized buffer objects.
    let a = CLBuffer::<isize>::new(&ctx, &vec_a[..], opencl::cl::CL_MEM_READ_ONLY);
    let b = CLBuffer::<isize>::new(&ctx, &vec_b[..], opencl::cl::CL_MEM_READ_ONLY);
    let c = CLBuffer::<isize>::new_uninitialized(&ctx, 8, opencl::cl::CL_MEM_WRITE_ONLY);

    // Create and build the program.
    let program = Program::new(&ctx, KERNEL_SRC);
    program.build(&device).ok().expect("Couldn't build program.");

    // Retrieve the kernel.
    let kernel = program.create_kernel("vector_add");

    // Set the kernel arguments.
    kernel.set_arg(0, &a);
    kernel.set_arg(1, &b);
    kernel.set_arg(2, &c);

    // Run the kernel.
    let event = queue.enqueue_async_kernel(&kernel, vec_a.len(), None, ());

    // Synchronously read the result.
    queue.read(&c, &mut vec_c[..], &event);

    println!("  {:?}", vec_a);
    println!("+ {:?}", vec_b);
    println!("= {:?}", vec_c);
}
```

With the kernel `demo.ocl`:
```opencl
__kernel void vector_add(__global const long *A, __global const long *B, __global long *C) {
	int i = get_global_id(0);
	C[i] = A[i] + B[i];
}
```
