# rust-opencl

OpenCL bindings and high-level interface for Rust.

## Installation

Add the following to your `Cargo.toml` file:

```rust
[dependencies] 
rust-opencl = "0.4"
```

## Example 

From 'examples/demo/main.rs':

```rust
extern crate opencl;

use opencl::mem::CLBuffer;
use std::fmt;

fn main()
{
    let ker = include_str!("demo.ocl");
    println!("ker {}", ker);

    let vec_a = vec![0isize, 1, 2, -3, 4, 5, 6, 7];
    let vec_b = vec![-7isize, -6, 5, -4, 0, -1, 2, 3];

    let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();

    println!("{}", device.name());

    let a: CLBuffer<isize> = ctx.create_buffer(vec_a.len(), opencl::cl::CL_MEM_READ_ONLY);
    let b: CLBuffer<isize> = ctx.create_buffer(vec_a.len(), opencl::cl::CL_MEM_READ_ONLY);
    let c: CLBuffer<isize> = ctx.create_buffer(vec_a.len(), opencl::cl::CL_MEM_WRITE_ONLY);

    queue.write(&a, &&vec_a[..], ());
    queue.write(&b, &&vec_b[..], ());

    let program = ctx.create_program_from_source(ker);
    program.build(&device).ok().expect("Couldn't build program.");


    let kernel = program.create_kernel("vector_add");

    kernel.set_arg(0, &a);
    kernel.set_arg(1, &b);
    kernel.set_arg(2, &c);

    let event = queue.enqueue_async_kernel(&kernel, vec_a.len(), None, ());

    let vec_c: Vec<isize> = queue.get(&c, &event);

    println!("  {}", string_from_slice(&vec_a[..]));
    println!("+ {}", string_from_slice(&vec_b[..]));
    println!("= {}", string_from_slice(&vec_c[..]));
}

fn string_from_slice<T: fmt::Display>(slice: &[T]) -> String {
    let mut st = String::from("[");
    let mut first = true;

    for i in slice.iter() {
        if !first {
            st.push_str(", ");
        }
        else {
            first = false;
        }
        st.push_str(&*i.to_string())
    }

    st.push_str("]");
    return st
}
```

With the kernel `demo.ocl`:
```opencl
__kernel void vector_add(__global const long *A, __global const long *B, __global long *C) {
	int i = get_global_id(0);
	C[i] = A[i] + B[i];
}
```
