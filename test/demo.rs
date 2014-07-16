#![allow(uppercase_variables)]

extern crate debug;
extern crate opencl;

use std::io::fs::File;
use std::io::Reader;
use std::str;

use opencl::mem::CLBuffer;

fn main()
{
    let ker = File::open(&std::path::Path::new("/pgrm/rust-opencl/test.ocl")).read_to_end().unwrap();
    let ker = str::from_utf8(ker.as_slice()).unwrap();

    let vec_a = &[0i, 1, 2, -3, 4, 5, 6, 7];
    let vec_b = &[-7i, -6, 5, -4, 0, -1, 2, 3];

    let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();

    println!("{:?}", device.name());

    let A: CLBuffer<int> = ctx.create_buffer(vec_a.len(), opencl::CL::CL_MEM_READ_ONLY);
    let B: CLBuffer<int> = ctx.create_buffer(vec_a.len(), opencl::CL::CL_MEM_READ_ONLY);
    let C: CLBuffer<int> = ctx.create_buffer(vec_a.len(), opencl::CL::CL_MEM_WRITE_ONLY);

    queue.write(&A, &vec_a, ());
    queue.write(&B, &vec_b, ());

    let program = ctx.create_program_from_source(ker);
    program.build(&device).ok().expect("Couldn't build program.");


    let kernel = program.create_kernel("vector_add");

    kernel.set_arg(0, &A);
    kernel.set_arg(1, &B);
    kernel.set_arg(2, &C);

    let event = queue.enqueue_async_kernel(&kernel, vec_a.len(), None, ());

    let vec_c: Vec<int> = queue.get(&C, &event);

    println!("  {}", vec_a);
    println!("+ {}", vec_b);
    println!("= {}", vec_c);
}
