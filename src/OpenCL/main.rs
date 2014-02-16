#[feature(link_args)];
extern crate OpenCL;

use std::io::fs::File;
use std::io::Reader;
use std::str;
use OpenCL::mem::CLBuffer;

#[nolink]
#[link_args = "-framework OpenCL"]
#[cfg(target_os = "macos")]
extern { }

#[nolink]
#[link_args = "-lOpenCL"]
#[cfg(target_os = "linux")]
extern { }

fn main()
{
    let ker = File::open(&std::path::Path::new("./test.ocl")).read_to_end();
    let ker = str::from_utf8(ker);

    let vec_a = &[0, 1, 2, -3, 4, 5, 6, 7];
    let vec_b = &[-7, -6, 5, -4, 0, -1, 2, 3];

    let (device, ctx, queue) = OpenCL::util::create_compute_context().unwrap();

    println!("{:?}", device.name());

    let A: CLBuffer<int> = ctx.create_buffer(vec_a.len(), OpenCL::CL::CL_MEM_READ_ONLY);
    let B: CLBuffer<int> = ctx.create_buffer(vec_a.len(), OpenCL::CL::CL_MEM_READ_ONLY);
    let C: CLBuffer<int> = ctx.create_buffer(vec_a.len(), OpenCL::CL::CL_MEM_WRITE_ONLY);

    queue.write(&A, &vec_a, ());
    queue.write(&B, &vec_b, ());

    let program = ctx.create_program_from_source(ker);

    program.build(&device);


    let kernel = program.create_kernel("vector_add");

    kernel.set_arg(0, &A);
    kernel.set_arg(1, &B);
    kernel.set_arg(2, &C);

    let event = queue.enqueue_async_kernel(&kernel, vec_a.len(), None, ());

    let vec_c: ~[int] = queue.get(&C, &event);

    println!("  {:?}", vec_a);
    println!("+ {:?}", vec_b);
    println!("= {:?}", vec_c);
}
