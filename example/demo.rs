extern crate debug;
extern crate opencl;

use opencl::mem::CLBuffer;

fn main()
{
    let ker = include_str!("demo.ocl");
    println!("ker {}", ker);

    let vec_a = vec![0i, 1, 2, -3, 4, 5, 6, 7];
    let vec_b = vec![-7i, -6, 5, -4, 0, -1, 2, 3];

    let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();

    println!("{:?}", device.name());

    let a: CLBuffer<int> = ctx.create_buffer(vec_a.len(), opencl::cl::CL_MEM_READ_ONLY);
    let b: CLBuffer<int> = ctx.create_buffer(vec_a.len(), opencl::cl::CL_MEM_READ_ONLY);
    let c: CLBuffer<int> = ctx.create_buffer(vec_a.len(), opencl::cl::CL_MEM_WRITE_ONLY);

    queue.write(&a, &vec_a.as_slice(), ());
    queue.write(&b, &vec_b.as_slice(), ());

    let program = ctx.create_program_from_source(ker);
    program.build(&device).ok().expect("Couldn't build program.");


    let kernel = program.create_kernel("vector_add");

    kernel.set_arg(0, &a);
    kernel.set_arg(1, &b);
    kernel.set_arg(2, &c);

    let event = queue.enqueue_async_kernel(&kernel, vec_a.len(), None, ());

    let vec_c: Vec<int> = queue.get(&c, &event);

    println!("  {}", vec_a);
    println!("+ {}", vec_b);
    println!("= {}", vec_c);
}
