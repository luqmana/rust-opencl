extern mod OpenCL;

use std::rt::io;
use std::rt::io::file;
use std::rt::io::Reader;
use std::str;

use OpenCL::hl::EventList;

fn main()
{
	let ker = file::open(&std::path::Path::new("./test.ocl"), io::Open, io::Read).read_to_end();
    let ker = str::from_utf8(ker);

	let vec_a: &[int] = &[0, 1, 2, -3, 4, 5, 6, 7];
	let vec_b: &[int] = &[-7, -6, 5, -4, 0, -1, 2, 3];

	let (_, device, ctx, queue) = OpenCL::util::create_compute_context().unwrap();

	println!("{:?}", device.name());

	let A : OpenCL::hl::CLBuffer<int> = ctx.create_buffer(vec_a.len(), OpenCL::CL::CL_MEM_READ_ONLY);
	let B : OpenCL::hl::CLBuffer<int> = ctx.create_buffer(vec_a.len(), OpenCL::CL::CL_MEM_READ_ONLY);
	let C : OpenCL::hl::CLBuffer<int> = ctx.create_buffer(vec_a.len(), OpenCL::CL::CL_MEM_WRITE_ONLY);

	queue.write_buffer(&A, 0, vec_a, ());
	queue.write_buffer(&B, 0, vec_b, ());

	let program = ctx.create_program_from_source(ker);

	program.build(device);

	let kernel = program.create_kernel("vector_add");

	kernel.set_arg(0, &A);
	kernel.set_arg(1, &B);
	kernel.set_arg(2, &C);

	queue.enqueue_async_kernel(&kernel, 8, None, ()).wait();

	let vec_c = queue.read(&C, ());

	println!("	{:?}", vec_a);
	println!("+	{:?}", vec_b);
	println!("=	{:?}", vec_c);
}
