extern mod OpenCL;

use std::mem;
use std::rt::io;
use std::rt::io::file;
use std::rt::io::extensions::ReaderUtil;
use std::str;

fn main()
{
	let ker = file::open(&std::path::Path::new("./test.ocl"), io::Open, io::Read).read_to_end();
    let ker = str::from_utf8(ker);

	let vec_a: &[int] = &[0, 1, 2, -3, 4, 5, 6, 7];
	let vec_b: &[int] = &[-7, -6, 5, -4, 0, -1, 2, 3];

	let ctx = OpenCL::hl::create_compute_context();

	println!("{:?}", ctx.borrow().device_name());

	let A = ctx.borrow().create_buffer(vec_a.len() * mem::size_of_val(&vec_a[0]), OpenCL::CL::CL_MEM_READ_ONLY);
	let B = ctx.borrow().create_buffer(vec_a.len() * mem::size_of_val(&vec_a[0]), OpenCL::CL::CL_MEM_READ_ONLY);
	let C = ctx.borrow().create_buffer(vec_a.len() * mem::size_of_val(&vec_a[0]), OpenCL::CL::CL_MEM_WRITE_ONLY);

	A.write(ctx.clone(), vec_a);
	B.write(ctx.clone(), vec_b);

	let program = ctx.borrow().create_program_from_source(ker);

	program.build(ctx.borrow().device);

	let kernel = program.create_kernel("vector_add");

	println!("{:?}", program);

	kernel.set_arg(0, &A);
	kernel.set_arg(1, &B);
	kernel.set_arg(2, &C);

	OpenCL::hl::enqueue_nd_range_kernel(&ctx.borrow().q, &kernel, 1, 0, 64, 64);

	let mut vec_c: ~[int];
	unsafe {
		vec_c = C.read(ctx);
	}

	println!("	{:?}", vec_a);
	println!("+	{:?}", vec_b);
	println!("=	{:?}", vec_c);
}
