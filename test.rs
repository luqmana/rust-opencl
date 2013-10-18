extern mod OpenCL;

use std::sys;
use std::io;

fn main()
{
	let ker = io::read_whole_file_str(&Path::new("./test.ocl")).unwrap();

	let vec_a: &[int] = &[0, 1, 2, -3, 4, 5, 6, 7];
	let vec_b: &[int] = &[-7, -6, 5, -4, 0, -1, 2, 3];

	let ctx = OpenCL::hl::create_compute_context();

	println(fmt!("%?", ctx.device_name()));

	let A = ctx.create_buffer(vec_a.len() * sys::size_of_val(&vec_a[0]), OpenCL::CL::CL_MEM_READ_ONLY);
	let B = ctx.create_buffer(vec_a.len() * sys::size_of_val(&vec_a[0]), OpenCL::CL::CL_MEM_READ_ONLY);
	let C = ctx.create_buffer(vec_a.len() * sys::size_of_val(&vec_a[0]), OpenCL::CL::CL_MEM_WRITE_ONLY);

	A.write(ctx, vec_a);
	B.write(ctx, vec_b);

	let program = ctx.create_program_from_source(ker);

	program.build(ctx.device);

	let kernel = program.create_kernel("vector_add");

	println(fmt!("%?", program));

	kernel.set_arg(0, &A);
	kernel.set_arg(1, &B);
	kernel.set_arg(2, &C);

	OpenCL::hl::enqueue_nd_range_kernel(&ctx.q, &kernel, 1, 0, 64, 64);

	let mut vec_c: ~[int];
	unsafe {
		vec_c = C.read(ctx);
	}

	println(fmt!("	%?", vec_a));
	println(fmt!("+	%?", vec_b));
	println(fmt!("=	%?", vec_c));
}
