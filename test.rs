extern mod OpenCL;

use std::sys;

fn main()
{
	let ker =
			~"__kernel void vector_add(__global const long *A, __global const long *B, __global long *C) {
				int i = get_global_id(0);
				C[i] = A[i] + B[i];
			 }";

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

	let vec_c: ~[int] = C.read(ctx);

	println(fmt!("	%?", vec_a));
	println(fmt!("+	%?", vec_b));
	println(fmt!("=	%?", vec_c));
}
