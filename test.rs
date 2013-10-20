#[feature(globs)];

extern mod OpenCL;
use OpenCL::CL::*;
use OpenCL::CL::ll::*;
use std::ptr;
use std::mem;
use std::libc;
use std::vec;
use std::cast;

#[fixed_stack_segment]
fn main()
{
	unsafe
	{
		let ker =
				~"__kernel void vector_add(__global const long *A, __global const long *B, __global long *C) {
					int i = get_global_id(0);
					C[i] = A[i] + B[i];
				 }";

		let sz: uint = 8;
		let vec_a = ~[0, 1, 2, -3, 4, 5, 6, 7];
		let vec_b = ~[-7, -6, 5, -4, 0, -1, 2, 3];

		let mut r: cl_int = 1;

		let platforms = OpenCL::hl::get_platforms();

		let device = OpenCL::hl::get_devices(platforms[0], CL_DEVICE_TYPE_ALL)[0];
		println(format!("{:s}", device.name()));

		let ctx = OpenCL::hl::create_context(device);

		let cque = OpenCL::hl::create_command_queue(&ctx, device);
		//let cque = comque.cqueue;
		let A = OpenCL::hl::create_buffer(&ctx, sz*8, CL_MEM_READ_ONLY);
		let B = OpenCL::hl::create_buffer(&ctx, sz*8, CL_MEM_READ_ONLY);
		let C = OpenCL::hl::create_buffer(&ctx, sz*8, CL_MEM_WRITE_ONLY);

		// Copy lists into memory buffers
		clEnqueueWriteBuffer(cque.cqueue,
			A.buffer,
			CL_TRUE,
			0,
			(sz * mem::size_of::<int>()) as libc::size_t,
			vec::raw::to_ptr(vec_a) as *libc::c_void,
			0,
			ptr::null(),
			ptr::null());

		clEnqueueWriteBuffer(cque.cqueue,
			B.buffer,
			CL_TRUE,
			0,
			(sz * mem::size_of::<int>()) as libc::size_t,
			vec::raw::to_ptr(vec_b) as *libc::c_void,
			0,
			ptr::null(),
			ptr::null());

		// Create a program from the kernel and build it
		do ker.as_imm_buf |bytes, len|
		{
			let prog = clCreateProgramWithSource(ctx.ctx,
						1,
						ptr::to_unsafe_ptr(&(bytes as *libc::c_char)),
						ptr::to_unsafe_ptr(&(len as libc::size_t)),
						ptr::to_unsafe_ptr(&r));

			r = clBuildProgram(prog,
				1,
				ptr::to_unsafe_ptr(&device.id),
				ptr::null(),
				cast::transmute(ptr::null::<&fn ()>()),
				ptr::null());

			if r != CL_SUCCESS as cl_int
			{ println(format!("Unable to build program [{:?}].", r)); }

			// Create the OpenCL kernel
			do "vector_add".as_imm_buf() |bytes, _|
			{
				let kernel = clCreateKernel(prog, bytes as *libc::c_char, ptr::to_unsafe_ptr(&r));
				if r != CL_SUCCESS as cl_int
				{ println(format!("Unable to create kernel [{:?}].", r)); }

				// Set the arguments of the kernel
				clSetKernelArg(kernel,
					0,
					mem::size_of::<cl_mem>() as libc::size_t,
					ptr::to_unsafe_ptr(&A.buffer)	 as *libc::c_void);

				clSetKernelArg(kernel,
					1,
					mem::size_of::<cl_mem>() as libc::size_t,
					ptr::to_unsafe_ptr(&B.buffer)	 as *libc::c_void);

				clSetKernelArg(kernel,
					2,
					mem::size_of::<cl_mem>() as libc::size_t,
					ptr::to_unsafe_ptr(&C.buffer)	 as *libc::c_void);

				let global_item_size: libc::size_t = (sz) as libc::size_t;
				let local_item_size:	libc::size_t = 8;

				// Execute the OpenCL kernel on the list
				clEnqueueNDRangeKernel(cque.cqueue,
					kernel,
					1,
					ptr::null(),
					ptr::to_unsafe_ptr(&global_item_size),
					ptr::to_unsafe_ptr(&local_item_size),
					0,
					ptr::null(),
					ptr::null());

				// Now let's read back the new list from the device
				let buf = libc::malloc((sz * mem::size_of::<int>()) as libc::size_t);

				clEnqueueReadBuffer(cque.cqueue,
					C.buffer,
					CL_TRUE,
					0,
					(sz * mem::size_of::<int>()) as libc::size_t,
					buf,
					0,
					ptr::null(),
					ptr::null());

				let vec_c = vec::from_buf(buf as *int, sz);

				libc::free(buf);

				println(format!("	{:?}", vec_a));
				println(format!("+	{:?}", vec_b));
				println(format!("=	{:?}", vec_c));

				// Cleanup
//				clReleaseKernel(kernel);
//				clReleaseProgram(prog);
			}
		}
	}
}
