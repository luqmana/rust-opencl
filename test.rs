extern mod OpenCL;
use OpenCL::CL::*;
use OpenCL::CL::ll::*;

fn main() unsafe {

    let ker = 
        ~"__kernel void vector_add(__global const int *A, __global const int *B, __global int *C) {
             int i = get_global_id(0);
             C[i] = A[i] + B[i];
         }";

    let sz = 8;
    let vec_a = ~[0, 1, 2, 3, 4, 5, 6, 7];
    let vec_b = ~[7, 6, 5, 4, 0, 1, 2, 3];

    let p_id: cl_platform_id = ptr::null();
    let device: cl_device_id = ptr::null();
    let np: cl_uint = 0;
    let nd: cl_uint = 0;
    let mut r: cl_int;    

    // Get the platform and device information
    clGetPlatformIDs(1, ptr::addr_of(&p_id), ptr::addr_of(&np));
    r = clGetDeviceIDs(p_id, CL_DEVICE_TYPE_GPU, 1, ptr::addr_of(&device), ptr::addr_of(&nd));
    if r != CL_SUCCESS {
        io::println(#fmt("Can't get device ID. [%?]", r));
    }

    // Create OpenCL context and command queue
    let ctx = clCreateContext(ptr::null(), nd, ptr::addr_of(&device), ptr::null(), ptr::null(), ptr::addr_of(&r));
    let cque = clCreateCommandQueue(ctx, device, 0, ptr::addr_of(&r));

    // Create memory buffers
    let A = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (sz * sys::size_of::<int>()) as libc::size_t, ptr::null(), ptr::addr_of(&r));
    let B = clCreateBuffer(ctx, CL_MEM_READ_ONLY, (sz * sys::size_of::<int>()) as libc::size_t, ptr::null(), ptr::addr_of(&r));
    let C = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (sz * sys::size_of::<int>()) as libc::size_t, ptr::null(), ptr::addr_of(&r));

    // Copy lists into memory buffers
    clEnqueueWriteBuffer(cque, A, CL_TRUE, 0, (sz * sys::size_of::<int>()) as libc::size_t,
                             vec::raw::to_ptr(vec_a) as *libc::c_void, 0, ptr::null(), ptr::null());
    clEnqueueWriteBuffer(cque, B, CL_TRUE, 0, (sz * sys::size_of::<int>()) as libc::size_t,
                             vec::raw::to_ptr(vec_b) as *libc::c_void, 0, ptr::null(), ptr::null());

    // Create a program from the kernel and build it
    let bytes = str::to_bytes(ker);
    let prog = clCreateProgramWithSource(ctx, 1, ptr::addr_of(&(vec::raw::to_ptr(bytes) as *libc::c_char)),
                                             ptr::addr_of(&(vec::len(bytes) as libc::size_t)), ptr::addr_of(&r));
    r = clBuildProgram(prog, nd, ptr::addr_of(&device), ptr::null(), ptr::null(), ptr::null());
    if r != CL_SUCCESS {
        io::println(#fmt("Unable to build program [%?].", r));
    }

    // Create the OpenCL kernel
    let bytes = str::to_bytes(@"vector_add");
    let kernel = clCreateKernel(prog, vec::raw::to_ptr(bytes) as *libc::c_char, ptr::addr_of(&r));
    if r != CL_SUCCESS {
        io::println(#fmt("Unable to create kernel [%?].", r));
    }

    // Set the arguments of the kernel
    clSetKernelArg(kernel, 0, sys::size_of::<cl_mem>() as libc::size_t, ptr::addr_of(&A) as *libc::c_void);
    clSetKernelArg(kernel, 1, sys::size_of::<cl_mem>() as libc::size_t, ptr::addr_of(&B) as *libc::c_void);
    clSetKernelArg(kernel, 2, sys::size_of::<cl_mem>() as libc::size_t, ptr::addr_of(&C) as *libc::c_void);

    let global_item_size: libc::size_t = (sz * sys::size_of::<int>()) as libc::size_t;
    let local_item_size: libc::size_t = 64;

    // Execute the OpenCL kernel on the list
    clEnqueueNDRangeKernel(cque, kernel, 1, ptr::null(), ptr::addr_of(&global_item_size),
                               ptr::addr_of(&local_item_size), 0, ptr::null(), ptr::null());

    // Now let's read back the new list from the device
    let buf = libc::malloc((sz * sys::size_of::<int>()) as libc::size_t);
    clEnqueueReadBuffer(cque, C, CL_TRUE, 0, (sz * sys::size_of::<int>()) as libc::size_t,
                            buf, 0, ptr::null(), ptr::null());
    let vec_c = vec::from_buf(buf as *int, sz);
    libc::free(buf);

    io::println(#fmt("   %?", vec_a));
    io::println(#fmt("+  %?", vec_b));
    io::println(#fmt("=  %?", vec_c));

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(prog);
    clReleaseMemObject(C);
    clReleaseMemObject(B);
    clReleaseMemObject(A);
    clReleaseCommandQueue(cque);
    clReleaseContext(ctx);

}
