extern mod OpenCL;
use OpenCL::CL::*;
use OpenCL::hl::*;


fn main() {
    let sz = 2;
    let vec_a = ~[1f, 1f];
    let vec_b = ~[1f, 2f];
    let vec_c = vec::to_mut(vec::from_elem(sz, 0f));

    let platforms = get_platforms();
    
    error!("%?", platforms);

    let devices = get_devices(platforms[0]);

    error!("%?", devices);

    let context = create_context(devices[0]);
    let cqueue = create_commandqueue(&context, devices[0]);
    
    let buf_a = create_buffer(&context, (sz * sys::size_of::<float>()) as int, CL_MEM_READ_ONLY);
    let buf_b = create_buffer(&context, (sz * sys::size_of::<float>()) as int, CL_MEM_READ_ONLY);
    let buf_c = create_buffer(&context, (sz * sys::size_of::<float>()) as int, CL_MEM_READ_ONLY);    

    enqueue_write_buffer(&cqueue, &buf_a, &vec_a);
    enqueue_write_buffer(&cqueue, &buf_b, &vec_b);

    let program = create_program_with_binary(&context, devices[0], &path::Path(@"rust-ptx.bin"));

    build_program(&program, devices[0]);

    let kernel = create_kernel(&program, @"_ZN9add_float17_d08d41c0c85935643_00E");

    set_kernel_arg(&kernel, 0, &ptr::null::<libc::c_void>());
    set_kernel_arg(&kernel, 1, &ptr::null::<libc::c_void>());
    set_kernel_arg(&kernel, 2, &buf_a);
    set_kernel_arg(&kernel, 3, &buf_b);
    set_kernel_arg(&kernel, 4, &buf_c);

    enqueue_nd_range_kernel(&cqueue, &kernel, 1, 0, sz as int, 2);

    enqueue_read_buffer(&cqueue, &buf_c, &vec_c);

    io::println(#fmt("=  %?", vec_c));

    io::println("Done executing vector add kernel!");
}
