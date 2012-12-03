extern mod OpenCL;
use OpenCL::CL::*;
use OpenCL::hl::*;

fn main() {
    let sz = 8;
    let vec_a = ~[0, 1, 2, 3, 4, 5, 6, 7];
    let vec_b = ~[1, 2, 3, 4, 5, 6, 7, 8];
    let vec_c = vec::to_mut(vec::from_elem(sz, 0));

    let platforms = get_platforms();
    
    error!("%?", platforms);

    let devices = get_devices(platforms[0]);

    error!("%?", devices);

    let context = create_context(devices[0]);
    let cqueue = create_commandqueue(&context, devices[0]);
    
    let buf_a = create_buffer(&context, (sz * sys::size_of::<int>()) as int, CL_MEM_READ_ONLY);
    let buf_b = create_buffer(&context, (sz * sys::size_of::<int>()) as int, CL_MEM_READ_ONLY);
    let buf_c = create_buffer(&context, (sz * sys::size_of::<int>()) as int, CL_MEM_READ_ONLY);    

    enqueue_write_buffer(&cqueue, &buf_a, &vec_a);
    enqueue_write_buffer(&cqueue, &buf_b, &vec_b);

    let program = create_program_with_binary(&context, devices[0], &path::Path(@"kernel.ptx"));

    build_program(&program, devices[0]);

    let kernel = create_kernel(&program, @"vector_add");

    set_kernel_arg(&kernel, 0, &buf_a);
    set_kernel_arg(&kernel, 1, &buf_b);
    set_kernel_arg(&kernel, 2, &buf_c);

    enqueue_nd_range_kernel(&cqueue, &kernel, 1, 0, (sz * sys::size_of::<int>()) as int, 64);

    enqueue_read_buffer(&cqueue, buf_c, vec_c);

    io::println(#fmt("=  %?", vec_c));

    io::println("Done executing vector add kernel!");
}
