extern mod OpenCL;
use OpenCL::CL::*;
use OpenCL::hl::*;

fn main() {
    let sz = 8;
    let vec_a = ~[0, 1, 2, 3, 4, 5, 6, 7];

    let platforms = get_platforms();
    
    error!("%?", platforms);

    let devices = get_devices(platforms[0]);

    error!("%?", devices);

    let context = create_context(devices[0]);
    let cqueue = create_commandqueue(&context, devices[0]);
    let buf_a = create_buffer(&context, sz * sys::size_of::<int>(), CL_MEM_READ_ONLY);
    enqueue_write_buffer(&cqueue, &buf_a, vec_a);
}
