extern crate opencl;

use opencl::{Program, Kernel, Buffer, PreferedType, MemoryAccess, EventList};

const KERNEL_SRC: &'static str = include_str!("histogram.ocl");

const DATA: [ u32; 121 ] = [
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 0, 0, 0, 0, 1, 0, 0, 0, 0, 9,
    9, 0, 0, 0, 1, 2, 1, 0, 0, 0, 9,
    9, 0, 0, 1, 2, 3, 2, 1, 0, 0, 9,
    9, 0, 1, 2, 2, 2, 2, 2, 1, 0, 9,
    9, 1, 1, 1, 1, 2, 1, 1, 1, 1, 9,
    9, 0, 0, 0, 1, 2, 1, 0, 0, 0, 9,
    9, 0, 0, 0, 1, 2, 1, 0, 0, 0, 9,
    9, 0, 0, 0, 1, 2, 1, 0, 0, 0, 9,
    9, 0, 0, 0, 1, 1, 1, 0, 0, 0, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
];

fn main() {
    let mut result = [ 0u32; 10 ];

    let (device, context, queue) = opencl::create_compute_context(false).unwrap();
    let program = Program::new(&context, KERNEL_SRC);
    program.build(&device).unwrap();

    let kernel = Kernel::new(&program, "histogram");
    let input  = Buffer::new(&context, &DATA[..], MemoryAccess::ReadOnly);
    let output = Buffer::new(&context, &result[..], MemoryAccess::WriteOnly);

    kernel.set_arg(0, &input);
    kernel.set_arg(1, &output);

    queue.enqueue_kernel(&kernel, DATA.len(), None, None);
    queue.read(&output, &mut result[..], None);

    println!("Histogram: {:?}", result);
}
