// Higher level OpenCL wrappers.

use CL::*;
use CL::ll::*;
use error::check;

struct Platform {
    id: cl_platform_id
}

impl Platform {
    fn get_devices() -> ~[Device] {
        get_devices(self)
    }

    fn name() -> ~str {
        let mut size = 0;
        
        clGetPlatformInfo(self.id,
                          CL_PLATFORM_NAME,
                          0,
                          ptr::null(),
                          ptr::mut_addr_of(&size));
        
        let name = str::repeat(" ", size as uint);

        do str::as_buf(name) |p, len| {
            clGetPlatformInfo(self.id,
                              CL_PLATFORM_NAME,
                              len as libc::size_t,
                              p as *libc::c_void,
                              ptr::mut_addr_of(&size));
        };

        move name
    }
}

pub fn get_platforms() -> ~[Platform] {
    let mut num_platforms = 0;

    // TODO: Check result status
    clGetPlatformIDs(0, ptr::null(), ptr::addr_of(&num_platforms));

    let ids = vec::to_mut(vec::from_elem(num_platforms as uint,
                                         0 as cl_platform_id));

    do vec::as_imm_buf(ids) |ids, len| {
        clGetPlatformIDs(len as cl_uint,
                         ids, ptr::addr_of(&num_platforms))
    };

    do ids.map |id| { Platform { id: *id } }
}

struct Device {
    id: cl_device_id
}

pub fn get_devices(platform: Platform) -> ~[Device] {
    let mut num_devices = 0;

    clGetDeviceIDs(platform.id, CL_DEVICE_TYPE_GPU, 0, ptr::null(), 
                   ptr::addr_of(&num_devices));
    
    let ids = vec::to_mut(vec::from_elem(num_devices as uint, 
                                         0 as cl_device_id));
    do vec::as_imm_buf(ids) |ids, len| {
        clGetDeviceIDs(platform.id, CL_DEVICE_TYPE_GPU, len as cl_uint, 
                       ids, ptr::addr_of(&num_devices));
    };

    do ids.map |id| { Device {id: *id }}
}

struct Context {
    ctx: cl_context,

    drop {
        clReleaseContext(self.ctx);
    }
}

pub fn create_context(device: Device) -> Context {
    // TODO: Support for multiple devices
    let mut errcode = 0;

    // TODO: Proper error messages
    let ctx = clCreateContext(ptr::null(), 1, ptr::addr_of(&device.id),
                              ptr::null(), ptr::null(), ptr::addr_of(&errcode));

    check(errcode, "Failed to create opencl context!");

    Context { ctx: ctx }
}

struct CommandQueue {
    cqueue: cl_command_queue,
    device: Device,

    drop {
        clReleaseCommandQueue(self.cqueue);
    }
}

pub fn create_commandqueue(ctx: & Context, device: Device) -> CommandQueue {
    let mut errcode = 0;

    let cqueue = clCreateCommandQueue(ctx.ctx, device.id, 0, ptr::addr_of(&errcode));

    check(errcode, "Failed to create command queue!");

    CommandQueue {
        cqueue: cqueue,
        device: device
    }
}

struct Buffer {
    buffer: cl_mem,
    size: int,

    drop {
        clReleaseMemObject(self.buffer);
    }
}


// TODO: How to make this function cleaner and nice
pub fn create_buffer(ctx: & Context, size: int, flags: cl_mem_flags) -> Buffer {
    let mut errcode = 0;

    let buffer = clCreateBuffer(ctx.ctx, flags, size as libc::size_t, ptr::null(), 
                                ptr::addr_of(&errcode));

    check(errcode, "Failed to create buffer!");

    Buffer { buffer: buffer, size: size }
}

pub fn enqueue_write_buffer<T: KernelArg>(
    cqueue: & CommandQueue,
    buf: & Buffer,
    host_vector: T) unsafe
{
    let ret = clEnqueueWriteBuffer(
        cqueue.cqueue, buf.buffer, CL_TRUE, 0, 
        buf.size as libc::size_t, 
        host_vector.get_value(),
        0, ptr::null(), ptr::null());
    check(ret, "Failed to enqueue write buffer!");
}

pub fn enqueue_read_buffer<T: KernelArg>(
    cqueue: & CommandQueue,
    buf: & Buffer,
    host_vector: T) unsafe
{
    let ret = clEnqueueReadBuffer(
        cqueue.cqueue, buf.buffer, CL_TRUE, 0, 
        buf.size as libc::size_t,
        host_vector.get_value(), 0, ptr::null(), ptr::null());
    
    check(ret, "Failed to enqueue read buffer!");
}

struct Program {
    prg: cl_program,

    drop {
        clReleaseProgram(self.prg);
    }
}

pub impl Program {
    fn build(&self, device: Device) -> Result<(), ~str> {
        build_program(self, device)
    }
}

// TODO: Support multiple devices
pub fn create_program_with_binary(ctx: & Context, device: Device,
                                  binary_path: & Path) -> Program {
    let mut errcode = 0;
    let binary = match move io::read_whole_file_str(binary_path) {
        result::Ok(move binary) => move binary,
        Err(e) => fail fmt!("%?", e)
    };
    let program = do str::as_c_str(binary) |kernel_binary| {
        clCreateProgramWithBinary(ctx.ctx, 1, ptr::addr_of(&device.id), 
                                  ptr::addr_of(&(binary.len() + 1)) as *libc::size_t, 
                                  ptr::addr_of(&kernel_binary) as **libc::c_uchar,
                                  ptr::null(),
                                  ptr::addr_of(&errcode))
    };
    
    check(errcode, "Failed to create open cl program with binary!");

    Program { prg: program }
}

pub fn build_program(program: & Program, device: Device) -> Result<(), ~str> {
    let ret = clBuildProgram(program.prg, 1, ptr::addr_of(&device.id), 
                             ptr::null(), ptr::null(), ptr::null());
    if ret == CL_SUCCESS as cl_int {
        Ok(())
    }
    else {
        let mut size = 0 as libc::size_t;
        let status = clGetProgramBuildInfo(
            program.prg,
            device.id,
            CL_PROGRAM_BUILD_LOG,
            0,
            ptr::null(),
            ptr::addr_of(&size));
        check(status, "Could not get build log");

        let buf = vec::from_elem(size as uint, 0u8);
        do vec::as_imm_buf(buf) |p, len| {
            let status = clGetProgramBuildInfo(
                program.prg,
                device.id,
                CL_PROGRAM_BUILD_LOG,
                len as libc::size_t,
                p as *libc::c_void,
                ptr::null());
            check(status, "Could not get build log");

            unsafe { Err(str::raw::from_c_str(p as *libc::c_char)) }
        }
    }
}


struct Kernel {
    kernel: cl_kernel,

    drop {
        clReleaseKernel(self.kernel);
    }
}

impl Kernel {
    fn set_arg<T: KernelArg>(i: uint, x: &T) {
        set_kernel_arg(&self, i as CL::cl_uint, x)
    }
}

pub fn create_kernel(program: & Program, kernel: & str) -> Kernel unsafe{
    let mut errcode = 0;
    let bytes = str::to_bytes(kernel);
    let kernel = clCreateKernel(program.prg, vec::raw::to_ptr(bytes) as *libc::c_char, ptr::addr_of(&errcode));

    check(errcode, "Failed to create kernel!");

    Kernel { kernel: kernel }
}

pub trait KernelArg{
    fn get_value() -> *libc::c_void;
}

pub fn set_kernel_arg<T: KernelArg>(kernel: & Kernel, position: cl_uint, arg: & T) unsafe{
    // TODO: How to set different argument types. Currently only support cl_mem
    let ret = clSetKernelArg(kernel.kernel, position, 
                             sys::size_of::<cl_mem>() as libc::size_t,
                             arg.get_value());
    
    check(ret, "Failed to set kernel arg!");
} 

pub fn enqueue_nd_range_kernel(cqueue: & CommandQueue, kernel: & Kernel, work_dim: cl_uint,
                               _global_work_offset: int, global_work_size: int, 
                               local_work_size: int) unsafe{
    let ret = clEnqueueNDRangeKernel(cqueue.cqueue, kernel.kernel, work_dim, 
                                     // ptr::addr_of(&global_work_offset) as *libc::size_t,
                                     ptr::null(),
                                     ptr::addr_of(&global_work_size) as *libc::size_t,
                                     ptr::addr_of(&local_work_size) as *libc::size_t,
                                     0, ptr::null(), ptr::null());
    check(ret, "Failed to enqueue nd range kernel!");
}

pub impl<T> &~[T]: KernelArg {
    fn get_value() -> *libc::c_void {
        do vec::as_imm_buf(*self) |p, _len| {
            p as *libc::c_void
        }
    }
}

pub impl<T> &~[mut T]: KernelArg {
    fn get_value() -> *libc::c_void {
        do vec::as_imm_buf(*self) |p, _len| {
            p as *libc::c_void
        }
    }
}

pub impl Buffer: KernelArg{
    fn get_value() -> *libc::c_void{
        ptr::addr_of(&self.buffer) as *libc::c_void
    }
} 

pub impl *libc::c_void: KernelArg{
    fn get_value() -> *libc::c_void {
        self
    }
}

/**
This packages an OpenCL context, a device, and a command queue to
simplify handling all these structures.
*/
struct ComputeContext {
    ctx: Context,
    device: Device,
    q: CommandQueue
}

impl ComputeContext {
    fn create_program_from_source(src: &str) -> Program {
        do str::as_c_str(src) |src| {
            let mut status = CL_SUCCESS as cl_int;
            let program = clCreateProgramWithSource(
                self.ctx.ctx,
                1,
                ptr::addr_of(&src),
                ptr::null(),
                ptr::addr_of(&status));
            check(status, "Could not create program");

            Program { prg: program }
        }
    }
}

pub fn create_compute_context() -> @ComputeContext {
    // Enumerate all platforms until we find a device that works.

    for get_platforms().each |p| {
        let devices = p.get_devices();
        if devices.len() > 0 {
            let device = devices[0];
            let ctx = create_context(device);
            let q = create_commandqueue(&ctx, device);
            return @ComputeContext {
                ctx: move ctx,
                device: move device,
                q: move q
            }
        }
    }
    fail ~"Could not find an acceptable device."
}

#[cfg(test)]
mod test {
    #[test]
    fn program_build() {
        let src = "__kernel void test(__global int *i) { \
                       *i += 1; \
                   }";
        let ctx = create_compute_context();
        let prog = ctx.create_program_from_source(src);
        prog.build(ctx.device);
    }
}