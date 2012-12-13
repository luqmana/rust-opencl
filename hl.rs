// Higher level OpenCL wrappers.

use CL::*;
use CL::ll::*;

struct Platform {
    id: cl_platform_id
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

    if errcode != CL_SUCCESS {
        fail ~"Failed to create opencl context!"
    }

    Context { ctx: ctx }
}

struct CommandQueue {
    cqueue: cl_command_queue,

    drop {
        clReleaseCommandQueue(self.cqueue);
    }
}

pub fn create_commandqueue(ctx: & Context, device: Device) -> CommandQueue {
    let mut errcode = 0;

    let cqueue = clCreateCommandQueue(ctx.ctx, device.id, 0, ptr::addr_of(&errcode));

    if errcode != CL_SUCCESS {
        fail ~"Failed to create command queue!"
    }

    CommandQueue { cqueue: cqueue }
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

    if errcode != CL_SUCCESS {
        fail ~"Failed to create buffer!"
    }

    Buffer { buffer: buffer, size: size }
}

pub fn enqueue_write_buffer(cqueue: & CommandQueue, buf: & Buffer, host_vector: & ~[float]) unsafe {
    let ret = clEnqueueWriteBuffer(cqueue.cqueue, buf.buffer, CL_TRUE, 0, 
                                   buf.size as libc::size_t, 
                                   vec::raw::to_ptr(*host_vector) as *libc::c_void,
                                   0, ptr::null(), ptr::null());
    if ret != CL_SUCCESS {
        fail ~"Failed to enqueue write buffer!"
    }
}

pub fn enqueue_read_buffer(cqueue: & CommandQueue, buf: & Buffer, host_vector: & ~[mut float]) unsafe {
    let mut ret = 0;
     do vec::as_imm_buf(*host_vector) |elements, _len| {
        ret = clEnqueueReadBuffer(cqueue.cqueue, buf.buffer, CL_TRUE, 0, 
                            buf.size as libc::size_t,
                            elements as *libc::c_void, 0, ptr::null(), ptr::null());
    };

    if ret != CL_SUCCESS {
        fail ~"Failed to enqueue read buffer!"
    }
}

struct Program {
    prg: cl_program,

    drop {
        clReleaseProgram(self.prg);
    }
}

// TODO: Support multiple devices
pub fn create_program_with_binary(ctx: & Context, device: Device, binary_path: & Path) -> Program{
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
    
    if errcode != CL_SUCCESS {
        fail ~"Failed to create open cl program with binary!"
    }

    Program { prg: program }
}

pub fn build_program(program: & Program, device: Device){
    let ret = clBuildProgram(program.prg, 1, ptr::addr_of(&device.id), 
                             ptr::null(), ptr::null(), ptr::null());
    if ret != CL_SUCCESS {
        let mut logv = ~"";
        for uint::range(0,1024*1204) |_i| {
            str::push_char(& mut logv, ' ');
        }
        do str::as_buf(logv) |logs, l| {
            let r = clGetProgramBuildInfo(program.prg, device.id, CL_PROGRAM_BUILD_LOG, l as libc::size_t, logs as *libc::c_void, ptr::null());
            if r != CL_SUCCESS{
                io::println(~"failed to get build info!");
            }
        }
        //io::println(logv);
        fail ~"Failure during program building!"
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

    if errcode != CL_SUCCESS {
        fail ~"Failed to create kernel!"
    }

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
    
    if ret != CL_SUCCESS {
        fail ~"Failed to set kernel arg!"
    }
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
    if ret != CL_SUCCESS {
        io::println((ret as int).to_str());
        fail ~"Failed to enqueue nd range kernel!"
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