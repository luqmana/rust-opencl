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

pub fn create_commandqueue(ctx: &Context, device: Device) -> CommandQueue {
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
pub fn create_buffer(ctx: &Context, size: int, flags: cl_mem_flags) -> Buffer {
    let mut errcode = 0;

    let buffer = clCreateBuffer(ctx.ctx, flags, size as libc::size_t, ptr::null(), 
                                ptr::addr_of(&errcode));

    if errcode != CL_SUCCESS {
        fail ~"Failed to create buffer!"
    }

    Buffer { buffer: buffer, size: size }
}

pub fn enqueue_write_buffer(cqueue: CommandQueue, buf: Buffer, host_vector: ~[int]) unsafe {
    let ret = clEnqueueWriteBuffer(cqueue.cqueue, buf.buffer, CL_TRUE, 0, 
                                   buf.size as libc::size_t, 
                                   vec::raw::to_ptr(host_vector) as *libc::c_void,
                                   0, ptr::null(), ptr::null());
    if ret != CL_SUCCESS {
        fail ~"Failed to enqueue write buffer!"
    }
}

struct Program {
    prg: cl_program,

    drop {
        clReleaseProgram(self.prg);
    }
}

// TODO: Support multiple devices
pub fn create_program_with_binary(ctx: Context, device: Device, binary_path: & Path){
    let mut errcode = 0;
    let binary = io::read_whole_file_str(binary_path).get();
    let program = do str::as_c_str(binary) |kernel_binary| {
        clCreateProgramWithBinary(ctx.ctx, 1, ptr::addr_of(&device.id), 
                                  ptr::addr_of(&(binary.leb() + 1)) as *libc::size_t, 
                                  ptr::addr_of(&kernel_binary) as **libc::cuchar,
                                  ptr::null(),
                                  ptr::addr_of(&errcode))
    };
    
    if errcode != CL_SUCCESS {
        fail ~"Failed to create open cl program with binary!"
    }

    Program { prg: program }
}

