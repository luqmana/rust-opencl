// Higher level OpenCL wrappers.

use CL::*;
use CL::ll::*;
use error::check;

struct Platform {
    id: cl_platform_id
}

enum DeviceType {
    CPU, GPU
}

fn convert_device_type(device: DeviceType) -> cl_device_type {
    match device {
        CPU => CL_DEVICE_TYPE_CPU,
        GPU => CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR
    }
}

impl Platform {
    fn get_devices() -> ~[Device] {
        get_devices(self, CL_DEVICE_TYPE_ALL)
    }

    fn get_devices_by_types(types: &[DeviceType]) -> ~[Device] {
        let mut dtype = 0;
        for types.each |&t| {
            dtype |= convert_device_type(t)
        }

        get_devices(self, dtype)
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

impl Device {
    fn name() -> ~str {
        let mut size = 0;
        let status = clGetDeviceInfo(
            self.id,
            CL_DEVICE_NAME,
            0,
            ptr::null(),
            ptr::addr_of(&size));
        check(status, "Could not determine name length");

        let buf = vec::from_elem(size as uint, 0);

        do vec::as_imm_buf(buf) |p, len| {
            let status = clGetDeviceInfo(
                self.id,
                CL_DEVICE_NAME,
                len as libc::size_t,
                p as *libc::c_void,
                ptr::null());
            check(status, "Could not get device name");
            
            unsafe { str::raw::from_c_str(p) }
        }
    }
}

pub fn get_devices(platform: Platform, dtype: cl_device_type) -> ~[Device] {
    let mut num_devices = 0;

    info!("Looking for devices matching %?", dtype);

    clGetDeviceIDs(platform.id, dtype, 0, ptr::null(), 
                   ptr::addr_of(&num_devices));
    
    let ids = vec::to_mut(vec::from_elem(num_devices as uint, 
                                         0 as cl_device_id));
    do vec::as_imm_buf(ids) |ids, len| {
        clGetDeviceIDs(platform.id, dtype, len as cl_uint, 
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

struct Program {
    prg: cl_program,

    context: Option<@ComputeContext>,

    drop {
        clReleaseProgram(self.prg);
    }
}

pub impl Program {
    fn build(&self, device: Device) -> Result<(), ~str> {
        build_program(self, device)
    }

    fn create_kernel(&self, name: &str) -> Kernel {
        let mut errcode = 0;
        do str::as_c_str(name) |bytes| {
            let kernel = clCreateKernel(
                self.prg,
                bytes,
                ptr::addr_of(&errcode));
        
            check(errcode, "Failed to create kernel!");
        
            Kernel { kernel: kernel, context: self.context }
        }
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

    Program {
        prg: program,
        context: None,
    }
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
    context: Option<@ComputeContext>,

    drop {
        clReleaseKernel(self.kernel);
    }
}

pub impl Kernel {
    fn set_arg<T: KernelArg>(i: uint, x: &T) {
        set_kernel_arg(&self, i as ::CL::cl_uint, x)
    }

    fn execute<I: KernelIndex>(&self, global: I, local: I) {
        match self.context {
            Some(ctx)
            => ctx.enqueue_async_kernel(self, global, local).wait(),
            
            None => fail ~"Kernel does not have an associated context."
        }
    }
}

pub fn create_kernel(program: & Program, kernel: & str) -> Kernel unsafe{
    let mut errcode = 0;
    let bytes = str::to_bytes(kernel);
    let kernel = clCreateKernel(program.prg, vec::raw::to_ptr(bytes) as *libc::c_char, ptr::addr_of(&errcode));

    check(errcode, "Failed to create kernel!");

    Kernel {
        kernel: kernel,
        context: None
    }
}

pub trait KernelArg {
    pure fn get_value(&self) -> (libc::size_t, *libc::c_void);
}

macro_rules! scalar_kernel_arg (
    ($t:ty) => (pub impl $t: KernelArg {
        pure fn get_value(&self) -> (libc::size_t, *libc::c_void) {
            (sys::size_of::<$t>() as libc::size_t,
             ptr::addr_of(self) as *libc::c_void)
        }
    })
)

scalar_kernel_arg!(int)
scalar_kernel_arg!(uint)

pub fn set_kernel_arg<T: KernelArg>(kernel: & Kernel,
                                    position: cl_uint,
                                    arg: &T)
{
    let (size, p) = arg.get_value();
    let ret = clSetKernelArg(kernel.kernel, position, 
                             size,
                             p);
    
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

pub impl Buffer: KernelArg{
    pure fn get_value(&self) -> (libc::size_t, *libc::c_void) {
        (sys::size_of::<cl_mem>() as libc::size_t,
         ptr::addr_of(&self.buffer) as *libc::c_void)
    }
} 

struct Event {
    event: cl_event,

    drop {
        clReleaseEvent(self.event);
    }
}

impl Event {
    fn wait() {
        let status = clWaitForEvents(1, ptr::addr_of(&self.event));
        check(status, "Error waiting for event");
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
    fn create_program_from_source(@self, src: &str) -> Program {
        do str::as_c_str(src) |src| {
            let mut status = CL_SUCCESS as cl_int;
            let program = clCreateProgramWithSource(
                self.ctx.ctx,
                1,
                ptr::addr_of(&src),
                ptr::null(),
                ptr::addr_of(&status));
            check(status, "Could not create program");

            Program {
                prg: program,
                context: Some(self),
            }
        }
    }

    fn create_program_from_binary(@self, bin: &str) -> Program {
        do str::as_c_str(bin) |src| {
            let mut status = CL_SUCCESS as cl_int;
            let len = bin.len() as libc::size_t;
            let program = clCreateProgramWithBinary(
                self.ctx.ctx,
                1,
                ptr::addr_of(&self.device.id),
                ptr::addr_of(&len),
                ptr::addr_of(&src) as **libc::c_uchar,
                ptr::null(),
                ptr::addr_of(&status));
            check(status, "Could not create program");

            Program {
                prg: program,
                context: Some(self),
            }
        }
    }

    fn enqueue_async_kernel<I: KernelIndex>(k: &Kernel, global: I, local: I)
        -> Event
    {
        let e: cl_event = ptr::null();
        let status = clEnqueueNDRangeKernel(
            self.q.cqueue,
            k.kernel,
            KernelIndex::num_dimensions::<I>(),
            ptr::null(),
            global.get_ptr(),
            local.get_ptr(),
            0,
            ptr::null(),
            ptr::addr_of(&e));
        check(status, "Error enqueuing kernel.");

        Event { event: e }
    }

    fn device_name() -> ~str {
        self.device.name()
    }
}

pub fn create_compute_context() -> @ComputeContext {
    create_compute_context_types([CPU, GPU])
}

pub fn create_compute_context_types(types: &[DeviceType]) -> @ComputeContext {
    // Enumerate all platforms until we find a device that works.

    for get_platforms().each |p| {
        let devices = p.get_devices_by_types(types);
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

trait KernelIndex {
    static pure fn num_dimensions() -> cl_uint;
    pure fn get_ptr(&self) -> *libc::size_t;
}

impl int: KernelIndex {
    static pure fn num_dimensions() -> cl_uint { 1 }

    pure fn get_ptr(&self) -> *libc::size_t {
        ptr::addr_of(self) as *libc::size_t
    }
}

impl (int, int): KernelIndex {
    static pure fn num_dimensions() -> cl_uint { 2 }

    pure fn get_ptr(&self) -> *libc::size_t {
        ptr::addr_of(self) as *libc::size_t
    }
}

impl uint: KernelIndex {
    static pure fn num_dimensions() -> cl_uint { 1 }

    pure fn get_ptr(&self) -> *libc::size_t {
        ptr::addr_of(self) as *libc::size_t
    }
}

impl (uint, uint): KernelIndex {
    static pure fn num_dimensions() -> cl_uint { 2 }

    pure fn get_ptr(&self) -> *libc::size_t {
        ptr::addr_of(self) as *libc::size_t
    }
}

#[cfg(test)]
mod test {
    macro_rules! expect (
        ($test: expr, $expected: expr) => ({
            let test = $test;
            let expected = $expected;
            if test != expected {
                fail fmt!("Test failure in %s: expected %?, got %?",
                          stringify!($test),
                          expected, test)
            }
        })
    )    

    use vector::*;

    #[test]
    fn program_build() {
        let src = "__kernel void test(__global int *i) { \
                       *i += 1; \
                   }";
        let ctx = create_compute_context();
        let prog = ctx.create_program_from_source(src);
        prog.build(ctx.device);
    }

    #[test]
    fn simple_kernel() {
        let src = "__kernel void test(__global int *i) { \
                       *i += 1; \
                   }";
        let ctx = create_compute_context();
        let prog = ctx.create_program_from_source(src);
        prog.build(ctx.device);

        let k = prog.create_kernel("test");
        
        let v = Vector::from_vec(ctx, [1]);
        
        k.set_arg(0, &v);
        
        enqueue_nd_range_kernel(
            &ctx.q,
            &k,
            1, 0, 1, 1);

        let v = v.to_vec();

        expect!(v[0], 2);
    }

    #[test]
    fn add_k() {
        let src = "__kernel void test(__global int *i, long int k) { \
                       *i += k; \
                   }";
        let ctx = create_compute_context();
        let prog = ctx.create_program_from_source(src);
        prog.build(ctx.device);

        let k = prog.create_kernel("test");
        
        let v = Vector::from_vec(ctx, [1]);
        
        k.set_arg(0, &v);
        k.set_arg(1, &42);
        
        enqueue_nd_range_kernel(
            &ctx.q,
            &k,
            1, 0, 1, 1);

        let v = v.to_vec();

        expect!(v[0], 43);
    }

    #[test]
    fn simple_kernel_index() {
        let src = "__kernel void test(__global int *i) { \
                       *i += 1; \
                   }";
        let ctx = create_compute_context();
        let prog = ctx.create_program_from_source(src);
        prog.build(ctx.device);

        let k = prog.create_kernel("test");
        
        let v = Vector::from_vec(ctx, [1]);
        
        k.set_arg(0, &v);
        
        ctx.enqueue_async_kernel(&k, 1, 1).wait();

        let v = v.to_vec();

        expect!(v[0], 2);
    }

    #[test]
    fn kernel_2d() {
        let src = "__kernel void test(__global long int *N) { \
                       int i = get_global_id(0); \
                       int j = get_global_id(1); \
                       int s = get_global_size(0); \
                       N[i * s + j] = i * j;
                   }";
        let ctx = create_compute_context();
        let prog = ctx.create_program_from_source(src);
        
        match prog.build(ctx.device) {
            Ok(()) => (),
            Err(build_log) => {
                io::println("Error building program:\n");
                io::println(build_log);
                fail
            }
        }

        let k = prog.create_kernel("test");
        
        let v = Vector::from_vec(ctx, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        
        k.set_arg(0, &v);
        
        ctx.enqueue_async_kernel(&k, (3, 3), (1, 1)).wait();

        let v = v.to_vec();

        expect!(v, ~[0, 0, 0, 0, 1, 2, 0, 2, 4]);
    }

    #[test]
    fn kernel_2d_execute() {
        let src = "__kernel void test(__global long int *N) { \
                       int i = get_global_id(0); \
                       int j = get_global_id(1); \
                       int s = get_global_size(0); \
                       N[i * s + j] = i * j;
                   }";
        let ctx = create_compute_context();
        let prog = ctx.create_program_from_source(src);
        
        match prog.build(ctx.device) {
            Ok(()) => (),
            Err(build_log) => {
                io::println("Error building program:\n");
                io::println(build_log);
                fail
            }
        }

        let k = prog.create_kernel("test");
        
        let v = Vector::from_vec(ctx, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        
        k.set_arg(0, &v);
        k.execute((3, 3), (1, 1));

        let v = v.to_vec();

        expect!(v, ~[0, 0, 0, 0, 1, 2, 0, 2, 4]);
    }
}