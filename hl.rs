// Higher level OpenCL wrappers.

use CL;
use CL::*;
use CL::ll::*;
use error::check;
use std::libc;
use std::vec;
use std::str;
use std::rt::io;
use std::rt::io::file;
use std::rt::io::extensions::ReaderUtil;
use std::mem;
use std::cast;
use std::ptr;

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
  fn get_devices(&self) -> ~[Device]
  {
    get_devices(*self, CL_DEVICE_TYPE_ALL)
  }

  fn get_devices_by_types(&self, types: &[DeviceType]) -> ~[Device]
  {
    let dtype = 0;
    for &t in types.iter() {
      dtype != convert_device_type(t);
    }

    get_devices(*self, dtype)
  }

  #[fixed_stack_segment] #[inline(never)]
  fn name(&self) -> ~str
  {
    unsafe
    {
      let mut size = 0;

      clGetPlatformInfo(self.id,
                        CL_PLATFORM_NAME,
                        0,
                        ptr::null(),
                        ptr::to_mut_unsafe_ptr(&mut size));

      let name = " ".repeat(size as uint);

      do name.as_imm_buf |p, len| {
        clGetPlatformInfo(self.id,
                          CL_PLATFORM_NAME,
                          len as libc::size_t,
                          p as *libc::c_void,
                          ptr::to_mut_unsafe_ptr(&mut size));
      };

      name
    }
  }
}

#[fixed_stack_segment] #[inline(never)]
pub fn get_platforms() -> ~[Platform]
{
    let num_platforms = 0;

    unsafe
    {
        let status = clGetPlatformIDs(0,
                                      ptr::null(),
                                      ptr::to_unsafe_ptr(&num_platforms));
        check(status, "could not get platform count.");

        let ids = vec::from_elem(num_platforms as uint, 0 as cl_platform_id);

        do ids.as_imm_buf |ids, len| {
            let status = clGetPlatformIDs(len as cl_uint,
                                          ids,
                                          ptr::to_unsafe_ptr(&num_platforms));
            check(status, "could not get platforms.");
        };

        do ids.map |id| { Platform { id: *id } }
    }
}

struct Device {
    id: cl_device_id
}

impl Device {
    #[fixed_stack_segment] #[inline(never)]
    pub fn name(&self) -> ~str { unsafe {
        let size = 0;
        let status = clGetDeviceInfo(
            self.id,
            CL_DEVICE_NAME,
            0,
            ptr::null(),
            ptr::to_unsafe_ptr(&size));
        check(status, "Could not determine name length");

        let buf = vec::from_elem(size as uint, 0);

        do buf.as_imm_buf |p, len| {
            let status = clGetDeviceInfo(
                self.id,
                CL_DEVICE_NAME,
                len as libc::size_t,
                p as *libc::c_void,
                ptr::null());
            check(status, "Could not get device name");

            str::raw::from_c_str(p as *i8)
        }
    } }

    #[fixed_stack_segment] #[inline(never)]
	pub fn computeUnits(&self) -> uint {
		unsafe {
			let mut ct: uint = 0;
            let status = clGetDeviceInfo(
                self.id,
                CL_DEVICE_MAX_COMPUTE_UNITS,
                8,
                ptr::to_mut_unsafe_ptr(&mut ct) as *libc::c_void,
                ptr::null());
            check(status, "Could not get number of device compute units.");
			return ct;
		}
	}

}

#[fixed_stack_segment] #[inline(never)]
pub fn get_devices(platform: Platform, dtype: cl_device_type) -> ~[Device]
{
    unsafe
    {
        let num_devices = 0;

        info!("Looking for devices matching {:?}", dtype);

        clGetDeviceIDs(platform.id, dtype, 0, ptr::null(),
                       ptr::to_unsafe_ptr(&num_devices));

        let ids = vec::from_elem(num_devices as uint, 0 as cl_device_id);
        do ids.as_imm_buf |ids, len| {
            clGetDeviceIDs(platform.id, dtype, len as cl_uint,
                           ids, ptr::to_unsafe_ptr(&num_devices));
        };

        do ids.map |id| { Device {id: *id }}
    }
}

struct Context {
    ctx: cl_context,
}

impl Drop for Context
{
    #[fixed_stack_segment] #[inline(never)]
    fn drop(&mut self) {
        unsafe {
            clReleaseContext(self.ctx);
        }
    }
}

#[fixed_stack_segment] #[inline(never)]
pub fn create_context(device: Device) -> Context
{
    unsafe
    {
        // TODO: Support for multiple devices
        let errcode = 0;

        // TODO: Proper error messages
        let ctx = clCreateContext(ptr::null(),
                                  1,
                                  ptr::to_unsafe_ptr(&device.id),
                                  cast::transmute(ptr::null::<&fn ()>()),
                                  ptr::null(),
                                  ptr::to_unsafe_ptr(&errcode));

        check(errcode, "Failed to create opencl context!");

        Context { ctx: ctx }
    }
}

struct CommandQueue {
    cqueue: cl_command_queue,
    device: Device,
}

impl Drop for CommandQueue
{
    #[fixed_stack_segment] #[inline(never)]
    fn drop(&mut self) {
        unsafe {
            clReleaseCommandQueue(self.cqueue);
        }
    }
}

#[fixed_stack_segment] #[inline(never)]
pub fn create_command_queue(ctx: & Context, device: Device) -> CommandQueue
{
    unsafe
    {
        let errcode = 0;

        let cqueue = clCreateCommandQueue(ctx.ctx, device.id, 0,
                                          ptr::to_unsafe_ptr(&errcode));

        check(errcode, "Failed to create command queue!");

        CommandQueue {
            cqueue: cqueue,
            device: device
        }
    }
}

struct Buffer
{
    buffer: cl_mem,
    size: uint,
}

impl Buffer
{
	#[fixed_stack_segment]
	pub fn write<T>(&self, ctx: @ComputeContext, inVec: &[T]) {
		unsafe {
			clEnqueueWriteBuffer(ctx.q.cqueue,
				self.buffer,
				CL_TRUE,
				0,
				self.size as libc::size_t,
				vec::raw::to_ptr(inVec) as *libc::c_void,
				0,
				ptr::null(),
				ptr::null());
		}
	}
	#[fixed_stack_segment]
	pub unsafe fn read<T>(&self, ctx: @ComputeContext) -> ~[T] {
		let mut v: ~[T] = vec::with_capacity(self.size / mem::size_of::<T>());
		clEnqueueReadBuffer(ctx.q.cqueue,
			self.buffer,
			CL_TRUE,
			0,
			self.size as libc::size_t,
			vec::raw::to_mut_ptr(v) as *libc::c_void,
			0,
			ptr::null(),
			ptr::null());
		vec::raw::set_len(&mut v, self.size / mem::size_of::<T>());
		return v
	}
}

impl Drop for Buffer
{
    #[fixed_stack_segment] #[inline(never)]
    fn drop(&mut self) {
        unsafe {
            clReleaseMemObject(self.buffer);
        }
    }
}



struct Program
{
    prg: cl_program,
}

impl Drop for Program
{
    #[fixed_stack_segment] #[inline(never)]
    fn drop(&mut self) {
        unsafe {
            clReleaseProgram(self.prg);
        }
    }
}

impl Program
{
    pub fn build(&self, device: Device) -> Result<(), ~str> {
        build_program(self, device)
    }

    pub fn create_kernel(&self, name: &str) -> Kernel {
        create_kernel(self, name)
    }
}

// TODO: Support multiple devices
#[fixed_stack_segment] #[inline(never)]
pub fn create_program_with_binary(ctx: & Context, device: Device,
                                  binary_path: & Path) -> Program
{
    unsafe
    {
        let errcode = 0;
        let mut file = file::open(binary_path, io::Open, io::Read);
        let binary = file.read_to_end();
        let program = do binary.to_c_str().with_ref |kernel_binary| {
            clCreateProgramWithBinary(ctx.ctx, 1, ptr::to_unsafe_ptr(&device.id),
                                      ptr::to_unsafe_ptr(&(binary.len() + 1)) as *libc::size_t,
                                      ptr::to_unsafe_ptr(&kernel_binary) as **libc::c_uchar,
                                      ptr::null(),
                                      ptr::to_unsafe_ptr(&errcode))
        };

        check(errcode, "Failed to create open cl program with binary!");

        Program {
            prg: program,
        }
    }
}

#[fixed_stack_segment] #[inline(never)]
pub fn build_program(program: & Program, device: Device) -> Result<(), ~str>
{
    unsafe
    {
        let ret = clBuildProgram(program.prg, 1, ptr::to_unsafe_ptr(&device.id),
                                 ptr::null(),
                                 cast::transmute(ptr::null::<&fn ()>()),
                                 ptr::null());
        if ret == CL_SUCCESS as cl_int {
            Ok(())
        }
        else {
            let size = 0 as libc::size_t;
            let status = clGetProgramBuildInfo(
                program.prg,
                device.id,
                CL_PROGRAM_BUILD_LOG,
                0,
                ptr::null(),
                ptr::to_unsafe_ptr(&size));
            check(status, "Could not get build log");

            let buf = vec::from_elem(size as uint, 0u8);
            do buf.as_imm_buf |p, len| {
                let status = clGetProgramBuildInfo(
                    program.prg,
                    device.id,
                    CL_PROGRAM_BUILD_LOG,
                    len as libc::size_t,
                    p as *libc::c_void,
                    ptr::null());
                check(status, "Could not get build log");

                Err(str::raw::from_c_str(p as *libc::c_char))
            }
        }
    }
}

struct Kernel {
    kernel: cl_kernel,
}

impl Drop for Kernel
{
    #[fixed_stack_segment] #[inline(never)]
    fn drop(&mut self) {
        unsafe {
            clReleaseKernel(self.kernel);
        }
    }
}

impl Kernel {
    pub fn set_arg<T: KernelArg>(&self, i: uint, x: &T)
    {
        set_kernel_arg(self, i as CL::cl_uint, x)
    }

/*
    pub fn execute<I: KernelIndex>(&self, global: I, local: I) {
        match self.context {
            Some(ctx)
            => ctx.enqueue_async_kernel(self, global, local).wait(),

            None => fail!("Kernel does not have an associated context.")
        }
    }

    pub fn work_group_size(&self) -> uint { unsafe {
        match self.context {
            Some(ctx) => {
                let mut size: libc::size_t = 0;
                let status = clGetKernelWorkGroupInfo(
                    self.kernel,
                    ctx.device.id,
                    CL_KERNEL_WORK_GROUP_SIZE,
                    mem::size_of::<libc::size_t>() as libc::size_t,
                    ptr::to_unsafe_ptr(&size) as *libc::c_void,
                    ptr::null());
                check(status, "Could not get work group info.");
                size as uint
            },
            None => fail!("Kernel does not have an associated context.")
        }
    } }

    pub fn local_mem_size(&self) -> uint {
        match self.context {
            Some(ctx) => {
                let mut size: cl_ulong = 0;
                let status = unsafe {
                    clGetKernelWorkGroupInfo(
                        self.kernel,
                        ctx.device.id,
                        CL_KERNEL_LOCAL_MEM_SIZE,
                        mem::size_of::<cl_ulong>() as libc::size_t,
                        ptr::to_unsafe_ptr(&size) as *libc::c_void,
                        ptr::null())
                };
                check(status, "Could not get work group info.");
                size as uint
            },
            None => fail!("Kernel does not have an associated context.")
        }
    }

    pub fn private_mem_size(&self) -> uint {
        match self.context {
            Some(ctx) => {
                let mut size: cl_ulong = 0;
                let status = unsafe {
                    clGetKernelWorkGroupInfo(
                        self.kernel,
                        ctx.device.id,
                        CL_KERNEL_PRIVATE_MEM_SIZE,
                        mem::size_of::<cl_ulong>() as libc::size_t,
                        ptr::to_unsafe_ptr(&size) as *libc::c_void,
                        ptr::null())
                };
                check(status, "Could not get work group info.");
                size as uint
            },
            None => fail!("Kernel does not have an associated context.")
        }
    }
*/
}

#[fixed_stack_segment] #[inline(never)]
pub fn create_kernel(program: & Program, kernel: & str) -> Kernel
{
    unsafe {
        let errcode = 0;
        // let bytes = str::to_bytes(kernel);
        do kernel.to_c_str().with_ref |str_ptr|
        {
            let kernel = clCreateKernel(program.prg,
                                        str_ptr,
                                        ptr::to_unsafe_ptr(&errcode));

            check(errcode, "Failed to create kernel!");

            Kernel {
                kernel: kernel,
            }
        }
    }
}

pub trait KernelArg {
  fn get_value(&self) -> (libc::size_t, *libc::c_void);
}

macro_rules! scalar_kernel_arg (
    ($t:ty) => (impl KernelArg for $t {
        fn get_value(&self) -> (libc::size_t, *libc::c_void) {
            (mem::size_of::<$t>() as libc::size_t,
             ptr::to_unsafe_ptr(self) as *libc::c_void)
        }
    })
)

scalar_kernel_arg!(int)
scalar_kernel_arg!(uint)
scalar_kernel_arg!(u32)
scalar_kernel_arg!(u64)
scalar_kernel_arg!(i32)
scalar_kernel_arg!(i64)
scalar_kernel_arg!(f32)
scalar_kernel_arg!(f64)

#[fixed_stack_segment] #[inline(never)]
pub fn set_kernel_arg<T: KernelArg>(kernel: & Kernel,
                                    position: cl_uint,
                                    arg: &T)
{
    unsafe
    {
        let (size, p) = arg.get_value();
        let ret = clSetKernelArg(kernel.kernel, position,
                                 size,
                                 p);

        check(ret, "Failed to set kernel arg!");
    }
}

#[fixed_stack_segment] #[inline(never)]
pub fn enqueue_nd_range_kernel(cqueue: & CommandQueue, kernel: & Kernel, work_dim: cl_uint,
                               _global_work_offset: int, global_work_size: int,
                               local_work_size: int)
{
  unsafe
    {
      let ret = clEnqueueNDRangeKernel(cqueue.cqueue, kernel.kernel, work_dim,
                                       // ptr::to_unsafe_ptr(&global_work_offset) as *libc::size_t,
                                       ptr::null(),
                                       ptr::to_unsafe_ptr(&global_work_size) as *libc::size_t,
                                       ptr::to_unsafe_ptr(&local_work_size) as *libc::size_t,
                                       0, ptr::null(), ptr::null());
      check(ret, "Failed to enqueue nd range kernel!");
  }
}

impl KernelArg for Buffer
{
    fn get_value(&self) -> (libc::size_t, *libc::c_void)
    {
        (mem::size_of::<cl_mem>() as libc::size_t,
         ptr::to_unsafe_ptr(&self.buffer) as *libc::c_void)
    }
}

pub struct Event
{
    event: cl_event,
}

impl Drop for Event
{
    #[fixed_stack_segment] #[inline(never)]
    fn drop(&mut self) {
        unsafe {
            clReleaseEvent(self.event);
        }
    }
}

trait EventList {
    fn as_event_list<T>(&self, &fn(*cl_event, cl_uint) -> T) -> T;

    #[fixed_stack_segment] #[inline(never)]
    fn wait(&self) {
        do self.as_event_list |p, len| {
            unsafe {
                let status = clWaitForEvents(len, p);
                check(status, "Error waiting for event(s)");
            }
        }
    }
}

impl EventList for Event {
    fn as_event_list<T>(&self, f: &fn(*cl_event, cl_uint) -> T) -> T
    {
        f(ptr::to_unsafe_ptr(&self.event), 1 as cl_uint)
    }
}

impl<T: EventList> EventList for Option<T> {
    fn as_event_list<T>(&self, f: &fn(*cl_event, cl_uint) -> T) -> T
    {
        match *self {
            None => f(ptr::null(), 0),
            Some(ref s) => s.as_event_list(f)
        }
    }
}

impl<'self> EventList for &'self ~[Event] {
    fn as_event_list<T>(&self, f: &fn(*cl_event, cl_uint) -> T) -> T
    {
        /* this is wasteful */
        let events = self.iter().map(|event| event.event).to_owned_vec();

        do events.as_imm_buf |p, len| {
            f(p as **libc::c_void, len as cl_uint)
        }
    }
}

/* this seems VERY hackey */
impl EventList for () {
    fn as_event_list<T>(&self, f: &fn(*cl_event, cl_uint) -> T) -> T
    {
        f(ptr::null(), 0)
    }
}

/**
This packages an OpenCL context, a device, and a command queue to
simplify handling all these structures.
*/
pub struct ComputeContext
{
    ctx:    Context,
    device: Device,
    q:      CommandQueue
}

impl ComputeContext
{
	// TODO: How to make this function cleaner and nice
	#[fixed_stack_segment] #[inline(never)]
	pub fn create_buffer(@self, len: uint, flags: cl_mem_flags) -> Buffer
	{
   		unsafe
    	{
        	let errcode = 0;
        	let buffer = clCreateBuffer(self.ctx.ctx,
									flags,
                                    len as libc::size_t,
									ptr::null(),
                                    ptr::to_unsafe_ptr(&errcode));

        check(errcode, "Failed to create buffer!");

        Buffer { buffer: buffer, size: len}
    	}
	}

    #[fixed_stack_segment] #[inline(never)]
    pub fn create_program_from_source(@self, src: &str) -> Program
    {
        unsafe
        {
            do src.to_c_str().with_ref |src| {
                let status = CL_SUCCESS as cl_int;
                let program = clCreateProgramWithSource(
                    self.ctx.ctx,
                    1,
                    ptr::to_unsafe_ptr(&src),
                    ptr::null(),
                    ptr::to_unsafe_ptr(&status));
                check(status, "Could not create program");

                Program { prg: program }
            }
        }
    }

    #[fixed_stack_segment] #[inline(never)]
    pub fn create_program_from_binary(@self, bin: &str) -> Program {
        do bin.to_c_str().with_ref |src| {
            let status = CL_SUCCESS as cl_int;
            let len = bin.len() as libc::size_t;
            let program = unsafe {
                clCreateProgramWithBinary(
                    self.ctx.ctx,
                    1,
                    ptr::to_unsafe_ptr(&self.device.id),
                    ptr::to_unsafe_ptr(&len),
                    ptr::to_unsafe_ptr(&src) as **libc::c_uchar,
                    ptr::null(),
                    ptr::to_unsafe_ptr(&status))
            };
            check(status, "Could not create program");

            Program {
                prg: program,
            }
        }
    }

    #[fixed_stack_segment] #[inline(never)]
    pub fn enqueue_async_kernel<I: KernelIndex, E: EventList>(&self, k: &Kernel, global: I, local: Option<I>, wait_on: E)
        -> Event
        {
            unsafe
            {
                do wait_on.as_event_list |event, event_count| {
                    let e: cl_event = ptr::null();
                    let status = clEnqueueNDRangeKernel(
                        self.q.cqueue,
                        k.kernel,
                        KernelIndex::num_dimensions(None::<I>),
                        ptr::null(),
                        global.get_ptr(),
                        match local {
                            Some(ref l) => l.get_ptr(),
                            None => ptr::null()
                        },
                        event_count,
                        event,
                        ptr::to_unsafe_ptr(&e));
                    check(status, "Error enqueuing kernel.");
                    Event { event: e }
                }
            }
        }

    pub fn device_name(&self) -> ~str {
        self.device.name()
    }
}

pub fn create_compute_context() -> @ComputeContext {
  // Enumerate all platforms until we find a device that works.

  let platforms = get_platforms();

  for p in platforms.iter()
  {
    let devices = p.get_devices();

    if devices.len() > 0
    {
      let device = devices[0];
      let ctx    = create_context(device);
      let q      = create_command_queue(&ctx, device);

      return @ComputeContext
      {
        ctx:    ctx,
        device: device,
        q:      q
      }
    }
  }

  fail!("No suitable device found")
}

pub fn create_compute_context_types(types: &[DeviceType]) -> @ComputeContext {
    // Enumerate all platforms until we find a device that works.

    let platforms = get_platforms();

    for p in platforms.iter() {
        let devices = p.get_devices_by_types(types);
        if devices.len() > 0 {
            let device = devices[0];
            let ctx = create_context(device);
            let q = create_command_queue(&ctx, device);
            return @ComputeContext {
                ctx: ctx,
                device: device,
                q: q
            }
        }
    }

    fail!(~"Could not find an acceptable device.")
}

trait KernelIndex
{
    fn num_dimensions(dummy_self: Option<Self>) -> cl_uint;
    fn get_ptr(&self) -> *libc::size_t;
}

impl KernelIndex for int
{
    fn num_dimensions(_: Option<int>) -> cl_uint { 1 }

    fn get_ptr(&self) -> *libc::size_t
    {
        ptr::to_unsafe_ptr(self) as *libc::size_t
    }
}

impl KernelIndex for (int, int) {
    fn num_dimensions(_: Option<(int, int)>) -> cl_uint { 2 }

    fn get_ptr(&self) -> *libc::size_t {
        ptr::to_unsafe_ptr(self) as *libc::size_t
    }
}

impl KernelIndex for uint
{
    fn num_dimensions(_: Option<uint>) -> cl_uint { 1 }

    fn get_ptr(&self) -> *libc::size_t {
        ptr::to_unsafe_ptr(self) as *libc::size_t
    }
}

impl KernelIndex for (uint, uint)
{
    fn num_dimensions(_: Option<(uint, uint)>) -> cl_uint { 2 }

    fn get_ptr(&self) -> *libc::size_t {
        ptr::to_unsafe_ptr(self) as *libc::size_t
    }
}

#[cfg(test)]
mod test {
    use hl::*;
    use vector::Vector;
    use std::rt::io;

    macro_rules! expect (
        ($test: expr, $expected: expr) => ({
            let test     = $test;
            let expected = $expected;
            if test != expected {
                fail!(format!("Test failure in {:s}: expected {:?}, got {:?}",
                           stringify!($test),
                           expected, test))
            }
        })
    )

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

        ctx.enqueue_async_kernel(&k, 1, None, ()).wait();
        let v = v.to_vec();

        expect!(v[0], 2);
    }

    #[test]
    fn chain_kernel_event() {
        let src = "__kernel void test(__global int *i) { \
                   *i += 1; \
                   }";
        let ctx = create_compute_context();
        let prog = ctx.create_program_from_source(src);
        prog.build(ctx.device);

        let k = prog.create_kernel("test");

        let v = Vector::from_vec(ctx, [1]);

        k.set_arg(0, &v);

        let mut e : Option<Event> = None;
        for _ in range(0, 8) {
            e = Some(ctx.enqueue_async_kernel(&k, 1, None, e));
        }
        e.wait();

        let v = v.to_vec();

        expect!(v[0], 9);
    }

    #[test]
    fn chain_kernel_event_list() {
        let src = "__kernel void inc(__global int *i) { \
                   *i += 1; \
                   } \
                   __kernel void add(__global int *a, __global int *b, __global int *c) { \
                   *c = *a + *b; \
                   }";
        let ctx = create_compute_context();
        let prog = ctx.create_program_from_source(src);
        prog.build(ctx.device);

        let k_incA = prog.create_kernel("inc");
        let k_incB = prog.create_kernel("inc");
        let k_add = prog.create_kernel("add");

        let a = Vector::from_vec(ctx, [1]);
        let b = Vector::from_vec(ctx, [1]);
        let c = Vector::from_vec(ctx, [1]);

        k_incA.set_arg(0, &a);
        k_incB.set_arg(0, &b);
        let event_list = ~[
            ctx.enqueue_async_kernel(&k_incA, 1, None, ()),
            ctx.enqueue_async_kernel(&k_incB, 1, None, ()),
        ];

        k_add.set_arg(0, &a);
        k_add.set_arg(1, &b);
        k_add.set_arg(2, &c);

        ctx.enqueue_async_kernel(&k_add, 1, None, &event_list).wait();
        let v = c.to_vec();

        expect!(v[0], 4);
    }

    #[test]
    fn kernel_2d()
    {
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
                println!("Error building program:\n");
                println!("{:s}", build_log);
                fail!("");
            }
        }

        let k = prog.create_kernel("test");

        let v = Vector::from_vec(ctx, [1, 2, 3, 4, 5, 6, 7, 8, 9]);

        k.set_arg(0, &v);

        ctx.enqueue_async_kernel(&k, (3, 3), None, ()).wait();

        let v = v.to_vec();

        expect!(v, ~[0, 0, 0, 0, 1, 2, 0, 2, 4]);
    }

/*
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
                fail!()
            }
        }

        let k = prog.create_kernel("test");

        let v = Vector::from_vec(ctx, [1, 2, 3, 4, 5, 6, 7, 8, 9]);

        k.set_arg(0, &v);
        k.execute((3, 3), (1, 1));

        let v = v.to_vec();

        expect!(v, ~[0, 0, 0, 0, 1, 2, 0, 2, 4]);
    }
*/
}
