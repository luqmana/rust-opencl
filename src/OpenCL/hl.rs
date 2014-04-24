//! A higher level API.

use CL;
use CL::*;
use CL::ll::*;
use error::check;
use libc;
use std::vec::Vec;
use std::str;
use std::mem;
use std::cast;
use std::ptr;
use sync::mutex;
use mem::{Put, Get, Write, Read, Buffer, CLBuffer};

pub enum DeviceType {
      CPU, GPU
}

fn convert_device_type(device: DeviceType) -> cl_device_type {
    match device {
        CPU => CL_DEVICE_TYPE_CPU,
        GPU => CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR
    }
}

pub struct Platform {
    id: cl_platform_id
}

impl Platform {
    fn get_devices_internal(&self, dtype: cl_device_type) -> ~[Device]
    {
        unsafe
        {
            let num_devices = 0;

            info!("Looking for devices matching {:?}", dtype);

            clGetDeviceIDs(self.id, dtype, 0, ptr::null(),
                           (&num_devices));

            let ids = Vec::from_elem(num_devices as uint, 0 as cl_device_id);
            clGetDeviceIDs(self.id, dtype, ids.len() as cl_uint,
                           ids.as_ptr(), (&num_devices));
            ids.iter().map(|id| { Device {id: *id }}).collect()
        }
    }

    pub fn get_devices(&self) -> ~[Device]
    {
        self.get_devices_internal(CL_DEVICE_TYPE_ALL)
    }

    pub fn get_devices_by_types(&self, types: &[DeviceType]) -> ~[Device]
    {
        let mut dtype = 0;
        for &t in types.iter() {
          dtype |= convert_device_type(t);
        }

        self.get_devices_internal(dtype)
    }

    fn profile_info(&self, name: cl_platform_info) -> ~str
    {
        unsafe {
            let mut size = 0;

            clGetPlatformInfo(self.id,
                            name,
                            0,
                            ptr::null(),
                            &mut size);

            let value = " ".repeat(size as uint);

            clGetPlatformInfo(self.id,
                              name,
                              value.len() as libc::size_t,
                              value.as_ptr() as *libc::c_void,
                              &mut size);
            value
        }
    }
    
    pub fn name(&self) -> ~str
    {
        self.profile_info(CL_PLATFORM_NAME)
    }
    
    pub fn version(&self) -> ~str
    {
        self.profile_info(CL_PLATFORM_VERSION)
    }
    
    pub fn profile(&self) -> ~str
    {
        self.profile_info(CL_PLATFORM_PROFILE)
    }
    
    pub fn vendor(&self) -> ~str
    {
        self.profile_info(CL_PLATFORM_VENDOR)
    }
    
    pub fn extensions(&self) -> ~str
    {
        self.profile_info(CL_PLATFORM_EXTENSIONS)
    }
}

// This mutex is used to work around weak OpenCL implementations.
// On some implementations concurrent calls to clGetPlatformIDs
// will cause the implantation to return invalid status. 
static mut platforms_mutex: mutex::StaticMutex = mutex::MUTEX_INIT;

pub fn get_platforms() -> ~[Platform]
{
    let num_platforms = 0;

    unsafe
    {
        let guard = platforms_mutex.lock();
        let status = clGetPlatformIDs(0,
                                      ptr::null(),
                                      (&num_platforms));
        // unlock this before the check in case the check fails
        check(status, "could not get platform count.");

        let ids = Vec::from_elem(num_platforms as uint, 0 as cl_platform_id);

        let status = clGetPlatformIDs(ids.len() as cl_uint,
                                      ids.as_ptr(),
                                      (&num_platforms));
        check(status, "could not get platforms.");
        
        let _ = guard;

        ids.iter().map(|id| { Platform { id: *id } }).collect()
    }
}

pub struct Device {
    id: cl_device_id
}

impl Device {
    pub fn name(&self) -> ~str {
        unsafe {
            let mut size = 0;
            let status = clGetDeviceInfo(
                self.id,
                CL_DEVICE_NAME,
                0,
                ptr::mut_null(),
                (&mut size));
            check(status, "Could not determine name length");
            
            let mut buf = Vec::from_elem(size as uint, 0);
            
            let status = clGetDeviceInfo(
                self.id,
                CL_DEVICE_NAME,
                buf.len() as libc::size_t,
                buf.as_mut_ptr() as *mut libc::c_void,
                ptr::mut_null());
            check(status, "Could not get device name");
            
            str::raw::from_c_str(buf.as_ptr() as *i8)
        }
    }

	pub fn computeUnits(&self) -> uint {
		unsafe {
			let mut ct: uint = 0;
            let status = clGetDeviceInfo(
                self.id,
                CL_DEVICE_MAX_COMPUTE_UNITS,
                8,
                (&mut ct as *mut uint) as *mut libc::c_void,
                ptr::mut_null());
            check(status, "Could not get number of device compute units.");
			return ct;
		}
	}


    pub fn create_context(&self) -> Context
    {
        unsafe
        {
            // TODO: Support for multiple devices
            let errcode = 0;

            // TODO: Proper error messages
            let ctx = clCreateContext(ptr::null(),
                                      1,
                                      &self.id,
                                      cast::transmute(ptr::null::<||>()),
                                      ptr::null(),
                                      (&errcode));

            check(errcode, "Failed to create opencl context!");

            Context { ctx: ctx }
        }
    }
}

pub struct Context {
    ctx: cl_context,
}

impl Context {
    pub fn create_buffer<T>(&self, size: uint, flags: cl_mem_flags) -> CLBuffer<T>
    {
        unsafe {
            let status = 0;
            let buf = clCreateBuffer(self.ctx,
                                     flags,
                                     (size*mem::size_of::<T>()) as libc::size_t ,
                                     ptr::null(),
                                     (&status));
            check(status, "Could not allocate buffer");
            CLBuffer{cl_buffer: buf}
        }
    }


    pub fn create_buffer_from<T, U, IN: Put<T, U>>(&self, create: IN, flags: cl_mem_flags) -> U
    {
        create.put(|p, len| {
            let status = 0;
            let buf = unsafe {
                clCreateBuffer(self.ctx,
                               flags | CL_MEM_COPY_HOST_PTR,
                               len,
                               p,
                               (&status))
            };
            check(status, "Could not allocate buffer");
            buf
        })

    }

    pub fn create_command_queue(&self, device: &Device) -> CommandQueue
    {
        unsafe
        {
            let errcode = 0;

            let cqueue = clCreateCommandQueue(self.ctx,
                                              device.id,
                                              CL_QUEUE_PROFILING_ENABLE,
                                              (&errcode));

            check(errcode, "Failed to create command queue!");

            CommandQueue {
                cqueue: cqueue
            }
        }
    }

    pub fn create_program_from_source(&self, src: &str) -> Program
    {
        unsafe
        {
            src.to_c_str().with_ref(|src| {
                let status = CL_SUCCESS as cl_int;
                let program = clCreateProgramWithSource(
                    self.ctx,
                    1,
                    (&src),
                    ptr::null(),
                    (&status));
                check(status, "Could not create program");

                Program { prg: program }
            })
        }
    }

    pub fn create_program_from_binary(&self, bin: &str, device: &Device) -> Program {
        bin.to_c_str().with_ref(|src| {
            let status = CL_SUCCESS as cl_int;
            let len = bin.len() as libc::size_t;
            let program = unsafe {
                clCreateProgramWithBinary(
                    self.ctx,
                    1,
                    &device.id,
                    (&len),
                    (&src as **i8) as **libc::c_uchar,
                    ptr::null(),
                    (&status))
            };
            check(status, "Could not create program");

            Program {prg: program}
        })
    }
}

impl Drop for Context
{
    fn drop(&mut self) {
        unsafe {
            clReleaseContext(self.ctx);
        }
    }
}

impl<'r, T> KernelArg for &'r Buffer<T> {
    fn get_value(&self) -> (libc::size_t, *libc::c_void)
    {
        unsafe {
            (mem::size_of::<cl_mem>() as libc::size_t,
             self.id_ptr() as *libc::c_void)
        }
    }
}

impl<T> KernelArg for ~Buffer<T> {
    fn get_value(&self) -> (libc::size_t, *libc::c_void)
    {
        unsafe {
            (mem::size_of::<cl_mem>() as libc::size_t,
             self.id_ptr() as *libc::c_void)
        }
    }
}


pub struct CommandQueue {
    cqueue: cl_command_queue
}

impl CommandQueue
{
    pub fn enqueue_async_kernel<I: KernelIndex, E: EventList>(&self, k: &Kernel, global: I, local: Option<I>, wait_on: E)
        -> Event
    {
        unsafe
        {
            wait_on.as_event_list(|event, event_count| {
                let e: cl_event = ptr::null();
                let status = clEnqueueNDRangeKernel(
                    self.cqueue,
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
                    (&e));
                check(status, "Error enqueuing kernel.");
                Event { event: e }
            })
        }
    }

    pub fn get<T, U, B: Buffer<T>, G: Get<B, U>, E: EventList>(&self, buf: &B, event: E) -> G
    {
        event.as_event_list(|evt, evt_len| {
            Get::get(buf, |offset, ptr, len| {
                unsafe {
                    let err = clEnqueueReadBuffer(self.cqueue,
                                                  buf.id(),
                                                  CL_TRUE,
                                                  offset as libc::size_t,
                                                  len,
                                                  ptr,
                                                  evt_len,
                                                  evt,
                                                  ptr::null());

                    check(err, "Failed to read buffer");
                }
            })
        })
    }

    pub fn write<U: Write, T, E: EventList, B: Buffer<T>>(&self, mem: &B, write: &U, event: E)
    {
        unsafe {
            event.as_event_list(|evt, evt_len| {
                write.write(|offset, p, len| {
                    let err = clEnqueueWriteBuffer(self.cqueue,
                                                   mem.id(),
                                                   CL_TRUE,
                                                   offset as libc::size_t,
                                                   len as libc::size_t,
                                                   p as *libc::c_void,
                                                   evt_len,
                                                   evt,
                                                   ptr::null());

                    check(err, "Failed to write buffer");
                })
            })
        }
    }

    pub fn read<T, U: Read, E: EventList, B: Buffer<T>>(&self, mem: &B, read: &mut U, event: E)
    {
        unsafe {
            event.as_event_list(|evt, evt_len| {
                read.read(|offset, p, len| {
                    let err = clEnqueueReadBuffer(self.cqueue,
                                                  mem.id(),
                                                  CL_TRUE,
                                                  offset as libc::size_t,
                                                  len as libc::size_t,
                                                  p as *mut libc::c_void,
                                                  evt_len,
                                                  evt,
                                                  ptr::null());

                    check(err, "Failed to read buffer");
                })
            })
        }
    }
}

impl Drop for CommandQueue
{
    fn drop(&mut self) {
        unsafe {
            clReleaseCommandQueue(self.cqueue);
        }
    }
}


pub struct Program
{
    prg: cl_program,
}

impl Drop for Program
{
    fn drop(&mut self) {
        unsafe {
            clReleaseProgram(self.prg);
        }
    }
}

impl Program
{
    pub fn build(&self, device: &Device) -> Result<(), ~str>
    {
        unsafe
        {
            let ret = clBuildProgram(self.prg, 1, &device.id,
                                     ptr::null(),
                                     cast::transmute(ptr::null::<||>()),
                                     ptr::null());
            if ret == CL_SUCCESS as cl_int {
                Ok(())
            }
            else {
                let mut size = 0 as libc::size_t;
                let status = clGetProgramBuildInfo(
                    self.prg,
                    device.id,
                    CL_PROGRAM_BUILD_LOG,
                    0,
                    ptr::mut_null(),
                    (&mut size));
                check(status, "Could not get build log");
                
                let mut buf = Vec::from_elem(size as uint, 0u8);
                let status = clGetProgramBuildInfo(
                    self.prg,
                    device.id,
                    CL_PROGRAM_BUILD_LOG,
                    buf.len() as libc::size_t,
                    buf.as_mut_ptr() as *mut libc::c_void,
                    ptr::mut_null());
                check(status, "Could not get build log");
                
                Err(str::raw::from_c_str(buf.as_ptr() as *libc::c_char))
            }
        }
    }

    pub fn create_kernel(&self, name: &str) -> Kernel {
        create_kernel(self, name)
    }
}

pub struct Kernel {
    kernel: cl_kernel,
}

impl Drop for Kernel
{
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
}

pub fn create_kernel(program: & Program, kernel: & str) -> Kernel
{
    unsafe {
        let errcode = 0;
        // let bytes = str::to_bytes(kernel);
        kernel.to_c_str().with_ref(|str_ptr|
        {
            let kernel = clCreateKernel(program.prg,
                                        str_ptr,
                                        (&errcode));

            check(errcode, "Failed to create kernel!");

            Kernel {
                kernel: kernel,
            }
        })
    }
}

pub trait KernelArg {
  fn get_value(&self) -> (libc::size_t, *libc::c_void);
}

macro_rules! scalar_kernel_arg (
    ($t:ty) => (impl KernelArg for $t {
        fn get_value(&self) -> (libc::size_t, *libc::c_void) {
            (mem::size_of::<$t>() as libc::size_t,
             (self as *$t) as *libc::c_void)
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


pub struct Event
{
    event: cl_event,
}

impl Event {
    fn get_time(&self, param: cl_uint) -> u64
    {
        unsafe {
            let time: cl_ulong = 0;
            let ret = clGetEventProfilingInfo(self.event,
                                    param,
                                    mem::size_of::<cl_ulong>() as libc::size_t,
                                    (&time as *u64) as *libc::c_void,
                                    ptr::null());

            check(ret, "Failed to get profiling info");
            time as u64
        }
    }

    pub fn queue_time(&self) -> u64
    {
        self.get_time(CL_PROFILING_COMMAND_QUEUED)
    }

    pub fn submit_time(&self) -> u64
    {
        self.get_time(CL_PROFILING_COMMAND_SUBMIT)
    }

    pub fn start_time(&self) -> u64
    {
        self.get_time(CL_PROFILING_COMMAND_START)
    }

    pub fn end_time(&self) -> u64
    {
        self.get_time(CL_PROFILING_COMMAND_END)
    }
}

impl Drop for Event
{
    fn drop(&mut self) {
        unsafe {
            clReleaseEvent(self.event);
        }
    }
}

pub trait EventList {
    fn as_event_list<T>(&self, |*cl_event, cl_uint| -> T) -> T;

    fn wait(&self) {
        self.as_event_list(|p, len| {
            unsafe {
                let status = clWaitForEvents(len, p);
                check(status, "Error waiting for event(s)");
            }
        })
    }
}

impl<'r> EventList for &'r Event {
    fn as_event_list<T>(&self, f: |*cl_event, cl_uint| -> T) -> T
    {
        f(&self.event, 1 as cl_uint)
    }
}

impl EventList for Event {
    fn as_event_list<T>(&self, f: |*cl_event, cl_uint| -> T) -> T
    {
        f(&self.event, 1 as cl_uint)
    }
}

impl<T: EventList> EventList for Option<T> {
    fn as_event_list<T>(&self, f: |*cl_event, cl_uint| -> T) -> T
    {
        match *self {
            None => f(ptr::null(), 0),
            Some(ref s) => s.as_event_list(f)
        }
    }
}

impl<'r> EventList for &'r [Event] {
    fn as_event_list<T>(&self, f: |*cl_event, cl_uint| -> T) -> T
    {
        /* this is wasteful */
        let events = self.iter().map(|event| event.event).collect::<~[*libc::c_void]>();

        f(events.as_ptr() as **libc::c_void, events.len() as cl_uint)
    }
}

/* this seems VERY hackey */
impl EventList for () {
    fn as_event_list<T>(&self, f: |*cl_event, cl_uint| -> T) -> T
    {
        f(ptr::null(), 0)
    }
}


pub trait KernelIndex
{
    fn num_dimensions(dummy_self: Option<Self>) -> cl_uint;
    fn get_ptr(&self) -> *libc::size_t;
}

impl KernelIndex for int
{
    fn num_dimensions(_: Option<int>) -> cl_uint { 1 }

    fn get_ptr(&self) -> *libc::size_t
    {
        (self as *int) as *libc::size_t
    }
}

impl KernelIndex for (int, int) {
    fn num_dimensions(_: Option<(int, int)>) -> cl_uint { 2 }

    fn get_ptr(&self) -> *libc::size_t {
        (self as *(int, int)) as *libc::size_t
    }
}

impl KernelIndex for (int, int, int)
{
    fn num_dimensions(_: Option<(int, int, int)>) -> cl_uint { 3 }

    fn get_ptr(&self) -> *libc::size_t {
        (self as *(int, int, int)) as *libc::size_t
    }
}

impl KernelIndex for uint
{
    fn num_dimensions(_: Option<uint>) -> cl_uint { 1 }

    fn get_ptr(&self) -> *libc::size_t {
        (self as *uint) as *libc::size_t
    }
}

impl KernelIndex for (uint, uint)
{
    fn num_dimensions(_: Option<(uint, uint)>) -> cl_uint { 2 }

    fn get_ptr(&self) -> *libc::size_t {
        (self as *(uint, uint)) as *libc::size_t
    }
}

impl KernelIndex for (uint, uint, uint)
{
    fn num_dimensions(_: Option<(uint, uint, uint)>) -> cl_uint { 3 }

    fn get_ptr(&self) -> *libc::size_t {
        (self as *(uint, uint, uint)) as *libc::size_t
    }
}
