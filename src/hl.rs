//! A higher level API.

use libc;
use std::ffi::CString;
use std::iter::repeat;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::string::String;
use std::vec::Vec;

use cl;
use cl::*;
use cl::ll::*;
use cl::CLStatus::CL_SUCCESS;
use error::check;
use mem::{Put, Get, Write, Read, Buffer, CLBuffer};

#[derive(Copy, Clone)]
pub enum DeviceType {
      CPU, GPU
}

fn convert_device_type(device: DeviceType) -> cl_device_type {
    match device {
        DeviceType::CPU => CL_DEVICE_TYPE_CPU,
        DeviceType::GPU => CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR
    }
}

pub struct Platform {
    pub id: cl_platform_id
}

impl Platform {
    fn get_devices_internal(&self, dtype: cl_device_type) -> Vec<Device>
    {
        unsafe
        {
            let mut num_devices = 0;

            info!("Looking for devices matching {}", dtype);

            clGetDeviceIDs(self.id, dtype, 0, ptr::null_mut(),
                           (&mut num_devices));

            let mut ids: Vec<cl_device_id> = repeat(0 as cl_device_id)
                .take(num_devices as usize).collect();
            clGetDeviceIDs(self.id, dtype, ids.len() as cl_uint,
                           ids.as_mut_ptr(), (&mut num_devices));
            ids.iter().map(|id| { Device {id: *id }}).collect()
        }
    }

    pub fn get_devices(&self) -> Vec<Device>
    {
        self.get_devices_internal(CL_DEVICE_TYPE_ALL)
    }

    pub fn get_devices_by_types(&self, types: &[DeviceType]) -> Vec<Device>
    {
        let mut dtype = 0;
        for &t in types.iter() {
          dtype |= convert_device_type(t);
        }

        self.get_devices_internal(dtype)
    }

    fn profile_info(&self, name: cl_platform_info) -> String
    {
        unsafe {
            let mut size = 0 as libc::size_t;

            let status = clGetPlatformInfo(self.id,
                              name,
                              0,
                              ptr::null_mut(),
                              &mut size);
            check(status, "Could not determine platform info string length");

            let mut buf : Vec<u8>
                = repeat(0u8).take(size as usize).collect();

            let status = clGetPlatformInfo(self.id,
                              name,
                              size,
                              buf.as_mut_ptr() as *mut libc::c_void,
                              ptr::null_mut());
            check(status, "Could not get platform info string");

            String::from_utf8_unchecked(buf)
        }
    }

    pub fn get_id(&self) -> cl_platform_id {
        self.id
    }

    pub fn name(&self) -> String
    {
        self.profile_info(CL_PLATFORM_NAME)
    }

    pub fn version(&self) -> String
    {
        self.profile_info(CL_PLATFORM_VERSION)
    }

    pub fn profile(&self) -> String
    {
        self.profile_info(CL_PLATFORM_PROFILE)
    }

    pub fn vendor(&self) -> String
    {
        self.profile_info(CL_PLATFORM_VENDOR)
    }

    pub fn extensions(&self) -> String
    {
        self.profile_info(CL_PLATFORM_EXTENSIONS)
    }

    pub unsafe fn from_platform_id(id: cl_platform_id) -> Platform {
        Platform { id: id }
    }
}

// This mutex is used to work around weak OpenCL implementations.
// On some implementations concurrent calls to clGetPlatformIDs
// will cause the implantation to return invalid status.
static mut platforms_mutex: std::sync::StaticMutex = std::sync::MUTEX_INIT;

pub fn get_platforms() -> Vec<Platform>
{
    let mut num_platforms = 0 as cl_uint;

    unsafe
    {
        let guard = platforms_mutex.lock();
        let status = clGetPlatformIDs(0,
                                          ptr::null_mut(),
                                          (&mut num_platforms));
        // unlock this before the check in case the check fails
        check(status, "could not get platform count.");

        let mut ids: Vec<cl_device_id> = repeat(0 as cl_device_id)
            .take(num_platforms as usize).collect();

        let status = clGetPlatformIDs(num_platforms,
                                      ids.as_mut_ptr(),
                                      (&mut num_platforms));
        check(status, "could not get platforms.");

        let _ = guard;

        ids.iter().map(|id| { Platform { id: *id } }).collect()
    }
}

pub fn create_context_with_properties(dev: &[Device], prop: &[cl_context_properties]) -> Context
{
    unsafe
    {
        // TODO: Support for multiple devices
        let mut errcode = 0;
        let dev: Vec<cl_device_id> = dev.iter().map(|dev| dev.id).collect();

        // TODO: Proper error messages
        let ctx = clCreateContext(&prop[0],
                                  dev.len() as u32,
                                  &dev[0],
                                  mem::transmute(ptr::null::<fn()>()),
                                  ptr::null_mut(),
                                  &mut errcode);

        check(errcode, "Failed to create opencl context!");

        Context { ctx: ctx }
    }
}

#[derive(Copy, Clone)]
pub struct Device {
    pub id: cl_device_id
}

unsafe impl Sync for Device {}
unsafe impl Send for Device {}

impl Device {
    fn profile_info(&self, name: cl_device_info) -> String
    {
        unsafe {
            let mut size = 0 as libc::size_t;

            let status = clGetDeviceInfo(
                self.id,
                name,
                0,
                ptr::null_mut(),
                &mut size);
            check(status, "Could not determine device info string length");

            let mut buf : Vec<u8>
                = repeat(0u8).take(size as usize).collect();

            let status = clGetDeviceInfo(self.id,
                                         name,
                                         size,
                                         buf.as_mut_ptr() as *mut libc::c_void,
                                         ptr::null_mut());
            check(status, "Could not get device info string");

            String::from_utf8_unchecked(buf)
        }
    }

    pub fn name(&self) -> String
    {
        self.profile_info(CL_DEVICE_NAME)
    }
    pub fn vendor(&self) -> String
    {
        self.profile_info(CL_DEVICE_VENDOR)
    }
    pub fn profile(&self) -> String
    {
        self.profile_info(CL_DEVICE_PROFILE)
    }
    pub fn device_type(&self) -> String
    {
        self.profile_info(CL_DEVICE_TYPE)
    }

    pub fn compute_units(&self) -> usize {
		unsafe {
			let mut ct: usize = 0;
            let status = clGetDeviceInfo(
                self.id,
                CL_DEVICE_MAX_COMPUTE_UNITS,
                8,
                (&mut ct as *mut usize) as *mut libc::c_void,
                ptr::null_mut());
            check(status, "Could not get number of device compute units.");
			return ct;
		}
	}


    pub fn create_context(&self) -> Context
    {
        unsafe
        {
            // TODO: Support for multiple devices
            let mut errcode = 0;

            // TODO: Proper error messages
            let ctx = clCreateContext(ptr::null(),
                                      1,
                                      &self.id,
                                      mem::transmute(ptr::null::<fn()>()),
                                      ptr::null_mut(),
                                      (&mut errcode));

            check(errcode, "Failed to create opencl context!");

            Context { ctx: ctx }
        }
    }
}

pub struct Context {
    pub ctx: cl_context,
}

unsafe impl Sync for Context {}
unsafe impl Send for Context {}

impl Context {
    pub fn create_buffer<T>(&self, size: usize, flags: cl_mem_flags) -> CLBuffer<T>
    {
        unsafe {
            let mut status = 0;
            let buf = clCreateBuffer(self.ctx,
                                     flags,
                                     (size*mem::size_of::<T>()) as libc::size_t ,
                                     ptr::null_mut(),
                                     (&mut status));
            check(status, "Could not allocate buffer");
            CLBuffer {
                cl_buffer: buf,
                phantom: PhantomData,
            }
        }
    }


    pub fn create_buffer_from<T, U, IN: Put<T, U>>(&self, create: IN, flags: cl_mem_flags) -> U
    {
        create.put(|p, len| {
            let mut status = 0;
            let buf = unsafe {
                clCreateBuffer(self.ctx,
                               flags | CL_MEM_COPY_HOST_PTR,
                               len,
                               mem::transmute(p),
                               (&mut status))
            };
            check(status, "Could not allocate buffer");
            buf
        })
    }

    pub fn create_command_queue(&self, device: &Device) -> CommandQueue
    {
        unsafe
        {
            let mut errcode = 0;

            let cqueue = clCreateCommandQueue(self.ctx,
                                              device.id,
                                              CL_QUEUE_PROFILING_ENABLE,
                                              (&mut errcode));

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
            let src = CString::new(src).unwrap();

            let mut status = CL_SUCCESS as cl_int;
            let program = clCreateProgramWithSource(
                self.ctx,
                1,
                &src.as_ptr(),
                ptr::null(),
                (&mut status));
            check(status, "Could not create program");

            Program { prg: program }
        }
    }

    pub fn create_program_from_binary(&self, bin: &str, device: &Device) -> Program {
        let src = CString::new(bin).unwrap();
        let mut status = CL_SUCCESS as cl_int;
        let len = bin.len() as libc::size_t;
        let program = unsafe {
            clCreateProgramWithBinary(
                self.ctx,
                1,
                &device.id,
                (&len),
                (src.as_ptr() as *const *const i8) as *const *const libc::c_uchar,
                ptr::null_mut(),
                (&mut status))
        };
        check(status, "Could not create program");

        Program {prg: program}
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

impl<'r, T> KernelArg for &'r (Buffer<T> + 'r) {
    fn get_value(&self) -> (libc::size_t, *const libc::c_void)
    {
        unsafe {
            (mem::size_of::<cl_mem>() as libc::size_t,
             self.id_ptr() as *const libc::c_void)
        }
    }
}

impl<'r, T> KernelArg for Box<Buffer<T> + 'r> {
    fn get_value(&self) -> (libc::size_t, *const libc::c_void)
    {
        unsafe {
            (mem::size_of::<cl_mem>() as libc::size_t,
             self.id_ptr() as *const libc::c_void)
        }
    }
}


pub struct CommandQueue {
    pub cqueue: cl_command_queue
}

unsafe impl Sync for CommandQueue {}
unsafe impl Send for CommandQueue {}

impl CommandQueue
{
    //synchronous
    pub fn enqueue_kernel<I: KernelIndex, E: EventList>(&self, k: &Kernel, global: I, local: Option<I>, wait_on: E)
        -> Event
    {
        unsafe
        {
            wait_on.as_event_list(|event_list, event_list_length| {
                let mut e: cl_event = ptr::null_mut();
                let mut status = clEnqueueNDRangeKernel(
                    self.cqueue,
                    k.kernel,
                    KernelIndex::num_dimensions(None::<I>),
                    ptr::null(),
                    global.get_ptr(),
                    match local {
                        Some(ref l) => l.get_ptr() as *const libc::size_t,
                        None => ptr::null()
                    },
                    event_list_length,
                    event_list,
                    (&mut e));
                check(status, "Error enqueuing kernel.");
                status = clFinish(self.cqueue);
                check(status, "Error finishing kernel.");
                Event { event: e }
            })
        }
    }

    //asynchronous
    pub fn enqueue_async_kernel<I: KernelIndex, E: EventList>(&self, k: &Kernel, global: I, local: Option<I>, wait_on: E)
        -> Event
    {
        unsafe
        {
            wait_on.as_event_list(|event_list, event_list_length| {
                let mut e: cl_event = ptr::null_mut();
                let status = clEnqueueNDRangeKernel(
                    self.cqueue,
                    k.kernel,
                    KernelIndex::num_dimensions(None::<I>),
                    ptr::null(),
                    global.get_ptr(),
                    match local {
                        Some(ref l) => l.get_ptr() as *const libc::size_t,
                        None => ptr::null()
                    },
                    event_list_length,
                    event_list,
                    (&mut e));
                check(status, "Error enqueuing kernel.");
                Event { event: e }
            })
        }
    }

    pub fn get<T, U, B: Buffer<T>, G: Get<B, U>, E: EventList>(&self, buf: &B, event: E) -> G
    {
        event.as_event_list(|event_list, event_list_length| {
            Get::get(buf, |offset, ptr, len| {
                unsafe {
                    let err = clEnqueueReadBuffer(self.cqueue,
                                                  buf.id(),
                                                  CL_TRUE,
                                                  offset as libc::size_t,
                                                  len,
                                                  ptr,
                                                  event_list_length,
                                                  event_list,
                                                  ptr::null_mut());

                    check(err, "Failed to read buffer");
                }
            })
        })
    }

    pub fn write<U: Write, T, E: EventList, B: Buffer<T>>(&self, mem: &B, write: &U, event: E)
    {
        unsafe {
            event.as_event_list(|event_list, event_list_length| {
                write.write(|offset, p, len| {
                    let err = clEnqueueWriteBuffer(self.cqueue,
                                                   mem.id(),
                                                   CL_TRUE,
                                                   offset as libc::size_t,
                                                   len as libc::size_t,
                                                   p as *const libc::c_void,
                                                   event_list_length,
                                                   event_list,
                                                   ptr::null_mut());

                    check(err, "Failed to write buffer");
                })
            })
        }
    }

    pub fn write_async<U: Write, T, E: EventList, B: Buffer<T>>(&self, mem: &B, write: &U, event: E) -> Event
    {
        let mut out_event = None;
        unsafe {
            event.as_event_list(|evt, evt_len| {
                write.write(|offset, p, len| {
                    let mut e: cl_event = ptr::null_mut();
                    let err = clEnqueueWriteBuffer(self.cqueue,
                                                   mem.id(),
                                                   CL_FALSE,
                                                   offset as libc::size_t,
                                                   len as libc::size_t,
                                                   p as *const libc::c_void,
                                                   evt_len,
                                                   evt,
                                                   &mut e);
                    out_event = Some(e);
                    check(err, "Failed to write buffer");
                })
            })
        }
        Event { event: out_event.unwrap() }
    }

    pub fn read<T, U: Read, E: EventList, B: Buffer<T>>(&self, mem: &B, read: &mut U, event: E)
    {
        event.as_event_list(|event_list, event_list_length| {
                read.read(|offset, p, len| {
                        unsafe {
                            let err = clEnqueueReadBuffer(self.cqueue,
                                                          mem.id(),
                                                          CL_TRUE,
                                                          offset as libc::size_t,
                                                          len as libc::size_t,
                                                          p as *mut libc::c_void,
                                                          event_list_length,
                                                          event_list,
                                                          ptr::null_mut());

                            check(err, "Failed to read buffer");
                        }
                    })
            })
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


/// Represents an OpenCL program, which is a collection of kernels.
///
/// Create these using
/// [`Context::create_program_from_source`](struct.Context.html#method.create_program_from_source)
/// or
/// [`Context::create_program_from_binary`](struct.Context.html#method.create_program_from_binary).
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
    /// Build the program for a given device.
    ///
    /// Both Ok and Err returns include the build log.
    pub fn build(&self, device: &Device) -> Result<String, String>
    {
        unsafe
        {
            let ret = clBuildProgram(self.prg, 1, &device.id,
                                     ptr::null(),
                                     mem::transmute(ptr::null::<fn()>()),
                                     ptr::null_mut());
            // Get the build log.
            let mut size = 0 as libc::size_t;
            let status = clGetProgramBuildInfo(
                self.prg,
                device.id,
                CL_PROGRAM_BUILD_LOG,
                0,
                ptr::null_mut(),
                (&mut size));
            check(status, "Could not get build log");

            let mut buf : Vec<u8> = repeat(0u8).take(size as usize).collect();
            let status = clGetProgramBuildInfo(
                self.prg,
                device.id,
                CL_PROGRAM_BUILD_LOG,
                buf.len() as libc::size_t,
                buf.as_mut_ptr() as *mut libc::c_void,
                ptr::null_mut());
            check(status, "Could not get build log");

            let log = String::from_utf8_lossy(&buf[..]);
            if ret == CL_SUCCESS as cl_int {
                Ok(log.into_owned())
            } else {
                Err(log.into_owned())
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
    pub fn set_arg<T: KernelArg>(&self, i: usize, x: &T)
    {
        set_kernel_arg(self, i as cl::cl_uint, x)
    }
}

pub fn create_kernel(program: &Program, kernel: & str) -> Kernel
{
    unsafe {
        let mut errcode = 0;
        let str = CString::new(kernel).unwrap();
        let kernel = clCreateKernel(program.prg,
                                    str.as_ptr(),
                                    (&mut errcode));

        check(errcode, "Failed to create kernel!");

        Kernel { kernel: kernel }
    }
}

pub trait KernelArg {
  fn get_value(&self) -> (libc::size_t, *const libc::c_void);
}

macro_rules! scalar_kernel_arg (
    ($t:ty) => (impl KernelArg for $t {
        fn get_value(&self) -> (libc::size_t, *const libc::c_void) {
            (mem::size_of::<$t>() as libc::size_t,
             (self as *const $t) as *const libc::c_void)
        }
    })
);

scalar_kernel_arg!(isize);
scalar_kernel_arg!(usize);
scalar_kernel_arg!(u32);
scalar_kernel_arg!(u64);
scalar_kernel_arg!(i32);
scalar_kernel_arg!(i64);
scalar_kernel_arg!(f32);
scalar_kernel_arg!(f64);
scalar_kernel_arg!([f32; 2]);
scalar_kernel_arg!([f64; 2]);

impl KernelArg for [f32; 3] {
    fn get_value(&self) -> (libc::size_t, *const libc::c_void) {
        (4 * mem::size_of::<f32>() as libc::size_t,
          (self as *const f32) as *const libc::c_void)
    }
}

impl KernelArg for [f64; 3] {
    fn get_value(&self) -> (libc::size_t, *const libc::c_void) {
        (4 * mem::size_of::<f64>() as libc::size_t,
          (self as *const f64) as *const libc::c_void)
    }
}

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
    pub event: cl_event,
}

impl Event {
    fn get_time(&self, param: cl_uint) -> u64
    {
        unsafe {
            let mut time: cl_ulong = 0;
            let ret = clGetEventProfilingInfo(self.event,
                                    param,
                                    mem::size_of::<cl_ulong>() as libc::size_t,
                                    (&mut time as *mut u64) as *mut libc::c_void,
                                    ptr::null_mut());

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
    fn as_event_list<T, F: FnOnce(*const cl_event, cl_uint) -> T>(&self, F) -> T;

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
    fn as_event_list<T, F>(&self, f: F) -> T
        where F: FnOnce(*const cl_event, cl_uint) -> T
    {
        f(&self.event, 1 as cl_uint)
    }
}

impl EventList for Event {
    fn as_event_list<T, F>(&self, f: F) -> T
        where F: FnOnce(*const cl_event, cl_uint) -> T
    {
        f(&self.event, 1 as cl_uint)
    }
}

impl<T: EventList> EventList for Option<T> {
    fn as_event_list<T2, F>(&self, f: F) -> T2
        where F: FnOnce(*const cl_event, cl_uint) -> T2
    {
        match *self {
            None => f(ptr::null(), 0),
            Some(ref s) => s.as_event_list(f)
        }
    }
}

impl<'r> EventList for &'r [Event] {
    fn as_event_list<T, F>(&self, f: F) -> T
        where F: FnOnce(*const cl_event, cl_uint) -> T
    {
        let mut vec: Vec<cl_event> = Vec::with_capacity(self.len());
        for item in self.iter(){
            vec.push(item.event);
        }

        f(vec.as_ptr(), vec.len() as cl_uint)
    }
}

/* this seems VERY hackey */
impl EventList for () {
    fn as_event_list<T, F>(&self, f: F) -> T
        where F: FnOnce(*const cl_event, cl_uint) -> T
    {
        f(ptr::null(), 0)
    }
}


pub trait KernelIndex
{
    fn num_dimensions(dummy_self: Option<Self>) -> cl_uint where Self: Sized;
    fn get_ptr(&self) -> *const libc::size_t;
}

impl KernelIndex for isize
{
    fn num_dimensions(_: Option<isize>) -> cl_uint { 1 }

    fn get_ptr(&self) -> *const libc::size_t
    {
        (self as *const isize) as *const libc::size_t
    }
}

impl KernelIndex for (isize, isize) {
    fn num_dimensions(_: Option<(isize, isize)>) -> cl_uint { 2 }

    fn get_ptr(&self) -> *const libc::size_t {
        (self as *const (isize, isize)) as *const libc::size_t
    }
}

impl KernelIndex for (isize, isize, isize)
{
    fn num_dimensions(_: Option<(isize, isize, isize)>) -> cl_uint { 3 }

    fn get_ptr(&self) -> *const libc::size_t {
        (self as *const (isize, isize, isize)) as *const libc::size_t
    }
}

impl KernelIndex for usize
{
    fn num_dimensions(_: Option<usize>) -> cl_uint { 1 }

    fn get_ptr(&self) -> *const libc::size_t {
        (self as *const usize) as *const libc::size_t
    }
}

impl KernelIndex for (usize, usize)
{
    fn num_dimensions(_: Option<(usize, usize)>) -> cl_uint { 2 }

    fn get_ptr(&self) -> *const libc::size_t {
        (self as *const (usize, usize)) as *const libc::size_t
    }
}

impl KernelIndex for (usize, usize, usize)
{
    fn num_dimensions(_: Option<(usize, usize, usize)>) -> cl_uint { 3 }

    fn get_ptr(&self) -> *const libc::size_t {
        (self as *const (usize, usize, usize)) as *const libc::size_t
    }
}
