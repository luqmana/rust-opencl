use libc;
use std::ffi::CString;
use std::mem;
use std::ptr;
use std::vec::Vec;

use cl::*;
use cl::ll::*;
use cl::CLStatus::CL_SUCCESS;
use error::check;
use mem::{Put, CLBuffer, CLBuffer1D, CLBuffer2D, CLBuffer3D};
use device::Device;
use command_queue::CommandQueue;
use program::Program;

/// An OpenCL context.
pub struct Context {
    ctx: cl_context
}

unsafe impl Sync for Context {}
unsafe impl Send for Context {}

impl Context {
    /// Creates a context for a device.
    pub fn new(dev: &Device) -> Context {
        let mut errcode = 0;
        let ctx = unsafe {
            clCreateContext(ptr::null(),
                            1,
                            &dev.cl_id(),
                            mem::transmute(ptr::null::<fn()>()),
                            ptr::null_mut(),
                            &mut errcode)
        };

        check(errcode, "Failed to create opencl context!");

        Context { ctx: ctx }
    }

    /// Creates an OpenCL context for a set of devices with the given properties.
    pub fn with_properties(dev: &[Device], prop: &[cl_context_properties]) -> Context {
        unsafe
        {
            // TODO: Support for multiple devices
            let mut errcode = 0;
            let dev: Vec<cl_device_id> = dev.iter().map(|dev| dev.cl_id()).collect();

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

    /// Creates a buffer on this context.
    pub fn create_buffer1d<T>(&self, size: usize, flags: cl_mem_flags) -> CLBuffer1D<T> {
        unsafe {
            let mut status = 0;
            let buf = clCreateBuffer(self.ctx,
                                     flags,
                                     (size * mem::size_of::<T>()) as libc::size_t ,
                                     ptr::null_mut(),
                                     (&mut status));
            check(status, "Could not allocate buffer");

            CLBuffer1D::new_unchecked(buf)
        }
    }

    /// Creates a buffer on this context.
    pub fn create_buffer2d<T>(&self, width: usize, height: usize, flags: cl_mem_flags) -> CLBuffer2D<T> {
        let buf: CLBuffer1D<T> = self.create_buffer1d(width * height, flags);

        unsafe { CLBuffer2D::new_unchecked(width, height, buf.cl_id()) }
    }

    /// Creates a buffer on this context.
    pub fn create_buffer3d<T>(&self, width: usize, height: usize, depth: usize, flags: cl_mem_flags) -> CLBuffer3D<T> {
        let buf: CLBuffer1D<T> = self.create_buffer1d(width * height * depth, flags);

        unsafe { CLBuffer3D::new_unchecked(width, height, depth, buf.cl_id()) }
    }


    /// Creates a buffer on this context.
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

    /// Creates a command queue for a device attached to this context.
    pub fn create_command_queue(&self, device: &Device) -> CommandQueue
    {
        unsafe
        {
            let mut errcode = 0;

            let cqueue = clCreateCommandQueue(self.ctx,
                                              device.cl_id(),
                                              CL_QUEUE_PROFILING_ENABLE,
                                              (&mut errcode));

            check(errcode, "Failed to create command queue!");

            CommandQueue::new_unchecked(cqueue)
        }
    }

    /// Creates a program from its OpenCL C source code.
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
            check(status, "Could not create the program");

            Program::new_unchecked(program)
        }
    }

    /// Creates a program from its pre-compiled binaries.
    pub fn create_program_from_binary(&self, bin: &str, device: &Device) -> Program {
        let src = CString::new(bin).unwrap();
        let mut status = CL_SUCCESS as cl_int;
        let len = bin.len() as libc::size_t;
        let program = unsafe {
            clCreateProgramWithBinary(
                self.ctx,
                1,
                &device.cl_id(),
                (&len),
                (src.as_ptr() as *const *const i8) as *const *const libc::c_uchar,
                ptr::null_mut(),
                (&mut status))
        };
        check(status, "Could not create the program");

        unsafe {
            Program::new_unchecked(program)
        }
    }
}

impl Drop for Context
{
    fn drop(&mut self) {
        unsafe {
            let status = clReleaseContext(self.ctx);
            check(status, "Could not release the context");
        }
    }
}
