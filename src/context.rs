use std::mem;
use std::ptr;
use std::vec::Vec;

use cl::*;
use cl::ll::*;
use error::check;
use device::Device;

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

    /// The underlying OpenCL context identifier.
    pub fn cl_id(&self) -> cl_context {
        self.ctx
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
