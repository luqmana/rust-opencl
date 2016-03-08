use libc::{size_t, c_void};
use std::marker::{PhantomData};
use std::mem;
use std::ptr;

use cl::*;
use cl::ll::*;

use program::KernelArg;
use error::check;

/// Trait implemented by valid OpenCL buffer objects.
pub trait Buffer<T> {
    /// A pointer to the underlying OpenCL buffer object.
    unsafe fn id_ptr(&self) -> *const cl_mem;

    /// The underlying buffer object.
    fn id(&self) -> cl_mem {
        unsafe {
            *self.id_ptr()
        }
    }

    /// The length in bytes of this buffer object.
    fn byte_len(&self) -> size_t
    {
        unsafe {
            let mut size : size_t = 0;
            let err = clGetMemObjectInfo(self.id(),
                                         CL_MEM_SIZE,
                                         mem::size_of::<size_t>() as size_t,
                                         (&mut size as *mut size_t) as *mut c_void,
                                         ptr::null_mut());

            check(err, "Failed to read memory size");
            size
        }
    }

    /// The number of element of type `T` on this buffer object.
    fn len(&self) -> usize { self.byte_len() as usize / mem::size_of::<T>() }
}

/// An 1-dimenisonal device-side OpenCL buffer object.
pub struct CLBuffer<T> {
    cl_buffer: cl_mem,
    phantom: PhantomData<T>,
}

impl<T> CLBuffer<T> {
    /// Unsafely wraps an OpenCL buffer object ID.
    /// 
    /// The provided identifier is not checked.
    pub unsafe fn new_unchecked(buffer: cl_mem) -> CLBuffer<T> {
        CLBuffer {
            cl_buffer: buffer,
            phantom:   PhantomData
        }
    }
}

impl<T> Drop for CLBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            let status = clReleaseMemObject(self.cl_buffer);
            check(status, "Could not release the buffer");
        }
    }
}

impl<T> Buffer<T> for CLBuffer<T> {
    unsafe fn id_ptr(&self) -> *const cl_mem
    {
        &self.cl_buffer as *const cl_mem
    }
}

impl<T> KernelArg for CLBuffer<T> {
    fn get_value(&self) -> (size_t, *const c_void)
    {
        unsafe {
            (mem::size_of::<cl_mem>() as size_t,
             self.id_ptr() as *const c_void)
        }
    }
}
