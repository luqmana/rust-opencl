use libc::{size_t, c_void};
use std::marker::PhantomData;
use std::mem;
use std::ptr;

use cl::*;
use cl::ll::*;

use context::Context;
use program::KernelArg;
use error::check;

/// Trait implemented by objects that can be used to initialize a device-side buffer.
pub trait BufferData<T: Copy> {
    /// Returns the raw data representation of this buffer and its size in bytes.
    fn as_raw_data<F, O>(&self, f: F) -> O
        where F: FnOnce(*const c_void, size_t) -> O;

    /// Returns the mutable raw data representation of this buffer and its size in bytes.
    fn as_raw_data_mut<F, O>(&mut self, f: F) -> O
        where F: FnOnce(*mut c_void, size_t) -> O;

    /// The number of element of type `T` on this buffer.
    fn len(&self) -> size_t;

    /// The number of bytes on this buffer object.
    fn bytes_len(&self) -> size_t {
        self.len() * mem::size_of::<T>() as size_t
    }
}

impl<T: Copy> BufferData<T> for [T] {
    fn as_raw_data<F, O>(&self, f: F) -> O
        where F: FnOnce(*const c_void, size_t) -> O {
        f(self.as_ptr() as *const c_void, self.bytes_len())
    }

    fn as_raw_data_mut<F, O>(&mut self, f: F) -> O
        where F: FnOnce(*mut c_void, size_t) -> O {
        f(self.as_mut_ptr() as *mut c_void, self.bytes_len())
    }

    fn len(&self) -> size_t {
        self.len() as size_t
    }
}

impl<T: Copy> BufferData<T> for Vec<T> {
    fn as_raw_data<F, O>(&self, f: F) -> O
        where F: FnOnce(*const c_void, size_t) -> O {
        f(self.as_ptr() as *const c_void, self.bytes_len())
    }

    fn as_raw_data_mut<F, O>(&mut self, f: F) -> O
        where F: FnOnce(*mut c_void, size_t) -> O {
        f(self.as_mut_ptr() as *mut c_void, self.bytes_len())
    }

    fn len(&self) -> size_t {
        self.len() as size_t
    }
}

/// A device-side OpenCL buffer object.
pub struct Buffer<T> {
    len:       size_t,
    cl_buffer: cl_mem,
    phantom:   PhantomData<T>,
}

impl<T: Copy> Buffer<T> {
    /// Creates a buffer initialized with the content of `data`.
    pub fn new<D: ?Sized>(context: &Context, data: &D, flags: cl_mem_flags) -> Buffer<T>
        where D: BufferData<T> {

        data.as_raw_data(|raw_data, sz| {
            let mut status = 0;

            let mem = unsafe {
                clCreateBuffer(context.cl_id(),
                               flags | CL_MEM_COPY_HOST_PTR,
                               sz,
                               mem::transmute(raw_data),
                               &mut status)
            };

            check(status, "Could not allocate buffer");

            Buffer {
                len:       data.len(),
                cl_buffer: mem,
                phantom:   PhantomData
            }
        })
    }

    // FIXME: should be unsafe?
    /// Creates a new uninitialized 1-dimensional buffer.
    pub fn new_uninitialized(context: &Context, len: usize, flags: cl_mem_flags) -> Buffer<T> {
        let mut status = 0;
        let mem = unsafe {
            clCreateBuffer(context.cl_id(),
                           flags,
                           (len * mem::size_of::<T>()) as size_t,
                           ptr::null_mut(),
                           &mut status)
        };

        check(status, "Could not allocate buffer");

        Buffer {
            len:       len as size_t,
            cl_buffer: mem,
            phantom:   PhantomData
        }
    }

    /// The underlying OpenCL identifier.
    pub fn cl_id_ptr(&self) -> cl_mem {
        self.cl_buffer
    }

    /// The underlying OpenCL identifier.
    pub fn cl_id(&self) -> cl_mem {
        self.cl_buffer
    }

    /// The length in bytes of this buffer.
    pub fn bytes_len(&self) -> size_t {
        self.len * mem::size_of::<T>() as size_t
    }

    /// The number of elements of type `T` on this buffer.
    pub fn len(&self) -> size_t {
        self.len
    }
}

impl<T> Drop for Buffer<T> {
    fn drop(&mut self) {
        unsafe {
            let status = clReleaseMemObject(self.cl_buffer);
            check(status, "Could not release the buffer");
        }
    }
}

impl<T> KernelArg for Buffer<T> {
    fn get_value(&self) -> (size_t, *const c_void) {
        (mem::size_of::<cl_mem>() as size_t, &self.cl_buffer as *const cl_mem as *const c_void)
    }
}
