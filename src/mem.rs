//! High level buffer management.

use libc::{size_t, c_void};
use std::marker::{PhantomData};
use std::mem;
use std::ptr;
use std::vec::Vec;

use cl::*;
use cl::ll::*;

use hl::KernelArg;
use error::check;

/// Trait implemented by valid OpenCL buffer ojects.
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

/// An 1-dimenisonal device-side-only OpenCL buffer object.
pub struct CLBuffer<T> {
    cl_buffer: cl_mem,
    phantom: PhantomData<T>,
}

impl<T> CLBuffer<T> {
    /// Unsafely wraps an OpenCL buffer object ID.
    /// 
    /// The provided identifier is not checked.
    pub unsafe fn new(buffer: cl_mem) -> CLBuffer<T> {
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

/* memory life cycle
 * | Trait  | Exists in rust | Exists in OpenCL | Direction      |
 * | Put    | X              |                  | rust -> opencl |
 * | Get    |                | X                | opencl -> rust |
 * | Write  | X              | X                | rust -> opencl |
 * | Read   | X              | X                | opencl -> rust |
 *mut */

/// Internal trait implemented by host-side buffer objects that can be transfered to the device, yielding a
/// device-side buffer object of type `B`.
pub trait Put<T, B> {
    /// Writes data to the device, and returns the corresponding device-side
    /// buffer.
    ///
    /// The provided callback of type `F` is responsible for actually writing raw data to the device.
    /// Do not use this directly, refer to the corresponding `CommandQueue::put` method.
    fn put<F>(&self, F) -> B
        where F: FnOnce(*const c_void, size_t) -> cl_mem;
}

/// Internal trait implemented by host-side buffer objects that can be retrieved from a device-side buffer
/// object of type `B`.
pub trait Get<B, T> {
    /// Reads data from the device, and returns the corresponding host-side
    /// buffer.
    ///
    /// The provided callback of type `F` is responsible for actually reading raw data from the device.
    /// Do not use this directly, refer to the corresponding `CommandQueue::get` method.
    fn get<F: FnOnce(size_t, *mut c_void, size_t)>(mem: &B, F) -> Self;
}

/// Internal trait implemented by host-side memory objects that can be written to a device-side memory
/// object.
pub trait Write {
    /// Writes data from `self` to the device.
    ///
    /// The provided callback of type `F` is responsible for actually writing raw data to the device.
    /// Do not use this directly, refer to the corresponding `CommandQueue::write` method.
    fn write<F: FnOnce(size_t, *const c_void, size_t)>(&self, F);
}

/// Internal trait implemented by host-side memory objects that can be filled with data read from the device.
pub trait Read {
    /// Reads data to `self` from the device.
    ///
    /// The provided callback of type `F` is responsible for actually reading raw data from the device.
    /// Do not use this directly, refer to the corresponding `CommandQueue::read` method.
    fn read<F: FnOnce(size_t, *mut c_void, size_t)>(&mut self, F);
}

impl<'r, T> Put<T, CLBuffer<T>> for &'r [T]
{
    fn put<F>(&self, f: F) -> CLBuffer<T>
        where F: FnOnce(*const c_void, size_t) -> cl_mem
    {
        CLBuffer {
            cl_buffer: f(self.as_ptr() as *const c_void,
                         (self.len() * mem::size_of::<T>()) as size_t),
            phantom: PhantomData,
        }
    }
}

impl<'r, T> Put<T, CLBuffer<T>> for &'r Vec<T>
{
    fn put<F>(&self, f: F) -> CLBuffer<T>
        where F: FnOnce(*const c_void, size_t) -> cl_mem
    {
        CLBuffer {
            cl_buffer: f(self.as_ptr() as *const c_void, (self.len() * mem::size_of::<T>()) as size_t),
            phantom: PhantomData,
        }
    }
}

impl<T> Put<T, CLBuffer<T>> for Vec<T>
{
    fn put<F>(&self, f: F) -> CLBuffer<T>
        where F: FnOnce(*const c_void, size_t) -> cl_mem
    {
        CLBuffer {
            cl_buffer: f(self.as_ptr() as *const c_void, (self.len() * mem::size_of::<T>()) as size_t),
            phantom: PhantomData,
        }
    }
}

impl<T> Get<CLBuffer<T>, T> for Vec<T>
{
    fn get<F>(mem: &CLBuffer<T>, f: F) -> Vec<T>
        where F: FnOnce(size_t, *mut c_void, size_t)
    {
        let mut v: Vec<T> = Vec::with_capacity(mem.len());
        unsafe {
            v.set_len(mem.len());
        }
        f(0, v.as_ptr() as *mut c_void, (v.len() * mem::size_of::<T>()) as size_t);
        v
    }
}

impl<'r, T> Write for &'r [T]
{
    fn write<F>(&self, f: F)
        where F: FnOnce(size_t, *const c_void, size_t)
    {
        f(0, self.as_ptr() as *const c_void, (self.len() * mem::size_of::<T>()) as size_t)
    }
}

impl<'r, T> Read for &'r mut [T]
{
    fn read<F>(&mut self, f: F)
        where F: FnOnce(size_t, *mut c_void, size_t)
    {
        let p = (*self).as_mut_ptr();
        let len = self.len();
        f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t)
    }
}

macro_rules! get_arg (
    ($t:ty) => (impl Get<CLBuffer<$t>, $t> for $t
        {
            fn get<F>(_: &CLBuffer<$t>, f: F) -> $t
                where F: FnOnce(size_t, *mut c_void, size_t)
            {
                let mut v: $t = 0 as $t;
                f(0, (&mut v as *mut $t) as *mut c_void, mem::size_of::<$t>() as size_t);
                v as $t
            }
        })
);

get_arg!(isize);
get_arg!(usize);
get_arg!(u32);
get_arg!(u64);
get_arg!(i32);
get_arg!(i64);
get_arg!(f32);
get_arg!(f64);

macro_rules! put_arg (
    ($t:ty) => (impl Put<$t, CLBuffer<$t>> for $t
        {
            fn put<F>(&self, f: F) -> CLBuffer<$t>
                where F: FnOnce(*const c_void, size_t) -> cl_mem
            {
                CLBuffer {
                    cl_buffer: f((self as *const $t) as *const c_void, mem::size_of::<$t>() as size_t),
                    phantom: PhantomData,
                }
            }
        }
    )
);

put_arg!(isize);
put_arg!(usize);
put_arg!(u32);
put_arg!(u64);
put_arg!(i32);
put_arg!(i64);
put_arg!(f32);
put_arg!(f64);

macro_rules! read_arg (
    ($t:ty) => (impl Read for $t
        {
            fn read<F>(&mut self, f: F)
                where F: FnOnce(size_t, *mut c_void, size_t)
            {
                f(0, (self as *mut $t) as *mut c_void, mem::size_of::<$t>() as size_t)
            }
        }
    )
);

read_arg!(isize);
read_arg!(usize);
read_arg!(u32);
read_arg!(u64);
read_arg!(i32);
read_arg!(i64);
read_arg!(f32);
read_arg!(f64);

macro_rules! write_arg (
    ($t:ty) => (impl Write for $t
        {
            fn write<F>(&self, f: F) 
                where F: FnOnce(size_t, *const c_void, size_t)
            {
                f(0, (self as *const $t) as *const c_void, mem::size_of::<$t>() as size_t)
            }
        }
    )
);

write_arg!(isize);
write_arg!(usize);
write_arg!(u32);
write_arg!(u64);
write_arg!(i32);
write_arg!(i64);
write_arg!(f32);
write_arg!(f64);
