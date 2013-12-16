use std::libc::{size_t, c_void};
use std::mem;
use std::ptr;
use std::vec;
use std::num::zero;

use CL::*;
use CL::ll::*;

use hl::KernelArg;
use error::check;

pub trait Buffer<T> {
    unsafe fn id_ptr(&self) -> *cl_mem;

    fn id(&self) -> cl_mem {
        unsafe {
            *self.id_ptr()
        }
    }

    fn byte_len(&self) -> size_t
    {
        unsafe {
            let size : size_t = 0;
            let err = clGetMemObjectInfo(self.id(),
                                         CL_MEM_SIZE,
                                         mem::size_of::<size_t>() as size_t,
                                         ptr::to_unsafe_ptr(&size) as *c_void,
                                         ptr::null());

            check(err, "Failed to read memory size");
            size
        }
    }

    fn len(&self) -> uint { self.byte_len() as uint / mem::size_of::<T>() }
}

pub struct CLBuffer<T> {
    cl_buffer: cl_mem
}

#[unsafe_destructor]
impl<T> Drop for CLBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            clReleaseMemObject(self.cl_buffer);
        }
    }
}

impl<T> Buffer<T> for CLBuffer<T> {
    unsafe fn id_ptr(&self) -> *cl_mem
    {
        ptr::to_unsafe_ptr(&self.cl_buffer)
    }
}

impl<T> KernelArg for CLBuffer<T> {
    fn get_value(&self) -> (size_t, *c_void)
    {
        unsafe {
            (mem::size_of::<cl_mem>() as size_t,
             self.id_ptr() as *c_void)
        }
    }
}

/* memory life cycle
 * | Trait  | Exists in rust | Exists in OpenCL | Direction      |
 * | Put    | X              |                  | rust -> opencl |
 * | Get    |                | X                | opencl -> rust |
 * | Write  | X              | X                | rust -> opencl |
 * | Read   | X              | X                | opencl -> rust |
 **/

pub trait Put<T, B> {
    fn put(&self, |ptr: *c_void, size: size_t| -> cl_mem) -> B;
}

pub trait Get<B, T> {
    fn get(mem: &B, |offset: size_t, ptr: *mut c_void, size: size_t|) -> Self;
}

pub trait Write {
    fn write(&self, |offset: size_t, ptr: *c_void, size: size_t|);
}

pub trait Read {
    fn read(&mut self, |offset: size_t, ptr: *mut c_void, size: size_t|);
}

impl<'r, T> Put<T, CLBuffer<T>> for &'r [T]
{
    fn put(&self, f: |ptr: *c_void, size: size_t| -> cl_mem) -> CLBuffer<T>
    {
        self.as_imm_buf(|p, len| {
            CLBuffer {
                cl_buffer: f(p as *c_void, (len * mem::size_of::<T>()) as size_t)
            }
        })
    }
}

impl<'r, T> Put<T, CLBuffer<T>> for &'r ~[T]
{
    fn put(&self, f: |ptr: *c_void, size: size_t| -> cl_mem) -> CLBuffer<T>
    {
        self.as_imm_buf(|p, len| {
            CLBuffer {
                cl_buffer: f(p as *c_void, (len * mem::size_of::<T>()) as size_t)
            }
        })
    }
}

impl<T> Put<T, CLBuffer<T>> for ~[T]
{
    fn put(&self, f: |ptr: *c_void, size: size_t| -> cl_mem) -> CLBuffer<T>
    {
        self.as_imm_buf(|p, len| {
            CLBuffer {
                cl_buffer: f(p as *c_void, (len * mem::size_of::<T>()) as size_t)
            }
        })
    }
}

impl<T> Get<CLBuffer<T>, T> for ~[T]
{
    fn get(mem: &CLBuffer<T>, f: |offset: size_t, ptr: *mut c_void, size: size_t|) -> ~[T]
    {
        let mut v: ~[T] = vec::with_capacity(mem.len());
        unsafe {
            v.set_len(mem.len());
        }
        v.as_imm_buf(|p, len| {
            f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t);
        });
        v
    }
}

impl<'r, T> Write for &'r [T]
{
    fn write(&self, f: |offset: size_t, ptr: *c_void, size: size_t|)
    {
        self.as_imm_buf(|p, len| {
            f(0, p as *c_void, (len * mem::size_of::<T>()) as size_t)
        })
    }
}

impl<'r, T> Read for &'r mut [T]
{
    fn read(&mut self, f: |offset: size_t, ptr: *mut c_void, size: size_t|)
    {
        self.as_mut_buf(|p, len| {
            f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t)
        })
    }
}

macro_rules! get_arg (
    ($t:ty) => (impl Get<CLBuffer<$t>, $t> for $t
        {
            fn get(_: &CLBuffer<$t>, f: |offset: size_t, ptr: *mut c_void, size: size_t|) -> $t {
                let mut v = zero();
                f(0, ptr::to_unsafe_ptr(&mut v) as *mut c_void, mem::size_of::<$t>() as size_t);
                v
            }
        })
)

get_arg!(int)
get_arg!(uint)
get_arg!(u32)
get_arg!(u64)
get_arg!(i32)
get_arg!(i64)
get_arg!(f32)
get_arg!(f64)

macro_rules! put_arg (
    ($t:ty) => (impl Put<$t, CLBuffer<$t>> for $t
        {
            fn put(&self, f: |ptr: *c_void, size: size_t| -> cl_mem) -> CLBuffer<$t> {
                CLBuffer {
                    cl_buffer: f(ptr::to_unsafe_ptr(self) as *c_void, mem::size_of::<$t>() as size_t)
                }
            }
        })
)

put_arg!(int)
put_arg!(uint)
put_arg!(u32)
put_arg!(u64)
put_arg!(i32)
put_arg!(i64)
put_arg!(f32)
put_arg!(f64)

macro_rules! read_arg (
    ($t:ty) => (
        impl Read for $t
        {
            fn read(&mut self, f: |offset: size_t, ptr: *mut c_void, size: size_t|) {
                f(0, ptr::to_unsafe_ptr(self) as *mut c_void, mem::size_of::<$t>() as size_t)
            }
        }
    )
)

read_arg!(int)
read_arg!(uint)
read_arg!(u32)
read_arg!(u64)
read_arg!(i32)
read_arg!(i64)
read_arg!(f32)
read_arg!(f64)

macro_rules! write_arg (
    ($t:ty) => (impl Write for $t
        {
            fn write(&self, f: |offset: size_t, ptr: *c_void, size: size_t|) {
                f(0, ptr::to_unsafe_ptr(self) as *c_void, mem::size_of::<$t>() as size_t)
            }
        })
)

write_arg!(int)
write_arg!(uint)
write_arg!(u32)
write_arg!(u64)
write_arg!(i32)
write_arg!(i64)
write_arg!(f32)
write_arg!(f64)
