use libc::{size_t, c_void};
use std::marker::{PhantomData};
use std::mem;
use std::ptr;

use cl::*;
use cl::ll::*;

use program::KernelArg;
use error::check;
use mem::{Put, Get, Read, Write};

/// Trait implemented by valid OpenCL buffer objects.
pub trait CLBuffer<T> {
    /// A pointer to the underlying OpenCL buffer object.
    unsafe fn cl_id_ptr(&self) -> *const cl_mem;

    /// The underlying buffer object.
    fn cl_id(&self) -> cl_mem {
        unsafe {
            *self.cl_id_ptr()
        }
    }

    /// The length in bytes of this buffer object.
    fn byte_len(&self) -> size_t
    {
        unsafe {
            let mut size : size_t = 0;
            let err = clGetMemObjectInfo(self.cl_id(),
                                         CL_MEM_SIZE,
                                         mem::size_of::<size_t>() as size_t,
                                         (&mut size as *mut size_t) as *mut c_void,
                                         ptr::null_mut());

            check(err, "Failed to read memory size");
            size
        }
    }

    /// The number of element of type `T` on this buffer object.
    fn len(&self) -> usize {
        self.byte_len() as usize / mem::size_of::<T>()
    }
}

/// A host-side 1D buffer.
pub type Buffer1D<T> = Vec<T>;

/// An 1-dimenisonal device-side OpenCL buffer object.
pub struct CLBuffer1D<T> {
    cl_buffer: cl_mem,
    phantom: PhantomData<T>,
}

impl<T> CLBuffer1D<T> {
    /// Unsafely wraps an OpenCL buffer object ID.
    /// 
    /// The provided identifier is not checked.
    pub unsafe fn new_unchecked(buffer: cl_mem) -> CLBuffer1D<T> {
        CLBuffer1D {
            cl_buffer: buffer,
            phantom:   PhantomData
        }
    }
}

impl<T> Drop for CLBuffer1D<T> {
    fn drop(&mut self) {
        unsafe {
            let status = clReleaseMemObject(self.cl_buffer);
            check(status, "Could not release the buffer");
        }
    }
}

impl<T> CLBuffer<T> for CLBuffer1D<T> {
    unsafe fn cl_id_ptr(&self) -> *const cl_mem {
        &self.cl_buffer as *const cl_mem
    }
}

impl<T> KernelArg for CLBuffer1D<T> {
    fn get_value(&self) -> (size_t, *const c_void) {
        unsafe {
            (mem::size_of::<cl_mem>() as size_t,
             self.cl_id_ptr() as *const c_void)
        }
    }
}

/// A host-side 3D buffer.
pub struct Buffer3D<T> {
    width: usize,
    height: usize,
    depth: usize,
    dat: Vec<T>
}

/// A device-side 3D buffer.
pub struct CLBuffer3D<T> {
    width: usize,
    height: usize,
    depth: usize,
    buf: cl_mem,
    phantom: PhantomData<T>,
}

impl<T: Clone> Buffer3D<T> {
    /// Creates a new host-side 3D buffer with elements initialized by a callback.
    pub fn new<F>(width: usize, height: usize, depth: usize, val: F) -> Buffer3D<T>
        where F: Fn(usize, usize, usize) -> T
    {
        let mut dat: Vec<T> = Vec::new();
        for x in 0 .. width {
            for y in 0 .. height {
                for z in 0 .. depth {
                    dat.push(val(x, y, z));
                }
            }
        }

        Buffer3D {
            width: width,
            height: height,
            depth: depth,
            dat: dat
        }
    }

    /// Sets an element of this buffer.
    pub fn set(&mut self, x: usize, y: usize, z: usize, val: T) {
        self.dat[self.width*self.height*z + self.width*y + x] = val;
    }

    /// Gets an element of this buffer.
    pub fn get(&self, x: usize, y: usize, z: usize) -> T {
        (&self.dat[..])[self.width*self.height*z + self.width*y + x].clone()
    }
}

impl<T> CLBuffer3D<T> {
    /// Creates a device-side 3D buffer from the given memory object.
    ///
    /// The memory object or the sizes are not checked.
    pub unsafe fn new_unchecked(width: usize, height: usize, depth: usize, buf: cl_mem) -> CLBuffer3D<T> {
        CLBuffer3D {
            width:   width,
            height:  height,
            depth:   depth,
            buf:     buf,
            phantom: PhantomData
        }
    }
}

impl<T> Drop for CLBuffer3D<T> {
    fn drop(&mut self) {
        unsafe {
            let status = clReleaseMemObject(self.buf);
            check(status, "Could not release the 3D buffer OpenCL buffer");
        }
    }
}

impl<'r, T> Put<Buffer3D<T>, CLBuffer3D<T>> for &'r Buffer3D<T>
{
    fn put<F>(&self, f: F) -> CLBuffer3D<T>
        where F: FnOnce(*const c_void, size_t) -> cl_mem
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        let out = f(p as *const c_void, (len * mem::size_of::<T>()) as size_t);

        CLBuffer3D{
            width: self.width,
            height: self.height,
            depth: self.depth,
            buf: out,
            phantom: PhantomData,
        }
    }
}


impl<T> Get<CLBuffer3D<T>, Buffer3D<T>> for Buffer3D<T>
{
    fn get<F>(arr: &CLBuffer3D<T>, f: F) -> Buffer3D<T>
        where F: FnOnce(size_t, *mut c_void, size_t)
    {
        let mut v: Vec<T> = Vec::with_capacity(arr.len());
        unsafe {
            v.set_len(arr.len());
        }

        let p = v.as_mut_ptr();
        let len = v.len();
        f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t);

        Buffer3D {
            width: arr.width,
            height: arr.height,
            depth: arr.depth,
            dat: v,
        }
    }
}

impl<T> Write for Buffer3D<T> {
    fn write<F>(&self, f: F)
        where F: FnOnce(size_t, *const c_void, size_t)
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        f(0, p as *const c_void, (len * mem::size_of::<T>()) as size_t)
    }
}

impl<T> Read for Buffer3D<T> {
    fn read<F>(&mut self, f: F)
        where F: FnOnce(size_t, *mut c_void, size_t)
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t)
    }
}

impl<T> CLBuffer<T> for CLBuffer3D<T> {
    unsafe fn cl_id_ptr(&self) -> *const cl_mem {
        &self.buf as *const cl_mem
    }

    fn len(&self) -> usize {
        self.width * self.height * self.depth
    }
}

impl<T> KernelArg for CLBuffer3D<T> {
    fn get_value(&self) -> (size_t, *const c_void)
    {
        (mem::size_of::<cl_mem>() as size_t,
         unsafe { self.cl_id_ptr() } as *const c_void)
    }
}

/// A host-side 2D buffer.
pub struct Buffer2D<T> {
    width: usize,
    height: usize,
    dat: Vec<T>,
}

/// A device-side 2D buffer.
pub struct CLBuffer2D<T> {
    width: usize,
    height: usize,
    buf: cl_mem,
    phantom: PhantomData<T>,
}

impl<T> CLBuffer2D<T> {
    /// Creates a device-side 2D buffer from the given memory object.
    ///
    /// The memory object or the sizes are not checked.
    pub unsafe fn new_unchecked(width: usize, height: usize, buf: cl_mem) -> CLBuffer2D<T> {
        CLBuffer2D {
            width:   width,
            height:  height,
            buf:     buf,
            phantom: PhantomData
        }
    }
}

impl<T: Clone> Buffer2D<T> {
    /// Creates a new host-side 2D buffer with elements initialized by a callback.
    pub fn new<F>(width: usize, height: usize, val: F) -> Buffer2D<T>
        where F: Fn(usize, usize) -> T
    {
        let mut dat: Vec<T> = Vec::new();
        for x in 0 .. width {
            for y in 0 .. height {
                dat.push(val(x, y));
            }
        }
        Buffer2D {
            width: width,
            height: height,
            dat: dat,
        }
    }

    /// Sets an element of this buffer.
    pub fn set(&mut self, x: usize, y: usize, val: T) {
        self.dat[self.width*y + x] = val;
    }

    /// Gets an element of this buffer.
    pub fn get(&self, x: usize, y: usize) -> T {
        (&self.dat[..])[self.width*y + x].clone()
    }
}

impl<T> Drop for CLBuffer2D<T> {
    fn drop(&mut self) {
        unsafe {
            let status = clReleaseMemObject(self.buf);
            check(status, "Could not release the 2D buffer OpenCL buffer");
        }
    }
}

impl<'r, T> Put<Buffer2D<T>, CLBuffer2D<T>> for &'r Buffer2D<T>
{
    fn put<F>(&self, f: F) -> CLBuffer2D<T>
        where F: FnOnce(*const c_void, size_t) -> cl_mem
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        let out = f(p as *const c_void, (len * mem::size_of::<T>()) as size_t);

        CLBuffer2D{
            width: self.width,
            height: self.height,
            buf: out,
            phantom: PhantomData,
        }
    }
}


impl<T> Get<CLBuffer2D<T>, Buffer2D<T>> for Buffer2D<T>
{
    fn get<F>(arr: &CLBuffer2D<T>, f: F)
           -> Buffer2D<T>
        where F: FnOnce(size_t, *mut c_void, size_t)
    {
        let mut v: Vec<T> = Vec::with_capacity(arr.len());
        unsafe {
            v.set_len(arr.len())
        }

        let p = v.as_mut_ptr();
        let len = v.len();
        f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t);

        Buffer2D {
            width: arr.width,
            height: arr.height,
            dat: v
        }
    }
}

impl<T> Write for Buffer2D<T> {
    fn write<F>(&self, f: F)
        where F: FnOnce(size_t, *const c_void, size_t)
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        f(0, p as *const c_void, (len * mem::size_of::<T>()) as size_t)
    }
}

impl<T> Read for Buffer2D<T> {
    fn read<F>(&mut self, f: F)
        where F: FnOnce(size_t, *mut c_void, size_t)
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t)
    }
}

impl<T> CLBuffer<T> for CLBuffer2D<T> {
    unsafe fn cl_id_ptr(&self) -> *const cl_mem {
        &self.buf as *const cl_mem
    }

    fn len(&self) -> usize {
        self.width * self.height
    }
}

impl<T> KernelArg for CLBuffer2D<T> {
    fn get_value(&self) -> (size_t, *const c_void)
    {
        (mem::size_of::<cl_mem>() as size_t,
         unsafe { self.cl_id_ptr() } as *const c_void)
    }
}
