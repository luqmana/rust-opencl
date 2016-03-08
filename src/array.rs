//! Two- and three-dimensional array support.

use cl::*;
use cl::ll::*;
use mem::*;
use std::marker::PhantomData;
use std::mem;
use std::vec::Vec;
use libc::{size_t, c_void};

use hl::KernelArg;
use error::check;

/// A host-side 3D array.
pub struct Array3D<T> {
    width: usize,
    height: usize,
    depth: usize,
    dat: Vec<T>
}

/// A device-side 3D array.
pub struct Array3DCL<T> {
    width: usize,
    height: usize,
    depth: usize,
    buf: cl_mem,
    phantom: PhantomData<T>,
}

impl<T: Clone> Array3D<T> {
    /// Creates a new host-side 3D array with elements initialized by a callback.
    pub fn new<F>(width: usize, height: usize, depth: usize, val: F) -> Array3D<T>
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

        Array3D {
            width: width,
            height: height,
            depth: depth,
            dat: dat
        }
    }

    /// Sets an element of this array.
    pub fn set(&mut self, x: usize, y: usize, z: usize, val: T) {
        self.dat[self.width*self.height*z + self.width*y + x] = val;
    }

    /// Gets an element of this array.
    pub fn get(&self, x: usize, y: usize, z: usize) -> T {
        (&self.dat[..])[self.width*self.height*z + self.width*y + x].clone()
    }
}

impl<T> Drop for Array3DCL<T> {
    fn drop(&mut self) {
        unsafe {
            let status = clReleaseMemObject(self.buf);
            check(status, "Could not release the 3D array");
        }
    }
}

impl<'r, T> Put<Array3D<T>, Array3DCL<T>> for &'r Array3D<T>
{
    fn put<F>(&self, f: F)
           -> Array3DCL<T>
        where F: FnOnce(*const c_void, size_t) -> cl_mem
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        let out = f(p as *const c_void, (len * mem::size_of::<T>()) as size_t);

        Array3DCL{
            width: self.width,
            height: self.height,
            depth: self.depth,
            buf: out,
            phantom: PhantomData,
        }
    }
}


impl<T> Get<Array3DCL<T>, Array3D<T>> for Array3D<T>
{
    fn get<F>(arr: &Array3DCL<T>, f: F)
           -> Array3D<T>
        where F: FnOnce(size_t, *mut c_void, size_t)
    {
        let mut v: Vec<T> = Vec::with_capacity(arr.len());
        unsafe {
            v.set_len(arr.len());
        }

        let p = v.as_mut_ptr();
        let len = v.len();
        f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t);

        Array3D {
            width: arr.width,
            height: arr.height,
            depth: arr.depth,
            dat: v,
        }
    }
}

impl<T> Write for Array3D<T> {
    fn write<F>(&self, f: F)
        where F: FnOnce(size_t, *const c_void, size_t)
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        f(0, p as *const c_void, (len * mem::size_of::<T>()) as size_t)
    }
}

impl<T> Read for Array3D<T> {
    fn read<F>(&mut self, f: F)
        where F: FnOnce(size_t, *mut c_void, size_t)
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t)
    }
}

impl<T> Buffer<T> for Array3DCL<T> {
    unsafe fn id_ptr(&self) -> *const cl_mem {
        &self.buf as *const cl_mem
    }

    fn len(&self) -> usize {
        self.width * self.height * self.depth
    }
}

impl<T> KernelArg for Array3DCL<T> {
    fn get_value(&self) -> (size_t, *const c_void)
    {
        (mem::size_of::<cl_mem>() as size_t,
         unsafe { self.id_ptr() } as *const c_void)
    }
}

/// A host-side 2D array.
pub struct Array2D<T> {
    width: usize,
    height: usize,
    dat: Vec<T>,
}

/// A device-side 2D array.
pub struct Array2DCL<T> {
    width: usize,
    height: usize,
    buf: cl_mem,
    phantom: PhantomData<T>,
}

impl<T: Clone> Array2D<T> {
    /// Creates a new host-side 2D array with elements initialized by a callback.
    pub fn new<F>(width: usize, height: usize, val: F) -> Array2D<T>
        where F: Fn(usize, usize) -> T
    {
        let mut dat: Vec<T> = Vec::new();
        for x in 0 .. width {
            for y in 0 .. height {
                dat.push(val(x, y));
            }
        }
        Array2D {
            width: width,
            height: height,
            dat: dat,
        }
    }

    /// Sets an element of this array.
    pub fn set(&mut self, x: usize, y: usize, val: T) {
        self.dat[self.width*y + x] = val;
    }

    /// Gets an element of this array.
    pub fn get(&self, x: usize, y: usize) -> T {
        (&self.dat[..])[self.width*y + x].clone()
    }
}

impl<T> Drop for Array2DCL<T> {
    fn drop(&mut self) {
        unsafe {
            let status = clReleaseMemObject(self.buf);
            check(status, "Could not release the 2D array");
        }
    }
}

impl<'r, T> Put<Array2D<T>, Array2DCL<T>> for &'r Array2D<T>
{
    fn put<F>(&self, f: F) -> Array2DCL<T>
        where F: FnOnce(*const c_void, size_t) -> cl_mem
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        let out = f(p as *const c_void, (len * mem::size_of::<T>()) as size_t);

        Array2DCL{
            width: self.width,
            height: self.height,
            buf: out,
            phantom: PhantomData,
        }
    }
}


impl<T> Get<Array2DCL<T>, Array2D<T>> for Array2D<T>
{
    fn get<F>(arr: &Array2DCL<T>, f: F)
           -> Array2D<T>
        where F: FnOnce(size_t, *mut c_void, size_t)
    {
        let mut v: Vec<T> = Vec::with_capacity(arr.len());
        unsafe {
            v.set_len(arr.len())
        }

        let p = v.as_mut_ptr();
        let len = v.len();
        f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t);

        Array2D {
            width: arr.width,
            height: arr.height,
            dat: v
        }
    }
}

impl<T> Write for Array2D<T> {
    fn write<F>(&self, f: F)
        where F: FnOnce(size_t, *const c_void, size_t)
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        f(0, p as *const c_void, (len * mem::size_of::<T>()) as size_t)
    }
}

impl<T> Read for Array2D<T> {
    fn read<F>(&mut self, f: F)
        where F: FnOnce(size_t, *mut c_void, size_t)
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t)
    }
}

impl<T> Buffer<T> for Array2DCL<T> {
    unsafe fn id_ptr(&self) -> *const cl_mem {
        &self.buf as *const cl_mem
    }

    fn len(&self) -> usize {
        self.width * self.height
    }
}

impl<T> KernelArg for Array2DCL<T> {
    fn get_value(&self) -> (size_t, *const c_void)
    {
        (mem::size_of::<cl_mem>() as size_t,
         unsafe { self.id_ptr() } as *const c_void)
    }
}
