use std::ptr;
use CL::*;
use CL::ll::*;
use mem::*;
use std::mem;
use std::vec;
use std::libc::{size_t, c_void};

use hl::KernelArg;

pub struct Array3D<T> {
    width: uint,
    height: uint,
    depth: uint,
    dat: ~[T]
}

pub struct Array3D_cl<T> {
    width: uint,
    height: uint,
    depth: uint,
    buf: cl_mem,
}

impl<T: Clone> Array3D<T> {
    pub fn new(width: uint, height: uint, depth: uint, val: |uint, uint, uint| -> T) -> Array3D<T>
    {
        let mut dat: ~[T] = ~[];
        for x in range(0, width) {
            for y in range(0, height) {
                for z in range(0, depth) {
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

    pub fn set(&mut self, x: uint, y: uint, z: uint, val: T)
    {
        self.dat[self.width*self.height*z + self.width*y + x] = val;
    }

    pub fn get(&self, x: uint, y: uint, z: uint) -> T
    {
        self.dat[self.width*self.height*z + self.width*y + x].clone()
    }
}

#[unsafe_destructor]
impl<T> Drop for Array3D_cl<T> {
    fn drop(&mut self) {
        unsafe {
            clReleaseMemObject(self.buf);
        }
    }
}

impl<'r, T> Put<Array3D<T>, Array3D_cl<T>> for &'r Array3D<T>
{
    fn put(&self, f: |ptr: *c_void, size: size_t| -> cl_mem) -> Array3D_cl<T>
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        let out = f(p as *c_void, (len * mem::size_of::<T>()) as size_t);

        Array3D_cl{
            width: self.width,
            height: self.height,
            depth: self.depth,
            buf: out
        }
    }
}


impl<T> Get<Array3D_cl<T>, Array3D<T>> for Array3D<T>
{
    fn get(arr: &Array3D_cl<T>, f: |offset: size_t, ptr: *mut c_void, size: size_t|) -> Array3D<T>
    {
        let mut v: ~[T] = vec::with_capacity(arr.len());
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
            dat: v
        }
    }
}

impl<T> Write for Array3D<T> {
    fn write(&self, f: |offset: size_t, ptr: *c_void, size: size_t|)
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        f(0, p as *c_void, (len * mem::size_of::<T>()) as size_t)
    }
}

impl<T> Read for Array3D<T> {
    fn read(&mut self, f: |offset: size_t, ptr: *mut c_void, size: size_t|)
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t)
    }
}

impl<T> Buffer<T> for Array3D_cl<T> {
    fn id_ptr(&self) -> *cl_mem {
        ptr::to_unsafe_ptr(&self.buf)
    }

    fn len(&self) -> uint {
        self.width * self.height * self.depth
    }
}

impl<T> KernelArg for Array3D_cl<T> {
    fn get_value(&self) -> (size_t, *c_void)
    {
        (mem::size_of::<cl_mem>() as size_t,
         self.id_ptr() as *c_void)
    }
}

pub struct Array2D<T> {
    width: uint,
    height: uint,
    dat: ~[T],
}

pub struct Array2D_cl<T> {
    width: uint,
    height: uint,
    buf: cl_mem,
}

impl<T: Clone> Array2D<T> {
    pub fn new(width: uint, height: uint, val: |uint, uint| -> T) -> Array2D<T>
    {
        let mut dat: ~[T] = ~[];
        for x in range(0, width) {
            for y in range(0, height) {
                dat.push(val(x, y));
            }
        }
        Array2D {
            width: width,
            height: height,
            dat: dat,
        }
    }

    pub fn set(&mut self, x: uint, y: uint, val: T) {
        self.dat[self.width*y + x] = val;
    }

    pub fn get(&self, x: uint, y: uint) -> T {
        self.dat[self.width*y + x].clone()
    }
}

#[unsafe_destructor]
impl<T> Drop for Array2D_cl<T> {
    fn drop(&mut self) {
        unsafe {
            clReleaseMemObject(self.buf);
        }
    }
}

impl<'r, T> Put<Array2D<T>, Array2D_cl<T>> for &'r Array2D<T>
{
    fn put(&self, f: |ptr: *c_void, size: size_t| -> cl_mem) -> Array2D_cl<T>
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        let out = f(p as *c_void, (len * mem::size_of::<T>()) as size_t);

        Array2D_cl{
            width: self.width,
            height: self.height,
            buf: out
        }
    }
}


impl<T> Get<Array2D_cl<T>, Array2D<T>> for Array2D<T>
{
    fn get(arr: &Array2D_cl<T>, f: |offset: size_t, ptr: *mut c_void, size: size_t|) -> Array2D<T>
    {
        let mut v: ~[T] = vec::with_capacity(arr.len());
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
    fn write(&self, f: |offset: size_t, ptr: *c_void, size: size_t|)
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        f(0, p as *c_void, (len * mem::size_of::<T>()) as size_t)
    }
}

impl<T> Read for Array2D<T> {
    fn read(&mut self, f: |offset: size_t, ptr: *mut c_void, size: size_t|)
    {
        let p = self.dat.as_ptr();
        let len = self.dat.len();
        f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t)
    }
}

impl<T> Buffer<T> for Array2D_cl<T> {
    fn id_ptr(&self) -> *cl_mem {
        ptr::to_unsafe_ptr(&self.buf)
    }

    fn len(&self) -> uint {
        self.width * self.height
    }
}

impl<T> KernelArg for Array2D_cl<T> {
    fn get_value(&self) -> (size_t, *c_void)
    {
        (mem::size_of::<cl_mem>() as size_t,
         self.id_ptr() as *c_void)
    }
}
