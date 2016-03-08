//! A higher level API.

use libc;
use std::ffi::CString;
use std::iter::repeat;
use std::mem;
use std::ptr;
use std::string::String;
use std::vec::Vec;

use cl::*;
use cl::ll::*;
use cl::CLStatus::CL_SUCCESS;
use error::check;
use mem::CLBuffer;
use device::Device;

/// Represents an OpenCL program, which is a collection of kernels.
///
/// Create these using
/// [`Context::create_program_from_source`](struct.Context.html#method.create_program_from_source)
/// or
/// [`Context::create_program_from_binary`](struct.Context.html#method.create_program_from_binary).
pub struct Program {
    prg: cl_program,
}

impl Drop for Program
{
    fn drop(&mut self) {
        unsafe {
            let status = clReleaseProgram(self.prg);
            check(status, "Colud not release the program.");
        }
    }
}

impl Program {
    /// Creates a new program from its OpenCL pointer.
    ///
    /// The pointer validity is not checked.
    pub unsafe fn new_unchecked(prg: cl_program) -> Program {
        Program {
            prg: prg
        }
    }

    /// Build the program for a given device.
    ///
    /// Both Ok and Err returns include the build log.
    pub fn build(&self, device: &Device) -> Result<String, String>
    {
        unsafe
        {
            let ret = clBuildProgram(self.prg, 1, &device.cl_id(),
                                     ptr::null(),
                                     mem::transmute(ptr::null::<fn()>()),
                                     ptr::null_mut());
            // Get the build log.
            let mut size = 0 as libc::size_t;
            let status = clGetProgramBuildInfo(
                self.prg,
                device.cl_id(),
                CL_PROGRAM_BUILD_LOG,
                0,
                ptr::null_mut(),
                (&mut size));
            check(status, "Could not get build log");

            let mut buf : Vec<u8> = repeat(0u8).take(size as usize).collect();
            let status = clGetProgramBuildInfo(
                self.prg,
                device.cl_id(),
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

    /// Retrieves a kernel object from its name.
    pub fn create_kernel(&self, name: &str) -> Kernel {
        create_kernel(self, name)
    }
}

/// An OpenCL kernel object.
pub struct Kernel {
    kernel: cl_kernel,
}

impl Drop for Kernel
{
    fn drop(&mut self) {
        unsafe {
            let status = clReleaseKernel(self.kernel);
            check(status, "Could not release the kernel");
        }
    }
}

impl Kernel {
    /// The underlying OpenCL kernel pointer.
    pub fn cl_id(&self) -> cl_kernel {
        self.kernel
    }

    /// Sets the i-th argument of this kernel.
    pub fn set_arg<T: KernelArg>(&self, i: usize, x: &T)
    {
        set_kernel_arg(self, i as cl_uint, x)
    }

    /// Allocates local memory for this kernel.
    pub fn alloc_local<T>(&self, i: usize, l: usize)
    {
        alloc_kernel_local::<T>(self, i as cl_uint, l as libc::size_t)
    }
}

/// Retrieves a kernel object from its name on the given program.
pub fn create_kernel(program: &Program, kernel: &str) -> Kernel
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

/// Trait implemented by valid kernel arguments.
pub trait KernelArg {
  /// Gets the size (in bytes) of this kernel argument and an OpenCL-compatible
  /// pointer to its value.
  fn get_value(&self) -> (libc::size_t, *const libc::c_void);
}

impl<'r, T> KernelArg for &'r (CLBuffer<T> + 'r) {
    fn get_value(&self) -> (libc::size_t, *const libc::c_void)
    {
        unsafe {
            (mem::size_of::<cl_mem>() as libc::size_t,
             self.cl_id_ptr() as *const libc::c_void)
        }
    }
}

impl<'r, T> KernelArg for Box<CLBuffer<T> + 'r> {
    fn get_value(&self) -> (libc::size_t, *const libc::c_void)
    {
        unsafe {
            (mem::size_of::<cl_mem>() as libc::size_t,
             self.cl_id_ptr() as *const libc::c_void)
        }
    }
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

/// Sets the i-th kernel argument of the given kernel.
pub fn set_kernel_arg<T: KernelArg>(kernel:   &Kernel,
                                    position: cl_uint,
                                    arg:      &T)
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

/// Allocates local memory for the given kernel.
///
/// It allocates enough memory to contain `length` elements of type `T`.
pub fn alloc_kernel_local<T>(kernel: &Kernel,
                             position: cl_uint,
                             // size: libc::size_t,
                             length: libc::size_t){
    unsafe
    {
        let tsize = mem::size_of::<T>() as libc::size_t;
        let ret = clSetKernelArg(kernel.kernel, position,
                                    tsize * length, ptr::null());
        check(ret, "Failed to set kernel arg!");
    }
}

/// Trait implemented by a valid kernel buffer index.
pub trait KernelIndex
{
    /// The number of dimensions (up to 3) of this kernel index.
    fn num_dimensions(dummy_self: Option<Self>) -> cl_uint where Self: Sized;

    /// Returns an OpenCL-compatible pointer to this index.
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
