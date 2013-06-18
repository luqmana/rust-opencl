use CL::*;
use CL::ll::*;
use hl;
use hl::*;
use error::check;
use std::sys;
use std::libc;
use std::ptr;
use std::vec;

// These are basically types that can be safely memcpyed.
trait VectorType {}

impl VectorType for int;
impl VectorType for i32;
impl VectorType for u32;
impl VectorType for float;
impl VectorType for f64;
impl VectorType for f32;

struct Vector<T> {
    cl_buffer: cl_mem,
    size:      uint,
    context:   @ComputeContext,
}

#[unsafe_destructor]
impl<T: VectorType> Drop for Vector<T>
{
  fn finalize(&self)
  { unsafe { clReleaseMemObject(self.cl_buffer); } }
}

impl<T: VectorType> Vector<T> {
  pub fn from_vec(ctx: @ComputeContext, v: &[const T]) -> Vector<T>
  {
    unsafe
    {
      do vec::as_const_buf(v) |p, len|
      {
        let status = 0;
        let byte_size = len * sys::size_of::<T>() as libc::size_t;

        let buf = clCreateBuffer(ctx.ctx.ctx,
                                 CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 byte_size,
                                 p as *libc::c_void,
                                 ptr::to_unsafe_ptr(&status));
        check(status, "Could not allocate buffer");

        //let status = clEnqueueWriteBuffer(
        //    ctx.q, buf, CL_TRUE, 0, byte_size, p as *libc::c_void,
        //    0, ptr::null(), ptr::null());

        Vector {
          cl_buffer: buf,
                       size: len,
                       context: ctx,
        }
      }
    }
  }

  pub fn to_vec(self) -> ~[T]
  {
    unsafe
    {
      let mut result = ~[];
      vec::reserve(&mut result, self.size);
      vec::raw::set_len(&mut result, self.size);
      do vec::as_imm_buf(result) |p, len| {
        clEnqueueReadBuffer(
          self.context.q.cqueue, self.cl_buffer, CL_TRUE, 0,
          len * sys::size_of::<T>() as libc::size_t,
          p as *libc::c_void, 0, ptr::null(), ptr::null());

      }
      result
    }
  }
}

impl<T: VectorType> hl::KernelArg for Vector<T>
{
    fn get_value(&self) -> (libc::size_t, *libc::c_void)
    {
        (sys::size_of::<cl_mem>() as libc::size_t, 
         ptr::to_unsafe_ptr(&self.cl_buffer) as *libc::c_void)
    }
}

#[cfg(test)]
mod test {
  use hl::*;
  use vector::*;

  macro_rules! expect (
      ($test: expr, $expected: expr) => ({
          let test = $test;
          let expected = $expected;
          if test != expected {
              fail!(fmt!("Test failure in %s: expected %?, got %?",
                         stringify!($test),
                         expected, test))
          }
      })
  )

  #[test]
  fn gpu_vector() {
      let ctx = create_compute_context();

      let x = ~[1, 2, 3, 4, 5];
      let gx = Vector::from_vec(ctx, x);
      let y = gx.to_vec();
      expect!(x, y);
  }
}
