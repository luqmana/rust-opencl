use CL::*;
use CL::ll::*;
use hl;
use hl::*;
use error::check;
use std::mem;
use std::libc;
use std::ptr;
use std::vec;
use std::cast;
use std::unstable;

struct Vector<T> {
    cl_buffer: cl_mem,
    size:      uint,
    context:   @ComputeContext,
}

#[unsafe_destructor]
impl<T> Drop for Vector<T>
{
    #[fixed_stack_segment] #[inline(never)]
    fn drop(&mut self)
    { unsafe { clReleaseMemObject(self.cl_buffer); } }
}

impl<T> Vector<T> {
    #[fixed_stack_segment] #[inline(never)]
    pub fn from_vec(ctx: @ComputeContext, v: &[T]) -> Vector<T> {
        unsafe {
            do v.as_imm_buf |p, len| {
                let status = 0;
                let byte_size = (len * mem::size_of::<T>()) as libc::size_t;

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
                    size:      len,
                    context:   ctx,
                }
            }
        }
    }
 
    #[fixed_stack_segment] #[inline(never)]
    pub fn rewrite(&mut self, v: &[T]) {
        if self.size < v.len() {
            fail!("Cannot copy cpu buffer on a smaller gpu buffer.")
        }

        unsafe {
            do v.as_imm_buf |p, len|
            {
                let byte_size = (len * mem::size_of::<T>()) as libc::size_t;

                let status = clEnqueueWriteBuffer(
                    self.context.q.cqueue, self.cl_buffer, CL_TRUE, 0, byte_size, p as *libc::c_void,
                    0, ptr::null(), ptr::null());

                check(status, "Could not write buffer");
            }
        }
    }

    pub fn to_vec(self) -> ~[T] {
        unsafe {
            let mut result = ~[];
            result.reserve(self.size);
            vec::raw::set_len(&mut result, self.size);

            self.to_existing_vec(result);

            result
        }
    }

    #[fixed_stack_segment] #[inline(never)]
    pub fn to_existing_vec(&self, out: &mut [T]) {
        if out.len() < self.size {
            fail!("Cannot copy gpu buffer on a smaller cpu buffer.")
        }

        unsafe {
            do out.as_imm_buf |p, len| {
                clEnqueueReadBuffer(
                    self.context.q.cqueue, self.cl_buffer, CL_TRUE, 0,
                    (len * mem::size_of::<T>()) as libc::size_t,
                    p as *libc::c_void, 0, ptr::null(), ptr::null());
            }
        }
    }
}

impl<T> hl::KernelArg for Vector<T>
{
    fn get_value(&self) -> (libc::size_t, *libc::c_void) {
        (mem::size_of::<cl_mem>() as libc::size_t, 
         ptr::to_unsafe_ptr(&self.cl_buffer) as *libc::c_void)
    }
}

struct CLBuffer {
    cl_buffer: cl_mem
}

pub struct Unique<T> {
    cl_buffer: CLBuffer,
    size: uint,
    context: @ComputeContext,
}

impl Drop for CLBuffer {
    #[fixed_stack_segment] #[inline(never)]
    fn drop(&mut self) {
        unsafe {
            clReleaseMemObject(self.cl_buffer);
        }
    }
}

impl<T> Unique<T> {
    #[fixed_stack_segment] #[inline(never)]
    pub fn from_vec(ctx: @ComputeContext, v: ~[T]) -> Unique<T> {
        unsafe
        {
            let byte_size
                = mem::size_of::<unstable::raw::Vec<T>>()
                - mem::size_of::<T>() 
                + v.len() * mem::size_of::<T>();
            let byte_size = byte_size as libc::size_t;
            
            let len = v.len();
            
            let addr: *libc::c_void = cast::transmute_copy(&v);
            
            let status = 0;
            let buf = clCreateBuffer(ctx.ctx.ctx,
                                     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     byte_size,
                                     addr,
                                     ptr::to_unsafe_ptr(&status));
            check(status, "Could not allocate buffer");
            
            Unique {
                cl_buffer: CLBuffer { cl_buffer: buf },
                size: len,
                context: ctx,
            }
        }
    }

    #[fixed_stack_segment] #[inline(never)]
    pub fn to_vec(self) -> ~[T] {
        unsafe {
            let mut result = ~[];
            result.reserve(self.size);
            vec::raw::set_len(&mut result, self.size);
            do result.as_imm_buf |p, len| {
                clEnqueueReadBuffer(
                    self.context.q.cqueue, self.cl_buffer.cl_buffer, CL_TRUE,
                    // Skip the header, we have a new one here.
                    (mem::size_of::<unstable::raw::Vec<T>>() - mem::size_of::<T>())  as libc::size_t,
                    (len * mem::size_of::<T>()) as libc::size_t,
                    p as *libc::c_void, 0, ptr::null(), ptr::null());

            }
            result
        } }

}

impl<T> ::hl::KernelArg for Unique<T> {
    fn get_value(&self) -> (libc::size_t, *libc::c_void) {
        (mem::size_of::<cl_mem>() as libc::size_t, 
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
              fail!(format!("Test failure in {:s}: expected {:?}, got {:?}",
                         stringify!($test),
                         expected, test))
          }
      })
  )

    #[test]
    fn unique_vector() {
        let ctx = create_compute_context();

        let x = ~[1, 2, 3, 4, 5, 6];
        let gx = Unique::from_vec(ctx, x.clone());
        let y = gx.to_vec();
        expect!(y, x);
    }

  #[test]
  fn gpu_vector() {
      let ctx = create_compute_context();

      let x = ~[1, 2, 3, 4, 5];
      let gx = Vector::from_vec(ctx, x);
      let y = gx.to_vec();
      expect!(x, y);
  }
}
