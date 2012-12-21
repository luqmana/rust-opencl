use CL::*;
use hl::*;
use error::check;

// These are basically types that can be safely memcpyed.
trait VectorType {}

impl int: VectorType;
impl i32: VectorType;
impl u32: VectorType;
impl float: VectorType;
impl f64: VectorType;
impl f32: VectorType;

struct Vector<T: VectorType> {
    cl_buffer: cl_mem,
    size: uint,
    context: @ComputeContext,

    drop {
        clReleaseMemObject(self.cl_buffer);
    }
}

impl<T: VectorType> Vector<T> {
    static fn from_vec(ctx: @ComputeContext, v: &[const T]) -> Vector<T> {
        do vec::as_const_buf(v) |p, len| {
            let mut status = 0;
            let byte_size = len * sys::size_of::<T>() as libc::size_t;
            
            let buf = clCreateBuffer(ctx.ctx.ctx,
                                     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     byte_size,
                                     p as *libc::c_void,
                                     ptr::addr_of(&status));
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

    fn to_vec(self) -> ~[T] unsafe {
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

impl<T: VectorType> Vector<T>: hl::KernelArg {
    pure fn get_value(&self) -> (libc::size_t, *libc::c_void) {
        (self.size * sys::size_of::<T>() as libc::size_t, 
         ptr::addr_of(&self.cl_buffer) as *libc::c_void)
    }
}

#[cfg(test)]
mod test {
    macro_rules! expect (
        ($test: expr, $expected: expr) => ({
            let test = $test;
            let expected = $expected;
            if test != expected {
                fail fmt!("Test failure in %s: expected %?, got %?",
                          stringify!($test),
                          expected, test)
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
