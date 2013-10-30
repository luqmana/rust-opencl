use std::libc::{size_t, c_void};
use std::mem;
use std::ptr;
use std::vec;
use std::num::zero;
use std::unstable;
use std::unstable::raw::Vec;
use std::cast;

use CL::*;
use CL::ll::*;

use error::check;

pub trait Buffer<T> {
    unsafe fn id_ptr(&self) -> *cl_mem;

    fn id(&self) -> cl_mem {
        unsafe {
            *self.id_ptr()
        }
    }

    #[fixed_stack_segment] #[inline(never)]
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
    #[fixed_stack_segment] #[inline(never)]
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

pub struct Unique<T>(~[T]);

impl<T> Unique<T>
{
    pub fn unwrap(self) -> ~[T]
    {
        match self {
            Unique(dat) => dat
        }
    }

    pub fn as_ref<'r>(&'r self) -> &'r ~[T]
    {
        match *self {
            Unique(ref dat) => dat
        }
    }

    pub fn as_mut_ref<'r>(&'r self) -> &'r ~[T]
    {
        match *self {
            Unique(ref dat) => dat
        }
    }
}

impl<'self, T> Put<Unique<T>> for &'self Unique<T> {
    fn put(&self, f: &fn(ptr: *c_void, size: size_t) -> cl_mem) -> ~Buffer<Unique<T>>
    {
        unsafe {
            let byte_size
                = mem::size_of::<unstable::raw::Vec<T>>()
                - mem::size_of::<T>() 
                + self.capacity() * mem::size_of::<T>();
            let byte_size = byte_size as size_t;
                    
            let addr: *c_void = cast::transmute_copy(self.as_mut_ref());

            let out: ~CLBuffer<Unique<T>> = ~CLBuffer{
                cl_buffer: f(addr, byte_size)
            };
            out as ~Buffer<Unique<T>>
        }
    }
}

impl<T> Get<Unique<T>> for Unique<T> {
    fn get(_: &Buffer<Unique<T>>, f: &fn(offset: size_t, ptr: *mut c_void, size: size_t)) -> Unique<T>
    {
        // read header first
        let mut fill: uint = 0;
        f(0, ptr::to_unsafe_ptr(&mut fill) as *mut c_void, mem::size_of::<uint>() as size_t);
        fill /= mem::size_of::<T>();

        unsafe {
            let mut result: ~[T] = ~[];
            result.reserve(fill);
            vec::raw::set_len(&mut result, fill);

            let header_offset = mem::size_of::<unstable::raw::Vec<T>>() - mem::size_of::<T>();
            do result.as_imm_buf |p, len| {
                f(header_offset as size_t, p as *mut c_void, (len * mem::size_of::<T>()) as size_t);
            }
            Unique(result)
        }
    }
}

impl<T> Write for Unique<T> {
    fn write(&self, f: &fn(offset: size_t, ptr: *c_void, size: size_t))
    {
        unsafe {
            let byte_size
                = mem::size_of::<unstable::raw::Vec<T>>()
                - mem::size_of::<T>() 
                + self.len() * mem::size_of::<T>();
            let byte_size = byte_size as size_t;

            let addr: &c_void = cast::transmute_copy(self.as_mut_ref());
            f(0, addr, byte_size);           
        }
    }
}

impl<T> Read for Unique<T> {
    fn read(&mut self, f: &fn(offset: size_t, ptr: *mut c_void, size: size_t))
    {
        // read header first
        let mut fill: uint = 0;
        f(0, ptr::to_unsafe_ptr(&mut fill) as *mut c_void, mem::size_of::<uint>() as size_t);
        fill /= mem::size_of::<T>();

        unsafe {
            self.reserve(fill);

            match self {
                &Unique(ref mut v) => vec::raw::set_len(v, fill)
            };

            let header_offset = mem::size_of::<unstable::raw::Vec<T>>() - mem::size_of::<T>();
            do self.as_imm_buf |p, len| {
                f(header_offset as size_t, p as *mut c_void, (len * mem::size_of::<T>()) as size_t);
            }
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

pub trait Put<T> {
    fn put(&self, &fn(ptr: *c_void, size: size_t) -> cl_mem) -> ~Buffer<T>;
}

pub trait Get<T> {
    fn get(mem: &Buffer<T>, &fn(offset: size_t, ptr: *mut c_void, size: size_t)) -> Self;
}

pub trait Write {
    fn write(&self, &fn(offset: size_t, ptr: *c_void, size: size_t));
}

pub trait Read {
    fn read(&mut self, &fn(offset: size_t, ptr: *mut c_void, size: size_t));
}

impl<'self, T> Put<T> for &'self [T]
{
    fn put(&self, f: &fn(ptr: *c_void, size: size_t) -> cl_mem) -> ~Buffer<T> {
        do self.as_imm_buf |p, len| {
            let out: ~CLBuffer<T> = ~CLBuffer {
                cl_buffer: f(p as *c_void, (len * mem::size_of::<T>()) as size_t)
            };
            out as ~Buffer<T>
        }
    }
}

impl<T> Get<T> for ~[T]
{
    fn get(mem: &Buffer<T>, f: &fn(offset: size_t, ptr: *mut c_void, size: size_t)) -> ~[T] {
        let mut v: ~[T] = vec::with_capacity(mem.len());
        unsafe {
            vec::raw::set_len(&mut v, mem.len());
        }
        do v.as_imm_buf |p, len| {
            f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t);
        }
        v
    }
}

impl<'self, T> Write for &'self [T]
{
    fn write(&self, f: &fn(offset: size_t, ptr: *c_void, size: size_t)) {
        do self.as_imm_buf |p, len| {
            f(0, p as *c_void, (len * mem::size_of::<T>()) as size_t)
        }
    }
}

impl<'self, T> Read for &'self mut [T]
{
    fn read(&mut self, f: &fn(offset: size_t, ptr: *mut c_void, size: size_t)) {
        do self.as_mut_buf |p, len| {
            f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t)
        }
    }
}

macro_rules! get_arg (
    ($t:ty) => (impl Get<$t> for $t
        {
            fn get(_: &Buffer<$t>, f: &fn(offset: size_t, ptr: *mut c_void, size: size_t)) -> $t {
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
    ($t:ty) => (impl Put<$t> for $t
        {
            fn put(&self, f: &fn(ptr: *c_void, size: size_t) -> cl_mem) -> ~Buffer<$t> {
                let out: ~CLBuffer<$t> = ~CLBuffer {
                    cl_buffer: f(ptr::to_unsafe_ptr(self) as *c_void, mem::size_of::<$t>() as size_t)
                };
                out as ~Buffer<$t>
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
            fn read(&mut self, f: &fn(offset: size_t, ptr: *mut c_void, size: size_t)) {
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
            fn write(&self, f: &fn(offset: size_t, ptr: *c_void, size: size_t)) {
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


#[cfg(test)]
mod test {
    use std::unstable::intrinsics;
    use std::vec;
    use mem::{Read, Write, Unique};

    macro_rules! expect (
        ($test: expr, $expected: expr) => ({
            let test     = $test;
            let expected = $expected;
            if test != expected {
                fail!(format!("Test failure in {:s}: expected {:?}, got {:?}",
                           stringify!($test),
                           expected, test))
            }
        })
    )

    fn read_write<R: Read, W: Write>(src: &W, dst: &mut R)
    {
        // find the max size of the input buffer
        let mut max = 0;
        do src.write |off, _, len| {
            if max < off + len {
                max = off + len;
            }
        }
        let max = max as uint;

        let mut buffer: ~[u8] = ~[];
        unsafe {
            buffer.reserve(max);
            vec::raw::set_len(&mut buffer, max);          
        }

        // copy from input into buffer
        do src.write |off, ptr, len| {
            assert!(buffer.len() >= (off + len) as uint);
            unsafe {
                intrinsics::memcpy64(buffer.unsafe_ref(off as uint) as *mut u8, ptr as *u8, len);
            }
        }

        // copy from buffer into output
        do dst.read |off, ptr, len| {
            assert!(buffer.len() >= (off + len) as uint);
            unsafe {
                intrinsics::memcpy64(ptr as *mut u8, buffer.unsafe_ref(off as uint) as *u8, len);
            }
        }
    }

    #[test]
    fn read_write_vec()
    {
        let input :&[int] = &[0, 1, 2, 3, 4, 5, 6, 7];
        let mut output :&mut [int] = &mut [0, 0, 0, 0, 0, 0, 0, 0]; 
        read_write(&input, &mut output);
        expect!(input, output);
    }

    #[test]
    fn read_write_int()
    {
        let input : int = 3141;
        let mut output : int = 0; 
        read_write(&input, &mut output);
        expect!(input, output);
    }

    #[test]
    fn read_write_uint()
    {
        let input : uint = 3141;
        let mut output : uint = 0; 
        read_write(&input, &mut output);
        expect!(input, output);
    }

    #[test]
    fn read_write_f32()
    {
        let input : f32 = 3141.;
        let mut output : f32 = 0.; 
        read_write(&input, &mut output);
        expect!(input, output);
    }

    #[test]
    fn read_write_f64()
    {
        let input : f64 = 3141.;
        let mut output : f64 = 0.; 
        read_write(&input, &mut output);
        expect!(input, output);
    }

    #[test]
    fn read_write_unique()
    {
        let input : Unique<int> = Unique(~[1, 2, 3, 4]);
        let mut output : Unique<int> = Unique(~[]); 
        read_write(&input, &mut output);
        expect!(input.unwrap(), output.unwrap());
    }
}