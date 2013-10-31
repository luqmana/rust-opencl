use std::ptr;
use CL::*;
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
    pub fn new(width: uint, height: uint, depth: uint, val: &fn(uint, uint, uint) -> T) -> Array3D<T>
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


impl<'self, T> Put<Array3D<T>, Array3D_cl<T>> for &'self Array3D<T>
{
    fn put(&self, f: &fn(ptr: *c_void, size: size_t) -> cl_mem) -> Array3D_cl<T>
    {
        let out = do self.dat.as_imm_buf |p, len| {
            f(p as *c_void, (len * mem::size_of::<T>()) as size_t)
        };

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
    fn get(arr: &Array3D_cl<T>, f: &fn(offset: size_t, ptr: *mut c_void, size: size_t)) -> Array3D<T>
    {
        let mut v: ~[T] = vec::with_capacity(arr.len());
        unsafe {
            vec::raw::set_len(&mut v, arr.len());
        }

        do v.as_mut_buf |p, len| {
            f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t)
        };

        Array3D {
            width: arr.width,
            height: arr.height,
            depth: arr.depth,
            dat: v
        }
    }
}

impl<T> Write for Array3D<T> {
    fn write(&self, f: &fn(offset: size_t, ptr: *c_void, size: size_t))
    {
        do self.dat.as_imm_buf |p, len| {
            f(0, p as *c_void, (len * mem::size_of::<T>()) as size_t)
        };
    }
}

impl<T> Read for Array3D<T> {
    fn read(&mut self, f: &fn(offset: size_t, ptr: *mut c_void, size: size_t))
    {
        do self.dat.as_mut_buf |p, len| {
            f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t)
        }; 
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
    pub fn new(width: uint, height: uint, val: &fn(uint, uint) -> T) -> Array2D<T>
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

impl<'self, T> Put<Array2D<T>, Array2D_cl<T>> for &'self Array2D<T>
{
    fn put(&self, f: &fn(ptr: *c_void, size: size_t) -> cl_mem) -> Array2D_cl<T>
    {
        let out = do self.dat.as_imm_buf |p, len| {
            f(p as *c_void, (len * mem::size_of::<T>()) as size_t)
        };

        Array2D_cl{
            width: self.width,
            height: self.height,
            buf: out
        }
    }
}


impl<T> Get<Array2D_cl<T>, Array2D<T>> for Array2D<T>
{
    fn get(arr: &Array2D_cl<T>, f: &fn(offset: size_t, ptr: *mut c_void, size: size_t)) -> Array2D<T>
    {
        let mut v: ~[T] = vec::with_capacity(arr.len());
        unsafe {
            vec::raw::set_len(&mut v, arr.len());
        }

        do v.as_mut_buf |p, len| {
            f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t)
        };

        Array2D {
            width: arr.width,
            height: arr.height,
            dat: v
        }
    }
}

impl<T> Write for Array2D<T> {
    fn write(&self, f: &fn(offset: size_t, ptr: *c_void, size: size_t))
    {
        do self.dat.as_imm_buf |p, len| {
            f(0, p as *c_void, (len * mem::size_of::<T>()) as size_t)
        };
    }
}

impl<T> Read for Array2D<T> {
    fn read(&mut self, f: &fn(offset: size_t, ptr: *mut c_void, size: size_t))
    {
        do self.dat.as_mut_buf |p, len| {
            f(0, p as *mut c_void, (len * mem::size_of::<T>()) as size_t)
        }; 
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

#[cfg(test)]
mod test {
    use util::create_compute_context;
    use array::*;
    use CL::CL_MEM_READ_WRITE;

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

    #[test]
    fn put_get_2D()
    {
        let (_, ctx, queue) = create_compute_context().unwrap();
        let arr_in = do Array2D::new(8, 8) |x, y| {(x+y) as int};
        let arr_cl = ctx.create_buffer_from(&arr_in, CL_MEM_READ_WRITE);
        let arr_out: Array2D<int> = queue.get(&arr_cl, ());

        for x in range(0u, 8u) {
            for y in range(0u, 8u) {
                expect!(arr_in.get(x, y), arr_out.get(x, y));
            }
        }
    }


    #[test]
    fn read_write_2D()
    {
        let (_, ctx, queue) = create_compute_context().unwrap();
        let added = do Array2D::new(8, 8) |x, y| {(x+y) as int};
        let zero = do Array2D::new(8, 8) |_, _| {(0) as int};
        let mut out = do Array2D::new(8, 8) |_, _| {(0) as int};

        /* both are zeroed */
        let a_cl = ctx.create_buffer_from(&zero, CL_MEM_READ_WRITE);

        queue.write(&a_cl, &added, ());
        queue.read(&a_cl, &mut out, ());

        for x in range(0u, 8u) {
            for y in range(0u, 8u) {
                expect!(added.get(x, y), out.get(x, y));
            }
        }
    }


    #[test]
    fn kernel_2D()
    {
        let (device, ctx, queue) = create_compute_context().unwrap();
        let mut a = do Array2D::new(8, 8) |_, _| {(0) as i32};
        let b = do Array2D::new(8, 8) |x, y| {(x*y) as i32};
        let a_cl = ctx.create_buffer_from(&a, CL_MEM_READ_WRITE);

        let src =  "__kernel void test(__global int *a) { \
                        int x = get_global_id(0); \
                        int y = get_global_id(1); \
                        int size_x = get_global_size(0); \
                        a[size_x*y + x] = x*y; \
                    }";
        let prog = ctx.create_program_from_source(src);
        match prog.build(&device) {
            Ok(()) => (),
            Err(build_log) => {
                println!("Error building program:\n");
                println!("{:s}", build_log);
                fail!("");
            }
        }
        let k = prog.create_kernel("test");

        k.set_arg(0, &a_cl);
        let event = queue.enqueue_async_kernel(&k, (8, 8), None, ());
        queue.read(&a_cl, &mut a, &event);

        for x in range(0u, 8u) {
            for y in range(0u, 8u) {
                expect!(a.get(x, y), b.get(x, y));
            }
        }
    }

    #[test]
    fn put_get_3D()
    {
        let (_, ctx, queue) = create_compute_context().unwrap();
        let arr_in = do Array3D::new(8, 8, 8) |x, y, z| {(x+y+z) as int};
        let arr_cl = ctx.create_buffer_from(&arr_in, CL_MEM_READ_WRITE);
        let arr_out: Array3D<int> = queue.get(&arr_cl, ());

        for x in range(0u, 8u) {
            for y in range(0u, 8u) {
                for z in range(0u, 8u) {
                    expect!(arr_in.get(x, y, z), arr_out.get(x, y, z));
                }
            }
        }
    }


    #[test]
    fn read_write_3D()
    {
        let (_, ctx, queue) = create_compute_context().unwrap();
        let added = do Array3D::new(8, 8, 8) |x, y, z| {(x+y+z) as int};
        let zero = do Array3D::new(8, 8, 8) |_, _, _| {(0) as int};
        let mut out = do Array3D::new(8, 8, 8) |_, _, _| {(0) as int};

        /* both are zeroed */
        let a_cl = ctx.create_buffer_from(&zero, CL_MEM_READ_WRITE);

        queue.write(&a_cl, &added, ());
        queue.read(&a_cl, &mut out, ());

        for x in range(0u, 8u) {
            for y in range(0u, 8u) {
                for z in range(0u, 8u) {
                    expect!(added.get(x, y, z), out.get(x, y, z));
                }
            }
        }
    }


    #[test]
    fn kernel_3D()
    {
        let (device, ctx, queue) = create_compute_context().unwrap();
        let mut a = do Array3D::new(8, 8, 8) |_, _, _| {(0) as i32};
        let b = do Array3D::new(8, 8, 8) |x, y, z| {(x*y*z) as i32};
        let a_cl = ctx.create_buffer_from(&a, CL_MEM_READ_WRITE);

        let src =  "__kernel void test(__global int *a) { \
                        int x = get_global_id(0); \
                        int y = get_global_id(1); \
                        int z = get_global_id(2); \
                        int size_x = get_global_size(0); \
                        int size_y = get_global_size(1); \
                        a[size_x*size_y*z + size_x*y + x] = x*y*z; \
                    }";
        let prog = ctx.create_program_from_source(src);
        match prog.build(&device) {
            Ok(()) => (),
            Err(build_log) => {
                println!("Error building program:\n");
                println!("{:s}", build_log);
                fail!("");
            }
        }
        let k = prog.create_kernel("test");

        k.set_arg(0, &a_cl);
        let event = queue.enqueue_async_kernel(&k, (8, 8, 8), None, ());
        queue.read(&a_cl, &mut a, &event);

        for x in range(0u, 8u) {
            for y in range(0u, 8u) {
                for z in range(0u, 8u) {
                    expect!(a.get(x, y, z), b.get(x, y, z));
                }
            }
        }
    }
}