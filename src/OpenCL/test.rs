#[feature(macro_rules)];
#[feature(globs)];

extern mod std;
extern mod OpenCL;

use OpenCL::hl::*;

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


pub fn test_all_platforms_devices(test: &fn(Device, Context, CommandQueue))
{
    let platforms = get_platforms();
    for p in platforms.iter() {
        let devices = p.get_devices();
        for d in devices.iter() {
            let context = d.create_context();
            let queue = context.create_command_queue(d);
            test(*d, context, queue);
        }
    }
}

mod mem {
    use std::vec;
    use std::cast;
    use OpenCL::mem::{Read, Write, Unique};

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
            let off = off as uint;
            let len = len as uint;
            assert!(buffer.len() >= (off + len) as uint);
            let target = buffer.mut_slice(off, off + len);
            unsafe {
                do vec::raw::buf_as_slice(ptr as *u8, len) |src| {
                    vec::bytes::copy_memory(target, src, len);
                }
            }
        }

        // copy from buffer into output
        do dst.read |off, ptr, len| {
            let off = off as uint;
            let len = len as uint;
            assert!(buffer.len() >= (off + len) as uint);
            let src = buffer.slice(off, off + len);
            unsafe {
                do vec::raw::mut_buf_as_slice(ptr as *mut u8, len) |dst| {
                    vec::bytes::copy_memory(dst, src, len);
                }
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
    #[cfg(disable)]
    fn read_write_unique()
    {
        let input : Unique<int> = Unique(~[1, 2, 3, 4]);
        let mut output : Unique<int> = Unique(~[]); 
        read_write(&input, &mut output);
        expect!(input.unwrap(), output.unwrap());
    }
}

#[cfg(test)]
mod hl {
    use OpenCL::CL::*;
    use OpenCL::hl::*;
    use OpenCL::mem::*;
    use OpenCL::util;

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
    fn program_build() {
        let src = "__kernel void test(__global int *i) { \
                   *i += 1; \
                   }";
        do ::test_all_platforms_devices |device, ctx, _| {
            let prog = ctx.create_program_from_source(src);
            prog.build(&device);            
        }
    }

    #[test]
    fn simple_kernel() {
        let src = "__kernel void test(__global int *i) { \
                   *i += 1; \
                   }";
        do ::test_all_platforms_devices |device, ctx, queue| {
            let prog = ctx.create_program_from_source(src);
            prog.build(&device);

            let k = prog.create_kernel("test");
            let v = ctx.create_buffer_from(&[1], CL_MEM_READ_WRITE);
            
            k.set_arg(0, &v);

            queue.enqueue_async_kernel(&k, 1, None, ()).wait();

            let v: ~[int] = queue.get(&v, ());

            expect!(v[0], 2);
        }
    }

    #[test]
    fn add_k() {
        let src = "__kernel void test(__global int *i, long int k) { \
                   *i += k; \
                   }";

        do ::test_all_platforms_devices |device, ctx, queue| {
            let prog = ctx.create_program_from_source(src);
            prog.build(&device);

            let k = prog.create_kernel("test");
            
            let v = ctx.create_buffer_from(&[1], CL_MEM_READ_WRITE);
            
            k.set_arg(0, &v);
            k.set_arg(1, &42);

            queue.enqueue_async_kernel(&k, 1, None, ()).wait();

            let v: ~[int] = queue.get(&v, ());

            expect!(v[0], 43);
        }
    }

    #[test]
    fn simple_kernel_index() {
        let src = "__kernel void test(__global int *i) { \
                   *i += 1; \
                   }";

        do ::test_all_platforms_devices |device, ctx, queue| {
            let prog = ctx.create_program_from_source(src);
            prog.build(&device);

            let k = prog.create_kernel("test");

            let v = ctx.create_buffer_from(&[1], CL_MEM_READ_WRITE);
          
            k.set_arg(0, &v);

            queue.enqueue_async_kernel(&k, 1, None, ()).wait();
          
            let v: ~[int] = queue.get(&v, ());

            expect!(v[0], 2);
        }
    }

    #[test]
    fn chain_kernel_event() {
        let src = "__kernel void test(__global int *i) { \
                   *i += 1; \
                   }";

        do ::test_all_platforms_devices |device, ctx, queue| {
            let prog = ctx.create_program_from_source(src);
            prog.build(&device);

            let k = prog.create_kernel("test");
            let v = ctx.create_buffer_from(&[1], CL_MEM_READ_WRITE);
          
            k.set_arg(0, &v);

            let mut e : Option<Event> = None;
            for _ in range(0, 8) {
                e = Some(queue.enqueue_async_kernel(&k, 1, None, e));
            }
            e.wait();
          
            let v: ~[int] = queue.get(&v, ());

            expect!(v[0], 9);
        }
    }

    #[test]
    fn chain_kernel_event_list() {
        let src = "__kernel void inc(__global int *i) { \
                   *i += 1; \
                   } \
                   __kernel void add(__global int *a, __global int *b, __global int *c) { \
                   *c = *a + *b; \
                   }";

        do ::test_all_platforms_devices |device, ctx, queue| {
            let prog = ctx.create_program_from_source(src);
            prog.build(&device);

            let k_incA = prog.create_kernel("inc");
            let k_incB = prog.create_kernel("inc");
            let k_add = prog.create_kernel("add");
            
            let a = ctx.create_buffer_from(&[1], CL_MEM_READ_WRITE);
            let b = ctx.create_buffer_from(&[1], CL_MEM_READ_WRITE);
            let c = ctx.create_buffer_from(&[1], CL_MEM_READ_WRITE);
          
            k_incA.set_arg(0, &a);
            k_incB.set_arg(0, &b);

            let event_list = &[
                queue.enqueue_async_kernel(&k_incA, 1, None, ()),
                queue.enqueue_async_kernel(&k_incB, 1, None, ()),
            ];

            k_add.set_arg(0, &a);
            k_add.set_arg(1, &b);
            k_add.set_arg(2, &c);

            let event = queue.enqueue_async_kernel(&k_add, 1, None, event_list);
          
            let v: ~[int] = queue.get(&c, event);

            expect!(v[0], 4);
        }
    }

    #[test]
    fn kernel_2d()
    {
        let src = "__kernel void test(__global long int *N) { \
                   int i = get_global_id(0); \
                   int j = get_global_id(1); \
                   int s = get_global_size(0); \
                   N[i * s + j] = i * j;
}";
        do ::test_all_platforms_devices |device, ctx, queue| {
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
            
            let v = ctx.create_buffer_from(&[1, 2, 3, 4, 5, 6, 7, 8, 9], CL_MEM_READ_ONLY);
            
            k.set_arg(0, &v);

            queue.enqueue_async_kernel(&k, (3, 3), None, ()).wait();
            
            let v: ~[int] = queue.get(&v, ());
            
            expect!(v, ~[0, 0, 0, 0, 1, 2, 0, 2, 4]);
        }
    }

    #[test]
    fn memory_read_write()
    {
        do ::test_all_platforms_devices |_, ctx, queue| {
            let buffer: CLBuffer<int> = ctx.create_buffer(8, CL_MEM_READ_ONLY);

            let input = &[0, 1, 2, 3, 4, 5, 6, 7];
            let mut output = &mut [0, 0, 0, 0, 0, 0, 0, 0];

            queue.write(&buffer, &input, ());
            queue.read(&buffer, &mut output, ());

            expect!(input, output);
        }
    }

    #[test]
    fn memory_read_vec()
    {
        do ::test_all_platforms_devices |_, ctx, queue| {
            let input = &[0, 1, 2, 3, 4, 5, 6, 7];
            let buffer = ctx.create_buffer_from(input, CL_MEM_READ_WRITE);
            let output: ~[int] = queue.get(&buffer, ());
            expect!(input, output);
        }
    }


    #[test]
    fn memory_read_owned()
    {
        do ::test_all_platforms_devices |_, ctx, queue| {
            let input = ~[0, 1, 2, 3, 4, 5, 6, 7];
            let buffer = ctx.create_buffer_from(&input, CL_MEM_READ_WRITE);
            let output: ~[int] = queue.get(&buffer, ());
            expect!(input, output);
        }
    }

    #[test]
    fn memory_read_owned_clone()
    {
        do ::test_all_platforms_devices |_, ctx, queue| {
            let input = ~[0, 1, 2, 3, 4, 5, 6, 7];
            let buffer = ctx.create_buffer_from(input.clone(), CL_MEM_READ_WRITE);
            let output: ~[int] = queue.get(&buffer, ());
            expect!(input, output);
        }
    }

    #[test]
    #[cfg(disable)]
    fn memory_read_unique()
    {
        do ::test_all_platforms_devices |_, ctx, queue| {
            let input = Unique(~[0, 1, 2, 3, 4, 5, 6, 7]);
            let buffer = ctx.create_buffer_from(&input, CL_MEM_READ_WRITE);
            let output: Unique<int> = queue.get(&buffer, ());
            expect!(input.unwrap(), output.unwrap());
        }
    }

    #[test]
    #[cfg(disable)]
    fn kernel_unique_size()
    {
        let src = " struct vec { \
                        long fill; \
                        long alloc; \
                    }; \
                    __kernel void test(__global struct vec *v) { \
                        int idx = get_global_id(0); \
                        global long *dat = (global long*)(v+1);
                        if (idx == 0) { \
                            v->fill = v->alloc; \
                        } \
                        if (idx < (v->alloc / sizeof(long))) { \
                            dat[idx] = idx*idx; \
                        } \
                    } \
                    ";

        do ::test_all_platforms_devices |device, ctx, queue| {
            let prog = ctx.create_program_from_source(src);

            match prog.build(&device) {
                Ok(()) => (),
                Err(build_log) => {
                    println!("Error building program:\n");
                    println!("{:s}", build_log);
                    fail!("");
                }
            }

            let mut expect: ~[int] = ~[];
            for i in range(0, 16) {
                expect.push(i*i);
            }

            let mut input: ~[int] = ~[];
            input.reserve(16);


            let k = prog.create_kernel("test");
            let v = ctx.create_buffer_from(&Unique(input), CL_MEM_READ_WRITE);
            
            let out_check: Unique<int> = queue.get(&v, ());
            let out_check = out_check.unwrap();

            expect!(out_check.len(), 0);

            k.set_arg(0, &v);
            queue.enqueue_async_kernel(&k, 16, None, ()).wait();
            
            let out_check: Unique<int> = queue.get(&v, ());
            let out_check = out_check.unwrap();

            expect!(expect, out_check);
        }
    }

    #[test]
    fn event_get_times() {
        let src = "__kernel void test(__global int *i) { \
                   *i += 1; \
                   }";

        let (device, ctx, queue) = util::create_compute_context().unwrap();
        let prog = ctx.create_program_from_source(src);
        prog.build(&device);

        let k = prog.create_kernel("test");
        let v = ctx.create_buffer_from(&[1], CL_MEM_READ_WRITE);
      
        k.set_arg(0, &v);

        let e = queue.enqueue_async_kernel(&k, 1, None, ());
        e.wait();

        // the that are returned are not useful for unit test, this test
        // is mostly testing that opencl returns no error
        e.queue_time();
        e.submit_time();
        e.start_time();
        e.end_time();
    }
}

#[cfg(test)]
mod array {
    use OpenCL::array::*;
    use OpenCL::CL::CL_MEM_READ_WRITE;

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
        do ::test_all_platforms_devices |_, ctx, queue| {
            let arr_in = do Array2D::new(8, 8) |x, y| {(x+y) as int};
            let arr_cl = ctx.create_buffer_from(&arr_in, CL_MEM_READ_WRITE);
            let arr_out: Array2D<int> = queue.get(&arr_cl, ());

            for x in range(0u, 8u) {
                for y in range(0u, 8u) {
                    expect!(arr_in.get(x, y), arr_out.get(x, y));
                }
            }
        }
    }


    #[test]
    fn read_write_2D()
    {
        do ::test_all_platforms_devices |_, ctx, queue| {
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
    }


    #[test]
    fn kernel_2D()
    {
        do ::test_all_platforms_devices |device, ctx, queue| {
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
    }

    #[test]
    fn put_get_3D()
    {
        do ::test_all_platforms_devices |_, ctx, queue| {
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
    }


    #[test]
    fn read_write_3D()
    {
        do ::test_all_platforms_devices |_, ctx, queue| {
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
    }


    #[test]
    fn kernel_3D()
    {
        do ::test_all_platforms_devices |device, ctx, queue| {
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
}

#[cfg(test)]
mod error {
    use OpenCL::CL::*;
    use OpenCL::error::*;

    #[test]
    fn test_convert() {
        expect!(convert(CL_INVALID_GLOBAL_OFFSET as cl_int),
                CL_INVALID_GLOBAL_OFFSET)
    }
    
    #[test]
    fn convert_to_str() {
        expect!(convert(CL_INVALID_BUFFER_SIZE as cl_int).to_str(),
                ~"CL_INVALID_BUFFER_SIZE");
    }
}
