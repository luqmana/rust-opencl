#![feature(slice_bytes)]

#[macro_use]
extern crate log;

extern crate opencl;

use opencl::{Platform, Device, Context, CommandQueue};

macro_rules! expect (
    ($test: expr, $expected: expr) => ({
            let test     = $test;
            let expected = $expected;
            if test != expected {
                panic!(format!("Test failure in {}:", // " expected {}, got {}",
                              stringify!($test)/*,
                              expected, test*/))
            }
        }
    )
);

pub fn test_all_platforms_devices<F>(test: &mut F)
    where F: FnMut(&Device, &Context, &CommandQueue)
{
    let platforms = Platform::all();
    for p in platforms.iter() {
        let devices = p.get_devices();
        for d in devices.iter() {
            let context = Context::new(d);
            let queue = context.create_command_queue(d);
            test(d, &context, &queue);
        }
    }
}

mod mem {
    use std::slice;
    use opencl::{Read, Write};

    fn read_write<W: Write, R: Read>(src: &W, dst: &mut R)
    {
        // find the max size of the input buffer
        let mut max = 0;
        src.write(|off, _, len| {
            if max < off + len {
                max = off + len;
            }
        });
        let max = max as usize;

        let mut buffer: Vec<u8> = Vec::new();
        unsafe {
            buffer.reserve(max);
            buffer.set_len(max);
        }

        // copy from input into buffer
        src.write(|off, ptr, len| {
            let off = off as usize;
            let len = len as usize;
            assert!(buffer.len() >= (off + len) as usize);
            let target = &mut buffer[off .. off + len];

            unsafe {
                let ptr = ptr as *const u8;
                let src = slice::from_raw_parts(ptr, len);
                slice::bytes::copy_memory(src, target);
            }
        });

        // copy from buffer into output
        dst.read(|off, ptr, len| {
            let off = off as usize;
            let len = len as usize;
            assert!(buffer.len() >= (off + len) as usize);
            let src = &buffer[off .. off + len];
            unsafe {
                let ptr = ptr as *mut u8;
                let mut dst = slice::from_raw_parts_mut(ptr, len);
                slice::bytes::copy_memory(src, dst);
            }
        })
    }

    #[test]
    fn read_write_slice()
    {
        let input: &[isize] = &[0, 1, 2, 3, 4, 5, 6, 7];
        let mut output: &mut [isize] = &mut [0, 0, 0, 0, 0, 0, 0, 0];
        read_write(&input, &mut output);
        expect!(input, output);
    }

    #[test]
    fn read_write_int()
    {
        let input: isize      = 3141;
        let mut output: isize = 0;
        read_write(&input, &mut output);
        expect!(input, output);
    }

    #[test]
    fn read_write_uint()
    {
        let input : usize = 3141;
        let mut output : usize = 0;
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
}

#[cfg(test)]
mod hl {
    use opencl::cl::*;
    use opencl::*;
    use opencl;

    #[test]
    fn program_build() {
        let src = "__kernel void test(__global int *i) { \
                   *i += 1; \
                   }";
        ::test_all_platforms_devices(&mut |device, ctx, _| {
            let prog = ctx.create_program_from_source(src);
            prog.build(device).unwrap();
        })
    }

    #[test]
    fn simple_kernel() {
        let src = "__kernel void test(__global int *i) { \
                   *i += 1; \
                   }";
        ::test_all_platforms_devices(&mut |device, ctx, queue| {
            let prog = ctx.create_program_from_source(src);
            prog.build(device).unwrap();

            let k = prog.create_kernel("test");
            let v = ctx.create_buffer_from(vec![1isize], CL_MEM_READ_WRITE);

            k.set_arg(0, &v);

            queue.enqueue_async_kernel(&k, 1isize, None, ()).wait();

            let v: Vec<isize> = queue.get(&v, ());

            expect!(v[0], 2);
        })
    }

    #[test]
    fn add_k() {
        let src = "__kernel void test(__global int *i, long int k) { \
                   *i += k; \
                   }";

        ::test_all_platforms_devices(&mut |device, ctx, queue| {
            let prog = ctx.create_program_from_source(src);
            prog.build(device).unwrap();

            let k = prog.create_kernel("test");

            let v = ctx.create_buffer_from(vec![1isize], CL_MEM_READ_WRITE);

            k.set_arg(0, &v);
            k.set_arg(1, &42isize);

            queue.enqueue_async_kernel(&k, 1isize, None, ()).wait();

            let v: Vec<isize> = queue.get(&v, ());

            expect!(v[0], 43);
        })
    }

    #[test]
    fn simple_kernel_index() {
        let src = "__kernel void test(__global int *i) { \
                   *i += 1; \
                   }";

        ::test_all_platforms_devices(&mut |device, ctx, queue| {
            let prog = ctx.create_program_from_source(src);
            prog.build(device).unwrap();

            let k = prog.create_kernel("test");

            let v = ctx.create_buffer_from(vec![1isize], CL_MEM_READ_WRITE);

            k.set_arg(0, &v);

            queue.enqueue_async_kernel(&k, 1isize, None, ()).wait();

            let v: Vec<isize> = queue.get(&v, ());

            expect!(v[0], 2);
        })
    }

    #[test]
    fn chain_kernel_event() {
        let src = "__kernel void test(__global int *i) { \
                   *i += 1; \
                   }";

        ::test_all_platforms_devices(&mut |device, ctx, queue| {
            let prog = ctx.create_program_from_source(src);
            prog.build(device).unwrap();

            let k = prog.create_kernel("test");
            let v = ctx.create_buffer_from(vec![1isize], CL_MEM_READ_WRITE);

            k.set_arg(0, &v);

            let mut e : Option<Event> = None;
            for _ in 0isize .. 8 {
                e = Some(queue.enqueue_async_kernel(&k, 1isize, None, e));
            }
            e.wait();

            let v: Vec<isize> = queue.get(&v, ());

            expect!(v[0], 9);
        })
    }

    #[test]
    fn chain_kernel_event_list() {
        let src = "__kernel void inc(__global int *i) { \
                   *i += 1; \
                   } \
                   __kernel void add(__global int *a, __global int *b, __global int *c) { \
                   *c = *a + *b; \
                   }";

        ::test_all_platforms_devices(&mut |device, ctx, queue| {
            let prog = ctx.create_program_from_source(src);
            prog.build(device).unwrap();

            let k_inc_a = prog.create_kernel("inc");
            let k_inc_b = prog.create_kernel("inc");
            let k_add = prog.create_kernel("add");

            let a = ctx.create_buffer_from(vec![1isize], CL_MEM_READ_WRITE);
            let b = ctx.create_buffer_from(vec![1isize], CL_MEM_READ_WRITE);
            let c = ctx.create_buffer_from(vec![1isize], CL_MEM_READ_WRITE);

            k_inc_a.set_arg(0, &a);
            k_inc_b.set_arg(0, &b);

            let event_list = [
                queue.enqueue_async_kernel(&k_inc_a, 1isize, None, ()),
                queue.enqueue_async_kernel(&k_inc_b, 1isize, None, ()),
            ];

            k_add.set_arg(0, &a);
            k_add.set_arg(1, &b);
            k_add.set_arg(2, &c);

            let event = queue.enqueue_async_kernel(&k_add, 1isize, None, &event_list[..]);

            let v: Vec<isize> = queue.get(&c, event);

            expect!(v[0], 4);
        })
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
        ::test_all_platforms_devices(&mut |device, ctx, queue| {
            let prog = ctx.create_program_from_source(src);

            match prog.build(device) {
                Ok(_) => (),
                Err(build_log) => {
                    println!("Error building program:\n");
                    println!("{}", build_log);
                    panic!("");
                }
            }

            let k = prog.create_kernel("test");

            let v = ctx.create_buffer_from(&[1isize, 2, 3, 4, 5, 6, 7, 8, 9][..], CL_MEM_READ_ONLY);

            k.set_arg(0, &v);

            queue.enqueue_async_kernel(&k, (3isize, 3isize), None, ()).wait();

            let v: Vec<isize> = queue.get(&v, ());

            expect!(v, vec!(0, 0, 0, 0, 1, 2, 0, 2, 4));
        })
    }

    #[test]
    fn memory_read_write()
    {
        ::test_all_platforms_devices(&mut |_, ctx, queue| {
            let buffer = ctx.create_buffer1d::<isize>(8, CL_MEM_READ_ONLY);

            let input = [0isize, 1, 2, 3, 4, 5, 6, 7];
            let mut output = [0isize, 0, 0, 0, 0, 0, 0, 0];

            queue.write(&buffer, &&input[..], ());
            queue.read(&buffer, &mut &mut output[..], ());

            expect!(input, output);
        })
    }

    #[test]
    fn memory_read_vec()
    {
        ::test_all_platforms_devices(&mut |_, ctx, queue| {
            let input = [0isize, 1, 2, 3, 4, 5, 6, 7];
            let buffer = ctx.create_buffer_from(&input[..], CL_MEM_READ_WRITE);
            let output: Vec<isize> = queue.get(&buffer, ());
            expect!(&input[..], &output[..]);
        })
    }


    #[test]
    fn memory_read_owned()
    {
        ::test_all_platforms_devices(&mut |_, ctx, queue| {
            let input = vec!(0isize, 1, 2, 3, 4, 5, 6, 7);
            let buffer = ctx.create_buffer_from(&input, CL_MEM_READ_WRITE);
            let output: Vec<isize> = queue.get(&buffer, ());
            expect!(input, output);
        })
    }

    #[test]
    fn memory_read_owned_clone()
    {
        ::test_all_platforms_devices(&mut |_, ctx, queue| {
            let input = vec!(0isize, 1, 2, 3, 4, 5, 6, 7);
            let buffer = ctx.create_buffer_from(input.clone(), CL_MEM_READ_WRITE);
            let output: Vec<isize> = queue.get(&buffer, ());
            expect!(input, output);
        })
    }

    #[test]
    fn memory_alloc_local()
    {
        let src = "__kernel void test(__global int *i, \
                                    __local int *t) { \
                   *t = *i; \
                   *t += 1; \
                   *i = *t; \
                   }";
        ::test_all_platforms_devices(&mut |device, ctx, queue| {
            let prog = ctx.create_program_from_source(src);
            prog.build(device).unwrap();

            let k = prog.create_kernel("test");
            let v = ctx.create_buffer_from(vec![1isize], CL_MEM_READ_WRITE);

            k.set_arg(0, &v);
            k.alloc_local::<isize>(1, 1);

            queue.enqueue_async_kernel(&k, 1isize, None, ()).wait();

            let v: Vec<isize> = queue.get(&v, ());

            expect!(v[0], 2);
        })

    }

    #[test]
    fn event_get_times() {
        let src = "__kernel void test(__global int *i) { \
                   *i += 1; \
                   }";

        let (device, ctx, queue) = opencl::create_compute_context().unwrap();
        let prog = ctx.create_program_from_source(src);
        prog.build(&device).unwrap();

        let k = prog.create_kernel("test");
        let v = ctx.create_buffer_from(vec![1isize], CL_MEM_READ_WRITE);

        k.set_arg(0, &v);

        let e = queue.enqueue_async_kernel(&k, 1isize, None, ());
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
    use opencl::{Buffer2D, Buffer3D};
    use opencl::cl::CL_MEM_READ_WRITE;

    #[test]
    fn put_get_2d()
    {
        ::test_all_platforms_devices(&mut |_, ctx, queue| {
            let arr_in = Buffer2D::new(8, 8, |x, y| {(x+y) as isize});
            let arr_cl = ctx.create_buffer_from(&arr_in, CL_MEM_READ_WRITE);
            let arr_out: Buffer2D<isize> = queue.get(&arr_cl, ());

            for x in 0usize.. 8usize {
                for y in 0usize..8usize {
                    expect!(arr_in.get(x, y), arr_out.get(x, y));
                }
            }
        })
    }


    #[test]
    fn read_write_2d()
    {
        ::test_all_platforms_devices(&mut |_, ctx, queue| {
            let added = Buffer2D::new(8, 8, |x, y| {(x+y) as isize});
            let zero = Buffer2D::new(8, 8, |_, _| {(0) as isize});
            let mut out = Buffer2D::new(8, 8, |_, _| {(0) as isize});

            /* both are zeroed */
            let a_cl = ctx.create_buffer_from(&zero, CL_MEM_READ_WRITE);

            queue.write(&a_cl, &added, ());
            queue.read(&a_cl, &mut out, ());

            for x in 0usize .. 8usize {
                for y in 0usize .. 8usize {
                    expect!(added.get(x, y), out.get(x, y));
                }
            }
        })
    }


    #[test]
    fn kernel_2d()
    {
        ::test_all_platforms_devices(&mut |device, ctx, queue| {
            let mut a = Buffer2D::new(8, 8, |_, _| {(0) as i32});
            let b = Buffer2D::new(8, 8, |x, y| {(x*y) as i32});
            let a_cl = ctx.create_buffer_from(&a, CL_MEM_READ_WRITE);

            let src =  "__kernel void test(__global int *a) { \
                            int x = get_global_id(0); \
                            int y = get_global_id(1); \
                            int size_x = get_global_size(0); \
                            a[size_x*y + x] = x*y; \
                        }";
            let prog = ctx.create_program_from_source(src);
            match prog.build(device) {
                Ok(_) => (),
                Err(build_log) => {
                    println!("Error building program:\n");
                    println!("{}", build_log);
                    panic!("");
                }
            }
            let k = prog.create_kernel("test");

            k.set_arg(0, &a_cl);
            let event = queue.enqueue_async_kernel(&k, (8isize, 8isize), None, ());
            queue.read(&a_cl, &mut a, &event);

            for x in 0usize .. 8usize {
                for y in 0usize .. 8usize {
                    expect!(a.get(x, y), b.get(x, y));
                }
            }
        })
    }

    #[test]
    fn put_get_3d()
    {
        ::test_all_platforms_devices(&mut |_, ctx, queue| {
            let arr_in = Buffer3D::new(8, 8, 8, |x, y, z| {(x+y+z) as isize});
            let arr_cl = ctx.create_buffer_from(&arr_in, CL_MEM_READ_WRITE);
            let arr_out: Buffer3D<isize> = queue.get(&arr_cl, ());

            for x in 0usize .. 8usize {
                for y in 0usize .. 8usize {
                    for z in 0usize .. 8usize {
                        expect!(arr_in.get(x, y, z), arr_out.get(x, y, z));
                    }
                }
            }
        })
    }


    #[test]
    fn read_write_3d()
    {
        ::test_all_platforms_devices(&mut |_, ctx, queue| {
            let added = Buffer3D::new(8, 8, 8, |x, y, z| {(x+y+z) as isize});
            let zero = Buffer3D::new(8, 8, 8, |_, _, _| {(0) as isize});
            let mut out = Buffer3D::new(8, 8, 8, |_, _, _| {(0) as isize});

            /* both are zeroed */
            let a_cl = ctx.create_buffer_from(&zero, CL_MEM_READ_WRITE);

            queue.write(&a_cl, &added, ());
            queue.read(&a_cl, &mut out, ());

            for x in 0usize .. 8usize {
                for y in 0usize .. 8usize {
                    for z in 0usize .. 8usize {
                        expect!(added.get(x, y, z), out.get(x, y, z));
                    }
                }
            }
        })
    }


    #[test]
    fn kernel_3d()
    {
        ::test_all_platforms_devices(&mut |device, ctx, queue| {
            let mut a = Buffer3D::new(8, 8, 8, |_, _, _| {(0) as i32});
            let b = Buffer3D::new(8, 8, 8, |x, y, z| {(x * y * z) as i32});
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
            match prog.build(device) {
                Ok(_) => (),
                Err(build_log) => {
                    println!("Error building program:\n");
                    println!("{}", build_log);
                    panic!("");
                }
            }
            let k = prog.create_kernel("test");

            k.set_arg(0, &a_cl);
            let event = queue.enqueue_async_kernel(&k, (8isize, 8isize, 8isize), None, ());
            queue.read(&a_cl, &mut a, &event);

            for x in 0usize .. 8usize {
                for y in 0usize .. 8usize {
                    for z in 0usize .. 8usize {
                        expect!(a.get(x, y, z), b.get(x, y, z));
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod ext {
    use opencl::ext;
    use opencl::Platform;

    #[test]
    fn try_load_all_extensions() {
        let platforms = Platform::all();

        for platform in platforms.into_iter() {
            let platform_id = platform.get_id();

            macro_rules! check_ext {
                ($ext:ident) => {
                    match ext::$ext::load(platform_id) {
                        Ok(_) => {
                            info!("Extension {} loaded successfully.",
                                  stringify!($ext))
                        }
                        Err(_) => {
                            info!("Error loading extension {}.",
                                  stringify!($ext))
                        }
                    }
                }
            }

            check_ext!(cl_khr_fp64);
            check_ext!(cl_khr_fp16);
            check_ext!(cl_APPLE_SetMemObjectDestructor);
            check_ext!(cl_APPLE_ContextLoggingFunctions);
            check_ext!(cl_khr_icd);
            check_ext!(cl_nv_device_attribute_query);
            check_ext!(cl_amd_device_attribute_query);
            check_ext!(cl_arm_printf);
            check_ext!(cl_ext_device_fission);
            check_ext!(cl_qcom_ext_host_ptr);
            check_ext!(cl_qcom_ion_host_ptr);
        }
    }
}

#[cfg(test)]
mod cl {
    use opencl::cl::CLStatus::*;

    #[test]
    fn clstatus_str() {
        let x = CL_SUCCESS;
        expect!(format!("{}", x), "CL_SUCCESS");

        let y = CL_DEVICE_NOT_FOUND;
        expect!(y.to_string(), "CL_DEVICE_NOT_FOUND");
    }
}
