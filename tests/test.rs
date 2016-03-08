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
            let queue = CommandQueue::new(&context, &d, false, false);
            test(d, &context, &queue);
        }
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
            let prog = Program::new(ctx, src);
            prog.build(device).unwrap();
        })
    }

    #[test]
    fn simple_kernel() {
        let src = "__kernel void test(__global int *i) { \
                        *i += 1; \
                   }";
        ::test_all_platforms_devices(&mut |device, ctx, queue| {
            let prog = Program::new(ctx, src);
            prog.build(device).unwrap();

            let k = prog.create_kernel("test");
            let v = Buffer::new(&ctx, &[ 1isize ][..], CL_MEM_READ_WRITE);

            k.set_arg(0, &v);

            queue.enqueue_async_kernel(&k, 1isize, None, ()).wait();

            let mut out = [ 0isize; 1 ];
            queue.read(&v, &mut out[..], ());

            expect!(out[0], 2);
        })
    }

    #[test]
    fn add_k() {
        let src = "__kernel void test(__global int *i, long int k) { \
                       *i += k; \
                   }";

        ::test_all_platforms_devices(&mut |device, ctx, queue| {
            let prog = Program::new(ctx, src);
            prog.build(device).unwrap();

            let k = prog.create_kernel("test");

            let v = Buffer::new(&ctx, &[1isize][..], CL_MEM_READ_WRITE);

            k.set_arg(0, &v);
            k.set_arg(1, &42isize);

            queue.enqueue_async_kernel(&k, 1isize, None, ()).wait();

            let mut out = [ 0isize; 1 ];
            queue.read(&v, &mut out[..], ());

            expect!(out[0], 43);
        })
    }

    #[test]
    fn simple_kernel_index() {
        let src = "__kernel void test(__global int *i) { \
                        *i += 1; \
                   }";

        ::test_all_platforms_devices(&mut |device, ctx, queue| {
            let prog = Program::new(ctx, src);
            prog.build(device).unwrap();

            let k = prog.create_kernel("test");

            let v = Buffer::new(&ctx, &[1isize][..], CL_MEM_READ_WRITE);

            k.set_arg(0, &v);

            queue.enqueue_async_kernel(&k, 1isize, None, ()).wait();

            let mut out = [ 0isize; 1 ];
            queue.read(&v, &mut out[..], ());

            expect!(out[0], 2);
        })
    }

    #[test]
    fn chain_kernel_event() {
        let src = "__kernel void test(__global int *i) { \
                        *i += 1; \
                   }";

        ::test_all_platforms_devices(&mut |device, ctx, queue| {
            let prog = Program::new(ctx, src);
            prog.build(device).unwrap();

            let k = prog.create_kernel("test");
            let v = Buffer::new(&ctx, &[1isize][..], CL_MEM_READ_WRITE);

            k.set_arg(0, &v);

            let mut e : Option<Event> = None;
            for _ in 0isize .. 8 {
                e = Some(queue.enqueue_async_kernel(&k, 1isize, None, e));
            }
            e.wait();

            let mut out = [ 0isize; 1 ];
            queue.read(&v, &mut out[..], ());

            expect!(out[0], 9);
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
            let prog = Program::new(ctx, src);
            prog.build(device).unwrap();

            let k_inc_a = prog.create_kernel("inc");
            let k_inc_b = prog.create_kernel("inc");
            let k_add = prog.create_kernel("add");

            let a = Buffer::new(&ctx, &[1isize][..], CL_MEM_READ_WRITE);
            let b = Buffer::new(&ctx, &[1isize][..], CL_MEM_READ_WRITE);
            let c = Buffer::new(&ctx, &[1isize][..], CL_MEM_READ_WRITE);

            k_inc_a.set_arg(0, &a);
            k_inc_b.set_arg(0, &b);

            let event_list = [
                queue.enqueue_async_kernel(&k_inc_a, 1isize, None, ()),
                queue.enqueue_async_kernel(&k_inc_b, 1isize, None, ()),
            ];

            k_add.set_arg(0, &a);
            k_add.set_arg(1, &b);
            k_add.set_arg(2, &c);

            queue.enqueue_async_kernel(&k_add, 1isize, None, &event_list[..]).wait();

            let mut out = [ 0isize; 1 ];
            queue.read(&c, &mut out[..], ());

            expect!(out[0], 4);
        })
    }

    #[test]
    fn kernel_2d()
    {
        let src = "__kernel void test(__global long int *N) { \
                       int i = get_global_id(0); \
                       int j = get_global_id(1); \
                       int s = get_global_size(0); \
                       N[i * s + j] = i * j; \
                   }";
        ::test_all_platforms_devices(&mut |device, ctx, queue| {
            let prog = Program::new(ctx, src);

            match prog.build(device) {
                Ok(_) => (),
                Err(build_log) => {
                    println!("Error building program:\n");
                    println!("{}", build_log);
                    panic!("");
                }
            }

            let k = prog.create_kernel("test");

            let v = Buffer::new(&ctx, &[1isize, 2, 3, 4, 5, 6, 7, 8, 9][..], CL_MEM_READ_WRITE);

            k.set_arg(0, &v);

            queue.enqueue_async_kernel(&k, (3isize, 3isize), None, ()).wait();

            let mut out = [ 0isize; 9 ];
            queue.read(&v, &mut out[..], ());

            expect!(out, [ 0, 0, 0, 0, 1, 2, 0, 2, 4 ]);
        })
    }

    #[test]
    fn memory_read_write()
    {
        ::test_all_platforms_devices(&mut |_, ctx, queue| {
            let buffer = Buffer::<isize>::new_uninitialized(&ctx, 8, CL_MEM_READ_ONLY);

            let input = [0isize, 1, 2, 3, 4, 5, 6, 7];
            let mut output = [0isize, 0, 0, 0, 0, 0, 0, 0];

            queue.write(&buffer, &input[..], ());
            queue.read(&buffer, &mut output[..], ());

            expect!(input, output);
        })
    }

    #[test]
    fn memory_read_vec()
    {
        ::test_all_platforms_devices(&mut |_, ctx, queue| {
            let input = [0isize, 1, 2, 3, 4, 5, 6, 7];
            let mut output = [0isize, 0, 0, 0, 0, 0, 0, 0];

            let buffer = Buffer::new(&ctx, &input[..], CL_MEM_READ_WRITE);

            queue.write(&buffer, &input[..], ());
            queue.read(&buffer, &mut output[..], ());

            expect!(&input[..], &output[..]);
        })
    }


    #[test]
    fn memory_read_owned()
    {
        ::test_all_platforms_devices(&mut |_, ctx, queue| {
            let input = vec!(0isize, 1, 2, 3, 4, 5, 6, 7);
            let mut output = vec!(0isize, 0, 0, 0, 0, 0, 0, 0);

            let buffer = Buffer::new(&ctx, &input[..], CL_MEM_READ_WRITE);

            queue.write(&buffer, &input[..], ());
            queue.read(&buffer, &mut output[..], ());

            expect!(input, output);
        })
    }

    #[test]
    fn memory_read_owned_clone()
    {
        ::test_all_platforms_devices(&mut |_, ctx, queue| {
            let input = vec!(0isize, 1, 2, 3, 4, 5, 6, 7);
            let mut output = vec!(0isize, 0, 0, 0, 0, 0, 0, 0);

            let buffer = Buffer::new(&ctx, &input.clone(), CL_MEM_READ_WRITE);

            queue.write(&buffer, &input[..], ());
            queue.read(&buffer, &mut output[..], ());

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
            let prog = Program::new(ctx, src);
            prog.build(device).unwrap();

            let k = prog.create_kernel("test");
            let v = Buffer::new(&ctx, &[1isize][..], CL_MEM_READ_WRITE);

            k.set_arg(0, &v);
            k.alloc_local::<isize>(1, 1);

            queue.enqueue_async_kernel(&k, 1isize, None, ()).wait();

            let mut out = [ 0isize; 1 ];
            queue.read(&v, &mut out[..], ());

            expect!(out[0], 2);
        })

    }

    #[test]
    fn event_get_times() {
        let src = "__kernel void test(__global int *i) { \
                      *i += 1; \
                   }";

        let (device, ctx, queue) = opencl::create_compute_context(true).unwrap();
        let prog = Program::new(&ctx, src);
        prog.build(&device).unwrap();

        let k = prog.create_kernel("test");
        let v = Buffer::new(&ctx, &[1isize][..], CL_MEM_READ_WRITE);

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
