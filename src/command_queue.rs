//! A higher level API.

use libc;
use std::ptr;

use cl::*;
use cl::ll::*;
use error::check;
use buffer::{BufferData, Buffer};
use program::{Kernel, KernelIndex};
use event::{Event, EventList};
use context::Context;
use device::Device;

/// An OpenCL command queue.
pub struct CommandQueue {
    cqueue: cl_command_queue
}

unsafe impl Sync for CommandQueue { }
unsafe impl Send for CommandQueue { }

impl CommandQueue {
    /// Creates a new command queue for the given device.
    pub fn new(context: &Context, device: &Device, profiling: bool, out_of_order: bool) -> CommandQueue {
        let mut errcode = 0;

        let mut props = 0;

        if profiling {
            props = props | CL_QUEUE_PROFILING_ENABLE;
        }

        if out_of_order {
            props = props | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        }

        let cqueue = unsafe {
            clCreateCommandQueue(context.cl_id(),
                                 device.cl_id(),
                                 props,
                                 (&mut errcode))
        };

        check(errcode, "Failed to create command queue!");

        CommandQueue {
            cqueue: cqueue
        }
    }

    /// Synchronously enqueues a kernel for execution on the device.
    pub fn enqueue_kernel<I: KernelIndex, E: EventList>(&self, k: &Kernel, global: I, local: Option<I>, wait_list: E) {
        self.enqueue_async_kernel(k, global, local, wait_list).wait()
    }

    /// Asynchronously enqueues a kernel for execution on the device.
    pub fn enqueue_async_kernel<I: KernelIndex, E: EventList>(&self, k: &Kernel, global: I, local: Option<I>, wait_list: E)
        -> Event {
        unsafe {
            wait_list.as_event_list(|event_list, event_list_length| {
                let mut e: cl_event = ptr::null_mut();
                let status = clEnqueueNDRangeKernel(
                    self.cqueue,
                    k.cl_id(),
                    KernelIndex::num_dimensions(None::<I>),
                    ptr::null(),
                    global.get_ptr(),
                    match local {
                        Some(ref l) => l.get_ptr() as *const libc::size_t,
                        None => ptr::null()
                    },
                    event_list_length,
                    event_list,
                    (&mut e));
                check(status, "Error enqueuing kernel.");

                Event::new_unchecked(e)
            })
        }
    }

    /// Synchronously Acquire OpenCL memory objects that have been created from OpenGL objects.
    pub fn enqueue_acquire_gl_buffer<T: Copy, E: EventList>(&self, mem: &Buffer<T>, wait_list: E) {
        self.enqueue_async_acquire_gl_buffer(mem, wait_list).wait()
    }

    /// Asynchronously Acquire OpenCL memory objects that have been created from OpenGL objects.
    pub fn enqueue_async_acquire_gl_buffer<T: Copy, E: EventList>(&self, mem: &Buffer<T>, wait_list: E)
        -> Event {

        unsafe {
            wait_list.as_event_list(|event_list, event_list_length| {
                let mut e: cl_event = ptr::null_mut();
                let status = clEnqueueAcquireGLObjects(
                    self.cqueue,
                    1,
                    &mem.cl_id(),
                    event_list_length,
                    event_list,
                    (&mut e));

                check(status, "Failed to acquire buffer");

                Event::new_unchecked(e)
            })
        }
    }

    /// Synchronously Release OpenCL memory objects that have been created from OpenGL objects.
    pub fn enqueue_release_gl_buffer<T: Copy, E: EventList>(&self, mem: &Buffer<T>, wait_list: E) {
        self.enqueue_async_release_gl_buffer(mem, wait_list).wait()
    }

    /// Asynchronously Release OpenCL memory objects that have been created from OpenGL objects.
    pub fn enqueue_async_release_gl_buffer<T: Copy, E: EventList>(&self, mem: &Buffer<T>, wait_list: E)
        -> Event {

        unsafe {
            wait_list.as_event_list(|event_list, event_list_length| {
                let mut e: cl_event = ptr::null_mut();
                let status = clEnqueueReleaseGLObjects(
                    self.cqueue,
                    1,
                    &mem.cl_id(),
                    event_list_length,
                    event_list,
                    (&mut e));

                check(status, "Failed to acquire buffer");

                Event::new_unchecked(e)
            })
        }
    }

    fn do_write<T: Copy, U: ?Sized, E>(&self, mem: &Buffer<T>, data: &U, wait_list: E, out_event: *mut cl_event)
        where U: BufferData<T>,
              E: EventList {
        unsafe {
            wait_list.as_event_list(|event_list, event_list_length| {
                data.as_raw_data(|raw_data, sz| {
                    assert!(sz == mem.bytes_len(), "Mismatched size for writing into a device buffer.");

                    let blocking = if out_event.is_null() { CL_TRUE } else { CL_FALSE };

                    let err = clEnqueueWriteBuffer(
                        self.cqueue,
                        mem.cl_id(),
                        blocking,
                        0,
                        sz as libc::size_t,
                        raw_data as *const libc::c_void,
                        event_list_length,
                        event_list,
                        out_event);

                    check(err, "Failed to write buffer");
                })
            })
        }
    }

    /// Synchronously writes `data` to a device-side memory object `mem`.
    pub fn write<T: Copy, U: ?Sized, E>(&self, mem: &Buffer<T>, data: &U, wait_list: E)
        where U: BufferData<T>,
              E: EventList {
        self.do_write(mem, data, wait_list, ptr::null_mut())
    }

    /// Asynchronously writes `data` to a device-side memory object `mem`.
    pub fn write_async<T: Copy, U: ?Sized, E>(&self, mem: &Buffer<T>, data: &U, wait_list: E) -> Event
        where U: BufferData<T>,
              E: EventList {
        let mut e: cl_event = ptr::null_mut();
        self.do_write(mem, data, wait_list, &mut e);

        assert!(!e.is_null(), "The event was not created properly.");
        unsafe { Event::new_unchecked(e) }
    }

    fn do_read<T: Copy, E>(&self,
                           mem:       &Buffer<T>,
                           raw_data:  *mut libc::c_void,
                           sz:        libc::size_t,
                           wait_list: E,
                           out_event: *mut cl_event)
        where E: EventList {
        unsafe {
            wait_list.as_event_list(|event_list, event_list_length| {
                assert!(sz == mem.bytes_len(), "Mismatched size for reading from a device buffer.");

                let blocking = if out_event.is_null() { CL_TRUE } else { CL_FALSE };

                let err = clEnqueueReadBuffer(self.cqueue,
                                              mem.cl_id(),
                                              blocking,
                                              0,
                                              sz as libc::size_t,
                                              raw_data as *mut libc::c_void,
                                              event_list_length,
                                              event_list,
                                              out_event);

                check(err, "Failed to reading buffer");
            })
        }
    }

    /// Synchronously reads `mem` data to the device-side memory object `out`.
    pub fn read<T: Copy, U: ?Sized, E>(&self, mem: &Buffer<T>, out: &mut U, wait_list: E)
        where U: BufferData<T>,
              E: EventList {
        out.as_raw_data_mut(|raw_data, sz| {
            self.do_read(mem, raw_data, sz, wait_list, ptr::null_mut())
        })
    }

    /// Asynchronously reads `mem` data to the device-side memory object `out`.
    pub fn read_async<T: Copy, U: ?Sized, E>(&self, mem: &Buffer<T>, out: &mut U, wait_list: E) -> Event
        where U: BufferData<T>,
              E: EventList {
        let mut e: cl_event = ptr::null_mut();

        out.as_raw_data_mut(|raw_data, sz| {
            self.do_read(mem, raw_data, sz, wait_list, &mut e)
        });

        assert!(!e.is_null(), "The event was not created properly.");
        unsafe { Event::new_unchecked(e) }
    }
}

impl Drop for CommandQueue {
    fn drop(&mut self) {
        unsafe {
            let status = clReleaseCommandQueue(self.cqueue);
            check(status, "Could not release the command queue.");
        }
    }
}
