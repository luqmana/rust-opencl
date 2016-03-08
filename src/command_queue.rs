//! A higher level API.

use libc;
use std::ptr;

use cl::*;
use cl::ll::*;
use error::check;
use mem::{Get, Write, Read, CLBuffer};
use program::{Kernel, KernelIndex};
use event::{Event, EventList};

/// An OpenCL command queue.
pub struct CommandQueue {
    cqueue: cl_command_queue
}

unsafe impl Sync for CommandQueue {}
unsafe impl Send for CommandQueue {}

impl CommandQueue {
    /// Creates a new command queue from its OpenCL raw pointer.
    ///
    /// The pointer validity is not checked.
    pub unsafe fn new_unchecked(cqueue: cl_command_queue) -> CommandQueue {
        CommandQueue {
            cqueue: cqueue
        }
    }

    /// Synchronously enqueues a kernel for execution on the device.
    pub fn enqueue_kernel<I: KernelIndex, E: EventList>(&self, k: &Kernel, global: I, local: Option<I>, wait_on: E)
        -> Event
    {
        unsafe
        {
            wait_on.as_event_list(|event_list, event_list_length| {
                let mut e: cl_event = ptr::null_mut();
                let mut status = clEnqueueNDRangeKernel(
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
                status = clFinish(self.cqueue);
                check(status, "Error finishing kernel.");

                Event::new_unchecked(e)
            })
        }
    }

    /// Asynchronously enqueues a kernel for execution on the device.
    pub fn enqueue_async_kernel<I: KernelIndex, E: EventList>(&self, k: &Kernel, global: I, local: Option<I>, wait_on: E)
        -> Event
    {
        unsafe
        {
            wait_on.as_event_list(|event_list, event_list_length| {
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

    /// Synchronously writes `data` to a device-side memory object `mem`.
    pub fn write<U: Write, T, E: EventList, B: CLBuffer<T>>(&self, mem: &B, data: &U, event: E)
    {
        unsafe {
            event.as_event_list(|event_list, event_list_length| {
                data.write(|offset, p, len| {
                    let err = clEnqueueWriteBuffer(self.cqueue,
                                                   mem.cl_id(),
                                                   CL_TRUE,
                                                   offset as libc::size_t,
                                                   len as libc::size_t,
                                                   p as *const libc::c_void,
                                                   event_list_length,
                                                   event_list,
                                                   ptr::null_mut());

                    check(err, "Failed to write buffer");
                })
            })
        }
    }

    /// Asynchronously writes `data` to a device-side memory object `mem`.
    pub fn write_async<U: Write, T, E: EventList, B: CLBuffer<T>>(&self, mem: &B, data: &U, event: E) -> Event
    {
        unsafe {
            let mut out_event = None;

            event.as_event_list(|evt, evt_len| {
                data.write(|offset, p, len| {
                    let mut e: cl_event = ptr::null_mut();
                    let err = clEnqueueWriteBuffer(self.cqueue,
                                                   mem.cl_id(),
                                                   CL_FALSE,
                                                   offset as libc::size_t,
                                                   len as libc::size_t,
                                                   p as *const libc::c_void,
                                                   evt_len,
                                                   evt,
                                                   &mut e);
                    out_event = Some(e);
                    check(err, "Failed to write buffer");
                })
            });

            Event::new_unchecked(out_event.unwrap())
        }
    }

    /// Synchronously reads `mem` to a host-side memory object of type `G`.
    pub fn get<T, U, B: CLBuffer<T>, G: Get<B, U>, E: EventList>(&self, mem: &B, event: E) -> G
    {
        event.as_event_list(|event_list, event_list_length| {
            Get::get(mem, |offset, ptr, len| {
                unsafe {
                    let err = clEnqueueReadBuffer(self.cqueue,
                                                  mem.cl_id(),
                                                  CL_TRUE,
                                                  offset as libc::size_t,
                                                  len,
                                                  ptr,
                                                  event_list_length,
                                                  event_list,
                                                  ptr::null_mut());

                    check(err, "Failed to read buffer");
                }
            })
        })
    }

    /// Synchronously reads `mem` data to the device-side memory object `out`.
    pub fn read<T, U: Read, E: EventList, B: CLBuffer<T>>(&self, mem: &B, out: &mut U, event: E)
    {
        event.as_event_list(|event_list, event_list_length| {
                out.read(|offset, p, len| {
                        unsafe {
                            let err = clEnqueueReadBuffer(self.cqueue,
                                                          mem.cl_id(),
                                                          CL_TRUE,
                                                          offset as libc::size_t,
                                                          len as libc::size_t,
                                                          p as *mut libc::c_void,
                                                          event_list_length,
                                                          event_list,
                                                          ptr::null_mut());
                            
                            check(err, "Failed to read buffer");
                        }
                    })
            })
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
