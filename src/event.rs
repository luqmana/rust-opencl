use libc;
use std::mem;
use std::ptr;
use std::vec::Vec;

use cl::*;
use cl::ll::*;
use error::check;

/// An OpenCL event.
pub struct Event {
    event: cl_event,
}

impl Event {
    /// Creates a new event from its OpenCL pointer.
    ///
    /// The pointer validity is not checked.
    pub unsafe fn new_unchecked(event: cl_event) -> Event {
        Event {
            event: event
        }
    }

    fn get_time(&self, param: cl_uint) -> u64
    {
        unsafe {
            let mut time: cl_ulong = 0;
            let ret = clGetEventProfilingInfo(self.event,
                                    param,
                                    mem::size_of::<cl_ulong>() as libc::size_t,
                                    (&mut time as *mut u64) as *mut libc::c_void,
                                    ptr::null_mut());

            check(ret, "Failed to get profiling info");
            time as u64
        }
    }

    /// Gets the time when the event was queued.
    pub fn queue_time(&self) -> u64
    {
        self.get_time(CL_PROFILING_COMMAND_QUEUED)
    }

    /// Gets the time when the event was submitted to the device.
    pub fn submit_time(&self) -> u64
    {
        self.get_time(CL_PROFILING_COMMAND_SUBMIT)
    }

    /// Gets the time when the event started.
    pub fn start_time(&self) -> u64
    {
        self.get_time(CL_PROFILING_COMMAND_START)
    }

    /// Gets the time when the event ended.
    pub fn end_time(&self) -> u64
    {
        self.get_time(CL_PROFILING_COMMAND_END)
    }
}

impl Drop for Event
{
    fn drop(&mut self) {
        unsafe {
            let status = clReleaseEvent(self.event);
            check(status, "Could not release the event");
        }
    }
}

/// Trait implemented by event lists.
pub trait EventList {
    /// Applies a user-defined function to this list of event.
    fn as_event_list<T, F: FnOnce(*const cl_event, cl_uint) -> T>(&self, F) -> T;

    /// Wait for all the events on this event list.
    fn wait(&self) {
        self.as_event_list(|p, len| {
            unsafe {
                let status = clWaitForEvents(len, p);
                check(status, "Error waiting for event(s)");
            }
        })
    }
}

impl<'r> EventList for &'r Event {
    fn as_event_list<T, F>(&self, f: F) -> T
        where F: FnOnce(*const cl_event, cl_uint) -> T
    {
        f(&self.event, 1 as cl_uint)
    }
}

impl EventList for Event {
    fn as_event_list<T, F>(&self, f: F) -> T
        where F: FnOnce(*const cl_event, cl_uint) -> T
    {
        f(&self.event, 1 as cl_uint)
    }
}

impl<T: EventList> EventList for Option<T> {
    fn as_event_list<T2, F>(&self, f: F) -> T2
        where F: FnOnce(*const cl_event, cl_uint) -> T2
    {
        match *self {
            None => f(ptr::null(), 0),
            Some(ref s) => s.as_event_list(f)
        }
    }
}

impl<'r> EventList for &'r [Event] {
    fn as_event_list<T, F>(&self, f: F) -> T
        where F: FnOnce(*const cl_event, cl_uint) -> T
    {
        let mut vec: Vec<cl_event> = Vec::with_capacity(self.len());
        for item in self.iter(){
            vec.push(item.event);
        }

        f(vec.as_ptr(), vec.len() as cl_uint)
    }
}

/* this seems VERY hackey */
impl EventList for () {
    fn as_event_list<T, F>(&self, f: F) -> T
        where F: FnOnce(*const cl_event, cl_uint) -> T
    {
        f(ptr::null(), 0)
    }
}
