//! Image chanel types.

#![allow(non_camel_case_types)]

use std::mem;
use std::ptr;
use libc::{size_t, c_void};
use std::marker::PhantomData;
use cl::*;
use cl::ll::*;

use error::check;
use program::KernelArg;
use buffer::BufferData;
use context::Context;
use image_channel_type::ImageChannelType;

/// A device-side 2D image.
pub struct Image2D<T: ImageChannelType> {
    width:     size_t,
    height:    size_t,
    cl_buffer: cl_mem,
    phantom:   PhantomData<T>
}

/// A device-side 3D image.
pub struct Image3D<T: ImageChannelType> {
    width:     size_t,
    height:    size_t,
    depth:     size_t,
    cl_buffer: cl_mem,
    phantom:   PhantomData<T>
}

impl<T: ImageChannelType> Image2D<T> {
    /// Creates a 2D image initialized with the content of `data`.
    pub fn new<D: ?Sized>(context:       &Context,
                          width:         usize,
                          height:        usize,
                          data:          &D,
                          channel_order: CLChannelOrder,
                          flags:         cl_mem_flags)
                          -> Image2D<T>
        where D: BufferData<T> {

        assert!(data.len() as usize == width * height , "Error: the data buffer size must match the image size.");

        data.as_raw_data(|raw_data, _| {
            let mut status = 0;

            let format = cl_image_format {
                image_channel_order:     channel_order as u32,
                image_channel_data_type: ImageChannelType::cl_flag(None::<T>)
            };

            let mem = unsafe {
                clCreateImage2D(context.cl_id(),
                                flags | CL_MEM_COPY_HOST_PTR,
                                &format,
                                width  as size_t,
                                height as size_t,
                                0,
                                mem::transmute(raw_data),
                                &mut status)
            };

            check(status, "Could not allocate image");

            Image2D {
                width:     width as size_t,
                height:    height as size_t,
                cl_buffer: mem,
                phantom:   PhantomData
            }
        })
    }

    // FIXME: should be unsafe?
    /// Creates a new uninitialized 2D image.
    pub fn new_uninitialized(context:       &Context,
                             width:         usize,
                             height:        usize,
                             channel_order: CLChannelOrder,
                             flags:         cl_mem_flags)
                             -> Image2D<T> {
        let mut status = 0;

        let format = cl_image_format {
            image_channel_order:     channel_order as u32,
            image_channel_data_type: ImageChannelType::cl_flag(None::<T>)
        };

        let mem = unsafe {
            clCreateImage2D(context.cl_id(),
                            flags,
                            &format,
                            width  as size_t,
                            height as size_t,
                            0,
                            ptr::null_mut(),
                            &mut status)
        };

        check(status, "Could not allocate image");

        Image2D {
            width:     width as size_t,
            height:    height as size_t,
            cl_buffer: mem,
            phantom:   PhantomData
        }
    }

    /// The underlying OpenCL identifier.
    pub fn cl_id_ptr(&self) -> cl_mem {
        self.cl_buffer
    }

    /// The underlying OpenCL identifier.
    pub fn cl_id(&self) -> cl_mem {
        self.cl_buffer
    }

    /// The length in bytes of this image.
    pub fn bytes_len(&self) -> size_t {
        self.len() * mem::size_of::<T>() as size_t
    }

    /// The number of elements of type `T` on this image.
    pub fn len(&self) -> size_t {
        self.width * self.height
    }
}


impl<T: ImageChannelType> Drop for Image2D<T> {
    fn drop(&mut self) {
        unsafe {
            let status = clReleaseMemObject(self.cl_buffer);
            check(status, "Could not release the buffer");
        }
    }
}

impl<T: ImageChannelType> KernelArg for Image2D<T> {
    fn get_value(&self) -> (size_t, *const c_void) {
        (mem::size_of::<cl_mem>() as size_t, &self.cl_buffer as *const cl_mem as *const c_void)
    }
}

impl<T: ImageChannelType> Image3D<T> {
    /// Creates a 3D image initialized with the content of `data`.
    pub fn new<D: ?Sized>(context:       &Context,
                          width:         usize,
                          height:        usize,
                          depth:         usize,
                          data:          &D,
                          channel_order: CLChannelOrder,
                          flags:         cl_mem_flags)
                          -> Image3D<T>
        where D: BufferData<T> {

        assert!(data.len() as usize == width * height * depth, "Error: the data buffer size must match the image size.");

        data.as_raw_data(|raw_data, _| {
            let mut status = 0;

            let format = cl_image_format {
                image_channel_order:     channel_order as u32,
                image_channel_data_type: ImageChannelType::cl_flag(None::<T>)
            };

            let mem = unsafe {
                clCreateImage3D(context.cl_id(),
                                flags | CL_MEM_COPY_HOST_PTR,
                                &format,
                                width as size_t,
                                height as size_t,
                                depth as size_t,
                                0,
                                0,
                                mem::transmute(raw_data),
                                &mut status)
            };

            check(status, "Could not allocate image");

            Image3D {
                width:     width as size_t,
                height:    height as size_t,
                depth:     depth as size_t,
                cl_buffer: mem,
                phantom:   PhantomData
            }
        })
    }

    // FIXME: should be unsafe?
    /// Creates a new uninitialized 3D image.
    pub fn new_uninitialized(context:       &Context,
                             width:         usize,
                             height:        usize,
                             depth:         usize,
                             channel_order: CLChannelOrder,
                             flags:         cl_mem_flags)
                             -> Image3D<T> {
        let mut status = 0;

        let format = cl_image_format {
            image_channel_order:     channel_order as u32,
            image_channel_data_type: ImageChannelType::cl_flag(None::<T>)
        };

        let mem = unsafe {
            clCreateImage3D(context.cl_id(),
                            flags,
                            &format,
                            width as size_t,
                            height as size_t,
                            depth as size_t,
                            0,
                            0,
                            ptr::null_mut(),
                            &mut status)
        };

        check(status, "Could not allocate image");

        Image3D {
            width:     width as size_t,
            height:    height as size_t,
            depth:     depth as size_t,
            cl_buffer: mem,
            phantom:   PhantomData
        }
    }

    /// The underlying OpenCL identifier.
    pub fn cl_id_ptr(&self) -> cl_mem {
        self.cl_buffer
    }

    /// The underlying OpenCL identifier.
    pub fn cl_id(&self) -> cl_mem {
        self.cl_buffer
    }

    /// The length in bytes of this image.
    pub fn bytes_len(&self) -> size_t {
        self.len() * mem::size_of::<T>() as size_t
    }

    /// The number of elements of type `T` on this image.
    pub fn len(&self) -> size_t {
        self.width * self.height * self.depth
    }
}


impl<T: ImageChannelType> Drop for Image3D<T> {
    fn drop(&mut self) {
        unsafe {
            let status = clReleaseMemObject(self.cl_buffer);
            check(status, "Could not release the buffer");
        }
    }
}

impl<T: ImageChannelType> KernelArg for Image3D<T> {
    fn get_value(&self) -> (size_t, *const c_void) {
        (mem::size_of::<cl_mem>() as size_t, &self.cl_buffer as *const cl_mem as *const c_void)
    }
}
