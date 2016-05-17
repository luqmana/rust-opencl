use libc;
use std::iter::repeat;
use std::ptr;
use std::string::String;
use std::vec::Vec;

use cl::*;
use cl::ll::*;
use error::check;

/// The type of selectable device.
#[derive(Copy, Clone)]
pub enum DeviceType {
    /// A CPU device.
    CPU,
    /// A GPU device.
    GPU
}

impl DeviceType {
    /// Converts this enumeration to the corresponding OpenCL flags.
    pub fn to_cl_device_type(self) -> cl_device_type {
        match self {
            DeviceType::CPU => CL_DEVICE_TYPE_CPU,
            DeviceType::GPU => CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR
        }
    }
}

/// And OpenCL device.
#[derive(Copy, Clone)]
pub struct Device {
    id: cl_device_id
}

unsafe impl Sync for Device {}
unsafe impl Send for Device {}

impl Device {
    fn profile_info(&self, name: cl_device_info) -> String
    {
        unsafe {
            let mut size = 0 as libc::size_t;

            let status = clGetDeviceInfo(
                self.id,
                name,
                0,
                ptr::null_mut(),
                &mut size);
            check(status, "Could not determine device info string length");

            let mut buf : Vec<u8>
                = repeat(0u8).take(size as usize).collect();

            let status = clGetDeviceInfo(self.id,
                                         name,
                                         size,
                                         buf.as_mut_ptr() as *mut libc::c_void,
                                         ptr::null_mut());
            check(status, "Could not get device info string");

            String::from_utf8_unchecked(buf)
        }
    }

    /// Creates a new device from its OpenCL identifier.
    ///
    /// The identifier validity is not checked.
    pub unsafe fn new_unchecked(id: cl_device_id) -> Device {
        Device {
            id: id
        }
    }

    /// The device name.
    pub fn name(&self) -> String
    {
        self.profile_info(CL_DEVICE_NAME)
    }
    /// The device vendor.
    pub fn vendor(&self) -> String
    {
        self.profile_info(CL_DEVICE_VENDOR)
    }

    /// The device profile.
    pub fn profile(&self) -> String
    {
        self.profile_info(CL_DEVICE_PROFILE)
    }

    /// The device type.
    pub fn device_type(&self) -> String
    {
        self.profile_info(CL_DEVICE_TYPE)
    }

    /// The maximum number of compute units of this device.
    pub fn compute_units(&self) -> usize {
		unsafe {
			let mut ct: usize = 0;
            let status = clGetDeviceInfo(
                self.id,
                CL_DEVICE_MAX_COMPUTE_UNITS,
                8,
                (&mut ct as *mut usize) as *mut libc::c_void,
                ptr::null_mut());
            check(status, "Could not get number of device compute units.");
			return ct;
		}
	}

    /// The global memory size of this device.
    pub fn global_mem_size(&self) -> usize {
        unsafe {
            let mut ct: usize = 0;
            let status = clGetDeviceInfo(
                self.id,
                CL_DEVICE_GLOBAL_MEM_SIZE,
                16,
                (&mut ct as *mut usize) as *mut libc::c_void,
                ptr::null_mut());
            check(status, "Could not get size of global memory.");
            return ct;
        }
    }

    /// The local memory size of this device.
    pub fn local_mem_size(&self) -> usize {
        unsafe {
            let mut ct: usize = 0;
            let status = clGetDeviceInfo(
                self.id,
                CL_DEVICE_LOCAL_MEM_SIZE,
                16,
                (&mut ct as *mut usize) as *mut libc::c_void,
                ptr::null_mut());
            check(status, "Could not get size of local memory.");
            return ct;
        }
    }

    /// The maximum memory allocation size of this device.
    pub fn max_mem_alloc_size(&self) -> usize {
        unsafe {
            let mut ct: usize = 0;
            let status = clGetDeviceInfo(
                self.id,
                CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                16,
                (&mut ct as *mut usize) as *mut libc::c_void,
                ptr::null_mut());
            check(status, "Could not get size of local memory.");
            return ct;
        }
    }

    /// The device OpenCL id.
    pub fn cl_id(&self) -> cl_device_id {
        self.id
    }
}
