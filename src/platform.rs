use libc;
use std;
use std::iter::repeat;
use std::ptr;
use std::string::String;
use std::vec::Vec;

use cl::*;
use cl::ll::*;
use error;
use device::{Device, DeviceType};

// This mutex is used to work around weak OpenCL implementations.
// On some implementations concurrent calls to clGetPlatformIDs
// will cause the implantation to return invalid status.
static mut platforms_mutex: std::sync::StaticMutex = std::sync::MUTEX_INIT;

/// Retrieves all the platform available in the system.
pub fn platforms() -> Vec<Platform>
{
    let mut num_platforms = 0 as cl_uint;

    unsafe
    {
        let guard = platforms_mutex.lock();
        let status = clGetPlatformIDs(0,
                                          ptr::null_mut(),
                                          (&mut num_platforms));
        // unlock this before the check in case the check fails
        error::check(status, "could not get platform count.");

        let mut ids: Vec<cl_device_id> = repeat(0 as cl_device_id)
            .take(num_platforms as usize).collect();

        let status = clGetPlatformIDs(num_platforms,
                                      ids.as_mut_ptr(),
                                      (&mut num_platforms));
        error::check(status, "could not get platforms.");

        let _ = guard;

        ids.iter().map(|id| { Platform { id: *id } }).collect()
    }
}

/// An OpenCL platform.
pub struct Platform {
    id: cl_platform_id
}

impl Platform {
    /// Retrieves all the platforms available in the system.
    pub fn all() -> Vec<Platform> {
        platforms()
    }

    /// Returns the first platform available on the system.
    pub fn first() -> Platform {
        Platform::all().swap_remove(0)
    }

    fn get_devices_internal(&self, dtype: cl_device_type) -> Vec<Device>
    {
        unsafe
        {
            let mut num_devices = 0;

            info!("Looking for devices matching {}", dtype);

            let status = clGetDeviceIDs(self.id, dtype, 0, ptr::null_mut(),
                                         (&mut num_devices));
            error::check(status, "Could not determine the number of devices");

            let mut ids: Vec<cl_device_id> = repeat(0 as cl_device_id)
                .take(num_devices as usize).collect();
            let status = clGetDeviceIDs(self.id, dtype, ids.len() as cl_uint,
                                        ids.as_mut_ptr(), (&mut num_devices));
            error::check(status, "Could not retrieve the list of devices");

            ids.iter().map(|id| { Device::new_unchecked(*id) }).collect()
        }
    }

    /// Gets all the devices available with this platform.
    pub fn get_devices(&self) -> Vec<Device>
    {
        self.get_devices_internal(CL_DEVICE_TYPE_ALL)
    }

    /// Gets all devices of the specified types available with this platform.
    pub fn get_devices_by_types(&self, types: &[DeviceType]) -> Vec<Device>
    {
        let mut dtype = 0;
        for &t in types.iter() {
          dtype |= t.to_cl_device_type();
        }

        self.get_devices_internal(dtype)
    }

    fn profile_info(&self, name: cl_platform_info) -> String
    {
        unsafe {
            let mut size = 0 as libc::size_t;

            let status = clGetPlatformInfo(self.id,
                              name,
                              0,
                              ptr::null_mut(),
                              &mut size);
            error::check(status, "Could not determine platform info string length");

            let mut buf : Vec<u8>
                = repeat(0u8).take(size as usize).collect();

            let status = clGetPlatformInfo(self.id,
                              name,
                              size,
                              buf.as_mut_ptr() as *mut libc::c_void,
                              ptr::null_mut());
            error::check(status, "Could not get platform info string");

            String::from_utf8_unchecked(buf)
        }
    }

    /// Gets the OpenCL platform identifier.
    pub fn get_id(&self) -> cl_platform_id {
        self.id
    }

    /// Gets the platform name.
    pub fn name(&self) -> String
    {
        self.profile_info(CL_PLATFORM_NAME)
    }

    /// Gets the platform version.
    pub fn version(&self) -> String
    {
        self.profile_info(CL_PLATFORM_VERSION)
    }

    /// Gets the platform profile.
    pub fn profile(&self) -> String
    {
        self.profile_info(CL_PLATFORM_PROFILE)
    }

    /// Gets the platform vendor.
    pub fn vendor(&self) -> String
    {
        self.profile_info(CL_PLATFORM_VENDOR)
    }

    /// Gets the support platform extensions.
    pub fn extensions(&self) -> String
    {
        self.profile_info(CL_PLATFORM_EXTENSIONS)
    }

    /// Unsafely creates a platform from its identifier.
    ///
    /// The identifier validity is not checked.
    pub unsafe fn from_platform_id(id: cl_platform_id) -> Platform {
        Platform { id: id }
    }
}
