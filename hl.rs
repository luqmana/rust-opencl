// Higher level OpenCL wrappers.

use CL::*;
use CL::ll::*;

struct Platform {
    id: cl_platform_id
}

pub fn get_platforms() -> ~[Platform] {
    let mut num_platforms = 0;

    // TODO: Check result status
    clGetPlatformIDs(0, ptr::null(), ptr::addr_of(&num_platforms));

    let ids = vec::to_mut(vec::from_elem(num_platforms as uint,
                                         0 as cl_platform_id));

    do vec::as_imm_buf(ids) |ids, len| {
        clGetPlatformIDs(len as cl_uint,
                         ids, ptr::addr_of(&num_platforms))
    };

    do ids.map |id| { Platform { id: *id } }
}

struct Device {
    id: cl_device_id
}

pub fn get_devices(platform: Platform) -> ~[Device] {
    let mut num_devices = 0;

    clGetDeviceIDs(platform.id, CL_DEVICE_TYPE_GPU, 0, ptr::null(), 
                   ptr::addr_of(&num_devices));
    
    let ids = vec::to_mut(vec::from_elem(num_devices as uint, 
                                         0 as cl_device_id));
    do vec::as_imm_buf(ids) |ids, len| {
        clGetDeviceIDs(platform.id, CL_DEVICE_TYPE_GPU, len as cl_uint, 
                       ids, ptr::addr_of(&num_devices));
    };

    do ids.map |id| { Device {id: *id }}
}

struct Context {
    ctx: cl_context,

    drop {
        clReleaseContext(self.ctx);
    }
}
