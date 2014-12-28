#![allow(unused, raw_pointer_deriving, non_camel_case_types, non_snake_case)]

/// All of the extensions defined for OpenCL 1.1, from
/// [`cl_ext.h`](https://www.khronos.org/registry/cl/api/1.1/cl_ext.h).

// Macro to define a struct-style extension pointer loader.
// Defines a (`Copy`, `Sync`) `struct Functions` that has extension function pointers as members (and methods,
// for convenience).
// Call `$ext::load()` to get an Option<$ext::Functions, String>.  (It's safe to call extension function pointers in other threads, right?)
macro_rules! cl_extension_loader {
    (
        $ext_name:expr;
        $(extern fn $function:ident ($($arg:ident : $arg_type:ty),*) -> $ret:ty;)*
    ) => (
        // Define struct Functions
        ext_struct_def!{ $($function ($($arg $arg_type)*) $ret)* }
        impl Functions {
            // Make function pointers available as methods so they don't have to be called as (struct.member)(arg)
            $( #[inline(always)] unsafe fn $function (&self, $($arg:$arg_type),*) -> $ret { (self.$function)($($arg),*) } )*
        }

        pub fn load(platform: cl_platform_id) -> Result<Functions, String> {
            use hl;
            use std::mem;
            use std::ptr;
            use cl::ll::clGetExtensionFunctionAddress;

            // Read in the available extensions
            // We have to do this, since loading function pointers for an 
            // unavailable extension can return non-NULL.
            // TODO read in extensions lazily and store them in a global HashSet?
            let available = unsafe {
                let hl_platform = hl::Platform::from_platform_id(platform);
                let available = hl_platform.extensions().contains($ext_name);
                mem::forget(hl_platform);
                available
            };
            if !available {
                return Err(format!("extension {} unavailable for platform with id {}", $ext_name, platform));
            }
            // Return a struct with all functions loaded
            Ok(ext_struct_literal!($ext_name platform $($function)*))
        }
    )
}
// We only need these helper macros so we can special-case for unit structs
// (Since writing `struct name {}` is a failing error for some reason)
macro_rules! ext_struct_def {
    () => (#[deriving(Copy, Sync)] pub struct Functions;);
    ($($function:ident ($($arg:ident $arg_type:ty)+) $ret:ty)+) =>
    (
        #[deriving(Copy, Sync)]
        pub struct Functions {
            $(pub $function: (extern fn ($($arg : $arg_type),+) -> $ret)),+
        }
    );
}
// (So is writing `name {}` as a literal)
macro_rules! ext_struct_literal {
    ($ext_name:expr $plat:ident) => (Functions);
    ($ext_name:expr $plat:ident $($function:ident)+) =>
    (
        Functions {
            $($function: {
                let mut fn_name = stringify!($function).to_c_str();
                // TODO use clGetExtensionFunctionAddressForPlatform() when it's available; more
                // reliable.
                let fn_ptr = unsafe { clGetExtensionFunctionAddress(fn_name.as_mut_ptr()) };
                if fn_ptr == ptr::null_mut() {
                    return Err(format!("extension {} apparently available for platform with id {}, but couldn't load function {}",
                                       $ext_name,
                                       $plat,
                                       stringify!($function)));
                }
                unsafe {
                    // Cast from *mut libc::void to the function pointer type we want
                    mem::transmute(fn_ptr)
                }
            }),+
        }
    );
}

pub mod cl_khr_fp64 {
    use cl::*;
    static CL_DEVICE_DOUBLE_FP_CONFIG: cl_uint = 0x1032;
}

pub mod cl_khr_fp16 {
    use cl::*;
    pub static CL_DEVICE_HALF_FP_CONFIG: cl_uint = 0x1033;
}

pub mod cl_APPLE_SetMemObjectDestructor {
    use libc;
    use cl::*;
    cl_extension_loader! {
        "cl_APPLE_SetMemObjectDestructor";
        extern fn clSetMemObjectDestructorAPPLE(memobj: cl_mem,
                                                pfn_notify: (extern fn(memobj: cl_mem,
                                                                       user_data: *mut libc::c_void)),
                                                user_data: *mut libc::c_void) -> (); // Note: returning () is necessary to satisfy macros
    }
}

pub mod cl_APPLE_ContextLoggingFunctions {
    use libc;
    use cl::*;
    cl_extension_loader! {
        "cl_APPLE_ContextLoggingFunctions";
        extern fn clLogMessagesToSystemLogAPPLE(errstr: *const libc::c_char,
                                                private_info: *const libc::c_void,
                                                cb: libc::size_t,
                                                user_data: *mut libc::c_void) -> ();
        extern fn clLogMessagesToStdoutAPPLE(errstr: *const libc::c_char,
                                             private_info: *const libc::c_void,
                                             cb: libc::size_t,
                                             user_data: *mut libc::c_void) -> ();
        extern fn clLogMessagesToStderrAPPLE(errstr: *const libc::c_char,
                                             private_info: *const libc::c_void,
                                             cb: libc::size_t,
                                             user_data: *mut libc::c_void) -> ();
    }
}

pub mod cl_khr_icd {
    use libc;
    use cl::*;
    pub static CL_PLATFORM_ICD_SUFFIX:      cl_uint = 0x0920;
    // Note: this is an error code, but we can't extend CLStatus with it... hmm.
    pub static CL_PLATFORM_NOT_FOUND_KHR:   cl_int  = -1001;
    cl_extension_loader! {
        "cl_khr_icd";
        extern fn clIcdGetPlatformIDsKHR(num_entries: cl_uint,
                                         platform: *mut cl_platform_id,
                                         num_platforms: *mut cl_uint) -> cl_int;
    }
}

pub mod cl_nv_device_attribute_query {
    use cl::*;
    pub static CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV:   cl_uint = 0x4000;
    pub static CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV:   cl_uint = 0x4001;
    pub static CL_DEVICE_REGISTERS_PER_BLOCK_NV:        cl_uint = 0x4002;
    pub static CL_DEVICE_WARP_SIZE_NV:                  cl_uint = 0x4003;
    pub static CL_DEVICE_GPU_OVERLAP_NV:                cl_uint = 0x4004;
    pub static CL_DEVICE_KERNEL_EXEC_TIMEOUT_NV:        cl_uint = 0x4005;
    pub static CL_DEVICE_INTEGRATED_MEMORY_NV:          cl_uint = 0x4006;
    cl_extension_loader! {
        "cl_nv_device_attribute_query";
    }
}

pub mod cl_amd_device_attribute_query {
    use cl::*;
    pub static CL_DEVICE_PROFILING_TIMER_OFFSET_AMD:    cl_uint = 0x4036;
    cl_extension_loader! {
        "cl_amd_device_attribute_query";
    }
}

pub mod cl_arm_printf {
    use cl::*;
    pub static CL_PRINTF_CALLBACK_ARM:      cl_uint = 0x40B0;
    pub static CL_PRINTF_BUFFERSIZE_ARM:    cl_uint = 0x40B1;
    cl_extension_loader! {
        "cl_arm_printf";
    }
}

pub mod cl_ext_device_fission {
    use std;
    use cl::*;
    pub type cl_device_partition_property_ext = cl_ulong;
    pub static CL_DEVICE_PARTITION_EQUALLY_EXT: cl_device_partition_property_ext            = 0x4050;
    pub static CL_DEVICE_PARTITION_BY_COUNTS_EXT: cl_device_partition_property_ext          = 0x4051;
    pub static CL_DEVICE_PARTITION_BY_NAMES_EXT: cl_device_partition_property_ext           = 0x4052;
    pub static CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN_EXT: cl_device_partition_property_ext = 0x4053;
    pub static CL_DEVICE_PARENT_DEVICE_EXT: cl_device_info                                  = 0x4054;
    pub static CL_DEVICE_PARTITION_TYPES_EXT: cl_device_info                                = 0x4055;
    pub static CL_DEVICE_AFFINITY_DOMAINS_EXT: cl_device_info                               = 0x4056;
    pub static CL_DEVICE_REFERENCE_COUNT_EXT: cl_device_info                                = 0x4057;
    pub static CL_DEVICE_PARTITION_STYLE_EXT: cl_device_info                                = 0x4058;
    pub static CL_DEVICE_PARTITION_FAILED_EXT: cl_int                                       = -1057;
    pub static CL_INVALID_PARTITION_COUNT_EXT: cl_int                                       = -1058;
    pub static CL_INVALID_PARTITION_NAME_EXT: cl_int                                        = -1059;
    pub static CL_AFFINITY_DOMAIN_L1_CACHE_EXT: cl_uint                                     = 0x1;
    pub static CL_AFFINITY_DOMAIN_L2_CACHE_EXT: cl_uint                                     = 0x2;
    pub static CL_AFFINITY_DOMAIN_L3_CACHE_EXT: cl_uint                                     = 0x3;
    pub static CL_AFFINITY_DOMAIN_L4_CACHE_EXT: cl_uint                                     = 0x4;
    pub static CL_AFFINITY_DOMAIN_NUMA_EXT: cl_uint                                         = 0x10;
    pub static CL_AFFINITY_DOMAIN_NEXT_FISSIONABLE_EXT: cl_uint                             = 0x100;
    pub static CL_PROPERTIES_LIST_END_EXT: cl_device_partition_property_ext                 = 0;
    pub static CL_PARTITION_BY_COUNTS_LIST_END_EXT: cl_device_partition_property_ext        = 0;
    pub static CL_PARTITION_BY_NAMES_LIST_END_EXT: cl_device_partition_property_ext         = std::u64::MAX;
    cl_extension_loader! {
        "cl_ext_device_fission";
        extern fn clReleaseDeviceEXT(device: cl_device_id) -> cl_int;
        extern fn clRetainDeviceEXT(device: cl_device_id) -> cl_int;
        extern fn clCreateSubDevicesExt(in_device: cl_device_id,
                                        properties: *const cl_device_partition_property_ext,
                                        num_entries: cl_uint,
                                        out_devices: *mut cl_device_id,
                                        num_devices: *mut cl_uint) -> cl_int;
    }
}

pub mod cl_qcom_ext_host_ptr {
    use libc;
    use cl::*;
    pub type cl_image_pitch_info_qcom = cl_uint;
    pub struct cl_mem_ext_host_ptr {
        pub allocation_type: cl_uint,
        pub host_cache_policy: cl_uint
    }
    pub static CL_MEM_EXT_HOST_PTR_QCOM:                cl_uint = (1 << 29);
    pub static CL_DEVICE_EXT_MEM_PADDING_IN_BYTES_QCOM: cl_uint = 0x40A0;
    pub static CL_DEVICE_PAGE_SIZE_QCOM:                cl_uint = 0x40A1;
    pub static CL_IMAGE_ROW_ALIGNMENT_QCOM:             cl_uint = 0x40A2;
    pub static CL_IMAGE_SLICE_ALIGNMENT_QCOM:           cl_uint = 0x40A3;
    pub static CL_MEM_HOST_UNCACHED_QCOM:               cl_uint = 0x40A4;
    pub static CL_MEM_HOST_WRITEBACK_QCOM:              cl_uint = 0x40A5;
    pub static CL_MEM_HOST_WRITETHROUGH_QCOM:           cl_uint = 0x40A6;
    pub static CL_MEM_HOST_WRITE_COMBINING_QCOM:        cl_uint = 0x40A7;
    cl_extension_loader! {
        "cl_qcom_ext_host_ptr";
        extern fn clGetDeviceImageInfoQCOM(device: cl_device_id,
                                           image_width: libc::size_t,
                                           image_height: libc::size_t,
                                           image_format: *const cl_image_format,
                                           param_name: cl_image_pitch_info_qcom,
                                           param_value_size: libc::size_t,
                                           param_value: *mut libc::c_void,
                                           param_value_size_ret: *mut libc::size_t) -> ();
    }
}

// This extension depends on the previous one. Should we try to express that?
pub mod cl_qcom_ion_host_ptr {
    use libc;
    use cl::*;
    use super::cl_qcom_ext_host_ptr;
    struct cl_mem_ion_host_ptr {
        pub ext_host_ptr: cl_qcom_ext_host_ptr::cl_mem_ext_host_ptr,
        pub ion_filedesc: libc::c_int,
        pub ion_hostptr: *mut libc::c_void
    }
    pub static CL_MEM_ION_HOST_PTR_QCOM: cl_uint = 0x40A8;

    cl_extension_loader! {
        "cl_qcom_ion_host_ptr";
    }
}
