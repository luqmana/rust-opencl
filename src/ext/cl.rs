use cl::cl_platform_id;
use cl::ll::clGetExtensionFunctionAddressForPlatform;
use hl;
use libc;
use std::mem;
use std::ptr;

macro_rules! cl_extension {
    ($ext:ident {
        $(extern fn $function:ident ($args:tt) -> $ret:ty);*
    }) => (
        /// The opencl extension $ext.
        /// This is a struct-style loader - you need to explicitly call 
        /// `$ext::load(my_cl_platform_id)` to get a struct with the
        /// functions from $ext available as methods.
        #[deriving(Copy,Sync)]
        pub struct $ext {
            $($function: (extern fn ($args) -> $ret)),+
        }

        impl $ext {
            pub fn load(platform: cl_platform_id) -> Result<$ext, String> {
                // Read in the available extensions
                // We have to do this, since loading function pointers for an 
                // unavailable extension can return non-NULL
                let available = unsafe {
                    let hl_platform = hl::Platform { id: cl_platform_id };
                    let available = hl_plat.extensions().contains(stringify!($ext));
                    forget(hl_platform)
                    available
                }
                if !available {
                    return Err(format!("extension {} unavailable for platform with id {}", stringify!(ext_name), cl_platform_id));
                }
                // Return a struct with all functions loaded
                $ext {
                    $($function: {
                        let fn_name = stringify!($function).to_c_str();
                        // TODO use clGetExtensionFunctionAddressForPlatform() when it's available-
                        // this 
                        let fn_ptr = unsafe { clGetExtensionFunctionAddress(fn_name.as_ptr()) };
                        if fn_ptr == ptr::null() {
                            return Err(format!("extension {} apparently available for platform with id {}, but couldn't load function {}",
                                               stringify!(ext_name),
                                               cl_platform_id,
                                               stringify!($function)));
                        }
                        unsafe {
                            // Cast from *mut libc::void to the function pointer type we want
                            mem::transmute(fn_ptr)
                        }
                    }),*
                }
            }
            // Make function pointers available as methods so they don't have to be called as
            // (struct.member)(arg)
            $(ext_fn_method!{$function ($args) $ret})*
        }
    )
}
// Second macro so we can match multiple arguments
// (You can't match repetitions recursively, can you?)
macro_rules! ext_fn_method {
    ($function:ident ($($arg:ident : $arg_type:ty),*) $ret:ty) =>
    (
        #[inline(always)] unsafe fn $function (&self, $($arg:$arg_type),*) -> $ret {
            (self.$function)($($arg),*)
        }
    )
}

cl_extension! {
    cl_APPLE_SetMemObjectDestructor {
        extern fn clSetMemObjectDestructorAPPLE(memobj: cl_mem,
                                         pfn_notify: extern fn(memobj: cl_mem,
                                                               user_data: *mut libc::c_void),
                                         user_data: *mut libc::c_void);
    }
}

cl_extension! {
    cl_APPLE_ContextLoggingFunctions {
        extern fn clLogMessagesToSystemLogAPPLE(errstr: *const libc::c_char,
                                                private_info: *const libc::c_void,
                                                cb: libc::size_t,
                                                user_data: *mut libc::c_void);
        extern fn clLogMessagesToStdoutAPPLE(errstr: *const libc::c_char,
                                                private_info: *const libc::c_void,
                                                cb: libc::size_t,
                                                user_data: *mut libc::c_void);
        extern fn clLogMessagesToStderrAPPLE(errstr: *const libc::c_char,
                                                private_info: *const libc::c_void,
                                                cb: libc::size_t,
                                                user_data: *mut libc::c_void);
    }
}

cl_extension! {
    cl_khr_icd {

