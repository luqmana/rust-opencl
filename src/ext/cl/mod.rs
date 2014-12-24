use ::cl::*;

// TODO
// copy clGetExtensionFunctionAddressForPlatform
// read in supported extensions and check if this extension is supported
// figure out how to implement this properly

macro_rules! cl_extension {
    ($ext_name:ident : $ext_title_string:ident {
        $(fn $function:ident ($args:tt) -> $ret:ty);+
    }) => (
        pub struct $ext_name {
            $($function: fn ($args) -> $ret),+
        }

        impl $ext_name {
            pub fn load(platform: cl_platform_id) -> Option<$ext_name> {
                
            }
        }
    )
}
