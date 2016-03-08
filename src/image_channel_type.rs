//! Image channel types.

#![allow(non_camel_case_types)]

use cl::*;

/// Normalized 8-bit integer value for images data type.
#[derive(Copy, Clone)]
pub struct norm_i8(i8);

/// Normalized 16-bit integer value for images data type.
#[derive(Copy, Clone)]
pub struct norm_i16(i16);

/// Normalized 8-bit unsigned integer value for images data type.
#[derive(Copy, Clone)]
pub struct norm_u8(u8);

/// Normalized 16-bit unsigned integer value for images data type.
#[derive(Copy, Clone)]
pub struct norm_u16(u16);

/// Normalized 16-bit unsigned integer value for images data type (5-6-5 layout).
#[derive(Copy, Clone)]
pub struct norm_u16_565(u16);

/// Normalized 16-bit unsigned integer value for images data type (5-5-5 layout).
#[derive(Copy, Clone)]
pub struct norm_u16_555(u16);

/// Normalized 32-bit unsigned integer value for images data type (10-10-10 layout).
#[derive(Copy, Clone)]
pub struct norm_u32_101010(u32);

/// Trait implemented by potential types for image channel elements.
pub trait ImageChannelType : Copy {
    /// Corresponding OpenCL value for the `cl_image_format::image_channel_data_type` field.
    fn cl_flag(Option<Self>) -> cl_uint;
}

macro_rules! impl_channel_type(
    ($t: ty, $flag: expr) => (
        impl ImageChannelType for $t {
            fn cl_flag(_: Option<$t>) -> cl_uint {
                $flag
            }
        }
    )
);

impl_channel_type!(norm_i8, CL_SNORM_INT8);
impl_channel_type!(norm_i16, CL_SNORM_INT16);
impl_channel_type!(norm_u8, CL_UNORM_INT8);
impl_channel_type!(norm_u16, CL_UNORM_INT16);
impl_channel_type!(norm_u16_565, CL_UNORM_SHORT_565);
impl_channel_type!(norm_u16_555, CL_UNORM_SHORT_555);
impl_channel_type!(norm_u32_101010, CL_UNORM_INT_101010);
impl_channel_type!(i8, CL_SIGNED_INT8);
impl_channel_type!(i16, CL_SIGNED_INT16);
impl_channel_type!(i32, CL_SIGNED_INT32);
impl_channel_type!(u8, CL_UNSIGNED_INT8);
impl_channel_type!(u16, CL_UNSIGNED_INT16);
impl_channel_type!(u32, CL_UNSIGNED_INT32);
impl_channel_type!(f32, CL_FLOAT);
