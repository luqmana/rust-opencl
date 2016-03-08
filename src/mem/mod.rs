//! High level memory management.

pub use self::buffer::{CLBuffer, Buffer};
pub use self::transfer::{Put, Get, Read, Write};
pub use self::array::{Array2D, CLArray2D, Array3D, CLArray3D};

mod buffer;
mod transfer;
mod array;
