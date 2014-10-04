#![crate_name = "opencl"]
#![crate_type = "lib"]
#![feature(macro_rules)]
#![feature(globs)]
#![feature(phase)]
#![feature(unsafe_destructor)]
#![allow(ctypes)]

//! OpenCL bindings for Rust.

extern crate debug;
extern crate libc;
extern crate sync;
#[phase(plugin, link)] extern crate log;

#[link(name = "OpenCL", kind = "framework")]
#[cfg(target_os = "macos")]
extern { }

#[link(name = "OpenCL")]
#[cfg(target_os = "linux")]
extern { }

/// Low-level OpenCL bindings. These should primarily be used by the
/// higher level features in this library.
pub mod cl;
pub mod error;
pub mod hl;
pub mod util;
pub mod mem;
pub mod array;
