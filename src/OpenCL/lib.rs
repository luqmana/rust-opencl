#![crate_id = "OpenCL#0.2"]
#![crate_type = "lib"]
#![feature(macro_rules)]
#![feature(globs)]
#![feature(managed_boxes)]
#![feature(link_args)]
#![feature(phase)]

//! OpenCL bindings for Rust.

extern crate libc;
extern crate sync;
#[phase(syntax, link)] extern crate log;

#[link_args = "-framework OpenCL"]
#[cfg(target_os = "macos")]
extern { }

#[link(name = "OpenCL")]
#[cfg(target_os = "linux")]
extern { }

/// Low-level OpenCL bindings. These should primarily be used by the
/// higher level features in this library.
pub mod CL;
pub mod error;
pub mod hl;
pub mod util;
pub mod mem;
pub mod array;
