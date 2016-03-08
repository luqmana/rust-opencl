/*! # rust-opencl

OpenCL bindings and high-level interface for Rust.

## Installation

Add the following to your `Cargo.toml` file:

```.ignore
[dependencies] 
rust-opencl = "0.4.0"
```
*/


#![allow(improper_ctypes)]
#![allow(missing_copy_implementations)]

#![deny(non_upper_case_globals)]
#![deny(non_camel_case_types)]
#![deny(unused_parens)]
#![deny(unused_qualifications)]
#![deny(unused_results)]
#![deny(unused_imports)]
#![warn(missing_docs)]

#![feature(static_mutex)]


extern crate libc;
#[macro_use]
extern crate log;

#[link(name = "OpenCL", kind = "framework")]
#[cfg(target_os = "macos")]
extern { }

#[link(name = "OpenCL")]
#[cfg(target_os = "linux")]
extern { }

pub mod cl;
pub mod ext;
pub mod error;
pub mod hl;
pub mod util;
pub mod mem;
pub mod array;
