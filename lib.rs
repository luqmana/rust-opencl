#[link(name = "OpenCL",
       vers = "0.1",
       uuid = "26bc3ce8-7720-41a7-9280-c522285f2e70")];
#[crate_type = "lib"];
#[feature(macro_rules)];
#[feature(globs)];
#[feature(managed_boxes)];

extern mod std;

#[nolink]
#[link_args = "-framework OpenCL"]
#[cfg(target_os = "macos")]
extern { }

#[nolink]
#[link_args = "-lOpenCL"]
#[cfg(target_os = "linux")]
extern { }

pub mod CL;
pub mod error;
pub mod hl;
pub mod util;
