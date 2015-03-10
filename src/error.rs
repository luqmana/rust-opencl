//! Error handling utilities.

use cl::{CLStatus, cl_int};
use cl::CLStatus::CL_SUCCESS;
use std::num::FromPrimitive;

fn error_str(status_code: cl_int) -> String {
    let status_opt: Option<CLStatus>
        = FromPrimitive::from_isize(status_code as isize);

    match status_opt {
        Some(status) => status.to_string(),
        None => format!("Unknown Error: {}", status_code)
    }
}

pub fn check(status: cl_int, message: &str) {
    if status != CL_SUCCESS as cl_int {
        panic!("{} ({})", message, error_str(status))
    }
}
