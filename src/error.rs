//! Error handling utilities.

use cl::{CLStatus, cl_int, CL_SUCCESS};

fn error_str(status_code: cl_int) -> String {
    let status_opt: Option<CLStatus> = FromPrimitive::from_int(status_code as int);

    match status_opt {
        Some(status) => status.to_string(),
        None => format!("Unknown Error: {}", status_code)
    }
}

pub fn check(status: cl_int, message: &str) {
    if status != CL_SUCCESS as cl_int {
        panic!(format!("{:s} ({})", message, error_str(status)))
    }
}
