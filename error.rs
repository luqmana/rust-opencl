use CL::*;

pub fn check(status: cl_int, message: &str) {
    if status != CL_SUCCESS as cl_int {
        fail fmt!("%s", message)
    }
}