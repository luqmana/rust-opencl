use CL::*;

impl CLStatus: ToStr {
    pure fn to_str() -> ~str {
        fmt!("%?", self)
    }
}

macro_rules! convert(
    ($e: expr, $x: ident) => (if $e == $x as cl_int {
        Some($x)
    } else { None });

    ($e: expr, $x: ident, $($xs: ident),+) => (if $e == $x as cl_int {
        Some($x)
    } else { convert!($e, $($xs),+) })
)

pub fn try_convert(status: cl_int) -> Option<CLStatus> {
    convert!(status,
             CL_SUCCESS,
             CL_DEVICE_NOT_FOUND,
             CL_DEVICE_NOT_AVAILABLE,
             CL_COMPILER_NOT_AVAILABLE,
             CL_MEM_OBJECT_ALLOCATION_FAILURE,
             CL_OUT_OF_RESOURCES,
             CL_OUT_OF_HOST_MEMORY,
             CL_PROFILING_INFO_NOT_AVAILABLE,
             CL_MEM_COPY_OVERLAP,
             CL_IMAGE_FORMAT_MISMATCH,
             CL_IMAGE_FORMAT_NOT_SUPPORTED,
             CL_BUILD_PROGRAM_FAILURE,
             CL_MAP_FAILURE,
             CL_MISALIGNED_SUB_BUFFER_OFFSET,
             CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
             CL_INVALID_VALUE,
             CL_INVALID_DEVICE_TYPE,
             CL_INVALID_PLATFORM,
             CL_INVALID_DEVICE,
             CL_INVALID_CONTEXT,
             CL_INVALID_QUEUE_PROPERTIES,
             CL_INVALID_COMMAND_QUEUE,
             CL_INVALID_HOST_PTR,
             CL_INVALID_MEM_OBJECT,
             CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
             CL_INVALID_IMAGE_SIZE,
             CL_INVALID_SAMPLER,
             CL_INVALID_BINARY,
             CL_INVALID_BUILD_OPTIONS,
             CL_INVALID_PROGRAM,
             CL_INVALID_PROGRAM_EXECUTABLE,
             CL_INVALID_KERNEL_NAME,
             CL_INVALID_KERNEL_DEFINITION,
             CL_INVALID_KERNEL,
             CL_INVALID_ARG_INDEX,
             CL_INVALID_ARG_VALUE,
             CL_INVALID_ARG_SIZE,
             CL_INVALID_KERNEL_ARGS,
             CL_INVALID_WORK_DIMENSION,
             CL_INVALID_WORK_GROUP_SIZE,
             CL_INVALID_WORK_ITEM_SIZE,
             CL_INVALID_GLOBAL_OFFSET,
             CL_INVALID_EVENT_WAIT_LIST,
             CL_INVALID_EVENT,
             CL_INVALID_OPERATION,
             CL_INVALID_GL_OBJECT,
             CL_INVALID_BUFFER_SIZE,
             CL_INVALID_MIP_LEVEL,
             CL_INVALID_GLOBAL_WORK_SIZE,
             CL_INVALID_PROPERTY)
}

pub fn convert(status: cl_int) -> CLStatus {
    match try_convert(status) {
        Some(s) => s,
        None => fail fmt!("Uknown OpenCL Status Code: %?", status)
    }
}

fn error_str(status: cl_int) -> ~str {
    match try_convert(status) {
        Some(s) => s.to_str(),
        None => fmt!("Unknown Error: %?", status)
    }
}

pub fn check(status: cl_int, message: &str) {
    if status != CL_SUCCESS as cl_int {
        fail fmt!("%s (%s)", message, error_str(status))
    }
}

macro_rules! expect (
    ($test: expr, $expected: expr) => ({
        let test = $test;
        let expected = $expected;
        if test != expected {
            fail fmt!("Test failure in %s: expected %?, got %?",
                      stringify!($test),
                      expected, test)
        }
    })
)

#[cfg(test)]
mod test {
    #[test]
    fn test_convert() {
        expect!(convert(CL_INVALID_GLOBAL_OFFSET as cl_int),
                CL_INVALID_GLOBAL_OFFSET)
    }

    #[test]
    fn convert_to_str() {
        expect!(convert(CL_INVALID_BUFFER_SIZE as cl_int).to_str(),
                ~"CL_INVALID_BUFFER_SIZE");
    }
}
