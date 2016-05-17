//! Error handling utilities.

use cl::{CLStatus, cl_int};

fn error_str(status_code: cl_int) -> String {
    match status_code {
    0 => CLStatus::CL_SUCCESS.to_string(),
    -1 => CLStatus::CL_DEVICE_NOT_FOUND.to_string(),
    -2 => CLStatus::CL_DEVICE_NOT_AVAILABLE.to_string(),
    -3 => CLStatus::CL_COMPILER_NOT_AVAILABLE.to_string(),
    -4 => CLStatus::CL_MEM_OBJECT_ALLOCATION_FAILURE.to_string(),
    -5 => CLStatus::CL_OUT_OF_RESOURCES.to_string(),
    -6 => CLStatus::CL_OUT_OF_HOST_MEMORY.to_string(),
    -7 => CLStatus::CL_PROFILING_INFO_NOT_AVAILABLE.to_string(),
    -8 => CLStatus::CL_MEM_COPY_OVERLAP.to_string(),
    -9 => CLStatus::CL_IMAGE_FORMAT_MISMATCH.to_string(),
    -10 => CLStatus::CL_IMAGE_FORMAT_NOT_SUPPORTED.to_string(),
    -11 => CLStatus::CL_BUILD_PROGRAM_FAILURE.to_string(),
    -12 => CLStatus::CL_MAP_FAILURE.to_string(),
    -13 => CLStatus::CL_MISALIGNED_SUB_BUFFER_OFFSET.to_string(),
    -14 => CLStatus::CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST.to_string(),
    -30 => CLStatus::CL_INVALID_VALUE.to_string(),
    -31 => CLStatus::CL_INVALID_DEVICE_TYPE.to_string(),
    -32 => CLStatus::CL_INVALID_PLATFORM.to_string(),
    -33 => CLStatus::CL_INVALID_DEVICE.to_string(),
    -34 => CLStatus::CL_INVALID_CONTEXT.to_string(),
    -35 => CLStatus::CL_INVALID_QUEUE_PROPERTIES.to_string(),
    -36 => CLStatus::CL_INVALID_COMMAND_QUEUE.to_string(),
    -37 => CLStatus::CL_INVALID_HOST_PTR.to_string(),
    -38 => CLStatus::CL_INVALID_MEM_OBJECT.to_string(),
    -39 => CLStatus::CL_INVALID_IMAGE_FORMAT_DESCRIPTOR.to_string(),
    -40 => CLStatus::CL_INVALID_IMAGE_SIZE.to_string(),
    -41 => CLStatus::CL_INVALID_SAMPLER.to_string(),
    -42 => CLStatus::CL_INVALID_BINARY.to_string(),
    -43 => CLStatus::CL_INVALID_BUILD_OPTIONS.to_string(),
    -44 => CLStatus::CL_INVALID_PROGRAM.to_string(),
    -45 => CLStatus::CL_INVALID_PROGRAM_EXECUTABLE.to_string(),
    -46 => CLStatus::CL_INVALID_KERNEL_NAME.to_string(),
    -47 => CLStatus::CL_INVALID_KERNEL_DEFINITION.to_string(),
    -48 => CLStatus::CL_INVALID_KERNEL.to_string(),
    -49 => CLStatus::CL_INVALID_ARG_INDEX.to_string(),
    -50 => CLStatus::CL_INVALID_ARG_VALUE.to_string(),
    -51 => CLStatus::CL_INVALID_ARG_SIZE.to_string(),
    -52 => CLStatus::CL_INVALID_KERNEL_ARGS.to_string(),
    -53 => CLStatus::CL_INVALID_WORK_DIMENSION.to_string(),
    -54 => CLStatus::CL_INVALID_WORK_GROUP_SIZE.to_string(),
    -55 => CLStatus::CL_INVALID_WORK_ITEM_SIZE.to_string(),
    -56 => CLStatus::CL_INVALID_GLOBAL_OFFSET.to_string(),
    -57 => CLStatus::CL_INVALID_EVENT_WAIT_LIST.to_string(),
    -58 => CLStatus::CL_INVALID_EVENT.to_string(),
    -59 => CLStatus::CL_INVALID_OPERATION.to_string(),
    -60 => CLStatus::CL_INVALID_GL_OBJECT.to_string(),
    -61 => CLStatus::CL_INVALID_BUFFER_SIZE.to_string(),
    -62 => CLStatus::CL_INVALID_MIP_LEVEL.to_string(),
    -63 => CLStatus::CL_INVALID_GLOBAL_WORK_SIZE.to_string(),
    -64 => CLStatus::CL_INVALID_PROPERTY.to_string(),
    -1001 => CLStatus::CL_PLATFORM_NOT_FOUND_KHR.to_string(),
        _ => format!("Unknown Error: {}", status_code)
    }
}

/// Checks and prints an OpenCL error code with a message in case of failure.
pub fn check(status: cl_int, message: &str) {
    if status != CLStatus::CL_SUCCESS as cl_int {
        panic!("{} ({})", message, error_str(status))
    }
}
