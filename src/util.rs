//! Utility functions

use hl::*;

/// Creates a complete compute context.
///
/// This creates a command queue and context for the first device of the first platform.
pub fn create_compute_context() -> Result<(Device, Context, CommandQueue), &'static str>
{
    let platforms = get_platforms();
    if platforms.len() == 0 {
        return Err("No platform found");
    }

    let mut devices = platforms[0].get_devices();
    if devices.len() == 0 {
        Err("No device found")
    } else {
        let device = devices.remove(0);
        let context = device.create_context();
        let queue = context.create_command_queue(&device);
        Ok((device, context, queue))
    }
}

/// The preferred device type for compute context creation.
#[derive(Copy, Clone)]
pub enum PreferedType {
    /// Pick the first available device.
    Any,

    /// Pick the first CPU device. Default to GPU in none is found.
    CPUPrefered,
    /// Pick the first GPU device. Default to CPU in none is found.
    GPUPrefered,

    /// Pick only the first available CPU device. Fail if none is found.
    CPUOnly,
    /// Pick only the first available GPU device. Fail if none is found.
    GPUOnly,
}

/// Attempt to create a complete compute context for the specified device type.
///
/// This creates a command queue and context for the first device of the specified type on the
/// first platform that contains it.
pub fn create_compute_context_prefer(cltype: PreferedType) -> Result<(Device, Context, CommandQueue), &'static str>
{
    let platforms = get_platforms();
    for platform in platforms.iter() {
        let types = match cltype {
            PreferedType::Any => vec![DeviceType::CPU, DeviceType::GPU],
            PreferedType::CPUPrefered | PreferedType::CPUOnly => vec![DeviceType::CPU],
            PreferedType::GPUPrefered | PreferedType::GPUOnly => vec![DeviceType::GPU]
        };

        let mut devices = platform.get_devices_by_types(&types[..]);
        if devices.len() > 0 {
            let device = devices.remove(0);
            let context = device.create_context();
            let queue = context.create_command_queue(&device);
            return Ok((device, context, queue))
        }
    }


    match cltype {
        PreferedType::Any |
        PreferedType::CPUPrefered |
        PreferedType::GPUPrefered => create_compute_context(),
        _ => Err("Could not find valid implementation")
    }
}
