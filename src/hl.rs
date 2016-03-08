//! A higher level API.

use device::{Device, DeviceType};
use context::Context;
use command_queue::CommandQueue;

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


/// Creates a complete compute context.
///
/// This creates a command queue and context for the first device of the first platform. The
/// command queue has profiling and out-of-order command execution both disabled.
pub fn create_compute_context(profiling: bool) -> Result<(Device, Context, CommandQueue), &'static str>
{
    let platforms = ::platforms();
    if platforms.len() == 0 {
        return Err("No platform found");
    }

    let mut devices = platforms[0].get_devices();
    if devices.len() == 0 {
        Err("No device found")
    } else {
        let device  = devices.remove(0);
        let context = Context::new(&device);
        let queue   = CommandQueue::new(&context, &device, profiling, false);
        Ok((device, context, queue))
    }
}

/// Attempt to create a complete compute context for the specified device type.
///
/// This creates a command queue and context for the first device of the specified type on the
/// first platform that contains it. The command queue has profiling and out-of-order command
/// execution both disabled.
pub fn create_compute_context_prefer(cltype: PreferedType, profiling: bool) -> Result<(Device, Context, CommandQueue), &'static str>
{
    let platforms = ::platforms();
    for platform in platforms.iter() {
        let types = match cltype {
            PreferedType::Any => vec![DeviceType::CPU, DeviceType::GPU],
            PreferedType::CPUPrefered | PreferedType::CPUOnly => vec![DeviceType::CPU],
            PreferedType::GPUPrefered | PreferedType::GPUOnly => vec![DeviceType::GPU]
        };

        let mut devices = platform.get_devices_by_types(&types[..]);
        if devices.len() > 0 {
            let device  = devices.remove(0);
            let context = Context::new(&device);
            let queue   = CommandQueue::new(&context, &device, profiling, false);
            return Ok((device, context, queue))
        }
    }


    match cltype {
        PreferedType::Any |
        PreferedType::CPUPrefered |
        PreferedType::GPUPrefered => create_compute_context(profiling),
        _ => Err("Could not find valid implementation")
    }
}
