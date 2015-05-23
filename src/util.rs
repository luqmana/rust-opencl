//! Utility functions

use hl::*;

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

#[derive(Copy, Clone)]
pub enum PreferedType {
    Any,

    CPUPrefered,
    GPUPrefered,

    CPUOnly,
    GPUOnly,
}

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
