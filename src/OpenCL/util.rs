//! Utility functions

use hl::*;

pub fn create_compute_context() -> Result<(Device, Context, CommandQueue), &str>
{
    let platforms = get_platforms();
    if platforms.len() == 0 {
        return Err("No platform found");
    }

    let devices = platforms.get(0).get_devices();
    if devices.len() == 0 {
        return Err("No devices found");
    }

    let context = devices.get(0).create_context();
    let queue = context.create_command_queue(devices.get(0));

    Ok((*devices.get(0), context, queue))
}


pub enum PreferedType {
    Any,

    CPUPrefered,
    GPUPrefered,

    CPUOnly,
    GPUOnly,

}

pub fn create_compute_context_prefer(cltype: PreferedType) -> Result<(Device, Context, CommandQueue), &str>
{
    let platforms = get_platforms();
    for platform in platforms.iter() {
        let types = match cltype {
            Any => ~[CPU, GPU],
            CPUPrefered | CPUOnly => ~[CPU],
            GPUPrefered | GPUOnly => ~[GPU]
        };

        let devices = platform.get_devices_by_types(types);
        if devices.len() != 0 {
            let context = devices.get(0).create_context();
            let queue = context.create_command_queue(devices.get(0));
            return Ok((*devices.get(0), context, queue));
        } 
    }


    match cltype {
        Any | CPUPrefered | GPUPrefered => create_compute_context(),
        _ => Err("Could not find valid implementation")   
    }
}
