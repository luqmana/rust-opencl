use hl::*;

pub fn create_compute_context() -> Result<(Device, Context, CommandQueue), ~str>
{
    let platforms = get_platforms();
    if platforms.len() == 0 {
        return Err(~"No platform found");
    }

    let devices = platforms[0].get_devices();
    if devices.len() == 0 {
        return Err(~"No devices found");
    }

    let context = devices[0].create_context();
    let queue = context.create_command_queue(&devices[0]);

    Ok((devices[0], context, queue))
}


pub enum PreferedType {
    ANY,

    CPU_PREFERED,
    GPU_PREFERED,

    CPU_ONLY,
    GPU_ONLY,

}

pub fn create_compute_context_prefer(cltype: PreferedType) -> Result<(Device, Context, CommandQueue), ~str>
{
    let platforms = get_platforms();
    for platform in platforms.iter() {
        let types = match cltype {
            ANY => &[CPU, GPU],
            CPU_PREFERED | CPU_ONLY => &[CPU],
            GPU_PREFERED | GPU_ONLY => &[GPU]
        };

        let devices = platform.get_devices_by_types(types);
        if devices.len() != 0 {
            let context = devices[0].create_context();
            let queue = context.create_command_queue(&devices[0]);
            return Ok((devices[0], context, queue));
        } 
    }


    match cltype {
        ANY | CPU_PREFERED | GPU_PREFERED => create_compute_context(),
        _ => Err(~"Could not find valid implementation")    
    }
}