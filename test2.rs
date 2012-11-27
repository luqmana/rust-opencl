extern mod OpenCL;
use OpenCL::CL::*;
use OpenCL::hl::*;

fn main() {
    let platforms = get_platforms();
    
    error!("%?", platforms);

    let devices = get_devices(platforms[0]);

    error!("%?", devices);
}
