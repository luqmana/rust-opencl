extern mod OpenCL;
use OpenCL::CL::*;
use OpenCL::hl::*;

fn main() {
    let platforms = get_platforms();
    
    error!("%?", platforms);

    get_devices(*platforms[0]);
}
